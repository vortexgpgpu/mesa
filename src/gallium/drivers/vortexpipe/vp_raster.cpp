/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_raster -- gfx_v2 on-device draw orchestration.
 *
 * The whole binning front end runs on the device: the embedded setup_k +
 * binning_k kernels (sw/gfx, built into libvortexpipe) do triangle clip +
 * setup and parallel bin-sort over device-resident memory, producing
 * RASTER's primbuf + dense tilebuf with no host graphics::Binning(). The
 * host only converts VS output to the front-end vertex record, sizes the
 * dense tile grid from the framebuffer, and sequences the launches.
 *
 * vp_raster_draw then programs the RASTER and OM DCRs from the
 * device-produced buffers and dispatches the fragment-shader kernel, which
 * polls vx_rast() for quads and submits shaded fragments to the OM unit via
 * vx_om() for depth-test, blend and colour + depth writeback.
 */

#include "vp_raster.h"

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "graphics.h"            /* vortex::graphics::rast_prim_t / rast_tile_header_t */
#include "gfx_frontend_abi.h"    /* pipe_arg_t, setup_vertex_t, PIPE_STAGE_*, SETUP_* */
#include "vp_gfx_frontend.h"     /* embedded setup_k + binning_k .vxbin */
#include "VX_types.h"            /* VX_DCR_RASTER_*, VX_DCR_OM_*, VX_OM_* */
#include "util/log.h"

namespace graphics = vortex::graphics;

/* gfx_v2 §6.3: the on-device front end bins at the coarse 128 px bin log
 * (VX_config.h VX_CFG_RASTER_BIN_LOGSIZE) and the RASTER unit descends
 * bin -> block -> quad. Must match the front-end kernel's PIPE_BIN_LOG
 * (sw/gfx/pipe_abi.h). */
#ifndef RASTER_BIN_LOGSIZE
#define RASTER_BIN_LOGSIZE 7
#endif

/* sizeof(graphics::rast_prim_t): vec3e_t edges[3] + rast_attribs_t. */
#define VP_RAST_PRIM_STRIDE 120

/* Hard error from the Vortex runtime — log as mesa_loge so the host /
 * test harness can detect it. Soft "feature not implemented" notices
 * remain mesa_logw; runtime API failures must not. */
#define VP_CHECK(call, what)                                            \
   do {                                                                 \
      vx_result_t _r = (call);                                          \
      if (_r != VX_SUCCESS) {                                           \
         mesa_loge("vortexpipe: raster: %s failed (%s)", (what),        \
                   vx_result_string(_r));                               \
         goto done;                                                     \
      }                                                                 \
   } while (0)

/* enqueue one DCR write, bailing to `done` on failure */
#define DCR(addr, val)                                                  \
   VP_CHECK(vx_enqueue_dcr_write(q, (addr), (uint32_t)(val), 0, NULL, NULL), \
            #addr)

/* exact log2 of a power-of-two texture dimension */
static inline uint32_t
vp_log2u(uint32_t n)
{
   uint32_t l = 0;
   while ((1u << l) < n)
      l++;
   return l;
}

extern "C" bool
vp_raster_draw(vx_device_h dev,
               const void *fs_vxbin, size_t fs_vxbin_size,
               uint64_t vsrec_addr, uint32_t vertex_count,
               const struct vp_vs_layout *layout,
               void *color, uint32_t width, uint32_t height,
               const struct vp_om_params *om,
               const struct vp_tex_params *tex)
{
   const uint32_t num_tris = vertex_count / 3;
   if (num_tris == 0) {
      mesa_logw("vortexpipe: raster: draw has no complete triangles");
      return false;
   }

   /* The resident VS output (vsrec_addr) is expanded to setup_vertex_t
    * on-device by expand_k below — no host readback or vertex expansion. */

   /* ---- dense tile grid, sized from the framebuffer ------------------- *
    * Bins == RASTER tiles, so the grid count (bin_cols*bin_rows) is the
    * RASTER TILE_COUNT — host-known, no num_tiles readback. The front end
    * emits one header per tile (empty tiles get pids_count=0). */
   const uint32_t BIN      = 1u << RASTER_BIN_LOGSIZE;
   const uint32_t bin_cols = (width  + BIN - 1) / BIN;
   const uint32_t bin_rows = (height + BIN - 1) / BIN;
   const uint32_t num_bins = bin_cols * bin_rows;
   const uint32_t MS       = SETUP_MAX_SUB;
   const uint32_t P_max    = num_tris * MS;          /* worst-case kept prims */

   /* keys[] holds one (bin,prim) entry per coarse 128 px bin a prim's bbox
    * covers; the exact count is meta[1], known only on-device. Size worst-case
    * (every kept prim covers every bin) so the launch chain needs no mid-frame
    * readback. The coarse bins make this far smaller than 32 px tiles, and
    * rast_bin_header_t's 32-bit fields lift the old 16-bit cap; tighter /
    * segmented allocation is the §6.2 follow-up. */
   const uint32_t keys_cap = P_max && num_bins ? P_max * num_bins : 1;

   const size_t PRIM_SZ = sizeof(graphics::rast_prim_t);
   const size_t HDR_SZ  = sizeof(graphics::rast_bin_header_t);
   const size_t BBOX_SZ = sizeof(setup_bbox_t);
   const size_t TILEBUF_SZ = (size_t)num_bins * HDR_SZ + (size_t)keys_cap * 4;
   const uint32_t cbuf_bytes = width * height * 4;

   bool ok = false;
   vx_queue_h  q     = NULL;
   vx_module_h femod = NULL, kmod = NULL;
   vx_kernel_h k_expand = NULL, k_setup = NULL, k_binning = NULL, kbuf = NULL;

   /* front-end resident buffer set (verts in; prim + tilebuf out; rest
    * scratch) + the colour/depth/texture buffers the FF units write/read */
   vx_buffer_h verts_buf = NULL, slot_prim_buf = NULL, slot_bbox_buf = NULL,
               keep_buf = NULL, offset_buf = NULL, tsum_buf = NULL,
               prim_buf = NULL, bbox_buf = NULL, bcount_buf = NULL,
               boffset_buf = NULL, keys_buf = NULL, btsum_buf = NULL,
               thist_buf = NULL, bincount_buf = NULL, binbase_buf = NULL,
               tilebuf_buf = NULL, meta_buf = NULL;
   vx_buffer_h cbuf = NULL, zbuf = NULL, xbuf = NULL;

   char vxpath[] = "/tmp/vortexpipe-fs.XXXXXX";
   int  vxfd = -1;

   vxfd = mkstemp(vxpath);
   if (vxfd < 0) {
      mesa_loge("vortexpipe: raster: mkstemp failed");
      return false;
   }
   if (write(vxfd, fs_vxbin, fs_vxbin_size) != (ssize_t)fs_vxbin_size) {
      mesa_loge("vortexpipe: raster: writing .vxbin failed");
      close(vxfd);
      unlink(vxpath);
      return false;
   }
   close(vxfd);

   {
      vx_queue_info_t qi = { sizeof(qi), NULL, VX_QUEUE_PRIORITY_NORMAL, 0 };
      VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");

      /* The front-end and fragment-shader .vxbins both link at the fixed
       * kernel STARTUP_ADDR, so only one can be resident at a time. Run the
       * front end first, drain it, then swap in the per-draw FS module; the
       * device buffers it produces (prim + tilebuf) are separate
       * allocations and survive the code-module swap.
       *
       * front-end module: the setup_k + binning_k blob embedded in the
       * driver, loaded straight from memory (no toolchain at runtime). */
      size_t fe_size = 0;
      const void *fe_bytes = vp_gfx_frontend_vxbin(&fe_size);
      VP_CHECK(vx_module_load_bytes(dev, fe_bytes, fe_size, &femod),
               "vx_module_load_bytes(frontend)");
      VP_CHECK(vx_module_get_kernel(femod, "expand_k", &k_expand),
               "vx_module_get_kernel(expand_k)");
      VP_CHECK(vx_module_get_kernel(femod, "setup_k", &k_setup),
               "vx_module_get_kernel(setup_k)");
      VP_CHECK(vx_module_get_kernel(femod, "binning_k", &k_binning),
               "vx_module_get_kernel(binning_k)");

      /* device-filling launch geometry: T = per-CTA threads, G = grid. */
      uint32_t one = 1, gdim[1] = { 1 }, bdim[1] = { 1 };
      VP_CHECK(vx_device_max_occupancy_grid(dev, 1, &one, gdim, bdim),
               "vx_device_max_occupancy_grid");
      const uint32_t T = bdim[0];
      const uint32_t G = gdim[0];

      /* Allocate the front-end buffer set up front (per draw) and capture
       * each device address into the shared pipe_arg_t. */
      pipe_arg_t arg{};
      arg.num_tris   = num_tris;
      arg.width      = width;
      arg.height     = height;
      arg.bin_stripe = (num_bins + G - 1) / G;   /* bins per CTA (HIST/SCATTER) */
      arg.bin_cols   = bin_cols;
      arg.num_bins   = num_bins;
      arg.cull_mode  = SETUP_CULL_NONE;          /* gfx-v1 two-sided default */

      struct { uint64_t bytes; uint64_t *addr; vx_buffer_h *h; const char *what; } spec[] = {
         { (uint64_t)vertex_count * sizeof(setup_vertex_t), &arg.verts_addr,     &verts_buf,     "verts"     },
         { (uint64_t)P_max * PRIM_SZ,                       &arg.slot_prim_addr, &slot_prim_buf, "slot_prim" },
         { (uint64_t)P_max * BBOX_SZ,                       &arg.slot_bbox_addr, &slot_bbox_buf, "slot_bbox" },
         { (uint64_t)num_tris * 4,                          &arg.keep_addr,      &keep_buf,      "keep"      },
         { (uint64_t)(num_tris + 1) * 4,                    &arg.offset_addr,    &offset_buf,    "offset"    },
         { (uint64_t)T * 4,                                 &arg.tsum_addr,      &tsum_buf,      "tsum"      },
         { (uint64_t)P_max * PRIM_SZ,                       &arg.prim_addr,      &prim_buf,      "prim"      },
         { (uint64_t)P_max * BBOX_SZ,                       &arg.bbox_addr,      &bbox_buf,      "bbox"      },
         { (uint64_t)P_max * 4,                             &arg.bcount_addr,    &bcount_buf,    "bcount"    },
         { (uint64_t)(P_max + 1) * 4,                       &arg.boffset_addr,   &boffset_buf,   "boffset"   },
         { (uint64_t)keys_cap * 4,                          &arg.keys_addr,      &keys_buf,      "keys"      },
         { (uint64_t)T * 4,                                 &arg.btsum_addr,     &btsum_buf,     "btsum"     },
         { (uint64_t)T * num_bins * 4,                      &arg.thist_addr,     &thist_buf,     "thist"     },
         { (uint64_t)num_bins * 4,                          &arg.bincount_addr,  &bincount_buf,  "bincount"  },
         { (uint64_t)num_bins * 4,                          &arg.binbase_addr,   &binbase_buf,   "binbase"   },
         { (uint64_t)TILEBUF_SZ,                            &arg.tilebuf_addr,   &tilebuf_buf,   "tilebuf"   },
         { (uint64_t)3 * 4,                                 &arg.meta_addr,      &meta_buf,      "meta"      },
      };
      for (auto &s : spec) {
         VP_CHECK(vx_buffer_create(dev, s.bytes ? s.bytes : 1, 0, s.h), s.what);
         VP_CHECK(vx_buffer_address(*s.h, s.addr), s.what);
      }

      /* on-device vertex assembly: expand the resident VS records into the
       * setup_vertex_t[] the front end consumes (verts_buf = arg.verts_addr),
       * replacing the host readback + expansion + re-upload. */
      expand_arg_t earg{};
      earg.vsrec_addr   = vsrec_addr;
      earg.verts_addr   = arg.verts_addr;
      earg.num_verts    = vertex_count;
      earg.vstride      = layout->stride;
      uint32_t nvary = layout->num_varyings;
      if (nvary > EXPAND_MAX_VARYINGS) nvary = EXPAND_MAX_VARYINGS;
      earg.num_varyings = nvary;
      for (uint32_t i = 0; i < nvary; i++)
         earg.varying_comps[i] = layout->varying_comps[i];

      vx_launch_info_t eli = vx_launch_info_t{};
      eli.struct_size = sizeof(eli);
      eli.kernel      = k_expand;
      eli.args_host   = &earg;
      eli.args_size   = sizeof(earg);
      eli.ndim        = 1;
      eli.grid_dim[0]  = G;
      eli.block_dim[0] = T;
      vx_event_h ev_x = NULL;
      VP_CHECK(vx_enqueue_launch(q, &eli, 0, NULL, &ev_x),
               "vx_enqueue_launch(expand)");

      /* CP-sequenced front end: nine chained launches over resident memory
       * (setup_k stages 0-2, binning_k stages 3-8). multi-CTA (grid=G)
       * except the three single-CTA scans. Each launch waits on the prior
       * stage's event — the launch drain is the device barrier. */
      const uint32_t NSTAGE = 9;
      const uint32_t sgrid[NSTAGE] = { G, 1, G,  G, 1, G,  G, 1, G };
      pipe_arg_t      kargs[NSTAGE];
      vx_launch_info_t li[NSTAGE];
      vx_event_h       ev[NSTAGE] = {};
      for (uint32_t s = 0; s < NSTAGE; ++s) {
         kargs[s] = arg; kargs[s].stage = s;
         li[s] = vx_launch_info_t{};
         li[s].struct_size = sizeof(li[s]);
         li[s].kernel      = (s < PIPE_STAGE_BCOUNT) ? k_setup : k_binning;
         li[s].args_host   = &kargs[s];
         li[s].args_size   = sizeof(pipe_arg_t);
         li[s].ndim        = 1;
         li[s].grid_dim[0]  = sgrid[s];
         li[s].block_dim[0] = T;
         vx_event_h dep = s ? ev[s - 1] : ev_x;
         VP_CHECK(vx_enqueue_launch(q, &li[s], 1, &dep, &ev[s]),
                  "vx_enqueue_launch(frontend)");
      }

      /* drain the front end, then free its code module so the FS module can
       * load at the shared STARTUP_ADDR. The resident prim + tilebuf the
       * front end produced are separate allocations and survive the swap. */
      VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish(frontend)");
      vx_kernel_release(k_binning); k_binning = NULL;
      vx_kernel_release(k_setup);   k_setup   = NULL;
      vx_kernel_release(k_expand);  k_expand  = NULL;
      vx_module_release(femod);     femod     = NULL;

      /* fragment-shader module (per-draw .vxbin) */
      VP_CHECK(vx_module_load_file(dev, vxpath, &kmod), "vx_module_load_file(fs)");
      VP_CHECK(vx_module_get_kernel(kmod, "main", &kbuf), "vx_module_get_kernel(fs)");

      /* ---- RASTER + FS + OM over the device-produced buffers ---------- */
      uint64_t prim_dev = arg.prim_addr;
      uint64_t tile_dev = arg.tilebuf_addr;

      VP_CHECK(vx_buffer_create(dev, cbuf_bytes, 0, &cbuf), "vx_buffer_create(color)");
      VP_CHECK(vx_buffer_create(dev, cbuf_bytes, 0, &zbuf), "vx_buffer_create(depth)");
      uint64_t color_dev = 0, depth_dev = 0;
      VP_CHECK(vx_buffer_address(cbuf, &color_dev), "vx_buffer_address(color)");
      VP_CHECK(vx_buffer_address(zbuf, &depth_dev), "vx_buffer_address(depth)");

      /* the FS kernel only needs the primitive buffer; the OM reaches the
       * colour/depth buffers through its DCRs. */
      uint64_t argblk[8] = { 0 };
      argblk[0] = prim_dev;

      /* clear depth to the far value (GREATER/GEQUAL clear to 0, else max) */
      uint8_t zfill = (om->depth_func == VX_OM_DEPTH_FUNC_GREATER ||
                       om->depth_func == VX_OM_DEPTH_FUNC_GEQUAL) ? 0x00 : 0xFF;
      std::vector<uint8_t> zclear(cbuf_bytes, zfill);
      std::vector<uint32_t> texbuf;   /* outlives vx_queue_finish (async write) */

      VP_CHECK(vx_enqueue_write(q, cbuf, 0, color, cbuf_bytes, 0, NULL, NULL),
               "vx_enqueue_write(color)");
      VP_CHECK(vx_enqueue_write(q, zbuf, 0, zclear.data(), cbuf_bytes, 0, NULL, NULL),
               "vx_enqueue_write(depth)");

      /* RASTER DCRs: device-produced tile + primitive buffers; TILE_COUNT
       * is the dense grid count (host-known, no readback). */
      DCR(VX_DCR_RASTER_TBUF_ADDR,   tile_dev / 64);
      DCR(VX_DCR_RASTER_TILE_COUNT,  num_bins);
      DCR(VX_DCR_RASTER_PBUF_ADDR,   prim_dev / 64);
      DCR(VX_DCR_RASTER_PBUF_STRIDE, VP_RAST_PRIM_STRIDE);
      DCR(VX_DCR_RASTER_SCISSOR_X,   (width  << 16) | 0);
      DCR(VX_DCR_RASTER_SCISSOR_Y,   (height << 16) | 0);

      /* OM DCRs: colour + depth buffers, depth test, blend. Stencil
       * disabled for gfx-v1 (ALWAYS / KEEP). */
      DCR(VX_DCR_OM_CBUF_ADDR,        color_dev / 64);
      DCR(VX_DCR_OM_CBUF_PITCH,       width * 4);
      DCR(VX_DCR_OM_CBUF_WRITEMASK,   om->colormask);
      DCR(VX_DCR_OM_ZBUF_ADDR,        depth_dev / 64);
      DCR(VX_DCR_OM_ZBUF_PITCH,       width * 4);
      DCR(VX_DCR_OM_DEPTH_FUNC,
          om->depth_test ? om->depth_func : VX_OM_DEPTH_FUNC_ALWAYS);
      DCR(VX_DCR_OM_DEPTH_WRITEMASK,
          (om->depth_test && om->depth_write) ? 1u : 0u);
      DCR(VX_DCR_OM_STENCIL_FUNC,      VX_OM_DEPTH_FUNC_ALWAYS);
      DCR(VX_DCR_OM_STENCIL_ZPASS,     VX_OM_STENCIL_OP_KEEP);
      DCR(VX_DCR_OM_STENCIL_ZFAIL,     VX_OM_STENCIL_OP_KEEP);
      DCR(VX_DCR_OM_STENCIL_FAIL,      VX_OM_STENCIL_OP_KEEP);
      DCR(VX_DCR_OM_STENCIL_REF,       0);
      DCR(VX_DCR_OM_STENCIL_MASK,      0xFF);
      DCR(VX_DCR_OM_STENCIL_WRITEMASK, 0);
      DCR(VX_DCR_OM_BLEND_MODE,        om->blend_mode);
      DCR(VX_DCR_OM_BLEND_FUNC,        om->blend_func);
      DCR(VX_DCR_OM_BLEND_CONST,       0);
      DCR(VX_DCR_OM_LOGIC_OP,          0);

      /* TEX unit (stage 0): convert the bound texture from R8G8B8A8 to the
       * A8R8G8B8 word the TEX unit unpacks, upload, program the sampler.
       * Untextured draws (tex == NULL) leave TEX state alone. */
      if (tex) {
         const uint32_t texel_count = tex->width * tex->height;
         texbuf.resize(texel_count);
         const uint8_t *src = static_cast<const uint8_t *>(tex->pixels);
         for (uint32_t i = 0; i < texel_count; i++) {
            uint32_t r = src[i * 4 + 0];
            uint32_t g = src[i * 4 + 1];
            uint32_t b = src[i * 4 + 2];
            uint32_t a = src[i * 4 + 3];
            texbuf[i] = (a << 24) | (r << 16) | (g << 8) | b;
         }
         const size_t texbuf_bytes = (size_t)texel_count * 4;

         VP_CHECK(vx_buffer_create(dev, texbuf_bytes, 0, &xbuf), "vx_buffer_create(tex)");
         uint64_t tex_dev = 0;
         VP_CHECK(vx_buffer_address(xbuf, &tex_dev), "vx_buffer_address(tex)");
         VP_CHECK(vx_enqueue_write(q, xbuf, 0, texbuf.data(), texbuf_bytes,
                                   0, NULL, NULL), "vx_enqueue_write(tex)");

         uint32_t logw = vp_log2u(tex->width);
         uint32_t logh = vp_log2u(tex->height);
         DCR(VX_DCR_TEX_STAGE,        0);
         DCR(VX_DCR_TEX_LOGDIM,       (logh << 16) | logw);
         DCR(VX_DCR_TEX_FORMAT,       VX_TEX_FORMAT_A8R8G8B8);
         DCR(VX_DCR_TEX_FILTER,       tex->filter);
         DCR(VX_DCR_TEX_WRAP,         (tex->wrap_v << 16) | tex->wrap_u);
         DCR(VX_DCR_TEX_ADDR,         tex_dev / 64);
         DCR(VX_DCR_TEX_MIPOFF_BASE,  0);
      }

      /* dispatch the fragment-shader kernel; threads poll vx_rast(). The
       * front end already drained (queue_finish above), so RASTER sees the
       * resident tilebuf + primbuf. Fill every HW lane: block = threads ×
       * warps (one CTA/core), grid = cores. */
      uint64_t nt = 1, nw = 1, nc = 1;
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt), "vx_device_query(NUM_THREADS)");
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &nw), "vx_device_query(NUM_WARPS)");
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,   &nc), "vx_device_query(NUM_CORES)");
      vx_launch_info_t fli = {
         sizeof(fli), NULL, kbuf, argblk, sizeof(argblk), 1,
         { (uint32_t)nc, 1, 1 }, { (uint32_t)(nt * nw), 1, 1 }, 0,
      };
      VP_CHECK(vx_enqueue_launch(q, &fli, 0, NULL, NULL), "vx_enqueue_launch(fs)");

      VP_CHECK(vx_enqueue_read(q, color, cbuf, 0, cbuf_bytes, 0, NULL, NULL),
               "vx_enqueue_read(color)");
      VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");
      ok = true;
   }

done:
   {
      vx_buffer_h all[] = {
         verts_buf, slot_prim_buf, slot_bbox_buf, keep_buf, offset_buf,
         tsum_buf, prim_buf, bbox_buf, bcount_buf, boffset_buf, keys_buf,
         btsum_buf, thist_buf, bincount_buf, binbase_buf, tilebuf_buf,
         meta_buf, cbuf, zbuf, xbuf,
      };
      for (vx_buffer_h b : all)
         if (b) vx_buffer_release(b);
   }
   if (k_binning) vx_kernel_release(k_binning);
   if (k_setup)   vx_kernel_release(k_setup);
   if (k_expand)  vx_kernel_release(k_expand);
   if (kbuf)      vx_kernel_release(kbuf);
   if (femod)     vx_module_release(femod);
   if (kmod)      vx_module_release(kmod);
   if (q)         vx_queue_release(q);
   unlink(vxpath);
   return ok;
}
