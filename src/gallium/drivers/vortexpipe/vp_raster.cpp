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
#include <new>
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

/* exact log2 of a power-of-two texture dimension */
static inline uint32_t
vp_log2u(uint32_t n)
{
   uint32_t l = 0;
   while ((1u << l) < n)
      l++;
   return l;
}

/* ---- persistent front-end working set (§6.6) -------------------------- *
 * The 17 binning buffers laid out once and reused across the frame's draws
 * instead of allocated per draw. prim + tilebuf (the RASTER AXI master's
 * inputs) are pinned over VX_MEM_PHYS; the rest is device-resident scratch.
 * The pool grows monotonically when a draw needs more capacity. addr caches
 * the device addresses (its count fields are unused — set per draw). */
struct vp_raster_pool {
   vx_buffer_h bufs[17];
   pipe_arg_t  addr;
   uint32_t    cap_tris;
   uint32_t    cap_bins;
   uint32_t    cap_keys;
   uint32_t    cap_T;
};

extern "C" struct vp_raster_pool *
vp_raster_pool_create(void)
{
   return new (std::nothrow) vp_raster_pool{};
}

extern "C" void
vp_raster_pool_destroy(struct vp_raster_pool *pool)
{
   if (!pool)
      return;
   for (vx_buffer_h b : pool->bufs)
      if (b) vx_buffer_release(b);
   delete pool;
}

/* Ensure the pool is sized for this draw (grow-only); (re)allocates and
 * re-caches addresses only when the request exceeds the current capacity.
 * Returns false on an allocation failure. */
static bool
vp_pool_ensure(vx_device_h dev, struct vp_raster_pool *pool,
               uint32_t num_tris, uint32_t num_bins,
               uint32_t keys_cap, uint32_t T)
{
   if (pool->bufs[0] &&
       num_tris <= pool->cap_tris && num_bins <= pool->cap_bins &&
       keys_cap <= pool->cap_keys && T == pool->cap_T)
      return true;   /* fits — reuse the resident set */

   for (vx_buffer_h &b : pool->bufs) { if (b) vx_buffer_release(b); b = NULL; }

   const uint32_t NT = num_tris > pool->cap_tris ? num_tris : pool->cap_tris;
   const uint32_t B  = num_bins > pool->cap_bins ? num_bins : pool->cap_bins;
   const uint32_t K  = keys_cap > pool->cap_keys ? keys_cap : pool->cap_keys;
   const uint32_t MS = SETUP_MAX_SUB;
   const uint32_t P_max = NT * MS;
   const size_t PRIM_SZ = sizeof(graphics::rast_prim_t);
   const size_t HDR_SZ  = sizeof(graphics::rast_bin_header_t);
   const size_t BBOX_SZ = sizeof(setup_bbox_t);
   const size_t TILEBUF_SZ = (size_t)B * HDR_SZ + (size_t)K * 4;

   const uint32_t W   = 0;                                          /* scratch */
   const uint32_t RWP = VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS;   /* RASTER-read */
   pipe_arg_t &a = pool->addr;
   struct { uint64_t bytes; uint64_t *addr; uint32_t flags; } spec[] = {
      { (uint64_t)3 * NT * sizeof(setup_vertex_t), &a.verts_addr,     W   },
      { (uint64_t)P_max * PRIM_SZ,                 &a.slot_prim_addr, W   },
      { (uint64_t)P_max * BBOX_SZ,                 &a.slot_bbox_addr, W   },
      { (uint64_t)NT * 4,                          &a.keep_addr,      W   },
      { (uint64_t)(NT + 1) * 4,                    &a.offset_addr,    W   },
      { (uint64_t)T * 4,                           &a.tsum_addr,      W   },
      { (uint64_t)P_max * PRIM_SZ,                 &a.prim_addr,      RWP },
      { (uint64_t)P_max * BBOX_SZ,                 &a.bbox_addr,      W   },
      { (uint64_t)P_max * 4,                       &a.bcount_addr,    W   },
      { (uint64_t)(P_max + 1) * 4,                 &a.boffset_addr,   W   },
      { (uint64_t)K * 4,                           &a.keys_addr,      W   },
      { (uint64_t)T * 4,                           &a.btsum_addr,     W   },
      { (uint64_t)T * B * 4,                       &a.thist_addr,     W   },
      { (uint64_t)B * 4,                           &a.bincount_addr,  W   },
      { (uint64_t)B * 4,                           &a.binbase_addr,   W   },
      { (uint64_t)TILEBUF_SZ,                      &a.tilebuf_addr,   RWP },
      { (uint64_t)3 * 4,                           &a.meta_addr,      W   },
   };
   for (uint32_t i = 0; i < 17; i++) {
      if (vx_buffer_create(dev, spec[i].bytes ? spec[i].bytes : 1,
                           spec[i].flags, &pool->bufs[i]) != VX_SUCCESS) {
         mesa_loge("vortexpipe: raster: front-end pool alloc failed");
         return false;
      }
      if (vx_buffer_address(pool->bufs[i], spec[i].addr) != VX_SUCCESS) {
         mesa_loge("vortexpipe: raster: front-end pool address failed");
         return false;
      }
   }
   pool->cap_tris = NT; pool->cap_bins = B; pool->cap_keys = K; pool->cap_T = T;
   return true;
}

extern "C" bool
vp_raster_draw(vx_device_h dev, struct vp_raster_pool *pool,
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
   if (!pool) {
      mesa_loge("vortexpipe: raster: no front-end pool");
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

   /* The front-end resident buffer set lives in the persistent pool; only the
    * per-draw colour/depth/texture buffers are owned here. */
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

      /* Load BOTH code modules resident: the per-draw FS at the default
       * STARTUP_ADDR (0x80000000) and the embedded front end at its distinct
       * base (kernels/gfx_frontend linked at 0x80200000). Co-residence lets
       * one CP command batch sequence expand -> setup -> binning -> raster ->
       * FS with no host module swap between stages (§6.4). */
      VP_CHECK(vx_module_load_file(dev, vxpath, &kmod), "vx_module_load_file(fs)");
      VP_CHECK(vx_module_get_kernel(kmod, "main", &kbuf), "vx_module_get_kernel(fs)");

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

      /* Lay out (or reuse) the persistent front-end working set, then build
       * the shared pipe_arg_t from the pool's cached device addresses plus
       * this draw's counts. */
      if (!vp_pool_ensure(dev, pool, num_tris, num_bins, keys_cap, T)) {
         mesa_loge("vortexpipe: raster: front-end pool sizing failed");
         goto done;
      }
      pipe_arg_t arg = pool->addr;               /* device addresses */
      arg.num_tris   = num_tris;
      arg.stage      = 0;
      arg.width      = width;
      arg.height     = height;
      arg.bin_stripe = (num_bins + G - 1) / G;   /* bins per CTA (HIST/SCATTER) */
      arg.bin_cols   = bin_cols;
      arg.num_bins   = num_bins;
      arg.cull_mode  = SETUP_CULL_NONE;          /* gfx-v1 two-sided default */

      /* expand_k arg: on-device vertex assembly expands the resident VS
       * records into the setup_vertex_t[] the front end consumes
       * (arg.verts_addr). */
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

      /* ---- colour / depth / texture buffers + uploads ----------------- *
       * The DMA uploads precede the command batch on the same queue (FIFO);
       * the batch itself is launches + DCR writes only. */
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
      uint64_t tex_dev = 0;

      VP_CHECK(vx_enqueue_write(q, cbuf, 0, color, cbuf_bytes, 0, NULL, NULL),
               "vx_enqueue_write(color)");
      VP_CHECK(vx_enqueue_write(q, zbuf, 0, zclear.data(), cbuf_bytes, 0, NULL, NULL),
               "vx_enqueue_write(depth)");
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
         VP_CHECK(vx_buffer_create(dev, (size_t)texel_count * 4, 0, &xbuf),
                  "vx_buffer_create(tex)");
         VP_CHECK(vx_buffer_address(xbuf, &tex_dev), "vx_buffer_address(tex)");
         VP_CHECK(vx_enqueue_write(q, xbuf, 0, texbuf.data(), (size_t)texel_count * 4,
                                   0, NULL, NULL), "vx_enqueue_write(tex)");
      }

      /* FS launch fills every HW lane: block = threads × warps (one CTA/core),
       * grid = cores. */
      uint64_t nt = 1, nw = 1, nc = 1;
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt), "vx_device_query(NUM_THREADS)");
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &nw), "vx_device_query(NUM_WARPS)");
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,   &nc), "vx_device_query(NUM_CORES)");

      /* Launch descriptors + per-stage args. These must outlive
       * vx_enqueue_commands (the runtime copies them at submit). */
      const uint32_t NSTAGE = 9;
      const uint32_t sgrid[NSTAGE] = { G, 1, G,  G, 1, G,  G, 1, G };
      pipe_arg_t       kargs[NSTAGE];
      vx_launch_info_t eli{}, li[NSTAGE], fli{};

      eli.struct_size = sizeof(eli); eli.kernel = k_expand;
      eli.args_host = &earg; eli.args_size = sizeof(earg);
      eli.ndim = 1; eli.grid_dim[0] = G; eli.block_dim[0] = T;

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
      }

      fli.struct_size = sizeof(fli); fli.kernel = kbuf;
      fli.args_host = argblk; fli.args_size = sizeof(argblk);
      fli.ndim = 1; fli.grid_dim[0] = (uint32_t)nc;
      fli.block_dim[0] = (uint32_t)(nt * nw);
      /* FWD-5: vx_frag_fetch stages the payload into the gfx window, not LMEM. */

      /* The whole draw as ONE CP command batch (§6.4): expand -> setup ->
       * binning, then RASTER/OM/TEX config, then FS. The CP retires the
       * commands in order, draining each launch before the next (the
       * inter-stage device barrier), so the host is untouched between stages
       * — no per-launch round-trip, no module swap. RASTER's device-produced
       * tile + prim buffers feed it; TILE_COUNT is the dense grid count
       * (host-known, no readback). Stencil disabled for gfx-v1 (ALWAYS/KEEP). */
      std::vector<vx_command_t> cmds;
      auto LAUNCH = [&](const vx_launch_info_t *l) {
         vx_command_t c{}; c.type = VX_COMMAND_LAUNCH; c.data.launch = l;
         cmds.push_back(c);
      };
      auto DCRW = [&](uint32_t a, uint32_t v) {
         vx_command_t c{}; c.type = VX_COMMAND_DCR_WRITE;
         c.data.dcr.addr = a; c.data.dcr.value = v; cmds.push_back(c);
      };

      LAUNCH(&eli);
      for (uint32_t s = 0; s < NSTAGE; ++s) LAUNCH(&li[s]);

      DCRW(VX_DCR_RASTER_TBUF_ADDR,   (uint32_t)(tile_dev / 64));
      DCRW(VX_DCR_RASTER_TILE_COUNT,  num_bins);
      DCRW(VX_DCR_RASTER_PBUF_ADDR,   (uint32_t)(prim_dev / 64));
      DCRW(VX_DCR_RASTER_PBUF_STRIDE, VP_RAST_PRIM_STRIDE);
      DCRW(VX_DCR_RASTER_SCISSOR_X,   (width  << 16) | 0);
      DCRW(VX_DCR_RASTER_SCISSOR_Y,   (height << 16) | 0);

      DCRW(VX_DCR_OM_CBUF_ADDR,        (uint32_t)(color_dev / 64));
      DCRW(VX_DCR_OM_CBUF_PITCH,       width * 4);
      DCRW(VX_DCR_OM_CBUF_WRITEMASK,   om->colormask);
      DCRW(VX_DCR_OM_ZBUF_ADDR,        (uint32_t)(depth_dev / 64));
      DCRW(VX_DCR_OM_ZBUF_PITCH,       width * 4);
      DCRW(VX_DCR_OM_DEPTH_FUNC,       om->depth_test ? om->depth_func : VX_OM_DEPTH_FUNC_ALWAYS);
      DCRW(VX_DCR_OM_DEPTH_WRITEMASK,  (om->depth_test && om->depth_write) ? 1u : 0u);
      DCRW(VX_DCR_OM_STENCIL_FUNC,      VX_OM_DEPTH_FUNC_ALWAYS);
      DCRW(VX_DCR_OM_STENCIL_ZPASS,     VX_OM_STENCIL_OP_KEEP);
      DCRW(VX_DCR_OM_STENCIL_ZFAIL,     VX_OM_STENCIL_OP_KEEP);
      DCRW(VX_DCR_OM_STENCIL_FAIL,      VX_OM_STENCIL_OP_KEEP);
      DCRW(VX_DCR_OM_STENCIL_REF,       0);
      DCRW(VX_DCR_OM_STENCIL_MASK,      0xFF);
      DCRW(VX_DCR_OM_STENCIL_WRITEMASK, 0);
      DCRW(VX_DCR_OM_BLEND_MODE,        om->blend_mode);
      DCRW(VX_DCR_OM_BLEND_FUNC,        om->blend_func);
      DCRW(VX_DCR_OM_BLEND_CONST,       0);
      DCRW(VX_DCR_OM_LOGIC_OP,          0);

      /* TEX unit (stage 0): the texels are already uploaded above (A8R8G8B8);
       * program the sampler. Untextured draws leave TEX state alone. */
      if (tex) {
         uint32_t logw = vp_log2u(tex->width);
         uint32_t logh = vp_log2u(tex->height);
         DCRW(VX_DCR_TEX_STAGE,        0);
         DCRW(VX_DCR_TEX_LOGDIM,       (logh << 16) | logw);
         DCRW(VX_DCR_TEX_FORMAT,       VX_TEX_FORMAT_A8R8G8B8);
         DCRW(VX_DCR_TEX_FILTER,       tex->filter);
         DCRW(VX_DCR_TEX_WRAP,         (tex->wrap_v << 16) | tex->wrap_u);
         DCRW(VX_DCR_TEX_ADDR,         (uint32_t)(tex_dev / 64));
         DCRW(VX_DCR_TEX_MIPOFF_BASE,  0);
      }

      LAUNCH(&fli);

      VP_CHECK(vx_enqueue_commands(q, cmds.data(), (uint32_t)cmds.size(),
                                   0, NULL, NULL), "vx_enqueue_commands");

      VP_CHECK(vx_enqueue_read(q, color, cbuf, 0, cbuf_bytes, 0, NULL, NULL),
               "vx_enqueue_read(color)");
      VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");
      ok = true;
   }

done:
   /* the front-end working set is pool-owned and persists; only the per-draw
    * colour / depth / texture buffers are released here. */
   if (xbuf) vx_buffer_release(xbuf);
   if (zbuf) vx_buffer_release(zbuf);
   if (cbuf) vx_buffer_release(cbuf);
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
