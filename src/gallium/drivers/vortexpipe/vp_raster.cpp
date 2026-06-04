/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_raster -- Phase 4/5: hardware-rasterizer + output-merger draw
 * orchestration.
 *
 * graphics::Binning() does triangle setup + tile binning; vp_raster_draw
 * uploads the tile/primitive buffers, programs the RASTER and OM DCRs,
 * and dispatches the fragment-shader kernel. The kernel polls vx_rast()
 * for quads and submits shaded fragments to the OM unit via vx_om(),
 * which depth-tests, blends and writes the colour + depth buffers.
 */

#include "vp_raster.h"

#include <vector>
#include <unordered_map>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "graphics.h"            /* graphics::Binning + on-wire types */
#include "VX_types.h"            /* VX_DCR_RASTER_*, VX_DCR_OM_*, VX_OM_* */
#include "util/log.h"

namespace graphics = vortex::graphics;

/* Tile size must match the hardware (VX_config.h RASTER_TILE_LOGSIZE). */
#ifndef RASTER_TILE_LOGSIZE
#define RASTER_TILE_LOGSIZE 5
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
               const void *xverts, uint32_t vertex_count,
               const struct vp_vs_layout *layout,
               void *color, uint32_t width, uint32_t height,
               const struct vp_om_params *om,
               const struct vp_tex_params *tex)
{
   /* ---- triangle setup + binning ---------------------------------- *
    * Convert the VS output records to graphics::vertex_t: record slot
    * 0 is the clip-space position, slots 1.. the generic varyings. The
    * RASTER unit interpolates a fixed colour + texcoord; each varying
    * is routed by its component count (2 -> texcoord, 3/4 -> colour),
    * the same gfx-v1 mapping the FS translator uses. */
   std::unordered_map<uint32_t, graphics::vertex_t> verts;
   std::vector<graphics::primitive_t> prims;
   const uint8_t *xv = static_cast<const uint8_t *>(xverts);

   for (uint32_t i = 0; i < vertex_count; i++) {
      const uint8_t *rec = xv + (size_t)i * layout->stride;
      const float *pos = reinterpret_cast<const float *>(rec);
      graphics::vertex_t v;
      v.pos[0] = pos[0]; v.pos[1] = pos[1];
      v.pos[2] = pos[2]; v.pos[3] = pos[3];
      v.color[0] = 1.0f; v.color[1] = 1.0f;
      v.color[2] = 1.0f; v.color[3] = 1.0f;
      v.texcoord[0] = 0.0f; v.texcoord[1] = 0.0f;
      for (uint32_t vi = 0; vi < layout->num_varyings; vi++) {
         const float *a = reinterpret_cast<const float *>(
            rec + 16u * (1u + vi));            /* slot 0 is gl_Position */
         uint32_t nc = layout->varying_comps[vi];
         if (nc == 2) {
            v.texcoord[0] = a[0]; v.texcoord[1] = a[1];
         } else if (nc >= 3) {
            v.color[0] = a[0]; v.color[1] = a[1];
            v.color[2] = a[2]; v.color[3] = nc >= 4 ? a[3] : 1.0f;
         }
      }
      verts[i] = v;
   }
   for (uint32_t i = 0; i + 2 < vertex_count; i += 3)
      prims.push_back({ i, i + 1, i + 2 });

   std::vector<uint8_t> tilebuf, primbuf;
   uint32_t num_tiles = graphics::Binning(tilebuf, primbuf, verts, prims,
                                          width, height, 0.0f, 1.0f,
                                          RASTER_TILE_LOGSIZE);
   if (num_tiles == 0 || primbuf.empty()) {
      mesa_logw("vortexpipe: raster: binning produced no tiles");
      return false;
   }

   /* ---- dispatch on the RASTER + OM units ------------------------- */
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_module_h kmod = NULL;
   vx_kernel_h kbuf = NULL;
   vx_buffer_h tbuf = NULL, pbuf = NULL,
               cbuf = NULL, zbuf = NULL, xbuf = NULL;
   char vxpath[] = "/tmp/vortexpipe-fs.XXXXXX";
   int  vxfd = -1;
   const uint32_t cbuf_bytes = width * height * 4;

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
      vx_queue_info_t qi = {
         sizeof(qi), NULL, VX_QUEUE_PRIORITY_NORMAL, 0,
      };
      VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");
      VP_CHECK(vx_module_load_file(dev, vxpath, &kmod),
               "vx_module_load_file");
      /* "main" is the public name vxbin.py assigns the single conventional
       * kernel (the C entry is "kernel_main"); match the native runtime. */
      VP_CHECK(vx_module_get_kernel(kmod, "main", &kbuf),
               "vx_module_get_kernel");

      /* tile / primitive / colour / depth device buffers */
      VP_CHECK(vx_buffer_create(dev, tilebuf.size(), 0, &tbuf),
               "vx_buffer_create(tiles)");
      VP_CHECK(vx_buffer_create(dev, primbuf.size(), 0, &pbuf),
               "vx_buffer_create(prims)");
      VP_CHECK(vx_buffer_create(dev, cbuf_bytes, 0, &cbuf),
               "vx_buffer_create(color)");
      VP_CHECK(vx_buffer_create(dev, cbuf_bytes, 0, &zbuf),
               "vx_buffer_create(depth)");

      uint64_t tile_dev = 0, prim_dev = 0, color_dev = 0, depth_dev = 0;
      VP_CHECK(vx_buffer_address(tbuf, &tile_dev),  "vx_buffer_address(tiles)");
      VP_CHECK(vx_buffer_address(pbuf, &prim_dev),  "vx_buffer_address(prims)");
      VP_CHECK(vx_buffer_address(cbuf, &color_dev), "vx_buffer_address(color)");
      VP_CHECK(vx_buffer_address(zbuf, &depth_dev), "vx_buffer_address(depth)");

      /* the FS kernel only needs the primitive buffer; the OM reaches
       * the colour/depth buffers through its DCRs. */
      uint64_t argblk[8] = { 0 };
      argblk[0] = prim_dev;

      /* clear the depth buffer to the far value (GREATER/GEQUAL clear
       * to 0, every other compare to max). */
      uint8_t zfill = (om->depth_func == VX_OM_DEPTH_FUNC_GREATER ||
                       om->depth_func == VX_OM_DEPTH_FUNC_GEQUAL)
                         ? 0x00 : 0xFF;
      std::vector<uint8_t> zclear(cbuf_bytes, zfill);

      /* The converted texture words. vx_enqueue_write is asynchronous --
       * the source memory must stay live until vx_queue_finish -- so the
       * texel buffer is declared at this scope, alongside zclear, rather
       * than inside the `if (tex)` block below. */
      std::vector<uint32_t> texbuf;

      /* upload everything (the colour buffer carries the cleared image) */
      VP_CHECK(vx_enqueue_write(q, tbuf, 0, tilebuf.data(), tilebuf.size(),
                                0, NULL, NULL), "vx_enqueue_write(tiles)");
      VP_CHECK(vx_enqueue_write(q, pbuf, 0, primbuf.data(), primbuf.size(),
                                0, NULL, NULL), "vx_enqueue_write(prims)");
      VP_CHECK(vx_enqueue_write(q, cbuf, 0, color, cbuf_bytes,
                                0, NULL, NULL), "vx_enqueue_write(color)");
      VP_CHECK(vx_enqueue_write(q, zbuf, 0, zclear.data(), cbuf_bytes,
                                0, NULL, NULL), "vx_enqueue_write(depth)");
      /* args passed inline via args_host below (no abuf needed) */

      /* RASTER DCRs (addresses are 64-byte block indices) */
      DCR(VX_DCR_RASTER_TBUF_ADDR,   tile_dev / 64);
      DCR(VX_DCR_RASTER_TILE_COUNT,  num_tiles);
      DCR(VX_DCR_RASTER_PBUF_ADDR,   prim_dev / 64);
      DCR(VX_DCR_RASTER_PBUF_STRIDE, VP_RAST_PRIM_STRIDE);
      DCR(VX_DCR_RASTER_SCISSOR_X,   (width  << 16) | 0);
      DCR(VX_DCR_RASTER_SCISSOR_Y,   (height << 16) | 0);

      /* OM DCRs: colour + depth buffers, depth test, blend. Stencil is
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

      /* TEX unit (stage 0): convert the bound texture from the Vulkan
       * R8G8B8A8 host layout to the A8R8G8B8 word the TEX unit unpacks,
       * upload it, and program the sampler DCRs. Untextured draws (tex
       * == NULL) leave the TEX state alone -- the FS kernel won't issue
       * vx_tex(). Only mip 0 is supplied; gfx-v1 fragment shading
       * samples at LOD 0. */
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
         const size_t texbuf_bytes = texel_count * 4;

         VP_CHECK(vx_buffer_create(dev, texbuf_bytes, 0, &xbuf),
                  "vx_buffer_create(tex)");
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
         DCR(VX_DCR_TEX_MIPOFF_BASE,  0);   /* mip 0 at the buffer base */
      }

      /* dispatch the fragment-shader kernel; threads poll vx_rast().
       * Args passed inline via args_host (see vp_launch).
       *
       * Fill every HW lane the device exposes so every warp on every
       * core races for vx_rast() pops. block_dim = num_threads ×
       * num_warps fills one CTA per core; grid_dim = num_cores
       * spreads CTAs across cores. (Previous shape grid=1/block=4
       * used 1 warp of 1 core — under 6% of a 4×4 device, less on
       * larger configs.) */
      uint64_t nt = 1, nw = 1, nc = 1;
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt),
               "vx_device_query(NUM_THREADS)");
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &nw),
               "vx_device_query(NUM_WARPS)");
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,   &nc),
               "vx_device_query(NUM_CORES)");
      vx_launch_info_t li = {
         sizeof(li), NULL, kbuf, argblk, sizeof(argblk), 1,
         { (uint32_t)nc, 1, 1 }, { (uint32_t)(nt * nw), 1, 1 }, 0,
      };
      VP_CHECK(vx_enqueue_launch(q, &li, 0, NULL, NULL), "vx_enqueue_launch");

      VP_CHECK(vx_enqueue_read(q, color, cbuf, 0, cbuf_bytes, 0, NULL, NULL),
               "vx_enqueue_read(color)");
      VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");
      ok = true;
   }

done:
   if (xbuf) vx_buffer_release(xbuf);
   if (zbuf) vx_buffer_release(zbuf);
   if (cbuf) vx_buffer_release(cbuf);
   if (pbuf) vx_buffer_release(pbuf);
   if (tbuf) vx_buffer_release(tbuf);
   if (kbuf) vx_kernel_release(kbuf);
   if (kmod) vx_module_release(kmod);
   if (q)    vx_queue_release(q);
   unlink(vxpath);
   return ok;
}
