/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_raster -- Phase 4 Step 3: hardware-rasterizer draw orchestration.
 *
 * graphics::Binning() does the triangle setup + tile binning -- the
 * same host-side step the draw3d/raster regression tests use -- and
 * produces the tile + primitive buffers the RASTER unit consumes.
 * vp_raster_draw() uploads those, programs the RASTER DCRs, and
 * dispatches the fragment-shader kernel, which polls vx_rast() for
 * quads and writes the colour buffer.
 */

#include "vp_raster.h"

#include <vector>
#include <unordered_map>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "gfxutil.h"             /* graphics::Binning, cocogfx::CGLTrace */
#include "util/log.h"

/* RASTER DCRs (VX_types.h). */
#define VX_DCR_RASTER_TBUF_ADDR    0x040
#define VX_DCR_RASTER_TILE_COUNT   0x041
#define VX_DCR_RASTER_PBUF_ADDR    0x042
#define VX_DCR_RASTER_PBUF_STRIDE  0x043
#define VX_DCR_RASTER_SCISSOR_X    0x044
#define VX_DCR_RASTER_SCISSOR_Y    0x045

/* Tile size must match the hardware (VX_config.h RASTER_TILE_LOGSIZE). */
#ifndef RASTER_TILE_LOGSIZE
#define RASTER_TILE_LOGSIZE 5
#endif

/* sizeof(graphics::rast_prim_t): vec3e_t edges[3] + rast_attribs_t. */
#define VP_RAST_PRIM_STRIDE 120

#define VP_CHECK(call, what)                                            \
   do {                                                                 \
      vx_result_t _r = (call);                                          \
      if (_r != VX_SUCCESS) {                                           \
         mesa_logw("vortexpipe: raster: %s failed (%s)", (what),        \
                   vx_result_string(_r));                               \
         goto done;                                                     \
      }                                                                 \
   } while (0)

extern "C" bool
vp_raster_draw(vx_device_h dev,
               const void *fs_vxbin, size_t fs_vxbin_size,
               const void *xverts, uint32_t vertex_count,
               const struct vp_vs_layout *layout,
               void *color, uint32_t width, uint32_t height)
{
   /* ---- triangle setup + binning ---------------------------------- *
    * Convert the VS output records to CGLTrace vertices: slot 0 is the
    * clip-space position, slot 1 the colour varying. */
   std::unordered_map<uint32_t, cocogfx::CGLTrace::vertex_t> verts;
   std::vector<cocogfx::CGLTrace::primitive_t> prims;
   const uint8_t *xv = static_cast<const uint8_t *>(xverts);

   for (uint32_t i = 0; i < vertex_count; i++) {
      const float *pos = reinterpret_cast<const float *>(
         xv + (size_t)i * layout->stride);
      const float *col = reinterpret_cast<const float *>(
         xv + (size_t)i * layout->stride + 16);  /* varying slot 0 */
      cocogfx::CGLTrace::vertex_t v;
      v.pos      = { pos[0], pos[1], pos[2], pos[3] };
      v.color    = { col[0], col[1], col[2], 1.0f };
      v.texcoord = { 0.0f, 0.0f };
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

   /* ---- dispatch on the RASTER unit ------------------------------- */
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_buffer_h kbuf = NULL, tbuf = NULL, pbuf = NULL,
               cbuf = NULL, abuf = NULL;
   char vxpath[] = "/tmp/vortexpipe-fs.XXXXXX";
   int  vxfd = -1;
   const uint32_t cbuf_bytes = width * height * 4;

   vxfd = mkstemp(vxpath);
   if (vxfd < 0) {
      mesa_logw("vortexpipe: raster: mkstemp failed");
      return false;
   }
   if (write(vxfd, fs_vxbin, fs_vxbin_size) != (ssize_t)fs_vxbin_size) {
      mesa_logw("vortexpipe: raster: writing .vxbin failed");
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
      VP_CHECK(vx_buffer_load_kernel_file(dev, q, vxpath, &kbuf),
               "vx_buffer_load_kernel_file");

      /* tile / primitive / colour device buffers */
      VP_CHECK(vx_buffer_create(dev, tilebuf.size(), 0, &tbuf),
               "vx_buffer_create(tiles)");
      VP_CHECK(vx_buffer_create(dev, primbuf.size(), 0, &pbuf),
               "vx_buffer_create(prims)");
      VP_CHECK(vx_buffer_create(dev, cbuf_bytes, 0, &cbuf),
               "vx_buffer_create(color)");

      uint64_t tile_dev = 0, prim_dev = 0, color_dev = 0;
      VP_CHECK(vx_buffer_address(tbuf, &tile_dev),  "vx_buffer_address(tiles)");
      VP_CHECK(vx_buffer_address(pbuf, &prim_dev),  "vx_buffer_address(prims)");
      VP_CHECK(vx_buffer_address(cbuf, &color_dev), "vx_buffer_address(color)");

      /* arg block: [0]=prim buffer, [1]=colour buffer, [2]=row pitch */
      uint64_t argblk[8] = { 0 };
      argblk[0] = prim_dev;
      argblk[1] = color_dev;
      argblk[2] = width * 4;
      VP_CHECK(vx_buffer_create(dev, sizeof(argblk), 0, &abuf),
               "vx_buffer_create(args)");

      /* upload everything (the colour buffer carries the cleared image) */
      VP_CHECK(vx_enqueue_write(q, tbuf, 0, tilebuf.data(), tilebuf.size(),
                                0, NULL, NULL), "vx_enqueue_write(tiles)");
      VP_CHECK(vx_enqueue_write(q, pbuf, 0, primbuf.data(), primbuf.size(),
                                0, NULL, NULL), "vx_enqueue_write(prims)");
      VP_CHECK(vx_enqueue_write(q, cbuf, 0, color, cbuf_bytes,
                                0, NULL, NULL), "vx_enqueue_write(color)");
      VP_CHECK(vx_enqueue_write(q, abuf, 0, argblk, sizeof(argblk),
                                0, NULL, NULL), "vx_enqueue_write(args)");

      /* program the RASTER DCRs (addresses are 64-byte block indices) */
      VP_CHECK(vx_enqueue_dcr_write(q, VX_DCR_RASTER_TBUF_ADDR,
                                    (uint32_t)(tile_dev / 64), 0, NULL, NULL),
               "dcr(TBUF_ADDR)");
      VP_CHECK(vx_enqueue_dcr_write(q, VX_DCR_RASTER_TILE_COUNT,
                                    num_tiles, 0, NULL, NULL),
               "dcr(TILE_COUNT)");
      VP_CHECK(vx_enqueue_dcr_write(q, VX_DCR_RASTER_PBUF_ADDR,
                                    (uint32_t)(prim_dev / 64), 0, NULL, NULL),
               "dcr(PBUF_ADDR)");
      VP_CHECK(vx_enqueue_dcr_write(q, VX_DCR_RASTER_PBUF_STRIDE,
                                    VP_RAST_PRIM_STRIDE, 0, NULL, NULL),
               "dcr(PBUF_STRIDE)");
      VP_CHECK(vx_enqueue_dcr_write(q, VX_DCR_RASTER_SCISSOR_X,
                                    (width << 16) | 0, 0, NULL, NULL),
               "dcr(SCISSOR_X)");
      VP_CHECK(vx_enqueue_dcr_write(q, VX_DCR_RASTER_SCISSOR_Y,
                                    (height << 16) | 0, 0, NULL, NULL),
               "dcr(SCISSOR_Y)");

      /* dispatch the fragment-shader kernel; threads poll vx_rast() */
      vx_launch_info_t li = {
         sizeof(li), NULL, kbuf, abuf, 1,
         { 1, 1, 1 }, { 4, 1, 1 }, 0,
      };
      VP_CHECK(vx_enqueue_launch(q, &li, 0, NULL, NULL), "vx_enqueue_launch");

      VP_CHECK(vx_enqueue_read(q, color, cbuf, 0, cbuf_bytes, 0, NULL, NULL),
               "vx_enqueue_read(color)");
      VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");
      ok = true;
   }

done:
   if (abuf) vx_buffer_release(abuf);
   if (cbuf) vx_buffer_release(cbuf);
   if (pbuf) vx_buffer_release(pbuf);
   if (tbuf) vx_buffer_release(tbuf);
   if (kbuf) vx_buffer_release(kbuf);
   if (q)    vx_queue_release(q);
   unlink(vxpath);
   return ok;
}
