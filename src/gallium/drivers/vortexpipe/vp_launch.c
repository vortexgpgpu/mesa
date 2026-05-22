/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_launch -- Phase 2 #5: dispatch a compiled kernel on the Vortex
 * device.
 *
 * Builds the kernel argument block the translated kernel expects
 * (vp_nir_to_llvm: kernel_main(ptr arg) reads buffer addresses as
 * arg[i] -- an array of i64 device addresses), copies the SSBO
 * host<->device, and drives the vortex2 runtime: queue -> upload ->
 * vx_enqueue_launch -> read back. add1/vecadd-class single-SSBO
 * kernels (binding 0); the multi-binding descriptor-stride case is
 * a later generalization.
 */

#define _GNU_SOURCE
#include "vp_launch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util/log.h"

/* i64 arg-block slots; slot N == base address of buffer set (N-1).
 * add1 uses slot 1 (descriptor set 0). */
#define VP_ARG_SLOTS 8

#define VP_CHECK(call, what)                                            \
   do {                                                                 \
      vx_result_t _r = (call);                                          \
      if (_r != VX_SUCCESS) {                                           \
         mesa_logw("vortexpipe: launch: %s failed (%s)", (what),        \
                   vx_result_string(_r));                               \
         goto done;                                                     \
      }                                                                 \
   } while (0)

bool
vp_launch(vx_device_h dev,
          const void *vxbin, size_t vxbin_size,
          void *ssbo_host, uint32_t ssbo_bytes,
          const uint32_t grid[3], const uint32_t block[3])
{
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_buffer_h kbuf = NULL, abuf = NULL, sbuf = NULL;
   char vxpath[] = "/tmp/vortexpipe-k.XXXXXX";
   int  vxfd = -1;

   /* materialize the .vxbin in a temp file for vx_buffer_load_kernel_file */
   vxfd = mkstemp(vxpath);
   if (vxfd < 0) {
      mesa_logw("vortexpipe: launch: mkstemp failed");
      return false;
   }
   if (write(vxfd, vxbin, vxbin_size) != (ssize_t)vxbin_size) {
      mesa_logw("vortexpipe: launch: writing .vxbin failed");
      close(vxfd);
      unlink(vxpath);
      return false;
   }
   close(vxfd);

   vx_queue_info_t qi = {
      .struct_size = sizeof(qi), .next = NULL,
      .priority = VX_QUEUE_PRIORITY_NORMAL, .flags = 0,
   };
   VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");

   VP_CHECK(vx_buffer_load_kernel_file(dev, q, vxpath, &kbuf),
            "vx_buffer_load_kernel_file");

   /* SSBO device buffer + its device address */
   VP_CHECK(vx_buffer_create(dev, ssbo_bytes, 0, &sbuf), "vx_buffer_create(ssbo)");
   uint64_t ssbo_dev = 0;
   VP_CHECK(vx_buffer_address(sbuf, &ssbo_dev), "vx_buffer_address");

   /* arg block: i64[VP_ARG_SLOTS]; slot 1 -> SSBO device address */
   uint64_t argblk[VP_ARG_SLOTS] = { 0 };
   argblk[1] = ssbo_dev;
   VP_CHECK(vx_buffer_create(dev, sizeof(argblk), 0, &abuf),
            "vx_buffer_create(args)");

   /* upload SSBO data + arg block */
   VP_CHECK(vx_enqueue_write(q, sbuf, 0, ssbo_host, ssbo_bytes, 0, NULL, NULL),
            "vx_enqueue_write(ssbo)");
   VP_CHECK(vx_enqueue_write(q, abuf, 0, argblk, sizeof(argblk), 0, NULL, NULL),
            "vx_enqueue_write(args)");

   /* dispatch */
   uint32_t ndim = (grid[2] > 1 || block[2] > 1) ? 3
                 : (grid[1] > 1 || block[1] > 1) ? 2 : 1;
   vx_launch_info_t li = {
      .struct_size = sizeof(li), .next = NULL,
      .kernel = kbuf, .args = abuf, .ndim = ndim,
      .grid_dim  = { grid[0],  grid[1],  grid[2]  },
      .block_dim = { block[0], block[1], block[2] },
      .lmem_size = 0,
   };
   VP_CHECK(vx_enqueue_launch(q, &li, 0, NULL, NULL), "vx_enqueue_launch");

   /* read the result back into the host SSBO */
   VP_CHECK(vx_enqueue_read(q, ssbo_host, sbuf, 0, ssbo_bytes, 0, NULL, NULL),
            "vx_enqueue_read");
   VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");

   ok = true;

done:
   if (sbuf) vx_buffer_release(sbuf);
   if (abuf) vx_buffer_release(abuf);
   if (kbuf) vx_buffer_release(kbuf);
   if (q)    vx_queue_release(q);
   unlink(vxpath);
   return ok;
}

/* attribute-table entry the VS kernel reads: { device base, stride }.
 * The table is indexed by VS input driver_location. */
#define VP_ATTR_ENTRY_BYTES 8
#define VP_ATTR_TABLE_LOCS  8

bool
vp_launch_vs(vx_device_h dev,
             const void *vxbin, size_t vxbin_size,
             void *out_host, uint32_t out_bytes,
             uint32_t vertex_count,
             const struct vp_vertex_input *vin)
{
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_buffer_h kbuf = NULL, abuf = NULL, obuf = NULL,
               vbuf = NULL, tbuf = NULL;
   char vxpath[] = "/tmp/vortexpipe-vs.XXXXXX";
   int  vxfd = -1;

   vxfd = mkstemp(vxpath);
   if (vxfd < 0) {
      mesa_logw("vortexpipe: vs launch: mkstemp failed");
      return false;
   }
   if (write(vxfd, vxbin, vxbin_size) != (ssize_t)vxbin_size) {
      mesa_logw("vortexpipe: vs launch: writing .vxbin failed");
      close(vxfd);
      unlink(vxpath);
      return false;
   }
   close(vxfd);

   vx_queue_info_t qi = {
      .struct_size = sizeof(qi), .next = NULL,
      .priority = VX_QUEUE_PRIORITY_NORMAL, .flags = 0,
   };
   VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");
   VP_CHECK(vx_buffer_load_kernel_file(dev, q, vxpath, &kbuf),
            "vx_buffer_load_kernel_file");

   /* output vertex-record buffer + its device address */
   VP_CHECK(vx_buffer_create(dev, out_bytes, 0, &obuf),
            "vx_buffer_create(out)");
   uint64_t out_dev = 0;
   VP_CHECK(vx_buffer_address(obuf, &out_dev), "vx_buffer_address");

   /* arg block: slot 0 -> output buffer device address,
    *            slot 1 -> vertex attribute table (0 if self-contained) */
   uint64_t argblk[VP_ARG_SLOTS] = { 0 };
   argblk[0] = out_dev;

   /* Vertex buffer + attribute table: upload the interleaved vertex
    * buffer, then a table indexed by driver_location holding the
    * device base address + stride of each attribute. The VS kernel
    * fetches input `loc` of vertex `vid` at table[loc].base +
    * vid*table[loc].stride.
    *
    * `table` is declared at function scope: vx_enqueue_write is
    * asynchronous (the source is read at vx_queue_finish), so it must
    * outlive the `if` block -- the same lifetime rule as argblk. */
   uint32_t table[VP_ATTR_TABLE_LOCS * 2] = { 0 };
   if (vin && vin->num_attrs) {
      VP_CHECK(vx_buffer_create(dev, vin->size, 0, &vbuf),
               "vx_buffer_create(vbuf)");
      uint64_t vbuf_dev = 0;
      VP_CHECK(vx_buffer_address(vbuf, &vbuf_dev), "vx_buffer_address(vbuf)");
      VP_CHECK(vx_enqueue_write(q, vbuf, 0, vin->data, vin->size,
                                0, NULL, NULL), "vx_enqueue_write(vbuf)");

      for (uint32_t i = 0; i < vin->num_attrs; i++) {
         uint32_t loc = vin->attr_loc[i];
         if (loc >= VP_ATTR_TABLE_LOCS)
            continue;
         table[loc * 2 + 0] = (uint32_t)vbuf_dev + vin->base_offset
                            + vin->attr_offset[i];
         table[loc * 2 + 1] = vin->attr_stride[i];
      }
      VP_CHECK(vx_buffer_create(dev, sizeof(table), 0, &tbuf),
               "vx_buffer_create(attrtab)");
      uint64_t tbuf_dev = 0;
      VP_CHECK(vx_buffer_address(tbuf, &tbuf_dev), "vx_buffer_address(attrtab)");
      VP_CHECK(vx_enqueue_write(q, tbuf, 0, table, sizeof(table),
                                0, NULL, NULL), "vx_enqueue_write(attrtab)");
      argblk[1] = tbuf_dev;
   }

   VP_CHECK(vx_buffer_create(dev, sizeof(argblk), 0, &abuf),
            "vx_buffer_create(args)");
   VP_CHECK(vx_enqueue_write(q, abuf, 0, argblk, sizeof(argblk), 0, NULL, NULL),
            "vx_enqueue_write(args)");

   /* one thread per vertex: a single block of `vertex_count` */
   vx_launch_info_t li = {
      .struct_size = sizeof(li), .next = NULL,
      .kernel = kbuf, .args = abuf, .ndim = 1,
      .grid_dim  = { 1, 1, 1 },
      .block_dim = { vertex_count, 1, 1 },
      .lmem_size = 0,
   };
   VP_CHECK(vx_enqueue_launch(q, &li, 0, NULL, NULL), "vx_enqueue_launch");

   VP_CHECK(vx_enqueue_read(q, out_host, obuf, 0, out_bytes, 0, NULL, NULL),
            "vx_enqueue_read");
   VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");

   ok = true;

done:
   if (tbuf) vx_buffer_release(tbuf);
   if (vbuf) vx_buffer_release(vbuf);
   if (obuf) vx_buffer_release(obuf);
   if (abuf) vx_buffer_release(abuf);
   if (kbuf) vx_buffer_release(kbuf);
   if (q)    vx_queue_release(q);
   unlink(vxpath);
   return ok;
}
