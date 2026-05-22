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
