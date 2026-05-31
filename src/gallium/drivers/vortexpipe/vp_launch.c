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

/* VP_CHECK wraps a Vortex runtime call: any vx_result_t != VX_SUCCESS is a
 * HARD ERROR — the operation we asked the runtime to perform on the device
 * failed. Log as mesa_loge so the host runtime / test harness can detect
 * the failure (vp_launch returns false on goto done; the test harness can
 * grep "MESA: error" or check stderr). DO NOT downgrade to mesa_logw: the
 * silent-fallback bug that ran tests on llvmpipe started here. */
#define VP_CHECK(call, what)                                            \
   do {                                                                 \
      vx_result_t _r = (call);                                          \
      if (_r != VX_SUCCESS) {                                           \
         mesa_loge("vortexpipe: launch: %s failed (%s)", (what),        \
                   vx_result_string(_r));                               \
         goto done;                                                     \
      }                                                                 \
   } while (0)

/* A buffer descriptor's lp_jit_buffer occupies the first bytes of its
 * struct lp_descriptor slot: the data pointer at +0 (8 bytes), the
 * byte size at +8 (4 bytes). VP_DESC_STRIDE lives in vp_launch.h. */
#define VP_JIT_BUF_PTR   0
#define VP_JIT_BUF_SIZE  8

/* ---- acceleration-structure (BVH) device copy (Phase 7.6) --------- *
 *
 * lavapipe builds the BVH on the CPU; the Vortex traversal kernel
 * walks it, so it must be copied into Vortex device memory. Layout
 * constants mirror struct lvp_bvh_header / lvp_bvh_instance_node in
 * src/gallium/frontends/lavapipe/lvp_acceleration_structure.h:
 *
 *   lvp_bvh_header { vk_aabb bounds;          // 24 B
 *                    uint32 serialization_size, instance_count,
 *                           leaf_nodes_offset, padding; }
 *
 * A box node's two children are BVH-relative uint32 offsets, so they
 * survive the copy unchanged. Only an instance node's bvh_ptr is an
 * absolute pointer (TLAS -> BLAS) and is relocated to the device copy.
 * serialization_size folds in a fixed serialization header plus eight
 * bytes per instance, so the buffer size is recoverable from it. */
#define VP_BVH_SERIALIZATION_SIZE  24   /* uint32, after vk_aabb bounds */
#define VP_BVH_INSTANCE_COUNT      28   /* uint32 */
#define VP_BVH_LEAF_NODES_OFFSET   32   /* uint32 */
#define VP_BVH_INSTANCE_NODE_SIZE  120  /* sizeof(struct lvp_bvh_instance_node) */
#define VP_ACCEL_SERIALIZATION_HDR 56   /* sizeof(lvp_accel_struct_serialization_header) */
#define VP_BVH_MAX_BYTES           (16u << 20)
#define VP_MAX_BVH                 64

/* Device buffers + host staging blobs created while copying one
 * acceleration structure's BVHs; both must outlive vx_queue_finish. */
struct vp_as_ctx {
   vx_device_h dev;
   vx_queue_h  q;
   vx_buffer_h bufs[VP_MAX_BVH];
   unsigned    n_bufs;
   void       *stages[VP_MAX_BVH];
   unsigned    n_stages;
   bool        ok;
};

/* Copy one lavapipe BVH (TLAS or BLAS) into Vortex device memory and
 * return its device address. A TLAS's instance nodes carry absolute
 * bvh_ptr links to their BLASes; those are copied recursively and the
 * link rewritten to the device address. Box-node children are
 * BVH-relative and need no fixup. */
static uint64_t
vp_copy_as(struct vp_as_ctx *c, const void *bvh_host)
{
   const uint8_t *h = bvh_host;
   uint32_t ser = 0, inst = 0, leaf_off = 0;
   memcpy(&ser,      h + VP_BVH_SERIALIZATION_SIZE, sizeof ser);
   memcpy(&inst,     h + VP_BVH_INSTANCE_COUNT,     sizeof inst);
   memcpy(&leaf_off, h + VP_BVH_LEAF_NODES_OFFSET,  sizeof leaf_off);

   if (ser <= VP_ACCEL_SERIALIZATION_HDR + 8u * inst) {
      mesa_loge("vortexpipe: launch: implausible BVH header");
      c->ok = false;
      return 0;
   }
   uint32_t size = ser - VP_ACCEL_SERIALIZATION_HDR - 8u * inst;
   if (size == 0 || size > VP_BVH_MAX_BYTES ||
       c->n_stages >= VP_MAX_BVH || c->n_bufs >= VP_MAX_BVH) {
      mesa_loge("vortexpipe: launch: BVH too large / too many BVHs");
      c->ok = false;
      return 0;
   }

   /* private copy so instance-node bvh_ptr fields can be relocated */
   uint8_t *stage = malloc(size);
   if (!stage) {
      c->ok = false;
      return 0;
   }
   memcpy(stage, bvh_host, size);
   c->stages[c->n_stages++] = stage;

   for (uint32_t i = 0; i < inst; i++) {
      uint8_t  *node = stage + leaf_off + (size_t)i * VP_BVH_INSTANCE_NODE_SIZE;
      uint64_t  blas_host = 0;
      memcpy(&blas_host, node, sizeof blas_host);   /* lvp_bvh_instance_node.bvh_ptr */
      if (blas_host) {
         uint64_t blas_dev = vp_copy_as(c, (const void *)(uintptr_t)blas_host);
         if (!c->ok)
            return 0;
         memcpy(node, &blas_dev, sizeof blas_dev);
      }
   }

   vx_buffer_h b = NULL;
   uint64_t dev_addr = 0;
   if (vx_buffer_create(c->dev, size, 0, &b) != VX_SUCCESS) {
      c->ok = false;
      return 0;
   }
   c->bufs[c->n_bufs++] = b;
   if (vx_buffer_address(b, &dev_addr) != VX_SUCCESS ||
       vx_enqueue_write(c->q, b, 0, stage, size, 0, NULL, NULL) != VX_SUCCESS) {
      c->ok = false;
      return 0;
   }
   return dev_addr;
}

bool
vp_launch(vx_device_h dev,
          const void *vxbin, size_t vxbin_size,
          const void *desc_host, uint32_t desc_bytes,
          const struct vp_desc *descs, uint32_t num_descs,
          const uint32_t grid[3], const uint32_t block[3],
          uint32_t lmem_size)
{
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_module_h kmod = NULL;
   vx_kernel_h kbuf = NULL;
   vx_buffer_h dbuf = NULL;
   vx_buffer_h res[VP_MAX_DESCS]      = { 0 };  /* per-descriptor device buffer */
   void       *res_host[VP_MAX_DESCS] = { 0 };  /* its host backing            */
   uint32_t    res_bytes[VP_MAX_DESCS]= { 0 };
   struct vp_as_ctx asc = { .ok = true };       /* acceleration-structure BVHs */
   uint8_t    *stage = NULL;
   char vxpath[] = "/tmp/vortexpipe-k.XXXXXX";
   int  vxfd = -1;

   /* materialize the .vxbin in a temp file for vx_module_load_file */
   vxfd = mkstemp(vxpath);
   if (vxfd < 0) {
      mesa_loge("vortexpipe: launch: mkstemp failed");
      return false;
   }
   if (write(vxfd, vxbin, vxbin_size) != (ssize_t)vxbin_size) {
      mesa_loge("vortexpipe: launch: writing .vxbin failed");
      close(vxfd);
      unlink(vxpath);
      return false;
   }
   close(vxfd);

   /* A private copy of the descriptor buffer: vp_launch rewrites the
    * resource pointers inside it to device addresses. It must outlive
    * vx_queue_finish (vx_enqueue_write reads it asynchronously). */
   stage = malloc(desc_bytes);
   if (!stage) {
      mesa_loge("vortexpipe: launch: descriptor staging OOM");
      unlink(vxpath);
      return false;
   }
   memcpy(stage, desc_host, desc_bytes);

   vx_queue_info_t qi = {
      .struct_size = sizeof(qi), .next = NULL,
      .priority = VX_QUEUE_PRIORITY_NORMAL, .flags = 0,
   };
   VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");
   asc.dev = dev;
   asc.q   = q;

   VP_CHECK(vx_module_load_file(dev, vxpath, &kmod),
            "vx_module_load_file");
   VP_CHECK(vx_module_get_kernel(kmod, "kernel_main", &kbuf),
            "vx_module_get_kernel");

   /* Relocate each descriptor into the staged descriptor blob:
    *  - VP_DESC_BUFFER: copy the resource into device memory, rewrite
    *    lp_jit_buffer.ptr so load_ssbo/store_ssbo dereference on-device.
    *  - VP_DESC_AS: copy the BVH (TLAS + its BLASes) into device
    *    memory with instance-node links relocated, rewrite the
    *    accel_struct device address. */
   for (uint32_t i = 0; i < num_descs; i++) {
      uint8_t *slot = stage + descs[i].offset;

      if (descs[i].kind == VP_DESC_AS) {
         uint64_t tlas_host = 0;
         memcpy(&tlas_host, slot, sizeof tlas_host);   /* lp_descriptor.accel_struct */
         if (!tlas_host)
            continue;
         uint64_t tlas_dev = vp_copy_as(&asc, (const void *)(uintptr_t)tlas_host);
         if (!asc.ok) {
            mesa_loge("vortexpipe: launch: acceleration-structure copy failed");
            goto done;
         }
         memcpy(slot, &tlas_dev, sizeof tlas_dev);
         continue;
      }

      /* VP_DESC_BUFFER */
      uint64_t host_ptr = 0;
      uint32_t size = 0;
      memcpy(&host_ptr, slot + VP_JIT_BUF_PTR,  sizeof host_ptr);
      memcpy(&size,     slot + VP_JIT_BUF_SIZE, sizeof size);
      if (!host_ptr || !size)
         continue;
      VP_CHECK(vx_buffer_create(dev, size, 0, &res[i]),
               "vx_buffer_create(resource)");
      uint64_t dev_addr = 0;
      VP_CHECK(vx_buffer_address(res[i], &dev_addr), "vx_buffer_address");
      VP_CHECK(vx_enqueue_write(q, res[i], 0, (void *)(uintptr_t)host_ptr,
                                size, 0, NULL, NULL),
               "vx_enqueue_write(resource)");
      memcpy(slot + VP_JIT_BUF_PTR, &dev_addr, sizeof dev_addr);
      res_host[i]  = (void *)(uintptr_t)host_ptr;
      res_bytes[i] = size;
   }

   /* upload the relocated descriptor buffer */
   VP_CHECK(vx_buffer_create(dev, desc_bytes, 0, &dbuf),
            "vx_buffer_create(descriptors)");
   uint64_t desc_dev = 0;
   VP_CHECK(vx_buffer_address(dbuf, &desc_dev), "vx_buffer_address(descriptors)");
   VP_CHECK(vx_enqueue_write(q, dbuf, 0, stage, desc_bytes, 0, NULL, NULL),
            "vx_enqueue_write(descriptors)");

   /* arg block: i64[VP_ARG_SLOTS]; slot 1 -> set-0 descriptor buffer.
    * In the current vortex2 API the runtime stages the arg blob into a
    * scratch slot at launch time — we pass it inline via args_host
    * instead of allocating an args buffer. */
   uint64_t argblk[VP_ARG_SLOTS] = { 0 };
   argblk[1] = desc_dev;

   /* dispatch */
   uint32_t ndim = (grid[2] > 1 || block[2] > 1) ? 3
                 : (grid[1] > 1 || block[1] > 1) ? 2 : 1;
   vx_launch_info_t li = {
      .struct_size = sizeof(li), .next = NULL,
      .kernel = kbuf,
      .args_host = argblk, .args_size = sizeof(argblk),
      .ndim = ndim,
      .grid_dim  = { grid[0],  grid[1],  grid[2]  },
      .block_dim = { block[0], block[1], block[2] },
      .lmem_size = lmem_size,
   };
   VP_CHECK(vx_enqueue_launch(q, &li, 0, NULL, NULL), "vx_enqueue_launch");

   /* copy every buffer back into its host backing */
   for (uint32_t i = 0; i < num_descs; i++) {
      if (!res[i])
         continue;
      VP_CHECK(vx_enqueue_read(q, res_host[i], res[i], 0, res_bytes[i],
                               0, NULL, NULL), "vx_enqueue_read(resource)");
   }
   VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");

   ok = true;

done:
   for (uint32_t i = 0; i < VP_MAX_DESCS; i++)
      if (res[i]) vx_buffer_release(res[i]);
   for (unsigned i = 0; i < asc.n_bufs; i++)
      if (asc.bufs[i]) vx_buffer_release(asc.bufs[i]);
   for (unsigned i = 0; i < asc.n_stages; i++)
      free(asc.stages[i]);
   if (dbuf) vx_buffer_release(dbuf);
   if (kbuf) vx_kernel_release(kbuf);
   if (kmod) vx_module_release(kmod);
   if (q)    vx_queue_release(q);
   free(stage);
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
   vx_module_h kmod = NULL;
   vx_kernel_h kbuf = NULL;
   vx_buffer_h obuf = NULL, vbuf = NULL, tbuf = NULL;
   char vxpath[] = "/tmp/vortexpipe-vs.XXXXXX";
   int  vxfd = -1;

   vxfd = mkstemp(vxpath);
   if (vxfd < 0) {
      mesa_loge("vortexpipe: vs launch: mkstemp failed");
      return false;
   }
   if (write(vxfd, vxbin, vxbin_size) != (ssize_t)vxbin_size) {
      mesa_loge("vortexpipe: vs launch: writing .vxbin failed");
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
   VP_CHECK(vx_module_load_file(dev, vxpath, &kmod),
            "vx_module_load_file");
   VP_CHECK(vx_module_get_kernel(kmod, "kernel_main", &kbuf),
            "vx_module_get_kernel");

   /* Query device geometry so the VS launch maximises warp utilization:
    *   block_dim = round_up(vertex_count, num_threads), capped at the
    *               max CTA size (num_threads × num_warps). This keeps
    *               every active warp's tmask full (no partial trailing
    *               warp) and lets a CTA saturate one whole core.
    *   grid_dim  = ceil(vertex_count / block_dim) so the work spreads
    *               across cores (KMU hands one CTA to each free core).
    *
    * Pre-fix shape was grid=(1,1,1) block=(vertex_count,1,1) which:
    *   - silently truncated vertex_count > max_block_size at the DCR
    *     write (KMU's CTA_TID_WIDTH+1 cap),
    *   - left the trailing warp partially-masked when vertex_count
    *     wasn't a multiple of num_threads (degraded warp util),
    *   - sat all work on one core (the other num_cores-1 idle).
    *
    * Padding policy: the device output buffer is sized to grid × block
    * × stride. Out-of-bounds threads (vid in [vertex_count,
    * grid*block)) write to the pad region; the host read-back copies
    * only `out_bytes` so the caller's buffer never sees the slack.
    * This avoids needing a bounds-check intrinsic in the VS NIR
    * lowering. */
   uint64_t nt = 0, nw = 0;
   VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt),
            "vx_device_query(NUM_THREADS)");
   VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &nw),
            "vx_device_query(NUM_WARPS)");
   const uint32_t num_threads  = (uint32_t)nt;
   const uint32_t num_warps    = (uint32_t)nw;
   const uint32_t cta_size_max = num_threads * num_warps;

   uint32_t block_x = (vertex_count + num_threads - 1u) / num_threads
                     * num_threads;          /* round up to nt multiple */
   if (block_x > cta_size_max) block_x = cta_size_max;
   if (block_x == 0)           block_x = num_threads;
   uint32_t grid_x  = (vertex_count + block_x - 1u) / block_x;
   uint32_t launched_threads = grid_x * block_x;
   uint32_t stride           = (vertex_count > 0) ? out_bytes / vertex_count : 0;
   uint32_t obuf_bytes       = launched_threads * stride;
   if (obuf_bytes < out_bytes) obuf_bytes = out_bytes;

   /* output vertex-record buffer + its device address (padded for the
    * trailing CTA's out-of-bounds threads). */
   VP_CHECK(vx_buffer_create(dev, obuf_bytes, 0, &obuf),
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

   /* One thread per vertex, sized to fill warps and saturate cores
    * (see geometry-query comment above). */
   vx_launch_info_t li = {
      .struct_size = sizeof(li), .next = NULL,
      .kernel = kbuf,
      .args_host = argblk, .args_size = sizeof(argblk),
      .ndim = 1,
      .grid_dim  = { grid_x,  1, 1 },
      .block_dim = { block_x, 1, 1 },
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
   if (kbuf) vx_kernel_release(kbuf);
   if (kmod) vx_module_release(kmod);
   if (q)    vx_queue_release(q);
   unlink(vxpath);
   return ok;
}
