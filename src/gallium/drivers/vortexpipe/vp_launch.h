/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_launch -- Phase 2 #5: dispatch a compiled kernel on the Vortex
 * device via the vortex2 runtime.
 */

#ifndef VP_LAUNCH_H
#define VP_LAUNCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "vortex2.h"
#include "vp_nir_to_llvm.h"      /* struct vp_desc */

#ifdef __cplusplus
extern "C" {
#endif

/* struct lp_descriptor stride in lavapipe's set descriptor buffer:
 * binding N's descriptor starts at N * VP_DESC_STRIDE. */
#define VP_DESC_STRIDE 256

/* Run a compiled Vortex compute kernel (.vxbin) on `dev`.
 *
 * `desc_host[0, desc_bytes)` is set 0's descriptor buffer -- an array
 * of struct lp_descriptor the kernel reaches through arg slot 1.
 * `descs[0, num_descs)` (from vp_scan_descriptors) names the buffer
 * descriptors inside it: each is copied to Vortex device memory and
 * its lp_jit_buffer.ptr rewritten to the device address, so the
 * kernel's load_ssbo/store_ssbo dereference resolves on-device. The
 * kernel runs over grid x block; writable buffers are copied back.
 * `lmem_size` is the per-workgroup shared-memory allocation.
 *
 * Returns true on success.
 */
bool vp_launch(vx_device_h dev,
               const void *vxbin, size_t vxbin_size,
               const void *desc_host, uint32_t desc_bytes,
               const struct vp_desc *descs, uint32_t num_descs,
               const uint32_t grid[3], const uint32_t block[3],
               uint32_t lmem_size);

/* The vertex-buffer geometry feeding a VS kernel: a single interleaved
 * host vertex buffer plus the per-attribute layout. The VS kernel
 * fetches attribute `loc` of vertex `vid` at
 *   data + base_offset + attr_offset[loc] + vid*attr_stride[loc].
 * NULL passed to vp_launch_vs means a self-contained VS (the corner
 * arrays are baked in -- gl_VertexIndex only, no vertex buffer). */
struct vp_vertex_input {
   const void *data;          /* host vertex-buffer bytes */
   uint32_t    size;          /* data length in bytes */
   uint32_t    base_offset;   /* pipe_vertex_buffer.buffer_offset */
   uint32_t    num_attrs;
   uint32_t    attr_loc[8];      /* VS input driver_location */
   uint32_t    attr_offset[8];   /* byte offset of the attribute in a vertex */
   uint32_t    attr_stride[8];   /* bytes between consecutive vertices */
};

/* Run a compiled Vortex vertex-shader kernel (.vxbin) on `dev`
 * (Phase 3). One thread per vertex; the kernel writes one transformed
 * record per vertex into a device buffer (arg slot 0). When `vin` is
 * non-NULL its vertex buffer is uploaded and an attribute table
 * (arg slot 1) lets the kernel fetch per-vertex inputs. On success
 * the `out_bytes` of output are copied into out_host. */
bool vp_launch_vs(vx_device_h dev,
                  const void *vxbin, size_t vxbin_size,
                  void *out_host, uint32_t out_bytes,
                  uint32_t vertex_count,
                  const struct vp_vertex_input *vin);

#ifdef __cplusplus
}
#endif

#endif /* VP_LAUNCH_H */
