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

/* A raw compute shader buffer bound at set_shader_buffers slot `slot`
 * (not a descriptor-set SSBO). vp_launch uploads `host[0,size)` to device
 * memory and writes its device address into arg[VP_ARG_SSBO_BASE + slot],
 * where the kernel's load_ssbo(imm slot, off) reads it. Upload-only: these
 * are input buffers (e.g. the RT trace-ray command buffer). */
struct vp_ssbo {
   const void *host;   /* mapped host bytes */
   uint32_t    size;   /* buffer_size */
   unsigned    slot;   /* set_shader_buffers binding index */
   bool        trace_cmd; /* this slot is an RT VkTraceRaysIndirectCommand2KHR:
                           * its SBT shader-record device-address fields are
                           * host pointers and must be relocated on upload. */
};

/* Run a compiled Vortex compute kernel (.vxbin) on `dev`.
 *
 * `desc_host[0, desc_bytes)` is set 0's descriptor buffer -- an array
 * of struct lp_descriptor the kernel reaches through arg slot 1.
 * `descs[0, num_descs)` (from vp_scan_descriptors) names the buffer
 * descriptors inside it: each is copied to Vortex device memory and
 * its lp_jit_buffer.ptr rewritten to the device address, so the
 * kernel's load_ssbo/store_ssbo dereference resolves on-device.
 * `ssbos[0, num_ssbos)` are raw set_shader_buffers slots relocated into
 * the arg block. The kernel runs over grid x block; writable descriptor
 * buffers are copied back. `lmem_size` is the per-workgroup shared memory.
 *
 * `module_io`/`kernel_io` are caller-owned residency slots, as on the draw
 * path: the kernel image is loaded onto the device by the first dispatch that
 * finds them NULL and reused by every dispatch after, so repeating a dispatch
 * costs no module reload. The caller owns the release, and must evict before
 * another shader takes the same device address -- compute and fragment shaders
 * both start at VP_STARTUP_FS, so only one of them can be resident.
 *
 * Returns true on success.
 */
bool vp_launch(vx_device_h dev,
               const void *vxbin, size_t vxbin_size,
               vx_module_h *module_io, vx_kernel_h *kernel_io,
               const void *desc_host, uint32_t desc_bytes,
               const struct vp_desc *descs, uint32_t num_descs,
               const struct vp_ssbo *ssbos, uint32_t num_ssbos,
               const uint32_t grid[3], const uint32_t block[3],
               const uint32_t grid_base[3],
               uint32_t lmem_size, bool has_rtu);

/* The vertex-buffer geometry feeding a VS kernel: the distinct vertex-buffer
 * resources the draw binds, plus the per-attribute layout. Each attribute may
 * come from its own buffer (an app that binds one buffer per attribute), so the
 * VS kernel fetches attribute `loc` of vertex `vid` at
 *   buf_data[attr_buf[loc]] + attr_offset[loc] + vid*attr_stride[loc].
 * NULL passed to vp_launch_vs means a self-contained VS (the corner arrays are
 * baked in -- gl_VertexIndex only, no vertex buffer). */
struct vp_vertex_input {
   uint32_t    num_bufs;         /* distinct bound vertex-buffer resources */
   const void *buf_data[8];      /* host bytes of each distinct buffer */
   uint32_t    buf_size[8];      /* length of each buffer in bytes */
   uint32_t    num_attrs;
   uint32_t    attr_loc[8];      /* VS input driver_location */
   uint32_t    attr_buf[8];      /* which buf_data[] this attribute fetches from */
   uint32_t    attr_offset[8];   /* byte offset of the attribute in its buffer */
   uint32_t    attr_stride[8];   /* bytes between consecutive vertices */
};

/* Run a compiled Vortex vertex-shader kernel (.vxbin) on `dev`. One
 * thread per vertex; the kernel writes one transformed record per vertex
 * into a device buffer (arg slot 0). When `vin` is non-NULL its vertex
 * buffer is uploaded and an attribute table (arg slot 1) lets the kernel
 * fetch per-vertex inputs.
 *
 * The transformed vertices are left RESIDENT: on success *out_buf / *out_addr
 * receive the device buffer + its base address, which the caller owns and
 * must vx_buffer_release. The on-device front end (vp_raster_draw's expand_k)
 * consumes them directly; only the llvmpipe-raster fallback reads them back to
 * host (vp_buffer_readback). `out_bytes` is the logical count*stride size. */
bool vp_launch_vs(vx_device_h dev,
                  const void *vxbin, size_t vxbin_size,
                  uint32_t vertex_count, uint32_t out_bytes,
                  const struct vp_vertex_input *vin,
                  vx_buffer_h *out_buf, uint64_t *out_addr);

/* Copy a resident device buffer back to host memory (one-shot, own queue).
 * Used on the llvmpipe-raster fallback to hand the kept-resident VS output to
 * llvmpipe as a host vertex buffer. */
bool vp_buffer_readback(vx_device_h dev, vx_buffer_h buf,
                        void *host, uint32_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* VP_LAUNCH_H */
