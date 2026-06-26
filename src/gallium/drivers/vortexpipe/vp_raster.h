/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_raster -- Phase 4: drive the Vortex hardware RASTER unit.
 *
 * Triangle setup + binning (graphics::Binning) turn the VS-transformed
 * vertices into the tile + primitive buffers the RASTER unit walks;
 * the RASTER DCRs are programmed and the fragment-shader kernel is
 * dispatched. The C++ implementation links the Vortex `graphics`
 * library; the interface here is plain C for the rest of vortexpipe.
 */

#ifndef VP_RASTER_H
#define VP_RASTER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "vortex2.h"
#include "vp_nir_to_llvm.h"      /* struct vp_vs_layout */

#ifdef __cplusplus
extern "C" {
#endif

/* The vertex-buffer geometry feeding the VS stage (defined in vp_launch.h). */
struct vp_vertex_input;

/* Persistent front-end working set (§6.6): the binning pipeline's resident
 * buffer set, laid out once over VX_MEM_PHYS and reused across the frame's
 * draws (grown on demand) instead of allocated per draw. Owned by the
 * context; opaque to the rest of vortexpipe. */
struct vp_raster_pool;
struct vp_raster_pool *vp_raster_pool_create(void);
void                   vp_raster_pool_destroy(struct vp_raster_pool *pool);

/* Output-merger state for a draw -- the Gallium depth-stencil + blend
 * state, translated to the Vortex OM encoding (see vp_context.c). */
struct vp_om_params {
   bool     depth_test;
   uint32_t depth_func;     /* VX_OM_DEPTH_FUNC_* */
   bool     depth_write;
   uint32_t blend_mode;     /* VX_DCR_OM_BLEND_MODE packed word */
   uint32_t blend_func;     /* VX_DCR_OM_BLEND_FUNC packed word */
   uint32_t colormask;      /* VX_DCR_OM_CBUF_WRITEMASK */
};

/* The texture bound for a draw -- mip 0 of the sampler-view image,
 * plus the sampler's filter/wrap. width/height must be powers of two.
 * NULL passed to vp_raster_draw means an untextured draw. */
struct vp_tex_params {
   const void *pixels;      /* width*height R8G8B8A8 host pixels */
   uint32_t    width;
   uint32_t    height;
   uint32_t    filter;      /* VX_TEX_FILTER_* */
   uint32_t    wrap_u;      /* VX_TEX_WRAP_* */
   uint32_t    wrap_v;
};

/* Run the WHOLE draw as one device-orchestrated command: the vertex shader
 * `vs_vxbin` is stage 0 of the draw program (linked at VP_STARTUP_VS so it
 * co-resides with the FS + front end), so its transformed output is consumed
 * by the on-device front end (expand_k) with no host round-trip — the VS no
 * longer runs as a separate host-blocking launch. The hardware RASTER unit
 * walks the binned primitives and `fs_vxbin` shades them; the OM unit
 * depth-tests, blends and writes the colour buffer.
 *
 *   vs_vxbin      the compiled vertex shader; launched as stage 0, one thread
 *                 per vertex, writing layout->stride-byte records expand_k
 *                 consumes. VS->setup->bin->FF->FS is one OP_DRAW.
 *   vertex_count  vertices in the draw (3 per triangle).
 *   layout        VS output record layout (stride + varyings) for expand_k.
 *   vin           vertex-buffer geometry the VS fetches inputs from, or NULL
 *                 for a self-contained VS (gl_VertexIndex only).
 *   color         a width*height R8G8B8A8 host buffer -- on entry the
 *                 cleared framebuffer, on return the rendered image.
 *   om            depth/blend state for the OM unit.
 *   tex           the texture bound to TEX stage 0, or NULL for an
 *                 untextured draw.
 *
 * Returns true on success; false leaves `color` untouched so the
 * caller can fall back. */
/* The VS/FS module + kernel handles are cached across draws via the *_io
 * pointers (the caller's CSO-resident slots): NULL on first use → loaded from
 * the vxbin and stored back; reused thereafter (compile-once / upload-once, no
 * /tmp round-trip). The front-end module is cached on `pool`. */
bool vp_raster_draw(vx_device_h dev, struct vp_raster_pool *pool,
                    const void *vs_vxbin, size_t vs_vxbin_size,
                    vx_module_h *vs_module_io, vx_kernel_h *vs_kernel_io,
                    const void *fs_vxbin, size_t fs_vxbin_size,
                    vx_module_h *fs_module_io, vx_kernel_h *fs_kernel_io,
                    uint32_t vertex_count,
                    const struct vp_vs_layout *layout,
                    const struct vp_vertex_input *vin,
                    void *color, uint32_t width, uint32_t height,
                    const struct vp_om_params *om,
                    const struct vp_tex_params *tex);

#ifdef __cplusplus
}
#endif

#endif /* VP_RASTER_H */
