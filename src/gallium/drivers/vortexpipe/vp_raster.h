/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_raster -- drive the Vortex hardware RASTER unit.
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
#include "gfx_fs_desc_abi.h"     /* GFX_FS_DESC_SLOTS */

#ifdef __cplusplus
extern "C" {
#endif

/* Fragment-shader constant buffers: for each lavapipe fragment
 * constant-buffer index i (0 = push constants, 1 = descriptor set-0 blob,
 * …) the host pointer + byte size of its contents, or {NULL,0} if unbound.
 * vp_raster_draw uploads each to device memory and builds the resident FS
 * descriptor table (i64[GFX_FS_DESC_SLOTS] of device base addresses) the
 * fragment kernel reads via load_ubo / load_push_constant / load_ssbo. A
 * NULL vp_fs_consts* is the descriptor-free draw (an all-zero table). */
struct vp_fs_consts {
   const void *data[GFX_FS_DESC_SLOTS];
   uint32_t    size[GFX_FS_DESC_SLOTS];
   /* set-0 descriptors living inside the descriptor blob at data[1] (their
    * byte offsets + kinds from vp_scan_descriptors). vp_raster relocates each
    * buffer descriptor's lp_jit_buffer.ptr host->device so the FS's UBO/SSBO
    * dereference resolves on device. NULL/0 = no descriptors (push-only). */
   const struct vp_desc *descs;
   uint32_t              num_descs;
};

/* The vertex-buffer geometry feeding the VS stage (defined in vp_launch.h). */
struct vp_vertex_input;

/* Persistent front-end working set: the binning pipeline's resident
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

/* Per-attachment output-merger state for a draw targeting >1 colour
 * attachment. num_color <= 1 (or a NULL pointer) selects the single-RT fast
 * path (the FF/SW OM writes one attachment via `om` + color_dev). num_color > 1
 * routes the FS to gfx_om_fragment_mrt_sw with a resident gfx_sw_omcolor_t[]:
 * one shared depth/stencil op (from `om`) then a per-attachment blend + colour
 * write. Requires the FS to be compiled for SW OM (sw_om). */
struct vp_mrt_params {
   uint32_t num_color;                       /* colour attachments (>1 => MRT) */
   uint64_t color_dev[GFX_OM_MAX_RT];        /* per-attachment device base */
   uint32_t pitch[GFX_OM_MAX_RT];            /* per-attachment row stride (bytes) */
   uint32_t blend_mode[GFX_OM_MAX_RT];       /* VX_DCR_OM_BLEND_MODE packed word */
   uint32_t blend_func[GFX_OM_MAX_RT];       /* VX_DCR_OM_BLEND_FUNC packed word */
   uint32_t colormask[GFX_OM_MAX_RT];        /* VX_DCR_OM_CBUF_WRITEMASK (RGBA) */
};

/* The texture bound for a draw -- mip 0 dims + the sampler's filter/wrap.
 * POT dims use the FF vx_tex4 unit; NPOT dims are sampled by the SW path
 * (multiply addressing). The texels themselves are uploaded +
 * kept device-resident by the caller (vp_context texture residency) and passed
 * to vp_raster_draw as a device address; a zero tex_dev means an untextured
 * draw (the params are then ignored). */
struct vp_tex_params {
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
 *   cull_mode     device SETUP_CULL_* face-cull mode (0 = none / two-sided),
 *                 translated from the bound rasterizer state.
 *   vp_sx/tx/sy/ty  viewport transform (screen = ndc*scale + bias) captured
 *                 from the app's VkViewport; a negative vp_sy is the Vulkan
 *                 y-flip. The caller only takes this path for a full-framebuffer
 *                 viewport (offset/partial viewports fall back), so cull and
 *                 orientation match the app's framebuffer winding.
 *
 * Returns true on success; false leaves `color` untouched so the
 * caller can fall back. */
/* The VS/FS module + kernel handles are cached across draws via the *_io
 * pointers (the caller's CSO-resident slots): NULL on first use → loaded from
 * the vxbin and stored back; reused thereafter (compile-once / upload-once, no
 * /tmp round-trip). The front-end module is cached on `pool`. */
/* color_dev / depth_dev are the device addresses of the render-pass-resident
 * colour + depth buffers (vp_context framebuffer residency): the OM renders
 * straight into them — no per-draw colour upload / readback, no per-draw depth
 * clear (the caller clears/initialises them once per pass). tex_dev is the
 * resident texture buffer (0 = untextured). */
bool vp_raster_draw(vx_device_h dev, struct vp_raster_pool *pool,
                    const void *vs_vxbin, size_t vs_vxbin_size,
                    vx_module_h *vs_module_io, vx_kernel_h *vs_kernel_io,
                    const void *fs_vxbin, size_t fs_vxbin_size,
                    vx_module_h *fs_module_io, vx_kernel_h *fs_kernel_io,
                    uint32_t vertex_count,
                    uint32_t instance_count, uint32_t first_instance,
                    const struct vp_vs_layout *layout,
                    const struct vp_vertex_input *vin,
                    uint64_t index_dev,
                    uint64_t color_dev, uint64_t depth_dev,
                    uint32_t width, uint32_t height,
                    const struct vp_om_params *om,
                    uint64_t tex_dev, const struct vp_tex_params *tex,
                    uint32_t cull_mode,
                    bool sw_tex, bool sw_om, bool sw_raster,
                    float vp_sx, float vp_tx, float vp_sy, float vp_ty,
                    const struct vp_fs_consts *fs_consts,
                    const struct vp_mrt_params *mrt);

#ifdef __cplusplus
}
#endif

#endif /* VP_RASTER_H */
