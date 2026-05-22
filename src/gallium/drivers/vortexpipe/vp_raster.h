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

/* Rasterize the Vortex-VS-transformed vertices on the hardware RASTER
 * unit and shade them with the fragment-shader kernel `fs_vxbin`; the
 * kernel submits fragments to the OM unit, which depth-tests, blends
 * and writes the colour buffer.
 *
 *   xverts        vertex_count records of layout->stride bytes; slot 0
 *                 is the clip-space gl_Position, slot 1 the colour
 *                 varying (the VS output layout).
 *   color         a width*height R8G8B8A8 host buffer -- on entry the
 *                 cleared framebuffer, on return the rendered image.
 *   om            depth/blend state for the OM unit.
 *
 * Returns true on success; false leaves `color` untouched so the
 * caller can fall back. */
bool vp_raster_draw(vx_device_h dev,
                    const void *fs_vxbin, size_t fs_vxbin_size,
                    const void *xverts, uint32_t vertex_count,
                    const struct vp_vs_layout *layout,
                    void *color, uint32_t width, uint32_t height,
                    const struct vp_om_params *om);

#ifdef __cplusplus
}
#endif

#endif /* VP_RASTER_H */
