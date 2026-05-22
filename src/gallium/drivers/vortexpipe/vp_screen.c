/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vortexpipe -- Vortex GPU Gallium driver.
 *
 * Phase 1 skeleton. vortexpipe is a thin Gallium driver layered on
 * llvmpipe: vortexpipe_create_screen() currently delegates straight
 * to llvmpipe_create_screen(). Subsequent phases progressively
 * override screen/context callbacks to dispatch rasterization,
 * texturing, the output merger, and compute/shader work to the
 * Vortex device -- see docs/proposals/vulkan_support_proposal.md in
 * the Vortex source tree.
 */

#include "vp_public.h"
#include "llvmpipe/lp_public.h"
#include "util/log.h"

struct pipe_screen *
vortexpipe_create_screen(struct sw_winsys *winsys)
{
   mesa_logi("vortexpipe: create_screen (Phase 1 skeleton -- llvmpipe passthrough)");

   /* Phase 1: passthrough to llvmpipe. Vortex specialization follows
    * in later phases (raster/tex/om HW models + SIMT shader codegen).
    */
   return llvmpipe_create_screen(winsys);
}
