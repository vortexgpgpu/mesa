/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_gfx_frontend -- accessor for the embedded gfx_v2 on-device front-end
 * kernel image. The setup_k + binning_k .vxbin is built from the Vortex SDK
 * source at driver build time (kernels/gfx_frontend/) and linked into
 * libvortexpipe as a byte array, so the draw path loads it from memory with
 * no runtime toolchain dependency -- the way a GPU ships its firmware.
 */

#ifndef VP_GFX_FRONTEND_H
#define VP_GFX_FRONTEND_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The embedded gfx_v2 front-end kernel image (.vxbin) for the requested
 * device XLEN (is64 = riscv64 image), exposing the "setup_k" and
 * "binning_k" entries. *size_out receives its byte length. */
const void *vp_gfx_frontend_vxbin(bool is64, size_t *size_out);

#ifdef __cplusplus
}
#endif

#endif /* VP_GFX_FRONTEND_H */
