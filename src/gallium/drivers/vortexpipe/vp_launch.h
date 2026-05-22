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

#ifdef __cplusplus
extern "C" {
#endif

/* Run a compiled Vortex compute kernel (.vxbin) on `dev`.
 *
 * The single shader-storage buffer's data is at ssbo_host[0,
 * ssbo_bytes): it is copied to device memory, the kernel runs over
 * grid x block, and the result is copied back into ssbo_host.
 *
 * Returns true on success (add1/vecadd-class single-SSBO kernels).
 */
bool vp_launch(vx_device_h dev,
               const void *vxbin, size_t vxbin_size,
               void *ssbo_host, uint32_t ssbo_bytes,
               const uint32_t grid[3], const uint32_t block[3]);

#ifdef __cplusplus
}
#endif

#endif /* VP_LAUNCH_H */
