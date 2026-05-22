/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_compile -- Phase 2 #4b: drive LLVM IR -> Vortex .vxbin.
 */

#ifndef VP_COMPILE_H
#define VP_COMPILE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile LLVM-IR text (a Vortex KMU kernel module, as produced by
 * vp_nir_to_llvm) into a Vortex .vxbin image. On success returns
 * true and stores a malloc'd blob + size in *out_blob / *out_size;
 * release it with vp_free_blob(). */
bool vp_compile_vxbin(const char *llvm_ir, void **out_blob, size_t *out_size);

void vp_free_blob(void *blob);

#ifdef __cplusplus
}
#endif

#endif /* VP_COMPILE_H */
