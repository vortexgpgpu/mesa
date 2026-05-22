/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_nir_to_llvm -- the scalar NIR -> LLVM-IR translator (Shape C).
 * See docs/proposals/vulkan_support_proposal.md §3 in the Vortex tree.
 */

#ifndef VP_NIR_TO_LLVM_H
#define VP_NIR_TO_LLVM_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nir_shader;

/* Translate a NIR shader to LLVM IR. On success returns true and, if
 * out_ir is non-NULL, stores a freshly allocated LLVM-IR text string
 * in *out_ir (release it with vp_free_ir). */
bool vp_nir_to_llvm(struct nir_shader *nir, char **out_ir);

/* Release a string returned via vp_nir_to_llvm()'s out_ir. */
void vp_free_ir(char *ir);

#ifdef __cplusplus
}
#endif

#endif /* VP_NIR_TO_LLVM_H */
