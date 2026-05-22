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

/* Vertex-shader output layout (Phase 3). vortexpipe's VS kernel
 * writes one record per vertex into a device buffer; the draw
 * integration reads it back with this layout. All sizes in bytes.
 * Each slot is a padded vec4 (16 bytes): slot 0 is gl_Position,
 * slots 1.. are the generic varyings in declaration order. */
#define VP_VS_MAX_VARYINGS 15

struct vp_vs_layout {
   unsigned stride;                         /* bytes per output vertex */
   unsigned num_varyings;                   /* generic varyings (after POS) */
   int      varying_loc[VP_VS_MAX_VARYINGS]; /* VARYING_SLOT_* per varying */
};

/* Translate a NIR shader (compute or vertex) to LLVM IR. On success
 * returns true and, if out_ir is non-NULL, stores a freshly
 * allocated LLVM-IR text string in *out_ir (release with
 * vp_free_ir). For a vertex shader, *out_vs (when non-NULL) is
 * filled with the output-record layout. */
bool vp_nir_to_llvm(struct nir_shader *nir, char **out_ir,
                    struct vp_vs_layout *out_vs);

/* Release a string returned via vp_nir_to_llvm()'s out_ir. */
void vp_free_ir(char *ir);

#ifdef __cplusplus
}
#endif

#endif /* VP_NIR_TO_LLVM_H */
