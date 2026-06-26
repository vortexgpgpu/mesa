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
   unsigned stride;                          /* bytes per output vertex */
   unsigned num_varyings;                    /* generic varyings (after POS) */
   int      varying_loc[VP_VS_MAX_VARYINGS];  /* VARYING_SLOT_* per varying */
   unsigned varying_comps[VP_VS_MAX_VARYINGS];/* component count per varying */
   bool     needs_vertex_input;              /* VS fetches vertex attributes */
};

/* gfx_v2 §5 per-unit software-fallback routing (compile-time). When a unit is
 * routed to software the FS calls the gfx_sw_abi entry points instead of the FF
 * intrinsic; the host feeds resident descriptors via the kernel arg block. A
 * NULL routing (or all-false) is the all-hardware path. */
struct vp_sw_routing {
   bool sw_tex;      /* sample via gfx_tex_sample_sw instead of vx_tex4 */
   bool sw_om;       /* merge via gfx_om_fragment_sw instead of vx_om4 */
   bool sw_raster;   /* (vp_raster_draw selects the iterate-bin-buffer kernel) */
};

/* Translate a NIR shader (compute or vertex) to LLVM IR. On success
 * returns true and, if out_ir is non-NULL, stores a freshly
 * allocated LLVM-IR text string in *out_ir (release with
 * vp_free_ir). For a vertex shader, *out_vs (when non-NULL) is
 * filled with the output-record layout. `routing` (may be NULL) selects the
 * per-unit HW/SW path for a fragment shader; ignored for VS/compute. */
bool vp_nir_to_llvm(struct nir_shader *nir, char **out_ir,
                    struct vp_vs_layout *out_vs,
                    const struct vp_sw_routing *routing);

/* Release a string returned via vp_nir_to_llvm()'s out_ir. */
void vp_free_ir(char *ir);

/* A descriptor a compute kernel reaches through set-0's descriptor
 * buffer (constant-buffer index 1):
 *  - VP_DESC_BUFFER: an SSBO -- lp_jit_buffer{ptr,size} at `offset`.
 *  - VP_DESC_AS: an acceleration structure -- accel_struct device
 *    address at `offset`. */
enum vp_desc_kind { VP_DESC_BUFFER, VP_DESC_AS };
struct vp_desc {
   unsigned          offset;   /* byte offset in the set-0 descriptor buffer */
   enum vp_desc_kind kind;
};
#define VP_MAX_DESCS 16

/* Scan a compute NIR for the set-0 descriptors it accesses (load_ssbo
 * / store_ssbo / load_ubo against constant-buffer index 1). Fills
 * out[0..*num_out) with the distinct descriptors found, capped at
 * VP_MAX_DESCS. */
void vp_scan_descriptors(struct nir_shader *nir,
                         struct vp_desc *out, unsigned *num_out);

/* Lower Vulkan ray-query / ray-tracing NIR (rq_* and trace_ray, left
 * intact by lavapipe when the driver advertises driver_ray_queries) into
 * the Vortex RTU vendor intrinsics (vortex_rt_wtrace/wait/get/cb_ret),
 * which vp_nir_to_llvm emits as CUSTOM1 .insn ops. Returns true if it
 * changed the shader. No-op (returns false) when the shader has no ray
 * queries. See vp_nir_lower_ray_tracing_to_rtu.c. */
bool vp_nir_lower_ray_tracing_to_rtu(struct nir_shader *nir);

#ifdef __cplusplus
}
#endif

#endif /* VP_NIR_TO_LLVM_H */
