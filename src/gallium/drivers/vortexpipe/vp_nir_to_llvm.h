/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_nir_to_llvm -- the scalar NIR -> LLVM-IR translator (Shape C).
 */

#ifndef VP_NIR_TO_LLVM_H
#define VP_NIR_TO_LLVM_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nir_shader;

/* Vertex-shader output layout. vortexpipe's VS kernel
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

/* Per-unit software-fallback routing (compile-time). When a unit is
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
 * per-unit HW/SW path for a fragment shader, `samples` its sample count
 * (0 or 1 = single-sample), and `bgra_mask` the channel order of each colour
 * attachment, one bit per render target (clear = red in the low byte, set =
 * blue); all three are ignored for VS/compute. Two fragment shaders differing
 * only in these are different kernels, which is why they are translation inputs
 * rather than draw-time arguments. */
bool vp_nir_to_llvm(struct nir_shader *nir, char **out_ir,
                    struct vp_vs_layout *out_vs,
                    const struct vp_sw_routing *routing,
                    unsigned samples, uint8_t bgra_mask);

/* Release a string returned via vp_nir_to_llvm()'s out_ir. */
void vp_free_ir(char *ir);

/* Kernel argument-block ABI (kernel_main(ptr arg) reads arg[i] as an i64
 * device address). Shared contract between vp_nir_to_llvm (emits arg[i]
 * reads) and vp_launch (fills argblk[i]):
 *   arg[0]                       -- push constants
 *   arg[1]                       -- set-0 descriptor blob (constant-buffer 1)
 *   arg[2]                       -- compute dispatch base: base_group_x in the
 *                                   low 32 bits, base_group_y in the high 32
 *                                   (vkCmdDispatchBase). 0 for a plain dispatch.
 *   arg[3]                       -- compute dispatch base: base_group_z (low 32).
 *   arg[VP_ARG_SSBO_BASE + slot] -- data address of a raw compute shader
 *                                   buffer bound at set_shader_buffers slot
 *                                   `slot`. Distinct from the descriptor-set
 *                                   SSBO path (those live in the set-0 blob):
 *                                   this is for internal buffers lavapipe binds
 *                                   directly, e.g. the RT trace-ray command
 *                                   buffer read as load_ssbo(imm slot, off). */
#define VP_ARG_GRID_BASE_XY 2   /* base_group_x | (base_group_y << 32) */
#define VP_ARG_GRID_BASE_Z  3   /* base_group_z                        */
#define VP_ARG_SSBO_BASE 4
#define VP_MAX_SSBO      4

/* The VERTEX stage overlays its own meanings on slots 0-5 (output record
 * buffer, attribute table, index buffer, verts-per-instance, base instance,
 * base vertex -- see vp_raster_draw's vs_argblk), so the slots above describe
 * COMPUTE only.
 * A vertex shader therefore cannot reach its constant buffers through slot 0/1
 * the way compute does; it gets its own descriptor table here, at a slot no
 * other stage assigns a meaning to. Its contents are the same
 * i64[GFX_FS_DESC_SLOTS] of constant-buffer device base addresses the fragment
 * stage receives, and the VS prologue routes it through t->arg so the shared
 * load_ubo / load_ssbo / load_push_constant lowering indexes the table rather
 * than the arg block. 0 for a VS that binds no constant buffers. */
#define VP_ARG_VS_DESC   (VP_ARG_SSBO_BASE + VP_MAX_SSBO)
#define VP_ARG_SLOTS     (VP_ARG_VS_DESC + 1)

/* A descriptor a compute kernel reaches through set-0's descriptor
 * buffer (constant-buffer index 1):
 *  - VP_DESC_BUFFER: an SSBO -- lp_jit_buffer{ptr,size} at `offset`.
 *  - VP_DESC_AS: an acceleration structure -- accel_struct device
 *    address at `offset`.
 *  - VP_DESC_IMAGE: a storage image -- lp_jit_image at `offset`; base at +0
 *    (aliases lp_jit_buffer.ptr), size derived from height*row_stride. */
enum vp_desc_kind { VP_DESC_BUFFER, VP_DESC_AS, VP_DESC_IMAGE };
struct vp_desc {
   unsigned          offset;   /* byte offset in the descriptor buffer */
   /* The constant-buffer index holding this descriptor = descriptor set + 1
    * (lavapipe binds set N's descriptor blob at constant-buffer index N+1). The
    * FS reaches it via load_const_buf_base_addr_lvp(cbuf_index); the relocation
    * must rewrite the lp_jit_buffer.ptr inside that set's blob, not always set 0. */
   unsigned          cbuf_index;
   enum vp_desc_kind kind;
   /* Byte multiplier for the descriptor's lp_jit_buffer.num_elements field (at
    * +8): an SSBO stores num_elements in bytes (1), a UBO in dwords (4, from
    * lp_jit_buffer_from_pipe_const's DIV_ROUND_UP(size, sizeof(float))). Used
    * by the descriptor relocation to size the device upload; 0 for AS/image. */
   unsigned          elem_bytes;
   /* True when this shader writes the descriptor: store_ssbo, an SSBO atomic,
    * an image store, or an image atomic. Only then does the device copy need
    * reading back into the host backing after the launch — a descriptor that is
    * only ever loaded holds the bytes it was uploaded with, and copying those
    * back is pure work. */
   bool              writable;
};
#define VP_MAX_DESCS 16

/* Scan a compute NIR for the set-0 descriptors it accesses (load_ssbo
 * / store_ssbo / load_ubo against constant-buffer index 1). Fills
 * out[0..*num_out) with the distinct descriptors found, capped at
 * VP_MAX_DESCS. */
void vp_scan_descriptors(struct nir_shader *nir,
                         struct vp_desc *out, unsigned *num_out);

/* Locate the FS's TEX-stage-0 sampled-image descriptor (cbuf_index, byte offset)
 * from its nir_tex_src_texture_handle, so a draw can read lp_jit_texture.base and
 * select the actually-sampled texture. Returns false if the FS has no bindless
 * texture handle (single-texture / non-bindless path uses the captured cur_tex). */
bool vp_scan_tex_descriptor(struct nir_shader *nir,
                            unsigned *cbuf_index, unsigned *offset);

/* Bitmask of set_shader_buffers slots read via constant-index load_ssbo/
 * store_ssbo (the RT trace-ray command buffer). See the definition. */
unsigned vp_scan_trace_cmd_slots(struct nir_shader *nir);

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
