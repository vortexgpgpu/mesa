/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_nir_to_llvm -- scalar NIR -> LLVM-IR translator (Shape C).
 *
 * Emits a Vortex KMU kernel: `void kernel_main(ptr %arg)`, marked
 * `vortex.kernel` via @llvm.global.annotations, targeting riscv32
 * or riscv64 depending on $MESA_VORTEX_XLEN (see vp_xlen_is_64). Two
 * shader stages are handled:
 *
 *   - Compute: one thread per work-item. SSBO base addresses come
 *     from the lavapipe descriptor buffer, reached through %arg.
 *   - Vertex: one thread per vertex. gl_VertexIndex is the
 *     CTA thread id; the kernel writes one padded-vec4 record per
 *     vertex into the output buffer whose device address is %arg[0].
 *
 * NIR SSA values are untyped bit patterns; each component is held as
 * an LLVM iN and ops bit-cast to the type they need. NIR derefs
 * resolve to iptr byte addresses (i32 on rv32, i64 on rv64).
 */

#include "vp_nir_to_llvm.h"
#include "vp_compile.h"      /* vp_xlen_is_64 */
#include "vp_private.h"      /* vp_dbg */
#include "gfx_fs_desc_abi.h" /* GFX_FS_ARG_DESC, GFX_FS_ARG_APERTURE */
#include "VX_types.h"        /* VX_MEM_OM_BASE_ADDR */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir.h"
#include "compiler/glsl_types.h"
#include "util/log.h"
#include "util/format/u_formats.h"

#include <llvm-c/Core.h>
#include <llvm-c/Analysis.h>

#define VP_MAXC 4    /* max vector components per NIR def */
#define VP_MAXV 64   /* max tracked variables per shader */

/* KMU CTA CSRs (VX_types.h): per-thread / per-block index registers. */
#define VX_CSR_CTA_THREAD_ID_X 0xCD3   /* local invocation id, +c for y/z */
#define VX_CSR_CTA_BLOCK_ID_X  0xCD6   /* workgroup id,        +c for y/z */
#define VX_CSR_CTA_BLOCK_DIM_X 0xCD9   /* workgroup size,      +c for y/z */
#define VX_CSR_CTA_ID          0xCD0   /* workgroup id (barrier id)        */
#define VX_CSR_CTA_SIZE        0xCD2   /* warps per workgroup              */
#define VX_CSR_CTA_GRID_DIM_X  0xCDC   /* workgroup count, +c for y/z      */
#define VX_CSR_CTA_LMEM_ADDR   0xCDF   /* shared-memory base for this CTA  */
#define VX_CSR_THREAD_ID       0xCC0   /* SIMD lane id (0..NUM_THREADS-1)  */
#define VX_CSR_WARP_ID         0xCC1   /* warp id within the core          */
#define VX_CSR_NUM_THREADS     0xFC0   /* threads per warp                 */

/* RISC-V custom-0 opcode -- vx_barrier lives here (custom-1 is graphics). */
#define VP_RISCV_CUSTOM0       11

/* RASTER CSRs (VX_types.h) -- latched per vx_rast() pop. */
#define VX_CSR_RASTER_BCOORD_X0 0x7C1  /* +i selects sub-pixel i (0..3) */
#define VX_CSR_RASTER_BCOORD_Y0 0x7C5
#define VX_CSR_RASTER_BCOORD_Z0 0x7C9
#define VX_CSR_RASTER_PID       0x7CD
#define VX_RASTER_DIM_BITS      15     /* pos_mask x/y field width + 1 */
/* A quad is 2x2 pixels, so a quad group is four adjacent lanes: lane L holds
 * corner L&3. Mirrors VX_FRAG_QUAD_LANES in the kernel ABI (sw/common/vx_gfx_abi.h).
 * The quad SHFL that carries a derivative permutes within this group. */
#define VX_FRAG_QUAD_LANES      4
/* Segment operands that scope a SHFL to a quad group: lane l's group is
 * [l & ~3, (l & ~3) | 3]. Mirrors VX_QUAD_CVAL / VX_QUAD_MASK in vx_intrinsics.h. */
#define VP_QUAD_SHFL_CVAL       3
#define VP_QUAD_SHFL_MASK       0x3c
/* frag_payload_t in the gfx window: word [0]=pos, [1]=pid.
 * The base sits outside the RTU object-ray range [8..13] at [19..20] so a
 * fragment shader can hold this record and an in-flight RTU query at once. Must
 * stay equal to the kernel ABI VX_GFX_FRAG_SLOT_BASE (VX_types.toml [gfx_window]). */
#define VP_FRAG_SLOT_BASE       19
/* gfx-window slot allocation for the FS (disjoint from the frag record 19..20):
 *   OM quad staging  : color[0..3] @ 0..3, depth[0..3] @ 4..7
 *   TEX windowed I/O : u @ 22, v @ 23, texel @ 24 */
#define VP_OM_SLOT_BASE         0
#define VP_TEX_IN_SLOT          22
#define VP_TEX_OUT_SLOT         24

/* vortex::graphics::rast_prim_t layout (sw/common/vx_gfx_abi.h, FIXEDPOINT):
 * vec3e_t edges[3] (36B), then rast_attribs_t {z,r,g,b,a,u,v,rhw}, each a
 * rast_attrib_t {x,y,z} of fixed24 (12B). r/g/b/a are the colour planes,
 * u/v the texcoord planes; rhw is the perspective 1/w plane (appended last,
 * so the z/r/g/b/a/u/v offsets below are unchanged). At w==1 the premultiplied
 * colour/uv planes equal the raw attributes and rhw is constant, so reading them
 * affinely is exact; the trailing rhw pushes the per-prim stride to 132B.
 * NOTE: duplicated as VP_RAST_PRIM_STRIDE in vp_raster.cpp (the DCR writer) —
 * keep the two in sync. */
#define VP_RAST_PRIM_STRIDE 132
#define VP_RAST_ATTR_Z       36
#define VP_RAST_ATTR_R       48
#define VP_RAST_ATTR_G       60
#define VP_RAST_ATTR_B       72
#define VP_RAST_ATTR_A       84
#define VP_RAST_ATTR_U       96
#define VP_RAST_ATTR_V      108
#define VP_RAST_ATTR_RHW    120

/* TEX unit: coordinates are S.23 fixed-point (VX_types.h VX_TEX_FXD_FRAC). */
#define VP_TEX_FXD_FRAC      23

/* Module target -- the LLVM triple + datalayout follow $MESA_VORTEX_XLEN.
 * rv32 datalayout: 32-bit pointers, 32-bit native int.
 * rv64 datalayout: 64-bit pointers (p:64:64), 64-bit native int (n64). */
static inline const char *vp_target_triple(void) {
   return vp_xlen_is_64() ? "riscv64-unknown-elf" : "riscv32-unknown-elf";
}
static inline const char *vp_target_datalayout(void) {
   return vp_xlen_is_64()
      ? "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
      : "e-m:e-p:32:32-i64:64-n32-S128";
}

/* A tracked NIR variable: either a function_temp (LLVM alloca) or a
 * shader_out (byte offset of its slot in the per-vertex record). */
struct vp_var {
   const nir_variable *var;
   LLVMValueRef        alloca;     /* function_temp storage, else NULL */
   int                 out_off;    /* shader_out slot offset, else -1 */
};

struct vp_tr {
   LLVMContextRef ctx;
   LLVMModuleRef  mod;
   LLVMBuilderRef b;
   LLVMTypeRef    i8, i32, i64, f32, f64, ptr;
   /* iptr: pointer-sized int (i32 on rv32, i64 on rv64). Every value
    * that holds a *device byte address* uses this type, so the LLVM
    * Add/Mul/IntToPtr ops survive on both XLENs. */
   LLVMTypeRef    iptr;
   LLVMValueRef   arg;          /* the kernel's %arg parameter (ptr) */
   LLVMValueRef   lmem_base;    /* compute: shared-memory base (CTA LMEM) */
   /* Per-thread scratch: nir->scratch_size bytes as one entry-block alloca,
    * created lazily on the first load_scratch/store_scratch (indexed scratch
    * that nir_lower_scratch_to_var could not promote to SSA vars). */
   unsigned       scratch_size;
   LLVMValueRef   scratch_base; /* iptr addr of the scratch alloca, or NULL */
   LLVMBasicBlockRef entry;     /* function entry block (alloca home) */
   /* SSA map: [def index][component] -> iN value (bit pattern).
    * For a deref instr, component 0 holds the iptr byte address. */
   LLVMValueRef  *val;
   unsigned       nval;
   /* vertex-shader state (is_vs only) */
   bool           is_vs;
   LLVMValueRef   vid;          /* i32 vertex id (index-resolved): gl_VertexIndex
                                 * + attribute fetch use this. */
   LLVMValueRef   vraw;         /* i32 sequential global id: VS output slot. For a
                                 * direct draw vraw == vid; for an indexed draw
                                 * vid = index_buf[vraw]. */
   /* Instancing (is_vs only). The VS runs instance_count × verts_per_instance
    * threads; instance = gid / vpi (0-based, gl_InstanceID) and first_instance =
    * gl_BaseInstance. gl_InstanceIndex = instance + first_instance (NIR lowers it
    * to load_instance_id + load_base_instance). Both null on the non-instanced
    * fast path (verts_per_instance == 0). */
   LLVMValueRef   instance;       /* i32 0-based instance id (gl_InstanceID) */
   LLVMValueRef   first_instance; /* i32 base instance (gl_BaseInstance) */
   LLVMValueRef   out_base;     /* iptr output-buffer device address */
   unsigned       out_stride;   /* bytes per output vertex record */
   LLVMValueRef   attr_table;   /* iptr addr of the {base,stride}[] table */
   /* fragment-shader state (is_fs only) */
   bool           is_fs;
   LLVMValueRef   fs_in_base;   /* iptr interpolated-varyings area */
   LLVMValueRef   fs_out_base;  /* iptr output-colour area */
   struct vp_var  vars[VP_MAXV];
   unsigned       nvars;
   /* Per-unit SW routing (FS only). fs_texstate is fs_main's 3rd
    * param: a resident gfx_sw_texstate_t[] (per sampler stage), used by
    * emit_vx_tex when sw_tex; null pointer when texturing is HW. */
   bool           sw_tex;
   bool           sw_om;
   bool           sw_raster;  /* FS is the one-warp-per-tile SW-raster kernel */
   LLVMValueRef   fs_texstate; /* ptr param (gfx_sw_texstate_t* table) */
   /* fs_main's 5th param: an i32 the body clears on discard/demote. A discarded
    * lane keeps running -- it is still a helper for its quad neighbours' derivatives
    * -- so the flag withholds its export rather than its execution. Its side effects
    * must still be withheld, though: a discarded invocation may not commit stores,
    * so an FS store retargets to fs_sink, a dead per-thread scratch word. */
   LLVMValueRef   fs_live;
   LLVMValueRef   fs_sink;   /* i64 addr of the scratch a discarded store lands in */

   /* OM aperture geometry, unpacked from the FS arg block (per-draw, so it cannot
    * be a JIT constant). A fragment export's address is formed by shifting. */
   LLVMValueRef   om_ap_xbits;
   LLVMValueRef   om_ap_ybits;
   LLVMValueRef   om_ap_shift;
   /* Number of colour attachments the FS writes (1 = single RT, the
    * byte-identical fast path). Colour output k occupies out slot k*16 in the
    * FS output area; the wrapper packs slots 0..num_color-1 into a src_colors[]
    * array and calls gfx_om_fragment_mrt_sw when num_color > 1. */
   unsigned       fs_num_color;
   unsigned       fs_out_words;  /* i32 words in the FS output area (incl. scratch) */
   bool           ok;
};

/* iptr-typed constant (i32 on rv32, i64 on rv64). */
static inline LLVMValueRef
vp_iptr_const(struct vp_tr *t, uint64_t v)
{
   return LLVMConstInt(t->iptr, v, false);
}

/* ZExt an i32 value to iptr (no-op on rv32). */
static inline LLVMValueRef
vp_to_iptr(struct vp_tr *t, LLVMValueRef v)
{
   if (t->iptr == t->i32)
      return v;
   return LLVMBuildZExt(t->b, v, t->iptr, "");
}

/* Narrow an iptr value to i32 (no-op on rv32). */
static inline LLVMValueRef
emit_to_i32(struct vp_tr *t, LLVMValueRef v)
{
   if (t->iptr == t->i32)
      return v;
   return LLVMBuildTrunc(t->b, v, t->i32, "");
}

static LLVMValueRef
ssa_get(struct vp_tr *t, unsigned idx, unsigned comp)
{
   if (idx >= t->nval || comp >= VP_MAXC)
      return NULL;
   return t->val[idx * VP_MAXC + comp];
}

static void
ssa_set(struct vp_tr *t, unsigned idx, unsigned comp, LLVMValueRef v)
{
   if (idx < t->nval && comp < VP_MAXC)
      t->val[idx * VP_MAXC + comp] = v;
}

static LLVMValueRef
alu_src(struct vp_tr *t, nir_alu_instr *alu, unsigned s, unsigned comp)
{
   return ssa_get(t, alu->src[s].src.ssa->index, alu->src[s].swizzle[comp]);
}

static LLVMValueRef
intr_src(struct vp_tr *t, nir_intrinsic_instr *in, unsigned s)
{
   return ssa_get(t, in->src[s].ssa->index, 0);
}

/* llvm.sqrt.f32 -> the RISC-V backend lowers it to the Vortex FPU fsqrt.s.
 * Declared once per module (LLVM recognises the reserved intrinsic name). */
static LLVMValueRef
emit_fsqrt(struct vp_tr *t, LLVMValueRef fa)
{
   LLVMTypeRef  fty = LLVMFunctionType(t->f32, &t->f32, 1, false);
   LLVMValueRef fn  = LLVMGetNamedFunction(t->mod, "llvm.sqrt.f32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "llvm.sqrt.f32", fty);
   return LLVMBuildCall2(t->b, fty, fn, &fa, 1, "fsqrt");
}

/* llvm.ctlz.i32 (count leading zeros, zero-is-not-poison) -> Vortex clz. */
static LLVMValueRef
emit_ctlz(struct vp_tr *t, LLVMValueRef v)
{
   LLVMTypeRef  i1   = LLVMInt1TypeInContext(t->ctx);
   LLVMTypeRef  args[2] = { t->i32, i1 };
   LLVMTypeRef  fty  = LLVMFunctionType(t->i32, args, 2, false);
   LLVMValueRef fn   = LLVMGetNamedFunction(t->mod, "llvm.ctlz.i32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "llvm.ctlz.i32", fty);
   LLVMValueRef a[2] = { v, LLVMConstInt(i1, 0, false) };
   return LLVMBuildCall2(t->b, fty, fn, a, 2, "ctlz");
}

/* llvm.pow.f32 (x**y). Lowered by the RISC-V backend to a powf libcall. */
static LLVMValueRef
emit_fpow(struct vp_tr *t, LLVMValueRef x, LLVMValueRef y)
{
   LLVMTypeRef  args[2] = { t->f32, t->f32 };
   LLVMTypeRef  fty = LLVMFunctionType(t->f32, args, 2, false);
   LLVMValueRef fn  = LLVMGetNamedFunction(t->mod, "llvm.pow.f32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "llvm.pow.f32", fty);
   LLVMValueRef a[2] = { x, y };
   return LLVMBuildCall2(t->b, fty, fn, a, 2, "fpow");
}

/* llvm.ctpop.i32 (population count / set-bit count). */
static LLVMValueRef
emit_ctpop(struct vp_tr *t, LLVMValueRef v)
{
   LLVMTypeRef  fty = LLVMFunctionType(t->i32, &t->i32, 1, false);
   LLVMValueRef fn  = LLVMGetNamedFunction(t->mod, "llvm.ctpop.i32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "llvm.ctpop.i32", fty);
   return LLVMBuildCall2(t->b, fty, fn, &v, 1, "ctpop");
}

/* Coerce integer value `v` to integer type `ty` by trunc/zext (identity when
 * already `ty`). Used to match a shift amount to its value's width — NIR allows
 * a narrower shift amount, LLVM requires both shift operands the same type. */
static LLVMValueRef
vp_int_cast(struct vp_tr *t, LLVMValueRef v, LLVMTypeRef ty)
{
   LLVMTypeRef vt = LLVMTypeOf(v);
   if (vt == ty)
      return v;
   return LLVMGetIntTypeWidth(vt) > LLVMGetIntTypeWidth(ty)
      ? LLVMBuildTrunc(t->b, v, ty, "")
      : LLVMBuildZExt(t->b, v, ty, "");
}

/* llvm.cttz.i32 (count trailing zeros, zero-is-not-poison). */
static LLVMValueRef
emit_cttz(struct vp_tr *t, LLVMValueRef v)
{
   LLVMTypeRef  i1   = LLVMInt1TypeInContext(t->ctx);
   LLVMTypeRef  args[2] = { t->i32, i1 };
   LLVMTypeRef  fty  = LLVMFunctionType(t->i32, args, 2, false);
   LLVMValueRef fn   = LLVMGetNamedFunction(t->mod, "llvm.cttz.i32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "llvm.cttz.i32", fty);
   LLVMValueRef a[2] = { v, LLVMConstInt(i1, 0, false) };
   return LLVMBuildCall2(t->b, fty, fn, a, 2, "cttz");
}

/* vx ballot (custom-0, funct3=3, funct7=1): rd = per-lane predicate reduced to
 * a warp bitmask (bit i = lane i's predicate). Side-effecting so the optimizer
 * cannot hoist it across the divergent control flow whose mask it reports. */
static LLVMValueRef
emit_ballot(struct vp_tr *t, LLVMValueRef pred_i32)
{
   char s[48];
   int n = snprintf(s, sizeof s, ".insn r %u, 3, 1, $0, $1, x0", VP_RISCV_CUSTOM0);
   LLVMTypeRef args[1] = { t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 1, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, (size_t)n, "=r,r", 4,
                                      /*HasSideEffects*/ true, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[1] = { pred_i32 };
   return LLVMBuildCall2(t->b, fnty, ia, a, 1, "ballot");
}

/* vx shuffle (custom-0, funct7=1): rd = value from a source lane selected by
 * `funct3` (4=up, 5=down, 6=bfly, 7=idx) and the packed `bc` control operand
 * (mask<<12 | cval<<6 | bval). Whole-warp scope uses cval=0x3f, mask=0 (both
 * truncated to the lane-index width by hardware) — since NUM_ALU_LANES ==
 * NUM_THREADS a warp is one shuffle group. Side-effecting so it is not hoisted
 * across the divergence whose active mask decides which source lanes are live. */
static LLVMValueRef
emit_shfl(struct vp_tr *t, unsigned funct3, LLVMValueRef value, LLVMValueRef bc)
{
   char s[48];
   int n = snprintf(s, sizeof s, ".insn r %u, %u, 1, $0, $1, $2",
                    VP_RISCV_CUSTOM0, funct3);
   LLVMTypeRef args[2] = { t->i32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 2, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, (size_t)n, "=r,r,r", 6,
                                      /*HasSideEffects*/ true, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[2] = { value, bc };
   return LLVMBuildCall2(t->b, fnty, ia, a, 2, "shfl");
}

/* shfl_up by a constant delta: bval=delta, cval=0x3f, mask=0 -> 0xFC0|delta. */
static LLVMValueRef
emit_shfl_up(struct vp_tr *t, LLVMValueRef value, unsigned delta)
{
   return emit_shfl(t, 4, value, LLVMConstInt(t->i32, 0xFC0u | delta, false));
}

/* shfl_idx (gather): every lane reads `value` from source lane `src` (dynamic).
 * bval=src, cval=0x3f, mask=0 -> 0xFC0 | src. */
static LLVMValueRef
emit_shfl_idx(struct vp_tr *t, LLVMValueRef value, LLVMValueRef src)
{
   LLVMValueRef bc = LLVMBuildOr(t->b, LLVMConstInt(t->i32, 0xFC0u, false),
                                 src, "shflidx_bc");
   return emit_shfl(t, 7, value, bc);
}

/* Combine two lane values under a subgroup reduction op (32-bit). Operands and
 * result are i32 bit patterns; float ops bitcast around the arithmetic. Returns
 * NULL (and clears t->ok) for an op with no device mapping. */
static LLVMValueRef
emit_scan_combine(struct vp_tr *t, nir_op rop, LLVMValueRef a, LLVMValueRef b)
{
   switch (rop) {
   case nir_op_iadd: return LLVMBuildAdd(t->b, a, b, "");
   case nir_op_imul: return LLVMBuildMul(t->b, a, b, "");
   case nir_op_iand: return LLVMBuildAnd(t->b, a, b, "");
   case nir_op_ior:  return LLVMBuildOr(t->b, a, b, "");
   case nir_op_ixor: return LLVMBuildXor(t->b, a, b, "");
   case nir_op_imin: return LLVMBuildSelect(t->b,
      LLVMBuildICmp(t->b, LLVMIntSLT, a, b, ""), a, b, "");
   case nir_op_imax: return LLVMBuildSelect(t->b,
      LLVMBuildICmp(t->b, LLVMIntSGT, a, b, ""), a, b, "");
   case nir_op_umin: return LLVMBuildSelect(t->b,
      LLVMBuildICmp(t->b, LLVMIntULT, a, b, ""), a, b, "");
   case nir_op_umax: return LLVMBuildSelect(t->b,
      LLVMBuildICmp(t->b, LLVMIntUGT, a, b, ""), a, b, "");
   case nir_op_fadd: case nir_op_fmul: case nir_op_fmin: case nir_op_fmax: {
      LLVMValueRef fa = LLVMBuildBitCast(t->b, a, t->f32, "");
      LLVMValueRef fb = LLVMBuildBitCast(t->b, b, t->f32, "");
      LLVMValueRef fr =
         rop == nir_op_fadd ? LLVMBuildFAdd(t->b, fa, fb, "")
       : rop == nir_op_fmul ? LLVMBuildFMul(t->b, fa, fb, "")
       : rop == nir_op_fmin ? LLVMBuildSelect(t->b,
            LLVMBuildFCmp(t->b, LLVMRealOLT, fa, fb, ""), fa, fb, "")
       :                      LLVMBuildSelect(t->b,
            LLVMBuildFCmp(t->b, LLVMRealOGT, fa, fb, ""), fa, fb, "");
      return LLVMBuildBitCast(t->b, fr, t->i32, "");
   }
   default:
      mesa_logw("vortexpipe: subgroup scan op %d unsupported", rop);
      t->ok = false;
      return NULL;
   }
}

/* read a RISC-V CSR via inline asm: `csrr <rd>, <csr>` -> i32 */
static LLVMValueRef
emit_csr_read(struct vp_tr *t, unsigned csr, const char *name)
{
   char s[32];
   int n = snprintf(s, sizeof s, "csrr $0, %u", csr);
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, NULL, 0, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, (size_t)n, "=r", 2,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   return LLVMBuildCall2(t->b, fnty, ia, NULL, 0, name);
}

/* vx_barrier(): a workgroup execution barrier (custom-0, funct3=4).
 * The barrier id is the CTA id and the count is the CTA's warp count,
 * matching vx_spawn2.h's __syncthreads(). */
static void
emit_vx_barrier(struct vp_tr *t)
{
   char s[40];
   int n = snprintf(s, sizeof s, ".insn r %u, 4, 0, x0, $0, $1",
                    VP_RISCV_CUSTOM0);
   LLVMTypeRef args[2] = { t->i32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                       args, 2, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, (size_t)n, "r,r", 3,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   LLVMValueRef a[2] = { emit_csr_read(t, VX_CSR_CTA_ID,   "cta_id"),
                         emit_csr_read(t, VX_CSR_CTA_SIZE, "cta_warps") };
   LLVMBuildCall2(t->b, fnty, ia, a, 2, "");
}

/* VS vertex-input fetch: the device address of attribute `loc` for the
 * current vertex. The attribute table (arg slot 1) holds, per VS input
 * driver_location, a { device base, stride } pair; the attribute lives
 * at base + vid*stride (vp_launch_vs builds the table). base/stride
 * are i32 fields on the wire even on rv64 (the table layout is the
 * same), so widen base to iptr before address arithmetic. */
static LLVMValueRef
emit_vs_attr_addr(struct vp_tr *t, unsigned loc)
{
   LLVMValueRef ent = LLVMBuildAdd(t->b, t->attr_table,
      vp_iptr_const(t, loc * 8u), "");
   LLVMValueRef base = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildIntToPtr(t->b, ent, t->ptr, ""), "attrbase");
   LLVMValueRef stride = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildIntToPtr(t->b,
         LLVMBuildAdd(t->b, ent, vp_iptr_const(t, 4), ""),
         t->ptr, ""), "attrstride");
   LLVMValueRef offset = LLVMBuildMul(t->b, t->vid, stride, "");
   return LLVMBuildAdd(t->b, vp_to_iptr(t, base),
                              vp_to_iptr(t, offset), "vsin");
}

/* byte size of a glsl type (32- or 64-bit scalars/vectors, arrays). */
static unsigned
glsl_bytes(const struct glsl_type *gt)
{
   if (glsl_type_is_array(gt))
      return glsl_get_length(gt) * glsl_bytes(glsl_get_array_element(gt));
   return glsl_get_components(gt) * (glsl_get_bit_size(gt) / 8u);
}

/* LLVM storage type for a glsl type; each scalar is an i32 or i64. */
static LLVMTypeRef
glsl_to_llvm(struct vp_tr *t, const struct glsl_type *gt)
{
   if (glsl_type_is_array(gt))
      return LLVMArrayType(glsl_to_llvm(t, glsl_get_array_element(gt)),
                           glsl_get_length(gt));
   LLVMTypeRef scalar = glsl_get_bit_size(gt) == 64 ? t->i64 : t->i32;
   unsigned n = glsl_get_components(gt);
   return n > 1 ? LLVMArrayType(scalar, n) : scalar;
}

static struct vp_var *
vp_var_find(struct vp_tr *t, const nir_variable *v)
{
   for (unsigned i = 0; i < t->nvars; i++)
      if (t->vars[i].var == v)
         return &t->vars[i];
   return NULL;
}

static void
emit_load_const(struct vp_tr *t, nir_load_const_instr *lc)
{
   for (unsigned c = 0; c < lc->def.num_components; c++) {
      LLVMValueRef v =
         lc->def.bit_size == 64 ? LLVMConstInt(t->i64, lc->value[c].u64, false)
       : lc->def.bit_size == 1  ? LLVMConstInt(LLVMInt1TypeInContext(t->ctx),
                                               lc->value[c].b, false)
       :                          LLVMConstInt(t->i32, lc->value[c].u32, false);
      ssa_set(t, lc->def.index, c, v);
   }
}

/* the LLVM float / integer type matching a NIR scalar bit size --
 * the translator carries f64 values as i64 bit patterns (rv32 has no
 * D extension; the .vxbin compiler lowers f64 ops to soft-float). */
static LLVMTypeRef
fty(struct vp_tr *t, unsigned bits) { return bits == 64 ? t->f64 : t->f32; }
static LLVMTypeRef
ity(struct vp_tr *t, unsigned bits) { return bits == 64 ? t->i64 : t->i32; }

/* quad-derivative helper (defined with the other Vortex intrinsics, below). */
static LLVMValueRef emit_quad_deriv(struct vp_tr *t, LLVMValueRef value,
                                    unsigned dir);

/* The address a store should actually use. A discarded fragment invocation keeps
 * running so its quad neighbours can still shuffle from it, but it may not commit
 * side effects -- so its stores are steered into a dead per-thread word instead.
 * Outside a fragment shader there is nothing to discard and the address stands. */
static LLVMValueRef
emit_store_addr(struct vp_tr *t, LLVMValueRef addr)
{
   if (!t->is_fs || !t->fs_live || !t->fs_sink)
      return addr;
   LLVMValueRef live = LLVMBuildLoad2(t->b, t->i32, t->fs_live, "live.v");
   LLVMValueRef ok = LLVMBuildICmp(t->b, LLVMIntNE, live,
                                   LLVMConstInt(t->i32, 0, false), "");
   /* the sink is pointer-sized; a store address may be wider (ssbo bases are i64) */
   LLVMValueRef sink = LLVMBuildZExtOrBitCast(t->b, t->fs_sink,
                                              LLVMTypeOf(addr), "sink");
   return LLVMBuildSelect(t->b, ok, addr, sink, "store_addr");
}

static void
emit_alu(struct vp_tr *t, nir_alu_instr *alu)
{
   for (unsigned c = 0; c < alu->def.num_components; c++) {
      LLVMValueRef r = NULL;
      switch (alu->op) {
      case nir_op_mov:
         r = alu_src(t, alu, 0, c);
         break;
      /* vecN: component c is built from source operand c. */
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4:
         r = alu_src(t, alu, c, 0);
         break;
      case nir_op_iadd:
         r = LLVMBuildAdd(t->b, alu_src(t, alu, 0, c),
                                alu_src(t, alu, 1, c), "iadd");
         break;
      case nir_op_imul:
         r = LLVMBuildMul(t->b, alu_src(t, alu, 0, c),
                                alu_src(t, alu, 1, c), "imul");
         break;
      case nir_op_ishl: {
         LLVMValueRef v = alu_src(t, alu, 0, c);
         r = LLVMBuildShl(t->b, v,
                vp_int_cast(t, alu_src(t, alu, 1, c), LLVMTypeOf(v)), "ishl");
         break;
      }
      case nir_op_fadd: case nir_op_fmul: {
         LLVMTypeRef ft = fty(t, alu->def.bit_size);
         LLVMValueRef a = LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), ft, "");
         LLVMValueRef b = LLVMBuildBitCast(t->b, alu_src(t, alu, 1, c), ft, "");
         LLVMValueRef res = alu->op == nir_op_fadd
            ? LLVMBuildFAdd(t->b, a, b, "fadd")
            : LLVMBuildFMul(t->b, a, b, "fmul");
         r = LLVMBuildBitCast(t->b, res, ity(t, alu->def.bit_size), "");
         break;
      }
      case nir_op_isub:
         r = LLVMBuildSub(t->b, alu_src(t, alu, 0, c),
                                alu_src(t, alu, 1, c), "isub");
         break;
      case nir_op_iand:
         r = LLVMBuildAnd(t->b, alu_src(t, alu, 0, c),
                                alu_src(t, alu, 1, c), "iand");
         break;
      case nir_op_ior:
         r = LLVMBuildOr(t->b, alu_src(t, alu, 0, c),
                               alu_src(t, alu, 1, c), "ior");
         break;
      case nir_op_ixor:
         r = LLVMBuildXor(t->b, alu_src(t, alu, 0, c),
                                alu_src(t, alu, 1, c), "ixor");
         break;
      case nir_op_inot:
         r = LLVMBuildNot(t->b, alu_src(t, alu, 0, c), "inot");
         break;
      case nir_op_ishr: {
         LLVMValueRef v = alu_src(t, alu, 0, c);
         r = LLVMBuildAShr(t->b, v,
                vp_int_cast(t, alu_src(t, alu, 1, c), LLVMTypeOf(v)), "ishr");
         break;
      }
      case nir_op_ushr: {
         LLVMValueRef v = alu_src(t, alu, 0, c);
         r = LLVMBuildLShr(t->b, v,
                vp_int_cast(t, alu_src(t, alu, 1, c), LLVMTypeOf(v)), "ushr");
         break;
      }
      /* integer comparisons -> a NIR bool (i1, or sign-extended i32). */
      case nir_op_ieq: case nir_op_ine:
      case nir_op_ilt: case nir_op_ige:
      case nir_op_ult: case nir_op_uge: {
         LLVMIntPredicate p = alu->op == nir_op_ieq ? LLVMIntEQ
                            : alu->op == nir_op_ine ? LLVMIntNE
                            : alu->op == nir_op_ilt ? LLVMIntSLT
                            : alu->op == nir_op_ige ? LLVMIntSGE
                            : alu->op == nir_op_ult ? LLVMIntULT
                                                    : LLVMIntUGE;
         LLVMValueRef cmp = LLVMBuildICmp(t->b, p, alu_src(t, alu, 0, c),
                                          alu_src(t, alu, 1, c), "icmp");
         r = (alu->def.bit_size == 1) ? cmp
           : LLVMBuildSExt(t->b, cmp, t->i32, "");
         break;
      }
      /* float comparisons (operands are float bit patterns). */
      case nir_op_feq: case nir_op_fneu:
      case nir_op_flt: case nir_op_fge: {
         LLVMRealPredicate p = alu->op == nir_op_feq ? LLVMRealOEQ
                             : alu->op == nir_op_fneu ? LLVMRealUNE
                             : alu->op == nir_op_flt ? LLVMRealOLT
                                                     : LLVMRealOGE;
         LLVMTypeRef ft = fty(t, nir_src_bit_size(alu->src[0].src));
         LLVMValueRef a = LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), ft, "");
         LLVMValueRef b = LLVMBuildBitCast(t->b, alu_src(t, alu, 1, c), ft, "");
         LLVMValueRef cmp = LLVMBuildFCmp(t->b, p, a, b, "fcmp");
         r = (alu->def.bit_size == 1) ? cmp
           : LLVMBuildSExt(t->b, cmp, t->i32, "");
         break;
      }
      /* integer min / max / abs / neg */
      case nir_op_imax: case nir_op_imin:
      case nir_op_umax: case nir_op_umin: {
         LLVMValueRef a = alu_src(t, alu, 0, c), b = alu_src(t, alu, 1, c);
         LLVMIntPredicate p = alu->op == nir_op_imax ? LLVMIntSGT
                            : alu->op == nir_op_imin ? LLVMIntSLT
                            : alu->op == nir_op_umax ? LLVMIntUGT
                                                     : LLVMIntULT;
         r = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, p, a, b, ""),
                             a, b, "minmax");
         break;
      }
      case nir_op_iabs: {
         LLVMValueRef a = alu_src(t, alu, 0, c);
         LLVMValueRef neg = LLVMBuildNeg(t->b, a, "");
         r = LLVMBuildSelect(t->b,
            LLVMBuildICmp(t->b, LLVMIntSLT, a,
                          LLVMConstInt(t->i32, 0, false), ""),
            neg, a, "iabs");
         break;
      }
      case nir_op_isign: {
         LLVMValueRef v = alu_src(t, alu, 0, c);
         LLVMValueRef z = LLVMConstInt(LLVMTypeOf(v), 0, false);
         r = LLVMBuildSub(t->b,
            LLVMBuildZExt(t->b, LLVMBuildICmp(t->b, LLVMIntSGT, v, z, ""),
                          LLVMTypeOf(v), ""),
            LLVMBuildZExt(t->b, LLVMBuildICmp(t->b, LLVMIntSLT, v, z, ""),
                          LLVMTypeOf(v), ""), "isign");
         break;
      }
      case nir_op_ineg:
         r = LLVMBuildNeg(t->b, alu_src(t, alu, 0, c), "ineg");
         break;
      /* integer <-> float conversions (floats held as i32 bit patterns) */
      case nir_op_i2f32:
         r = LLVMBuildBitCast(t->b,
            LLVMBuildSIToFP(t->b, alu_src(t, alu, 0, c), t->f32, ""),
            t->i32, "");
         break;
      case nir_op_u2f32:
         r = LLVMBuildBitCast(t->b,
            LLVMBuildUIToFP(t->b, alu_src(t, alu, 0, c), t->f32, ""),
            t->i32, "");
         break;
      case nir_op_f2i32:
         r = LLVMBuildFPToSI(t->b,
            LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), t->f32, ""),
            t->i32, "");
         break;
      case nir_op_f2u32:
         r = LLVMBuildFPToUI(t->b,
            LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), t->f32, ""),
            t->i32, "");
         break;
      /* float neg / div / min / max / abs / fused-multiply-add / sign */
      case nir_op_fneg: case nir_op_fdiv:
      case nir_op_fmin: case nir_op_fmax:
      case nir_op_fabs: case nir_op_ffma: case nir_op_fsign:
      case nir_op_fsqrt: case nir_op_frsq: {
         LLVMTypeRef  ft = fty(t, alu->def.bit_size);
         LLVMValueRef fa = LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), ft, "");
         LLVMValueRef z  = LLVMConstReal(ft, 0.0);
         LLVMValueRef res;
         if (alu->op == nir_op_fsqrt) {
            res = emit_fsqrt(t, fa);
         } else if (alu->op == nir_op_frsq) {
            res = LLVMBuildFDiv(t->b, LLVMConstReal(ft, 1.0),
                                emit_fsqrt(t, fa), "frsq");
         } else if (alu->op == nir_op_fneg) {
            res = LLVMBuildFNeg(t->b, fa, "");
         } else if (alu->op == nir_op_fabs) {
            res = LLVMBuildSelect(t->b,
               LLVMBuildFCmp(t->b, LLVMRealOLT, fa, z, ""),
               LLVMBuildFNeg(t->b, fa, ""), fa, "");
         } else if (alu->op == nir_op_fsign) {
            res = LLVMBuildSelect(t->b,
               LLVMBuildFCmp(t->b, LLVMRealOGT, fa, z, ""),
               LLVMConstReal(ft, 1.0),
               LLVMBuildSelect(t->b,
                  LLVMBuildFCmp(t->b, LLVMRealOLT, fa, z, ""),
                  LLVMConstReal(ft, -1.0), z, ""), "");
         } else {
            LLVMValueRef fb = LLVMBuildBitCast(t->b, alu_src(t, alu, 1, c),
                                               ft, "");
            if (alu->op == nir_op_fdiv)
               res = LLVMBuildFDiv(t->b, fa, fb, "");
            else if (alu->op == nir_op_fmin)
               res = LLVMBuildSelect(t->b,
                  LLVMBuildFCmp(t->b, LLVMRealOLT, fa, fb, ""), fa, fb, "");
            else if (alu->op == nir_op_fmax)
               res = LLVMBuildSelect(t->b,
                  LLVMBuildFCmp(t->b, LLVMRealOGT, fa, fb, ""), fa, fb, "");
            else { /* ffma */
               LLVMValueRef fc = LLVMBuildBitCast(t->b, alu_src(t, alu, 2, c),
                                                  ft, "");
               res = LLVMBuildFAdd(t->b, LLVMBuildFMul(t->b, fa, fb, ""),
                                   fc, "");
            }
         }
         r = LLVMBuildBitCast(t->b, res, ity(t, alu->def.bit_size), "");
         break;
      }
      /* float width conversions — branch on the actual LLVM operand width so
       * the bitcasts stay legal even when the source value was materialized at
       * a different width than its NIR bit_size (a same-width convert is the
       * identity on the bit pattern). */
      case nir_op_f2f64: {
         LLVMValueRef v = alu_src(t, alu, 0, c);
         if (LLVMTypeOf(v) == t->i64) {
            r = v;   /* already an f64 bit pattern */
         } else {
            r = LLVMBuildBitCast(t->b, LLVMBuildFPExt(t->b,
               LLVMBuildBitCast(t->b, v, t->f32, ""), t->f64, ""),
               t->i64, "f2f64");
         }
         break;
      }
      case nir_op_f2f32: {
         LLVMValueRef v = alu_src(t, alu, 0, c);
         if (LLVMTypeOf(v) == t->i32) {
            r = v;   /* already an f32 bit pattern */
         } else {
            r = LLVMBuildBitCast(t->b, LLVMBuildFPTrunc(t->b,
               LLVMBuildBitCast(t->b, v, t->f64, ""), t->f32, ""),
               t->i32, "f2f32");
         }
         break;
      }
      /* integer modulo + 64-bit pack */
      case nir_op_idiv:
         r = LLVMBuildSDiv(t->b, alu_src(t, alu, 0, c),
                                 alu_src(t, alu, 1, c), "idiv");
         break;
      case nir_op_udiv:
         r = LLVMBuildUDiv(t->b, alu_src(t, alu, 0, c),
                                 alu_src(t, alu, 1, c), "udiv");
         break;
      case nir_op_imod:
         r = LLVMBuildSRem(t->b, alu_src(t, alu, 0, c),
                                 alu_src(t, alu, 1, c), "imod");
         break;
      case nir_op_umod:
         r = LLVMBuildURem(t->b, alu_src(t, alu, 0, c),
                                 alu_src(t, alu, 1, c), "umod");
         break;
      case nir_op_pack_64_2x32_split:
         r = LLVMBuildOr(t->b,
            LLVMBuildZExt(t->b, alu_src(t, alu, 0, c), t->i64, ""),
            LLVMBuildShl(t->b,
               LLVMBuildZExt(t->b, alu_src(t, alu, 1, c), t->i64, ""),
               LLVMConstInt(t->i64, 32, false), ""), "pack64");
         break;
      case nir_op_unpack_64_2x32_split_x:
         r = LLVMBuildTrunc(t->b, alu_src(t, alu, 0, c), t->i32, "lo32");
         break;
      case nir_op_unpack_64_2x32_split_y:
         r = LLVMBuildTrunc(t->b,
            LLVMBuildLShr(t->b, alu_src(t, alu, 0, c),
                          LLVMConstInt(t->i64, 32, false), ""), t->i32, "hi32");
         break;
      /* bool -> int / float */
      case nir_op_b2i32: case nir_op_b2f32: {
         LLVMValueRef v = alu_src(t, alu, 0, c);
         LLVMValueRef cond = (LLVMTypeOf(v) == LLVMInt1TypeInContext(t->ctx))
            ? v
            : LLVMBuildICmp(t->b, LLVMIntNE, v,
                            LLVMConstInt(LLVMTypeOf(v), 0, false), "");
         if (alu->op == nir_op_b2i32)
            r = LLVMBuildSelect(t->b, cond, LLVMConstInt(t->i32, 1, false),
                                LLVMConstInt(t->i32, 0, false), "");
         else
            r = LLVMBuildBitCast(t->b,
               LLVMBuildSelect(t->b, cond, LLVMConstReal(t->f32, 1.0),
                               LLVMConstReal(t->f32, 0.0), ""), t->i32, "");
         break;
      }
      /* width conversions */
      case nir_op_u2u64:
         r = LLVMBuildZExt(t->b, alu_src(t, alu, 0, c), t->i64, "u2u64");
         break;
      case nir_op_i2i64:
         r = LLVMBuildSExt(t->b, alu_src(t, alu, 0, c), t->i64, "i2i64");
         break;
      case nir_op_u2u32: case nir_op_i2i32:
         /* 64-bit source -> 32: truncate; 32-bit source: identity. */
         r = (LLVMTypeOf(alu_src(t, alu, 0, c)) == t->i64)
            ? LLVMBuildTrunc(t->b, alu_src(t, alu, 0, c), t->i32, "")
            : alu_src(t, alu, 0, c);
         break;
      /* high 32 bits of a 32x32 multiply */
      case nir_op_umul_high: case nir_op_imul_high: {
         LLVMTypeRef ext = t->i64;
         LLVMValueRef a = alu->op == nir_op_umul_high
            ? LLVMBuildZExt(t->b, alu_src(t, alu, 0, c), ext, "")
            : LLVMBuildSExt(t->b, alu_src(t, alu, 0, c), ext, "");
         LLVMValueRef b = alu->op == nir_op_umul_high
            ? LLVMBuildZExt(t->b, alu_src(t, alu, 1, c), ext, "")
            : LLVMBuildSExt(t->b, alu_src(t, alu, 1, c), ext, "");
         r = LLVMBuildTrunc(t->b,
            LLVMBuildLShr(t->b, LLVMBuildMul(t->b, a, b, ""),
                          LLVMConstInt(ext, 32, false), ""), t->i32, "mulhi");
         break;
      }
      /* float reciprocal */
      case nir_op_frcp:
         r = LLVMBuildBitCast(t->b,
            LLVMBuildFDiv(t->b, LLVMConstReal(t->f32, 1.0),
               LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), t->f32, ""), ""),
            t->i32, "frcp");
         break;
      /* ufind_msb: index of the highest set bit, -1 when the source is 0.
       * ctlz(0)=32, so 31-ctlz naturally yields -1 for a zero source. */
      case nir_op_ufind_msb:
         r = LLVMBuildSub(t->b, LLVMConstInt(t->i32, 31, false),
                          emit_ctlz(t, alu_src(t, alu, 0, c)), "ufind_msb");
         break;
      case nir_op_bit_count:
         r = emit_ctpop(t, alu_src(t, alu, 0, c));
         break;
      case nir_op_fpow: {
         LLVMTypeRef  ft = fty(t, alu->def.bit_size);
         LLVMValueRef x  = LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), ft, "");
         LLVMValueRef y  = LLVMBuildBitCast(t->b, alu_src(t, alu, 1, c), ft, "");
         r = LLVMBuildBitCast(t->b, emit_fpow(t, x, y),
                              ity(t, alu->def.bit_size), "");
         break;
      }
      /* b2b1: normalize a bool to a 1-bit bool (nonzero -> true). */
      case nir_op_b2b1: {
         LLVMValueRef v = alu_src(t, alu, 0, c);
         r = (LLVMTypeOf(v) == LLVMInt1TypeInContext(t->ctx))
            ? v
            : LLVMBuildICmp(t->b, LLVMIntNE, v,
                            LLVMConstInt(LLVMTypeOf(v), 0, false), "b2b1");
         break;
      }
      /* bcsel: select(cond, a, b) -- cond is a NIR bool. */
      case nir_op_bcsel: {
         LLVMValueRef cond = alu_src(t, alu, 0, c);
         if (LLVMTypeOf(cond) != LLVMInt1TypeInContext(t->ctx))
            cond = LLVMBuildICmp(t->b, LLVMIntNE, cond,
                                 LLVMConstInt(LLVMTypeOf(cond), 0, false), "");
         r = LLVMBuildSelect(t->b, cond, alu_src(t, alu, 1, c),
                             alu_src(t, alu, 2, c), "bcsel");
         break;
      }
      default:
         mesa_logw("vortexpipe: vp_nir_to_llvm: unhandled nir_op '%s'",
                   nir_op_infos[alu->op].name);
         t->ok = false;
         return;
      }
      ssa_set(t, alu->def.index, c, r);
   }
}

/* A NIR deref resolves to an i32 byte address, stored in component 0
 * of its SSA def. function_temp vars are LLVM allocas; shader_out
 * vars address into the per-vertex output record. */
static void
emit_deref(struct vp_tr *t, nir_deref_instr *d)
{
   LLVMValueRef addr = NULL;

   switch (d->deref_type) {
   case nir_deref_type_var: {
      nir_variable *v = d->var;
      if (v->data.mode == nir_var_function_temp) {
         struct vp_var *e = vp_var_find(t, v);
         if (!e) {
            if (t->nvars >= VP_MAXV) { t->ok = false; return; }
            /* allocas must sit in the entry block so every deref --
             * including ones inside loops/branches -- is dominated. */
            LLVMBasicBlockRef cur = LLVMGetInsertBlock(t->b);
            LLVMValueRef first = LLVMGetFirstInstruction(t->entry);
            if (first)
               LLVMPositionBuilderBefore(t->b, first);
            else
               LLVMPositionBuilderAtEnd(t->b, t->entry);
            LLVMValueRef a = LLVMBuildAlloca(t->b,
               glsl_to_llvm(t, v->type), v->name ? v->name : "tmp");
            LLVMPositionBuilderAtEnd(t->b, cur);
            e = &t->vars[t->nvars++];
            e->var = v; e->alloca = a; e->out_off = -1;
         }
         addr = LLVMBuildPtrToInt(t->b, e->alloca, t->iptr, "");
      } else if (v->data.mode == nir_var_shader_out) {
         struct vp_var *e = vp_var_find(t, v);
         if (!e || e->out_off < 0) { t->ok = false; return; }
         LLVMValueRef off = vp_iptr_const(t, (unsigned)e->out_off);
         if (t->is_vs) {
            /* vertex shader: out_base + vraw * stride + slot_offset. The output
             * slot is the sequential global id (vraw), not the index-resolved
             * vid, so an indexed draw writes records[0..count) in draw order for
             * in-order triangle assembly. */
            LLVMValueRef voff = LLVMBuildMul(t->b, vp_to_iptr(t, t->vraw),
               vp_iptr_const(t, t->out_stride), "");
            addr = LLVMBuildAdd(t->b, t->out_base, voff, "");
            addr = LLVMBuildAdd(t->b, addr, off, "vsout");
         } else {
            /* fragment shader: fs_out_base + slot_offset */
            addr = LLVMBuildAdd(t->b, t->fs_out_base, off, "fsout");
         }
      } else if (v->data.mode == nir_var_shader_in && t->is_fs) {
         /* fragment shader input: an interpolated varying. */
         struct vp_var *e = vp_var_find(t, v);
         if (!e || e->out_off < 0) { t->ok = false; return; }
         addr = LLVMBuildAdd(t->b, t->fs_in_base,
            vp_iptr_const(t, (unsigned)e->out_off), "fsin");
      } else if (v->data.mode == nir_var_shader_in && t->is_vs) {
         /* vertex shader input: a per-vertex attribute fetched from
          * the bound vertex buffer (deref-based NIR path). */
         addr = emit_vs_attr_addr(t, v->data.driver_location);
      } else if (v->data.mode == nir_var_uniform) {
         /* a combined image-sampler handle. gfx-v1 binds a single
          * texture to TEX stage 0 through DCRs, so the deref carries
          * no address -- emit_tex ignores the texture/sampler deref
          * sources. A placeholder keeps the SSA chain well-formed. */
         addr = vp_iptr_const(t, 0);
      } else {
         mesa_logw("vortexpipe: vp_nir_to_llvm: deref of unsupported "
                   "var mode %d", (int)v->data.mode);
         t->ok = false;
         return;
      }
      break;
   }
   case nir_deref_type_array: {
      LLVMValueRef base = ssa_get(t, d->parent.ssa->index, 0);
      LLVMValueRef idx  = ssa_get(t, d->arr.index.ssa->index, 0);
      if (!base || !idx) { t->ok = false; return; }
      LLVMValueRef off = LLVMBuildMul(t->b, vp_to_iptr(t, idx),
         vp_iptr_const(t, glsl_bytes(d->type)), "");
      addr = LLVMBuildAdd(t->b, base, off, "elem");
      break;
   }
   default:
      mesa_logw("vortexpipe: vp_nir_to_llvm: unhandled deref type %d",
                (int)d->deref_type);
      t->ok = false;
      return;
   }

   ssa_set(t, d->def.index, 0, addr);
}

/* RTU emit helpers (defined after emit_vx_tex, below). */
static LLVMValueRef emit_vx_rt_get(struct vp_tr *t, unsigned slot, LLVMValueRef status);
static LLVMValueRef emit_vx_rt_wtrace(struct vp_tr *t, LLVMValueRef scene,
                                      LLVMValueRef flags_cull,
                                      LLVMValueRef ray[8]);
static LLVMValueRef emit_vx_rt_wait(struct vp_tr *t, LLVMValueRef handle);
static void         emit_vx_rt_cb_ret(struct vp_tr *t, LLVMValueRef action);

/* Emit an atomicrmw / cmpxchg at device pointer p, setting the intrinsic's def
 * to the old value. Data sources start at src index `di` (rmw: [di]=value;
 * swap: [di]=compare, [di+1]=new). 32-bit integer only — the lane ABI is 32-bit
 * and the device A-extension AMOs are .w; float/64-bit atomics fail-compile.
 * Shared by the ssbo / global / shared atomic intrinsics, which differ only in
 * how the address p is formed. */
static void
emit_atomic_at(struct vp_tr *t, nir_intrinsic_instr *in, LLVMValueRef p,
               unsigned di, bool is_swap)
{
   if (in->def.bit_size != 32) {
      mesa_logw("vortexpipe: %u-bit atomic unsupported (32-bit only)",
                in->def.bit_size);
      t->ok = false;
      return;
   }
   nir_atomic_op aop = nir_intrinsic_atomic_op(in);
   if (is_swap) {
      if (aop != nir_atomic_op_cmpxchg) {
         mesa_logw("vortexpipe: atomic_swap op %d unsupported", aop);
         t->ok = false;
         return;
      }
      LLVMValueRef r = LLVMBuildAtomicCmpXchg(t->b, p,
         intr_src(t, in, di), intr_src(t, in, di + 1),
         LLVMAtomicOrderingSequentiallyConsistent,
         LLVMAtomicOrderingSequentiallyConsistent, /*singleThread*/ false);
      ssa_set(t, in->def.index, 0, LLVMBuildExtractValue(t->b, r, 0, "amocas"));
      return;
   }
   LLVMAtomicRMWBinOp op;
   bool supported = true;
   switch (aop) {
   case nir_atomic_op_iadd: op = LLVMAtomicRMWBinOpAdd;  break;
   case nir_atomic_op_xchg: op = LLVMAtomicRMWBinOpXchg; break;
   case nir_atomic_op_iand: op = LLVMAtomicRMWBinOpAnd;  break;
   case nir_atomic_op_ior:  op = LLVMAtomicRMWBinOpOr;   break;
   case nir_atomic_op_ixor: op = LLVMAtomicRMWBinOpXor;  break;
   case nir_atomic_op_imin: op = LLVMAtomicRMWBinOpMin;  break;
   case nir_atomic_op_imax: op = LLVMAtomicRMWBinOpMax;  break;
   case nir_atomic_op_umin: op = LLVMAtomicRMWBinOpUMin; break;
   case nir_atomic_op_umax: op = LLVMAtomicRMWBinOpUMax; break;
   default:                 op = LLVMAtomicRMWBinOpAdd;
                            supported = false;           break;
   }
   if (!supported) {
      mesa_logw("vortexpipe: atomic op %d unsupported (int32 only)", aop);
      t->ok = false;
      return;
   }
   LLVMValueRef r = LLVMBuildAtomicRMW(t->b, op, p, intr_src(t, in, di),
      LLVMAtomicOrderingSequentiallyConsistent, /*singleThread*/ false);
   ssa_set(t, in->def.index, 0, r);
}

/* Lazily allocate the per-thread scratch region (nir->scratch_size bytes) as an
 * entry-block alloca; return its base as an iptr byte address. */
static LLVMValueRef
emit_scratch_base(struct vp_tr *t)
{
   if (!t->scratch_base) {
      unsigned sz = t->scratch_size ? t->scratch_size : 4u;
      LLVMBasicBlockRef cur = LLVMGetInsertBlock(t->b);
      LLVMValueRef first = LLVMGetFirstInstruction(t->entry);
      if (first)
         LLVMPositionBuilderBefore(t->b, first);
      else
         LLVMPositionBuilderAtEnd(t->b, t->entry);
      LLVMValueRef a = LLVMBuildArrayAlloca(t->b, t->i8,
         LLVMConstInt(t->i32, sz, false), "scratch");
      LLVMPositionBuilderAtEnd(t->b, cur);
      t->scratch_base = LLVMBuildPtrToInt(t->b, a, t->iptr, "");
   }
   return t->scratch_base;
}

/* lp_jit_image field byte offsets inside a struct lp_descriptor slot. The image
 * union member starts at offset 0, so base aliases an SSBO's data pointer and is
 * read the same way (*(i64*)desc). vp_launch rewrites base host->device. */
#define VP_JIT_IMG_BASE        0
#define VP_JIT_IMG_ROW_STRIDE  24
#define VP_JIT_IMG_BASE_OFFSET 40

/* Load a 32-bit lp_jit_image field at `byte_off` from the descriptor address. */
static LLVMValueRef
img_desc_u32(struct vp_tr *t, LLVMValueRef desc, unsigned byte_off)
{
   LLVMValueRef a = LLVMBuildAdd(t->b, desc,
      LLVMConstInt(LLVMTypeOf(desc), byte_off, false), "");
   LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
   return LLVMBuildLoad2(t->b, t->i32, p, "imgfld");
}

/* f32 in [0,1] -> unorm8 (round-to-nearest), value given as its i32 bit pattern. */
static LLVMValueRef
f32bits_to_unorm8(struct vp_tr *t, LLVMValueRef vi)
{
   LLVMValueRef f   = LLVMBuildBitCast(t->b, vi, t->f32, "");
   LLVMValueRef z   = LLVMConstReal(t->f32, 0.0);
   LLVMValueRef one = LLVMConstReal(t->f32, 1.0);
   f = LLVMBuildSelect(t->b, LLVMBuildFCmp(t->b, LLVMRealOGT, f, z, ""), f, z, "");
   f = LLVMBuildSelect(t->b, LLVMBuildFCmp(t->b, LLVMRealOLT, f, one, ""), f, one, "");
   LLVMValueRef s = LLVMBuildFAdd(t->b,
      LLVMBuildFMul(t->b, f, LLVMConstReal(t->f32, 255.0), ""),
      LLVMConstReal(t->f32, 0.5), "");
   LLVMValueRef b = LLVMBuildFPToUI(t->b, s, t->i32, "");
   return LLVMBuildAnd(t->b, b, LLVMConstInt(t->i32, 0xff, false), "");
}

/* unorm8 (low byte of `word`) -> f32 in [0,1], returned as its i32 bit pattern. */
static LLVMValueRef
unorm8_to_f32bits(struct vp_tr *t, LLVMValueRef word)
{
   LLVMValueRef byte = LLVMBuildAnd(t->b, word, LLVMConstInt(t->i32, 0xff, false), "");
   LLVMValueRef f = LLVMBuildFMul(t->b, LLVMBuildUIToFP(t->b, byte, t->f32, ""),
                                 LLVMConstReal(t->f32, 1.0 / 255.0), "");
   return LLVMBuildBitCast(t->b, f, t->i32, "");
}

static void
emit_intrinsic(struct vp_tr *t, nir_intrinsic_instr *in)
{
   switch (in->intrinsic) {
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_num_workgroups: {
      unsigned base =
         in->intrinsic == nir_intrinsic_load_workgroup_id ? VX_CSR_CTA_BLOCK_ID_X
       : in->intrinsic == nir_intrinsic_load_num_workgroups ? VX_CSR_CTA_GRID_DIM_X
                                                          : VX_CSR_CTA_THREAD_ID_X;
      for (unsigned c = 0; c < in->def.num_components; c++)
         ssa_set(t, in->def.index, c, emit_csr_read(t, base + c, "id"));
      break;
   }
   /* gl_VertexIndex: one Vortex thread per vertex, so it is the CTA
    * thread id (vp_draw_vbo launches a single block of `count`). */
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_vertex_id_zero_base:
      ssa_set(t, in->def.index, 0, t->vid);
      break;
   /* Instancing. gl_InstanceIndex lowers (nir_lower_system_values) to
    * load_instance_id + load_base_instance; the VS prologue resolves the
    * 0-based instance id and the base-instance from arg slots 3/4. On the
    * non-instanced fast path instance == 0 and first_instance == 0. */
   case nir_intrinsic_load_instance_id:
      ssa_set(t, in->def.index, 0,
              t->instance ? t->instance : LLVMConstInt(t->i32, 0, false));
      break;
   case nir_intrinsic_load_base_instance:
      ssa_set(t, in->def.index, 0,
              t->first_instance ? t->first_instance : LLVMConstInt(t->i32, 0, false));
      break;
   /* vertex-attribute fetch (lowered-IO NIR path): read each component
    * from the bound vertex buffer at attr base + component offset. */
   case nir_intrinsic_load_input: {
      unsigned loc  = nir_intrinsic_base(in);
      unsigned comp = nir_intrinsic_component(in);
      LLVMValueRef attr = emit_vs_attr_addr(t, loc);
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, attr,
            vp_iptr_const(t, (comp + c) * 4u), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, t->i32, p, "in"));
      }
      break;
   }
   case nir_intrinsic_load_const_buf_base_addr_lvp: {
      /* Base device address of constant buffer `index` = arg[index]. For the
       * FS arg is the resident descriptor table: index 1 is the set-0
       * descriptor blob, which feeds the UBO/SSBO descriptor dereference; a
       * combined-image-sampler's value only feeds a vx_tex handle source
       * (emit_tex ignores it). A shader with no table (compute VS placeholder)
       * keeps it well-formed with 0. */
      if (!t->arg) {
         ssa_set(t, in->def.index, 0, LLVMConstInt(t->i64, 0, false));
         break;
      }
      LLVMValueRef idx = intr_src(t, in, 0);
      LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, t->arg, &idx, 1, "");
      LLVMValueRef v   = LLVMBuildLoad2(t->b, t->i64, gep, "bufbase");
      ssa_set(t, in->def.index, 0, v);
      break;
   }
   /* buffer load: base (i64) + offset; one load per component, the
    * width taken from the def's bit size (32- or 64-bit).
    *
    * load_ssbo and load_ubo differ in operand 0:
    *  - load_ubo's is a constant-buffer INDEX; the descriptor lives at
    *    arg[index] + offset and IS the value the shader wants (lavapipe
    *    lowers an acceleration-structure read to load_ubo(cbuf,0), and
    *    the descriptor slot holds the AS device address verbatim).
    *  - load_ssbo's is the device ADDRESS of the buffer's descriptor
    *    (a struct lp_descriptor / lp_jit_buffer); the buffer's data
    *    pointer is the first 8 bytes of that descriptor, so the access
    *    dereferences one level: data = *(i64*)desc + offset.
    * vp_launch builds the device descriptor buffer with these pointer
    * fields rewritten from host to Vortex device addresses. */
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ubo: {
      LLVMValueRef base;
      if (in->intrinsic == nir_intrinsic_load_ubo &&
          nir_src_is_const(in->src[0])) {
         /* Constant-buffer index form: base = arg[index] read directly. Push
          * constants (load_ubo(0, off)) take this path (src[0] is the constant
          * 0); the compute acceleration-structure read load_ubo(1, 0) does too
          * (src[0] is the constant cbuf index). A UBO reached through a computed
          * const_buf_base+binding address (src[0] non-constant, both FS and
          * compute) is a descriptor dereference — the else branch below. */
         if (t->arg) {
            LLVMValueRef idx = intr_src(t, in, 0);
            LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, t->arg, &idx, 1, "");
            base = LLVMBuildLoad2(t->b, t->i64, gep, "ubobase");
         } else {
            base = LLVMConstInt(t->i64, 0, false);
         }
      } else if (in->intrinsic == nir_intrinsic_load_ssbo &&
                 nir_src_is_const(in->src[0])) {
         /* Raw shader-buffer slot: a constant src[0] is a set_shader_buffers
          * binding index (not a descriptor address), so its data base is
          * arg[VP_ARG_SSBO_BASE + slot] directly — no descriptor dereference.
          * lavapipe binds the RT trace-ray command buffer this way and reads it
          * as load_ssbo(imm 0, off); vp_launch relocates it into that arg slot. */
         unsigned slot = (unsigned)nir_src_as_uint(in->src[0]);
         if (t->arg && slot < VP_MAX_SSBO) {
            LLVMValueRef idx = LLVMConstInt(t->i64, VP_ARG_SSBO_BASE + slot, false);
            LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, t->arg, &idx, 1, "");
            base = LLVMBuildLoad2(t->b, t->i64, gep, "ssbobase");
         } else {
            base = LLVMConstInt(t->i64, 0, false);
         }
      } else {
         /* Descriptor-address form: src[0] is a descriptor's device address —
          * load_ssbo, and a UBO (FS or compute) whose NIR is
          * load_ubo(load_const_buf_base_addr_lvp(set)+binding, off). The buffer's
          * data base is the descriptor's lp_jit_buffer.ptr (its first 8 bytes),
          * which vp_launch relocates host->device before upload. */
         LLVMValueRef desc = intr_src(t, in, 0);              /* descriptor addr */
         LLVMValueRef dp   = LLVMBuildIntToPtr(t->b, desc, t->ptr, "");
         base = LLVMBuildLoad2(t->b, t->i64, dp, "ssbobase"); /* lp_jit_buffer.ptr */
      }
      LLVMValueRef off  = LLVMBuildZExt(t->b, intr_src(t, in, 1),
                                        t->i64, "");
      LLVMValueRef addr = LLVMBuildAdd(t->b, base, off, "");
      LLVMTypeRef  lt   = ity(t, in->def.bit_size);
      unsigned     esz  = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i64, c * esz, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, lt, p, "buf"));
      }
      break;
   }
   case nir_intrinsic_store_ssbo: {
      /* operand 1 is the buffer descriptor's device address; the data
       * pointer is its first 8 bytes (see load_ssbo above). A constant
       * operand 1 is a raw shader-buffer slot: its data base is
       * arg[VP_ARG_SSBO_BASE + slot] directly (see load_ssbo). */
      LLVMValueRef base;
      if (nir_src_is_const(in->src[1])) {
         unsigned slot = (unsigned)nir_src_as_uint(in->src[1]);
         if (t->arg && slot < VP_MAX_SSBO) {
            LLVMValueRef idx = LLVMConstInt(t->i64, VP_ARG_SSBO_BASE + slot, false);
            LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, t->arg, &idx, 1, "");
            base = LLVMBuildLoad2(t->b, t->i64, gep, "ssbobase");
         } else {
            base = LLVMConstInt(t->i64, 0, false);
         }
      } else {
         LLVMValueRef desc = intr_src(t, in, 1);
         LLVMValueRef dp   = LLVMBuildIntToPtr(t->b, desc, t->ptr, "");
         base = LLVMBuildLoad2(t->b, t->i64, dp, "ssbobase");
      }
      LLVMValueRef off  = LLVMBuildZExt(t->b, intr_src(t, in, 2),
                                        t->i64, "");
      LLVMValueRef addr = emit_store_addr(t, LLVMBuildAdd(t->b, base, off, ""));
      unsigned mask = nir_intrinsic_write_mask(in);
      unsigned nc   = nir_src_num_components(in->src[0]);
      unsigned esz  = nir_src_bit_size(in->src[0]) / 8u;
      for (unsigned c = 0; c < nc; c++) {
         if (!(mask & (1u << c))) continue;
         LLVMValueRef v = ssa_get(t, in->src[0].ssa->index, c);
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i64, c * esz, false), "");
         if (v)
            LLVMBuildStore(t->b, v, LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
      }
      break;
   }
   /* SSBO atomics: read-modify-write at the SSBO byte address, lowered onto the
    * device's RISC-V A-extension. The address is *(desc)+off exactly as
    * store_ssbo (operand layout: src[0]=descriptor addr, src[1]=byte offset,
    * src[2]=data; the _swap form adds src[2]=compare, src[3]=new value). The
    * device dcache is the atomic ordering point.
    *
    *   ssbo_atomic      -> atomicrmw  (add/xchg/and/or/xor/{i,u}min/{i,u}max)
    *                       => amoadd.w / amoswap.w / amoand.w / amoor.w /
    *                          amoxor.w / amomin[u].w / amomax[u].w
    *   ssbo_atomic_swap -> cmpxchg    => LR.W/SC.W retry loop
    *
    * 32-bit integer only — the lane ABI is 32-bit and the device AMOs are .w.
    * Float atomics (fadd/fmin/fmax/fcmpxchg) and 64-bit are rejected
    * (fail-compile) since there is no device AMO for them. */
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap: {
      if (in->def.bit_size != 32) {
         mesa_logw("vortexpipe: %u-bit SSBO atomic unsupported (32-bit only)",
                   in->def.bit_size);
         t->ok = false;
         break;
      }
      nir_atomic_op aop = nir_intrinsic_atomic_op(in);
      LLVMValueRef desc = intr_src(t, in, 0);
      LLVMValueRef dp   = LLVMBuildIntToPtr(t->b, desc, t->ptr, "");
      LLVMValueRef base = LLVMBuildLoad2(t->b, t->i64, dp, "ssbobase");
      LLVMValueRef off  = LLVMBuildZExt(t->b, intr_src(t, in, 1), t->i64, "");
      LLVMValueRef addr = LLVMBuildAdd(t->b, base, off, "");
      LLVMValueRef p    = LLVMBuildIntToPtr(t->b, addr, t->ptr, "");
      if (in->intrinsic == nir_intrinsic_ssbo_atomic_swap) {
         /* compare-and-swap: src[2]=compare, src[3]=new. cmpxchg returns
          * {oldval, i1 success}; the intrinsic wants the old value. */
         if (aop != nir_atomic_op_cmpxchg) {
            mesa_logw("vortexpipe: SSBO atomic_swap op %d unsupported", aop);
            t->ok = false;
            break;
         }
         LLVMValueRef cmp = intr_src(t, in, 2);
         LLVMValueRef nv  = intr_src(t, in, 3);
         LLVMValueRef r = LLVMBuildAtomicCmpXchg(t->b, p, cmp, nv,
            LLVMAtomicOrderingSequentiallyConsistent,
            LLVMAtomicOrderingSequentiallyConsistent, /*singleThread*/ false);
         ssa_set(t, in->def.index, 0,
                 LLVMBuildExtractValue(t->b, r, 0, "amocas"));
         break;
      }
      LLVMAtomicRMWBinOp op;
      bool supported = true;
      switch (aop) {
      case nir_atomic_op_iadd: op = LLVMAtomicRMWBinOpAdd;  break;
      case nir_atomic_op_xchg: op = LLVMAtomicRMWBinOpXchg; break;
      case nir_atomic_op_iand: op = LLVMAtomicRMWBinOpAnd;  break;
      case nir_atomic_op_ior:  op = LLVMAtomicRMWBinOpOr;   break;
      case nir_atomic_op_ixor: op = LLVMAtomicRMWBinOpXor;  break;
      case nir_atomic_op_imin: op = LLVMAtomicRMWBinOpMin;  break;
      case nir_atomic_op_imax: op = LLVMAtomicRMWBinOpMax;  break;
      case nir_atomic_op_umin: op = LLVMAtomicRMWBinOpUMin; break;
      case nir_atomic_op_umax: op = LLVMAtomicRMWBinOpUMax; break;
      default:                 op = LLVMAtomicRMWBinOpAdd;
                               supported = false;           break;
      }
      if (!supported) {
         /* float atomics (fadd/fmin/fmax), inc/dec_wrap: no device AMO */
         mesa_logw("vortexpipe: SSBO atomic op %d unsupported (int32 only)",
                   aop);
         t->ok = false;
         break;
      }
      LLVMValueRef val = intr_src(t, in, 2);
      LLVMValueRef r = LLVMBuildAtomicRMW(t->b, op, p, val,
         LLVMAtomicOrderingSequentiallyConsistent, /*singleThread*/ false);
      ssa_set(t, in->def.index, 0, r);
      break;
   }
   /* Raw global-address atomics: src[0] is the device address; data follows
    * at src[1] (swap: src[1]=compare, src[2]=new). */
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap: {
      LLVMValueRef p = LLVMBuildIntToPtr(t->b, intr_src(t, in, 0), t->ptr, "");
      emit_atomic_at(t, in, p, 1,
                     in->intrinsic == nir_intrinsic_global_atomic_swap);
      break;
   }
   /* Shared-memory atomics: addr = CTA local-mem base + nir_base + src[0];
    * data follows at src[1] (swap: src[1]=compare, src[2]=new). */
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap: {
      if (!t->lmem_base) { t->ok = false; break; }
      LLVMValueRef addr = LLVMBuildAdd(t->b,
         LLVMBuildAdd(t->b, t->lmem_base,
            vp_iptr_const(t, nir_intrinsic_base(in)), ""),
         vp_to_iptr(t, intr_src(t, in, 0)), "shatomaddr");
      LLVMValueRef p = LLVMBuildIntToPtr(t->b, addr, t->ptr, "");
      emit_atomic_at(t, in, p, 1,
                     in->intrinsic == nir_intrinsic_shared_atomic_swap);
      break;
   }
   /* Push constants: lavapipe usually lowers these to load_ubo(0, off), but
    * a load_push_constant survives on some paths. Read table[0] (the push-
    * constant constant-buffer base) + off directly, mirroring load_ubo(0). */
   case nir_intrinsic_load_push_constant: {
      LLVMValueRef base;
      if (t->arg) {
         LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
         LLVMValueRef gep  = LLVMBuildGEP2(t->b, t->i64, t->arg, &zero, 1, "");
         base = LLVMBuildLoad2(t->b, t->i64, gep, "pushbase");
      } else {
         base = LLVMConstInt(t->i64, 0, false);
      }
      LLVMValueRef off  = LLVMBuildZExt(t->b, intr_src(t, in, 0), t->i64, "");
      LLVMValueRef addr = LLVMBuildAdd(t->b, base, off, "");
      LLVMTypeRef  lt   = ity(t, in->def.bit_size);
      unsigned     esz  = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i64, c * esz, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, lt, p, "push"));
      }
      break;
   }
   /* raw global memory: the operand IS the device address. */
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant: {
      LLVMValueRef addr = intr_src(t, in, 0);
      LLVMTypeRef  lt   = ity(t, in->def.bit_size);
      unsigned     esz  = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(LLVMTypeOf(addr), c * esz, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, lt, p, "gld"));
      }
      break;
   }
   case nir_intrinsic_store_global: {
      LLVMValueRef addr = emit_store_addr(t, intr_src(t, in, 1));
      unsigned mask = nir_intrinsic_write_mask(in);
      unsigned nc   = nir_src_num_components(in->src[0]);
      unsigned esz  = nir_src_bit_size(in->src[0]) / 8u;
      for (unsigned c = 0; c < nc; c++) {
         if (!(mask & (1u << c))) continue;
         LLVMValueRef v = ssa_get(t, in->src[0].ssa->index, c);
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(LLVMTypeOf(addr), c * esz, false), "");
         if (v)
            LLVMBuildStore(t->b, v, LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
      }
      break;
   }
   /* shared memory: addr = CTA local-mem base + nir_base + dynamic off. */
   case nir_intrinsic_load_shared: {
      if (!t->lmem_base) { t->ok = false; break; }
      LLVMValueRef addr = LLVMBuildAdd(t->b,
         LLVMBuildAdd(t->b, t->lmem_base,
            vp_iptr_const(t, nir_intrinsic_base(in)), ""),
         vp_to_iptr(t, intr_src(t, in, 0)), "shaddr");
      LLVMTypeRef lt  = ity(t, in->def.bit_size);
      unsigned    esz = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            vp_iptr_const(t, c * esz), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, lt, p, "sld"));
      }
      break;
   }
   case nir_intrinsic_store_shared: {
      if (!t->lmem_base) { t->ok = false; break; }
      LLVMValueRef addr = LLVMBuildAdd(t->b,
         LLVMBuildAdd(t->b, t->lmem_base,
            vp_iptr_const(t, nir_intrinsic_base(in)), ""),
         vp_to_iptr(t, intr_src(t, in, 1)), "shaddr");
      unsigned mask = nir_intrinsic_write_mask(in);
      unsigned nc   = nir_src_num_components(in->src[0]);
      unsigned esz  = nir_src_bit_size(in->src[0]) / 8u;
      for (unsigned c = 0; c < nc; c++) {
         if (!(mask & (1u << c))) continue;
         LLVMValueRef v = ssa_get(t, in->src[0].ssa->index, c);
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            vp_iptr_const(t, c * esz), "");
         if (v)
            LLVMBuildStore(t->b, v, LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
      }
      break;
   }
   /* Per-thread scratch: addr = scratch base + dynamic offset. */
   case nir_intrinsic_load_scratch: {
      LLVMValueRef addr = LLVMBuildAdd(t->b, emit_scratch_base(t),
         vp_to_iptr(t, intr_src(t, in, 0)), "scaddr");
      LLVMTypeRef lt  = ity(t, in->def.bit_size);
      unsigned    esz = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr, vp_iptr_const(t, c * esz), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, lt, p, "scld"));
      }
      break;
   }
   case nir_intrinsic_store_scratch: {
      LLVMValueRef addr = LLVMBuildAdd(t->b, emit_scratch_base(t),
         vp_to_iptr(t, intr_src(t, in, 1)), "scaddr");
      unsigned mask = nir_intrinsic_write_mask(in);
      unsigned nc   = nir_src_num_components(in->src[0]);
      unsigned esz  = nir_src_bit_size(in->src[0]) / 8u;
      for (unsigned c = 0; c < nc; c++) {
         if (!(mask & (1u << c))) continue;
         LLVMValueRef v = ssa_get(t, in->src[0].ssa->index, c);
         LLVMValueRef a = LLVMBuildAdd(t->b, addr, vp_iptr_const(t, c * esz), "");
         if (v)
            LLVMBuildStore(t->b, v, LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
      }
      break;
   }
   /* A Vortex warp is a subgroup: the warp ballot reduces each lane's
    * predicate to a lane bitmask. The NIR predicate is a bool; widen to the
    * i32 the instruction consumes. */
   case nir_intrinsic_ballot: {
      LLVMValueRef pred = intr_src(t, in, 0);
      if (LLVMTypeOf(pred) == LLVMInt1TypeInContext(t->ctx))
         pred = LLVMBuildZExt(t->b, pred, t->i32, "");
      ssa_set(t, in->def.index, 0, emit_ballot(t, pred));
      break;
   }
   /* subgroup == warp: the subgroup index within the workgroup is the warp id
    * (one CTA occupies one core's warps). */
   case nir_intrinsic_load_subgroup_id:
      ssa_set(t, in->def.index, 0,
              emit_csr_read(t, VX_CSR_WARP_ID, "subgroup_id"));
      break;
   /* lane index within the subgroup (warp) = Vortex thread id. */
   case nir_intrinsic_load_subgroup_invocation:
      ssa_set(t, in->def.index, 0,
              emit_csr_read(t, VX_CSR_THREAD_ID, "subgroup_inv"));
      break;
   /* read_invocation: broadcast src[0] from the lane src[1] to all lanes. */
   case nir_intrinsic_read_invocation:
      ssa_set(t, in->def.index, 0,
              emit_shfl_idx(t, intr_src(t, in, 0), intr_src(t, in, 1)));
      break;
   /* read_first_invocation: broadcast from the lowest active lane (lowest set
    * bit of the active-lane ballot). */
   case nir_intrinsic_read_first_invocation: {
      LLVMValueRef low = emit_cttz(t, emit_ballot(t, LLVMConstInt(t->i32, 1, false)));
      ssa_set(t, in->def.index, 0, emit_shfl_idx(t, intr_src(t, in, 0), low));
      break;
   }
   /* elect: exactly the lowest active lane returns true. The active-lane
    * ballot's lowest set bit is that lane; compare against this lane's id. */
   case nir_intrinsic_elect: {
      LLVMValueRef mask = emit_ballot(t, LLVMConstInt(t->i32, 1, false));
      LLVMValueRef low  = emit_cttz(t, mask);
      LLVMValueRef lane = emit_csr_read(t, VX_CSR_THREAD_ID, "lane");
      ssa_set(t, in->def.index, 0,
              LLVMBuildICmp(t->b, LLVMIntEQ, lane, low, "elect"));
      break;
   }
   /* Inclusive prefix scan across the warp (subgroup): Hillis-Steele over
    * shfl_up with a lane-id guard so a lane below the step distance keeps its
    * own accumulator (shfl_up returns own value out of range — accumulating it
    * would double-count). Unrolled to cover NT up to 32; steps past NT no-op. */
   case nir_intrinsic_inclusive_scan: {
      if (in->def.bit_size != 32) {
         mesa_logw("vortexpipe: %u-bit inclusive_scan unsupported (32-bit only)",
                   in->def.bit_size);
         t->ok = false;
         break;
      }
      nir_op rop = nir_intrinsic_reduction_op(in);
      LLVMValueRef acc  = intr_src(t, in, 0);
      LLVMValueRef lane = emit_csr_read(t, VX_CSR_THREAD_ID, "lane");
      for (unsigned d = 1; d <= 16u; d <<= 1) {
         LLVMValueRef nbr  = emit_shfl_up(t, acc, d);
         LLVMValueRef comb = emit_scan_combine(t, rop, acc, nbr);
         if (!t->ok)
            break;
         LLVMValueRef pred = LLVMBuildICmp(t->b, LLVMIntUGE, lane,
                                           LLVMConstInt(t->i32, d, false), "");
         acc = LLVMBuildSelect(t->b, pred, comb, acc, "scan");
      }
      if (t->ok)
         ssa_set(t, in->def.index, 0, acc);
      break;
   }
   /* Flat CTA-linear lane id = warp_id * NT + lane (one CTA per core, so the
    * warp id within the core is the warp id within the workgroup). */
   case nir_intrinsic_load_local_invocation_index: {
      LLVMValueRef wid = emit_csr_read(t, VX_CSR_WARP_ID,     "wid");
      LLVMValueRef nt  = emit_csr_read(t, VX_CSR_NUM_THREADS, "nt");
      LLVMValueRef tid = emit_csr_read(t, VX_CSR_THREAD_ID,   "tid");
      ssa_set(t, in->def.index, 0,
              LLVMBuildAdd(t->b, LLVMBuildMul(t->b, wid, nt, ""), tid, "lii"));
      break;
   }
   /* workgroup barrier: only the execution-scoped form needs a sync;
    * a pure memory barrier is a no-op in this per-thread model. */
   case nir_intrinsic_barrier:
      if (nir_intrinsic_execution_scope(in) != SCOPE_NONE)
         emit_vx_barrier(t);
      break;
   /* deref load/store: the deref operand is an iptr byte address; each
    * component is a separate scalar load/store, width from bit size. */
   case nir_intrinsic_load_deref: {
      LLVMValueRef addr = intr_src(t, in, 0);
      if (!addr) { t->ok = false; break; }
      LLVMTypeRef lt  = ity(t, in->def.bit_size);
      unsigned    esz = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            vp_iptr_const(t, c * esz), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, lt, p, "ld"));
      }
      break;
   }
   case nir_intrinsic_store_deref: {
      LLVMValueRef addr = intr_src(t, in, 0);
      if (!addr) { t->ok = false; break; }
      unsigned mask = nir_intrinsic_write_mask(in);
      unsigned nc   = nir_src_num_components(in->src[1]);
      unsigned esz  = nir_src_bit_size(in->src[1]) / 8u;
      for (unsigned c = 0; c < nc; c++) {
         if (!(mask & (1u << c)))
            continue;
         LLVMValueRef v = ssa_get(t, in->src[1].ssa->index, c);
         if (!v) { t->ok = false; break; }
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            vp_iptr_const(t, c * esz), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         LLVMBuildStore(t->b, v, p);
      }
      break;
   }
   case nir_intrinsic_vortex_rt_get: {
      unsigned slot = nir_intrinsic_base(in);
      LLVMValueRef status = ssa_get(t, in->src[0].ssa->index, 0);
      ssa_set(t, in->def.index, 0, emit_vx_rt_get(t, slot, status));
      break;
   }
   case nir_intrinsic_vortex_rt_wtrace: {
      /* src = { scene, flags|cull, origin(3), dir(3), tmin, tmax } */
      LLVMValueRef scene = ssa_get(t, in->src[0].ssa->index, 0);
      LLVMValueRef fc    = ssa_get(t, in->src[1].ssa->index, 0);
      LLVMValueRef ray[8] = {
         ssa_get(t, in->src[2].ssa->index, 0),   /* origin.x */
         ssa_get(t, in->src[2].ssa->index, 1),   /* origin.y */
         ssa_get(t, in->src[2].ssa->index, 2),   /* origin.z */
         ssa_get(t, in->src[3].ssa->index, 0),   /* dir.x    */
         ssa_get(t, in->src[3].ssa->index, 1),   /* dir.y    */
         ssa_get(t, in->src[3].ssa->index, 2),   /* dir.z    */
         ssa_get(t, in->src[4].ssa->index, 0),   /* tmin     */
         ssa_get(t, in->src[5].ssa->index, 0),   /* tmax     */
      };
      ssa_set(t, in->def.index, 0, emit_vx_rt_wtrace(t, scene, fc, ray));
      break;
   }
   case nir_intrinsic_vortex_rt_wait: {
      LLVMValueRef h = ssa_get(t, in->src[0].ssa->index, 0);
      ssa_set(t, in->def.index, 0, emit_vx_rt_wait(t, h));
      break;
   }
   case nir_intrinsic_vortex_rt_cb_ret:
      emit_vx_rt_cb_ret(t, ssa_get(t, in->src[0].ssa->index, 0));
      break;

   /* Screen-space derivatives: the difference against this lane's quad neighbour
    * (dir 1 = horizontal, 2 = vertical). One lane is one pixel and a quad is four
    * adjacent lanes, so this is a single SHFL -- fine and coarse are the same
    * permute on Vortex. */
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddx_coarse:
      for (unsigned c = 0; c < in->def.num_components; c++)
         ssa_set(t, in->def.index, c,
                 emit_quad_deriv(t, ssa_get(t, in->src[0].ssa->index, c), 1));
      break;
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddy_coarse:
      for (unsigned c = 0; c < in->def.num_components; c++)
         ssa_set(t, in->def.index, c,
                 emit_quad_deriv(t, ssa_get(t, in->src[0].ssa->index, c), 2));
      break;

   /* discard / demote: clear this lane's live flag. The lane keeps executing --
    * a covered neighbour in its quad may still shuffle a value out of it for a
    * derivative -- and the wrapper ANDs the flag into coverage, so the export is
    * what gets withheld, not the shader. */
   case nir_intrinsic_demote:
   case nir_intrinsic_terminate:
      LLVMBuildStore(t->b, LLVMConstInt(t->i32, 0, false), t->fs_live);
      break;
   case nir_intrinsic_demote_if:
   case nir_intrinsic_terminate_if: {
      LLVMValueRef cond = ssa_get(t, in->src[0].ssa->index, 0);
      if (LLVMTypeOf(cond) != LLVMInt1TypeInContext(t->ctx))
         cond = LLVMBuildICmp(t->b, LLVMIntNE, cond,
                              LLVMConstInt(LLVMTypeOf(cond), 0, false), "");
      LLVMValueRef cur = LLVMBuildLoad2(t->b, t->i32, t->fs_live, "live.v");
      LLVMBuildStore(t->b,
         LLVMBuildSelect(t->b, cond, LLVMConstInt(t->i32, 0, false), cur, "live"),
         t->fs_live);
      break;
   }
   /* Storage-image load/store. The image is a bindless descriptor whose device
    * address is src[0]; the pixel address is base + base_offset + x*bpp +
    * y*row_stride (single-mip 2D; z and multi-level strides are unused by the
    * RT output/accumulation images). The RT path uses two formats: the
    * accumulation image is RGBA32F (four 32-bit lanes stored verbatim) and the
    * tonemapped output image is RGBA8_UNORM (four bytes packed into one word).
    * Any other format fails loud rather than writing wrong pixels. */
   case nir_intrinsic_image_load:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store: {
      bool store = (in->intrinsic == nir_intrinsic_image_store ||
                    in->intrinsic == nir_intrinsic_bindless_image_store);
      enum pipe_format fmt = nir_intrinsic_format(in);
      unsigned bpp;
      bool is_f32;
      if (fmt == PIPE_FORMAT_R32G32B32A32_FLOAT) {
         bpp = 16; is_f32 = true;
      } else if (fmt == PIPE_FORMAT_R8G8B8A8_UNORM) {
         bpp = 4;  is_f32 = false;
      } else {
         mesa_logw("vortexpipe: image %s: unsupported format %d "
                   "(RGBA32F / RGBA8_UNORM only)", store ? "store" : "load", fmt);
         t->ok = false;
         break;
      }
      LLVMValueRef desc = intr_src(t, in, 0);
      LLVMValueRef dp   = LLVMBuildIntToPtr(t->b, desc, t->ptr, "");
      LLVMValueRef base = LLVMBuildLoad2(t->b, t->i64, dp, "imgbase");
      LLVMValueRef boff = LLVMBuildZExt(t->b,
         img_desc_u32(t, desc, VP_JIT_IMG_BASE_OFFSET), t->i64, "");
      LLVMValueRef row  = LLVMBuildZExt(t->b,
         img_desc_u32(t, desc, VP_JIT_IMG_ROW_STRIDE), t->i64, "");
      LLVMValueRef x = LLVMBuildZExt(t->b, ssa_get(t, in->src[1].ssa->index, 0),
                                     t->i64, "");
      LLVMValueRef y = LLVMBuildZExt(t->b, ssa_get(t, in->src[1].ssa->index, 1),
                                     t->i64, "");
      LLVMValueRef off = LLVMBuildAdd(t->b, boff,
         LLVMBuildAdd(t->b, LLVMBuildMul(t->b, x, LLVMConstInt(t->i64, bpp, false), ""),
                            LLVMBuildMul(t->b, y, row, ""), ""), "");
      LLVMValueRef addr = LLVMBuildAdd(t->b, base, off, "");
      if (store) {
         addr = emit_store_addr(t, addr);
         unsigned nc = nir_src_num_components(in->src[3]);
         if (is_f32) {
            for (unsigned c = 0; c < nc; c++) {
               LLVMValueRef v = ssa_get(t, in->src[3].ssa->index, c);
               if (!v) continue;
               LLVMValueRef a = LLVMBuildAdd(t->b, addr,
                  LLVMConstInt(t->i64, c * 4u, false), "");
               LLVMBuildStore(t->b, v, LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
            }
         } else {
            LLVMValueRef packed = LLVMConstInt(t->i32, 0, false);
            for (unsigned c = 0; c < nc; c++) {
               LLVMValueRef v = ssa_get(t, in->src[3].ssa->index, c);
               if (!v) continue;
               LLVMValueRef b = f32bits_to_unorm8(t, v);
               packed = LLVMBuildOr(t->b, packed,
                  LLVMBuildShl(t->b, b, LLVMConstInt(t->i32, c * 8u, false), ""), "");
            }
            LLVMBuildStore(t->b, packed, LLVMBuildIntToPtr(t->b, addr, t->ptr, ""));
         }
      } else {
         if (is_f32) {
            LLVMTypeRef lt = ity(t, in->def.bit_size);
            for (unsigned c = 0; c < in->def.num_components; c++) {
               LLVMValueRef a = LLVMBuildAdd(t->b, addr,
                  LLVMConstInt(t->i64, c * 4u, false), "");
               LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
               ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, lt, p, "img"));
            }
         } else {
            LLVMValueRef packed = LLVMBuildLoad2(t->b, t->i32,
               LLVMBuildIntToPtr(t->b, addr, t->ptr, ""), "img");
            for (unsigned c = 0; c < in->def.num_components; c++) {
               LLVMValueRef w = LLVMBuildLShr(t->b, packed,
                  LLVMConstInt(t->i32, c * 8u, false), "");
               ssa_set(t, in->def.index, c, unorm8_to_f32bits(t, w));
            }
         }
      }
      break;
   }
   default:
      mesa_logw("vortexpipe: vp_nir_to_llvm: unhandled intrinsic '%s'",
                nir_intrinsic_infos[in->intrinsic].name);
      t->ok = false;
   }
}

/* vx_gfx_set(slot, val): write one gfx-window slot (SETW, custom-1 funct3=6
 * funct2=1; slot in funct7[6:2], value in rs1, no rd). */
static void
emit_vx_gfx_set(struct vp_tr *t, unsigned slot, LLVMValueRef val)
{
   char s[48];
   snprintf(s, sizeof s, ".insn r 43, 6, %u, x0, $0, x0", (slot << 2) | 1u);
   LLVMTypeRef args[1] = { t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                       args, 1, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "r", 1,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   LLVMValueRef a[1] = { val };
   LLVMBuildCall2(t->b, fnty, ia, a, 1, "");
}

/* vx_gfx_get_after(slot, handle): read one gfx-window slot into rd (single-slot
 * GETW, custom-1 funct3=6 funct2=3, count=1 via rs2=x1), chained on `handle`
 * (rs1) so the scoreboard stalls the read until the producer wrote the slot. */
static LLVMValueRef
emit_vx_gfx_get_after(struct vp_tr *t, unsigned slot, LLVMValueRef handle)
{
   char s[48];
   snprintf(s, sizeof s, ".insn r 43, 6, %u, $0, $1, x1", (slot << 2) | 3u);
   LLVMTypeRef args[1] = { t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 1, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "=r,r", 4,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   LLVMValueRef a[1] = { handle };
   return LLVMBuildCall2(t->b, fnty, ia, a, 1, "texw");
}

/* vx_tex: sample TEX stage 0 (custom-1 funct3=5, R4-type). u/v/lod ride
 * rs1/rs2/rs3 and the packed A8R8G8B8 texel returns in rd. */
static LLVMValueRef
emit_vx_tex(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v,
            LLVMValueRef lod)
{
   /* Software sampler: call gfx_tex_sample_sw(&texstate[0], u, v, lod) on the
    * resident descriptor table (fs_main's 3rd param) instead of the FF TEX unit.
    * Same fixed-point u/v/lod convention; returns the packed A8R8G8B8 texel. */
   if (t->sw_tex) {
      LLVMTypeRef params[4] = { t->ptr, t->i32, t->i32, t->i32 };
      LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 4, false);
      LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_sw");
      if (!fn)
         fn = LLVMAddFunction(t->mod, "gfx_tex_sample_sw", fty);
      LLVMValueRef a[4] = { t->fs_texstate, u, v, lod };  /* stage 0 = table[0] */
      return LLVMBuildCall2(t->b, fty, fn, a, 4, "texsw");
   }

   /* vx_tex: rs1=u, rs2=v, rs3=lod, rd=texel, funct2=stage(0). The TEX unit takes
    * its operands in registers -- no window staging, no handle chaining. */
   const char *s = ".insn r4 43, 5, 0, $0, $1, $2, $3";
   LLVMTypeRef args[3] = { t->i32, t->i32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 3, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "=r,r,r,r", 8,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   LLVMValueRef a[3] = { u, v, lod };
   return LLVMBuildCall2(t->b, fnty, ia, a, 3, "texel");
}

/* ── RTU (ray-tracing unit) ops — ISA v2 window ABI ──────────────────
 * CUSTOM1 (opcode 43). vortex_rt_wtrace (funct3=7, funct2=0) issues one ray:
 * the per-trace config lane-packs into rs1 via wgather, the per-thread ray
 * geometry rides the f0..f7 FP register window. vortex_rt_wait (funct3=7,
 * funct2=1) is a single-op block. vortex_rt_get reads one hit slot post-
 * terminal (GETW, funct3=6 funct2=3, count=1). funct3=6 funct2=0 is cb_ret.
 * Mirrors sw/kernel/include/vx_raytrace.h. */

/* vx_wgather: lane-scatter {self,v1,v2,v3} across 4 lanes of one register
 * (lane0<-self, lane1<-v1, lane2<-v2, lane3<-v3). R4-type, funct3=0, funct2=0;
 * rd is tied in/out (holds self on input, the gathered result on output). */
static LLVMValueRef
emit_vx_wgather(struct vp_tr *t, LLVMValueRef self, LLVMValueRef v1,
                LLVMValueRef v2, LLVMValueRef v3)
{
   /* rd=$0 (tied to the self input, "0" below). The matching/tied input
    * consumes operand index $1, so the three scatter sources v1/v2/v3 are
    * $2/$3/$4 — referencing $1/$2/$3 here would alias the self register and
    * drop v3 (lane3). rs1<-v1(lane1), rs2<-v2(lane2), rs3<-v3(lane3). */
   const char *s = ".insn r4 43, 0, 0, $0, $2, $3, $4";
   const char *c = "=r,0,r,r,r";
   LLVMTypeRef args[4] = { t->i32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 4, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), c, strlen(c),
                                      /*HasSideEffects*/ true, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[4] = { self, v1, v2, v3 };
   return LLVMBuildCall2(t->b, fnty, ia, a, 4, "wgather");
}

/* vx_rt_get: rd <- hit slot (GETW, funct3=6 funct2=3, count=1; rs2=x1); rs1 =
 * status (scoreboard ordering token so the read stalls until wait's terminal
 * staged the hit). Reads the f32-bit slot into a GP register. */
static LLVMValueRef
emit_vx_rt_get(struct vp_tr *t, unsigned slot, LLVMValueRef status)
{
   char s[48];
   int n = snprintf(s, sizeof s, ".insn r 43, 6, %u, $0, $1, x1",
                    ((slot & 0x1f) << 2) | 3u);
   LLVMTypeRef args[1] = { t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 1, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, (size_t)n, "=r,r", 4,
                                      /*HasSideEffects*/ true, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[1] = { status };
   return LLVMBuildCall2(t->b, fnty, ia, a, 1, "rtget");
}

/* vx_rt_trace: rd = handle <- trace(rs1 = lane-packed config; f0..f7 = ray).
 * config = wgather(0, scene, 0, flags|cull): lane1=scene, lane2=payload(0),
 * lane3=flags|cull (the RTU reads rs1 lanes 1/2/3). The eight ray floats
 * are bound to f0..f7 — read by HW convention, like the tensor unit's fragment
 * window; the encoding itself only names rd/rs1, so the window operands ride
 * the operand list unreferenced. SSA values are i32 bit-patterns, so the ray
 * geometry is bitcast to float to land in the FP registers. */
static LLVMValueRef
emit_vx_rt_wtrace(struct vp_tr *t, LLVMValueRef scene, LLVMValueRef flags_cull,
                  LLVMValueRef ray[8])
{
   LLVMValueRef z = LLVMConstInt(t->i32, 0, false);
   LLVMValueRef cfg = emit_vx_wgather(t, z, scene, z, flags_cull);

   const char *s = ".insn r 43, 7, 0, $0, $1, x0";
   const char *c = "=r,r,{f0},{f1},{f2},{f3},{f4},{f5},{f6},{f7}";
   LLVMTypeRef args[9];
   args[0] = t->i32;                              /* cfg (rs1) */
   for (int i = 0; i < 8; i++) args[1 + i] = t->f32;  /* f0..f7 */
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 9, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), c, strlen(c),
                                      /*HasSideEffects*/ true, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[9];
   a[0] = cfg;
   for (int i = 0; i < 8; i++)
      a[1 + i] = LLVMBuildBitCast(t->b, ray[i], t->f32, "");
   return LLVMBuildCall2(t->b, fnty, ia, a, 9, "rttrace");
}

/* vx_rt_wait: rd = status <- wait(rs1 = handle). Single-op block (funct3=7,
 * funct2=1); it parks/revives like the regfile WAIT so it survives a callback
 * trap. The hit-window reads (vortex_rt_get/GETW) chain on the returned status. */
static LLVMValueRef
emit_vx_rt_wait(struct vp_tr *t, LLVMValueRef handle)
{
   const char *s = ".insn r 43, 7, 1, $0, $1, x0";
   LLVMTypeRef args[1] = { t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 1, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "=r,r", 4,
                                      /*HasSideEffects*/ true, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[1] = { handle };
   return LLVMBuildCall2(t->b, fnty, ia, a, 1, "rtwait");
}

/* vx_rt_cb_ret: release the parked context with rs1 = action. No result. */
static void
emit_vx_rt_cb_ret(struct vp_tr *t, LLVMValueRef action)
{
   const char *s = ".insn r 43, 6, 0, x0, $0, x0";
   LLVMTypeRef args[1] = { t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), args, 1, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "r", 1,
                                      /*HasSideEffects*/ true, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[1] = { action };
   LLVMBuildCall2(t->b, fnty, ia, a, 1, "");
}

/* A NIR texture op: gfx-v1 supports a plain 2D `texture()` sampling
 * the single bound texture (TEX stage 0) at LOD 0. The interpolated
 * texcoord is the coord source; the texture/sampler deref sources are
 * fixed-function (TEX DCRs) and ignored. The result vec4 is the
 * unpacked A8R8G8B8 texel as four floats in [0,1]. */
static void
emit_tex(struct vp_tr *t, nir_tex_instr *tex)
{
   LLVMValueRef u = NULL, v = NULL;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_coord) {
         u = ssa_get(t, tex->src[i].src.ssa->index, 0);
         v = ssa_get(t, tex->src[i].src.ssa->index, 1);
      }
   }
   /* Accept auto-LOD (tex) and explicit-LOD/bias (txl/txb) sampling; the device
    * path samples base level, so the LOD/bias source is ignored. Ray-tracing and
    * compute shaders emit txl since they have no implicit derivatives. */
   if ((tex->op != nir_texop_tex && tex->op != nir_texop_txl &&
        tex->op != nir_texop_txb) || !u || !v) {
      mesa_logw("vortexpipe: vp_nir_to_llvm: unsupported texture op %d", tex->op);
      t->ok = false;
      return;
   }

   /* float UV -> the TEX unit's S.23 fixed-point coordinate. */
   LLVMValueRef scale = LLVMConstReal(t->f32, (double)(1u << VP_TEX_FXD_FRAC));
   LLVMValueRef ux = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b,
      LLVMBuildBitCast(t->b, u, t->f32, ""), scale, ""), t->i32, "");
   LLVMValueRef vx = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b,
      LLVMBuildBitCast(t->b, v, t->f32, ""), scale, ""), t->i32, "");

   LLVMValueRef texel = emit_vx_tex(t, ux, vx,
                                    LLVMConstInt(t->i32, 0, false));

   /* unpack A8R8G8B8 -> {r,g,b,a} floats in [0,1]. */
   static const unsigned shift[4] = { 16, 8, 0, 24 };
   for (unsigned c = 0; c < tex->def.num_components && c < 4; c++) {
      LLVMValueRef byte = LLVMBuildAnd(t->b,
         LLVMBuildLShr(t->b, texel,
                       LLVMConstInt(t->i32, shift[c], false), ""),
         LLVMConstInt(t->i32, 0xff, false), "");
      LLVMValueRef f = LLVMBuildFMul(t->b,
         LLVMBuildUIToFP(t->b, byte, t->f32, ""),
         LLVMConstReal(t->f32, 1.0 / 255.0), "");
      ssa_set(t, tex->def.index, c, LLVMBuildBitCast(t->b, f, t->i32, ""));
   }
}

/* A NIR phi -> one LLVM phi per component. The incoming values are
 * wired up by emit_cfg's deferred pass, once every block + value
 * exists (a loop's header phi reads a value defined in its body). */
static void
emit_phi(struct vp_tr *t, nir_phi_instr *phi)
{
   LLVMTypeRef ty = (phi->def.bit_size == 64) ? t->i64
                  : (phi->def.bit_size == 1)
                       ? LLVMInt1TypeInContext(t->ctx) : t->i32;
   for (unsigned c = 0; c < phi->def.num_components; c++)
      ssa_set(t, phi->def.index, c, LLVMBuildPhi(t->b, ty, "phi"));
}

static bool
emit_instr(struct vp_tr *t, nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
      emit_load_const(t, nir_instr_as_load_const(instr));
      break;
   case nir_instr_type_alu:
      emit_alu(t, nir_instr_as_alu(instr));
      break;
   case nir_instr_type_deref:
      emit_deref(t, nir_instr_as_deref(instr));
      break;
   case nir_instr_type_intrinsic:
      emit_intrinsic(t, nir_instr_as_intrinsic(instr));
      break;
   case nir_instr_type_tex:
      emit_tex(t, nir_instr_as_tex(instr));
      break;
   case nir_instr_type_phi:
      emit_phi(t, nir_instr_as_phi(instr));
      break;
   case nir_instr_type_undef: {
      /* an undefined SSA value -- zero of the def's width is a safe
       * materialization (the type must match its later uses / phis). */
      nir_undef_instr *u = nir_instr_as_undef(instr);
      LLVMTypeRef ut = u->def.bit_size == 64 ? t->i64
                     : u->def.bit_size == 1  ? LLVMInt1TypeInContext(t->ctx)
                                             : t->i32;
      for (unsigned c = 0; c < u->def.num_components; c++)
         ssa_set(t, u->def.index, c, LLVMConstInt(ut, 0, false));
      break;
   }
   case nir_instr_type_jump:
      break;   /* the branch is realised from the block's successors */
   default:
      mesa_logw("vortexpipe: vp_nir_to_llvm: unhandled nir_instr_type %d",
                (int)instr->type);
      t->ok = false;
      break;
   }
   return t->ok;
}

/* Translate a function's control-flow graph. One LLVM basic block per
 * NIR block; the branch out of each is taken straight from NIR's
 * per-block successors -- a following nir_if becomes a conditional
 * branch, a lone successor an unconditional one, the end block a
 * return. NIR phis become LLVM phis whose incoming edges are wired in
 * a deferred pass, once every block + value exists. */
static void
emit_cfg(struct vp_tr *t, nir_function_impl *impl,
         LLVMValueRef fn, LLVMBasicBlockRef entry)
{
   nir_metadata_require(impl, nir_metadata_block_index);
   LLVMBasicBlockRef *bb = calloc(impl->num_blocks, sizeof(*bb));
   if (!bb) { t->ok = false; return; }

   /* one LLVM block per NIR block; block 0 reuses `entry` so the
    * kernel prologue stays contiguous with the first NIR block. */
   unsigned n = 0;
   nir_foreach_block(blk, impl)
      bb[blk->index] = (n++ == 0) ? entry
         : LLVMAppendBasicBlockInContext(t->ctx, fn, "b");

   /* instructions + a terminator for every block */
   nir_foreach_block(blk, impl) {
      LLVMPositionBuilderAtEnd(t->b, bb[blk->index]);
      nir_foreach_instr(instr, blk) {
         if (!emit_instr(t, instr)) { free(bb); return; }
      }
      nir_if *nif = nir_block_get_following_if(blk);
      if (nif) {
         LLVMValueRef c = ssa_get(t, nif->condition.ssa->index, 0);
         if (!c) { t->ok = false; free(bb); return; }
         if (LLVMTypeOf(c) != LLVMInt1TypeInContext(t->ctx))
            c = LLVMBuildICmp(t->b, LLVMIntNE, c,
                   LLVMConstInt(LLVMTypeOf(c), 0, false), "");
         LLVMBuildCondBr(t->b, c, bb[blk->successors[0]->index],
                                  bb[blk->successors[1]->index]);
      } else if (blk->successors[0] &&
                 blk->successors[0] != impl->end_block) {
         LLVMBuildBr(t->b, bb[blk->successors[0]->index]);
      } else {
         LLVMBuildRetVoid(t->b);
      }
   }

   /* deferred pass: wire each phi's incoming { value, predecessor }. */
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type != nir_instr_type_phi)
            continue;
         nir_phi_instr *phi = nir_instr_as_phi(instr);
         for (unsigned c = 0; c < phi->def.num_components; c++) {
            LLVMValueRef llphi = ssa_get(t, phi->def.index, c);
            if (!llphi)
               continue;
            nir_foreach_phi_src(src, phi) {
               LLVMValueRef     v    = ssa_get(t, src->src.ssa->index, c);
               LLVMBasicBlockRef pred = bb[src->pred->index];
               if (v)
                  LLVMAddIncoming(llphi, &v, &pred, 1);
            }
         }
      }
   }
   free(bb);
}

/* Mark `fn` as a Vortex kernel entry: emit the @llvm.global.annotations
 * record carrying "vortex.kernel" (what __attribute__((annotate(...)))
 * lowers to -- the llvm_vortex backend keys on it, emitting the
 * __vx_kentry_<name> alias the VXSYMTAB footer records). Also place `fn`
 * in @llvm.used so it survives --gc-sections: the device dispatches by
 * address, so nothing references the kernel statically. */
static void
emit_kernel_annotation(struct vp_tr *t, LLVMValueRef fn)
{
   const char *tag = "vortex.kernel";
   const char *file = "vortexpipe";

   LLVMValueRef tag_g = LLVMAddGlobal(t->mod,
      LLVMArrayType(t->i8, strlen(tag) + 1), ".vp.anno.tag");
   LLVMSetInitializer(tag_g,
      LLVMConstStringInContext(t->ctx, tag, strlen(tag), false));
   LLVMSetLinkage(tag_g, LLVMPrivateLinkage);

   LLVMValueRef file_g = LLVMAddGlobal(t->mod,
      LLVMArrayType(t->i8, strlen(file) + 1), ".vp.anno.file");
   LLVMSetInitializer(file_g,
      LLVMConstStringInContext(t->ctx, file, strlen(file), false));
   LLVMSetLinkage(file_g, LLVMPrivateLinkage);

   LLVMTypeRef elem_fields[5] = { t->ptr, t->ptr, t->ptr, t->i32, t->ptr };
   LLVMTypeRef elem_ty = LLVMStructTypeInContext(t->ctx, elem_fields, 5, false);
   LLVMValueRef elem_vals[5] = {
      fn, tag_g, file_g,
      LLVMConstInt(t->i32, 0, false),
      LLVMConstNull(t->ptr),
   };
   LLVMValueRef elem = LLVMConstNamedStruct(elem_ty, elem_vals, 5);

   LLVMValueRef arr = LLVMConstArray2(elem_ty, &elem, 1);
   LLVMValueRef g = LLVMAddGlobal(t->mod, LLVMArrayType(elem_ty, 1),
                                  "llvm.global.annotations");
   LLVMSetInitializer(g, arr);
   LLVMSetLinkage(g, LLVMAppendingLinkage);
   LLVMSetSection(g, "llvm.metadata");

   /* @llvm.used = appending global [1 x ptr] [ptr @fn] -> SHF_GNU_RETAIN. */
   LLVMValueRef used_elem = fn;
   LLVMValueRef used_arr = LLVMConstArray2(t->ptr, &used_elem, 1);
   LLVMValueRef used_g = LLVMAddGlobal(t->mod, LLVMArrayType(t->ptr, 1),
                                       "llvm.used");
   LLVMSetInitializer(used_g, used_arr);
   LLVMSetLinkage(used_g, LLVMAppendingLinkage);
   LLVMSetSection(used_g, "llvm.metadata");
}

/* Scan the shader's outputs and assign each a 16-byte slot in the
 * per-vertex record: slot 0 is gl_Position, slots 1.. are the
 * generic varyings in declaration order. */
static void
vs_scan_outputs(struct vp_tr *t, struct nir_shader *nir,
                struct vp_vs_layout *out_vs)
{
   unsigned next = 16;   /* slot 0 reserved for gl_Position */
   nir_foreach_shader_out_variable(var, nir) {
      if (t->nvars >= VP_MAXV) { t->ok = false; return; }
      int off;
      if (var->data.location == VARYING_SLOT_POS) {
         off = 0;
      } else {
         off = (int)next;
         next += 16;
         if (out_vs && out_vs->num_varyings < VP_VS_MAX_VARYINGS) {
            out_vs->varying_loc[out_vs->num_varyings]   = var->data.location;
            out_vs->varying_comps[out_vs->num_varyings] =
               glsl_get_components(var->type);
         }
         if (out_vs)
            out_vs->num_varyings++;
      }
      t->vars[t->nvars].var     = var;
      t->vars[t->nvars].alloca  = NULL;
      t->vars[t->nvars].out_off = off;
      t->nvars++;
   }
   t->out_stride = next;   /* 16 * (1 + num_varyings) */
   if (out_vs)
      out_vs->stride = t->out_stride;
}

/* Assign each fragment-shader input varying and output a 16-byte slot,
 * in declaration order. Inputs index the interpolated-varyings area
 * the kernel wrapper fills; outputs index the colour-output area. */
static void
fs_scan_io(struct vp_tr *t, struct nir_shader *nir)
{
   unsigned off = 0;
   nir_foreach_shader_in_variable(var, nir) {
      if (t->nvars >= VP_MAXV) { t->ok = false; return; }
      t->vars[t->nvars].var     = var;
      t->vars[t->nvars].alloca  = NULL;
      t->vars[t->nvars].out_off = (int)off;
      t->nvars++;
      off += 16;
   }
   /* Output slots are keyed by render-target index: a colour output at
    * FRAG_RESULT_DATA0+k lands at out slot k*16, so the wrapper can pack colours
    * 0..num_color-1 deterministically regardless of declaration order. Non-colour
    * outputs (gl_FragDepth/stencil — unsupported on gfx-v1, depth comes from the
    * plane) get a scratch slot past the colour area so their stores never corrupt
    * a colour and are never read back. */
   unsigned num_color = 0, scratch = 0;
   nir_foreach_shader_out_variable(var, nir) {
      if (t->nvars >= VP_MAXV) { t->ok = false; return; }
      unsigned loc = var->data.location;
      int rt;
      if (loc == FRAG_RESULT_COLOR) {
         rt = 0;                              /* broadcast colour -> RT0 */
      } else if (loc >= FRAG_RESULT_DATA0) {
         rt = (int)(loc - FRAG_RESULT_DATA0); /* explicit MRT index */
      } else {
         rt = -1;                             /* depth/stencil: scratch slot */
      }
      unsigned slot;
      if (rt < 0) {
         slot = GFX_OM_MAX_RT + scratch;      /* scratch area past the colours */
         scratch++;
      } else {
         if ((unsigned)rt >= GFX_OM_MAX_RT) { /* bound the RT index */
            mesa_loge("vortexpipe: FS colour output RT%d exceeds VX_OM_MAX_RT=%u",
                      rt, GFX_OM_MAX_RT);
            t->ok = false; return;
         }
         slot = (unsigned)rt;
         if ((unsigned)rt + 1 > num_color) num_color = (unsigned)rt + 1;
      }
      t->vars[t->nvars].var     = var;
      t->vars[t->nvars].alloca  = NULL;
      t->vars[t->nvars].out_off = (int)(slot * 16);
      t->nvars++;
   }
   t->fs_num_color = num_color ? num_color : 1;
   t->fs_out_words = (GFX_OM_MAX_RT + scratch) * 4;
}

/* ---- fragment-kernel wrapper ----------------------- *
 * A fragment shader runs on Vortex as a rasterizer-driven kernel: every
 * thread polls vx_rast() for quads, the wrapper interpolates the
 * varyings from the bcoord CSRs + the primitive buffer, calls the
 * translated fragment-shader body (fs_main), and writes the framebuffer.
 * The wrapper is hand-emitted -- it is the driver's fixed-function glue
 * around the programmable stage, and the translator's first emission of
 * control flow (a loop + per-pixel branches).                          */

/* address + constant byte offset. `v` is an iptr (i32 on rv32, i64 on
 * rv64); the constant matches so LLVM's Add doesn't see a width mismatch. */
static LLVMValueRef
addk(struct vp_tr *t, LLVMValueRef v, unsigned k)
{
   return LLVMBuildAdd(t->b, v, vp_iptr_const(t, k), "");
}

/* load an i32 from an i32 byte address */
static LLVMValueRef
emit_load_i32(struct vp_tr *t, LLVMValueRef addr)
{
   return LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildIntToPtr(t->b, addr, t->ptr, ""), "");
}

static void
emit_store_i32(struct vp_tr *t, LLVMValueRef addr, LLVMValueRef val)
{
   LLVMBuildStore(t->b, val,
      LLVMBuildIntToPtr(t->b, addr, t->ptr, ""));
}

/* This lane's fragment payload. The raster engine packs it into the launch message
 * and the core lands it in the warp's launch registers before the warp is
 * activated, so reading it is a CSR read -- no window op, no memory traffic.
 *   word 0 = pos (x[15:0] | y[30:16] | covered[31])
 *   word 1 = pid
 */
static LLVMValueRef
emit_vx_frag_payload(struct vp_tr *t, unsigned word)
{
   return emit_csr_read(t, word ? VX_CSR_FRAG_PID : VX_CSR_FRAG_POS,
                        word ? "frag_pid" : "frag_pos");
}

/* vx_shfl_bfly scoped to a quad: lane l exchanges with lane l^dir (dir 1 =
 * horizontal neighbour, 2 = vertical). custom-0 funct3=6 funct2=1; rs2 packs the
 * segment operands that confine the permute to four adjacent lanes.
 *
 * A shuffle whose source lane is INACTIVE returns the reader's own value, so the
 * difference collapses to zero rather than faulting -- which is why the four lanes
 * of a quad must all run the shader, helper lanes included. */
static LLVMValueRef
emit_vx_quad_swap(struct vp_tr *t, LLVMValueRef value, unsigned dir)
{
   const uint32_t bc = (VP_QUAD_SHFL_MASK << 12) | (VP_QUAD_SHFL_CVAL << 6) | dir;
   const char *s = ".insn r 11, 6, 1, $0, $1, $2";
   const char *c = "=r,r,r";
   LLVMTypeRef args[2] = { t->i32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 2, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, (char *)s, strlen(s), (char *)c,
                                      strlen(c), /*HasSideEffects*/ false, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[2] = { value, LLVMConstInt(t->i32, bc, false) };
   return LLVMBuildCall2(t->b, fnty, ia, a, 2, "quad_swap");
}

/* Screen-space derivative of a float: the difference against this lane's quad
 * neighbour. Vortex has one derivative precision, so fine/coarse both land here.
 *
 * The exchange is a SYMMETRIC xor-swap, so a bare (neighbour - self) would hand the
 * two lanes of a pair opposite signs. The derivative must point the same way for the
 * whole quad -- increasing x for ddx, increasing y for ddy -- so the lane that holds
 * the FAR pixel of the pair (sub & dir) subtracts the other way.
 * (|d| is unaffected, which is why an fwidth()-only shader would never notice.) */
static LLVMValueRef
emit_quad_deriv(struct vp_tr *t, LLVMValueRef value, unsigned dir)
{
   LLVMValueRef self = LLVMBuildBitCast(t->b, value, t->i32, "");
   LLVMValueRef nb   = emit_vx_quad_swap(t, self, dir);
   LLVMValueRef fs   = LLVMBuildBitCast(t->b, self, t->f32, "");
   LLVMValueRef fn_  = LLVMBuildBitCast(t->b, nb,   t->f32, "");

   LLVMValueRef lane = emit_csr_read(t, VX_CSR_CTA_THREAD_ID_X, "lane");
   LLVMValueRef far  = LLVMBuildICmp(t->b, LLVMIntNE,
      LLVMBuildAnd(t->b, lane, LLVMConstInt(t->i32, dir, false), ""),
      LLVMConstInt(t->i32, 0, false), "far");

   LLVMValueRef fwd = LLVMBuildFSub(t->b, fn_, fs, "");   /* near lane: nb - self */
   LLVMValueRef rev = LLVMBuildFSub(t->b, fs, fn_, "");   /* far lane:  self - nb */
   return LLVMBuildBitCast(t->b,
      LLVMBuildSelect(t->b, far, rev, fwd, "deriv"), t->i32, "");
}

/* vx_om_export: emit one fragment as a STORE into the OM aperture (custom-1
 * funct3=3, R4-type, rd=x0). rs1=aperture address, rs2=colour, rs3=depth;
 * funct2 = {has_depth, has_colour} = 3 (both). The cluster's OM steer peels the
 * store off the L1->L2 trunk and the OM ingress turns it back into a fragment,
 * so the LSU never learns that OM exists. */
static void
emit_vx_om_export(struct vp_tr *t, LLVMValueRef addr, LLVMValueRef colour,
                  LLVMValueRef depth)
{
   const char *s = ".insn r4 43, 3, 3, x0, $0, $1, $2";
   LLVMTypeRef args[3] = { t->i32, t->i32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                       args, 3, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "r,r,r", 6,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   LLVMValueRef a[3] = { addr, colour, depth };
   LLVMBuildCall2(t->b, fnty, ia, a, 3, "");
}

/* Aperture address of one pixel:
 *   base + ((face << (xbits+ybits)) | (y << xbits) | x) << record_shift
 * xbits/ybits/record_shift are unpacked from the FS arg block (per-draw). */
static LLVMValueRef
emit_om_aperture_addr(struct vp_tr *t, LLVMValueRef xbits, LLVMValueRef ybits,
                      LLVMValueRef shift, LLVMValueRef x, LLVMValueRef y,
                      LLVMValueRef face)
{
   LLVMValueRef idx = LLVMBuildOr(t->b,
      LLVMBuildShl(t->b, face, LLVMBuildAdd(t->b, xbits, ybits, ""), ""),
      LLVMBuildOr(t->b, LLVMBuildShl(t->b, y, xbits, ""), x, ""), "");
   return LLVMBuildAdd(t->b,
      LLVMConstInt(t->i32, (uint32_t)VX_MEM_OM_BASE_ADDR, false),
      LLVMBuildShl(t->b, idx, shift, ""), "");
}

/* reinterpret a raw fixed-point i32 as float: (float)raw / 2^frac */
static LLVMValueRef
emit_fixed_to_float(struct vp_tr *t, LLVMValueRef raw, unsigned frac)
{
   LLVMValueRef f = LLVMBuildSIToFP(t->b, raw, t->f32, "");
   return LLVMBuildFMul(t->b, f,
      LLVMConstReal(t->f32, 1.0 / (double)(1u << frac)), "");
}

/* fixed24 product, (i64(a)*i64(b)) >> 24 truncated to i32 -- the exact
 * fixed_t<24> operator* of the native FS (vx_gfx_abi.h). */
static LLVMValueRef
emit_fx24_mul(struct vp_tr *t, LLVMValueRef a, LLVMValueRef b)
{
   LLVMValueRef p = LLVMBuildMul(t->b,
      LLVMBuildSExt(t->b, a, t->i64, ""),
      LLVMBuildSExt(t->b, b, t->i64, ""), "");
   return LLVMBuildTrunc(t->b,
      LLVMBuildAShr(t->b, p, LLVMConstInt(t->i64, 24, false), ""), t->i32, "");
}

/* interpolate one rast_attrib_t {x,y,z} plane in Q7.24 integer arithmetic,
 * bit-identical to the native FS: x*dxq + z + y*dyq with fixed24 products and
 * wrapping i32 sums. dxq/dyq are the Q7.24-quantized gradients; returns the
 * raw Q7.24 result bits. */
static LLVMValueRef
emit_interp(struct vp_tr *t, LLVMValueRef attr,
            LLVMValueRef dxq, LLVMValueRef dyq)
{
   LLVMValueRef ax = emit_load_i32(t, attr);
   LLVMValueRef ay = emit_load_i32(t, addk(t, attr, 4));
   LLVMValueRef az = emit_load_i32(t, addk(t, attr, 8));
   LLVMValueRef r  = LLVMBuildAdd(t->b, emit_fx24_mul(t, ax, dxq), az, "");
   return LLVMBuildAdd(t->b, emit_fx24_mul(t, ay, dyq), r, "");
}

/* float colour component -> 8-bit integer, bit-identical to the native FS's
 * (uint8_t)(fixed24 * 255): quantize to Q7.24 (trunc toward zero), then the
 * wrapping i32 *255 >> 24 byte pack. The [-127,127] pre-clamp only guards
 * FPToSI poison; every representable Q7.24 value is inside it. */
static LLVMValueRef
emit_to_byte(struct vp_tr *t, LLVMValueRef f)
{
   LLVMValueRef lo = LLVMConstReal(t->f32, -127.0);
   LLVMValueRef hi = LLVMConstReal(t->f32,  127.0);
   f = LLVMBuildSelect(t->b, LLVMBuildFCmp(t->b, LLVMRealOGT, f, lo, ""),
                       f, lo, "");
   f = LLVMBuildSelect(t->b, LLVMBuildFCmp(t->b, LLVMRealOLT, f, hi, ""),
                       f, hi, "");
   LLVMValueRef q = LLVMBuildFPToSI(t->b,
      LLVMBuildFMul(t->b, f, LLVMConstReal(t->f32, 16777216.0), ""),
      t->i32, "");
   LLVMValueRef v = LLVMBuildMul(t->b, q,
      LLVMConstInt(t->i32, 255, false), "");
   v = LLVMBuildAShr(t->b, v, LLVMConstInt(t->i32, 24, false), "");
   return LLVMBuildAnd(t->b, v, LLVMConstInt(t->i32, 0xff, false), "");
}

/* read arg-block slot k (an i64 device address) widened/narrowed to iptr */
static LLVMValueRef
emit_arg_i32(struct vp_tr *t, LLVMValueRef arg, unsigned k)
{
   LLVMValueRef idx = LLVMConstInt(t->i32, k, false);
   LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, arg, &idx, 1, "");
   LLVMValueRef v64 = LLVMBuildLoad2(t->b, t->i64, gep, "");
   if (t->iptr == t->i64)
      return v64;
   return LLVMBuildTrunc(t->b, v64,
                         t->i32, "");
}

/* Fill the fragment shader's input varyings for one covered pixel.
 * The RASTER unit interpolates a fixed set of attribute planes --
 * colour (rgba) and one texcoord (uv); each FS input variable is
 * filled from one or the other by its component count: a <=2-component
 * varying is the texcoord, a wider one is the colour. This is the
 * gfx-v1 fixed-function varying mapping -- a real GPU interpolates
 * arbitrary user varyings, gfx-v1 has exactly these planes. */
static void
emit_fs_fill_varyings(struct vp_tr *t, LLVMValueRef prim,
                      LLVMValueRef in_addr,
                      LLVMValueRef dxq, LLVMValueRef dyq)
{
   /* The front end interpolates 6 scalar planes; expand_k packed the VS varyings
    * into them in declaration order [u,v,r,g,b,a] (gfx_frontend_k.h). Read them
    * back the same way: each FS input varying claims the next nc lanes, so a draw
    * carrying several varyings never collides them onto one plane (the old
    * nc<=2->texcoord / else->colour heuristic aliased every >=3-component varying
    * onto r,g,b,a). */
   static const unsigned lane[6] = {
      VP_RAST_ATTR_U, VP_RAST_ATTR_V,
      VP_RAST_ATTR_R, VP_RAST_ATTR_G, VP_RAST_ATTR_B, VP_RAST_ATTR_A };

   /* Perspective recovery: setup premultiplied every colour/uv plane by 1/w and
    * carries a separate 1/w plane, so the true attribute is
    * interp(a/w)/interp(1/w) (gfx_setup.h). For a screen-aligned triangle 1/w is
    * constant, so this is an exact divide. */
   LLVMValueRef rhw_f = emit_fixed_to_float(t,
      emit_interp(t, addk(t, prim, VP_RAST_ATTR_RHW), dxq, dyq), 24);
   LLVMValueRef nz = LLVMBuildFCmp(t->b, LLVMRealONE, rhw_f,
                                   LLVMConstReal(t->f32, 0.0), "");
   LLVMValueRef inv_rhw = LLVMBuildFDiv(t->b, LLVMConstReal(t->f32, 1.0),
      LLVMBuildSelect(t->b, nz, rhw_f, LLVMConstReal(t->f32, 1.0), ""), "");

   unsigned li = 0;
   for (unsigned i = 0; i < t->nvars; i++) {
      const nir_variable *var = t->vars[i].var;
      if (!var || var->data.mode != nir_var_shader_in ||
          t->vars[i].out_off < 0)
         continue;
      unsigned nc = glsl_get_components(var->type);
      LLVMValueRef slot = addk(t, in_addr, (unsigned)t->vars[i].out_off);
      for (unsigned c = 0; c < nc && li < 6u; c++, li++) {
         LLVMValueRef q = emit_interp(t, addk(t, prim, lane[li]), dxq, dyq);
         LLVMValueRef f = LLVMBuildFMul(t->b, emit_fixed_to_float(t, q, 24),
                                        inv_rhw, "");
         emit_store_i32(t, addk(t, slot, c * 4),
                        LLVMBuildBitCast(t->b, f, t->i32, ""));
      }
   }
}

/* Source of a covered quad's per-corner edge (barycentric) values, the one
 * thing that differs between the two raster variants: the HW path recomputes
 * them in-shader from the primitive's edge planes (P2 dropped the per-corner
 * bcoord payload from the frag window — it now carries only {pos_mask, pid}); the
 * SW path reads them from a resident gfx_rast_quad_t in memory (`quad_addr`,
 * words 1+axis*4+corner, pos_mask is word 0). `from_window` selects the HW path;
 * otherwise `quad_addr` is the SW source. */
struct vp_bc_src {
   bool         from_window;  /* HW: recompute from the primitive's edge planes */
   LLVMValueRef quad_addr;    /* SW: iptr address of the gfx_rast_quad_t */
   LLVMValueRef sub;          /* SW: this lane's corner within that quad (0..3) */
};

/* Edge value F[axis] at this lane's pixel, as a float (fixed_t<16> raw).
 * HW path: recompute F = ex*px + ey*py + ez from the primitive's edge plane
 * {x,y,z} (Q16.16 at `prim` + axis*12) — a 32-bit integer MAC bit-identical to the
 * raster HW bcoord and the native FS (vx_graphics.h). SW path reads the corner the
 * walk already computed, and ignores prim/px/py. */
static LLVMValueRef
emit_bc(struct vp_tr *t, const struct vp_bc_src *bc, LLVMValueRef prim,
        LLVMValueRef px, LLVMValueRef py, unsigned axis)
{
   LLVMValueRef raw;
   if (bc->from_window) {
      LLVMValueRef ex = emit_load_i32(t, addk(t, prim, axis * 12 + 0));
      LLVMValueRef ey = emit_load_i32(t, addk(t, prim, axis * 12 + 4));
      LLVMValueRef ez = emit_load_i32(t, addk(t, prim, axis * 12 + 8));
      raw = LLVMBuildAdd(t->b,
         LLVMBuildAdd(t->b, LLVMBuildMul(t->b, ex, px, ""),
                            LLVMBuildMul(t->b, ey, py, ""), ""), ez, "bcraw");
   } else {
      /* gfx_rast_quad_t: pos_mask @word0, bcoords[axis*4+corner] @word 1+..
       * The corner is this lane's sub, a runtime value, so index it dynamically. */
      LLVMValueRef off = LLVMBuildShl(t->b,
         LLVMBuildAdd(t->b, bc->sub,
            LLVMConstInt(t->i32, 1 + axis * 4, false), ""),
         LLVMConstInt(t->i32, 2, false), "bcoff");
      raw = emit_load_i32(t,
         LLVMBuildAdd(t->b, bc->quad_addr, vp_to_iptr(t, off), ""));
   }
   return emit_fixed_to_float(t, raw, 16);
}

/* Unpack the OM aperture geometry from the FS arg block (GFX_FS_ARG_APERTURE):
 * bits [7:0]=xbits, [15:8]=ybits, [23:16]=record_shift. Only the FF OM path uses
 * it; the SW OM path merges through gfx_om_fragment_sw and never forms an
 * aperture address. */
static void
emit_om_aperture_load(struct vp_tr *t, LLVMValueRef arg)
{
   if (t->sw_om) {
      t->om_ap_xbits = t->om_ap_ybits = t->om_ap_shift = NULL;
      return;
   }
   /* The aperture word is a packed 32-bit geometry field, not a pointer, so
    * force it to i32: emit_arg_i32 returns pointer-width (i64 at XLEN=64) and the
    * unpack below is all i32 arithmetic. Without this the ap_xbits/ybits/shift
    * ands and the shl shift-amounts mix i64 with i32 and the FS module fails LLVM
    * verification, silently dropping the fragment stage onto llvmpipe. */
   LLVMValueRef w = emit_arg_i32(t, arg, GFX_FS_ARG_APERTURE);
   if (LLVMTypeOf(w) != t->i32)
      w = LLVMBuildTrunc(t->b, w, t->i32, "ap_word");
   LLVMValueRef m8 = LLVMConstInt(t->i32, 0xffu, false);
   t->om_ap_xbits = LLVMBuildAnd(t->b, w, m8, "ap_xbits");
   t->om_ap_ybits = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, w, LLVMConstInt(t->i32, 8, false), ""), m8, "ap_ybits");
   t->om_ap_shift = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, w, LLVMConstInt(t->i32, 16, false), ""), m8, "ap_shift");
}

/* Shade ONE pixel — this lane's. STRAIGHT-LINE by design, like the native FS: a
 * lane the primitive misses is a helper invocation, so it runs the shader anyway
 * (its covered neighbours in the quad may shuffle a value out of it for a
 * derivative) and `cov` — never the thread mask — is what withholds its export.
 * Shared by both raster wrappers; `bc` selects the edge-value source. */
static void
emit_shade_pixel(struct vp_tr *t, LLVMValueRef fn,
                 LLVMValueRef fs_main, LLVMTypeRef fs_main_ty,
                 LLVMValueRef prim, LLVMValueRef in_scr, LLVMValueRef out_scr,
                 LLVMValueRef in_addr, LLVMValueRef out_addr,
                 LLVMValueRef texstate_ptr, LLVMValueRef omstate_ptr,
                 LLVMValueRef desc_ptr, LLVMValueRef mrt_ptr,
                 LLVMValueRef mrt_colors_addr, LLVMValueRef live,
                 LLVMValueRef pxc, LLVMValueRef pyc, LLVMValueRef cov,
                 const struct vp_bc_src *bc)
{
   /* A colour attachment is 2D, so there is no cube face to select. */
   LLVMValueRef face = LLVMConstInt(t->i32, 0, false);

   {
      /* barycentric gradient: dx = F0/(F0+F1+F2), dy = F1/sum, quantized to
       * Q7.24 exactly like the native FS's FloatA(recip * F0) -- the f32
       * product truncated toward zero. All downstream interpolation is then
       * integer fixed24, bit-identical to the native kernel. */
      LLVMValueRef f0 = emit_bc(t, bc, prim, pxc, pyc, 0);
      LLVMValueRef f1 = emit_bc(t, bc, prim, pxc, pyc, 1);
      LLVMValueRef f2 = emit_bc(t, bc, prim, pxc, pyc, 2);
      LLVMValueRef sum = LLVMBuildFAdd(t->b,
         LLVMBuildFAdd(t->b, f0, f1, ""), f2, "");
      LLVMValueRef recip = LLVMBuildFDiv(t->b,
         LLVMConstReal(t->f32, 1.0), sum, "recip");
      LLVMValueRef k24 = LLVMConstReal(t->f32, 16777216.0);
      LLVMValueRef dxq = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b,
         LLVMBuildFMul(t->b, recip, f0, ""), k24, ""), t->i32, "dxq");
      LLVMValueRef dyq = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b,
         LLVMBuildFMul(t->b, recip, f1, ""), k24, ""), t->i32, "dyq");

      /* interpolate the RASTER attribute planes into the FS input
       * varyings (colour and/or texcoord, by declaration). */
      emit_fs_fill_varyings(t, prim, in_addr, dxq, dyq);

      /* run the programmable fragment shader (3rd arg = SW texstate table,
       * 4th = resident FS descriptor table, 5th = the discard/demote flag). A
       * discarded lane clears the flag but still runs, so its quad neighbours keep
       * a value to shuffle; folding the flag into coverage is what drops its
       * export. The slot lives in the entry block (the SW-raster wrapper shades
       * inside a loop nest, and a non-static alloca would grow the stack per quad);
       * only the reset is per-pixel. */
      LLVMBuildStore(t->b, LLVMConstInt(t->i32, 1, false), live);
      LLVMValueRef cargs[5] = { in_scr, out_scr, texstate_ptr, desc_ptr, live };
      LLVMBuildCall2(t->b, fs_main_ty, fs_main, cargs, 5, "");
      cov = LLVMBuildAnd(t->b, cov,
         LLVMBuildLoad2(t->b, t->i32, live, "live.v"), "cov_live");

      /* pack a render target's FS output (4 floats at out_addr + rt*16) into an
       * R8G8B8A8 pixel. */
      unsigned num_color = t->fs_num_color ? t->fs_num_color : 1;
      LLVMValueRef rgba_rt[GFX_OM_MAX_RT];
      for (unsigned rt = 0; rt < num_color && rt < GFX_OM_MAX_RT; rt++) {
         LLVMValueRef rgba = LLVMConstInt(t->i32, 0, false);
         for (unsigned c = 0; c < 4; c++) {
            LLVMValueRef fc = LLVMBuildBitCast(t->b,
               emit_load_i32(t, addk(t, out_addr, rt * 16 + c * 4)), t->f32, "");
            LLVMValueRef bc8 = LLVMBuildShl(t->b, emit_to_byte(t, fc),
               LLVMConstInt(t->i32, c * 8, false), "");
            rgba = LLVMBuildOr(t->b, rgba, bc8, "");
         }
         rgba_rt[rt] = rgba;
      }
      LLVMValueRef rgba = rgba_rt[0];   /* RT0 (single-RT / FF paths) */

      /* fragment depth: the fixed-function screen-space plane MAC over
       * rast_attribs.z {A',B',C'} (Q7.24), bit-identical to the raster
       * early-Z and the native FS's
       * PLANE_Z: zbits = trunc32(A'*px + B'*py + C'), saturated to the OM's
       * 24-bit depth range. */
      LLVMValueRef zpx = emit_load_i32(t, addk(t, prim, VP_RAST_ATTR_Z));
      LLVMValueRef zpy = emit_load_i32(t, addk(t, prim, VP_RAST_ATTR_Z + 4));
      LLVMValueRef zpz = emit_load_i32(t, addk(t, prim, VP_RAST_ATTR_Z + 8));
      LLVMValueRef acc = LLVMBuildAdd(t->b,
         LLVMBuildAdd(t->b,
            LLVMBuildMul(t->b, LLVMBuildSExt(t->b, zpx, t->i64, ""),
                               LLVMBuildSExt(t->b, pxc, t->i64, ""), ""),
            LLVMBuildMul(t->b, LLVMBuildSExt(t->b, zpy, t->i64, ""),
                               LLVMBuildSExt(t->b, pyc, t->i64, ""), ""), ""),
         LLVMBuildSExt(t->b, zpz, t->i64, ""), "");
      LLVMValueRef z32 = LLVMBuildTrunc(t->b, acc, t->i32, "zbits");
      LLVMValueRef zmask = LLVMConstInt(t->i32, 0xffffff, false);
      LLVMValueRef depth_i = LLVMBuildSelect(t->b,
         LLVMBuildICmp(t->b, LLVMIntSLT, z32, LLVMConstInt(t->i32, 0, false), ""),
         LLVMConstInt(t->i32, 0, false),
         LLVMBuildSelect(t->b,
            LLVMBuildICmp(t->b, LLVMIntSGT, z32, zmask, ""), zmask, z32, ""),
         "depth");

      if (t->sw_om && num_color > 1) {
         /* >1 colour attachment merges via gfx_om_fragment_mrt_sw(
          * omstate, rt[], num_color, covered, px, py, face, colours[], depth) —
          * one shared depth op then a per-attachment blend + colour write. Pack
          * this pixel's per-RT colours into the reused src_colors[] slot and
          * submit. MRT is only valid on the SW-OM path. */
         for (unsigned rt = 0; rt < num_color && rt < GFX_OM_MAX_RT; rt++)
            emit_store_i32(t, addk(t, mrt_colors_addr, rt * 4), rgba_rt[rt]);
         LLVMTypeRef params[9] = { t->ptr, t->ptr, t->i32, t->i32, t->i32,
                                   t->i32, t->i32, t->ptr, t->i32 };
         LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                            params, 9, false);
         LLVMValueRef ofn = LLVMGetNamedFunction(t->mod, "gfx_om_fragment_mrt_sw");
         if (!ofn)
            ofn = LLVMAddFunction(t->mod, "gfx_om_fragment_mrt_sw", fty);
         LLVMValueRef colors_ptr = LLVMBuildIntToPtr(t->b, mrt_colors_addr,
                                                     t->ptr, "colors");
         LLVMValueRef a[9] = { omstate_ptr, mrt_ptr,
                               LLVMConstInt(t->i32, num_color, false),
                               cov, pxc, pyc, face, colors_ptr, depth_i };
         LLVMBuildCall2(t->b, fty, ofn, a, 9, "");
      } else if (t->sw_om) {
         /* SW output-merger: merge this pixel via gfx_om_fragment_sw(
          * omstate, covered, px, py, face, colour, depth) over the LSU (no
          * window staging, no vx_om4). The callee applies the coverage bit —
          * keeping this wrapper straight-line. */
         LLVMTypeRef params[7] = { t->ptr, t->i32, t->i32, t->i32, t->i32,
                                   t->i32, t->i32 };
         LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                            params, 7, false);
         LLVMValueRef ofn = LLVMGetNamedFunction(t->mod, "gfx_om_fragment_sw");
         if (!ofn)
            ofn = LLVMAddFunction(t->mod, "gfx_om_fragment_sw", fty);
         LLVMValueRef a[7] = { omstate_ptr, cov, pxc, pyc, face, rgba, depth_i };
         LLVMBuildCall2(t->b, fty, ofn, a, 7, "");
      } else {
         /* FF output-merger: export this pixel as a store into the OM aperture.
          * An uncovered lane must be skipped -- there is no coverage mask
          * downstream, and the ingress turns every aperture store into a
          * fragment. The branch diverges across lanes; the thread mask handles
          * it, and it is placed AFTER the shader body so a helper lane still
          * runs the shader. */
         LLVMBasicBlockRef bb_do   = LLVMAppendBasicBlockInContext(t->ctx, fn, "om_do");
         LLVMBasicBlockRef bb_skip = LLVMAppendBasicBlockInContext(t->ctx, fn, "om_skip");
         LLVMBuildCondBr(t->b,
            LLVMBuildICmp(t->b, LLVMIntNE, cov,
                          LLVMConstInt(t->i32, 0, false), ""),
            bb_do, bb_skip);

         LLVMPositionBuilderAtEnd(t->b, bb_do);
         emit_vx_om_export(t,
            emit_om_aperture_addr(t, t->om_ap_xbits, t->om_ap_ybits,
                                  t->om_ap_shift, pxc, pyc, face),
            rgba, depth_i);
         LLVMBuildBr(t->b, bb_skip);

         LLVMPositionBuilderAtEnd(t->b, bb_skip);
      }
   }
}

/* Build kernel_main: the straight-line run-once wrapper that drives the
 * translated fragment body `fs_main`. RASTER dispatch v2 is PUSH: the raster work
 * distributor launches one 1-warp fragment CTA per covered-quad wave and seeds
 * that wave's per-lane frag_payload_t into the warp's gfx register window at
 * launch. The FS therefore runs once, reads its pre-seeded payload back
 * with GETW, shades that one pixel, and returns — no poll loop, no fetch
 * op, no begin op. arg block: [0]=primitive buffer, [1]=texstate, [2]=omstate. */
static LLVMValueRef
emit_fs_wrapper(struct vp_tr *t, LLVMValueRef fs_main, LLVMTypeRef fs_main_ty)
{
   LLVMTypeRef  p1[1] = { t->ptr };
   LLVMTypeRef  kty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                       p1, 1, false);
   LLVMValueRef fn  = LLVMAddFunction(t->mod, "kernel_main", kty);
   LLVMValueRef arg = LLVMGetParam(fn, 0);

   LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(t->ctx, fn, "entry");

   /* entry: read the arg block, allocate per-pixel scratch. The
    * colour/depth buffers are reached by the OM unit through its
    * DCRs, so the kernel only needs the primitive buffer (arg[0]). */
   LLVMPositionBuilderAtEnd(t->b, entry);
   LLVMValueRef prim_base = emit_arg_i32(t, arg, 0);
   /* SW sampler: arg[1] is the resident gfx_sw_texstate_t[] device address
    * (the host fills it when texturing is routed to software); null on the HW
    * path. Passed to fs_main as its 3rd param. */
   LLVMValueRef texstate_ptr;
   if (t->sw_tex) {
      LLVMValueRef ts_addr = emit_arg_i32(t, arg, 1);
      texstate_ptr = LLVMBuildIntToPtr(t->b, ts_addr, t->ptr, "texstate");
   } else {
      texstate_ptr = LLVMConstNull(t->ptr);
   }
   /* SW output-merger: arg[2] is the resident gfx_sw_omstate_t device address
    * (host-filled when OM is routed to software). The wrapper merges each covered
    * sub-pixel via gfx_om_fragment_sw over the LSU instead of staging + vx_om4. */
   LLVMValueRef omstate_ptr = LLVMConstNull(t->ptr);
   if (t->sw_om) {
      LLVMValueRef os_addr = emit_arg_i32(t, arg, 2);
      omstate_ptr = LLVMBuildIntToPtr(t->b, os_addr, t->ptr, "omstate");
   }
   emit_om_aperture_load(t, arg);
   /* arg[GFX_FS_ARG_DESC] is the resident FS descriptor-table device address
    * (always supplied by vp_raster; zero-filled when the FS is descriptor-free). */
   LLVMValueRef desc_ptr = LLVMBuildIntToPtr(t->b,
      emit_arg_i32(t, arg, GFX_FS_ARG_DESC), t->ptr, "desc");
   /* arg[GFX_FS_ARG_MRT] is the resident gfx_sw_omcolor_t[] device
    * address (per-attachment blend/write-mask). Read only for a >1-RT FS on the
    * SW-OM path; a single-RT draw leaves it null. */
   LLVMValueRef mrt_ptr = LLVMConstNull(t->ptr);
   LLVMValueRef mrt_colors_addr = LLVMConstNull(t->iptr);
   unsigned nrt = t->fs_num_color ? t->fs_num_color : 1;
   LLVMValueRef in_scr  = LLVMBuildAlloca(t->b, LLVMArrayType(t->i32, 16),
                                          "fs_in");
   LLVMValueRef out_scr = LLVMBuildAlloca(t->b,
      LLVMArrayType(t->i32, t->fs_out_words ? t->fs_out_words : 4), "fs_out");
   LLVMValueRef in_addr  = LLVMBuildPtrToInt(t->b, in_scr,  t->iptr, "");
   LLVMValueRef out_addr = LLVMBuildPtrToInt(t->b, out_scr, t->iptr, "");
   LLVMValueRef live     = LLVMBuildAlloca(t->b, t->i32, "fs_live");
   if (t->sw_om && nrt > 1) {
      mrt_ptr = LLVMBuildIntToPtr(t->b, emit_arg_i32(t, arg, GFX_FS_ARG_MRT),
                                  t->ptr, "mrt");
      LLVMValueRef col_scr = LLVMBuildAlloca(t->b,
         LLVMArrayType(t->i32, nrt), "fs_colors");
      mrt_colors_addr = LLVMBuildPtrToInt(t->b, col_scr, t->iptr, "");
   }

   /* Read this lane's pre-seeded PIXEL payload from the window (GETW): one lane is
    * one pixel, and the quad it belongs to is the four adjacent lanes. `covered`
    * says whether this lane may export; an uncovered lane is a helper and still
    * runs the shader. */
   LLVMValueRef pos = emit_vx_frag_payload(t, 0);
   LLVMValueRef pid = emit_vx_frag_payload(t, 1);
   LLVMValueRef px  = LLVMBuildAnd(t->b, pos,
      LLVMConstInt(t->i32, 0xffff, false), "px");
   LLVMValueRef py  = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, pos, LLVMConstInt(t->i32, 16, false), ""),
      LLVMConstInt(t->i32, 0x7fff, false), "py");
   LLVMValueRef cov = LLVMBuildLShr(t->b, pos,
      LLVMConstInt(t->i32, 31, false), "cov");
   LLVMValueRef poff = LLVMBuildMul(t->b, pid,
      LLVMConstInt(t->i32, VP_RAST_PRIM_STRIDE, false), "");
   LLVMValueRef prim = LLVMBuildAdd(t->b, prim_base,
                                    vp_to_iptr(t, poff), "prim");

   /* HW raster: edge values are recomputed in-shader from prim[pid]'s edge planes
    * (the window carries only pos + pid; P2 dropped the bcoord payload). */
   struct vp_bc_src bc = { .from_window = true, .quad_addr = NULL, .sub = NULL };
   emit_shade_pixel(t, fn, fs_main, fs_main_ty, prim, in_scr, out_scr,
                    in_addr, out_addr, texstate_ptr, omstate_ptr, desc_ptr,
                    mrt_ptr, mrt_colors_addr, live, px, py, cov, &bc);

   LLVMBuildRetVoid(t->b);
   return fn;
}

/* SW-raster FS wrapper (one WARP per screen tile; lanes cooperate).
 * When RASTER is routed to software the FS is NOT the frag-window poll loop. Each
 * WARP owns one 8x8 screen tile; its lanes share the tile and split the covered
 * quads. One lane is one pixel and a quad is four adjacent lanes, so lane L takes
 * corner L&3 of quad L>>2 and the warp advances NT/4 quads at a time. All lanes walk
 * every primitive over the SAME tile in draw order (gfx_rast_walk_tile_sw, redundant
 * but identical across the warp — uniform control flow), then split the quad list.
 * This is the CudaRaster fine-rasterizer mapping and, critically, it is SIMT-safe:
 * the loops are uniform-trip (same for every lane) and the only divergence is
 * per-quad coverage, which is uniform across a quad's four lanes — so a derivative
 * SHFL always reads a live neighbour. (The earlier one-thread-per-tile shape gave
 * each lane its own tile, so the per-lane covered-quad counts diverged and the SIMT
 * reconvergence dropped fragments at full-warp occupancy.) The per-pixel OM ordering
 * (one tile owned by
 * one warp, prims in draw order) holds because each pixel belongs to exactly one
 * tile = one warp, and a warp serializes its prims. arg block: [0]=prim base,
 * [1]=texstate, [2]=omstate, [3]=num_prims, [4]=nx (tiles/row), [5]=num_tiles,
 * [6]=tile_logsize (reserved; matches the constant below), [7]=scissor_w,
 * [8]=scissor_h. */
#define VP_SW_RAST_TILE_LOG  3u                                  /* 8x8 px tile  */
#define VP_SW_RAST_MAX_QUADS (1u << (2 * (VP_SW_RAST_TILE_LOG - 1)))/* (tile/2)^2 */
#define VP_RAST_QUAD_WORDS   13u                  /* sizeof(gfx_rast_quad_t)/4    */

static LLVMValueRef
emit_fs_wrapper_sw_raster(struct vp_tr *t, LLVMValueRef fs_main,
                          LLVMTypeRef fs_main_ty)
{
   LLVMTypeRef  p1[1] = { t->ptr };
   LLVMTypeRef  kty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                       p1, 1, false);
   LLVMValueRef fn  = LLVMAddFunction(t->mod, "kernel_main", kty);
   LLVMValueRef arg = LLVMGetParam(fn, 0);

   LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(t->ctx, fn, "entry");
   LLVMBasicBlockRef setup = LLVMAppendBasicBlockInContext(t->ctx, fn, "setup");
   LLVMBasicBlockRef ploop = LLVMAppendBasicBlockInContext(t->ctx, fn, "ploop");
   LLVMBasicBlockRef pbody = LLVMAppendBasicBlockInContext(t->ctx, fn, "pbody");
   LLVMBasicBlockRef qloop = LLVMAppendBasicBlockInContext(t->ctx, fn, "qloop");
   LLVMBasicBlockRef qchk  = LLVMAppendBasicBlockInContext(t->ctx, fn, "qchk");
   LLVMBasicBlockRef qbody = LLVMAppendBasicBlockInContext(t->ctx, fn, "qbody");
   LLVMBasicBlockRef qinc  = LLVMAppendBasicBlockInContext(t->ctx, fn, "qinc");
   LLVMBasicBlockRef pnext = LLVMAppendBasicBlockInContext(t->ctx, fn, "pnext");
   LLVMBasicBlockRef exit  = LLVMAppendBasicBlockInContext(t->ctx, fn, "exit");

   /* entry: read the arg block + per-lane scratch (counters/quads in allocas so
    * mem2reg promotes the loop induction; no hand-built phi). */
   LLVMPositionBuilderAtEnd(t->b, entry);
   LLVMValueRef prim_base = emit_arg_i32(t, arg, 0);
   LLVMValueRef texstate_ptr = LLVMConstNull(t->ptr);
   if (t->sw_tex)
      texstate_ptr = LLVMBuildIntToPtr(t->b, emit_arg_i32(t, arg, 1), t->ptr, "texstate");
   LLVMValueRef omstate_ptr = LLVMConstNull(t->ptr);
   if (t->sw_om)
      omstate_ptr = LLVMBuildIntToPtr(t->b, emit_arg_i32(t, arg, 2), t->ptr, "omstate");
   emit_om_aperture_load(t, arg);
   /* Resident FS descriptor table (arg[GFX_FS_ARG_DESC]). */
   LLVMValueRef desc_ptr = LLVMBuildIntToPtr(t->b,
      emit_arg_i32(t, arg, GFX_FS_ARG_DESC), t->ptr, "desc");
   /* arg-block counts are i64 device words; narrow to i32 for the loop math. */
   /* arg[3] is the front-end meta buffer address; meta[0] is the kept-prim count
    * P (after clip + face cull), known only on-device. Load it so the walk covers
    * exactly the dense primbuf — a fully-culled draw yields P==0 (no stale walk). */
   LLVMValueRef meta_ptr  = LLVMBuildIntToPtr(t->b, emit_arg_i32(t, arg, 3),
                                              t->ptr, "meta");
   LLVMValueRef num_prims = LLVMBuildLoad2(t->b, t->i32, meta_ptr, "num_prims");
   LLVMValueRef nx        = emit_to_i32(t, emit_arg_i32(t, arg, 4));
   LLVMValueRef num_tiles = emit_to_i32(t, emit_arg_i32(t, arg, 5));
   LLVMValueRef scis_w    = emit_to_i32(t, emit_arg_i32(t, arg, 7));
   LLVMValueRef scis_h    = emit_to_i32(t, emit_arg_i32(t, arg, 8));

   /* Per-attachment colour state (arg[GFX_FS_ARG_MRT]) + a src_colors[]
    * scratch, read only for a >1-RT FS on the SW-OM path. */
   LLVMValueRef mrt_ptr = LLVMConstNull(t->ptr);
   LLVMValueRef mrt_colors_addr = LLVMConstNull(t->iptr);
   unsigned nrt = t->fs_num_color ? t->fs_num_color : 1;
   LLVMValueRef in_scr  = LLVMBuildAlloca(t->b, LLVMArrayType(t->i32, 16), "fs_in");
   LLVMValueRef out_scr = LLVMBuildAlloca(t->b,
      LLVMArrayType(t->i32, t->fs_out_words ? t->fs_out_words : 4), "fs_out");
   if (t->sw_om && nrt > 1) {
      mrt_ptr = LLVMBuildIntToPtr(t->b, emit_arg_i32(t, arg, GFX_FS_ARG_MRT),
                                  t->ptr, "mrt");
      LLVMValueRef col_scr = LLVMBuildAlloca(t->b,
         LLVMArrayType(t->i32, nrt), "fs_colors");
      mrt_colors_addr = LLVMBuildPtrToInt(t->b, col_scr, t->iptr, "");
   }
   LLVMValueRef quadbuf = LLVMBuildAlloca(t->b,
      LLVMArrayType(t->i32, VP_SW_RAST_MAX_QUADS * VP_RAST_QUAD_WORDS), "quads");
   /* Entry-block slot: this wrapper shades inside a loop nest, and an alloca built
    * at the shade site would grow the stack once per quad. */
   LLVMValueRef live      = LLVMBuildAlloca(t->b, t->i32, "fs_live");
   LLVMValueRef pid_slot  = LLVMBuildAlloca(t->b, t->i32, "pid");
   LLVMValueRef base_slot = LLVMBuildAlloca(t->b, t->i32, "base");
   LLVMValueRef cnt_slot  = LLVMBuildAlloca(t->b, t->i32, "cnt");
   LLVMValueRef in_addr   = LLVMBuildPtrToInt(t->b, in_scr,  t->iptr, "");
   LLVMValueRef out_addr  = LLVMBuildPtrToInt(t->b, out_scr, t->iptr, "");
   LLVMValueRef quad_base = LLVMBuildPtrToInt(t->b, quadbuf, t->iptr, "");

   /* tile id = blockIdx.x (one warp == one CTA == one tile, set by the launch
    * geometry); lane = threadIdx.x (0..NT-1); NT = blockDim.x. tile_idx is
    * UNIFORM across the warp, so the bounds branch is uniform — no divergence. */
   LLVMValueRef tile_idx = emit_csr_read(t, VX_CSR_CTA_BLOCK_ID_X, "tile_idx");
   LLVMValueRef lane     = emit_csr_read(t, VX_CSR_CTA_THREAD_ID_X, "lane");
   LLVMValueRef ntv      = emit_csr_read(t, VX_CSR_CTA_BLOCK_DIM_X, "nt");
   LLVMBuildCondBr(t->b,
      LLVMBuildICmp(t->b, LLVMIntULT, tile_idx, num_tiles, ""), setup, exit);

   /* setup: tile origin in pixels, then start the prim walk at pid 0. */
   LLVMPositionBuilderAtEnd(t->b, setup);
   LLVMValueRef logc = LLVMConstInt(t->i32, VP_SW_RAST_TILE_LOG, false);
   LLVMValueRef tx = LLVMBuildShl(t->b,
      LLVMBuildURem(t->b, tile_idx, nx, ""), logc, "tx");
   LLVMValueRef ty = LLVMBuildShl(t->b,
      LLVMBuildUDiv(t->b, tile_idx, nx, ""), logc, "ty");
   LLVMBuildStore(t->b, LLVMConstInt(t->i32, 0, false), pid_slot);
   LLVMBuildBr(t->b, ploop);

   /* ploop: for each primitive in draw order (uniform: num_prims is warp-wide). */
   LLVMPositionBuilderAtEnd(t->b, ploop);
   LLVMValueRef pid = LLVMBuildLoad2(t->b, t->i32, pid_slot, "pid.v");
   LLVMBuildCondBr(t->b,
      LLVMBuildICmp(t->b, LLVMIntULT, pid, num_prims, ""), pbody, exit);

   /* pbody: every lane walks this prim over the SAME tile — identical args →
    * identical control flow, so the call is uniform across the warp (no divergent
    * recursion/loops). Each lane fills its own quad buffer; the lanes then split
    * the quad list below. (Redundant walk; a shared-memory single walk is the perf
    * follow-up.) */
   LLVMPositionBuilderAtEnd(t->b, pbody);
   LLVMValueRef poff = LLVMBuildMul(t->b, pid,
      LLVMConstInt(t->i32, VP_RAST_PRIM_STRIDE, false), "");
   LLVMValueRef prim = LLVMBuildAdd(t->b, prim_base, vp_to_iptr(t, poff), "prim");
   LLVMTypeRef wparams[9] = { t->ptr, t->i32, t->i32, t->i32, t->i32,
                              t->i32, t->i32, t->ptr, t->i32 };
   LLVMTypeRef wty = LLVMFunctionType(t->i32, wparams, 9, false);
   LLVMValueRef wfn = LLVMGetNamedFunction(t->mod, "gfx_rast_walk_tile_sw");
   if (!wfn)
      wfn = LLVMAddFunction(t->mod, "gfx_rast_walk_tile_sw", wty);
   LLVMValueRef wargs[9] = {
      LLVMBuildIntToPtr(t->b, prim, t->ptr, "primp"), pid, tx, ty, logc,
      scis_w, scis_h, quadbuf, LLVMConstInt(t->i32, VP_SW_RAST_MAX_QUADS, false) };
   LLVMValueRef cnt = LLVMBuildCall2(t->b, wty, wfn, wargs, 9, "cnt");
   LLVMBuildStore(t->b, cnt, cnt_slot);
   /* One lane is one pixel, so a quad is four ADJACENT lanes: lane L takes corner
    * L&3 of quad L>>2, and the warp holds NT/4 quads at a time. */
   LLVMBuildStore(t->b,
      LLVMBuildLShr(t->b, lane, LLVMConstInt(t->i32, 2, false), ""), base_slot);
   LLVMValueRef sub = LLVMBuildAnd(t->b, lane,
      LLVMConstInt(t->i32, VX_FRAG_QUAD_LANES - 1, false), "sub");
   /* NT/4 quads per warp. Clamp to >=1: a warp narrower than a quad group cannot
    * honour the lane law anyway, and a zero stride would spin qloop forever. */
   LLVMValueRef qstride_raw = LLVMBuildLShr(t->b, ntv,
      LLVMConstInt(t->i32, 2, false), "");
   LLVMValueRef qstride = LLVMBuildSelect(t->b,
      LLVMBuildICmp(t->b, LLVMIntEQ, qstride_raw,
         LLVMConstInt(t->i32, 0, false), ""),
      LLVMConstInt(t->i32, 1, false), qstride_raw, "qstride");
   LLVMBuildBr(t->b, qloop);

   /* qloop: split the tile's covered quads across the warp's quad groups. The loop
    * is UNIFORM — every lane steps base by NT/4 until base >= MAX, the same trip
    * count for all lanes (MAX is a constant) — so reconvergence is trivial. Quad
    * group G handles quads G, G+NT/4, G+2*NT/4, … */
   LLVMPositionBuilderAtEnd(t->b, qloop);
   LLVMValueRef base = LLVMBuildLoad2(t->b, t->i32, base_slot, "base.v");
   LLVMBuildCondBr(t->b,
      LLVMBuildICmp(t->b, LLVMIntULT, base,
         LLVMConstInt(t->i32, VP_SW_RAST_MAX_QUADS, false), ""), qchk, pnext);

   /* qchk: shade this group's slot only if it holds a real quad (base < cnt). A
    * divergent *if* (not a loop), uniform within a quad group — so the four lanes
    * of a quad stay in lockstep and a cross-lane derivative still reads its true
    * neighbour. */
   LLVMPositionBuilderAtEnd(t->b, qchk);
   LLVMValueRef cntv = LLVMBuildLoad2(t->b, t->i32, cnt_slot, "cnt.v");
   LLVMBuildCondBr(t->b,
      LLVMBuildICmp(t->b, LLVMIntULT, base, cntv, ""), qbody, qinc);

   /* qbody: shade this lane's corner of quad #base. Every lane walked the same
    * prim over the same tile, so the four lanes of a group read identical quad
    * buffers — the split is over corners, not over data. */
   LLVMPositionBuilderAtEnd(t->b, qbody);
   LLVMValueRef qoff = LLVMBuildMul(t->b, base,
      LLVMConstInt(t->i32, VP_RAST_QUAD_WORDS * 4, false), "");
   LLVMValueRef quad_addr = LLVMBuildAdd(t->b, quad_base, vp_to_iptr(t, qoff), "quad");
   LLVMValueRef pos_mask = emit_load_i32(t, quad_addr);

   /* pos_mask: mask@[3:0], qx@[4+:DIM-1], qy@[4+DIM-1+:DIM-1]. This lane's pixel is
    * px=(qx<<1)|(sub&1), py=(qy<<1)|(sub>>1), covered = mask[sub]. */
   const uint32_t dim_mask = (1u << (VX_RASTER_DIM_BITS - 1)) - 1;
   LLVMValueRef one = LLVMConstInt(t->i32, 1, false);
   LLVMValueRef qx = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, pos_mask, LLVMConstInt(t->i32, 4, false), ""),
      LLVMConstInt(t->i32, dim_mask, false), "qx");
   LLVMValueRef qy = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, pos_mask,
         LLVMConstInt(t->i32, 4 + VX_RASTER_DIM_BITS - 1, false), ""),
      LLVMConstInt(t->i32, dim_mask, false), "qy");
   LLVMValueRef px = LLVMBuildOr(t->b, LLVMBuildShl(t->b, qx, one, ""),
      LLVMBuildAnd(t->b, sub, one, ""), "px");
   LLVMValueRef py = LLVMBuildOr(t->b, LLVMBuildShl(t->b, qy, one, ""),
      LLVMBuildLShr(t->b, sub, one, ""), "py");
   LLVMValueRef cov = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, pos_mask, sub, ""), one, "cov");

   struct vp_bc_src bc = { .from_window = false, .quad_addr = quad_addr,
                           .sub = sub };
   emit_shade_pixel(t, fn, fs_main, fs_main_ty, prim, in_scr, out_scr,
                    in_addr, out_addr, texstate_ptr, omstate_ptr, desc_ptr,
                    mrt_ptr, mrt_colors_addr, live, px, py, cov, &bc);
   /* builder is now at emit_shade_pixel's trailing fall-through block. */
   LLVMBuildBr(t->b, qinc);

   /* qinc: advance by NT/4 (uniform stride) to this group's next quad. */
   LLVMPositionBuilderAtEnd(t->b, qinc);
   LLVMBuildStore(t->b, LLVMBuildAdd(t->b, base, qstride, ""), base_slot);
   LLVMBuildBr(t->b, qloop);

   /* pnext: advance to the next primitive. */
   LLVMPositionBuilderAtEnd(t->b, pnext);
   LLVMBuildStore(t->b,
      LLVMBuildAdd(t->b, pid, LLVMConstInt(t->i32, 1, false), ""), pid_slot);
   LLVMBuildBr(t->b, ploop);

   LLVMPositionBuilderAtEnd(t->b, exit);
   LLVMBuildRetVoid(t->b);
   return fn;
}

bool
vp_nir_to_llvm(struct nir_shader *nir, char **out_ir,
               struct vp_vs_layout *out_vs,
               const struct vp_sw_routing *routing)
{
   if (out_ir)
      *out_ir = NULL;
   if (out_vs)
      memset(out_vs, 0, sizeof *out_vs);
   if (!nir)
      return false;

   if (getenv("VORTEXPIPE_DEBUG_NIR")) {
      fprintf(stderr, "=== vortexpipe: lavapipe-lowered NIR ===\n");
      nir_print_shader(nir, stderr);
      fprintf(stderr, "=== end NIR ===\n");
   }

   struct vp_tr t = {0};
   t.ok    = true;
   t.is_vs = (nir->info.stage == MESA_SHADER_VERTEX);
   t.is_fs = (nir->info.stage == MESA_SHADER_FRAGMENT);
   /* Per-unit SW routing (FS only). */
   t.sw_tex    = (t.is_fs && routing && routing->sw_tex);
   t.sw_om     = (t.is_fs && routing && routing->sw_om);
   t.sw_raster = (t.is_fs && routing && routing->sw_raster);
   t.ctx   = LLVMContextCreate();
   t.mod   = LLVMModuleCreateWithNameInContext("vortex_shader", t.ctx);
   LLVMSetTarget(t.mod, vp_target_triple());
   LLVMSetDataLayout(t.mod, vp_target_datalayout());
   t.b   = LLVMCreateBuilderInContext(t.ctx);
   t.i8  = LLVMInt8TypeInContext(t.ctx);
   t.i32 = LLVMInt32TypeInContext(t.ctx);
   t.i64 = LLVMInt64TypeInContext(t.ctx);
   /* Pointer-sized int -- matches the module's target datalayout above. */
   t.iptr = vp_xlen_is_64() ? t.i64 : t.i32;
   t.f32 = LLVMFloatTypeInContext(t.ctx);
   t.f64 = LLVMDoubleTypeInContext(t.ctx);
   t.ptr = LLVMPointerTypeInContext(t.ctx, 0);

   /* The function the NIR body is walked into. Compute and vertex
    * shaders ARE kernel_main; a fragment shader is fs_main(in,out) --
    * emit_fs_wrapper then builds the kernel_main that drives it. The
    * KMU startup (vx_start.S in libvortex2.a) calls `kernel_main`. */
   LLVMValueRef fn;
   LLVMTypeRef  fs_main_ty = NULL;
   if (t.is_fs) {
      /* fs_main(varyings_in, colour_out, texstate_table, desc_table, live). The 3rd
       * param carries the resident gfx_sw_texstate_t[] for the SW sampler
       * (null on the all-HW path); the 4th is the resident FS descriptor
       * table (i64[] of constant-buffer base addresses) that load_ubo /
       * load_push_constant / load_ssbo read (null when the FS is descriptor-free,
       * in which case those intrinsics are never emitted); the 5th is the
       * discard/demote flag the wrapper folds into coverage. */
      LLVMTypeRef p5[5] = { t.ptr, t.ptr, t.ptr, t.ptr, t.ptr };
      fs_main_ty = LLVMFunctionType(LLVMVoidTypeInContext(t.ctx),
                                    p5, 5, false);
      fn = LLVMAddFunction(t.mod, "fs_main", fs_main_ty);
      LLVMSetLinkage(fn, LLVMInternalLinkage);
      t.fs_live = LLVMGetParam(fn, 4);
   } else {
      LLVMTypeRef p1[1] = { t.ptr };
      LLVMTypeRef kty = LLVMFunctionType(LLVMVoidTypeInContext(t.ctx),
                                         p1, 1, false);
      fn = LLVMAddFunction(t.mod, "kernel_main", kty);
      t.arg = LLVMGetParam(fn, 0);
      LLVMSetValueName2(t.arg, "arg", 3);
   }

   LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(t.ctx, fn, "entry");
   LLVMPositionBuilderAtEnd(t.b, entry);
   t.entry = entry;
   t.scratch_size = nir->scratch_size;

   /* Where a discarded fragment's stores go to die. Sized for the widest store the
    * translator emits (4 components x 8 bytes), so a retargeted vec4 stays inside
    * it. Never read back. */
   if (t.is_fs) {
      LLVMValueRef sink = LLVMBuildAlloca(t.b, LLVMArrayType(t.i32, 8),
                                          "discard_sink");
      t.fs_sink = LLVMBuildPtrToInt(t.b, sink, t.iptr, "");
   }

   /* Vertex-shader prologue: assign output slots, then read the
    * vertex id and the output-buffer base from %arg[0]. */
   if (t.is_vs) {
      vs_scan_outputs(&t, nir, out_vs);
      if (out_vs)
         out_vs->needs_vertex_input = (nir->info.inputs_read != 0);
      /* Global vertex id = blockIdx.x × blockDim.x + threadIdx.x.
       * Pre-fix this read only THREAD_ID_X, which made sense when
       * vp_launch_vs ran a single CTA whose block_dim covered every
       * vertex. The driver now uses grid_x × block_x to fill warps
       * AND spread CTAs across cores; the kernel must compute the
       * global id to match. Out-of-bounds vids (trailing CTA's
       * unused threads, when vertex_count isn't a multiple of
       * block_dim) write into the padded device output buffer; the
       * host read-back stops at the real vertex_count so the slack
       * never leaves the device. */
      LLVMValueRef vid_block_id  = emit_csr_read(&t, VX_CSR_CTA_BLOCK_ID_X, "bid");
      LLVMValueRef vid_block_dim = emit_csr_read(&t, VX_CSR_CTA_BLOCK_DIM_X, "bdim");
      LLVMValueRef vid_thread_id = emit_csr_read(&t, VX_CSR_CTA_THREAD_ID_X, "tid");
      t.vid = LLVMBuildAdd(t.b,
                LLVMBuildMul(t.b, vid_block_id, vid_block_dim, "blkofs"),
                vid_thread_id, "vid");
      /* %arg[0] / %arg[1] are i64 device addresses from the host runtime.
       * On rv64 the device stack base is 0x1FFFF0000 (33-bit), so we keep
       * iptr width: cast the i64 down only on rv32. */
      LLVMValueRef ob64 = LLVMBuildLoad2(t.b, t.i64, t.arg, "outbase64");
      t.out_base = (t.iptr == t.i64) ? ob64
                                     : LLVMBuildTrunc(t.b, ob64, t.i32, "outbase");
      /* arg slot 1: the vertex-attribute table (vp_launch_vs) -- 0 for
       * a self-contained VS that fetches no vertex-buffer inputs. */
      LLVMValueRef one = LLVMConstInt(t.i32, 1, false);
      LLVMValueRef atp = LLVMBuildGEP2(t.b, t.i64, t.arg, &one, 1, "");
      LLVMValueRef at64 = LLVMBuildLoad2(t.b, t.i64, atp, "attrtab64");
      t.attr_table = (t.iptr == t.i64) ? at64
                                       : LLVMBuildTrunc(t.b, at64, t.i32, "attrtab");
      /* arg slot 2: index buffer base (a u32 index per vertex), or 0 for a
       * direct (non-indexed) draw. On an indexed draw the i-th VS thread must
       * render index_buf[i], so the vertex id that drives gl_VertexIndex AND
       * vertex-attribute fetch is resolved through the index buffer here. When
       * arg[2] == 0 the vid stays the sequential global id, so the non-indexed
       * path is byte-identical to the prior ABI. */
      LLVMValueRef two   = LLVMConstInt(t.i32, 2, false);
      LLVMValueRef ibp   = LLVMBuildGEP2(t.b, t.i64, t.arg, &two, 1, "");
      LLVMValueRef ib64  = LLVMBuildLoad2(t.b, t.i64, ibp, "idxbuf64");
      LLVMValueRef hasix = LLVMBuildICmp(t.b, LLVMIntNE, ib64,
                                         LLVMConstInt(t.i64, 0, false), "hasidx");
      /* Instancing: arg slot 3 = verts-per-instance (0 => non-instanced fast
       * path, byte-identical to the pre-instancing ABI), slot 4 = base instance
       * (gl_BaseInstance). Thread gid maps to instance = gid / vpi and the
       * in-instance vertex vert = gid % vpi. The VS output slot stays the
       * sequential gid so expand_k reads the records densely; only the vid that
       * drives gl_VertexIndex / attribute fetch and the per-instance index-buffer
       * lookup use vert. */
      LLVMValueRef gid   = t.vid;                       /* sequential global id */
      LLVMValueRef three = LLVMConstInt(t.i32, 3, false);
      LLVMValueRef four  = LLVMConstInt(t.i32, 4, false);
      LLVMValueRef vpip  = LLVMBuildGEP2(t.b, t.i64, t.arg, &three, 1, "");
      LLVMValueRef vpi64 = LLVMBuildLoad2(t.b, t.i64, vpip, "vpi64");
      LLVMValueRef vpi   = LLVMBuildTrunc(t.b, vpi64, t.i32, "vpi");
      LLVMValueRef fip   = LLVMBuildGEP2(t.b, t.i64, t.arg, &four, 1, "");
      LLVMValueRef fi64  = LLVMBuildLoad2(t.b, t.i64, fip, "firstinst64");
      t.first_instance   = LLVMBuildTrunc(t.b, fi64, t.i32, "firstinst");
      LLVMValueRef vpi_is0   = LLVMBuildICmp(t.b, LLVMIntEQ, vpi,
                                             LLVMConstInt(t.i32, 0, false), "vpi0");
      /* guard the div/rem against a zero divisor on the non-instanced fast path */
      LLVMValueRef vpi_safe  = LLVMBuildSelect(t.b, vpi_is0,
                                               LLVMConstInt(t.i32, 1, false), vpi, "vpisafe");
      LLVMValueRef inst_q    = LLVMBuildUDiv(t.b, gid, vpi_safe, "instq");
      LLVMValueRef vert_r    = LLVMBuildURem(t.b, gid, vpi_safe, "vertr");
      t.instance = LLVMBuildSelect(t.b, vpi_is0,
                                   LLVMConstInt(t.i32, 0, false), inst_q, "instance");
      LLVMValueRef logical_vert = LLVMBuildSelect(t.b, vpi_is0, gid, vert_r, "logvert");

      LLVMValueRef raw_vid = logical_vert; /* in-instance vertex -> vid / index lookup */
      t.vraw = gid;                        /* sequential global id -> VS output slot */
      LLVMBasicBlockRef from_bb = LLVMGetInsertBlock(t.b);
      LLVMBasicBlockRef idx_bb  = LLVMAppendBasicBlockInContext(t.ctx, fn, "vid_idx");
      LLVMBasicBlockRef cont_bb = LLVMAppendBasicBlockInContext(t.ctx, fn, "vid_cont");
      LLVMBuildCondBr(t.b, hasix, idx_bb, cont_bb);
      LLVMPositionBuilderAtEnd(t.b, idx_bb);
      LLVMValueRef ib_iptr = (t.iptr == t.i64) ? ib64
                                               : LLVMBuildTrunc(t.b, ib64, t.i32, "idxbuf");
      LLVMValueRef ioff = LLVMBuildMul(t.b, vp_to_iptr(&t, raw_vid),
                                       vp_iptr_const(&t, 4), "");
      LLVMValueRef iea  = LLVMBuildAdd(t.b, ib_iptr, ioff, "");
      LLVMValueRef idx_vid = LLVMBuildLoad2(t.b, t.i32,
                               LLVMBuildIntToPtr(t.b, iea, t.ptr, ""), "idxvid");
      LLVMBuildBr(t.b, cont_bb);
      LLVMPositionBuilderAtEnd(t.b, cont_bb);
      LLVMValueRef vphi = LLVMBuildPhi(t.b, t.i32, "vid");
      LLVMValueRef vin_vals[2]  = { raw_vid, idx_vid };
      LLVMBasicBlockRef vin_bb[2] = { from_bb, idx_bb };
      LLVMAddIncoming(vphi, vin_vals, vin_bb, 2);
      t.vid = vphi;
   }

   /* Fragment-shader prologue: assign varying/output slots. fs_main's
    * two ptr params are the per-pixel interpolated-varyings input and
    * the colour-output area; the wrapper (emit_fs_wrapper) fills them. */
   if (t.is_fs) {
      fs_scan_io(&t, nir);
      t.fs_in_base  = LLVMBuildPtrToInt(t.b, LLVMGetParam(fn, 0),
                                        t.iptr, "fsin");
      t.fs_out_base = LLVMBuildPtrToInt(t.b, LLVMGetParam(fn, 1),
                                        t.iptr, "fsout");
      t.fs_texstate = LLVMGetParam(fn, 2);   /* gfx_sw_texstate_t* (SW tex) */
      /* The resident FS descriptor table (i64[] of constant-buffer base
       * addresses) is fs_main's 4th param. Route it through t.arg so the shared
       * load_ubo / load_ssbo / load_const_buf_base_addr_lvp / load_push_constant
       * lowering reads table[index] exactly as the compute path reads its arg
       * block. Null-safe: a descriptor-free FS never emits those intrinsics. */
      t.arg = LLVMGetParam(fn, 3);
      LLVMSetValueName2(t.arg, "desc", 4);
   }

   /* Compute-shader prologue: the workgroup's shared-memory base, read
    * once so load/store_shared can address it. CSR-read is i32; widen
    * to iptr so address arithmetic against it has matching widths. */
   if (!t.is_vs && !t.is_fs)
      t.lmem_base = vp_to_iptr(&t,
         emit_csr_read(&t, VX_CSR_CTA_LMEM_ADDR, "lmem"));

   nir_foreach_function_impl(impl, nir) {
      t.nval = impl->ssa_alloc;
      t.val  = calloc((size_t)t.nval * VP_MAXC, sizeof(LLVMValueRef));
      if (!t.val) { t.ok = false; break; }

      /* walk the control-flow graph (emits each block's terminator,
       * including the function return). NIR block 0 reuses the prologue's
       * current tail block so they stay contiguous — for the VS this is the
       * index-resolve merge block (cont_bb), not the original entry, which the
       * index-buffer branch already terminated. */
      emit_cfg(&t, impl, fn, LLVMGetInsertBlock(t.b));

      free(t.val);
      t.val = NULL;
      break;   /* one entrypoint impl per shader */
   }

   /* Fragment shaders: wrap fs_main in the rasterizer poll-loop
    * kernel_main. Compute/vertex shaders are already kernel_main. */
   LLVMValueRef kfn = fn;
   if (t.is_fs && t.ok)
      kfn = t.sw_raster ? emit_fs_wrapper_sw_raster(&t, fn, fs_main_ty)
                        : emit_fs_wrapper(&t, fn, fs_main_ty);

   if (t.ok) {
      /* Annotate the kernel "vortex.kernel" + retain it. The llvm-vortex
       * backend emits the __vx_kentry_<name> alias of the kernel body;
       * vxbin.py records it in the VXSYMTAB footer and the runtime resolves
       * it by name (kernel_main -> "main"). The KMU launches __vx_cta_entry,
       * which brings up the C environment and dispatches to the kernel
       * address it reads from VX_CSR_CTA_ENTRY. */
      emit_kernel_annotation(&t, kfn);
   }

   char *err = NULL;
   bool ok = t.ok &&
             LLVMVerifyModule(t.mod, LLVMReturnStatusAction, &err) == 0;
   if (ok) {
      vp_dbg("vortexpipe: vp_nir_to_llvm: translated %s shader to a "
             "Vortex kernel module",
             t.is_vs ? "vertex" : t.is_fs ? "fragment" : "compute");
      char *ir = LLVMPrintModuleToString(t.mod);
      if (getenv("VORTEXPIPE_DEBUG_IR"))
         fprintf(stderr, "=== vortexpipe: generated LLVM IR ===\n%s"
                         "=== end IR ===\n", ir);
      if (out_ir)
         *out_ir = ir;
      else
         LLVMDisposeMessage(ir);
   } else if (!t.ok) {
      mesa_logw("vortexpipe: vp_nir_to_llvm: shader not translatable yet");
   } else {
      mesa_logw("vortexpipe: vp_nir_to_llvm: module verify failed: %s",
                err ? err : "(no message)");
      if (getenv("VORTEXPIPE_DEBUG_IR")) {
         char *ir = LLVMPrintModuleToString(t.mod);
         fprintf(stderr, "=== vortexpipe: rejected LLVM IR ===\n%s"
                         "=== end IR ===\n", ir);
         LLVMDisposeMessage(ir);
      }
   }
   LLVMDisposeMessage(err);

   LLVMDisposeBuilder(t.b);
   LLVMDisposeModule(t.mod);
   LLVMContextDispose(t.ctx);
   return ok;
}

void
vp_free_ir(char *ir)
{
   if (ir)
      LLVMDisposeMessage(ir);
}

/* ---- descriptor scan ----------------------------------- *
 *
 * A compute kernel reaches its set-0 resources through the descriptor
 * buffer bound at constant-buffer index 1. vp_launch_grid must copy
 * that buffer (and the resources it points at) into Vortex device
 * memory, which means it needs to know each descriptor's byte offset
 * and kind. lavapipe's lowered NIR encodes exactly that:
 *
 *   load_ssbo  (desc_addr, off)         -- SSBO   at desc_addr
 *   store_ssbo (val, desc_addr, off)    -- SSBO   at desc_addr
 *   load_ubo   (cbuf_index, off)        -- AS     at cbuf[index]+off
 *
 * where desc_addr is load_const_buf_base_addr_lvp(1) optionally plus a
 * constant byte offset. */

/* Resolve an SSBO/UBO descriptor-address SSA value to its byte offset within its
 * descriptor buffer: load_const_buf_base_addr_lvp(idx) is offset 0, that plus a
 * constant is the constant. Also returns (via *cbuf_index) the constant-buffer
 * index the base came from — index N+1 is descriptor set N's blob, so a
 * descriptor in set >= 1 is relocated inside the right blob, not always set 0.
 * -1 if unrecognized. */
static int
vp_desc_addr_offset(nir_def *def, unsigned *cbuf_index)
{
   nir_instr *p = def->parent_instr;
   if (p->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *i = nir_instr_as_intrinsic(p);
      if (i->intrinsic == nir_intrinsic_load_const_buf_base_addr_lvp) {
         if (cbuf_index && nir_src_is_const(i->src[0]))
            *cbuf_index = (unsigned)nir_src_as_uint(i->src[0]);
         return 0;
      }
   } else if (p->type == nir_instr_type_alu) {
      nir_alu_instr *a = nir_instr_as_alu(p);
      if (a->op == nir_op_iadd) {
         for (int s = 0; s < 2; s++) {
            nir_instr *o = a->src[s].src.ssa->parent_instr;
            if (o->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(o)->intrinsic ==
                   nir_intrinsic_load_const_buf_base_addr_lvp &&
                nir_src_is_const(a->src[!s].src)) {
               nir_intrinsic_instr *b = nir_instr_as_intrinsic(o);
               if (cbuf_index && nir_src_is_const(b->src[0]))
                  *cbuf_index = (unsigned)nir_src_as_uint(b->src[0]);
               return (int)nir_src_as_uint(a->src[!s].src);
            }
         }
      }
   }
   return -1;
}

void
vp_scan_descriptors(struct nir_shader *nir,
                    struct vp_desc *out, unsigned *num_out)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(blk, impl) {
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            int off = -1;
            enum vp_desc_kind kind = VP_DESC_BUFFER;
            unsigned elem_bytes = 1;            /* SSBO: num_elements is bytes */
            unsigned cbuf_index = 1;            /* default set-0 blob (index 1) */
            switch (in->intrinsic) {
            case nir_intrinsic_load_ssbo:
               off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
               break;
            case nir_intrinsic_store_ssbo:
               off = vp_desc_addr_offset(in->src[1].ssa, &cbuf_index);
               break;
            case nir_intrinsic_ssbo_atomic:
            case nir_intrinsic_ssbo_atomic_swap:
               /* atomic RMW targets the same SSBO descriptor (src[0]); its
                * data pointer must be relocated like a load/store_ssbo. */
               off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
               break;
            case nir_intrinsic_image_load:
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_image_store:
            case nir_intrinsic_bindless_image_store:
               /* storage image: src[0] is the bindless descriptor address, in
                * the same const_buf_base+binding form as an SSBO. lp_jit_image
                * sizing (height*row_stride) is done in the launch relocation. */
               off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
               kind = VP_DESC_IMAGE;
               elem_bytes = 0;
               break;
            case nir_intrinsic_load_ubo:
               if (nir->info.stage == MESA_SHADER_FRAGMENT) {
                  /* A fragment UBO is a buffer descriptor reached via
                   * load_ubo(load_const_buf_base_addr_lvp(set+1)+binding, off) —
                   * relocate its lp_jit_buffer.ptr like an SSBO. A constant src[0]
                   * is a push-constant read (bound directly, no descriptor). A UBO
                   * descriptor stores num_elements in dwords (sizeof(float)). */
                  if (nir_src_is_const(in->src[0]))
                     continue;
                  off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
                  elem_bytes = 4;
               } else if (nir_intrinsic_range(in) == -1) {
                  /* compute: lavapipe reads the acceleration-structure handle as
                   * an UNBOUNDED ubo (range == -1), so only that read is an AS —
                   * src[0] is the constant cbuf index and off is 0. */
                  if (nir_src_is_const(in->src[1]))
                     off = (int)nir_src_as_uint(in->src[1]);
                  if (nir_src_is_const(in->src[0]))
                     cbuf_index = (unsigned)nir_src_as_uint(in->src[0]);
                  kind = VP_DESC_AS;
                  elem_bytes = 0;
               } else {
                  /* compute: a finite-range load_ubo is a data UBO (camera /
                   * scene uniforms) reached through load_ubo(const_buf_base(set)
                   * +binding, off) — the same descriptor-dereference form as the
                   * FS UBO. Relocate its lp_jit_buffer.ptr like an SSBO so the
                   * shader dereferences on-device. A constant src[0] is a push
                   * constant (bound inline, no descriptor): vp_desc_addr_offset
                   * returns -1 and it is skipped below. num_elements is dwords. */
                  off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
                  elem_bytes = 4;
               }
               break;
            default:
               continue;
            }
            if (off < 0)
               continue;
            /* Distinct by (cbuf_index, offset): the same byte offset in two sets
             * is two different descriptors. */
            bool dup = false;
            for (unsigned k = 0; k < n; k++)
               if (out[k].offset == (unsigned)off &&
                   out[k].cbuf_index == cbuf_index) { dup = true; break; }
            if (dup || n >= VP_MAX_DESCS)
               continue;
            out[n].offset     = (unsigned)off;
            out[n].cbuf_index = cbuf_index;
            out[n].kind       = kind;
            out[n].elem_bytes = elem_bytes;
            n++;
         }
      }
   }
   *num_out = n;
}

/* Bitmask of set_shader_buffers slots the shader reads as a CONSTANT-index
 * load_ssbo(imm slot, off). That form is unique to a raw shader buffer bound
 * outside set-0 (the RT trace-ray command buffer via lvp_push_internal_buffer);
 * ordinary SSBOs reach the data through a descriptor address (non-const src0).
 * vp_launch uses this to relocate the SBT shader-record device pointers that
 * live inside that command buffer. */
unsigned
vp_scan_trace_cmd_slots(struct nir_shader *nir)
{
   unsigned slots = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(blk, impl) {
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if ((in->intrinsic == nir_intrinsic_load_ssbo ||
                 in->intrinsic == nir_intrinsic_store_ssbo) &&
                nir_src_is_const(in->src[in->intrinsic == nir_intrinsic_store_ssbo ? 1 : 0])) {
               unsigned slot = (unsigned)nir_src_as_uint(
                  in->src[in->intrinsic == nir_intrinsic_store_ssbo ? 1 : 0]);
               if (slot < 32)
                  slots |= (1u << slot);
            }
         }
      }
   }
   return slots;
}
