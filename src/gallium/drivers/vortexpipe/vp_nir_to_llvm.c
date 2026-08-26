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
#include "gfx_sw_abi.h"      /* gfx_sw_texstate_t (logdim offset for auto-LOD) */
#include "VX_types.h"        /* VX_MEM_OM_BASE_ADDR */

#include <assert.h>          /* static_assert */
#include <stddef.h>          /* offsetof */
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
 * vec3e_t edges[3] (36B), then rast_attribs_t {z,r,g,b,a,u,v,rhw,w0..w5}, each a
 * rast_attrib_t {x,y,z} of fixed24 (12B). r/g/b/a/u/v/w0..w5 are the twelve
 * generic varying planes (declaration-order [u,v,r,g,b,a,w0..w5]); rhw is the
 * perspective 1/w plane. The w0..w5 planes are appended after rhw, so the
 * z/r/g/b/a/u/v/rhw offsets below are unchanged. At w==1 the premultiplied
 * varying planes equal the raw attributes and rhw is constant, so reading them
 * affinely is exact. The record pitch itself is VP_RAST_PRIM_STRIDE
 * (vp_nir_to_llvm.h), shared with the DCR writer in vp_raster.cpp. */
#define VP_RAST_ATTR_Z       36
#define VP_RAST_ATTR_R       48
#define VP_RAST_ATTR_G       60
#define VP_RAST_ATTR_B       72
#define VP_RAST_ATTR_A       84
#define VP_RAST_ATTR_U       96
#define VP_RAST_ATTR_V      108
#define VP_RAST_ATTR_RHW    120
#define VP_RAST_ATTR_W0     132
#define VP_RAST_ATTR_W1     144
#define VP_RAST_ATTR_W2     156
#define VP_RAST_ATTR_W3     168
#define VP_RAST_ATTR_W4     180
#define VP_RAST_ATTR_W5     192
/* The two per-primitive scalars that sit after the planes (rast_prim_t): the
 * winding EdgeEquation flipped away, which is gl_FrontFacing, and the factor
 * the rhw plane was premultiplied by, which gl_FragCoord.w must undo. */
/* i32 words of the fragment-shader input record: four 16-byte slots for the
 * declared inputs, plus one for the system values that have no variable of
 * their own (gl_FrontFacing). */
#define VP_FS_IN_WORDS 20

#define VP_RAST_PRIM_FACING    204
#define VP_RAST_PRIM_RHW_SCALE 208

/* Scalar planes available to carry varyings: [u,v,r,g,b,a] plus w0..w5. The
 * VS packs into them in declaration order and the FS reads back the same way,
 * so this bounds the two together. */
#define VP_RAST_MAX_PLANES   12

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
   LLVMValueRef   tex_f32_slot; /* float[4] the float texture ABI writes, or NULL */
   LLVMValueRef   tex_i32_slot; /* int32[4] the integer texture ABI writes, or NULL */
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
   LLVMValueRef   base_vertex;    /* i32 base vertex (gl_BaseVertex); added to
                                   * gl_VertexIndex only -- see the vid comment */
   LLVMValueRef   out_base;     /* iptr output-buffer device address */
   unsigned       out_stride;   /* bytes per output vertex record */
   LLVMValueRef   attr_table;   /* iptr addr of the {base,stride}[] table */
   /* fragment-shader state (is_fs only) */
   bool           is_fs;
   unsigned       fs_samples;   /* samples per pixel; 1 = single-sample */
   uint8_t        fs_bgra_mask; /* per-RT: blue-first, not red-first */
   LLVMValueRef   fs_in_base;   /* iptr interpolated-varyings area */
   LLVMValueRef   fs_out_base;  /* iptr output-colour area */
   struct vp_var  vars[VP_MAXV];
   unsigned       nvars;
   /* Per-unit SW routing (FS only). fs_texstate is fs_main's 3rd
    * param: a resident gfx_sw_texstate_t[] (per sampler stage). The SW sampler
    * reads all of it; the HW-tex FS reads logdim/filter to derive the mip LOD and
    * the min/mag tap. Zero (unused) for an untextured draw. */
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
   int            fs_depth_off;  /* byte offset of gl_FragDepth's slot; -1 if unwritten */
   /* gl_FragCoord is declared as a shader input and so owns a slot in the input
    * record, but it is a system value: the wrapper synthesises it rather than
    * interpolating a plane. -1 when the shader does not read it. */
   int            fs_pos_off;
   /* gl_FrontFacing arrives as an intrinsic, not a variable, so it has no slot
    * of its own. One is reserved past the declared inputs and filled by the
    * wrapper, which is where the primitive record is in scope. */
   unsigned       fs_sysval_off;
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

static LLVMTypeRef cty(struct vp_tr *t, unsigned bits);

/* Width-aware unary float intrinsic (llvm.<base>.f32/f64). `base` is the
 * intrinsic stem without the type suffix ("floor","ceil","trunc","round",
 * "roundeven","exp2","log2","sin","cos"). `fa` is a compute-typed float (from
 * as_float, always f32/f64 — never half), and the result is the same type; the
 * caller rounds back to f16 via from_float. The RISC-V backend lowers these to
 * FPU sequences or libcalls. */
static LLVMValueRef
emit_funary_intrin(struct vp_tr *t, const char *base, LLVMValueRef fa,
                   unsigned bits)
{
   const char  *suf = bits == 64 ? "f64" : "f32";
   char         nm[32];
   snprintf(nm, sizeof(nm), "llvm.%s.%s", base, suf);
   LLVMTypeRef  ft  = cty(t, bits);
   LLVMTypeRef  fnty = LLVMFunctionType(ft, &ft, 1, false);
   LLVMValueRef fn  = LLVMGetNamedFunction(t->mod, nm);
   if (!fn)
      fn = LLVMAddFunction(t->mod, nm, fnty);
   return LLVMBuildCall2(t->b, fnty, fn, &fa, 1, base);
}

/* copysign on a compute float: magnitude of `mag`, sign of `sgn`. Implemented
 * with integer bit ops (clear the sign bit of mag, OR in the sign bit of sgn)
 * so it lowers cleanly in divergent codegen. Restores the sign of a zero result
 * from rounding (floor(-0.0) = -0.0, trunc(-0.3) = -0.0, ...). */
static LLVMValueRef
emit_fcopysign(struct vp_tr *t, LLVMValueRef mag, LLVMValueRef sgn, unsigned bits)
{
   int          is64 = (bits == 64);
   LLVMTypeRef  it   = is64 ? t->i64 : t->i32;
   LLVMTypeRef  ft   = is64 ? t->f64 : t->f32;
   uint64_t     smask = is64 ? 0x8000000000000000ull : 0x80000000ull;
   LLVMValueRef mb   = LLVMBuildBitCast(t->b, mag, it, "");
   LLVMValueRef sb   = LLVMBuildBitCast(t->b, sgn, it, "");
   LLVMValueRef absv = LLVMBuildAnd(t->b, mb,
                          LLVMConstInt(it, ~smask, false), "");
   LLVMValueRef sv   = LLVMBuildAnd(t->b, sb,
                          LLVMConstInt(it, smask, false), "");
   return LLVMBuildBitCast(t->b, LLVMBuildOr(t->b, absv, sv, ""), ft, "");
}

/* trunc / floor / ceil, synthesized from fptosi/sitofp. The Vortex LLVM backend
 * cannot lower llvm.trunc/floor/ceil/round in divergent codegen (they expand to
 * a rounding pseudo the divergent custom-inserter rejects), but int<->float
 * converts and float selects lower fine. A value with |x| >= 2^24 (f32) / 2^53
 * (f64) is already integral, so the round-trip (which would overflow the integer
 * range) is guarded off and x passes through unchanged. `fa` is a compute float
 * (f32/f64); the result is the same type. */
static LLVMValueRef
emit_ftrunc(struct vp_tr *t, LLVMValueRef fa, unsigned bits)
{
   LLVMTypeRef  ft   = cty(t, bits);
   int          is64 = (bits == 64);
   LLVMTypeRef  it   = is64 ? t->i64 : t->i32;
   LLVMValueRef iv   = LLVMBuildFPToSI(t->b, fa, it, "");
   LLVMValueRef rv   = LLVMBuildSIToFP(t->b, iv, ft, "");
   LLVMValueRef ax   = emit_funary_intrin(t, "fabs", fa, bits);
   LLVMValueRef big  = LLVMBuildFCmp(t->b, LLVMRealOGE, ax,
                          LLVMConstReal(ft, is64 ? 9007199254740992.0
                                                 : 16777216.0), "");
   return emit_fcopysign(t, LLVMBuildSelect(t->b, big, fa, rv, "ftrunc"),
                         fa, bits);
}
static LLVMValueRef
emit_ffloor(struct vp_tr *t, LLVMValueRef fa, unsigned bits)
{
   LLVMTypeRef  ft = cty(t, bits);
   LLVMValueRef tr = emit_ftrunc(t, fa, bits);
   /* trunc rounds toward zero; for x<tr (negative non-integers) subtract one. */
   LLVMValueRef gt = LLVMBuildFCmp(t->b, LLVMRealOGT, tr, fa, "");
   LLVMValueRef m1 = LLVMBuildFSub(t->b, tr, LLVMConstReal(ft, 1.0), "");
   return emit_fcopysign(t, LLVMBuildSelect(t->b, gt, m1, tr, "ffloor"),
                         fa, bits);
}
static LLVMValueRef
emit_fceil(struct vp_tr *t, LLVMValueRef fa, unsigned bits)
{
   LLVMTypeRef  ft = cty(t, bits);
   LLVMValueRef tr = emit_ftrunc(t, fa, bits);
   LLVMValueRef lt = LLVMBuildFCmp(t->b, LLVMRealOLT, tr, fa, "");
   LLVMValueRef p1 = LLVMBuildFAdd(t->b, tr, LLVMConstReal(ft, 1.0), "");
   return emit_fcopysign(t, LLVMBuildSelect(t->b, lt, p1, tr, "fceil"),
                         fa, bits);
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

/* llvm.bitreverse.i32 (reverse bit order) -> Vortex has no native op; the RISC-V
 * backend expands it, but declaring the intrinsic keeps codegen one node wide. */
static LLVMValueRef
emit_bitreverse(struct vp_tr *t, LLVMValueRef v)
{
   LLVMTypeRef  fty = LLVMFunctionType(t->i32, &t->i32, 1, false);
   LLVMValueRef fn  = LLVMGetNamedFunction(t->mod, "llvm.bitreverse.i32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "llvm.bitreverse.i32", fty);
   return LLVMBuildCall2(t->b, fty, fn, &v, 1, "bitreverse");
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
 * bval=src, cval=0x3f, mask=0 -> 0xFC0 | src. The shfl instruction is i32-typed,
 * so a sub-i32 value (a bool arrives as i1 from the subgroup broadcast / shuffle
 * / quad lowerings) is widened for transport and narrowed back — the inline-asm
 * call would otherwise fail IR verification. Type-preserving, so every caller
 * (shuffle, read_invocation, vote_ieq, ...) sees the value's original type. */
static LLVMValueRef
emit_shfl_idx(struct vp_tr *t, LLVMValueRef value, LLVMValueRef src)
{
   LLVMTypeRef  vt  = LLVMTypeOf(value);
   LLVMValueRef v32 = (vt == t->i32) ? value
                    : LLVMBuildZExt(t->b, value, t->i32, "shfl_w");
   LLVMValueRef bc  = LLVMBuildOr(t->b, LLVMConstInt(t->i32, 0xFC0u, false),
                                  src, "shflidx_bc");
   LLVMValueRef r   = emit_shfl(t, 7, v32, bc);
   return (vt == t->i32) ? r : LLVMBuildTrunc(t->b, r, vt, "shfl_n");
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

/* The identity element of a subgroup reduction op, as the i32 lane bit pattern.
 * Used to seed the lane-0 slot of an exclusive scan. Returns NULL (clears ok)
 * for an op with no identity mapping. */
static LLVMValueRef
emit_scan_identity(struct vp_tr *t, nir_op rop)
{
   switch (rop) {
   case nir_op_iadd: case nir_op_ior: case nir_op_ixor:
      return LLVMConstInt(t->i32, 0, false);
   case nir_op_imul:
      return LLVMConstInt(t->i32, 1, false);
   case nir_op_iand:
      return LLVMConstInt(t->i32, 0xFFFFFFFFu, false);
   case nir_op_umin:
      return LLVMConstInt(t->i32, 0xFFFFFFFFu, false);
   case nir_op_umax:
      return LLVMConstInt(t->i32, 0, false);
   case nir_op_imin:
      return LLVMConstInt(t->i32, 0x7FFFFFFFu, false);
   case nir_op_imax:
      return LLVMConstInt(t->i32, 0x80000000u, false);
   case nir_op_fadd:
      return LLVMConstInt(t->i32, 0x00000000u, false);          /* +0.0f */
   case nir_op_fmul:
      return LLVMConstInt(t->i32, 0x3F800000u, false);          /* 1.0f  */
   case nir_op_fmin:
      return LLVMConstInt(t->i32, 0x7F800000u, false);          /* +inf  */
   case nir_op_fmax:
      return LLVMConstInt(t->i32, 0xFF800000u, false);          /* -inf  */
   default:
      mesa_logw("vortexpipe: subgroup scan op %d has no identity", rop);
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

/* Inclusive prefix scan of `val` across the warp (subgroup) under reduction op
 * `rop`: Hillis-Steele over shfl_up. Three guards decide whether a lane combines
 * its neighbour at step d:
 *   - in range: the source lane (lane-d) must exist (lane >= d);
 *   - source active: the source lane must be an active invocation;
 *   - same cluster: for a clustered reduction, the source must not cross the
 *     cluster boundary.
 * The second matters under divergence — a shfl from an inactive lane returns the
 * reader's OWN value, so combining it unconditionally would double-count for a
 * non-idempotent op (add/mul/xor); an active source at a further power-of-two
 * distance is still picked up in a later step. Unrolled to cover NT up to 32.
 *
 * `cluster` is the width of the lane group a clustered reduction folds over, or
 * 0 for the whole warp. It is a power of two, so a lane's offset within its
 * cluster is `lane & (cluster-1)` and a source at distance d stays inside the
 * cluster exactly when that offset is at least d. No lane reaches past its own
 * cluster, so the scan stops at d == cluster.
 *
 * Returns the inclusive result (NULL + clears ok on an unmapped op). */
static LLVMValueRef
emit_subgroup_incl_scan(struct vp_tr *t, nir_op rop, LLVMValueRef val,
                        unsigned cluster)
{
   LLVMValueRef acc    = val;
   LLVMValueRef lane   = emit_csr_read(t, VX_CSR_THREAD_ID, "lane");
   LLVMValueRef active = emit_ballot(t, LLVMConstInt(t->i32, 1, false));
   LLVMValueRef one    = LLVMConstInt(t->i32, 1, false);
   /* Offset within the cluster; for the unclustered scan this is just the lane. */
   LLVMValueRef pos    = cluster
      ? LLVMBuildAnd(t->b, lane, LLVMConstInt(t->i32, cluster - 1u, false), "cpos")
      : lane;
   unsigned span = cluster ? cluster : 32u;
   for (unsigned d = 1; d < span && d <= 16u; d <<= 1) {
      LLVMValueRef dc   = LLVMConstInt(t->i32, d, false);
      LLVMValueRef nbr  = emit_shfl_up(t, acc, d);
      LLVMValueRef comb = emit_scan_combine(t, rop, acc, nbr);
      if (!t->ok)
         return NULL;
      LLVMValueRef inrange = LLVMBuildICmp(t->b, LLVMIntUGE, pos, dc, "");
      LLVMValueRef srcbit  = LLVMBuildAnd(t->b,
         LLVMBuildLShr(t->b, active, LLVMBuildSub(t->b, lane, dc, ""), ""),
         one, "");
      LLVMValueRef srcact  = LLVMBuildICmp(t->b, LLVMIntNE, srcbit,
         LLVMConstInt(t->i32, 0, false), "");
      LLVMValueRef doit = LLVMBuildAnd(t->b, inrange, srcact, "");
      acc = LLVMBuildSelect(t->b, doit, comb, acc, "scan");
   }
   return acc;
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
   LLVMValueRef call = LLVMBuildCall2(t->b, fnty, ia, a, 2, "");
   /* Mark the barrier convergent: it must execute with the CTA's lanes
    * converged, so mid-level CFG passes (SimplifyCFG merging two branches on the
    * same divergent condition) must not duplicate it into, or sink it out of,
    * divergent control flow. Without this the barrier is placed inside a
    * divergent if and the workgroup never actually synchronizes. */
   unsigned conv = LLVMGetEnumAttributeKindForName("convergent", 10);
   LLVMAddCallSiteAttribute(call, LLVMAttributeFunctionIndex,
                            LLVMCreateEnumAttribute(t->ctx, conv, 0));
}

/* VS vertex-input fetch: the device address of attribute `loc` for the
 * current vertex. The attribute table (arg slot 1) holds, per VS input
 * driver_location, a { base, stride, divisor } entry; the attribute lives at
 * base + index*stride (vp_launch_vs builds the table). The index is the vertex
 * id, or for a non-zero divisor the instance id divided by it, so an
 * instance-rate attribute advances once every `divisor` instances rather than
 * once per instance. The fields are i32 on the wire even on rv64 (the table
 * layout is the same), so widen base to iptr before address arithmetic. */
static LLVMValueRef
emit_vs_attr_addr(struct vp_tr *t, unsigned loc)
{
   LLVMValueRef ent = LLVMBuildAdd(t->b, t->attr_table,
      vp_iptr_const(t, loc * VP_ATTR_ENTRY_BYTES), "");
   LLVMValueRef base = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildIntToPtr(t->b, ent, t->ptr, ""), "attrbase");
   LLVMValueRef stride = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildIntToPtr(t->b,
         LLVMBuildAdd(t->b, ent, vp_iptr_const(t, 4), ""),
         t->ptr, ""), "attrstride");
   LLVMValueRef divisor = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildIntToPtr(t->b,
         LLVMBuildAdd(t->b, ent, vp_iptr_const(t, 8), ""),
         t->ptr, ""), "attrdiv");
   LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
   LLVMValueRef per_vertex = LLVMBuildICmp(t->b, LLVMIntEQ, divisor, zero, "");
   /* The divide is guarded to 1 on the per-vertex path: it is dead there, but
    * a udiv by the zero that marks that path is poison, not an ignored value. */
   LLVMValueRef safe_div = LLVMBuildSelect(t->b, per_vertex,
      LLVMConstInt(t->i32, 1, false), divisor, "");
   LLVMValueRef inst = t->instance ? t->instance : zero;
   LLVMValueRef index = LLVMBuildSelect(t->b, per_vertex, t->vid,
      LLVMBuildUDiv(t->b, inst, safe_div, ""), "attridx");
   LLVMValueRef offset = LLVMBuildMul(t->b, index, stride, "");
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
fty(struct vp_tr *t, unsigned bits) {
   return bits == 64 ? t->f64
        : bits == 16 ? LLVMHalfTypeInContext(t->ctx)
                     : t->f32;
}
static LLVMTypeRef
ity(struct vp_tr *t, unsigned bits) { return bits == 64 ? t->i64 : t->i32; }

/* Compute float type. Vortex has an f32/f64 FPU but NO half hardware (Zfh is
 * off), and the RISC-V backend cannot lower `half` ops in Vortex's divergent
 * codegen. So fp16 is computed in f32 exactly as a GPU without native f16 does:
 * promote to f32, do the op, round the single result back to f16. cty() is the
 * type the arithmetic runs in (f32 for 16- and 32-bit, f64 for 64-bit). */
static LLVMTypeRef
cty(struct vp_tr *t, unsigned bits) { return bits == 64 ? t->f64 : t->f32; }

/* Float ALU operands live in the lane as their IEEE bit pattern in an integer:
 * i32 for 16- and 32-bit (a 16-bit value occupies the low half), i64 for 64-bit.
 * as_float() recovers the compute-typed float for an op; from_float() packs a
 * result back. A 16-bit float is unpacked as its half bit pattern then widened
 * to f32 for the arithmetic; from_float() truncates the f32 result back to f16
 * (one correctly-rounded step), so denormals and rounding stay f16-accurate
 * without ever emitting a native `half` operation. */
static LLVMValueRef
as_float(struct vp_tr *t, LLVMValueRef laneval, unsigned bits) {
   if (bits == 16) {
      LLVMValueRef h = LLVMBuildTrunc(t->b, laneval,
                                      LLVMInt16TypeInContext(t->ctx), "");
      LLVMValueRef hf = LLVMBuildBitCast(t->b, h,
                                         LLVMHalfTypeInContext(t->ctx), "");
      return LLVMBuildFPExt(t->b, hf, t->f32, "");
   }
   return LLVMBuildBitCast(t->b, laneval, fty(t, bits), "");
}
static LLVMValueRef
from_float(struct vp_tr *t, LLVMValueRef fval, unsigned bits) {
   if (bits == 16) {
      LLVMValueRef hf = LLVMBuildFPTrunc(t->b, fval,
                                         LLVMHalfTypeInContext(t->ctx), "");
      LLVMValueRef h  = LLVMBuildBitCast(t->b, hf,
                                         LLVMInt16TypeInContext(t->ctx), "");
      return LLVMBuildZExt(t->b, h, t->i32, "");
   }
   return LLVMBuildBitCast(t->b, fval, ity(t, bits), "");
}

/* Normalize a `bits`-wide integer held in the i32 lane before a signed (sgn) or
 * unsigned read. Sub-32-bit arithmetic (iadd/imul/ishl/...) is done in i32 and
 * can leave dirty high bits, so a signed consumer must sign-extend and an
 * unsigned consumer must mask from the true width. No-op for >=32-bit sources
 * or values already at their native LLVM width (i64). */
static LLVMValueRef
norm_int_src(struct vp_tr *t, LLVMValueRef v, unsigned bits, bool sgn) {
   if (bits >= 32 || LLVMTypeOf(v) != t->i32)
      return v;
   if (sgn) {
      unsigned sh = 32 - bits;
      return LLVMBuildAShr(t->b,
                LLVMBuildShl(t->b, v, LLVMConstInt(t->i32, sh, false), ""),
                LLVMConstInt(t->i32, sh, false), "sext");
   }
   return LLVMBuildAnd(t->b, v, LLVMConstInt(t->i32, (1u << bits) - 1, false),
                       "zext");
}

/* Exact-width integer type for a memory access element (i8/i16/i32/i64).
 * Distinct from the lane ABI type ity() (only i32/i64): a sub-32-bit access
 * must touch exactly bit_size/8 bytes. Loading/storing at the wider lane type
 * would over-read or over-write the neighbouring element — and a widened store
 * that straddles a 64B device block corrupts memory outright. */
static LLVMTypeRef
mem_ty(struct vp_tr *t, unsigned bits) {
   return LLVMIntTypeInContext(t->ctx, bits);
}

/* Load a bit_size-wide integer from p, widened (zero-extended) to the lane ABI
 * type so downstream ops see a normal i32/i64 SSA value. */
static LLVMValueRef
vp_load_mem(struct vp_tr *t, unsigned bits, LLVMValueRef p, const char *nm) {
   LLVMValueRef raw = LLVMBuildLoad2(t->b, mem_ty(t, bits), p, nm);
   if (bits < 32)
      return LLVMBuildZExt(t->b, raw, ity(t, bits), "");
   return raw;
}

/* Truncate a lane ABI value to the bit_size-wide memory element before storing,
 * so the backend emits a byte/half store rather than a full word. */
static LLVMValueRef
vp_store_mem_val(struct vp_tr *t, unsigned bits, LLVMValueRef v) {
   if (bits < 32)
      return LLVMBuildTrunc(t->b, v, mem_ty(t, bits), "");
   return v;
}

/* quad-derivative helper (defined with the other Vortex intrinsics, below). */
static LLVMValueRef emit_quad_deriv(struct vp_tr *t, LLVMValueRef value,
                                    unsigned dir);

/* i32 load from an iptr address (defined with the fragment wrapper, below). */
static LLVMValueRef emit_load_i32(struct vp_tr *t, LLVMValueRef addr);

/* An identity the optimizer cannot see through: an empty asm whose output is
 * tied to its input, so it costs no instruction and yields a value with no
 * known provenance. */
static LLVMValueRef
emit_opaque(struct vp_tr *t, LLVMValueRef v)
{
   LLVMTypeRef ty = LLVMTypeOf(v);
   LLVMTypeRef args[1] = { ty };
   LLVMTypeRef fnty = LLVMFunctionType(ty, args, 1, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, "", 0, "=r,0", 4,
                                      /*HasSideEffects*/ false, false,
                                      LLVMInlineAsmDialectATT, false);
   LLVMValueRef a[1] = { v };
   return LLVMBuildCall2(t->b, fnty, ia, a, 1, "opaque");
}

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
         unsigned bs = alu->def.bit_size;
         LLVMValueRef a = as_float(t, alu_src(t, alu, 0, c), bs);
         LLVMValueRef b = as_float(t, alu_src(t, alu, 1, c), bs);
         LLVMValueRef res = alu->op == nir_op_fadd
            ? LLVMBuildFAdd(t->b, a, b, "fadd")
            : LLVMBuildFMul(t->b, a, b, "fmul");
         r = from_float(t, res, bs);
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
         /* Normalize sub-32-bit operands by width: signed compares sign-extend,
          * unsigned/equality mask. Sub-32-bit arithmetic runs in the i32 lane
          * and leaves dirty high bits (e.g. a 16-bit iadd that overflowed in a
          * saturating-dot), which would corrupt an i32 icmp. */
         unsigned sbs = nir_src_bit_size(alu->src[0].src);
         bool sgn = (alu->op == nir_op_ilt || alu->op == nir_op_ige);
         LLVMValueRef ca = norm_int_src(t, alu_src(t, alu, 0, c), sbs, sgn);
         LLVMValueRef cb = norm_int_src(t, alu_src(t, alu, 1, c), sbs, sgn);
         LLVMValueRef cmp = LLVMBuildICmp(t->b, p, ca, cb, "icmp");
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
         unsigned sbs = nir_src_bit_size(alu->src[0].src);
         LLVMValueRef a = as_float(t, alu_src(t, alu, 0, c), sbs);
         LLVMValueRef b = as_float(t, alu_src(t, alu, 1, c), sbs);
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
      case nir_op_i2f16: case nir_op_i2f32: case nir_op_i2f64: {
         unsigned db  = alu->def.bit_size;
         unsigned sbs = nir_src_bit_size(alu->src[0].src);
         LLVMValueRef s = norm_int_src(t, alu_src(t, alu, 0, c), sbs, true);
         r = from_float(t, LLVMBuildSIToFP(t->b, s, cty(t, db), ""), db);
         break;
      }
      case nir_op_u2f16: case nir_op_u2f32: case nir_op_u2f64: {
         unsigned db  = alu->def.bit_size;
         unsigned sbs = nir_src_bit_size(alu->src[0].src);
         LLVMValueRef s = norm_int_src(t, alu_src(t, alu, 0, c), sbs, false);
         r = from_float(t, LLVMBuildUIToFP(t->b, s, cty(t, db), ""), db);
         break;
      }
      case nir_op_f2i8:  case nir_op_f2i16:
      case nir_op_f2i32: case nir_op_f2i64: {
         unsigned db = alu->def.bit_size;
         LLVMValueRef v = LLVMBuildFPToSI(t->b,
            as_float(t, alu_src(t, alu, 0, c), nir_src_bit_size(alu->src[0].src)),
            LLVMIntTypeInContext(t->ctx, db), "");
         r = db < 32 ? LLVMBuildZExt(t->b, v, t->i32, "") : v;
         break;
      }
      case nir_op_f2u8:  case nir_op_f2u16:
      case nir_op_f2u32: case nir_op_f2u64: {
         unsigned db = alu->def.bit_size;
         LLVMValueRef v = LLVMBuildFPToUI(t->b,
            as_float(t, alu_src(t, alu, 0, c), nir_src_bit_size(alu->src[0].src)),
            LLVMIntTypeInContext(t->ctx, db), "");
         r = db < 32 ? LLVMBuildZExt(t->b, v, t->i32, "") : v;
         break;
      }
      /* float neg / div / min / max / abs / fused-multiply-add / sign */
      case nir_op_fneg: case nir_op_fdiv:
      case nir_op_fmin: case nir_op_fmax:
      case nir_op_fabs: case nir_op_ffma: case nir_op_fsign:
      case nir_op_fsqrt: case nir_op_frsq: {
         unsigned     bs = alu->def.bit_size;
         LLVMTypeRef  ft = cty(t, bs);
         LLVMValueRef fa = as_float(t, alu_src(t, alu, 0, c), bs);
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
            /* llvm.fabs clears the sign bit outright; a select(x<0,-x,x) would
             * mishandle -0.0 (-0.0 < 0.0 is false, so it would keep the sign). */
            res = emit_funary_intrin(t, "fabs", fa, bs);
         } else if (alu->op == nir_op_fsign) {
            res = LLVMBuildSelect(t->b,
               LLVMBuildFCmp(t->b, LLVMRealOGT, fa, z, ""),
               LLVMConstReal(ft, 1.0),
               LLVMBuildSelect(t->b,
                  LLVMBuildFCmp(t->b, LLVMRealOLT, fa, z, ""),
                  LLVMConstReal(ft, -1.0), z, ""), "");
         } else {
            LLVMValueRef fb = as_float(t, alu_src(t, alu, 1, c), bs);
            if (alu->op == nir_op_fdiv)
               res = LLVMBuildFDiv(t->b, fa, fb, "");
            else if (alu->op == nir_op_fmin)
               res = LLVMBuildSelect(t->b,
                  LLVMBuildFCmp(t->b, LLVMRealOLT, fa, fb, ""), fa, fb, "");
            else if (alu->op == nir_op_fmax)
               res = LLVMBuildSelect(t->b,
                  LLVMBuildFCmp(t->b, LLVMRealOGT, fa, fb, ""), fa, fb, "");
            else { /* ffma */
               LLVMValueRef fc = as_float(t, alu_src(t, alu, 2, c), bs);
               res = LLVMBuildFAdd(t->b, LLVMBuildFMul(t->b, fa, fb, ""),
                                   fc, "");
            }
         }
         r = from_float(t, res, bs);
         break;
      }
      /* float width conversions — branch on the actual LLVM operand width so
       * the bitcasts stay legal even when the source value was materialized at
       * a different width than its NIR bit_size (a same-width convert is the
       * identity on the bit pattern). */
      /* Float width conversions. Drive off the NIR source/dest bit sizes (not the
       * LLVM operand type — a 16-bit float shares the i32 lane rep with f32) so
       * f2f16/f2f32/f2f64 pick the right FPExt/FPTrunc/identity in every case. */
      /* f2f16_rtne is f2f16 with an explicit round-to-nearest-even request;
       * from_float's FPTrunc-to-half already rounds RTNE under the default mode,
       * so it shares the f2f16 path. SPIR-V OpQuantizeToF16 lowers to this. */
      case nir_op_f2f16_rtne:
      case nir_op_f2f16:
      case nir_op_f2f32:
      case nir_op_f2f64: {
         unsigned sb = nir_src_bit_size(alu->src[0].src);
         unsigned db = alu->def.bit_size;
         /* as_float yields the compute type (f32 for 16/32, f64 for 64); convert
          * between compute types, then from_float rounds to f16 when db==16. An
          * f16<->f32 convert is a no-op here — the round happens in from_float. */
         LLVMValueRef s = as_float(t, alu_src(t, alu, 0, c), sb);
         LLVMValueRef res =
              (cty(t, db) == cty(t, sb)) ? s
            : (db == 64) ? LLVMBuildFPExt(t->b, s, t->f64, "f2f")
                         : LLVMBuildFPTrunc(t->b, s, t->f32, "f2f");
         r = from_float(t, res, db);
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
      /* bool -> int / float (any result width). Floats round through from_float
       * so b2f16 lands as a valid f16 bit pattern; sub-32-bit ints are held in
       * the i32 lane. */
      case nir_op_b2i8: case nir_op_b2i16:
      case nir_op_b2i32: case nir_op_b2i64:
      case nir_op_b2f16: case nir_op_b2f32: case nir_op_b2f64: {
         unsigned db = alu->def.bit_size;
         LLVMValueRef v = alu_src(t, alu, 0, c);
         LLVMValueRef cond = (LLVMTypeOf(v) == LLVMInt1TypeInContext(t->ctx))
            ? v
            : LLVMBuildICmp(t->b, LLVMIntNE, v,
                            LLVMConstInt(LLVMTypeOf(v), 0, false), "");
         int is_float = (alu->op == nir_op_b2f16 || alu->op == nir_op_b2f32 ||
                         alu->op == nir_op_b2f64);
         if (is_float) {
            LLVMTypeRef ft = cty(t, db);
            r = from_float(t, LLVMBuildSelect(t->b, cond,
                   LLVMConstReal(ft, 1.0), LLVMConstReal(ft, 0.0), ""), db);
         } else {
            LLVMTypeRef it = (db == 64) ? t->i64 : t->i32;
            r = LLVMBuildSelect(t->b, cond, LLVMConstInt(it, 1, false),
                                LLVMConstInt(it, 0, false), "");
         }
         break;
      }
      /* width conversions. Widening normalizes the source from its true bit size
       * first (sign- or zero-extend), because sub-32-bit arithmetic runs in i32
       * and can leave dirty high bits (e.g. a 16-bit iadd that wrapped). */
      case nir_op_u2u64:
         r = LLVMBuildZExt(t->b,
               norm_int_src(t, alu_src(t, alu, 0, c),
                            nir_src_bit_size(alu->src[0].src), false),
               t->i64, "u2u64");
         break;
      case nir_op_i2i64:
         r = LLVMBuildSExt(t->b,
               norm_int_src(t, alu_src(t, alu, 0, c),
                            nir_src_bit_size(alu->src[0].src), true),
               t->i64, "i2i64");
         break;
      case nir_op_u2u32: case nir_op_i2i32: {
         unsigned sbs = nir_src_bit_size(alu->src[0].src);
         LLVMValueRef s = alu_src(t, alu, 0, c);
         /* 64-bit source -> 32: truncate; otherwise normalize from sbs. */
         r = (LLVMTypeOf(s) == t->i64)
            ? LLVMBuildTrunc(t->b, s, t->i32, "")
            : norm_int_src(t, s, sbs, alu->op == nir_op_i2i32);
         break;
      }
      /* Sub-32-bit int conversions. The lane ABI keeps every integer in an i32
       * (ity() never yields i16/i8). Both the SOURCE and DEST widths matter: the
       * source is first normalized from its own bit size (an i8 source may sit in
       * the lane zero-extended, so a signed i2i16 must sign-extend from bit 8, not
       * bit 16), then the result is produced at the destination width. Used by
       * packed 8/16-bit math (16bit_storage add-immediate, integer-dot-product). */
      case nir_op_u2u16: case nir_op_u2u8: {
         unsigned sbs = nir_src_bit_size(alu->src[0].src);
         unsigned dbs = (alu->op == nir_op_u2u16) ? 16u : 8u;
         LLVMValueRef s = alu_src(t, alu, 0, c);
         if (LLVMTypeOf(s) == t->i64)
            s = LLVMBuildTrunc(t->b, s, t->i32, "");
         s = norm_int_src(t, s, sbs, false);           /* clean unsigned source */
         r = LLVMBuildAnd(t->b, s,
                LLVMConstInt(t->i32, (1u << dbs) - 1u, false), "u2u");
         break;
      }
      case nir_op_i2i16: case nir_op_i2i8: {
         unsigned sbs = nir_src_bit_size(alu->src[0].src);
         unsigned dbs = (alu->op == nir_op_i2i16) ? 16u : 8u;
         LLVMValueRef s = alu_src(t, alu, 0, c);
         if (LLVMTypeOf(s) == t->i64)
            s = LLVMBuildTrunc(t->b, s, t->i32, "");
         s = norm_int_src(t, s, sbs, true);            /* signed value from source */
         unsigned sh = 32u - dbs;                       /* canonicalize as dbs-bit signed */
         r = LLVMBuildAShr(t->b,
                LLVMBuildShl(t->b, s, LLVMConstInt(t->i32, sh, false), ""),
                LLVMConstInt(t->i32, sh, false), "i2i");
         break;
      }
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
      case nir_op_frcp: {
         unsigned bs = alu->def.bit_size;
         r = from_float(t,
            LLVMBuildFDiv(t->b, LLVMConstReal(cty(t, bs), 1.0),
               as_float(t, alu_src(t, alu, 0, c), bs), ""), bs);
         break;
      }
      /* ufind_msb: index of the highest set bit, -1 when the source is 0.
       * ctlz(0)=32, so 31-ctlz naturally yields -1 for a zero source. */
      case nir_op_ufind_msb:
         r = LLVMBuildSub(t->b, LLVMConstInt(t->i32, 31, false),
                          emit_ctlz(t, alu_src(t, alu, 0, c)), "ufind_msb");
         break;
      case nir_op_bit_count:
         r = emit_ctpop(t, alu_src(t, alu, 0, c));
         break;
      case nir_op_bitfield_reverse:
         r = emit_bitreverse(t, alu_src(t, alu, 0, c));
         break;
      /* find_lsb: index of the lowest set bit, -1 when the source is 0. cttz(0)
       * is 32 (zero-is-not-poison), so a zero source is mapped to -1 explicitly. */
      case nir_op_find_lsb: {
         LLVMValueRef v  = alu_src(t, alu, 0, c);
         LLVMValueRef tz = emit_cttz(t, v);
         r = LLVMBuildSelect(t->b,
             LLVMBuildICmp(t->b, LLVMIntEQ, v, LLVMConstInt(t->i32, 0, false), ""),
             LLVMConstInt(t->i32, -1, true), tz, "find_lsb");
         break;
      }
      case nir_op_fpow: {
         unsigned bs = alu->def.bit_size;
         LLVMValueRef x = as_float(t, alu_src(t, alu, 0, c), bs);
         LLVMValueRef y = as_float(t, alu_src(t, alu, 1, c), bs);
         r = from_float(t, emit_fpow(t, x, y), bs);
         break;
      }
      /* Unary float rounding + transcendentals. floor/ceil/trunc are synthesized
       * (see emit_ftrunc) because their intrinsics don't lower in divergent
       * codegen; fround_even maps to nearbyint (identical under the default
       * round-to-nearest-even mode); exp2/log2/sin/cos use LLVM intrinsics that
       * lower to libcalls. */
      case nir_op_ffloor: case nir_op_fceil: case nir_op_ftrunc:
      case nir_op_fround_even:
      case nir_op_fexp2: case nir_op_flog2:
      case nir_op_fsin: case nir_op_fcos: {
         unsigned     bs = alu->def.bit_size;
         LLVMValueRef fa = as_float(t, alu_src(t, alu, 0, c), bs);
         LLVMValueRef res;
         switch (alu->op) {
         case nir_op_ffloor:      res = emit_ffloor(t, fa, bs); break;
         case nir_op_fceil:       res = emit_fceil(t, fa, bs);  break;
         case nir_op_ftrunc:      res = emit_ftrunc(t, fa, bs); break;
         case nir_op_fround_even: res = emit_funary_intrin(t, "nearbyint", fa, bs); break;
         case nir_op_fexp2:       res = emit_funary_intrin(t, "exp2", fa, bs); break;
         case nir_op_flog2:       res = emit_funary_intrin(t, "log2", fa, bs); break;
         case nir_op_fsin:        res = emit_funary_intrin(t, "sin",  fa, bs); break;
         default:                 res = emit_funary_intrin(t, "cos",  fa, bs); break;
         }
         r = from_float(t, res, bs);
         break;
      }
      /* ffract: x - floor(x). */
      case nir_op_ffract: {
         unsigned     bs = alu->def.bit_size;
         LLVMValueRef fa = as_float(t, alu_src(t, alu, 0, c), bs);
         LLVMValueRef fl = emit_ffloor(t, fa, bs);
         r = from_float(t, LLVMBuildFSub(t->b, fa, fl, "ffract"), bs);
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

/* RTU emit helpers (defined after the TEX emitters, below). */
static LLVMValueRef emit_vx_rt_get(struct vp_tr *t, unsigned slot, LLVMValueRef status);
static LLVMValueRef emit_vx_rt_wtrace(struct vp_tr *t, LLVMValueRef scene,
                                      LLVMValueRef flags_cull,
                                      LLVMValueRef ray[8]);
static LLVMValueRef emit_vx_rt_wait(struct vp_tr *t, LLVMValueRef handle);
static void         emit_vx_rt_cb_ret(struct vp_tr *t, LLVMValueRef action);
static void         emit_vx_rt_continue(struct vp_tr *t, LLVMValueRef action,
                                        LLVMValueRef tval, LLVMValueRef attr);

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

/* f32 in [0,1] -> unsigned-normalized integer of `bits` width (round-to-nearest),
 * value given as its i32 bit pattern. Covers the 8-bit channels of RGBA8_UNORM and
 * the 10/2-bit channels of R10G10B10A2_UNORM. */
static LLVMValueRef
f32bits_to_unorm(struct vp_tr *t, LLVMValueRef vi, unsigned bits)
{
   double maxv = (double)((1u << bits) - 1u);
   LLVMValueRef f   = LLVMBuildBitCast(t->b, vi, t->f32, "");
   LLVMValueRef z   = LLVMConstReal(t->f32, 0.0);
   LLVMValueRef one = LLVMConstReal(t->f32, 1.0);
   f = LLVMBuildSelect(t->b, LLVMBuildFCmp(t->b, LLVMRealOGT, f, z, ""), f, z, "");
   f = LLVMBuildSelect(t->b, LLVMBuildFCmp(t->b, LLVMRealOLT, f, one, ""), f, one, "");
   LLVMValueRef s = LLVMBuildFAdd(t->b,
      LLVMBuildFMul(t->b, f, LLVMConstReal(t->f32, maxv), ""),
      LLVMConstReal(t->f32, 0.5), "");
   LLVMValueRef b = LLVMBuildFPToUI(t->b, s, t->i32, "");
   return LLVMBuildAnd(t->b, b, LLVMConstInt(t->i32, (1u << bits) - 1u, false), "");
}

/* `bits`-wide unsigned-normalized field (low bits of `word`) -> f32 in [0,1],
 * returned as its i32 bit pattern. */
static LLVMValueRef
unorm_to_f32bits(struct vp_tr *t, LLVMValueRef word, unsigned bits)
{
   LLVMValueRef m = LLVMBuildAnd(t->b, word,
      LLVMConstInt(t->i32, (1u << bits) - 1u, false), "");
   LLVMValueRef f = LLVMBuildFMul(t->b, LLVMBuildUIToFP(t->b, m, t->f32, ""),
      LLVMConstReal(t->f32, 1.0 / (double)((1u << bits) - 1u)), "");
   return LLVMBuildBitCast(t->b, f, t->i32, "");
}

/* f32 (as i32 bits) -> unsigned float, 5-bit exponent + `mbits` mantissa, no sign
 * (the packed floats of R11G11B10: R,G use mbits=6, B uses mbits=5). These share
 * the IEEE half's 5-bit exponent, so the value is relayed through a half: fptrunc
 * supplies correct rounding/denormals/overflow, and the field is the top (5+mbits)
 * bits of the half with the always-zero sign dropped. */
static LLVMValueRef
f32bits_to_ufloat(struct vp_tr *t, LLVMValueRef vi, unsigned mbits)
{
   unsigned shift = 10 - mbits;                 /* a half carries 10 mantissa bits */
   unsigned mask  = (1u << (5 + mbits)) - 1u;
   LLVMValueRef f = LLVMBuildBitCast(t->b, vi, t->f32, "");
   f = LLVMBuildSelect(t->b,                    /* unsigned: clamp <=0 and NaN to 0 */
      LLVMBuildFCmp(t->b, LLVMRealOGT, f, LLVMConstReal(t->f32, 0.0), ""),
      f, LLVMConstReal(t->f32, 0.0), "");
   LLVMValueRef hi = LLVMBuildZExt(t->b,
      LLVMBuildBitCast(t->b, LLVMBuildFPTrunc(t->b, f,
         LLVMHalfTypeInContext(t->ctx), ""), LLVMInt16TypeInContext(t->ctx), ""),
      t->i32, "");
   /* round the dropped low mantissa bits (round half up); a carry propagates
    * through the mantissa into the exponent, matching IEEE overflow to inf. */
   hi = LLVMBuildAdd(t->b, hi, LLVMConstInt(t->i32, 1u << (shift - 1), false), "");
   return LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, hi, LLVMConstInt(t->i32, shift, false), ""),
      LLVMConstInt(t->i32, mask, false), "");
}

/* inverse of f32bits_to_ufloat: unsigned-float field (low bits of `word`) -> f32
 * bit pattern, via the same half relay. */
static LLVMValueRef
ufloat_to_f32bits(struct vp_tr *t, LLVMValueRef word, unsigned mbits)
{
   unsigned shift = 10 - mbits;
   unsigned mask  = (1u << (5 + mbits)) - 1u;
   LLVMValueRef fld = LLVMBuildAnd(t->b, word,
      LLVMConstInt(t->i32, mask, false), "");
   LLVMValueRef h16 = LLVMBuildTrunc(t->b,
      LLVMBuildShl(t->b, fld, LLVMConstInt(t->i32, shift, false), ""),
      LLVMInt16TypeInContext(t->ctx), "");
   LLVMValueRef ff = LLVMBuildFPExt(t->b,
      LLVMBuildBitCast(t->b, h16, LLVMHalfTypeInContext(t->ctx), ""), t->f32, "");
   return LLVMBuildBitCast(t->b, ff, t->i32, "");
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
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef id = emit_csr_read(t, base + c, "id");
         /* gl_WorkGroupID must include the vkCmdDispatchBase base offset
          * (arg[2]=base_x|base_y<<32, arg[3]=base_z). The hardware block id is
          * base-relative; the base rides in as a kernel dispatch parameter so
          * this works identically on SimX and RTL with no KMU change. */
         if (in->intrinsic == nir_intrinsic_load_workgroup_id && t->arg) {
            unsigned slot = (c == 2) ? VP_ARG_GRID_BASE_Z : VP_ARG_GRID_BASE_XY;
            LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, t->arg,
               (LLVMValueRef[]){ LLVMConstInt(t->i64, slot, false) }, 1, "");
            LLVMValueRef packed = LLVMBuildLoad2(t->b, t->i64, gep, "gbase");
            if (c == 1)
               packed = LLVMBuildLShr(t->b, packed, LLVMConstInt(t->i64, 32, false), "");
            LLVMValueRef off = LLVMBuildTrunc(t->b, packed, t->i32, "");
            id = LLVMBuildAdd(t->b, id, off, "wgid");
         }
         ssa_set(t, in->def.index, c, id);
      }
      break;
   }
   /* gl_VertexIndex is firstVertex + the draw position, while the vid is the
    * 0-based position (the host folds firstVertex into the attribute bases, so
    * the fetch must not see it twice). The base is zero on an indexed draw,
    * where the vid is already the absolute index value, and on the
    * zero-base opcode, which is 0-based by definition. */
   case nir_intrinsic_load_vertex_id:
      ssa_set(t, in->def.index, 0,
              t->base_vertex ? LLVMBuildAdd(t->b, t->vid, t->base_vertex, "vidx")
                             : t->vid);
      break;
   case nir_intrinsic_load_vertex_id_zero_base:
      ssa_set(t, in->def.index, 0, t->vid);
      break;
   case nir_intrinsic_load_base_vertex:
      ssa_set(t, in->def.index, 0,
              t->base_vertex ? t->base_vertex : LLVMConstInt(t->i32, 0, false));
      break;
   /* Instancing. gl_InstanceIndex lowers (nir_lower_system_values) to
    * load_instance_id + load_base_instance; the VS prologue resolves the
    * 0-based instance id and the base-instance from arg slots 3/4. On the
    * non-instanced fast path instance == 0 and first_instance == 0. */
   /* gl_FrontFacing: the wrapper resolved the primitive's winding into the
    * reserved system-value slot, because the primitive record is in scope there
    * and not here. NIR wants a 1-bit value. */
   case nir_intrinsic_load_front_face:
      ssa_set(t, in->def.index, 0,
              LLVMBuildICmp(t->b, LLVMIntNE,
                 emit_load_i32(t, LLVMBuildAdd(t->b, t->fs_in_base,
                    vp_iptr_const(t, t->fs_sysval_off), "")),
                 LLVMConstInt(t->i32, 0, false), "frontface"));
      break;
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
      unsigned     esz  = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i64, c * esz, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, vp_load_mem(t, in->def.bit_size, p, "buf"));
      }
      break;
   }
   case nir_intrinsic_get_ssbo_size: {
      /* Runtime length of an unsized SSBO array (GLSL .length()). src[0] is the
       * buffer's descriptor device address; its lp_jit_buffer.num_elements
       * (u32 at offset 8, == byte size for an SSBO) is what the intrinsic
       * returns. The frontend divides by the array stride to get the count. */
      LLVMValueRef desc = intr_src(t, in, 0);
      LLVMValueRef sp = LLVMBuildIntToPtr(t->b,
         LLVMBuildAdd(t->b, desc, LLVMConstInt(t->i64, 8, false), ""),
         t->ptr, "");
      ssa_set(t, in->def.index, 0, LLVMBuildLoad2(t->b, t->i32, sp, "ssbosize"));
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
            LLVMBuildStore(t->b, vp_store_mem_val(t, nir_src_bit_size(in->src[0]), v),
                           LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
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
      /* A suppressed fragment may not touch memory, and an atomic is a write
       * like any other -- steer it to the sink exactly as store_ssbo is. */
      LLVMValueRef addr = emit_store_addr(t, LLVMBuildAdd(t->b, base, off, ""));
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
      LLVMValueRef p = LLVMBuildIntToPtr(t->b,
         emit_store_addr(t, intr_src(t, in, 0)), t->ptr, "");
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
      LLVMValueRef p = LLVMBuildIntToPtr(t->b, emit_store_addr(t, addr),
                                         t->ptr, "");
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
      unsigned     esz  = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i64, c * esz, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, vp_load_mem(t, in->def.bit_size, p, "push"));
      }
      break;
   }
   /* raw global memory: the operand IS the device address. */
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant: {
      LLVMValueRef addr = intr_src(t, in, 0);
      unsigned     esz  = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(LLVMTypeOf(addr), c * esz, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, vp_load_mem(t, in->def.bit_size, p, "gld"));
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
            LLVMBuildStore(t->b, vp_store_mem_val(t, nir_src_bit_size(in->src[0]), v),
                           LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
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
      unsigned    esz = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            vp_iptr_const(t, c * esz), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, vp_load_mem(t, in->def.bit_size, p, "sld"));
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
            LLVMBuildStore(t->b, vp_store_mem_val(t, nir_src_bit_size(in->src[0]), v),
                           LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
      }
      break;
   }
   /* Per-thread scratch: addr = scratch base + dynamic offset. */
   case nir_intrinsic_load_scratch: {
      LLVMValueRef addr = LLVMBuildAdd(t->b, emit_scratch_base(t),
         vp_to_iptr(t, intr_src(t, in, 0)), "scaddr");
      unsigned    esz = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr, vp_iptr_const(t, c * esz), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, vp_load_mem(t, in->def.bit_size, p, "scld"));
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
            LLVMBuildStore(t->b, vp_store_mem_val(t, nir_src_bit_size(in->src[0]), v),
                           LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
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
   /* shuffle: each lane reads src[0] from the lane numbered src[1]. shfl_idx
    * takes a dynamic source lane and preserves the value type, so this is a
    * direct gather. */
   case nir_intrinsic_shuffle:
      ssa_set(t, in->def.index, 0,
              emit_shfl_idx(t, intr_src(t, in, 0), intr_src(t, in, 1)));
      break;
   /* vote_all / vote_any: reduce a per-lane predicate over the ACTIVE lanes.
    * active = ballot(1); votes = ballot(pred). all = (votes & active)==active;
    * any = (votes & active)!=0. Restricting to the active mask is what makes the
    * result correct under divergence. */
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_any: {
      LLVMValueRef pred = intr_src(t, in, 0);
      if (LLVMTypeOf(pred) == LLVMInt1TypeInContext(t->ctx))
         pred = LLVMBuildZExt(t->b, pred, t->i32, "");
      LLVMValueRef active = emit_ballot(t, LLVMConstInt(t->i32, 1, false));
      LLVMValueRef votes  = LLVMBuildAnd(t->b, emit_ballot(t, pred), active, "");
      LLVMValueRef r = in->intrinsic == nir_intrinsic_vote_all
         ? LLVMBuildICmp(t->b, LLVMIntEQ, votes, active, "vote_all")
         : LLVMBuildICmp(t->b, LLVMIntNE, votes,
                         LLVMConstInt(t->i32, 0, false), "vote_any");
      ssa_set(t, in->def.index, 0, r);
      break;
   }
   /* vote_ieq / vote_feq: true iff src[0] is equal across all active lanes.
    * Broadcast the lowest active lane's value, compare per-lane, then vote_all
    * that comparison. feq uses ordered float equality (NaN => not-all-equal,
    * which is the defined result). */
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq: {
      LLVMValueRef val    = intr_src(t, in, 0);
      LLVMValueRef active = emit_ballot(t, LLVMConstInt(t->i32, 1, false));
      LLVMValueRef low    = emit_cttz(t, active);
      LLVMValueRef first  = emit_shfl_idx(t, val, low);
      LLVMValueRef eq;
      if (in->intrinsic == nir_intrinsic_vote_feq) {
         unsigned bs = in->src[0].ssa->bit_size;
         eq = LLVMBuildFCmp(t->b, LLVMRealOEQ,
                            as_float(t, val, bs), as_float(t, first, bs), "");
      } else {
         eq = LLVMBuildICmp(t->b, LLVMIntEQ, val, first, "");
      }
      LLVMValueRef votes = LLVMBuildAnd(t->b,
         emit_ballot(t, LLVMBuildZExt(t->b, eq, t->i32, "")), active, "");
      ssa_set(t, in->def.index, 0,
              LLVMBuildICmp(t->b, LLVMIntEQ, votes, active, "vote_eq"));
      break;
   }
   /* Subgroup scans/reduction over the warp. All three share the inclusive
    * Hillis-Steele scan (emit_subgroup_incl_scan):
    *   inclusive[i] = combine(lanes 0..i)
    *   exclusive[i] = combine(lanes 0..i-1); lane 0 = identity  (= incl shifted up 1)
    *   reduce       = combine(all active lanes) = inclusive of the highest active lane
    * 32-bit lane values only (the shfl transport width).
    *
    * A reduce may additionally carry a cluster size: it then folds over each
    * group of that many consecutive lanes independently rather than over the
    * whole warp. Only reduce carries one — the scans never do — and a cluster
    * spanning the entire subgroup has already been normalised to 0 upstream. */
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
   case nir_intrinsic_reduce: {
      if (in->def.bit_size != 32) {
         mesa_logw("vortexpipe: %u-bit subgroup scan/reduce unsupported (32-bit only)",
                   in->def.bit_size);
         t->ok = false;
         break;
      }
      unsigned cluster = in->intrinsic == nir_intrinsic_reduce
         ? nir_intrinsic_cluster_size(in) : 0u;
      /* The cluster arithmetic below builds a power-of-two lane mask that must
       * fit the 32-bit ballot, both of which Vulkan guarantees for a cluster no
       * wider than the subgroup. Refuse anything else rather than fold over the
       * wrong lanes. */
      if ((cluster & (cluster - 1u)) || cluster > 32u) {
         mesa_logw("vortexpipe: unsupported subgroup reduce cluster size %u",
                   cluster);
         t->ok = false;
         break;
      }
      if (cluster == 1u) {
         /* Each lane is its own cluster: the reduction is the lane's own value. */
         ssa_set(t, in->def.index, 0, intr_src(t, in, 0));
         break;
      }
      nir_op rop  = nir_intrinsic_reduction_op(in);
      LLVMValueRef incl = emit_subgroup_incl_scan(t, rop, intr_src(t, in, 0),
                                                  cluster);
      if (!t->ok)
         break;
      LLVMValueRef r;
      if (in->intrinsic == nir_intrinsic_inclusive_scan) {
         r = incl;
      } else if (in->intrinsic == nir_intrinsic_exclusive_scan) {
         /* exclusive[i] = the inclusive value of the nearest LOWER active lane;
          * the first active lane gets the op identity. Probe lower lanes at
          * power-of-two distances and take the first in-range active one — the
          * `got` flag keeps it to the NEAREST. Divergence-safe like the inclusive
          * scan; a fixed shfl_up(1) would read an inactive neighbour under a
          * strided active mask and deliver the wrong lane's value. */
         LLVMValueRef lane   = emit_csr_read(t, VX_CSR_THREAD_ID, "lane");
         LLVMValueRef active = emit_ballot(t, LLVMConstInt(t->i32, 1, false));
         LLVMValueRef one    = LLVMConstInt(t->i32, 1, false);
         LLVMValueRef zero   = LLVMConstInt(t->i32, 0, false);
         LLVMTypeRef  i1t    = LLVMInt1TypeInContext(t->ctx);
         r = emit_scan_identity(t, rop);
         if (!t->ok)
            break;
         LLVMValueRef got = LLVMConstInt(i1t, 0, false);
         for (unsigned d = 1; d <= 16u; d <<= 1) {
            LLVMValueRef dc      = LLVMConstInt(t->i32, d, false);
            LLVMValueRef cand    = emit_shfl_up(t, incl, d);
            LLVMValueRef inrange = LLVMBuildICmp(t->b, LLVMIntUGE, lane, dc, "");
            LLVMValueRef srcbit  = LLVMBuildAnd(t->b,
               LLVMBuildLShr(t->b, active,
                             LLVMBuildSub(t->b, lane, dc, ""), ""), one, "");
            LLVMValueRef srcact  = LLVMBuildICmp(t->b, LLVMIntNE, srcbit, zero, "");
            LLVMValueRef here    = LLVMBuildAnd(t->b, inrange, srcact, "");
            LLVMValueRef take    = LLVMBuildAnd(t->b, here,
                                                LLVMBuildNot(t->b, got, ""), "");
            r   = LLVMBuildSelect(t->b, take, cand, r, "excl");
            got = LLVMBuildOr(t->b, got, here, "");
         }
      } else { /* reduce: broadcast the highest active lane's inclusive value */
         LLVMValueRef active = emit_ballot(t, LLVMConstInt(t->i32, 1, false));
         if (cluster) {
            /* Confine the search to this lane's cluster, or the broadcast would
             * pull in the inclusive value of a lane the cluster does not
             * contain -- which is the whole-warp answer, not the cluster's. */
            LLVMValueRef lane = emit_csr_read(t, VX_CSR_THREAD_ID, "lane");
            LLVMValueRef base = LLVMBuildAnd(t->b, lane,
               LLVMConstInt(t->i32, ~(cluster - 1u), false), "cbase");
            LLVMValueRef cmask = LLVMBuildShl(t->b,
               LLVMConstInt(t->i32, (1u << cluster) - 1u, false), base, "cmask");
            active = LLVMBuildAnd(t->b, active, cmask, "cactive");
         }
         LLVMValueRef hi = LLVMBuildSub(t->b, LLVMConstInt(t->i32, 31, false),
                                        emit_ctlz(t, active), "hi_lane");
         r = emit_shfl_idx(t, incl, hi);
      }
      ssa_set(t, in->def.index, 0, r);
      break;
   }
   /* Flat CTA-linear lane id = warp_id * NT + lane (one CTA per core, so the
    * warp id within the core is the warp id within the workgroup). */
   case nir_intrinsic_load_local_invocation_index: {
      /* gl_LocalInvocationIndex is the row-major linearization of the 3D local
       * invocation id over the workgroup size (Vulkan spec):
       *   (z*dimY + y)*dimX + x.
       * Derive it from the CTA-local thread id + block dims — NOT warp_id*NT+tid,
       * which uses the PHYSICAL warp id (VX_CSR_WARP_ID) and is wrong whenever a
       * CTA lands on a nonzero physical warp (e.g. single-thread workgroups on
       * later warps read warp_id>0 → index = warp_id*NT instead of 0). */
      LLVMValueRef x  = emit_csr_read(t, VX_CSR_CTA_THREAD_ID_X,     "ltx");
      LLVMValueRef y  = emit_csr_read(t, VX_CSR_CTA_THREAD_ID_X + 1, "lty");
      LLVMValueRef z  = emit_csr_read(t, VX_CSR_CTA_THREAD_ID_X + 2, "ltz");
      LLVMValueRef dx = emit_csr_read(t, VX_CSR_CTA_BLOCK_DIM_X,     "bdx");
      LLVMValueRef dy = emit_csr_read(t, VX_CSR_CTA_BLOCK_DIM_X + 1, "bdy");
      LLVMValueRef idx = LLVMBuildAdd(t->b,
         LLVMBuildMul(t->b,
            LLVMBuildAdd(t->b, LLVMBuildMul(t->b, z, dy, ""), y, ""), dx, ""),
         x, "lii");
      ssa_set(t, in->def.index, 0, idx);
      break;
   }
   /* workgroup barrier. The memory-scoped form must drain this thread's
    * outstanding memory ops before proceeding: a shared/global atomic whose
    * result is discarded is not scoreboard-tracked, so without a fence a later
    * access to the same location can issue while the atomic is still in flight
    * (observed: atomicAdd then a barrier then atomicExchange reordering). A real
    * RISC-V fence maps to the LSU FENCE, which stalls the warp until its pending
    * requests retire. The execution-scoped form additionally syncs CTA warps. */
   case nir_intrinsic_barrier:
      if (nir_intrinsic_memory_scope(in) != SCOPE_NONE)
         LLVMBuildFence(t->b, LLVMAtomicOrderingSequentiallyConsistent,
                        /*singleThread*/ false, "");
      if (nir_intrinsic_execution_scope(in) != SCOPE_NONE)
         emit_vx_barrier(t);
      break;
   /* deref load/store: the deref operand is an iptr byte address; each
    * component is a separate scalar load/store, width from bit size. */
   case nir_intrinsic_load_deref: {
      LLVMValueRef addr = intr_src(t, in, 0);
      if (!addr) { t->ok = false; break; }
      unsigned    esz = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            vp_iptr_const(t, c * esz), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, vp_load_mem(t, in->def.bit_size, p, "ld"));
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
         LLVMBuildStore(t->b, vp_store_mem_val(t, nir_src_bit_size(in->src[1]), v), p);
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

   case nir_intrinsic_vortex_rt_continue:
      emit_vx_rt_continue(t, ssa_get(t, in->src[0].ssa->index, 0),
                             ssa_get(t, in->src[1].ssa->index, 0),
                             ssa_get(t, in->src[2].ssa->index, 0));
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
      /* Storage layouts by class: verbatim 32-bit lanes (RAW32: R/RG/RGBA x
       * FLOAT/UINT/SINT, raw bits copied), one 16-bit half (F16), and single
       * 32-bit packed words (UNORM8; R10G10B10A2 UNORM/UINT; R11G11B10 float).
       * Only the format's own channels are touched. Single-mip 2D addressing. */
      unsigned bpp, nchan;
      enum { IMG_RAW32, IMG_UNORM8, IMG_F16, IMG_RGB10A2_UNORM,
             IMG_RGB10A2_UINT, IMG_RG11B10F } cls;
      bool is_float;
      switch (fmt) {
      case PIPE_FORMAT_R32G32B32A32_FLOAT: bpp=16; nchan=4; cls=IMG_RAW32; is_float=true;  break;
      case PIPE_FORMAT_R32G32B32A32_UINT:
      case PIPE_FORMAT_R32G32B32A32_SINT:  bpp=16; nchan=4; cls=IMG_RAW32; is_float=false; break;
      case PIPE_FORMAT_R32G32_FLOAT:       bpp=8;  nchan=2; cls=IMG_RAW32; is_float=true;  break;
      case PIPE_FORMAT_R32G32_UINT:
      case PIPE_FORMAT_R32G32_SINT:        bpp=8;  nchan=2; cls=IMG_RAW32; is_float=false; break;
      case PIPE_FORMAT_R32_FLOAT:          bpp=4;  nchan=1; cls=IMG_RAW32; is_float=true;  break;
      case PIPE_FORMAT_R32_UINT:
      case PIPE_FORMAT_R32_SINT:           bpp=4;  nchan=1; cls=IMG_RAW32; is_float=false; break;
      case PIPE_FORMAT_R8G8B8A8_UNORM:     bpp=4;  nchan=4; cls=IMG_UNORM8;        is_float=true;  break;
      case PIPE_FORMAT_R16_FLOAT:          bpp=2;  nchan=1; cls=IMG_F16;           is_float=true;  break;
      case PIPE_FORMAT_R10G10B10A2_UNORM:  bpp=4;  nchan=4; cls=IMG_RGB10A2_UNORM; is_float=true;  break;
      case PIPE_FORMAT_R10G10B10A2_UINT:   bpp=4;  nchan=4; cls=IMG_RGB10A2_UINT;  is_float=false; break;
      case PIPE_FORMAT_R11G11B10_FLOAT:    bpp=4;  nchan=3; cls=IMG_RG11B10F;      is_float=true;  break;
      default:
         mesa_logw("vortexpipe: image %s: unsupported format %d",
                   store ? "store" : "load", fmt);
         t->ok = false;
         break;
      }
      if (!t->ok) break;
      /* Field bit-offset and (for the two non-RAW packed classes) the per-channel
       * width, indexed by component. RGB10A2: R,G,B are 10-bit, A is 2-bit at
       * bit 30. R11G11B10: R,G are 11-bit at 0/11, B is 10-bit at 22. */
      const unsigned f_off[4]   = { 0, 10, 20, 30 };  /* RGB10A2 field offsets */
      const unsigned f_ubits[4] = { 10, 10, 10, 2 };  /* RGB10A2 field widths  */
      const unsigned uf_off[3]  = { 0, 11, 22 };      /* R11G11B10 field offsets */
      const unsigned uf_mant[3] = { 6, 6, 5 };        /* R11G11B10 mantissa bits */
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
      LLVMTypeRef i16 = LLVMInt16TypeInContext(t->ctx);
      LLVMTypeRef half = LLVMHalfTypeInContext(t->ctx);
      if (store) {
         addr = emit_store_addr(t, addr);
         if (cls == IMG_RAW32) {
            for (unsigned c = 0; c < nchan; c++) {
               LLVMValueRef v = ssa_get(t, in->src[3].ssa->index, c);
               if (!v) continue;
               LLVMValueRef a = LLVMBuildAdd(t->b, addr,
                  LLVMConstInt(t->i64, c * 4u, false), "");
               LLVMBuildStore(t->b, v, LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
            }
         } else if (cls == IMG_F16) {
            /* one 16-bit half (relay through the same fptrunc as from_float) */
            LLVMValueRef v = ssa_get(t, in->src[3].ssa->index, 0);
            LLVMValueRef h = LLVMBuildBitCast(t->b,
               LLVMBuildFPTrunc(t->b, LLVMBuildBitCast(t->b, v, t->f32, ""), half, ""),
               i16, "");
            LLVMBuildStore(t->b, h, LLVMBuildIntToPtr(t->b, addr, t->ptr, ""));
         } else {
            /* single 32-bit packed word: UNORM8, R10G10B10A2 (UNORM/UINT), R11G11B10F */
            LLVMValueRef packed = LLVMConstInt(t->i32, 0, false);
            for (unsigned c = 0; c < nchan; c++) {
               LLVMValueRef v = ssa_get(t, in->src[3].ssa->index, c);
               if (!v) continue;
               LLVMValueRef fld;
               unsigned off;
               switch (cls) {
               case IMG_UNORM8:
                  fld = f32bits_to_unorm(t, v, 8);  off = c * 8u; break;
               case IMG_RGB10A2_UNORM:
                  fld = f32bits_to_unorm(t, v, f_ubits[c]); off = f_off[c]; break;
               case IMG_RGB10A2_UINT:
                  fld = LLVMBuildAnd(t->b, v,
                     LLVMConstInt(t->i32, (1u << f_ubits[c]) - 1u, false), "");
                  off = f_off[c]; break;
               default: /* IMG_RG11B10F */
                  fld = f32bits_to_ufloat(t, v, uf_mant[c]); off = uf_off[c]; break;
               }
               packed = LLVMBuildOr(t->b, packed,
                  LLVMBuildShl(t->b, fld, LLVMConstInt(t->i32, off, false), ""), "");
            }
            LLVMBuildStore(t->b, packed, LLVMBuildIntToPtr(t->b, addr, t->ptr, ""));
         }
      } else {
         /* Image loads always yield a vec4; channels the format lacks read back
          * as 0 (G,B) and 1 (A) per Vulkan storage-image semantics. */
         LLVMValueRef word = NULL;
         if (cls == IMG_UNORM8 || cls == IMG_RGB10A2_UNORM ||
             cls == IMG_RGB10A2_UINT || cls == IMG_RG11B10F)
            word = LLVMBuildLoad2(t->b, t->i32,
               LLVMBuildIntToPtr(t->b, addr, t->ptr, ""), "img");
         LLVMTypeRef lt = ity(t, in->def.bit_size);
         unsigned one = is_float ? 0x3f800000u : 1u;
         for (unsigned c = 0; c < in->def.num_components; c++) {
            LLVMValueRef val;
            if (c >= nchan) {
               val = LLVMConstInt(lt, (c == 3) ? one : 0u, false);
            } else switch (cls) {
            case IMG_RAW32: {
               LLVMValueRef a = LLVMBuildAdd(t->b, addr,
                  LLVMConstInt(t->i64, c * 4u, false), "");
               val = LLVMBuildLoad2(t->b, lt,
                  LLVMBuildIntToPtr(t->b, a, t->ptr, ""), "img");
               break;
            }
            case IMG_F16: {
               LLVMValueRef h = LLVMBuildLoad2(t->b, i16,
                  LLVMBuildIntToPtr(t->b, addr, t->ptr, ""), "img");
               val = LLVMBuildBitCast(t->b,
                  LLVMBuildFPExt(t->b, LLVMBuildBitCast(t->b, h, half, ""), t->f32, ""),
                  t->i32, "");
               break;
            }
            case IMG_UNORM8:
               val = unorm_to_f32bits(t,
                  LLVMBuildLShr(t->b, word, LLVMConstInt(t->i32, c * 8u, false), ""), 8);
               break;
            case IMG_RGB10A2_UNORM:
               val = unorm_to_f32bits(t,
                  LLVMBuildLShr(t->b, word, LLVMConstInt(t->i32, f_off[c], false), ""),
                  f_ubits[c]);
               break;
            case IMG_RGB10A2_UINT:
               val = LLVMBuildAnd(t->b,
                  LLVMBuildLShr(t->b, word, LLVMConstInt(t->i32, f_off[c], false), ""),
                  LLVMConstInt(t->i32, (1u << f_ubits[c]) - 1u, false), "");
               break;
            default: /* IMG_RG11B10F */
               val = ufloat_to_f32bits(t,
                  LLVMBuildLShr(t->b, word, LLVMConstInt(t->i32, uf_off[c], false), ""),
                  uf_mant[c]);
               break;
            }
            ssa_set(t, in->def.index, c, val);
         }
      }
      break;
   }
   case nir_intrinsic_image_atomic:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_bindless_image_atomic_swap: {
      /* Storage-image atomics. Vulkan permits them only on 32-bit single-channel
       * formats (R32_UINT/SINT, R32_FLOAT), so the pixel is one 4-byte word at
       * the same single-mip 2D address as image load/store. The RMW reuses the
       * device A-extension path: src[0]=image descriptor, src[1]=coordinate,
       * src[2]=sample, src[3]=data (swap: src[3]=compare, src[4]=new). */
      bool is_swap = (in->intrinsic == nir_intrinsic_image_atomic_swap ||
                      in->intrinsic == nir_intrinsic_bindless_image_atomic_swap);
      LLVMValueRef desc = intr_src(t, in, 0);
      LLVMValueRef dp   = LLVMBuildIntToPtr(t->b, desc, t->ptr, "");
      LLVMValueRef base = LLVMBuildLoad2(t->b, t->i64, dp, "imgbase");
      LLVMValueRef boff = LLVMBuildZExt(t->b,
         img_desc_u32(t, desc, VP_JIT_IMG_BASE_OFFSET), t->i64, "");
      LLVMValueRef row  = LLVMBuildZExt(t->b,
         img_desc_u32(t, desc, VP_JIT_IMG_ROW_STRIDE), t->i64, "");
      LLVMValueRef x = LLVMBuildZExt(t->b, ssa_get(t, in->src[1].ssa->index, 0),
                                     t->i64, "");
      LLVMValueRef y = (nir_src_num_components(in->src[1]) >= 2)
         ? LLVMBuildZExt(t->b, ssa_get(t, in->src[1].ssa->index, 1), t->i64, "")
         : LLVMConstInt(t->i64, 0, false);
      LLVMValueRef off = LLVMBuildAdd(t->b, boff,
         LLVMBuildAdd(t->b, LLVMBuildMul(t->b, x, LLVMConstInt(t->i64, 4, false), ""),
                            LLVMBuildMul(t->b, y, row, ""), ""), "");
      LLVMValueRef addr = emit_store_addr(t, LLVMBuildAdd(t->b, base, off, ""));
      LLVMValueRef p = LLVMBuildIntToPtr(t->b, addr, t->ptr, "");
      emit_atomic_at(t, in, p, 3, is_swap);
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

/* Software sampler: gfx_tex_sample_sw(&texstate[0], u, v, lod, filter) on the
 * resident descriptor table (fs_main's 3rd param). Same fixed-point u/v/lod
 * convention; `filter` (tap in bit0, mip-linear in bit1) is resolved by the caller.
 * Returns the packed A8R8G8B8 texel. */
static LLVMValueRef
emit_tex_sw(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v, LLVMValueRef lod,
            LLVMValueRef filter)
{
   LLVMTypeRef params[5] = { t->ptr, t->i32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 5, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_sw");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_sample_sw", fty);
   LLVMValueRef a[5] = { t->fs_texstate, u, v, lod, filter };  /* stage 0 = table[0] */
   return LLVMBuildCall2(t->b, fty, fn, a, 5, "texsw");
}

/* vx_tex: sample TEX stage 0 (custom-1 funct3=5, R4-type). u/v/lod ride rs1/rs2/rs3
 * and the packed A8R8G8B8 texel returns in rd. The tap filter is a per-draw DCR, so
 * this path is only used for non-mipmapped draws (single filter, base level). */
static LLVMValueRef
emit_tex_hw(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v, LLVMValueRef lod)
{
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

/* The float[4] slot the float-returning texture ABI writes through. Allocated once
 * in the entry block and reused: an alloca built at the emit site would be dynamic,
 * growing the stack on every trip of a loop containing the fetch. The slot is dead
 * across calls -- each fetch fills all four channels before they are read. */
static LLVMValueRef
tex_f32_scratch(struct vp_tr *t)
{
   if (!t->tex_f32_slot) {
      LLVMBasicBlockRef cur = LLVMGetInsertBlock(t->b);
      LLVMValueRef first = LLVMGetFirstInstruction(t->entry);
      if (first)
         LLVMPositionBuilderBefore(t->b, first);
      else
         LLVMPositionBuilderAtEnd(t->b, t->entry);
      t->tex_f32_slot = LLVMBuildAlloca(t->b, LLVMArrayType(t->f32, 4), "texel_f32");
      LLVMPositionBuilderAtEnd(t->b, cur);
   }
   return t->tex_f32_slot;
}

/* texelFetch returning floats: gfx_tex_fetch_f32(&texstate[0], x, y, lod, out) --
 * the exact texel at integer (x,y) of integer lod, decoded to four float channels
 * in RGBA order. Returns the scratch slot the callee filled. */
static LLVMValueRef
emit_tex_fetch_f32(struct vp_tr *t, LLVMValueRef x, LLVMValueRef y, LLVMValueRef lod)
{
   LLVMTypeRef params[5] = { t->ptr, t->i32, t->i32, t->i32, t->ptr };
   LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), params, 5, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_fetch_f32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_fetch_f32", fty);
   LLVMValueRef out = tex_f32_scratch(t);
   LLVMValueRef a[5] = { t->fs_texstate, x, y, lod, out };
   LLVMBuildCall2(t->b, fty, fn, a, 5, "");
   return out;
}

/* The SW sampler delivered as four floats: gfx_tex_sample_f32(&texstate[0], u, v,
 * lod, filter, out). A float-format texture holds values outside [0,1] and cannot be
 * filtered through the 8-bit ARGB working space; the callee decodes and blends it in
 * float, and delegates every other format to the packed sampler so those results stay
 * bit-identical. Returns the scratch slot the callee filled. */
static LLVMValueRef
emit_tex_sw_f32(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v, LLVMValueRef lod,
                LLVMValueRef filter)
{
   LLVMTypeRef params[6] = { t->ptr, t->i32, t->i32, t->i32, t->i32, t->ptr };
   LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), params, 6, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_f32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_sample_f32", fty);
   LLVMValueRef out = tex_f32_scratch(t);
   LLVMValueRef a[6] = { t->fs_texstate, u, v, lod, filter, out };
   LLVMBuildCall2(t->b, fty, fn, a, 6, "");
   return out;
}

/* The int32[4] slot the integer texture ABI writes through; entry-block allocated
 * and reused for the same reason as tex_f32_scratch. */
static LLVMValueRef
tex_i32_scratch(struct vp_tr *t)
{
   if (!t->tex_i32_slot) {
      LLVMBasicBlockRef cur = LLVMGetInsertBlock(t->b);
      LLVMValueRef first = LLVMGetFirstInstruction(t->entry);
      if (first)
         LLVMPositionBuilderBefore(t->b, first);
      else
         LLVMPositionBuilderAtEnd(t->b, t->entry);
      t->tex_i32_slot = LLVMBuildAlloca(t->b, LLVMArrayType(t->i32, 4), "texel_i32");
      LLVMPositionBuilderAtEnd(t->b, cur);
   }
   return t->tex_i32_slot;
}

/* texelFetch for an integer sampler: gfx_tex_fetch_i32(&texstate[0], x, y, lod,
 * layer, out) -- the texel's four channels as raw 0..255 values. Returns the
 * scratch slot the callee filled. */
static LLVMValueRef
emit_tex_fetch_i32(struct vp_tr *t, LLVMValueRef x, LLVMValueRef y,
                   LLVMValueRef lod, LLVMValueRef layer)
{
   LLVMTypeRef params[6] = { t->ptr, t->i32, t->i32, t->i32, t->i32, t->ptr };
   LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), params, 6, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_fetch_i32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_fetch_i32", fty);
   LLVMValueRef out = tex_i32_scratch(t);
   LLVMValueRef a[6] = { t->fs_texstate, x, y, lod, layer, out };
   LLVMBuildCall2(t->b, fty, fn, a, 6, "");
   return out;
}

/* texelFetch on a 2D array returning floats: the layer selects the slice, the
 * channels come back as gfx_tex_fetch_f32's do. */
static LLVMValueRef
emit_tex_fetch_array_f32(struct vp_tr *t, LLVMValueRef x, LLVMValueRef y,
                         LLVMValueRef layer, LLVMValueRef lod)
{
   LLVMTypeRef params[6] = { t->ptr, t->i32, t->i32, t->i32, t->i32, t->ptr };
   LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), params, 6, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_fetch_array_f32");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_fetch_array_f32", fty);
   LLVMValueRef out = tex_f32_scratch(t);
   LLVMValueRef a[6] = { t->fs_texstate, x, y, layer, lod, out };
   LLVMBuildCall2(t->b, fty, fn, a, 6, "");
   return out;
}

/* textureGather: gfx_tex_gather_sw(&texstate[0], x, y, comp) -- channel `comp` of
 * the 2x2 footprint at (x,y), base level, packed in GL gather order as bytes
 * x | y<<8 | z<<16 | w<<24. Unpack to the def's vec4 as floats in [0,1]. An array
 * sampler passes its layer, which selects the slice the footprint is taken from. */
static void
emit_tex_gather(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef x, LLVMValueRef y,
                LLVMValueRef layer)
{
   const unsigned n = layer ? 5 : 4;
   LLVMTypeRef params[5] = { t->ptr, t->i32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, n, false);
   const char *name = layer ? "gfx_tex_gather_array_sw" : "gfx_tex_gather_sw";
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, name);
   if (!fn)
      fn = LLVMAddFunction(t->mod, name, fty);
   LLVMValueRef a[5] = { t->fs_texstate, x, y,
                         LLVMConstInt(t->i32, tex->component, false), layer };
   LLVMValueRef packed = LLVMBuildCall2(t->b, fty, fn, a, n, "gather");
   for (unsigned c = 0; c < tex->def.num_components && c < 4; c++) {
      LLVMValueRef byte = LLVMBuildAnd(t->b,
         LLVMBuildLShr(t->b, packed, LLVMConstInt(t->i32, c * 8, false), ""),
         LLVMConstInt(t->i32, 0xff, false), "");
      LLVMValueRef f = LLVMBuildFMul(t->b,
         LLVMBuildUIToFP(t->b, byte, t->f32, ""),
         LLVMConstReal(t->f32, 1.0 / 255.0), "");
      ssa_set(t, tex->def.index, c, LLVMBuildBitCast(t->b, f, t->i32, ""));
   }
}

/* textureGatherCmp: compare each of the 2x2 depth taps against ref_bits and
 * return the vec4 of 0/1 results (gfx_tex_gather_cmp_sw packs 0xff/0x00 per tap;
 * the /255 unpack yields 0.0/1.0, matching the colour gather). */
static void
emit_tex_gather_cmp(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef x,
                    LLVMValueRef y, LLVMValueRef ref_bits, LLVMValueRef layer)
{
   unsigned n = layer ? 5 : 4;
   LLVMTypeRef params[5] = { t->ptr, t->i32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, n, false);
   const char *name = layer ? "gfx_tex_gather_cmp_array_sw" : "gfx_tex_gather_cmp_sw";
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, name);
   if (!fn)
      fn = LLVMAddFunction(t->mod, name, fty);
   LLVMValueRef a[5] = { t->fs_texstate, x, y, ref_bits, layer };
   LLVMValueRef packed = LLVMBuildCall2(t->b, fty, fn, a, n, "gathercmp");
   for (unsigned c = 0; c < tex->def.num_components && c < 4; c++) {
      LLVMValueRef byte = LLVMBuildAnd(t->b,
         LLVMBuildLShr(t->b, packed, LLVMConstInt(t->i32, c * 8, false), ""),
         LLVMConstInt(t->i32, 0xff, false), "");
      LLVMValueRef f = LLVMBuildFMul(t->b,
         LLVMBuildUIToFP(t->b, byte, t->f32, ""),
         LLVMConstReal(t->f32, 1.0 / 255.0), "");
      ssa_set(t, tex->def.index, c, LLVMBuildBitCast(t->b, f, t->i32, ""));
   }
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

/* CONTINUE: resume traversal for the candidate this lane was handed, with its
 * verdict in rs1, the hit distance in the FP operand and the attribute in rs3.
 * An any-hit lane computes no distance of its own and passes the candidate's
 * back. No result — the next vx_rt_wait on the handle collects the response. */
static void
emit_vx_rt_continue(struct vp_tr *t, LLVMValueRef action, LLVMValueRef tval,
                    LLVMValueRef attr)
{
   const char *s = ".insn r4 43, 6, 0, x0, $0, $1, $2";
   LLVMTypeRef args[3] = { t->i32, t->f32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), args, 3, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "r,f,r", 5,
                                      /*HasSideEffects*/ true, false,
                                      LLVMInlineAsmDialectATT, false);
   /* The hit window is read as untyped 32-bit words, so a distance arrives as
    * an integer-shaped value and has to be reinterpreted for the FP operand. */
   if (LLVMTypeOf(tval) != t->f32)
      tval = LLVMBuildBitCast(t->b, tval, t->f32, "rtcont_t");
   LLVMValueRef a[3] = { action, tval, attr };
   LLVMBuildCall2(t->b, fnty, ia, a, 3, "");
}

/* llvm.ctlz.i64: leading-zero count (is_zero_undef=false). */
static LLVMValueRef
emit_ctlz64(struct vp_tr *t, LLVMValueRef v)
{
   LLVMTypeRef  i1   = LLVMInt1TypeInContext(t->ctx);
   LLVMTypeRef  args[2] = { t->i64, i1 };
   LLVMTypeRef  fty  = LLVMFunctionType(t->i64, args, 2, false);
   LLVMValueRef fn   = LLVMGetNamedFunction(t->mod, "llvm.ctlz.i64");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "llvm.ctlz.i64", fty);
   LLVMValueRef a[2] = { v, LLVMConstInt(i1, 0, false) };
   return LLVMBuildCall2(t->b, fty, fn, a, 2, "ctlz64");
}

/* One texel-space gradient: |a-b| widened to i64 and shifted left by the axis's
 * log2 dimension. abs-vs-0 (not a sign-correct derivative) is all the LOD needs. */
static LLVMValueRef
emit_lod_grad(struct vp_tr *t, LLVMValueRef a, LLVMValueRef b, LLVMValueRef logdim)
{
   LLVMValueRef d   = LLVMBuildSub(t->b, a, b, "");
   LLVMValueRef abs = LLVMBuildSelect(t->b,
      LLVMBuildICmp(t->b, LLVMIntSLT, d, LLVMConstInt(t->i32, 0, false), ""),
      LLVMBuildNeg(t->b, d, ""), d, "");
   return LLVMBuildShl(t->b, LLVMBuildZExt(t->b, abs, t->i64, ""),
                       LLVMBuildZExt(t->b, logdim, t->i64, ""), "grad");
}

static LLVMValueRef
emit_umax64(struct vp_tr *t, LLVMValueRef a, LLVMValueRef b)
{
   return LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntUGT, a, b, ""), a, b, "");
}

/* Gradient-vector length sqrt(a^2 + b^2) of two orthogonal texel-space components.
 * Computed in f32 (the FPU fsqrt.s) rather than i128 integer arithmetic: the LOD
 * only needs ~log2(rho), so f32's precision is ample, and this is exact to the
 * tolerance rather than a max+min approximation (which perturbs the trilinear blend). */
static LLVMValueRef
emit_grad_len(struct vp_tr *t, LLVMValueRef a, LLVMValueRef b)
{
   LLVMValueRef af = LLVMBuildUIToFP(t->b, a, t->f32, "");
   LLVMValueRef bf = LLVMBuildUIToFP(t->b, b, t->f32, "");
   LLVMValueRef sq = LLVMBuildFAdd(t->b, LLVMBuildFMul(t->b, af, af, ""),
                                   LLVMBuildFMul(t->b, bf, bf, ""), "");
   return LLVMBuildFPToUI(t->b, emit_fsqrt(t, sq), t->i64, "grad_len");
}

/* The lane quad's texel-space gradient rho (S.FXD_FRAC fixed-point, i64): the max
 * over the two screen directions of the gradient-vector length. log2(rho) - FXD_FRAC
 * is lambda. logw/logh come from the resident TEX descriptor (fs_texstate) because the
 * TEX DCRs are host-write-only; quad neighbours arrive through the quad-scoped SHFL
 * (bfly), so this must run with the whole quad active. */
static LLVMValueRef
emit_tex_grad_rho(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v)
{
   LLVMValueRef off = LLVMConstInt(t->i32,
      offsetof(gfx_sw_texstate_t, logdim), false);
   LLVMValueRef p   = LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &off, 1, "logdim_p");
   LLVMValueRef logdim = LLVMBuildLoad2(t->b, t->i32, p, "logdim");
   LLVMValueRef logw = LLVMBuildAnd(t->b, logdim,
      LLVMConstInt(t->i32, 0xFFFF, false), "logw");
   LLVMValueRef logh = LLVMBuildLShr(t->b, logdim,
      LLVMConstInt(t->i32, 16, false), "logh");

   /* quad bfly control: (mask 0x3c << 12) | (cval 3 << 6) | dir. */
   LLVMValueRef bch = LLVMConstInt(t->i32, (0x3c << 12) | (3 << 6) | 1, false);
   LLVMValueRef bcv = LLVMConstInt(t->i32, (0x3c << 12) | (3 << 6) | 2, false);
   LLVMValueRef uh = emit_shfl(t, 6, u, bch), uv = emit_shfl(t, 6, u, bcv);
   LLVMValueRef vh = emit_shfl(t, 6, v, bch), vv = emit_shfl(t, 6, v, bcv);

   /* rho = max over the two screen directions (x = horizontal neighbour, y =
    * vertical) of the gradient-vector length sqrt((du*W)^2 + (dv*H)^2), not the
    * per-axis max of the four components. The per-axis max underestimates a
    * diagonal gradient (du and dv varying together) by up to sqrt(2) ~ 0.5 LOD,
    * which exceeds the deqp mipmap tolerance. */
   LLVMValueRef gux = emit_lod_grad(t, u, uh, logw);
   LLVMValueRef guy = emit_lod_grad(t, u, uv, logw);
   LLVMValueRef gvx = emit_lod_grad(t, v, vh, logh);
   LLVMValueRef gvy = emit_lod_grad(t, v, vv, logh);
   return emit_umax64(t, emit_grad_len(t, gux, gvx), emit_grad_len(t, guy, gvy));
}

/* Load the resident TEX descriptor's filter word (fs_texstate). */
static LLVMValueRef
emit_tex_filter_word(struct vp_tr *t)
{
   LLVMValueRef off = LLVMConstInt(t->i32,
      offsetof(gfx_sw_texstate_t, filter), false);
   LLVMValueRef p = LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &off, 1, "filter_p");
   return LLVMBuildLoad2(t->b, t->i32, p, "filter");
}

/* The bound view's component swizzle word (gfx_sw_texstate_t.swizzle):
 * r | g<<3 | b<<6 | a<<9, each field a PIPE_SWIZZLE_* (0..3 = R/G/B/A source
 * channel, 4 = 0.0, 5 = 1.0). Identity packs to 1672. */
static LLVMValueRef
emit_tex_swizzle_word(struct vp_tr *t)
{
   LLVMValueRef off = LLVMConstInt(t->i32,
      offsetof(gfx_sw_texstate_t, swizzle), false);
   LLVMValueRef p = LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &off, 1, "swz_p");
   return LLVMBuildLoad2(t->b, t->i32, p, "swizzle");
}

/* Read the sampler's LOD bias (signed Q(VX_TEX_LOD_FRAC_BITS)) from texstate. */
static LLVMValueRef
emit_tex_lod_bias(struct vp_tr *t)
{
   LLVMValueRef o = LLVMConstInt(t->i32, offsetof(gfx_sw_texstate_t, lod_bias), false);
   return LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &o, 1, ""), "lod_bias");
}

/* Clamp a signed Q(VX_TEX_LOD_FRAC_BITS) λ to the sampler's [min_lod, max_lod].
 * Default sampler (min_lod=0, max_lod=LOD_MAX) makes this the identity for a
 * minified λ and pins magnified/uniform λ (<0) to level 0. */
static LLVMValueRef
emit_tex_lod_clamp(struct vp_tr *t, LLVMValueRef lam)
{
   LLVMValueRef o_min = LLVMConstInt(t->i32, offsetof(gfx_sw_texstate_t, min_lod), false);
   LLVMValueRef o_max = LLVMConstInt(t->i32, offsetof(gfx_sw_texstate_t, max_lod), false);
   LLVMValueRef vmin = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &o_min, 1, ""), "min_lod");
   LLVMValueRef vmax = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &o_max, 1, ""), "max_lod");
   lam = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntSLT, lam, vmin, ""), vmin, lam, "");
   lam = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntSGT, lam, vmax, ""), vmax, lam, "");
   return lam;
}

/* Implicit-LOD sample of a mipmapped texture through the SW sampler: the HW TEX
 * unit has one per-draw filter DCR and no inter-level blend, so a mipmapped sampler
 * routes here. Resolve the tap filter per fragment from the sign of lambda (minified
 * -> min tap, magnified -> mag tap), pick the mip level (round-to-nearest for
 * mip-nearest, Q(VX_TEX_LOD_FRAC_BITS) fractional lambda for mip-linear), and sample.
 * Must run with the whole quad active (the derivative SHFL). */
static void
emit_tex_resolve_auto_lod(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v,
                          LLVMValueRef shader_bias, LLVMValueRef *out_lod,
                          LLVMValueRef *out_filter)
{
   LLVMValueRef rho  = emit_tex_grad_rho(t, u, v);
   LLVMValueRef filt = emit_tex_filter_word(t);
   LLVMValueRef zero64 = LLVMConstInt(t->i64, 0, false);
   LLVMValueRef zero32 = LLVMConstInt(t->i32, 0, false);

   LLVMValueRef mip_lin = LLVMBuildICmp(t->b, LLVMIntNE,
      LLVMBuildAnd(t->b, filt,
         LLVMConstInt(t->i32, VX_TEX_FILTER_MIP_LINEAR, false), ""),
      zero32, "mip_lin");
   LLVMValueRef mip_en = LLVMBuildICmp(t->b, LLVMIntNE,
      LLVMBuildAnd(t->b, filt,
         LLVMConstInt(t->i32, GFX_SW_TEX_FILTER_MIP_ENABLE, false), ""),
      zero32, "mip_en");

   /* Continuous signed LOD λ in Q(VX_TEX_LOD_FRAC_BITS) from the gradient rho
    * (which carries VP_TEX_FXD_FRAC frac bits): λ = log2(rho) − FXD. Magnified
    * fragments (rho < 2^FXD) give a negative integer part. */
   LLVMValueRef msb = LLVMBuildSub(t->b, LLVMConstInt(t->i64, 63, false),
                                   emit_ctlz64(t, rho), "msb");
   LLVMValueRef one_msb = LLVMBuildShl(t->b, LLVMConstInt(t->i64, 1, false), msb, "one_msb");
   LLVMValueRef ip = LLVMBuildShl(t->b,
      LLVMBuildSub(t->b, msb, LLVMConstInt(t->i64, VP_TEX_FXD_FRAC, false), ""),
      LLVMConstInt(t->i64, VX_TEX_LOD_FRAC_BITS, false), "");
   LLVMValueRef mant = LLVMBuildAnd(t->b, rho,
      LLVMBuildSub(t->b, one_msb, LLVMConstInt(t->i64, 1, false), ""), "");
   LLVMValueRef fp = LLVMBuildLShr(t->b,
      LLVMBuildShl(t->b, mant, LLVMConstInt(t->i64, VX_TEX_LOD_FRAC_BITS, false), ""),
      msb, "");
   LLVMValueRef lam64 = LLVMBuildAdd(t->b, ip, fp, "lambda_q8");
   /* rho==0 (uniform quad): λ = −inf → a large negative sentinel (the msb/one_msb
    * math is poison for rho==0; the select discards it). */
   lam64 = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntEQ, rho, zero64, ""),
      LLVMConstInt(t->i64, (uint64_t)(int64_t)(-(64 << VX_TEX_LOD_FRAC_BITS)), true),
      lam64, "");

   /* Vulkan LOD pipeline: λ' = λ + lodBias (min/mag decision uses λ', BEFORE the
    * min/max clamp); the sampled level uses clamp(λ', minLod, maxLod). */
   LLVMValueRef lam_b = LLVMBuildAdd(t->b, LLVMBuildTrunc(t->b, lam64, t->i32, ""),
                                     emit_tex_lod_bias(t), "lam_biased");
   /* textureBias(): the shader's per-fragment LOD bias adds on top of the sampler
    * mipLodBias, before the min/max clamp. */
   if (shader_bias)
      lam_b = LLVMBuildAdd(t->b, lam_b, shader_bias, "lam_shbias");
   /* Sampled λ = clamp(λ + bias, minLod, maxLod). Vulkan/deqp derive min-vs-mag
    * from the CLAMPED λ (so minLod>0 forces minification, and the level follows);
    * a non-mipmapped sampler always reads the base level. */
   LLVMValueRef lam_c = emit_tex_lod_clamp(t, lam_b);
   LLVMValueRef minified = LLVMBuildICmp(t->b, LLVMIntSGT, lam_c, zero32, "minified");
   LLVMValueRef lam = LLVMBuildSelect(t->b, mip_en, lam_c, zero32, "");

   /* tap: min (bit3) when minified, mag (bit0) otherwise. */
   LLVMValueRef mag_tap = LLVMBuildAnd(t->b, filt, LLVMConstInt(t->i32, 1, false), "mag_tap");
   LLVMValueRef min_tap = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, filt, LLVMConstInt(t->i32, 3, false), ""),
      LLVMConstInt(t->i32, 1, false), "min_tap");
   LLVMValueRef tap = LLVMBuildSelect(t->b, minified, min_tap, mag_tap, "tap");

   /* Inter-level blend only for a minified fragment of a mip-linear sampler with a
    * real mip chain (mip-enable). mip-linear consumes the Q8 λ; otherwise the
    * rounded integer level (0 when magnified or non-mipmapped). */
   LLVMValueRef mip_active = LLVMBuildAnd(t->b, LLVMBuildAnd(t->b, minified, mip_lin, ""),
                                          mip_en, "mip_active");
   LLVMValueRef level = LLVMBuildAShr(t->b,
      LLVMBuildAdd(t->b, lam,
         LLVMConstInt(t->i32, 1 << (VX_TEX_LOD_FRAC_BITS - 1), false), ""),
      LLVMConstInt(t->i32, VX_TEX_LOD_FRAC_BITS, false), "level");
   LLVMValueRef lod = LLVMBuildSelect(t->b, mip_active, lam, level, "lod");
   LLVMValueRef mip_bit = LLVMBuildSelect(t->b, mip_active,
      LLVMConstInt(t->i32, VX_TEX_FILTER_MIP_LINEAR, false), zero32, "");
   *out_filter = LLVMBuildOr(t->b, tap, mip_bit, "sw_filter");
   *out_lod = lod;
}

static LLVMValueRef
emit_tex_sw_resolved(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v,
                     LLVMValueRef shader_bias)
{
   LLVMValueRef lod, filter;
   emit_tex_resolve_auto_lod(t, u, v, shader_bias, &lod, &filter);
   return emit_tex_sw(t, u, v, lod, filter);
}

/* Same sample, delivered as four floats in the scratch slot. */
static LLVMValueRef
emit_tex_sw_resolved_f32(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v,
                         LLVMValueRef shader_bias)
{
   LLVMValueRef lod, filter;
   emit_tex_resolve_auto_lod(t, u, v, shader_bias, &lod, &filter);
   return emit_tex_sw_f32(t, u, v, lod, filter);
}

/* Encode an EXPLICIT float lambda into the LOD value the SW sampler expects for
 * the bound sampler's mip mode: Q(VX_TEX_LOD_FRAC_BITS) fixed-point when mip-linear
 * (the sampler splits it into floor level + blend), else round(lambda) as an
 * integer level; a non-mipmapped sampler (mip-enable clear) forces level 0. Shared
 * by the 2D textureLod path and the array/cube entries (which read the same
 * descriptor filter). `lam_bits` is the lod source's raw i32 (f32 bits). */
static LLVMValueRef
emit_encode_explicit_lod(struct vp_tr *t, LLVMValueRef lam_bits)
{
   LLVMValueRef filt = emit_tex_filter_word(t);
   LLVMValueRef zerof = LLVMConstReal(t->f32, 0.0);
   LLVMValueRef lam = LLVMBuildBitCast(t->b, lam_bits, t->f32, "lambda");
   LLVMValueRef minified = LLVMBuildFCmp(t->b, LLVMRealOGT, lam, zerof, "minified");
   LLVMValueRef lam_c = LLVMBuildSelect(t->b, minified, lam, zerof, "lam_c");

   LLVMValueRef mip_lin = LLVMBuildICmp(t->b, LLVMIntNE,
      LLVMBuildAnd(t->b, filt,
         LLVMConstInt(t->i32, VX_TEX_FILTER_MIP_LINEAR, false), ""),
      LLVMConstInt(t->i32, 0, false), "mip_lin");

   LLVMValueRef maxlod_i = LLVMConstInt(t->i32, VX_TEX_LOD_MAX, false);
   LLVMValueRef lod_near = LLVMBuildFPToSI(t->b,
      LLVMBuildFAdd(t->b, lam_c, LLVMConstReal(t->f32, 0.5), ""), t->i32, "lod_near");
   lod_near = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntSGT, lod_near, maxlod_i, ""),
                              maxlod_i, lod_near, "");
   LLVMValueRef maxlod_q = LLVMConstInt(t->i32, VX_TEX_LOD_MAX << VX_TEX_LOD_FRAC_BITS, false);
   LLVMValueRef lod_lin = LLVMBuildFPToSI(t->b,
      LLVMBuildFMul(t->b, lam_c,
         LLVMConstReal(t->f32, (double)(1u << VX_TEX_LOD_FRAC_BITS)), ""), t->i32, "lod_lin");
   lod_lin = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntSGT, lod_lin, maxlod_q, ""),
                             maxlod_q, lod_lin, "");

   LLVMValueRef lod = LLVMBuildSelect(t->b, mip_lin, lod_lin, lod_near, "lod");
   LLVMValueRef mip_en = LLVMBuildICmp(t->b, LLVMIntNE,
      LLVMBuildAnd(t->b, filt,
         LLVMConstInt(t->i32, GFX_SW_TEX_FILTER_MIP_ENABLE, false), ""),
      LLVMConstInt(t->i32, 0, false), "mip_en");
   return LLVMBuildSelect(t->b, mip_en, lod, LLVMConstInt(t->i32, 0, false), "");
}

/* textureLod: sample with an EXPLICIT lambda (float lod source) rather than the
 * quad-gradient-derived rho. Same tap + mip resolution as emit_tex_sw_resolved,
 * but driven by lambda directly: minified when lambda>0 (pick the min tap, else
 * mag); the mip level is encoded by emit_encode_explicit_lod. `lam_bits` is the
 * lod source's raw i32 (f32 bits). */
static void
emit_tex_resolve_explicit_lod(struct vp_tr *t, LLVMValueRef lam_bits,
                              LLVMValueRef *out_lod, LLVMValueRef *out_filter)
{
   LLVMValueRef filt = emit_tex_filter_word(t);
   LLVMValueRef zerof = LLVMConstReal(t->f32, 0.0);
   LLVMValueRef lam = LLVMBuildBitCast(t->b, lam_bits, t->f32, "lambda");
   LLVMValueRef minified = LLVMBuildFCmp(t->b, LLVMRealOGT, lam, zerof, "minified");

   /* tap: min (bit3) when minified, mag (bit0) otherwise. */
   LLVMValueRef mag_tap = LLVMBuildAnd(t->b, filt, LLVMConstInt(t->i32, 1, false), "mag_tap");
   LLVMValueRef min_tap = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, filt, LLVMConstInt(t->i32, 3, false), ""),
      LLVMConstInt(t->i32, 1, false), "min_tap");
   LLVMValueRef tap = LLVMBuildSelect(t->b, minified, min_tap, mag_tap, "tap");

   LLVMValueRef mip_lin = LLVMBuildICmp(t->b, LLVMIntNE,
      LLVMBuildAnd(t->b, filt,
         LLVMConstInt(t->i32, VX_TEX_FILTER_MIP_LINEAR, false), ""),
      LLVMConstInt(t->i32, 0, false), "mip_lin");

   *out_lod = emit_encode_explicit_lod(t, lam_bits);
   LLVMValueRef mip_bit = LLVMBuildSelect(t->b,
      LLVMBuildAnd(t->b, minified, mip_lin, ""),
      LLVMConstInt(t->i32, VX_TEX_FILTER_MIP_LINEAR, false),
      LLVMConstInt(t->i32, 0, false), "");
   *out_filter = LLVMBuildOr(t->b, tap, mip_bit, "sw_filter");
}

static LLVMValueRef
emit_tex_sw_lod(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v, LLVMValueRef lam_bits)
{
   LLVMValueRef lod, filter;
   emit_tex_resolve_explicit_lod(t, lam_bits, &lod, &filter);
   return emit_tex_sw(t, u, v, lod, filter);
}

/* Same sample, delivered as four floats in the scratch slot. */
static LLVMValueRef
emit_tex_sw_lod_f32(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v, LLVMValueRef lam_bits)
{
   LLVMValueRef lod, filter;
   emit_tex_resolve_explicit_lod(t, lam_bits, &lod, &filter);
   return emit_tex_sw_f32(t, u, v, lod, filter);
}

/* mip-0 {width,height} from the resident TEX descriptor, as i32. A descriptor
 * whose width/height is 0 is a power-of-two texture; derive the dim from logdim. */
static void
emit_tex_mip0_dims(struct vp_tr *t, LLVMValueRef *out_w, LLVMValueRef *out_h)
{
   LLVMValueRef one = LLVMConstInt(t->i32, 1, false);
   LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
   LLVMValueRef off_ld = LLVMConstInt(t->i32, offsetof(gfx_sw_texstate_t, logdim), false);
   LLVMValueRef logdim = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &off_ld, 1, ""), "logdim");
   LLVMValueRef off_w = LLVMConstInt(t->i32, offsetof(gfx_sw_texstate_t, width), false);
   LLVMValueRef w0 = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &off_w, 1, ""), "twidth");
   LLVMValueRef off_h = LLVMConstInt(t->i32, offsetof(gfx_sw_texstate_t, height), false);
   LLVMValueRef h0 = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &off_h, 1, ""), "theight");
   LLVMValueRef potw = LLVMBuildShl(t->b, one,
      LLVMBuildAnd(t->b, logdim, LLVMConstInt(t->i32, 0xffff, false), ""), "");
   LLVMValueRef poth = LLVMBuildShl(t->b, one,
      LLVMBuildLShr(t->b, logdim, LLVMConstInt(t->i32, 16, false), ""), "");
   *out_w = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntNE, w0, zero, ""), w0, potw, "w0");
   *out_h = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntNE, h0, zero, ""), h0, poth, "h0");
}

/* textureSize (also the intermediate emitted when textureGrad's txd is lowered to
 * a size query + explicit-LOD sample): the (width,height) of mip level `lod` from
 * the resident TEX descriptor -- no sample. The def is integer: component c =
 * max(dim >> lod, 1). */
static void
emit_tex_size(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef lod_int)
{
   LLVMValueRef lod = lod_int ? lod_int : LLVMConstInt(t->i32, 0, false);
   LLVMValueRef one  = LLVMConstInt(t->i32, 1, false);
   LLVMValueRef w0, h0;
   emit_tex_mip0_dims(t, &w0, &h0);

   /* level dim = max(dim >> lod, 1). */
   LLVMValueRef wl = LLVMBuildLShr(t->b, w0, lod, "");
   LLVMValueRef hl = LLVMBuildLShr(t->b, h0, lod, "");
   wl = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntULT, wl, one, ""), one, wl, "wl");
   hl = LLVMBuildSelect(t->b, LLVMBuildICmp(t->b, LLVMIntULT, hl, one, ""), one, hl, "hl");

   /* Third component: the descriptor's depth field carries the layer count for
    * arrays (cube count for cube arrays) and the slice count for 3D. Layers do
    * not shrink with the mip level; 3D slices halve like the other axes. */
   LLVMValueRef dims[3] = { wl, hl, one };
   if (tex->def.num_components >= 3) {
      LLVMValueRef off_d = LLVMConstInt(t->i32,
         offsetof(gfx_sw_texstate_t, depth), false);
      LLVMValueRef d0 = LLVMBuildLoad2(t->b, t->i32,
         LLVMBuildGEP2(t->b, t->i8, t->fs_texstate, &off_d, 1, ""), "tdepth");
      LLVMValueRef dl = (tex->sampler_dim == GLSL_SAMPLER_DIM_3D)
         ? LLVMBuildLShr(t->b, d0, lod, "")
         : d0;
      dims[2] = LLVMBuildSelect(t->b,
         LLVMBuildICmp(t->b, LLVMIntULT, dl, one, ""), one, dl, "dl");
   }
   for (unsigned c = 0; c < tex->def.num_components && c < 3; c++)
      ssa_set(t, tex->def.index, c, dims[c]);
}

/* Write a sampled vec4 to the def, applying the view's component swizzle. `chan`
 * holds the four source channels (R,G,B,A) as i32 bit patterns, and `czero`/`cone`
 * are the swizzle's constant arms in the same encoding -- a float sampler passes
 * bitcast 0.0/1.0, an integer sampler passes 0/1. Output component c takes source
 * channel map (0..3 = R/G/B/A), or a constant (map 4 -> czero, map 5 -> cone);
 * those stay literals, so a VK_COMPONENT_SWIZZLE_ONE returns one whatever the texel
 * holds. Identity (1672) is a passthrough. */
static void
emit_tex_swizzle_store(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef chan[4],
                       LLVMValueRef czero, LLVMValueRef cone)
{
   LLVMValueRef swz = emit_tex_swizzle_word(t);
   for (unsigned c = 0; c < tex->def.num_components && c < 4; c++) {
      LLVMValueRef map = LLVMBuildAnd(t->b,
         LLVMBuildLShr(t->b, swz, LLVMConstInt(t->i32, c * 3, false), ""),
         LLVMConstInt(t->i32, 0x7, false), "map");
      LLVMValueRef is0 = LLVMBuildICmp(t->b, LLVMIntEQ, map, LLVMConstInt(t->i32, 0, false), "");
      LLVMValueRef is1 = LLVMBuildICmp(t->b, LLVMIntEQ, map, LLVMConstInt(t->i32, 1, false), "");
      LLVMValueRef is2 = LLVMBuildICmp(t->b, LLVMIntEQ, map, LLVMConstInt(t->i32, 2, false), "");
      LLVMValueRef v = LLVMBuildSelect(t->b, is0, chan[0],
                       LLVMBuildSelect(t->b, is1, chan[1],
                       LLVMBuildSelect(t->b, is2, chan[2], chan[3], ""), ""), "src_chan");
      LLVMValueRef is4 = LLVMBuildICmp(t->b, LLVMIntEQ, map, LLVMConstInt(t->i32, 4, false), "");
      LLVMValueRef is5 = LLVMBuildICmp(t->b, LLVMIntEQ, map, LLVMConstInt(t->i32, 5, false), "");
      v = LLVMBuildSelect(t->b, is4, czero, v, "");
      v = LLVMBuildSelect(t->b, is5, cone, v, "");
      ssa_set(t, tex->def.index, c, v);
   }
}

/* The swizzle constants for a float sampler: 0.0 and 1.0 as i32 bit patterns. */
static void
tex_swizzle_float_consts(struct vp_tr *t, LLVMValueRef *czero, LLVMValueRef *cone)
{
   *czero = LLVMBuildBitCast(t->b, LLVMConstReal(t->f32, 0.0), t->i32, "");
   *cone  = LLVMBuildBitCast(t->b, LLVMConstReal(t->f32, 1.0), t->i32, "");
}

/* Write four stored bytes to an integer sampler's def as the raw values they hold:
 * a signed sampler reinterprets each as int8 (the stored value spans -128..127), an
 * unsigned one takes it as-is. No normalisation -- an isampler/usampler yields the
 * stored integer and the shader's own vec4() cast converts.
 *
 * Exact only because the integer formats reach the sampler as VX_TEX_FORMAT_A8R8G8B8
 * (vp_vx_tex_format), whose decode preserves every byte. Giving them their own VX
 * format above VX_TEX_FORMAT_FF_MAX would break this silently -- TexDecodeExtended's
 * default arm returns a=0xff, rgb=0. */
static void
emit_tex_store_int8_chan(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef byte[4],
                         bool is_signed)
{
   LLVMValueRef chan[4];
   for (unsigned s = 0; s < 4; s++) {
      /* byte -> int8: (v ^ 0x80) - 0x80 */
      chan[s] = is_signed
         ? LLVMBuildSub(t->b,
              LLVMBuildXor(t->b, byte[s], LLVMConstInt(t->i32, 0x80, false), ""),
              LLVMConstInt(t->i32, 0x80, false), "sext8")
         : byte[s];
   }
   emit_tex_swizzle_store(t, tex, chan, LLVMConstInt(t->i32, 0, false),
                          LLVMConstInt(t->i32, 1, false));
}

/* Unpack a float[4] scratch slot (emit_tex_fetch_f32) to the def's vec4. The
 * channels are already the sampled values -- no /255 -- so a float-format texel
 * keeps its real magnitude. */
static void
emit_tex_unpack_f32(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef out)
{
   LLVMValueRef chan[4];
   for (unsigned s = 0; s < 4; s++) {
      LLVMValueRef idx[2] = { LLVMConstInt(t->i32, 0, false),
                              LLVMConstInt(t->i32, s, false) };
      LLVMValueRef p = LLVMBuildGEP2(t->b, LLVMArrayType(t->f32, 4), out, idx, 2, "");
      chan[s] = LLVMBuildBitCast(t->b,
         LLVMBuildLoad2(t->b, t->f32, p, "texel_f"), t->i32, "");
   }
   LLVMValueRef czero, cone;
   tex_swizzle_float_consts(t, &czero, &cone);
   emit_tex_swizzle_store(t, tex, chan, czero, cone);
}

/* Unpack an int32[4] scratch slot (emit_tex_fetch_i32) to an integer sampler's
 * def. The channels arrive as the raw 0..255 stored bytes. */
static void
emit_tex_unpack_i32(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef out,
                    bool is_signed)
{
   LLVMValueRef byte[4];
   for (unsigned s = 0; s < 4; s++) {
      LLVMValueRef idx[2] = { LLVMConstInt(t->i32, 0, false),
                              LLVMConstInt(t->i32, s, false) };
      LLVMValueRef p = LLVMBuildGEP2(t->b, LLVMArrayType(t->i32, 4), out, idx, 2, "");
      byte[s] = LLVMBuildLoad2(t->b, t->i32, p, "texel_i");
   }
   emit_tex_store_int8_chan(t, tex, byte, is_signed);
}

/* Does this sampler yield integers (isampler/usampler) rather than floats? Decides
 * which carrier a sample travels in and how it is written to the def.
 *
 * The BASE type only, never the size bits: lavapipe runs nir_opt_16bit_tex_image with
 * int/uint in opt_tex_dest_types, so dest_type may arrive narrowed. */
static bool
tex_dest_is_int(const nir_tex_instr *tex)
{
   nir_alu_type base = nir_alu_type_get_base_type(tex->dest_type);
   return base == nir_type_int || base == nir_type_uint;
}

/* Split a packed A8R8G8B8 word into its four source channel bytes (R,G,B,A). */
static void
tex_argb_bytes(struct vp_tr *t, LLVMValueRef texel, LLVMValueRef byte[4])
{
   static const unsigned shift[4] = { 16, 8, 0, 24 };
   for (unsigned s = 0; s < 4; s++)
      byte[s] = LLVMBuildAnd(t->b,
         LLVMBuildLShr(t->b, texel, LLVMConstInt(t->i32, shift[s], false), ""),
         LLVMConstInt(t->i32, 0xff, false), "");
}

/* Scale four channel bytes to [0,1]. The multiply-by-reciprocal (not a divide) is
 * what the SW sampler's float decode also uses, so the two agree bit-for-bit on a
 * non-float format. */
static void
tex_bytes_to_f32(struct vp_tr *t, LLVMValueRef byte[4], LLVMValueRef out[4])
{
   LLVMValueRef c1 = LLVMConstReal(t->f32, 1.0 / 255.0);
   for (unsigned s = 0; s < 4; s++)
      out[s] = LLVMBuildFMul(t->b, LLVMBuildUIToFP(t->b, byte[s], t->f32, ""), c1, "");
}

/* Unpack a packed A8R8G8B8 texel into the tex def's components: floats [0,1] for a
 * float sampler, the raw stored integers for an isampler/usampler. This is the common
 * tail of every packed-texel sampler (2D, array, cube, cube-array, 3D) on both the SW
 * and HW paths, so the sampler's result type is resolved here rather than in each. */
static void
emit_tex_unpack(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef texel)
{
   LLVMValueRef byte[4];
   tex_argb_bytes(t, texel, byte);

   /* An integer sampler must yield the stored value, not a normalised colour. Integer
    * formats sample point-only (Vulkan gates VK_FILTER_LINEAR on a format feature no
    * integer format advertises), so the packed word holds the four stored bytes
    * unblended. */
   if (tex_dest_is_int(tex)) {
      const bool is_signed =
         nir_alu_type_get_base_type(tex->dest_type) == nir_type_int;
      emit_tex_store_int8_chan(t, tex, byte, is_signed);
      return;
   }

   LLVMValueRef f[4], chan[4];
   tex_bytes_to_f32(t, byte, f);
   for (unsigned s = 0; s < 4; s++)
      chan[s] = LLVMBuildBitCast(t->b, f[s], t->i32, "");
   LLVMValueRef czero, cone;
   tex_swizzle_float_consts(t, &czero, &cone);
   emit_tex_swizzle_store(t, tex, chan, czero, cone);
}

/* Expand a packed A8R8G8B8 texel into the float[4] scratch slot, so a packed
 * producer (the HW TEX unit) can feed the same float unpack the float-returning SW
 * sampler feeds. Returns the slot. */
static LLVMValueRef
emit_tex_argb_to_f32_scratch(struct vp_tr *t, LLVMValueRef texel)
{
   LLVMValueRef byte[4], f[4];
   tex_argb_bytes(t, texel, byte);
   tex_bytes_to_f32(t, byte, f);
   LLVMValueRef out = tex_f32_scratch(t);
   for (unsigned s = 0; s < 4; s++) {
      LLVMValueRef idx[2] = { LLVMConstInt(t->i32, 0, false),
                              LLVMConstInt(t->i32, s, false) };
      LLVMValueRef p = LLVMBuildGEP2(t->b, LLVMArrayType(t->f32, 4), out, idx, 2, "");
      LLVMBuildStore(t->b, f[s], p);
   }
   return out;
}

/* The software arm of a 2D sampling op, in whichever carrier the sampler's result
 * type calls for: textureLod samples the explicit lambda, textureBias adds its bias
 * to the quad-gradient LOD, and plain texture() takes the gradient alone. */
static LLVMValueRef
emit_tex_sw_arm(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef ux, LLVMValueRef vx,
                LLVMValueRef lod_bits, LLVMValueRef bias_q8, bool as_f32)
{
   if (tex->op == nir_texop_txl)
      return as_f32 ? emit_tex_sw_lod_f32(t, ux, vx, lod_bits)
                    : emit_tex_sw_lod(t, ux, vx, lod_bits);

   LLVMValueRef bias = (tex->op == nir_texop_txb) ? bias_q8 : NULL;
   return as_f32 ? emit_tex_sw_resolved_f32(t, ux, vx, bias)
                 : emit_tex_sw_resolved(t, ux, vx, bias);
}

/* sampler2DShadow: gfx_tex_shadow_sw(&texstate[0], x, y, ref_bits, filter) --
 * sample the depth texture at (x,y), compare each tap against ref with the
 * descriptor's compare_func, and return the 0..1 result as a float bit-pattern
 * (0/1 point, PCF fraction for a bilinear sampler). The result is a scalar float,
 * so write it directly to every def component (bypassing the ARGB unpack). */
static void
emit_tex_shadow(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef x, LLVMValueRef y,
                LLVMValueRef ref_bits, LLVMValueRef lod)
{
   LLVMTypeRef params[6] = { t->ptr, t->i32, t->i32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 6, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_shadow_sw");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_shadow_sw", fty);
   LLVMValueRef a[6] = { t->fs_texstate, x, y, ref_bits, emit_tex_filter_word(t), lod };
   LLVMValueRef res = LLVMBuildCall2(t->b, fty, fn, a, 6, "shadow");
   for (unsigned c = 0; c < tex->def.num_components && c < 4; c++)
      ssa_set(t, tex->def.index, c, res);
}

/* sampler2DArrayShadow: like emit_tex_shadow, but the integer `layer` selects the
 * array slice before the depth compare (gfx_tex_shadow_array_sw). */
static void
emit_tex_shadow_array(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef x,
                      LLVMValueRef y, LLVMValueRef layer, LLVMValueRef ref_bits,
                      LLVMValueRef lod)
{
   LLVMTypeRef params[7] = { t->ptr, t->i32, t->i32, t->i32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 7, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_shadow_array_sw");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_shadow_array_sw", fty);
   LLVMValueRef a[7] = { t->fs_texstate, x, y, layer, ref_bits,
                         emit_tex_filter_word(t), lod };
   LLVMValueRef res = LLVMBuildCall2(t->b, fty, fn, a, 7, "shadowarray");
   for (unsigned c = 0; c < tex->def.num_components && c < 4; c++)
      ssa_set(t, tex->def.index, c, res);
}

/* samplerCubeShadow: the (sc,tc,rc) direction picks the face + projects; compare
 * against ref at that face's slice (gfx_tex_shadow_cube_sw). */
static void
emit_tex_shadow_cube(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef sc,
                     LLVMValueRef tc, LLVMValueRef rc, LLVMValueRef ref_bits,
                     LLVMValueRef lod)
{
   LLVMTypeRef params[7] = { t->ptr, t->f32, t->f32, t->f32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 7, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_shadow_cube_sw");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_shadow_cube_sw", fty);
   LLVMValueRef a[7] = { t->fs_texstate, sc, tc, rc, ref_bits,
                         emit_tex_filter_word(t), lod };
   LLVMValueRef res = LLVMBuildCall2(t->b, fty, fn, a, 7, "shadowcube");
   for (unsigned c = 0; c < tex->def.num_components && c < 4; c++)
      ssa_set(t, tex->def.index, c, res);
}

/* samplerCubeArrayShadow: array_index selects the cube, (sc,tc,rc) the face;
 * compare against ref at slice array_index*6 + face. */
static void
emit_tex_shadow_cube_array(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef sc,
                           LLVMValueRef tc, LLVMValueRef rc, LLVMValueRef array_index,
                           LLVMValueRef ref_bits, LLVMValueRef lod)
{
   LLVMTypeRef params[8] = { t->ptr, t->f32, t->f32, t->f32, t->i32, t->i32, t->i32,
                             t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 8, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_shadow_cube_array_sw");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_shadow_cube_array_sw", fty);
   LLVMValueRef a[8] = { t->fs_texstate, sc, tc, rc, array_index, ref_bits,
                         emit_tex_filter_word(t), lod };
   LLVMValueRef res = LLVMBuildCall2(t->b, fty, fn, a, 8, "shadowcubearray");
   for (unsigned c = 0; c < tex->def.num_components && c < 4; c++)
      ssa_set(t, tex->def.index, c, res);
}

/* sampler2DArray: gfx_tex_sample_array_sw(&texstate[0], x, y, layer, lod) -- sample
 * the integer `layer` slice at (x,y) of the given LOD, then unpack the A8R8G8B8
 * texel to the def's vec4. The layer stride comes from the resident descriptor. */
static void
emit_tex_array(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef x, LLVMValueRef y,
               LLVMValueRef layer, LLVMValueRef lod)
{
   /* A float sampler takes the four-float form: a float-format texel leaves [0,1] and
    * cannot survive the packed word. An array sample is always software, so unlike the
    * 2D path there is no second arm to merge. */
   if (!tex_dest_is_int(tex)) {
      LLVMTypeRef params[6] = { t->ptr, t->i32, t->i32, t->i32, t->i32, t->ptr };
      LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), params, 6, false);
      LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_array_f32");
      if (!fn)
         fn = LLVMAddFunction(t->mod, "gfx_tex_sample_array_f32", fty);
      LLVMValueRef out = tex_f32_scratch(t);
      LLVMValueRef a[6] = { t->fs_texstate, x, y, layer, lod, out };
      LLVMBuildCall2(t->b, fty, fn, a, 6, "");
      emit_tex_unpack_f32(t, tex, out);
      return;
   }

   LLVMTypeRef params[5] = { t->ptr, t->i32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 5, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_array_sw");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_sample_array_sw", fty);
   LLVMValueRef a[5] = { t->fs_texstate, x, y, layer, lod };
   emit_tex_unpack(t, tex, LLVMBuildCall2(t->b, fty, fn, a, 5, "texarray"));
}

/* samplerCube: gfx_tex_sample_cube_sw(&texstate[0], sc, tc, rc, lod) -- pick the
 * face from the major axis of the (sc,tc,rc) direction, project, and sample; the
 * six faces are slices of the resident descriptor (layer_stride apart). Unpack the
 * A8R8G8B8 texel to the def's vec4. */
static void
emit_tex_cube(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef sc, LLVMValueRef tc,
              LLVMValueRef rc, LLVMValueRef lod)
{
   /* A float sampler takes the four-float form (see emit_tex_array); a cube sample is
    * always software, so there is no second arm to merge. */
   if (!tex_dest_is_int(tex)) {
      LLVMTypeRef params[6] = { t->ptr, t->f32, t->f32, t->f32, t->i32, t->ptr };
      LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), params, 6, false);
      LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_cube_f32");
      if (!fn)
         fn = LLVMAddFunction(t->mod, "gfx_tex_sample_cube_f32", fty);
      LLVMValueRef out = tex_f32_scratch(t);
      LLVMValueRef a[6] = { t->fs_texstate, sc, tc, rc, lod, out };
      LLVMBuildCall2(t->b, fty, fn, a, 6, "");
      emit_tex_unpack_f32(t, tex, out);
      return;
   }

   LLVMTypeRef params[5] = { t->ptr, t->f32, t->f32, t->f32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 5, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_cube_sw");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_sample_cube_sw", fty);
   LLVMValueRef a[5] = { t->fs_texstate, sc, tc, rc, lod };
   emit_tex_unpack(t, tex, LLVMBuildCall2(t->b, fty, fn, a, 5, "texcube"));
}

/* samplerCubeArray: `array_index` selects the cube, (sc,tc,rc) the face; the SW
 * entry addresses the slice array_index*6 + face. */
static void
emit_tex_cube_array(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef sc,
                    LLVMValueRef tc, LLVMValueRef rc, LLVMValueRef array_index,
                    LLVMValueRef lod)
{
   /* A float sampler takes the four-float form (see emit_tex_array); a cube-array
    * sample is always software, so there is no second arm to merge. */
   if (!tex_dest_is_int(tex)) {
      LLVMTypeRef params[7] = { t->ptr, t->f32, t->f32, t->f32, t->i32, t->i32, t->ptr };
      LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), params, 7, false);
      LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_cube_array_f32");
      if (!fn)
         fn = LLVMAddFunction(t->mod, "gfx_tex_sample_cube_array_f32", fty);
      LLVMValueRef out = tex_f32_scratch(t);
      LLVMValueRef a[7] = { t->fs_texstate, sc, tc, rc, array_index, lod, out };
      LLVMBuildCall2(t->b, fty, fn, a, 7, "");
      emit_tex_unpack_f32(t, tex, out);
      return;
   }

   LLVMTypeRef params[6] = { t->ptr, t->f32, t->f32, t->f32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 6, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_cube_array_sw");
   if (!fn)
      fn = LLVMAddFunction(t->mod, "gfx_tex_sample_cube_array_sw", fty);
   LLVMValueRef a[6] = { t->fs_texstate, sc, tc, rc, array_index, lod };
   emit_tex_unpack(t, tex, LLVMBuildCall2(t->b, fty, fn, a, 6, "texcubearray"));
}

/* sampler3D: sample at S.23 (u,v,w). The LOD and the min/mag tap come from the
 * shared resolvers, so a 3D sample follows the same lambda rules as a 2D one. A
 * mip-linear sampler passes the Q8 lambda with the mip-linear bit set so the sampler
 * blends the two bracketing levels (full trilinear = 2 levels x 2 slices x bilinear);
 * otherwise it passes the rounded integer level (nearest-mip). The sampler picks the
 * depth slice from w (nearest slice, or a linear blend of the two bracketing slices
 * on a linear tap). A non-mipmapped sampler forces level 0.
 *
 * The auto-LOD lambda still comes from the u,v gradient alone, so a volume whose
 * depth axis is the steepest gradient picks too low a level. */
static void
emit_tex_3d(struct vp_tr *t, nir_tex_instr *tex, LLVMValueRef u, LLVMValueRef v,
            LLVMValueRef w, LLVMValueRef lod_bits, LLVMValueRef bias_q8)
{
   /* Resolve the mip level and per-fragment tap exactly as the 2D path does -- same
    * op-to-resolver mapping as emit_tex_sw_arm -- so every sampler shape derives its
    * LOD through one implementation. textureLod takes the explicit lambda; texture()
    * and textureBias take the quad gradient, the latter adding the shader bias. */
   LLVMValueRef lod, sw_filter;
   if (tex->op == nir_texop_txl) {
      emit_tex_resolve_explicit_lod(t, lod_bits, &lod, &sw_filter);
   } else {
      emit_tex_resolve_auto_lod(t, u, v,
                                tex->op == nir_texop_txb ? bias_q8 : NULL,
                                &lod, &sw_filter);
   }

   /* A float sampler takes the four-float form (see emit_tex_array); a 3D sample is
    * always software, so there is no second arm to merge. The lambda and tap above
    * are shared -- only the final call differs. */
   if (!tex_dest_is_int(tex)) {
      LLVMTypeRef params[7] = { t->ptr, t->i32, t->i32, t->i32, t->i32, t->i32, t->ptr };
      LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx), params, 7, false);
      LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_3d_f32");
      if (!fn) {
         fn = LLVMAddFunction(t->mod, "gfx_tex_sample_3d_f32", fty);
      }
      LLVMValueRef out = tex_f32_scratch(t);
      LLVMValueRef a[7] = { t->fs_texstate, u, v, w, lod, sw_filter, out };
      LLVMBuildCall2(t->b, fty, fn, a, 7, "");
      emit_tex_unpack_f32(t, tex, out);
      return;
   }

   LLVMTypeRef params[6] = { t->ptr, t->i32, t->i32, t->i32, t->i32, t->i32 };
   LLVMTypeRef fty = LLVMFunctionType(t->i32, params, 6, false);
   LLVMValueRef fn = LLVMGetNamedFunction(t->mod, "gfx_tex_sample_3d_sw");
   if (!fn) {
      fn = LLVMAddFunction(t->mod, "gfx_tex_sample_3d_sw", fty);
   }
   LLVMValueRef a[6] = { t->fs_texstate, u, v, w, lod, sw_filter };
   emit_tex_unpack(t, tex, LLVMBuildCall2(t->b, fty, fn, a, 6, "tex3d"));
}

/* A NIR texture op: a 2D `texture()`/`textureLod()`/`textureBias()` sampling the
 * single bound texture (TEX stage 0), or `texelFetch()` (integer-coord fetch, no
 * filter). The interpolated texcoord is the coord source; the texture/sampler
 * deref sources are fixed-function (TEX DCRs). For implicit-LOD `tex`, the mip
 * level is derived from the quad's coordinate gradients; explicit-LOD/bias take
 * level 0 (no bias arithmetic yet). The result vec4 is the unpacked A8R8G8B8
 * texel as four floats in [0,1]. A sampler2DShadow op returns a single depth-
 * compare float instead (emit_tex_shadow). */
static void
emit_tex(struct vp_tr *t, nir_tex_instr *tex)
{
   /* Every texture route -- the SW sampler and the HW TEX path alike -- reads the
    * resident descriptor table for the filter word and the log2 dimensions. Only
    * the fragment entry carries that table, so a texture op in any other stage has
    * no descriptor to read. Refuse the shader rather than build a GEP on a null. */
   if (!t->fs_texstate) {
      t->ok = false;
      return;
   }
   /* The gather footprint is addressed in one face's 2D space, so a cube gather
    * would have to resolve taps that cross a face edge. The cube branches below
    * precede the gather one and would otherwise take a cube tg4 as an ordinary
    * cube sample -- a wrong result rather than a refusal. */
   if (tex->op == nir_texop_tg4 &&
       tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
      mesa_logw("vortexpipe: vp_nir_to_llvm: textureGather on a cube sampler "
                "is unimplemented");
      t->ok = false;
      return;
   }
   LLVMValueRef u = NULL, v = NULL, lod_int = NULL, bias_f = NULL;
   LLVMValueRef off_x = NULL, off_y = NULL, cmp = NULL;
   unsigned coord_ssa = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_coord) {
         coord_ssa = tex->src[i].src.ssa->index;
         u = ssa_get(t, tex->src[i].src.ssa->index, 0);
         v = ssa_get(t, tex->src[i].src.ssa->index, 1);
      } else if (tex->src[i].src_type == nir_tex_src_lod) {
         lod_int = ssa_get(t, tex->src[i].src.ssa->index, 0);
      } else if (tex->src[i].src_type == nir_tex_src_bias) {
         bias_f = ssa_get(t, tex->src[i].src.ssa->index, 0);
      } else if (tex->src[i].src_type == nir_tex_src_offset) {
         off_x = ssa_get(t, tex->src[i].src.ssa->index, 0);
         off_y = ssa_get(t, tex->src[i].src.ssa->index, 1);
      } else if (tex->src[i].src_type == nir_tex_src_comparator) {
         cmp = ssa_get(t, tex->src[i].src.ssa->index, 0);
      }
   }
   /* textureBias(): the shader LOD bias as signed Q(VX_TEX_LOD_FRAC_BITS). */
   LLVMValueRef bias_q8 = bias_f
      ? LLVMBuildFPToSI(t->b,
           LLVMBuildFMul(t->b, LLVMBuildBitCast(t->b, bias_f, t->f32, ""),
              LLVMConstReal(t->f32, (double)(1 << VX_TEX_LOD_FRAC_BITS)), ""),
           t->i32, "bias_q8")
      : NULL;

   /* Shadow mip level: textureLod / textureGrad (lowered to txl) carry an explicit
    * LOD; encode it as the colour path does (mip-nearest level or mip-linear Q,
    * gated by the sampler's mip-enable). texture()/textureProj sample the base
    * level (auto-LOD shadow is magnified in the demanded tests). */
   LLVMValueRef shadow_lod = (tex->op == nir_texop_txl && lod_int)
      ? emit_encode_explicit_lod(t, lod_int)
      : LLVMConstInt(t->i32, 0, false);

   /* sampler2DArrayShadow: coord.z is the array layer, the comparator src (or, when
    * folded into the coord, component 3) is the reference. Select the layer slice,
    * then compare. Handled before the plain 2D-shadow branch (which drops the
    * layer) and before the array-colour branch (which drops the compare). */
   if (tex->is_shadow && tex->is_array && tex->op != nir_texop_tg4 &&
       tex->sampler_dim == GLSL_SAMPLER_DIM_2D && u && v) {
      LLVMValueRef ref = cmp ? cmp : ssa_get(t, coord_ssa, 3);
      LLVMValueRef zf = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 2), t->f32, "");
      LLVMValueRef layer = LLVMBuildFPToSI(t->b,
         LLVMBuildFAdd(t->b, zf, LLVMConstReal(t->f32, 0.5), ""), t->i32, "layer");
      LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
      layer = LLVMBuildSelect(t->b,
         LLVMBuildICmp(t->b, LLVMIntSLT, layer, zero, ""), zero, layer, "");
      LLVMValueRef uf = LLVMBuildBitCast(t->b, u, t->f32, "uf");
      LLVMValueRef vf = LLVMBuildBitCast(t->b, v, t->f32, "vf");
      LLVMValueRef scale = LLVMConstReal(t->f32, (double)(1u << VP_TEX_FXD_FRAC));
      LLVMValueRef ux = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, uf, scale, ""), t->i32, "");
      LLVMValueRef vx = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, vf, scale, ""), t->i32, "");
      emit_tex_shadow_array(t, tex, ux, vx, layer, ref, shadow_lod);
      return;
   }

   /* samplerCubeArrayShadow: coord.xyz is the direction, coord.w the array index,
    * the comparator src the reference. Select the face + project, then compare at
    * slice array_index*6 + face. Handled before the plain cube-shadow (!is_array). */
   if (tex->is_shadow && tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
       tex->is_array && u && v) {
      LLVMValueRef sc = LLVMBuildBitCast(t->b, u, t->f32, "sc");
      LLVMValueRef tc = LLVMBuildBitCast(t->b, v, t->f32, "tc");
      LLVMValueRef rc = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 2), t->f32, "rc");
      LLVMValueRef wf = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 3), t->f32, "");
      LLVMValueRef idx = LLVMBuildFPToSI(t->b,
         LLVMBuildFAdd(t->b, wf, LLVMConstReal(t->f32, 0.5), ""), t->i32, "cubeidx");
      LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
      idx = LLVMBuildSelect(t->b,
         LLVMBuildICmp(t->b, LLVMIntSLT, idx, zero, ""), zero, idx, "");
      emit_tex_shadow_cube_array(t, tex, sc, tc, rc, idx, cmp, shadow_lod);
      return;
   }

   /* samplerCubeShadow: coord.xyz is the direction vector, the comparator src (or
    * folded coord component 3) is the reference. Select the face + project, then
    * compare. Handled before the plain 2D-shadow branch and the cube-colour branch. */
   if (tex->is_shadow && tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
       !tex->is_array && u && v) {
      LLVMValueRef ref = cmp ? cmp : ssa_get(t, coord_ssa, 3);
      LLVMValueRef sc = LLVMBuildBitCast(t->b, u, t->f32, "sc");
      LLVMValueRef tc = LLVMBuildBitCast(t->b, v, t->f32, "tc");
      LLVMValueRef rc = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 2), t->f32, "rc");
      emit_tex_shadow_cube(t, tex, sc, tc, rc, ref, shadow_lod);
      return;
   }

   /* sampler2DShadow: the depth-compare reference is the comparator src, or, when
    * a lowering folds it in, coord component 2. Sample the depth texture, compare
    * against ref per the sampler's compareOp, and return the 0..1 result (scalar).
    * A shadow op is still nir_texop_tex/txl, so handle it before that dispatch. A
    * textureGatherCmp (tg4) is a 2x2 gather, not a single compare -- fall through. */
   if (tex->is_shadow && tex->op != nir_texop_tg4 && u && v) {
      LLVMValueRef ref = cmp ? cmp : ssa_get(t, coord_ssa, 2);
      LLVMValueRef uf = LLVMBuildBitCast(t->b, u, t->f32, "uf");
      LLVMValueRef vf = LLVMBuildBitCast(t->b, v, t->f32, "vf");
      LLVMValueRef scale = LLVMConstReal(t->f32, (double)(1u << VP_TEX_FXD_FRAC));
      LLVMValueRef ux = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, uf, scale, ""), t->i32, "");
      LLVMValueRef vx = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, vf, scale, ""), t->i32, "");
      emit_tex_shadow(t, tex, ux, vx, ref, shadow_lod);
      return;
   }

   /* sampler2DArray: coord.z is the array layer (round to nearest, clamp >= 0);
    * sample that slice via the SW array entry (base + layer*layer_stride). Handled
    * before the 2D dispatch since an array sample is still nir_texop_tex/txl. */
   if (tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
       tex->op != nir_texop_txf && tex->op != nir_texop_tg4 && u && v) {
      LLVMValueRef zf = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 2), t->f32, "");
      LLVMValueRef layer = LLVMBuildFPToSI(t->b,
         LLVMBuildFAdd(t->b, zf, LLVMConstReal(t->f32, 0.5), ""), t->i32, "layer");
      LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
      layer = LLVMBuildSelect(t->b,
         LLVMBuildICmp(t->b, LLVMIntSLT, layer, zero, ""), zero, layer, "");
      LLVMValueRef uf = LLVMBuildBitCast(t->b, u, t->f32, "");
      LLVMValueRef vf = LLVMBuildBitCast(t->b, v, t->f32, "");
      LLVMValueRef scale = LLVMConstReal(t->f32, (double)(1u << VP_TEX_FXD_FRAC));
      LLVMValueRef ux = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, uf, scale, ""), t->i32, "");
      LLVMValueRef vx = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, vf, scale, ""), t->i32, "");
      /* Explicit-LOD (textureLod) carries a float LOD; the array entry reads the
       * descriptor filter, so encode the LOD the same way (mip-linear Q or
       * mip-nearest level). Auto-LOD and base take level 0. */
      LLVMValueRef lod = (tex->op == nir_texop_txl && lod_int)
         ? emit_encode_explicit_lod(t, lod_int) : zero;
      emit_tex_array(t, tex, ux, vx, layer, lod);
      return;
   }

   /* samplerCube: coord.xyz is the direction vector; the SW cube entry selects the
    * face and projects. Handled before the 2D dispatch (a cube sample is still
    * nir_texop_tex/txl). */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && !tex->is_array &&
       tex->op != nir_texop_txf && u && v) {
      LLVMValueRef sc = LLVMBuildBitCast(t->b, u, t->f32, "");
      LLVMValueRef tc = LLVMBuildBitCast(t->b, v, t->f32, "");
      LLVMValueRef rc = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 2), t->f32, "");
      LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
      /* txl carries a float LOD; the cube entry reads the descriptor filter, so
       * encode the LOD the same way (mip-linear Q or mip-nearest level). Auto-LOD
       * and base take level 0. */
      LLVMValueRef lod = (tex->op == nir_texop_txl && lod_int)
         ? emit_encode_explicit_lod(t, lod_int) : zero;
      emit_tex_cube(t, tex, sc, tc, rc, lod);
      return;
   }

   /* samplerCubeArray: coord.xyz is the direction, coord.w the array index. The
    * SW entry addresses the slice array_index*6 + face. Handled before the plain
    * cube (which requires !is_array) and the 2D dispatch. */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && tex->is_array &&
       !tex->is_shadow && tex->op != nir_texop_txf && u && v) {
      LLVMValueRef sc = LLVMBuildBitCast(t->b, u, t->f32, "");
      LLVMValueRef tc = LLVMBuildBitCast(t->b, v, t->f32, "");
      LLVMValueRef rc = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 2), t->f32, "");
      LLVMValueRef wf = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 3), t->f32, "");
      LLVMValueRef idx = LLVMBuildFPToSI(t->b,
         LLVMBuildFAdd(t->b, wf, LLVMConstReal(t->f32, 0.5), ""), t->i32, "cubeidx");
      LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
      idx = LLVMBuildSelect(t->b,
         LLVMBuildICmp(t->b, LLVMIntSLT, idx, zero, ""), zero, idx, "");
      LLVMValueRef lod = (tex->op == nir_texop_txl && lod_int)
         ? emit_encode_explicit_lod(t, lod_int) : zero;
      emit_tex_cube_array(t, tex, sc, tc, rc, idx, lod);
      return;
   }

   /* texelFetch: integer (x,y,lod), no wrap/filter/mip -- the coord and lod are
    * already integers, so skip the float->fixed conversion below. */
   if (tex->op == nir_texop_txf) {
      if (!u || !v) {
         mesa_logw("vortexpipe: vp_nir_to_llvm: texelFetch missing coord");
         t->ok = false;
         return;
      }
      LLVMValueRef lod = lod_int ? lod_int : LLVMConstInt(t->i32, 0, false);
      /* texelFetchOffset: the offset is a texel delta added to the integer coord. */
      LLVMValueRef x = off_x ? LLVMBuildAdd(t->b, u, off_x, "") : u;
      LLVMValueRef y = off_y ? LLVMBuildAdd(t->b, v, off_y, "") : v;
      /* 2D array: coord.z is the integer layer slice. */
      LLVMValueRef layer = tex->is_array ? ssa_get(t, coord_ssa, 2)
                                         : LLVMConstInt(t->i32, 0, false);
      /* An isampler/usampler must yield the stored integer; everything else takes
       * the float path, where a float-format texel keeps its real magnitude (its
       * range is not [0,1]) and a non-float format decodes to the same [0,1]
       * channels the packed-ARGB unpack yields. Branch on the base type only --
       * the size bits are not guaranteed. */
      nir_alu_type base = nir_alu_type_get_base_type(tex->dest_type);
      if (base == nir_type_int || base == nir_type_uint) {
         emit_tex_unpack_i32(t, tex, emit_tex_fetch_i32(t, x, y, lod, layer),
                             base == nir_type_int);
      } else if (tex->is_array) {
         emit_tex_unpack_f32(t, tex, emit_tex_fetch_array_f32(t, x, y, layer, lod));
      } else {
         emit_tex_unpack_f32(t, tex, emit_tex_fetch_f32(t, x, y, lod));
      }
      return;
   }

   /* textureSize: a dimension query from the descriptor, no coord/sample. Also the
    * intermediate a lowered textureGrad emits to scale its gradients. */
   if (tex->op == nir_texop_txs) {
      emit_tex_size(t, tex, lod_int);
      return;
   }

   /* textureGather: the 2x2 footprint's `comp` channel, base level, SW-only. */
   if (tex->op == nir_texop_tg4) {
      if (!u || !v) {
         mesa_logw("vortexpipe: vp_nir_to_llvm: textureGather missing coord");
         t->ok = false;
         return;
      }
      LLVMValueRef guf = LLVMBuildBitCast(t->b, u, t->f32, "");
      LLVMValueRef gvf = LLVMBuildBitCast(t->b, v, t->f32, "");
      /* textureGatherOffset: a constant texel offset added to the coord as
       * offset/dim (mip-0 dims), so the 2x2 footprint shifts by whole texels. */
      if (off_x || off_y) {
         LLVMValueRef w0, h0;
         emit_tex_mip0_dims(t, &w0, &h0);
         if (off_x)
            guf = LLVMBuildFAdd(t->b, guf,
               LLVMBuildFDiv(t->b, LLVMBuildSIToFP(t->b, off_x, t->f32, ""),
                             LLVMBuildUIToFP(t->b, w0, t->f32, ""), ""), "");
         if (off_y)
            gvf = LLVMBuildFAdd(t->b, gvf,
               LLVMBuildFDiv(t->b, LLVMBuildSIToFP(t->b, off_y, t->f32, ""),
                             LLVMBuildUIToFP(t->b, h0, t->f32, ""), ""), "");
      }
      LLVMValueRef gscale = LLVMConstReal(t->f32, (double)(1u << VP_TEX_FXD_FRAC));
      LLVMValueRef gux = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, guf, gscale, ""), t->i32, "");
      LLVMValueRef gvx = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, gvf, gscale, ""), t->i32, "");
      LLVMValueRef glayer = NULL;
      if (tex->is_array) {
         LLVMValueRef zf = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 2), t->f32, "");
         LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
         glayer = LLVMBuildFPToSI(t->b,
            LLVMBuildFAdd(t->b, zf, LLVMConstReal(t->f32, 0.5), ""), t->i32, "glayer");
         glayer = LLVMBuildSelect(t->b,
            LLVMBuildICmp(t->b, LLVMIntSLT, glayer, zero, ""), zero, glayer, "");
      }
      /* textureGatherCmp (sampler2DShadow): compare each tap against the reference. */
      if (tex->is_shadow)
         emit_tex_gather_cmp(t, tex, gux, gvx,
            cmp ? cmp : ssa_get(t, coord_ssa, tex->is_array ? 3 : 2), glayer);
      else
         emit_tex_gather(t, tex, gux, gvx, glayer);
      return;
   }

   /* Accept auto-LOD (tex) and explicit-LOD/bias (txl/txb) sampling. Ray-tracing
    * and compute shaders emit txl since they have no implicit derivatives. */
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txl &&
       tex->op != nir_texop_txb) {
      mesa_logw("vortexpipe: vp_nir_to_llvm: unsupported texture op %d", tex->op);
      t->ok = false;
      return;
   }
   /* A 1D texture is one row of a 2D one, so supplying the second coordinate
    * lets the whole 2D path -- filtering, mip selection, wrap -- apply as it
    * stands. A 1D array carries its layer in the second component instead and
    * needs its own handling, so it is left to the guard below. */
   if (!v && u && tex->sampler_dim == GLSL_SAMPLER_DIM_1D && !tex->is_array) {
      v = LLVMConstInt(t->i32, 0, false);          /* 0.0f is all-zero bits */
   }
   /* What still reaches here without a second component is a 1D array, whose
    * second coordinate is a layer rather than an axis. That is a missing
    * dimension, not a rejected op, and reporting it as one would send the
    * reader to the op switch. */
   if (!u || !v) {
      mesa_logw("vortexpipe: vp_nir_to_llvm: texture op %d: no %s coordinate "
                "component (unsupported sampler dimensionality)",
                tex->op, u ? "second" : "first");
      t->ok = false;
      return;
   }

   LLVMValueRef uf = LLVMBuildBitCast(t->b, u, t->f32, "uf");
   LLVMValueRef vf = LLVMBuildBitCast(t->b, v, t->f32, "vf");

   /* textureOffset: a constant texel offset added to the normalized coord as
    * offset/dim (mip-0 dims -- deqp's offset cases sample the base level). */
   if (off_x || off_y) {
      LLVMValueRef w0, h0;
      emit_tex_mip0_dims(t, &w0, &h0);
      if (off_x)
         uf = LLVMBuildFAdd(t->b, uf,
            LLVMBuildFDiv(t->b, LLVMBuildSIToFP(t->b, off_x, t->f32, ""),
                          LLVMBuildUIToFP(t->b, w0, t->f32, ""), ""), "uoff");
      if (off_y)
         vf = LLVMBuildFAdd(t->b, vf,
            LLVMBuildFDiv(t->b, LLVMBuildSIToFP(t->b, off_y, t->f32, ""),
                          LLVMBuildUIToFP(t->b, h0, t->f32, ""), ""), "voff");
   }

   /* float UV -> the TEX unit's S.23 fixed-point coordinate. */
   LLVMValueRef scale = LLVMConstReal(t->f32, (double)(1u << VP_TEX_FXD_FRAC));
   LLVMValueRef ux = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, uf, scale, ""), t->i32, "");
   LLVMValueRef vx = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, vf, scale, ""), t->i32, "");

   /* The explicit-LOD source, as the raw f32 bits every LOD resolver expects; an op
    * without one never reads it. */
   LLVMValueRef lod_bits = lod_int ? lod_int : LLVMConstInt(t->i32, 0, false);

   /* sampler3D: the third coordinate selects the depth slice (SW-sampled). */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_3D && !tex->is_array) {
      LLVMValueRef wf = LLVMBuildBitCast(t->b, ssa_get(t, coord_ssa, 2), t->f32, "wf");
      LLVMValueRef wx = LLVMBuildFPToSI(t->b, LLVMBuildFMul(t->b, wf, scale, ""), t->i32, "");
      emit_tex_3d(t, tex, ux, vx, wx, lod_bits, bias_q8);
      return;
   }

   /* Sample. Implicit-LOD `texture()` on a HW-TEX device: a mipmapped sampler
    * (texstate.filter mip-enable bit) routes to the SW sampler, which resolves the
    * min/mag tap per fragment and the mip level (nearest or trilinear); a non-
    * mipmapped sampler stays on the fast HW TEX path at the base level. mip-enable is
    * a uniform descriptor bit, so the branch never diverges. On a TEX-less device
    * (t->sw_tex) every sample is software. Explicit-LOD/bias (txl/txb) have no
    * derivatives and take the base level.
    *
    * The sample is carried as four floats for a float sampler -- a float-format texel
    * leaves [0,1] and cannot survive the packed 8-bit word -- and as that packed word
    * for an integer sampler, whose formats are all 8-bit and FF. The carrier follows
    * the sampler's static result type, so only one form is ever emitted. */
   const bool as_f32 = !tex_dest_is_int(tex);

   if (t->sw_tex) {
      /* TEX-less device: software always. The SW arm resolves min/mag + mip itself
       * (a non-mipmapped sampler falls out as magnified -> base level). */
      LLVMValueRef r = emit_tex_sw_arm(t, tex, ux, vx, lod_bits, bias_q8, as_f32);
      if (as_f32)
         emit_tex_unpack_f32(t, tex, r);
      else
         emit_tex_unpack(t, tex, r);
      return;
   }

   /* Route to the SW sampler when the sampler is mipmapped, the texture is NPOT (the
    * FF vx_tex4 unit is POT-only), or its format is above the FF set (the FF unit has
    * no decoder or texel stride for it); otherwise the fast HW path. All are uniform
    * descriptor bits, so the branch never diverges (derivatives safe). A float format
    * always sets the extended-format bit, so the HW unit never sees a wide texel.
    *
    * A border wrap is NOT among them: the unit carries a border colour of its own
    * and substitutes the taps that leave the texture. A mipmapped border sampler
    * still lands in software, on the mip bit -- which is the one border combination
    * the unit has no test for. */
   LLVMValueRef use_sw = LLVMBuildICmp(t->b, LLVMIntNE,
      LLVMBuildAnd(t->b, emit_tex_filter_word(t),
         LLVMConstInt(t->i32,
            GFX_SW_TEX_FILTER_MIP_ENABLE | GFX_SW_TEX_FILTER_NPOT |
            GFX_SW_TEX_FILTER_EXT_FORMAT, false), ""),
      LLVMConstInt(t->i32, 0, false), "tex_use_sw");
   LLVMValueRef fn = LLVMGetBasicBlockParent(LLVMGetInsertBlock(t->b));
   LLVMBasicBlockRef bb_sw = LLVMAppendBasicBlockInContext(t->ctx, fn, "tex_sw");
   LLVMBasicBlockRef bb_hw = LLVMAppendBasicBlockInContext(t->ctx, fn, "tex_hw");
   LLVMBasicBlockRef bb_mg = LLVMAppendBasicBlockInContext(t->ctx, fn, "tex_merge");
   LLVMBuildCondBr(t->b, use_sw, bb_sw, bb_hw);

   LLVMPositionBuilderAtEnd(t->b, bb_sw);
   LLVMValueRef texel_sw = emit_tex_sw_arm(t, tex, ux, vx, lod_bits, bias_q8, as_f32);
   LLVMBasicBlockRef end_sw = LLVMGetInsertBlock(t->b);
   LLVMBuildBr(t->b, bb_mg);

   LLVMPositionBuilderAtEnd(t->b, bb_hw);
   LLVMValueRef texel_hw = emit_tex_hw(t, ux, vx, LLVMConstInt(t->i32, 0, false));
   if (as_f32)
      emit_tex_argb_to_f32_scratch(t, texel_hw);
   LLVMBasicBlockRef end_hw = LLVMGetInsertBlock(t->b);
   LLVMBuildBr(t->b, bb_mg);

   LLVMPositionBuilderAtEnd(t->b, bb_mg);
   if (as_f32) {
      /* Both arms filled the one scratch slot, so there is nothing to merge: the
       * slot is the entry block's alloca, which dominates this block whichever arm
       * ran. */
      emit_tex_unpack_f32(t, tex, texel_sw);
      return;
   }
   LLVMValueRef texel = LLVMBuildPhi(t->b, t->i32, "texel");
   LLVMValueRef vals[2] = { texel_sw, texel_hw };
   LLVMBasicBlockRef bbs[2] = { end_sw, end_hw };
   LLVMAddIncoming(texel, vals, bbs, 2);
   emit_tex_unpack(t, tex, texel);
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
   unsigned vs_scalars = 0;
   nir_foreach_shader_out_variable(var, nir) {
      if (t->nvars >= VP_MAXV) { t->ok = false; return; }
      int off;
      if (var->data.location == VARYING_SLOT_POS) {
         off = 0;
      } else {
         off = (int)next;
         next += 16;
         if (out_vs) {
            if (out_vs->num_varyings >= VP_VS_MAX_VARYINGS) {
               mesa_logw("vortexpipe: VS declares more than %u generic "
                         "varyings; this draw runs on llvmpipe",
                         VP_VS_MAX_VARYINGS);
               t->ok = false; return;
            }
            out_vs->varying_loc[out_vs->num_varyings]   = var->data.location;
            out_vs->varying_comps[out_vs->num_varyings] =
               glsl_get_components(var->type);
            out_vs->num_varyings++;
            vs_scalars += glsl_get_components(var->type);
         }
      }
      t->vars[t->nvars].var     = var;
      t->vars[t->nvars].alloca  = NULL;
      t->vars[t->nvars].out_off = off;
      t->nvars++;
   }
   /* The front end carries varyings in VP_RAST_MAX_PLANES scalar interpolation
    * planes and drops whatever does not fit, so a shader that writes more than
    * that loses varyings with no diagnostic and the fragment reads a
    * fixed-function default in their place. Refuse it here instead: the draw
    * takes the llvmpipe fallback and produces the right answer slowly. */
   if (vs_scalars > VP_RAST_MAX_PLANES) {
      mesa_logw("vortexpipe: VS varyings need %u interpolation planes, "
                "device carries %u; this draw runs on llvmpipe",
                vs_scalars, VP_RAST_MAX_PLANES);
      t->ok = false; return;
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
   unsigned fs_scalars = 0;
   t->fs_pos_off = -1;
   nir_foreach_shader_in_variable(var, nir) {
      if (t->nvars >= VP_MAXV) { t->ok = false; return; }
      /* A flat varying bypasses the interpolator: its bit pattern is not
       * necessarily a number -- which is why every integer varying is flat --
       * and the plane path would premultiply it by 1/w and quantise it to
       * Q7.24, turning a small integer into zero. Setup carries the provoking
       * vertex's words verbatim in a side array instead, which the fill below
       * reads without arithmetic.
       *
       * noperspective has no such carry: it needs a plane that is interpolated
       * but not premultiplied, which is a third path neither side implements.
       * Refuse it rather than treat it as smooth -- silently mistreating an
       * interpolation mode is what this whole path exists to stop doing. */
      if (var->data.interpolation == INTERP_MODE_NOPERSPECTIVE) {
         mesa_logw("vortexpipe: FS input at location %u uses noperspective "
                   "interpolation, which the device varying path does not "
                   "implement; this draw runs on llvmpipe",
                   var->data.location);
         t->ok = false; return;
      }
      /* gl_FragCoord is a system value wearing an input's clothes. It takes a
       * slot in the record like any other input, but the wrapper writes it from
       * the pixel address and the primitive rather than from an interpolated
       * plane -- so it must not claim one, or every varying declared after it
       * reads its neighbour's. */
      if (var->data.location == VARYING_SLOT_POS) {
         t->fs_pos_off = (int)off;
      } else {
         fs_scalars += glsl_get_components(var->type);
      }
      t->vars[t->nvars].var     = var;
      t->vars[t->nvars].alloca  = NULL;
      t->vars[t->nvars].out_off = (int)off;
      t->nvars++;
      off += 16;
   }
   /* One slot past the declared inputs for the system values with no variable
    * of their own (gl_FrontFacing). */
   t->fs_sysval_off = off;
   /* The record is a fixed-size alloca in the wrapper, so a shader declaring
    * more inputs than it holds would have the wrapper write past it. The plane
    * budget below does not bound this: twelve scalars are twelve slots. */
   if (off + 16u > VP_FS_IN_WORDS * 4u) {
      mesa_logw("vortexpipe: FS declares %u input slots, record holds %u; "
                "this draw runs on llvmpipe",
                off / 16u, (VP_FS_IN_WORDS * 4u) / 16u - 1u);
      t->ok = false; return;
   }
   /* Same plane budget as the VS side, checked here too because the two are
    * compiled independently: a fragment shader reading past the last plane is
    * left holding the fixed-function default, silently. */
   if (fs_scalars > VP_RAST_MAX_PLANES) {
      mesa_logw("vortexpipe: FS inputs need %u interpolation planes, "
                "device carries %u; this draw runs on llvmpipe",
                fs_scalars, VP_RAST_MAX_PLANES);
      t->ok = false; return;
   }
   /* Output slots are keyed by render-target index: a colour output at
    * FRAG_RESULT_DATA0+k lands at out slot k*16, so the wrapper can pack colours
    * 0..num_color-1 deterministically regardless of declaration order. Non-colour
    * outputs get a slot past the colour area, where a store cannot corrupt a
    * colour; gl_FragDepth's is read back for the fragment's depth, and anything
    * else there is written and ignored. */
   unsigned num_color = 0, scratch = 0;
   t->fs_depth_off = -1;
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
      if (loc == FRAG_RESULT_DEPTH)
         t->fs_depth_off = (int)(slot * 16);
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
   /* The address is pointer-wide: the LSU decodes the aperture by comparing the
    * whole address against the aperture range, and rv64 holds a 32-bit value in
    * a register SIGN-extended -- so an i32 base of 0xE0000000 would arrive as
    * 0xffffffff_e0000000 and match nothing. Colour and depth are payload and
    * stay 32-bit. */
   LLVMTypeRef args[3] = { t->iptr, t->i32, t->i32 };
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

/* Aperture address of one pixel of one colour attachment:
 *   base + ((((rt << 1) | face) << (xbits+ybits)) | (y << xbits) | x) << record_shift
 * xbits/ybits/record_shift are unpacked from the FS arg block (per-draw). The
 * attachment index sits ABOVE the cube face, so rt == 0 reproduces the
 * single-attachment encoding bit for bit. */
static LLVMValueRef
emit_om_aperture_addr(struct vp_tr *t, LLVMValueRef xbits, LLVMValueRef ybits,
                      LLVMValueRef shift, LLVMValueRef x, LLVMValueRef y,
                      LLVMValueRef face, LLVMValueRef rt)
{
   LLVMValueRef sel = LLVMBuildOr(t->b,
      LLVMBuildShl(t->b, rt, LLVMConstInt(t->i32, 1, false), ""), face, "");
   LLVMValueRef idx = LLVMBuildOr(t->b,
      LLVMBuildShl(t->b, sel, LLVMBuildAdd(t->b, xbits, ybits, ""), ""),
      LLVMBuildOr(t->b, LLVMBuildShl(t->b, y, xbits, ""), x, ""), "");
   return LLVMBuildAdd(t->b,
      LLVMConstInt(t->iptr, (uint32_t)VX_MEM_OM_BASE_ADDR, false),
      vp_to_iptr(t, LLVMBuildShl(t->b, idx, shift, "")), "");
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
emit_fs_fill_varyings(struct vp_tr *t, LLVMValueRef prim, LLVMValueRef flat,
                      LLVMValueRef in_addr, LLVMValueRef px, LLVMValueRef py,
                      LLVMValueRef dxq, LLVMValueRef dyq)
{
   /* The front end interpolates 12 scalar planes; expand_k packed the VS varyings
    * into them in declaration order [u,v,r,g,b,a,w0..w5] (gfx_frontend_k.h). Read
    * them back the same way: each FS input varying claims the next nc lanes, so a
    * draw may carry any mix of varyings (a texcoord + a scalar lod, or a
    * samplerCube textureGrad's coord + dPdx + dPdy = 9 scalars) without the planes
    * colliding. Twelve planes are the [u,v,r,g,b,a] six plus w0..w5. */
   static const unsigned lane[VP_RAST_MAX_PLANES] = {
      VP_RAST_ATTR_U,  VP_RAST_ATTR_V,
      VP_RAST_ATTR_R,  VP_RAST_ATTR_G,  VP_RAST_ATTR_B,  VP_RAST_ATTR_A,
      VP_RAST_ATTR_W0, VP_RAST_ATTR_W1, VP_RAST_ATTR_W2,
      VP_RAST_ATTR_W3, VP_RAST_ATTR_W4, VP_RAST_ATTR_W5 };

   /* Perspective recovery: setup premultiplied every colour/uv plane by 1/w and
    * carries a separate 1/w plane, so the true attribute is interp(a/w)/interp(1/w)
    * (gfx_setup.h). setup also folds a common power-of-2 into 1/w to keep large
    * texcoords inside the Q7.24 range; that same divide undoes it. For a
    * screen-aligned triangle 1/w is constant 1, so this is an exact divide by 1. */
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
      if (var->data.location == VARYING_SLOT_POS) {
         continue;   /* system value; written below, and claims no plane */
      }
      /* A flat varying is held constant over the primitive, so it is read from
       * the words setup copied out of the provoking vertex rather than
       * interpolated. No arithmetic is applied on the way: the value may be an
       * integer whose float reinterpretation is a denormal, and the plane path
       * would quantise that to zero. */
      const bool is_flat = (var->data.interpolation == INTERP_MODE_FLAT);
      for (unsigned c = 0; c < nc && li < VP_RAST_MAX_PLANES; c++, li++) {
         if (is_flat) {
            emit_store_i32(t, addk(t, slot, c * 4),
                           emit_load_i32(t, addk(t, flat, li * 4)));
            continue;
         }
         LLVMValueRef q = emit_interp(t, addk(t, prim, lane[li]), dxq, dyq);
         LLVMValueRef f = LLVMBuildFMul(t->b, emit_fixed_to_float(t, q, 24),
                                        inv_rhw, "");
         emit_store_i32(t, addk(t, slot, c * 4),
                        LLVMBuildBitCast(t->b, f, t->i32, ""));
      }
   }

   /* gl_FragCoord. x,y are the pixel CENTRE, which is the sample position the
    * whole pipeline shades at. z is the same fixed-function depth plane the
    * merger and the early-Z read, so a shader comparing gl_FragCoord.z against
    * what it wrote sees one value, not two. w is 1/w_clip: the rhw plane
    * interpolates that directly, but setup premultiplied it by a normalization
    * factor that cancels in a varying's interp(a*rhw)/interp(rhw) and does NOT
    * cancel here, so it is divided back out. */
   if (t->fs_pos_off >= 0) {
      LLVMValueRef slot = addk(t, in_addr, (unsigned)t->fs_pos_off);
      LLVMValueRef half = LLVMConstReal(t->f32, 0.5);
      LLVMValueRef xf = LLVMBuildFAdd(t->b,
         LLVMBuildSIToFP(t->b, px, t->f32, ""), half, "fcx");
      LLVMValueRef yf = LLVMBuildFAdd(t->b,
         LLVMBuildSIToFP(t->b, py, t->f32, ""), half, "fcy");

      LLVMValueRef zpx = emit_load_i32(t, addk(t, prim, VP_RAST_ATTR_Z));
      LLVMValueRef zpy = emit_load_i32(t, addk(t, prim, VP_RAST_ATTR_Z + 4));
      LLVMValueRef zpz = emit_load_i32(t, addk(t, prim, VP_RAST_ATTR_Z + 8));
      LLVMValueRef zacc = LLVMBuildAdd(t->b,
         LLVMBuildAdd(t->b,
            LLVMBuildMul(t->b, LLVMBuildSExt(t->b, zpx, t->i64, ""),
                               LLVMBuildSExt(t->b, px, t->i64, ""), ""),
            LLVMBuildMul(t->b, LLVMBuildSExt(t->b, zpy, t->i64, ""),
                               LLVMBuildSExt(t->b, py, t->i64, ""), ""), ""),
         LLVMBuildSExt(t->b, zpz, t->i64, ""), "");
      LLVMValueRef z32 = LLVMBuildTrunc(t->b, zacc, t->i32, "");
      LLVMValueRef zmask = LLVMConstInt(t->i32, 0xffffff, false);
      LLVMValueRef zsat = LLVMBuildSelect(t->b,
         LLVMBuildICmp(t->b, LLVMIntSLT, z32, LLVMConstInt(t->i32, 0, false), ""),
         LLVMConstInt(t->i32, 0, false),
         LLVMBuildSelect(t->b,
            LLVMBuildICmp(t->b, LLVMIntSGT, z32, zmask, ""), zmask, z32, ""), "");
      LLVMValueRef zf = LLVMBuildFDiv(t->b,
         LLVMBuildUIToFP(t->b, zsat, t->f32, ""),
         LLVMConstReal(t->f32, (double)0xffffff), "fcz");

      LLVMValueRef scale = LLVMBuildBitCast(t->b,
         emit_load_i32(t, addk(t, prim, VP_RAST_PRIM_RHW_SCALE)), t->f32, "");
      LLVMValueRef snz = LLVMBuildFCmp(t->b, LLVMRealONE, scale,
                                       LLVMConstReal(t->f32, 0.0), "");
      LLVMValueRef wf = LLVMBuildFDiv(t->b, rhw_f,
         LLVMBuildSelect(t->b, snz, scale, LLVMConstReal(t->f32, 1.0), ""), "fcw");

      emit_store_i32(t, addk(t, slot, 0),
                     LLVMBuildBitCast(t->b, xf, t->i32, ""));
      emit_store_i32(t, addk(t, slot, 4),
                     LLVMBuildBitCast(t->b, yf, t->i32, ""));
      emit_store_i32(t, addk(t, slot, 8),
                     LLVMBuildBitCast(t->b, zf, t->i32, ""));
      emit_store_i32(t, addk(t, slot, 12),
                     LLVMBuildBitCast(t->b, wf, t->i32, ""));
   }

   /* gl_FrontFacing. EdgeEquation flips a backward-wound triangle's edges so the
    * interior is positive either way, which erases the winding; setup records it
    * beforehand, and this is the only place it can still be read. */
   emit_store_i32(t, addk(t, in_addr, t->fs_sysval_off),
      LLVMBuildZExt(t->b,
         LLVMBuildICmp(t->b, LLVMIntEQ,
            emit_load_i32(t, addk(t, prim, VP_RAST_PRIM_FACING)),
            LLVMConstInt(t->i32, 0, false), ""),
         t->i32, "frontfacing"));
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
   unsigned     bcoord_word0; /* SW: word index of bcoords[0] in that record --
                               * 1 single-sample, 5 multisample (the wider record
                               * carries sample_masks[4] between the two) */
   LLVMValueRef sample_mask;  /* SW multisample: this corner's per-sample coverage
                               * (bit k = sample k). NULL on every single-sample
                               * path, and on the fixed-function path, which has no
                               * per-sample coverage to give. */
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
      /* bcoords[axis*4+corner], at bcoord_word0 + .. in the quad record (word 1
       * single-sample, word 5 multisample). The corner is this lane's sub, a
       * runtime value, so index it dynamically. */
      LLVMValueRef off = LLVMBuildShl(t->b,
         LLVMBuildAdd(t->b, bc->sub,
            LLVMConstInt(t->i32, bc->bcoord_word0 + axis * 4, false), ""),
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
    * unpack below is all i32 arithmetic. */
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
                 LLVMValueRef prim, LLVMValueRef flat,
                 LLVMValueRef in_scr, LLVMValueRef out_scr,
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
      emit_fs_fill_varyings(t, prim, flat, in_addr, pxc, pyc, dxq, dyq);

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

      /* pack a render target's FS output (4 floats at out_addr + rt*16) into a
       * single pixel word, in that attachment's own channel order: red in the
       * low byte, or blue there when it is blue-first. The order is per target,
       * so a draw writing two attachments may pack them differently. The merger
       * stores the word unmodified and the host reads the buffer back byte for
       * byte, so this is the only place the order is decided. Alpha stays in
       * the high byte either way, which is what keeps every alpha blend factor
       * order-independent. */
      unsigned num_color = t->fs_num_color ? t->fs_num_color : 1;
      LLVMValueRef rgba_rt[GFX_OM_MAX_RT];
      for (unsigned rt = 0; rt < num_color && rt < GFX_OM_MAX_RT; rt++) {
         const bool bgra = (t->fs_bgra_mask >> rt) & 1u;
         LLVMValueRef rgba = LLVMConstInt(t->i32, 0, false);
         for (unsigned c = 0; c < 4; c++) {
            const unsigned byte = (bgra && c < 3) ? (2u - c) : c;
            LLVMValueRef fc = LLVMBuildBitCast(t->b,
               emit_load_i32(t, addk(t, out_addr, rt * 16 + c * 4)), t->f32, "");
            LLVMValueRef bc8 = LLVMBuildShl(t->b, emit_to_byte(t, fc),
               LLVMConstInt(t->i32, byte * 8, false), "");
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

      /* gl_FragDepth replaces the interpolated plane: the shader's [0,1] value
       * scales to the same 24-bit range the plane path saturates to, so the
       * merger tests and stores it without knowing which produced it. */
      if (t->fs_depth_off >= 0) {
         LLVMValueRef zf = LLVMBuildBitCast(t->b,
            emit_load_i32(t, addk(t, out_addr, (unsigned)t->fs_depth_off)),
            t->f32, "fragdepth");
         LLVMValueRef zero_f = LLVMConstReal(t->f32, 0.0);
         LLVMValueRef one_f  = LLVMConstReal(t->f32, 1.0);
         /* Ordered compares select the bound on an unordered result, so a NaN
          * depth resolves to a defined value instead of poisoning fptoui. */
         zf = LLVMBuildSelect(t->b,
            LLVMBuildFCmp(t->b, LLVMRealOLT, zf, one_f, ""), zf, one_f, "");
         zf = LLVMBuildSelect(t->b,
            LLVMBuildFCmp(t->b, LLVMRealOGT, zf, zero_f, ""), zf, zero_f, "");
         LLVMValueRef zi = LLVMBuildFPToUI(t->b,
            LLVMBuildFAdd(t->b,
               LLVMBuildFMul(t->b, zf,
                  LLVMConstReal(t->f32, (double)0xffffff), ""),
               LLVMConstReal(t->f32, 0.5), ""), t->i32, "fragdepth_i");
         /* 0xffffff+0.5 is not representable in f32 and rounds up, so z == 1.0
          * lands one past the field; saturate as the plane path does. */
         depth_i = LLVMBuildSelect(t->b,
            LLVMBuildICmp(t->b, LLVMIntUGT, zi, zmask, ""), zmask, zi, "depth");
      }

      /* Multisample merge is legal to emit only when this kernel was translated
       * for it AND this call site actually carries per-sample coverage. The two
       * clauses check different things, and both are load-bearing: the
       * fixed-function wrapper fails the second unconditionally, which is what
       * keeps the shared emitter from taking this arm where there is no mask to
       * read. A kernel built for multisample whose source cannot feed it is a
       * translator bug, not a runtime case -- bail rather than silently emit the
       * single-sample arm and render subtly wrong coverage. */
      const bool msaa = t->fs_samples > 1 && bc->sample_mask && t->sw_om;
      if (t->fs_samples > 1 && !msaa) {
         mesa_logw("vortexpipe: multisample FS variant lacks a per-sample "
                   "coverage source or a software merger; runs on llvmpipe");
         t->ok = false;
      }
      /* MRT under multisampling has no device ABI -- gfx_om_fragment_mrt_sw is
       * single-sample only, and merging through it would write plausible garbage
       * into buffers strided for S samples. Refuse the translation so the variant
       * never exists and the draw falls to llvmpipe, which renders it correctly. */
      if (msaa && num_color > 1) {
         mesa_logw("vortexpipe: multisample MRT has no device merge path; "
                   "this shader runs on llvmpipe");
         t->ok = false;
      }

      if (msaa && num_color <= 1) {
         /* Per-sample merge: gfx_om_fragment_msaa_sw(omstate, samples,
          * sample_mask, px, py, face, colour, depth). `cov` was already folded
          * with the shader's discard flag above, so gating the whole mask on it
          * applies discard exactly once -- the per-fragment flag cannot be
          * expressed inside a per-sample mask any other way, which is why the
          * callee's contract puts the fold on us. An all-zero mask is dropped by
          * the callee, keeping this straight-line like the single-sample call. */
         LLVMValueRef mask_live = LLVMBuildSelect(t->b,
            LLVMBuildICmp(t->b, LLVMIntNE, cov,
                          LLVMConstInt(t->i32, 0, false), ""),
            bc->sample_mask, LLVMConstInt(t->i32, 0, false), "smask_live");
         LLVMTypeRef params[8] = { t->ptr, t->i32, t->i32, t->i32, t->i32,
                                   t->i32, t->i32, t->i32 };
         LLVMTypeRef fty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                            params, 8, false);
         LLVMValueRef ofn = LLVMGetNamedFunction(t->mod, "gfx_om_fragment_msaa_sw");
         if (!ofn)
            ofn = LLVMAddFunction(t->mod, "gfx_om_fragment_msaa_sw", fty);
         LLVMValueRef a[8] = { omstate_ptr,
                               LLVMConstInt(t->i32, t->fs_samples, false),
                               mask_live, pxc, pyc, face, rgba, depth_i };
         LLVMBuildCall2(t->b, fty, ofn, a, 8, "");
      } else if (t->sw_om && num_color > 1) {
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
         /* FF output-merger: export this pixel as a store into the OM aperture,
          * once per colour attachment. An uncovered lane must be skipped --
          * there is no coverage mask downstream, and the ingress turns every
          * aperture store into a fragment. The branch diverges across lanes; the
          * thread mask handles it, and it is placed AFTER the shader body so a
          * helper lane still runs the shader.
          *
          * Each export re-runs the depth/stencil stage, so several of them are
          * only self-consistent while that stage cannot change state between
          * them -- the draw path keeps a depth- or stencil-WRITING multi-
          * attachment draw on the software merger, and both the RTL and the
          * SimX model assert it rather than trusting that. num_color == 1
          * unrolls to exactly the single-attachment export. */
         LLVMBasicBlockRef bb_do   = LLVMAppendBasicBlockInContext(t->ctx, fn, "om_do");
         LLVMBasicBlockRef bb_skip = LLVMAppendBasicBlockInContext(t->ctx, fn, "om_skip");
         LLVMBuildCondBr(t->b,
            LLVMBuildICmp(t->b, LLVMIntNE, cov,
                          LLVMConstInt(t->i32, 0, false), ""),
            bb_do, bb_skip);

         LLVMPositionBuilderAtEnd(t->b, bb_do);
         for (unsigned rt = 0; rt < num_color && rt < GFX_OM_MAX_RT; rt++) {
            emit_vx_om_export(t,
               emit_om_aperture_addr(t, t->om_ap_xbits, t->om_ap_ybits,
                                     t->om_ap_shift, pxc, pyc, face,
                                     LLVMConstInt(t->i32, rt, false)),
               rgba_rt[rt], depth_i);
         }
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
   LLVMValueRef flat_base = emit_arg_i32(t, arg, GFX_FS_ARG_FLAT);
   /* arg[1] is the resident gfx_sw_texstate_t[] device address (host-filled for
    * any textured draw). The SW sampler reads the whole descriptor; the HW-tex FS
    * reads only logdim from it to compute the mip LOD. Untextured draws leave
    * arg[1] zero and never dereference it. Passed to fs_main as its 3rd param. */
   LLVMValueRef texstate_ptr = LLVMBuildIntToPtr(t->b,
      emit_arg_i32(t, arg, 1), t->ptr, "texstate");
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
   LLVMValueRef in_scr  = LLVMBuildAlloca(t->b, LLVMArrayType(t->i32, VP_FS_IN_WORDS),
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
   /* The provoking vertex's flat words for this primitive, in a side array the
    * front end fills in step with the primitive buffer and indexed by the same
    * id. Zero when nothing bound declares a flat input, and then never read. */
   LLVMValueRef flat = LLVMBuildAdd(t->b, flat_base,
      vp_to_iptr(t, LLVMBuildMul(t->b, pid,
         LLVMConstInt(t->i32, GFX_FS_FLAT_WORDS * 4u, false), "")), "flat");

   /* HW raster: edge values are recomputed in-shader from prim[pid]'s edge planes
    * (the window carries only pos + pid; P2 dropped the bcoord payload). */
   struct vp_bc_src bc = { .from_window = true, .quad_addr = NULL, .sub = NULL };
   emit_shade_pixel(t, fn, fs_main, fs_main_ty, prim, flat, in_scr, out_scr,
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
/* Quad-record widths, derived from the device ABI rather than copied from it:
 * both structs come from gfx_sw_abi.h, so a change there cannot silently
 * desynchronise the stride the emitter walks with. */
#define VP_RAST_QUAD_WORDS      ((unsigned)(sizeof(gfx_rast_quad_t) / 4))
#define VP_RAST_MSAA_QUAD_WORDS ((unsigned)(sizeof(gfx_rast_msaa_quad_t) / 4))
static_assert(VP_RAST_QUAD_WORDS == 13,
              "gfx_rast_quad_t width moved; the FF frag payload layout changed");
static_assert(VP_RAST_MSAA_QUAD_WORDS == 17,
              "gfx_rast_msaa_quad_t width moved; check the bcoord base word");

/* The emitter builds the call to the walk by hand -- it writes the LLVM
 * function type rather than calling the C declaration -- so nothing in the
 * compiler checks it against gfx_sw_abi.h, and a parameter list that grows on
 * the device side leaves the arguments shifted. The failure is silent: the walk
 * reads a scissor bound out of a pointer, rejects every sample, and the draw
 * renders black. Bind each declaration to the shape the emitter assumes, so the
 * next change over there is a build error here instead. */
typedef uint32_t (*vp_walk_fn)(const void *, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                               gfx_rast_quad_t *, uint32_t);
typedef uint32_t (*vp_walk_msaa_fn)(const void *, uint32_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t, uint32_t,
                                    uint32_t, gfx_rast_msaa_quad_t *, uint32_t);
static const vp_walk_fn      vp_walk_sig      = gfx_rast_walk_tile_sw;
static const vp_walk_msaa_fn vp_walk_msaa_sig = gfx_rast_walk_tile_msaa_sw;

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
   LLVMValueRef flat_base = emit_arg_i32(t, arg, GFX_FS_ARG_FLAT);
   /* arg[1] = resident texstate for any textured draw (HW-tex reads logdim from
    * it for the mip LOD, SW sampler reads all of it); zero + unused if untextured. */
   LLVMValueRef texstate_ptr = LLVMBuildIntToPtr(t->b,
      emit_arg_i32(t, arg, 1), t->ptr, "texstate");
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
   LLVMValueRef in_scr  = LLVMBuildAlloca(t->b, LLVMArrayType(t->i32, VP_FS_IN_WORDS), "fs_in");
   LLVMValueRef out_scr = LLVMBuildAlloca(t->b,
      LLVMArrayType(t->i32, t->fs_out_words ? t->fs_out_words : 4), "fs_out");
   if (t->sw_om && nrt > 1) {
      mrt_ptr = LLVMBuildIntToPtr(t->b, emit_arg_i32(t, arg, GFX_FS_ARG_MRT),
                                  t->ptr, "mrt");
      LLVMValueRef col_scr = LLVMBuildAlloca(t->b,
         LLVMArrayType(t->i32, nrt), "fs_colors");
      mrt_colors_addr = LLVMBuildPtrToInt(t->b, col_scr, t->iptr, "");
   }
   /* Multisample quads carry per-sample coverage, so their record is wider and
    * the walk that fills it is a different entry point. Both are selected here,
    * once, from the variant's sample count -- a single-sample kernel keeps the
    * narrower per-lane quad buffer rather than paying for a field it cannot use. */
   const bool ms = t->fs_samples > 1;
   const unsigned quad_words = ms ? VP_RAST_MSAA_QUAD_WORDS : VP_RAST_QUAD_WORDS;
   const char *walk_name = ms ? "gfx_rast_walk_tile_msaa_sw"
                              : "gfx_rast_walk_tile_sw";
   LLVMValueRef quadbuf = LLVMBuildAlloca(t->b,
      LLVMArrayType(t->i32, VP_SW_RAST_MAX_QUADS * quad_words), "quads");
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
   LLVMValueRef flat = LLVMBuildAdd(t->b, flat_base,
      vp_to_iptr(t, LLVMBuildMul(t->b, pid,
         LLVMConstInt(t->i32, GFX_FS_FLAT_WORDS * 4u, false), "")), "flat");
   /* The walk confines coverage to a rect, not an extent: left and top are
    * separate parameters because a scissored sub-rectangle has a non-zero
    * origin. The arg block carries only the extent today, so the origin is the
    * framebuffer's -- the same rect the walk was confined to when the callee
    * took a width and a height and pinned its own corner to (0,0). Handing it
    * an app scissor origin is a separate change, and needs the host to put one
    * in the arg block first. */
   LLVMValueRef zero = LLVMConstInt(t->i32, 0, false);
   LLVMTypeRef wparams[11] = { t->ptr, t->i32, t->i32, t->i32, t->i32,
                               t->i32, t->i32, t->i32, t->i32, t->ptr, t->i32 };
   LLVMTypeRef wty = LLVMFunctionType(t->i32, wparams, 11, false);
   LLVMValueRef wfn = LLVMGetNamedFunction(t->mod, walk_name);
   if (!wfn)
      wfn = LLVMAddFunction(t->mod, walk_name, wty);
   LLVMValueRef wargs[11] = {
      LLVMBuildIntToPtr(t->b, prim, t->ptr, "primp"), pid, tx, ty, logc,
      zero, zero, scis_w, scis_h,
      quadbuf, LLVMConstInt(t->i32, VP_SW_RAST_MAX_QUADS, false) };
   LLVMValueRef cnt = LLVMBuildCall2(t->b, wty, wfn, wargs, 11, "cnt");
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
      LLVMConstInt(t->i32, quad_words * 4, false), "");
   LLVMValueRef quad_addr = LLVMBuildAdd(t->b, quad_base, vp_to_iptr(t, qoff), "quad");
   LLVMValueRef pos_mask = emit_load_i32(t, quad_addr);

   /* pos_mask: mask@[3:0], qx@[4+:DIM-1], qy@[4+DIM-1+:DIM-1]. This lane's pixel is
    * px=(qx<<1)|(sub&1), py=(qy<<1)|(sub>>1), covered = mask[sub].
    *
    * The position fields decode identically in both records, but the coverage
    * nibble does NOT: rast_emit_quad_msaa builds pos_mask from the quad position
    * alone and leaves [3:0] zero, because multisample coverage is "any sample
    * inside" and cannot be expressed as one bit per pixel. Reading it there would
    * mask every fragment off and render an empty image with no diagnostic, so the
    * multisample coverage comes from sample_masks[sub] instead. */
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
   LLVMValueRef smask = NULL;
   LLVMValueRef cov;
   if (ms) {
      /* sample_masks[sub] at word 1+sub; covered = any sample of this corner. */
      LLVMValueRef soff = LLVMBuildShl(t->b,
         LLVMBuildAdd(t->b, sub, one, ""),
         LLVMConstInt(t->i32, 2, false), "smoff");
      smask = emit_load_i32(t,
         LLVMBuildAdd(t->b, quad_addr, vp_to_iptr(t, soff), ""));
      cov = LLVMBuildZExt(t->b,
         LLVMBuildICmp(t->b, LLVMIntNE, smask,
                       LLVMConstInt(t->i32, 0, false), ""), t->i32, "cov");
   } else {
      cov = LLVMBuildAnd(t->b,
         LLVMBuildLShr(t->b, pos_mask, sub, ""), one, "cov");
   }

   struct vp_bc_src bc = { .from_window = false, .quad_addr = quad_addr,
                           .sub = sub,
                           .bcoord_word0 = ms ? 5u : 1u,
                           .sample_mask = smask };
   emit_shade_pixel(t, fn, fs_main, fs_main_ty, prim, flat, in_scr, out_scr,
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
               const struct vp_sw_routing *routing,
               unsigned samples, uint8_t bgra_mask)
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

   /* A descriptor the scan could not record is a descriptor the relocation
    * never rewrites, leaving the device to dereference a host address. That is
    * a host-side array bound, invisible to everything else here, so it is
    * turned into an ordinary translation failure: the shader falls back to
    * llvmpipe, which has no such limit. Refused before the module is built,
    * since nothing downstream can change the answer -- but after the NIR dump
    * above, so the shader that tripped it can still be looked at. */
   if (vp_descriptors_overflow(nir)) {
      mesa_logw("vortexpipe: shader touches more than %u distinct descriptors, "
                "which is more than the relocation can carry", VP_MAX_DESCS);
      return false;
   }

   /* lavapipe numbers a descriptor set's constant buffer set+1, without
    * compacting, so the highest index a shader reaches says which set it needs
    * and the bound to compare it against is whatever that stage can address. */
   const unsigned max_cbuf = vp_descriptors_max_cbuf(nir);

   if (nir->info.stage == MESA_SHADER_COMPUTE) {
      /* Compute's argument block carries set 0's blob and has no slot for a
       * second, and no table to index either. Refusing matters more than it
       * looks -- two sets number their bindings from zero independently, so a
       * set-1 descriptor carries the same offset as its set-0 counterpart, and
       * the launch relocation would rewrite one on top of the other rather than
       * merely miss it. */
      if (max_cbuf > VP_CBUF_SET0) {
         mesa_logw("vortexpipe: compute shader reaches a descriptor set other "
                   "than set 0, which the launch argument block has no slot for");
         return false;
      }
   } else if (max_cbuf >= GFX_FS_DESC_SLOTS) {
      /* The vertex and fragment stages read their blob bases from a resident
       * table of GFX_FS_DESC_SLOTS entries, indexed by the shader with no bound
       * check of its own -- an index past the end is loaded as readily as one
       * inside, and whatever follows the table is dereferenced as a
       * constant-buffer base. The host side is sized to match and simply never
       * records that buffer, so nothing downstream notices either. */
      mesa_logw("vortexpipe: shader reaches constant buffer %u, past the "
                "%u-entry descriptor table", max_cbuf,
                (unsigned)GFX_FS_DESC_SLOTS);
      return false;
   }

   struct vp_tr t = {0};
   t.ok    = true;
   t.is_vs = (nir->info.stage == MESA_SHADER_VERTEX);
   t.is_fs = (nir->info.stage == MESA_SHADER_FRAGMENT);
   /* Per-unit SW routing (FS only). */
   t.sw_tex    = (t.is_fs && routing && routing->sw_tex);
   t.sw_om     = (t.is_fs && routing && routing->sw_om);
   t.sw_raster = (t.is_fs && routing && routing->sw_raster);
   /* Samples per pixel this kernel is built for. Above 1 the merge is per-sample
    * and the coverage the rasterizer hands over is a sample mask rather than a
    * single bit, so it changes what is emitted, not just what is passed in. */
   t.fs_samples = (t.is_fs && samples > 1) ? samples : 1u;
   /* Colour attachment channel order, one bit per render target. The wrapper
    * stores each merged pixel as a word and nothing downstream reinterprets it,
    * so the order the shader packs is the order that reaches memory -- and the
    * targets of one draw need not agree, so this is a mask. */
   t.fs_bgra_mask = t.is_fs ? bgra_mask : 0u;
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
      /* Blinded: the sink is never read, so an optimizer that can still see the
       * alloca proves a store to it dead and rewrites the branchless address
       * select into a conditional store. That branch tests a per-lane predicate,
       * and the warp takes or skips it as one -- so a single suppressed lane
       * suppresses its neighbours' stores too. The address has to stay an
       * address for the store to remain unconditional. */
      t.fs_sink = emit_opaque(&t, LLVMBuildPtrToInt(t.b, sink, t.iptr, ""));
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
      /* Stop the threads the padding created, before anything they would touch.
       * The padded output slots are harmless -- the buffer is sized for them and
       * the host stops reading at the real count -- but the index-buffer load
       * below and any store the shader itself performs are addressed by the
       * vertex id, and for a thread past the end both land outside the draw. */
      LLVMValueRef vcnt_idx = LLVMConstInt(t.i32, VP_ARG_VS_COUNT, false);
      LLVMValueRef vcnt_p   = LLVMBuildGEP2(t.b, t.i64, t.arg, &vcnt_idx, 1, "");
      LLVMValueRef vcnt64   = LLVMBuildLoad2(t.b, t.i64, vcnt_p, "vcount64");
      LLVMValueRef vcnt     = LLVMBuildTrunc(t.b, vcnt64, t.i32, "vcount");
      LLVMValueRef in_draw  = LLVMBuildICmp(t.b, LLVMIntULT, t.vid, vcnt,
                                            "vs_in_draw");
      LLVMBasicBlockRef vs_go_bb  =
         LLVMAppendBasicBlockInContext(t.ctx, fn, "vs_go");
      LLVMBasicBlockRef vs_end_bb =
         LLVMAppendBasicBlockInContext(t.ctx, fn, "vs_end");
      LLVMBuildCondBr(t.b, in_draw, vs_go_bb, vs_end_bb);
      LLVMPositionBuilderAtEnd(t.b, vs_end_bb);
      LLVMBuildRetVoid(t.b);
      LLVMPositionBuilderAtEnd(t.b, vs_go_bb);
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
      /* arg slot 5: base vertex (gl_BaseVertex / the firstVertex half of
       * gl_VertexIndex); 0 on an indexed draw and on a draw that starts at
       * vertex 0, which keeps the emitted code identical to the prior ABI. */
      LLVMValueRef five  = LLVMConstInt(t.i32, 5, false);
      LLVMValueRef bvp   = LLVMBuildGEP2(t.b, t.i64, t.arg, &five, 1, "");
      LLVMValueRef bv64  = LLVMBuildLoad2(t.b, t.i64, bvp, "basevert64");
      t.base_vertex      = LLVMBuildTrunc(t.b, bv64, t.i32, "basevert");
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

      /* Every arg-block slot this stage reads is resolved above, so t.arg can
       * now be repointed at the VS constant-buffer table (VP_ARG_VS_DESC). The
       * shared load_ubo / load_ssbo / load_push_constant lowering indexes t.arg
       * by constant-buffer index, exactly as the fragment stage does; slots 0-4
       * carry vertex meanings, so reading the arg block directly would return
       * the attribute table's address for a UBO and the output buffer's for a
       * push constant. vp_raster_draw always supplies the table (zero-filled
       * when the VS binds nothing), so this never yields a null t.arg. */
      LLVMValueRef dsc   = LLVMConstInt(t.i32, VP_ARG_VS_DESC, false);
      LLVMValueRef dscp  = LLVMBuildGEP2(t.b, t.i64, t.arg, &dsc, 1, "");
      LLVMValueRef dsc64 = LLVMBuildLoad2(t.b, t.i64, dscp, "vsdesc64");
      t.arg = LLVMBuildIntToPtr(t.b, dsc64, t.ptr, "vsdesc");
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

/* Record the distinct descriptors a shader touches, up to `cap`, and return how
 * many were found -- a count that stops rising at `cap`, so a caller wanting to
 * know whether the shader exceeds a smaller limit has to pass room above it. */
static unsigned
vp_scan_descs_into(struct nir_shader *nir, struct vp_desc *out, unsigned cap)
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
            unsigned cbuf_index = VP_CBUF_SET0;  /* until the address says otherwise */
            bool writable = false;
            switch (in->intrinsic) {
            case nir_intrinsic_load_ssbo:
               off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
               break;
            case nir_intrinsic_store_ssbo:
               off = vp_desc_addr_offset(in->src[1].ssa, &cbuf_index);
               writable = true;
               break;
            case nir_intrinsic_ssbo_atomic:
            case nir_intrinsic_ssbo_atomic_swap:
               /* atomic RMW targets the same SSBO descriptor (src[0]); its
                * data pointer must be relocated like a load/store_ssbo. */
               off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
               writable = true;
               break;
            case nir_intrinsic_image_store:
            case nir_intrinsic_bindless_image_store:
            case nir_intrinsic_image_atomic:
            case nir_intrinsic_bindless_image_atomic:
            case nir_intrinsic_image_atomic_swap:
            case nir_intrinsic_bindless_image_atomic_swap:
               writable = true;
               FALLTHROUGH;
            case nir_intrinsic_image_load:
            case nir_intrinsic_bindless_image_load:
               /* storage image: src[0] is the bindless descriptor address, in
                * the same const_buf_base+binding form as an SSBO. lp_jit_image
                * sizing (height*row_stride) is done in the launch relocation.
                * The atomics take the descriptor in src[0] as well, and a shader
                * whose only access to an image is an atomic still needs the
                * descriptor relocated for that atomic to land on the device
                * copy rather than the host one. */
               off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
               kind = VP_DESC_IMAGE;
               elem_bytes = 0;
               break;
            case nir_intrinsic_load_ubo:
               if (nir_intrinsic_range(in) == -1) {
                  /* lavapipe reads the acceleration-structure handle as an
                   * UNBOUNDED ubo, and gives every other ubo read a real byte
                   * range — including the push constants it rewrites into one.
                   * The unbounded range is therefore the marker, and it marks an
                   * AS in every stage: a fragment shader's ray query reaches its
                   * structure through exactly this read, and a stage test ahead
                   * of this one would classify that as an ordinary buffer and
                   * relocate it as one. src[0] is the constant cbuf index. */
                  if (nir_src_is_const(in->src[1]))
                     off = (int)nir_src_as_uint(in->src[1]);
                  if (nir_src_is_const(in->src[0]))
                     cbuf_index = (unsigned)nir_src_as_uint(in->src[0]);
                  kind = VP_DESC_AS;
                  elem_bytes = 0;
               } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
                  /* A fragment UBO is a buffer descriptor reached via
                   * load_ubo(load_const_buf_base_addr_lvp(set+1)+binding, off) —
                   * relocate its lp_jit_buffer.ptr like an SSBO. A constant src[0]
                   * is a push-constant read (bound directly, no descriptor). A UBO
                   * descriptor stores num_elements in dwords (sizeof(float)). */
                  if (nir_src_is_const(in->src[0]))
                     continue;
                  off = vp_desc_addr_offset(in->src[0].ssa, &cbuf_index);
                  elem_bytes = 4;
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
             * is two different descriptors. A descriptor that is both loaded
             * and stored appears once per access, so writability accumulates
             * onto the entry already recorded -- dropping the duplicate
             * outright would lose the store when the load was seen first. */
            bool dup = false;
            for (unsigned k = 0; k < n; k++)
               if (out[k].offset == (unsigned)off &&
                   out[k].cbuf_index == cbuf_index) {
                  out[k].writable |= writable;
                  dup = true;
                  break;
               }
            if (dup || n >= cap)
               continue;
            out[n].offset     = (unsigned)off;
            out[n].cbuf_index = cbuf_index;
            out[n].kind       = kind;
            out[n].elem_bytes = elem_bytes;
            out[n].writable   = writable;
            n++;
         }
      }
   }
   return n;
}

/* Enough room above VP_MAX_DESCS to tell "exactly at the limit" from "past it".
 * A shader with more distinct descriptors than even this is past the limit by
 * any measure, so saturating here loses nothing. */
#define VP_DESC_SCAN_MAX 128

void
vp_scan_descriptors(struct nir_shader *nir,
                    struct vp_desc *out, unsigned *num_out)
{
   *num_out = vp_scan_descs_into(nir, out, VP_MAX_DESCS);
}

bool
vp_descriptors_overflow(struct nir_shader *nir)
{
   struct vp_desc scratch[VP_DESC_SCAN_MAX];
   return vp_scan_descs_into(nir, scratch, VP_DESC_SCAN_MAX) > VP_MAX_DESCS;
}

unsigned
vp_descriptors_max_cbuf(struct nir_shader *nir)
{
   struct vp_desc scratch[VP_DESC_SCAN_MAX];
   unsigned n = vp_scan_descs_into(nir, scratch, VP_DESC_SCAN_MAX);
   unsigned max = 0;
   for (unsigned i = 0; i < n; i++)
      if (scratch[i].cbuf_index > max)
         max = scratch[i].cbuf_index;
   return max;
}

/* Locate the FS's TEX-stage-0 texture descriptor: the sampled image's lp_descriptor
 * within its descriptor-set blob, from the first tex instruction's
 * nir_tex_src_texture_handle (the bindless descriptor address, in the same
 * const_buf_base+binding form as a storage image). Returns true + (cbuf_index,
 * offset) so the draw can read lp_jit_texture.base and select the sampled texture.
 * gfx-v1 drives a single TEX stage, so the first textured sample suffices. */
bool
vp_scan_tex_descriptor(struct nir_shader *nir, unsigned *cbuf_index, unsigned *offset)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(blk, impl) {
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            for (unsigned i = 0; i < tex->num_srcs; i++) {
               if (tex->src[i].src_type != nir_tex_src_texture_handle)
                  continue;
               unsigned cb = 1;
               int off = vp_desc_addr_offset(tex->src[i].src.ssa, &cb);
               if (off < 0)
                  return false;
               *cbuf_index = cb;
               *offset     = (unsigned)off;
               return true;
            }
         }
      }
   }
   return false;
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
