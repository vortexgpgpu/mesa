/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_nir_to_llvm -- scalar NIR -> LLVM-IR translator (Shape C).
 *
 * Emits a Vortex KMU kernel: `void kernel_main(ptr %arg)`, marked
 * `vortex.kernel` via @llvm.global.annotations, targeting riscv32.
 * Two shader stages are handled:
 *
 *   - Compute: one thread per work-item. SSBO base addresses come
 *     from the lavapipe descriptor buffer, reached through %arg.
 *   - Vertex (Phase 3): one thread per vertex. gl_VertexIndex is the
 *     CTA thread id; the kernel writes one padded-vec4 record per
 *     vertex into the output buffer whose device address is %arg[0].
 *
 * NIR SSA values are untyped bit patterns; each component is held as
 * an LLVM iN and ops bit-cast to the type they need. NIR derefs
 * resolve to i32 byte addresses (riscv32 pointers).
 *
 * See docs/proposals/vulkan_support_proposal.md §3 in the Vortex tree.
 */

#include "vp_nir_to_llvm.h"
#include "vp_private.h"      /* vp_dbg */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir.h"
#include "compiler/glsl_types.h"
#include "util/log.h"

#include <llvm-c/Core.h>
#include <llvm-c/Analysis.h>

#define VP_MAXC 4    /* max vector components per NIR def */
#define VP_MAXV 64   /* max tracked variables per shader */

/* KMU CTA CSRs (VX_types.h): per-thread / per-block index registers. */
#define VX_CSR_CTA_THREAD_ID_X 0xCD3   /* local invocation id, +c for y/z */
#define VX_CSR_CTA_BLOCK_ID_X  0xCD6   /* workgroup id,        +c for y/z */
#define VX_CSR_CTA_ID          0xCD0   /* workgroup id (barrier id)        */
#define VX_CSR_CTA_SIZE        0xCD2   /* warps per workgroup              */
#define VX_CSR_CTA_GRID_DIM_X  0xCDC   /* workgroup count, +c for y/z      */
#define VX_CSR_CTA_LMEM_ADDR   0xCDF   /* shared-memory base for this CTA  */

/* RISC-V custom-0 opcode -- vx_barrier lives here (custom-1 is graphics). */
#define VP_RISCV_CUSTOM0       11

/* RASTER CSRs (VX_types.h) -- latched per vx_rast() pop. */
#define VX_CSR_RASTER_BCOORD_X0 0x7C1  /* +i selects sub-pixel i (0..3) */
#define VX_CSR_RASTER_BCOORD_Y0 0x7C5
#define VX_CSR_RASTER_BCOORD_Z0 0x7C9
#define VX_CSR_RASTER_PID       0x7CD
#define VX_RASTER_DIM_BITS      15     /* pos_mask x/y field width + 1 */

/* graphics::rast_prim_t layout (sw/common/graphics.h, FIXEDPOINT):
 * vec3e_t edges[3] (36B), then rast_attribs_t {z,r,g,b,a,u,v}, each a
 * rast_attrib_t {x,y,z} of fixed24. r/g/b/a are the colour planes,
 * u/v the texcoord planes. */
#define VP_RAST_PRIM_STRIDE 120
#define VP_RAST_ATTR_Z       36
#define VP_RAST_ATTR_R       48
#define VP_RAST_ATTR_G       60
#define VP_RAST_ATTR_B       72
#define VP_RAST_ATTR_A       84
#define VP_RAST_ATTR_U       96
#define VP_RAST_ATTR_V      108

/* TEX unit: coordinates are S.23 fixed-point (VX_types.h VX_TEX_FXD_FRAC). */
#define VP_TEX_FXD_FRAC      23

/* riscv32 module target (XLEN=32 build). */
#define VP_TRIPLE      "riscv32-unknown-elf"
#define VP_DATALAYOUT  "e-m:e-p:32:32-i64:64-n32-S128"

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
   LLVMValueRef   arg;          /* the kernel's %arg parameter (ptr) */
   LLVMValueRef   lmem_base;    /* compute: shared-memory base (CTA LMEM) */
   LLVMBasicBlockRef entry;     /* function entry block (alloca home) */
   /* SSA map: [def index][component] -> iN value (bit pattern).
    * For a deref instr, component 0 holds the i32 byte address. */
   LLVMValueRef  *val;
   unsigned       nval;
   /* vertex-shader state (is_vs only) */
   bool           is_vs;
   LLVMValueRef   vid;          /* i32 vertex id */
   LLVMValueRef   out_base;     /* i32 output-buffer device address */
   unsigned       out_stride;   /* bytes per output vertex record */
   LLVMValueRef   attr_table;   /* i32 addr of the {base,stride}[] table */
   /* fragment-shader state (is_fs only) */
   bool           is_fs;
   LLVMValueRef   fs_in_base;   /* i32 interpolated-varyings area */
   LLVMValueRef   fs_out_base;  /* i32 output-colour area */
   struct vp_var  vars[VP_MAXV];
   unsigned       nvars;
   bool           ok;
};

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
 * at base + vid*stride (vp_launch_vs builds the table). */
static LLVMValueRef
emit_vs_attr_addr(struct vp_tr *t, unsigned loc)
{
   LLVMValueRef ent = LLVMBuildAdd(t->b, t->attr_table,
      LLVMConstInt(t->i32, loc * 8u, false), "");
   LLVMValueRef base = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildIntToPtr(t->b, ent, t->ptr, ""), "attrbase");
   LLVMValueRef stride = LLVMBuildLoad2(t->b, t->i32,
      LLVMBuildIntToPtr(t->b,
         LLVMBuildAdd(t->b, ent, LLVMConstInt(t->i32, 4, false), ""),
         t->ptr, ""), "attrstride");
   return LLVMBuildAdd(t->b, base,
      LLVMBuildMul(t->b, t->vid, stride, ""), "vsin");
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
      case nir_op_ishl:
         r = LLVMBuildShl(t->b, alu_src(t, alu, 0, c),
                                alu_src(t, alu, 1, c), "ishl");
         break;
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
      case nir_op_ishr:
         r = LLVMBuildAShr(t->b, alu_src(t, alu, 0, c),
                                 alu_src(t, alu, 1, c), "ishr");
         break;
      case nir_op_ushr:
         r = LLVMBuildLShr(t->b, alu_src(t, alu, 0, c),
                                 alu_src(t, alu, 1, c), "ushr");
         break;
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
      case nir_op_fabs: case nir_op_ffma: case nir_op_fsign: {
         LLVMTypeRef  ft = fty(t, alu->def.bit_size);
         LLVMValueRef fa = LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), ft, "");
         LLVMValueRef z  = LLVMConstReal(ft, 0.0);
         LLVMValueRef res;
         if (alu->op == nir_op_fneg) {
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
      /* float width conversions */
      case nir_op_f2f64:
         r = LLVMBuildBitCast(t->b, LLVMBuildFPExt(t->b,
            LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), t->f32, ""),
            t->f64, ""), t->i64, "f2f64");
         break;
      case nir_op_f2f32:
         r = LLVMBuildBitCast(t->b, LLVMBuildFPTrunc(t->b,
            LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c), t->f64, ""),
            t->f32, ""), t->i32, "f2f32");
         break;
      /* integer modulo + 64-bit pack */
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
         addr = LLVMBuildPtrToInt(t->b, e->alloca, t->i32, "");
      } else if (v->data.mode == nir_var_shader_out) {
         struct vp_var *e = vp_var_find(t, v);
         if (!e || e->out_off < 0) { t->ok = false; return; }
         LLVMValueRef off = LLVMConstInt(t->i32, (unsigned)e->out_off, false);
         if (t->is_vs) {
            /* vertex shader: out_base + vid * stride + slot_offset */
            LLVMValueRef voff = LLVMBuildMul(t->b, t->vid,
               LLVMConstInt(t->i32, t->out_stride, false), "");
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
            LLVMConstInt(t->i32, (unsigned)e->out_off, false), "fsin");
      } else if (v->data.mode == nir_var_shader_in && t->is_vs) {
         /* vertex shader input: a per-vertex attribute fetched from
          * the bound vertex buffer (deref-based NIR path). */
         addr = emit_vs_attr_addr(t, v->data.driver_location);
      } else if (v->data.mode == nir_var_uniform) {
         /* a combined image-sampler handle. gfx-v1 binds a single
          * texture to TEX stage 0 through DCRs, so the deref carries
          * no address -- emit_tex ignores the texture/sampler deref
          * sources. A placeholder keeps the SSA chain well-formed. */
         addr = LLVMConstInt(t->i32, 0, false);
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
      LLVMValueRef off = LLVMBuildMul(t->b, idx,
         LLVMConstInt(t->i32, glsl_bytes(d->type), false), "");
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
   /* vertex-attribute fetch (lowered-IO NIR path): read each component
    * from the bound vertex buffer at attr base + component offset. */
   case nir_intrinsic_load_input: {
      unsigned loc  = nir_intrinsic_base(in);
      unsigned comp = nir_intrinsic_component(in);
      LLVMValueRef attr = emit_vs_attr_addr(t, loc);
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, attr,
            LLVMConstInt(t->i32, (comp + c) * 4u, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, t->i32, p, "in"));
      }
      break;
   }
   case nir_intrinsic_load_const_buf_base_addr_lvp: {
      /* SSBO/UBO base address = arg_block[index] (array of i64). A
       * fragment shader has no arg block -- the only descriptor it
       * touches in gfx-v1 is the combined image-sampler, which is
       * bound to TEX stage 0 by DCRs and addressed by stage, not by
       * a handle. So the value only ever feeds a vx_tex handle source
       * (which emit_tex ignores); a placeholder keeps it well-formed. */
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
      if (in->intrinsic == nir_intrinsic_load_ubo) {
         if (t->arg) {
            LLVMValueRef idx = intr_src(t, in, 0);
            LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, t->arg, &idx, 1, "");
            base = LLVMBuildLoad2(t->b, t->i64, gep, "ubobase");
         } else {
            base = LLVMConstInt(t->i64, 0, false);
         }
      } else {
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
       * pointer is its first 8 bytes (see load_ssbo above). */
      LLVMValueRef desc = intr_src(t, in, 1);
      LLVMValueRef dp   = LLVMBuildIntToPtr(t->b, desc, t->ptr, "");
      LLVMValueRef base = LLVMBuildLoad2(t->b, t->i64, dp, "ssbobase");
      LLVMValueRef off  = LLVMBuildZExt(t->b, intr_src(t, in, 2),
                                        t->i64, "");
      LLVMValueRef addr = LLVMBuildAdd(t->b, base, off, "");
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
      LLVMValueRef addr = intr_src(t, in, 1);
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
            LLVMConstInt(t->i32, nir_intrinsic_base(in), false), ""),
         intr_src(t, in, 0), "shaddr");
      LLVMTypeRef lt  = ity(t, in->def.bit_size);
      unsigned    esz = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i32, c * esz, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, lt, p, "sld"));
      }
      break;
   }
   case nir_intrinsic_store_shared: {
      if (!t->lmem_base) { t->ok = false; break; }
      LLVMValueRef addr = LLVMBuildAdd(t->b,
         LLVMBuildAdd(t->b, t->lmem_base,
            LLVMConstInt(t->i32, nir_intrinsic_base(in), false), ""),
         intr_src(t, in, 1), "shaddr");
      unsigned mask = nir_intrinsic_write_mask(in);
      unsigned nc   = nir_src_num_components(in->src[0]);
      unsigned esz  = nir_src_bit_size(in->src[0]) / 8u;
      for (unsigned c = 0; c < nc; c++) {
         if (!(mask & (1u << c))) continue;
         LLVMValueRef v = ssa_get(t, in->src[0].ssa->index, c);
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i32, c * esz, false), "");
         if (v)
            LLVMBuildStore(t->b, v, LLVMBuildIntToPtr(t->b, a, t->ptr, ""));
      }
      break;
   }
   /* workgroup barrier: only the execution-scoped form needs a sync;
    * a pure memory barrier is a no-op in this per-thread model. */
   case nir_intrinsic_barrier:
      if (nir_intrinsic_execution_scope(in) != SCOPE_NONE)
         emit_vx_barrier(t);
      break;
   /* deref load/store: the deref operand is an i32 byte address; each
    * component is a separate riscv32 load/store, width from bit size. */
   case nir_intrinsic_load_deref: {
      LLVMValueRef addr = intr_src(t, in, 0);
      if (!addr) { t->ok = false; break; }
      LLVMTypeRef lt  = ity(t, in->def.bit_size);
      unsigned    esz = in->def.bit_size / 8u;
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i32, c * esz, false), "");
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
            LLVMConstInt(t->i32, c * esz, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         LLVMBuildStore(t->b, v, p);
      }
      break;
   }
   default:
      mesa_logw("vortexpipe: vp_nir_to_llvm: unhandled intrinsic '%s'",
                nir_intrinsic_infos[in->intrinsic].name);
      t->ok = false;
   }
}

/* vx_tex(): sample TEX stage 0 (custom-1, funct3=1, R4-type, funct2=
 * stage). Returns the filtered texel as a packed A8R8G8B8 word. */
static LLVMValueRef
emit_vx_tex(struct vp_tr *t, LLVMValueRef u, LLVMValueRef v,
            LLVMValueRef lod)
{
   const char *s = ".insn r4 43, 1, 0, $0, $1, $2, $3";
   LLVMTypeRef args[3] = { t->i32, t->i32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, args, 3, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "=r,r,r,r", 8,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   LLVMValueRef a[3] = { u, v, lod };
   return LLVMBuildCall2(t->b, fnty, ia, a, 3, "tex");
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
   if (tex->op != nir_texop_tex || !u || !v) {
      mesa_logw("vortexpipe: vp_nir_to_llvm: unsupported texture op");
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
 * lowers to -- the llvm_vortex backend keys on it). */
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
   off = 0;
   nir_foreach_shader_out_variable(var, nir) {
      if (t->nvars >= VP_MAXV) { t->ok = false; return; }
      t->vars[t->nvars].var     = var;
      t->vars[t->nvars].alloca  = NULL;
      t->vars[t->nvars].out_off = (int)off;
      t->nvars++;
      off += 16;
   }
}

/* ---- fragment-kernel wrapper (Phase 4 Step 2) ----------------------- *
 * A fragment shader runs on Vortex as a rasterizer-driven kernel: every
 * thread polls vx_rast() for quads, the wrapper interpolates the
 * varyings from the bcoord CSRs + the primitive buffer, calls the
 * translated fragment-shader body (fs_main), and writes the framebuffer.
 * The wrapper is hand-emitted -- it is the driver's fixed-function glue
 * around the programmable stage, and the translator's first emission of
 * control flow (a loop + per-pixel branches).                          */

static LLVMValueRef
addk(struct vp_tr *t, LLVMValueRef v, unsigned k)
{
   return LLVMBuildAdd(t->b, v, LLVMConstInt(t->i32, k, false), "");
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

/* vx_rast(): pop a quad from the raster unit (custom-1, funct3=3).
 * Returns the pos_mask word; 0 means the queue is drained. */
static LLVMValueRef
emit_vx_rast(struct vp_tr *t)
{
   const char *s = ".insn r 43, 3, 0, $0, x0, x0";
   LLVMTypeRef fnty = LLVMFunctionType(t->i32, NULL, 0, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "=r", 3,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   return LLVMBuildCall2(t->b, fnty, ia, NULL, 0, "rast");
}

/* vx_om: submit a fragment to the output-merger unit (custom-1,
 * funct3=2, R4-type). pos_face = (y<<16)|(x<<1)|face; the OM does
 * depth/stencil/blend and writes the colour + depth buffers. */
static void
emit_vx_om(struct vp_tr *t, LLVMValueRef pos_face,
           LLVMValueRef color, LLVMValueRef depth)
{
   const char *s = ".insn r4 43, 2, 0, x0, $0, $1, $2";
   LLVMTypeRef args[3] = { t->i32, t->i32, t->i32 };
   LLVMTypeRef fnty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                       args, 3, false);
   LLVMValueRef ia = LLVMGetInlineAsm(fnty, s, strlen(s), "r,r,r", 5,
                                      /*HasSideEffects*/ true,
                                      /*IsAlignStack*/ false,
                                      LLVMInlineAsmDialectATT,
                                      /*CanThrow*/ false);
   LLVMValueRef a[3] = { pos_face, color, depth };
   LLVMBuildCall2(t->b, fnty, ia, a, 3, "");
}

/* reinterpret a raw fixed-point i32 as float: (float)raw / 2^frac */
static LLVMValueRef
emit_fixed_to_float(struct vp_tr *t, LLVMValueRef raw, unsigned frac)
{
   LLVMValueRef f = LLVMBuildSIToFP(t->b, raw, t->f32, "");
   return LLVMBuildFMul(t->b, f,
      LLVMConstReal(t->f32, 1.0 / (double)(1u << frac)), "");
}

/* interpolate one rast_attrib_t {x,y,z} plane: x*dx + y*dy + z. */
static LLVMValueRef
emit_interp(struct vp_tr *t, LLVMValueRef attr,
            LLVMValueRef dx, LLVMValueRef dy)
{
   LLVMValueRef ax = emit_fixed_to_float(t, emit_load_i32(t, attr), 24);
   LLVMValueRef ay = emit_fixed_to_float(t, emit_load_i32(t, addk(t, attr, 4)), 24);
   LLVMValueRef az = emit_fixed_to_float(t, emit_load_i32(t, addk(t, attr, 8)), 24);
   LLVMValueRef r  = LLVMBuildFMul(t->b, ax, dx, "");
   r = LLVMBuildFAdd(t->b, r, LLVMBuildFMul(t->b, ay, dy, ""), "");
   return LLVMBuildFAdd(t->b, r, az, "");
}

/* float colour component in [0,1] -> 8-bit integer (clamped). */
static LLVMValueRef
emit_to_byte(struct vp_tr *t, LLVMValueRef f)
{
   LLVMValueRef z = LLVMConstReal(t->f32, 0.0);
   LLVMValueRef o = LLVMConstReal(t->f32, 1.0);
   f = LLVMBuildSelect(t->b, LLVMBuildFCmp(t->b, LLVMRealOGT, f, z, ""),
                       f, z, "");
   f = LLVMBuildSelect(t->b, LLVMBuildFCmp(t->b, LLVMRealOLT, f, o, ""),
                       f, o, "");
   return LLVMBuildFPToUI(t->b,
      LLVMBuildFMul(t->b, f, LLVMConstReal(t->f32, 255.0), ""), t->i32, "");
}

/* read arg-block slot k (an i64 device address) truncated to i32 */
static LLVMValueRef
emit_arg_i32(struct vp_tr *t, LLVMValueRef arg, unsigned k)
{
   LLVMValueRef idx = LLVMConstInt(t->i32, k, false);
   LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, arg, &idx, 1, "");
   return LLVMBuildTrunc(t->b, LLVMBuildLoad2(t->b, t->i64, gep, ""),
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
                      LLVMValueRef dx, LLVMValueRef dy)
{
   static const unsigned colour[4]   = {
      VP_RAST_ATTR_R, VP_RAST_ATTR_G, VP_RAST_ATTR_B, VP_RAST_ATTR_A };
   static const unsigned texcoord[2] = {
      VP_RAST_ATTR_U, VP_RAST_ATTR_V };

   for (unsigned i = 0; i < t->nvars; i++) {
      const nir_variable *var = t->vars[i].var;
      if (!var || var->data.mode != nir_var_shader_in ||
          t->vars[i].out_off < 0)
         continue;
      unsigned nc = glsl_get_components(var->type);
      const unsigned *plane  = (nc <= 2) ? texcoord : colour;
      unsigned        planes = (nc <= 2) ? 2u : 4u;
      LLVMValueRef    slot   = addk(t, in_addr, (unsigned)t->vars[i].out_off);
      for (unsigned c = 0; c < nc && c < planes; c++) {
         LLVMValueRef f = emit_interp(t, addk(t, prim, plane[c]), dx, dy);
         emit_store_i32(t, addk(t, slot, c * 4),
                        LLVMBuildBitCast(t->b, f, t->i32, ""));
      }
   }
}

/* Build kernel_main: the rasterizer poll-loop wrapper that drives the
 * translated fragment body `fs_main`. arg block: [0]=primitive buffer,
 * [1]=colour buffer, [2]=colour-buffer row pitch (bytes). */
static LLVMValueRef
emit_fs_wrapper(struct vp_tr *t, LLVMValueRef fs_main, LLVMTypeRef fs_main_ty)
{
   LLVMTypeRef  p1[1] = { t->ptr };
   LLVMTypeRef  kty = LLVMFunctionType(LLVMVoidTypeInContext(t->ctx),
                                       p1, 1, false);
   LLVMValueRef fn  = LLVMAddFunction(t->mod, "kernel_main", kty);
   LLVMValueRef arg = LLVMGetParam(fn, 0);

   LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(t->ctx, fn, "entry");
   LLVMBasicBlockRef loop  = LLVMAppendBasicBlockInContext(t->ctx, fn, "loop");
   LLVMBasicBlockRef body  = LLVMAppendBasicBlockInContext(t->ctx, fn, "body");
   LLVMBasicBlockRef exit  = LLVMAppendBasicBlockInContext(t->ctx, fn, "exit");

   /* entry: read the arg block, allocate per-pixel scratch. The
    * colour/depth buffers are reached by the OM unit through its
    * DCRs, so the kernel only needs the primitive buffer (arg[0]). */
   LLVMPositionBuilderAtEnd(t->b, entry);
   LLVMValueRef prim_base = emit_arg_i32(t, arg, 0);
   LLVMValueRef in_scr  = LLVMBuildAlloca(t->b, LLVMArrayType(t->i32, 16),
                                          "fs_in");
   LLVMValueRef out_scr = LLVMBuildAlloca(t->b, LLVMArrayType(t->i32, 4),
                                          "fs_out");
   LLVMValueRef in_addr  = LLVMBuildPtrToInt(t->b, in_scr,  t->i32, "");
   LLVMValueRef out_addr = LLVMBuildPtrToInt(t->b, out_scr, t->i32, "");
   LLVMBuildBr(t->b, loop);

   /* loop: pop a quad; stop when the raster queue drains. */
   LLVMPositionBuilderAtEnd(t->b, loop);
   LLVMValueRef pos_mask = emit_vx_rast(t);
   LLVMBuildCondBr(t->b,
      LLVMBuildICmp(t->b, LLVMIntEQ, pos_mask,
                    LLVMConstInt(t->i32, 0, false), ""),
      exit, body);

   /* body: unpack the quad, shade its covered sub-pixels. */
   LLVMPositionBuilderAtEnd(t->b, body);
   LLVMValueRef pid  = emit_csr_read(t, VX_CSR_RASTER_PID, "pid");
   LLVMValueRef prim = LLVMBuildAdd(t->b, prim_base,
      LLVMBuildMul(t->b, pid,
                   LLVMConstInt(t->i32, VP_RAST_PRIM_STRIDE, false), ""),
      "prim");
   LLVMValueRef mask = LLVMBuildAnd(t->b, pos_mask,
      LLVMConstInt(t->i32, 0xf, false), "mask");
   unsigned dim_mask = (1u << (VX_RASTER_DIM_BITS - 1)) - 1;
   LLVMValueRef qx = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, pos_mask, LLVMConstInt(t->i32, 4, false), ""),
      LLVMConstInt(t->i32, dim_mask, false), "qx");
   LLVMValueRef qy = LLVMBuildAnd(t->b,
      LLVMBuildLShr(t->b, pos_mask,
                    LLVMConstInt(t->i32, 4 + VX_RASTER_DIM_BITS - 1, false), ""),
      LLVMConstInt(t->i32, dim_mask, false), "qy");

   for (unsigned i = 0; i < 4; i++) {
      LLVMBasicBlockRef px  = LLVMAppendBasicBlockInContext(t->ctx, fn, "px");
      LLVMBasicBlockRef nxt = LLVMAppendBasicBlockInContext(t->ctx, fn, "nxt");
      LLVMValueRef cov = LLVMBuildAnd(t->b,
         LLVMBuildLShr(t->b, mask, LLVMConstInt(t->i32, i, false), ""),
         LLVMConstInt(t->i32, 1, false), "");
      LLVMBuildCondBr(t->b,
         LLVMBuildICmp(t->b, LLVMIntNE, cov,
                       LLVMConstInt(t->i32, 0, false), ""), px, nxt);

      LLVMPositionBuilderAtEnd(t->b, px);
      /* barycentric gradient: dx = F0/(F0+F1+F2), dy = F1/sum. */
      LLVMValueRef f0 = emit_fixed_to_float(t,
         emit_csr_read(t, VX_CSR_RASTER_BCOORD_X0 + i, "f0"), 16);
      LLVMValueRef f1 = emit_fixed_to_float(t,
         emit_csr_read(t, VX_CSR_RASTER_BCOORD_Y0 + i, "f1"), 16);
      LLVMValueRef f2 = emit_fixed_to_float(t,
         emit_csr_read(t, VX_CSR_RASTER_BCOORD_Z0 + i, "f2"), 16);
      LLVMValueRef sum = LLVMBuildFAdd(t->b,
         LLVMBuildFAdd(t->b, f0, f1, ""), f2, "");
      LLVMValueRef recip = LLVMBuildFDiv(t->b,
         LLVMConstReal(t->f32, 1.0), sum, "recip");
      LLVMValueRef dx = LLVMBuildFMul(t->b, recip, f0, "dx");
      LLVMValueRef dy = LLVMBuildFMul(t->b, recip, f1, "dy");

      /* interpolate the RASTER attribute planes into the FS input
       * varyings (colour and/or texcoord, by declaration). */
      emit_fs_fill_varyings(t, prim, in_addr, dx, dy);

      /* run the programmable fragment shader */
      LLVMValueRef cargs[2] = { in_scr, out_scr };
      LLVMBuildCall2(t->b, fs_main_ty, fs_main, cargs, 2, "");

      /* pack the FS output (4 floats) into an R8G8B8A8 pixel. */
      LLVMValueRef rgba = LLVMConstInt(t->i32, 0, false);
      for (unsigned c = 0; c < 4; c++) {
         LLVMValueRef fc = LLVMBuildBitCast(t->b,
            emit_load_i32(t, addk(t, out_addr, c * 4)), t->f32, "");
         LLVMValueRef bc = LLVMBuildShl(t->b, emit_to_byte(t, fc),
            LLVMConstInt(t->i32, c * 8, false), "");
         rgba = LLVMBuildOr(t->b, rgba, bc, "");
      }

      /* interpolate the fragment depth (rast_attribs.z, [0,1]) and
       * convert to the OM's 24-bit fixed-point depth. */
      LLVMValueRef dz = emit_interp(t, addk(t, prim, VP_RAST_ATTR_Z),
                                    dx, dy);
      LLVMValueRef depth_i = LLVMBuildFPToUI(t->b,
         LLVMBuildFMul(t->b, dz, LLVMConstReal(t->f32, 16777216.0), ""),
         t->i32, "");

      /* submit the fragment to the OM unit: it does depth/stencil/
       * blend and writes the colour + depth buffers. pos_face packs
       * (y<<16)|(x<<1)|face -- face 0 (front). */
      LLVMValueRef px_i = LLVMBuildAdd(t->b,
         LLVMBuildShl(t->b, qx, LLVMConstInt(t->i32, 1, false), ""),
         LLVMConstInt(t->i32, i & 1, false), "");
      LLVMValueRef py_i = LLVMBuildAdd(t->b,
         LLVMBuildShl(t->b, qy, LLVMConstInt(t->i32, 1, false), ""),
         LLVMConstInt(t->i32, i >> 1, false), "");
      LLVMValueRef pos_face = LLVMBuildOr(t->b,
         LLVMBuildShl(t->b, py_i, LLVMConstInt(t->i32, 16, false), ""),
         LLVMBuildShl(t->b, px_i, LLVMConstInt(t->i32, 1, false), ""), "");
      emit_vx_om(t, pos_face, rgba, depth_i);
      LLVMBuildBr(t->b, nxt);

      LLVMPositionBuilderAtEnd(t->b, nxt);   /* fall through to next pixel */
   }
   LLVMBuildBr(t->b, loop);

   LLVMPositionBuilderAtEnd(t->b, exit);
   LLVMBuildRetVoid(t->b);
   return fn;
}

bool
vp_nir_to_llvm(struct nir_shader *nir, char **out_ir,
               struct vp_vs_layout *out_vs)
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
   t.ctx   = LLVMContextCreate();
   t.mod   = LLVMModuleCreateWithNameInContext("vortex_shader", t.ctx);
   LLVMSetTarget(t.mod, VP_TRIPLE);
   LLVMSetDataLayout(t.mod, VP_DATALAYOUT);
   t.b   = LLVMCreateBuilderInContext(t.ctx);
   t.i8  = LLVMInt8TypeInContext(t.ctx);
   t.i32 = LLVMInt32TypeInContext(t.ctx);
   t.i64 = LLVMInt64TypeInContext(t.ctx);
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
      LLVMTypeRef p2[2] = { t.ptr, t.ptr };
      fs_main_ty = LLVMFunctionType(LLVMVoidTypeInContext(t.ctx),
                                    p2, 2, false);
      fn = LLVMAddFunction(t.mod, "fs_main", fs_main_ty);
      LLVMSetLinkage(fn, LLVMInternalLinkage);
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

   /* Vertex-shader prologue: assign output slots, then read the
    * vertex id and the output-buffer base from %arg[0]. */
   if (t.is_vs) {
      vs_scan_outputs(&t, nir, out_vs);
      if (out_vs)
         out_vs->needs_vertex_input = (nir->info.inputs_read != 0);
      t.vid = emit_csr_read(&t, VX_CSR_CTA_THREAD_ID_X, "vid");
      LLVMValueRef ob64 = LLVMBuildLoad2(t.b, t.i64, t.arg, "outbase64");
      t.out_base = LLVMBuildTrunc(t.b, ob64, t.i32, "outbase");
      /* arg slot 1: the vertex-attribute table (vp_launch_vs) -- 0 for
       * a self-contained VS that fetches no vertex-buffer inputs. */
      LLVMValueRef one = LLVMConstInt(t.i32, 1, false);
      LLVMValueRef atp = LLVMBuildGEP2(t.b, t.i64, t.arg, &one, 1, "");
      t.attr_table = LLVMBuildTrunc(t.b,
         LLVMBuildLoad2(t.b, t.i64, atp, "attrtab64"), t.i32, "attrtab");
   }

   /* Fragment-shader prologue: assign varying/output slots. fs_main's
    * two ptr params are the per-pixel interpolated-varyings input and
    * the colour-output area; the wrapper (emit_fs_wrapper) fills them. */
   if (t.is_fs) {
      fs_scan_io(&t, nir);
      t.fs_in_base  = LLVMBuildPtrToInt(t.b, LLVMGetParam(fn, 0),
                                        t.i32, "fsin");
      t.fs_out_base = LLVMBuildPtrToInt(t.b, LLVMGetParam(fn, 1),
                                        t.i32, "fsout");
   }

   /* Compute-shader prologue: the workgroup's shared-memory base, read
    * once so load/store_shared can address it. */
   if (!t.is_vs && !t.is_fs)
      t.lmem_base = emit_csr_read(&t, VX_CSR_CTA_LMEM_ADDR, "lmem");

   nir_foreach_function_impl(impl, nir) {
      t.nval = impl->ssa_alloc;
      t.val  = calloc((size_t)t.nval * VP_MAXC, sizeof(LLVMValueRef));
      if (!t.val) { t.ok = false; break; }

      /* walk the control-flow graph (emits each block's terminator,
       * including the function return). */
      emit_cfg(&t, impl, fn, entry);

      free(t.val);
      t.val = NULL;
      break;   /* one entrypoint impl per shader */
   }

   /* Fragment shaders: wrap fs_main in the rasterizer poll-loop
    * kernel_main. Compute/vertex shaders are already kernel_main. */
   LLVMValueRef kfn = fn;
   if (t.is_fs && t.ok)
      kfn = emit_fs_wrapper(&t, fn, fs_main_ty);

   if (t.ok)
      emit_kernel_annotation(&t, kfn);

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

/* ---- descriptor scan (Phase 7.5) ----------------------------------- *
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

/* Resolve an SSBO descriptor-address SSA value to its byte offset in
 * the set-0 descriptor buffer: load_const_buf_base_addr_lvp(1) is
 * offset 0, that plus a constant is the constant. -1 if unrecognized. */
static int
vp_desc_addr_offset(nir_def *def)
{
   nir_instr *p = def->parent_instr;
   if (p->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *i = nir_instr_as_intrinsic(p);
      if (i->intrinsic == nir_intrinsic_load_const_buf_base_addr_lvp)
         return 0;
   } else if (p->type == nir_instr_type_alu) {
      nir_alu_instr *a = nir_instr_as_alu(p);
      if (a->op == nir_op_iadd) {
         for (int s = 0; s < 2; s++) {
            nir_instr *o = a->src[s].src.ssa->parent_instr;
            if (o->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(o)->intrinsic ==
                   nir_intrinsic_load_const_buf_base_addr_lvp &&
                nir_src_is_const(a->src[!s].src))
               return (int)nir_src_as_uint(a->src[!s].src);
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
            switch (in->intrinsic) {
            case nir_intrinsic_load_ssbo:
               off = vp_desc_addr_offset(in->src[0].ssa);
               break;
            case nir_intrinsic_store_ssbo:
               off = vp_desc_addr_offset(in->src[1].ssa);
               break;
            case nir_intrinsic_load_ubo:
               /* lavapipe lowers an acceleration-structure read to
                * load_ubo(cbuf_index, byte_offset). */
               if (nir_src_is_const(in->src[1]))
                  off = (int)nir_src_as_uint(in->src[1]);
               kind = VP_DESC_AS;
               break;
            default:
               continue;
            }
            if (off < 0)
               continue;
            bool dup = false;
            for (unsigned k = 0; k < n; k++)
               if (out[k].offset == (unsigned)off) { dup = true; break; }
            if (dup || n >= VP_MAX_DESCS)
               continue;
            out[n].offset = (unsigned)off;
            out[n].kind   = kind;
            n++;
         }
      }
   }
   *num_out = n;
}
