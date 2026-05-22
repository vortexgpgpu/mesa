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
   LLVMTypeRef    i8, i32, i64, f32, ptr;
   LLVMValueRef   arg;          /* the kernel's %arg parameter (ptr) */
   /* SSA map: [def index][component] -> iN value (bit pattern).
    * For a deref instr, component 0 holds the i32 byte address. */
   LLVMValueRef  *val;
   unsigned       nval;
   /* vertex-shader state (is_vs only) */
   bool           is_vs;
   LLVMValueRef   vid;          /* i32 vertex id */
   LLVMValueRef   out_base;     /* i32 output-buffer device address */
   unsigned       out_stride;   /* bytes per output vertex record */
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

/* byte size of a glsl type (32-bit scalars/vectors, arrays thereof) */
static unsigned
glsl_bytes(const struct glsl_type *gt)
{
   if (glsl_type_is_array(gt))
      return glsl_get_length(gt) * glsl_bytes(glsl_get_array_element(gt));
   return glsl_get_components(gt) * 4u;
}

/* LLVM storage type for a glsl type (every scalar is an i32). */
static LLVMTypeRef
glsl_to_llvm(struct vp_tr *t, const struct glsl_type *gt)
{
   if (glsl_type_is_array(gt))
      return LLVMArrayType(glsl_to_llvm(t, glsl_get_array_element(gt)),
                           glsl_get_length(gt));
   unsigned n = glsl_get_components(gt);
   return n > 1 ? LLVMArrayType(t->i32, n) : t->i32;
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
      LLVMValueRef v = (lc->def.bit_size == 64)
         ? LLVMConstInt(t->i64, lc->value[c].u64, false)
         : LLVMConstInt(t->i32, lc->value[c].u32, false);
      ssa_set(t, lc->def.index, c, v);
   }
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
      case nir_op_ishl:
         r = LLVMBuildShl(t->b, alu_src(t, alu, 0, c),
                                alu_src(t, alu, 1, c), "ishl");
         break;
      case nir_op_fadd: {
         LLVMValueRef a = LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c),
                                           t->f32, "");
         LLVMValueRef b = LLVMBuildBitCast(t->b, alu_src(t, alu, 1, c),
                                           t->f32, "");
         r = LLVMBuildBitCast(t->b, LLVMBuildFAdd(t->b, a, b, "fadd"),
                              t->i32, "");
         break;
      }
      case nir_op_fmul: {
         LLVMValueRef a = LLVMBuildBitCast(t->b, alu_src(t, alu, 0, c),
                                           t->f32, "");
         LLVMValueRef b = LLVMBuildBitCast(t->b, alu_src(t, alu, 1, c),
                                           t->f32, "");
         r = LLVMBuildBitCast(t->b, LLVMBuildFMul(t->b, a, b, "fmul"),
                              t->i32, "");
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
            LLVMValueRef a = LLVMBuildAlloca(t->b,
               glsl_to_llvm(t, v->type), v->name ? v->name : "tmp");
            e = &t->vars[t->nvars++];
            e->var = v; e->alloca = a; e->out_off = -1;
         }
         addr = LLVMBuildPtrToInt(t->b, e->alloca, t->i32, "");
      } else if (v->data.mode == nir_var_shader_out) {
         struct vp_var *e = vp_var_find(t, v);
         if (!e || e->out_off < 0) { t->ok = false; return; }
         /* out_base + vid * stride + slot_offset */
         LLVMValueRef voff = LLVMBuildMul(t->b, t->vid,
            LLVMConstInt(t->i32, t->out_stride, false), "");
         addr = LLVMBuildAdd(t->b, t->out_base, voff, "");
         addr = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i32, (unsigned)e->out_off, false), "vsout");
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
   case nir_intrinsic_load_local_invocation_id: {
      unsigned base = (in->intrinsic == nir_intrinsic_load_workgroup_id)
                         ? VX_CSR_CTA_BLOCK_ID_X : VX_CSR_CTA_THREAD_ID_X;
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
   case nir_intrinsic_load_const_buf_base_addr_lvp: {
      /* SSBO base address = arg_block[index] (array of i64). */
      LLVMValueRef idx = intr_src(t, in, 0);
      LLVMValueRef gep = LLVMBuildGEP2(t->b, t->i64, t->arg, &idx, 1, "");
      LLVMValueRef v   = LLVMBuildLoad2(t->b, t->i64, gep, "bufbase");
      ssa_set(t, in->def.index, 0, v);
      break;
   }
   case nir_intrinsic_load_ssbo: {
      LLVMValueRef base = intr_src(t, in, 0);                 /* i64 */
      LLVMValueRef off  = LLVMBuildZExt(t->b, intr_src(t, in, 1),
                                        t->i64, "");
      LLVMValueRef addr = LLVMBuildAdd(t->b, base, off, "");
      LLVMValueRef ptr  = LLVMBuildIntToPtr(t->b, addr, t->ptr, "");
      LLVMValueRef v    = LLVMBuildLoad2(t->b, t->i32, ptr, "ssbo");
      ssa_set(t, in->def.index, 0, v);
      break;
   }
   case nir_intrinsic_store_ssbo: {
      LLVMValueRef val  = intr_src(t, in, 0);                 /* i32 */
      LLVMValueRef base = intr_src(t, in, 1);                 /* i64 */
      LLVMValueRef off  = LLVMBuildZExt(t->b, intr_src(t, in, 2),
                                        t->i64, "");
      LLVMValueRef addr = LLVMBuildAdd(t->b, base, off, "");
      LLVMValueRef ptr  = LLVMBuildIntToPtr(t->b, addr, t->ptr, "");
      LLVMBuildStore(t->b, val, ptr);
      break;
   }
   /* deref load/store: the deref operand is an i32 byte address;
    * each 32-bit component is a separate riscv32 load/store. */
   case nir_intrinsic_load_deref: {
      LLVMValueRef addr = intr_src(t, in, 0);
      if (!addr) { t->ok = false; break; }
      for (unsigned c = 0; c < in->def.num_components; c++) {
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i32, c * 4u, false), "");
         LLVMValueRef p = LLVMBuildIntToPtr(t->b, a, t->ptr, "");
         ssa_set(t, in->def.index, c, LLVMBuildLoad2(t->b, t->i32, p, "ld"));
      }
      break;
   }
   case nir_intrinsic_store_deref: {
      LLVMValueRef addr = intr_src(t, in, 0);
      if (!addr) { t->ok = false; break; }
      unsigned mask = nir_intrinsic_write_mask(in);
      unsigned nc   = nir_src_num_components(in->src[1]);
      for (unsigned c = 0; c < nc; c++) {
         if (!(mask & (1u << c)))
            continue;
         LLVMValueRef v = ssa_get(t, in->src[1].ssa->index, c);
         if (!v) { t->ok = false; break; }
         LLVMValueRef a = LLVMBuildAdd(t->b, addr,
            LLVMConstInt(t->i32, c * 4u, false), "");
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
   case nir_instr_type_jump:
      break;   /* straight-line: only the implicit return */
   default:
      mesa_logw("vortexpipe: vp_nir_to_llvm: unhandled nir_instr_type %d",
                (int)instr->type);
      t->ok = false;
      break;
   }
   return t->ok;
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
         if (out_vs && out_vs->num_varyings < VP_VS_MAX_VARYINGS)
            out_vs->varying_loc[out_vs->num_varyings] = var->data.location;
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
   t.ctx   = LLVMContextCreate();
   t.mod   = LLVMModuleCreateWithNameInContext("vortex_shader", t.ctx);
   LLVMSetTarget(t.mod, VP_TRIPLE);
   LLVMSetDataLayout(t.mod, VP_DATALAYOUT);
   t.b   = LLVMCreateBuilderInContext(t.ctx);
   t.i8  = LLVMInt8TypeInContext(t.ctx);
   t.i32 = LLVMInt32TypeInContext(t.ctx);
   t.i64 = LLVMInt64TypeInContext(t.ctx);
   t.f32 = LLVMFloatTypeInContext(t.ctx);
   t.ptr = LLVMPointerTypeInContext(t.ctx, 0);

   /* the Vortex KMU kernel entry: void kernel_main(ptr arg).
    * The name is fixed -- the KMU startup (vx_start.S in
    * libvortex2.a) calls `kernel_main` from its .init. */
   LLVMTypeRef  params[1] = { t.ptr };
   LLVMTypeRef  fnty = LLVMFunctionType(LLVMVoidTypeInContext(t.ctx),
                                        params, 1, false);
   LLVMValueRef fn   = LLVMAddFunction(t.mod, "kernel_main", fnty);
   t.arg = LLVMGetParam(fn, 0);
   LLVMSetValueName2(t.arg, "arg", 3);

   LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(t.ctx, fn, "entry");
   LLVMPositionBuilderAtEnd(t.b, entry);

   /* Vertex-shader prologue: assign output slots, then read the
    * vertex id and the output-buffer base from %arg[0]. */
   if (t.is_vs) {
      vs_scan_outputs(&t, nir, out_vs);
      t.vid = emit_csr_read(&t, VX_CSR_CTA_THREAD_ID_X, "vid");
      LLVMValueRef ob64 = LLVMBuildLoad2(t.b, t.i64, t.arg, "outbase64");
      t.out_base = LLVMBuildTrunc(t.b, ob64, t.i32, "outbase");
   }

   nir_foreach_function_impl(impl, nir) {
      t.nval = impl->ssa_alloc;
      t.val  = calloc((size_t)t.nval * VP_MAXC, sizeof(LLVMValueRef));
      if (!t.val) { t.ok = false; break; }

      nir_foreach_block(blk, impl) {
         nir_foreach_instr(instr, blk) {
            if (!emit_instr(&t, instr))
               goto done;
         }
      }
   done:
      free(t.val);
      t.val = NULL;
      break;   /* one entrypoint impl per shader */
   }

   LLVMBuildRetVoid(t.b);

   if (t.ok)
      emit_kernel_annotation(&t, fn);

   char *err = NULL;
   bool ok = t.ok &&
             LLVMVerifyModule(t.mod, LLVMReturnStatusAction, &err) == 0;
   if (ok) {
      vp_dbg("vortexpipe: vp_nir_to_llvm: translated %s shader to a "
             "Vortex kernel module", t.is_vs ? "vertex" : "compute");
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
