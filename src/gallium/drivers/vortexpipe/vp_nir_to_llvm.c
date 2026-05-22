/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_nir_to_llvm -- scalar NIR -> LLVM-IR translator (Shape C).
 *
 * Phase 2 #3: per-instruction scalar emission.
 * Phase 2 #4a: emit a Vortex KMU kernel -- the function is
 *   `void main(ptr %arg)`, marked `vortex.kernel` via
 *   @llvm.global.annotations, targeting riscv32. The device-ABI
 *   intrinsics are lowered inline: workgroup/local id via `csrr`
 *   of the KMU CTA CSRs; the SSBO base address via a load from the
 *   %arg block (an array of i64 buffer addresses that vortexpipe's
 *   launch_grid fills -- increment #5).
 *
 * NIR SSA values are untyped bit patterns; each component is held
 * as an LLVM iN and ops bit-cast to the type they need.
 *
 * See docs/proposals/vulkan_support_proposal.md §3 in the Vortex tree.
 */

#include "vp_nir_to_llvm.h"
#include "vp_private.h"      /* vp_dbg */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir.h"
#include "util/log.h"

#include <llvm-c/Core.h>
#include <llvm-c/Analysis.h>

#define VP_MAXC 4   /* max vector components per NIR def */

/* KMU CTA CSRs (VX_types.h): per-thread / per-block index registers. */
#define VX_CSR_CTA_THREAD_ID_X 0xCD3   /* local invocation id, +c for y/z */
#define VX_CSR_CTA_BLOCK_ID_X  0xCD6   /* workgroup id,        +c for y/z */

/* riscv32 module target (XLEN=32 build). */
#define VP_TRIPLE      "riscv32-unknown-elf"
#define VP_DATALAYOUT  "e-m:e-p:32:32-i64:64-n32-S128"

struct vp_tr {
   LLVMContextRef ctx;
   LLVMModuleRef  mod;
   LLVMBuilderRef b;
   LLVMTypeRef    i8, i32, i64, f32, ptr;
   LLVMValueRef   arg;          /* the kernel's %arg parameter (ptr) */
   /* SSA map: [def index][component] -> iN value (bit pattern) */
   LLVMValueRef  *val;
   unsigned       nval;
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

bool
vp_nir_to_llvm(struct nir_shader *nir, char **out_ir)
{
   if (out_ir)
      *out_ir = NULL;
   if (!nir)
      return false;

   if (getenv("VORTEXPIPE_DEBUG_NIR")) {
      fprintf(stderr, "=== vortexpipe: lavapipe-lowered NIR ===\n");
      nir_print_shader(nir, stderr);
      fprintf(stderr, "=== end NIR ===\n");
   }

   struct vp_tr t = {0};
   t.ok  = true;
   t.ctx = LLVMContextCreate();
   t.mod = LLVMModuleCreateWithNameInContext("vortex_shader", t.ctx);
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
      break;   /* one entrypoint impl for compute */
   }

   LLVMBuildRetVoid(t.b);

   if (t.ok)
      emit_kernel_annotation(&t, fn);

   char *err = NULL;
   bool ok = t.ok &&
             LLVMVerifyModule(t.mod, LLVMReturnStatusAction, &err) == 0;
   if (ok) {
      vp_dbg("vortexpipe: vp_nir_to_llvm: translated shader to a "
                "Vortex kernel module");
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
