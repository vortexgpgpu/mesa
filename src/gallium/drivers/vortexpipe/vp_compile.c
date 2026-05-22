/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_compile -- Phase 2 #4b: LLVM IR -> Vortex .vxbin.
 *
 * Drives the existing Vortex device toolchain on the kernel module
 * vp_nir_to_llvm emits: llvm_vortex's clang compiles + links the IR
 * (riscv32, +xvortex) against the KMU device kernel library
 * (libvortex2.a) and the baremetal libc/compiler-rt, then vxbin.py
 * packages the ELF into a .vxbin. First cut: fork/exec the tools
 * via system(); moving in-process is a later optimization (§3 n.4).
 *
 * Paths come from VORTEX_HOME (the Vortex tree) and TOOLDIR (the
 * $HOME/tools install root) -- the meson-baked defaults below,
 * overridable by the VORTEX_HOME / VORTEX_TOOLDIR env vars. The
 * Vortex out-of-tree build dir (where libvortex2.a is built) is
 * VORTEX_HOME/build by convention; VORTEX_BUILD overrides it for
 * trees configured under a differently named build dir (e.g. the
 * CI build32/build64).
 */

#define _GNU_SOURCE
#include "vp_compile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/log.h"

#ifndef VP_VORTEX_HOME
#define VP_VORTEX_HOME ""
#endif
#ifndef VP_TOOLDIR
#define VP_TOOLDIR ""
#endif

static bool
write_text(const char *path, const char *data)
{
   FILE *f = fopen(path, "wb");
   if (!f)
      return false;
   size_t n = strlen(data);
   bool ok = fwrite(data, 1, n, f) == n;
   fclose(f);
   return ok;
}

static void *
read_blob(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   void *buf = (sz > 0) ? malloc((size_t)sz) : NULL;
   if (buf && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
      free(buf);
      buf = NULL;
   }
   fclose(f);
   if (buf)
      *out_size = (size_t)sz;
   return buf;
}

bool
vp_compile_vxbin(const char *llvm_ir, void **out_blob, size_t *out_size)
{
   *out_blob = NULL;
   *out_size = 0;

   const char *vh = getenv("VORTEX_HOME");
   if (!vh || !*vh) vh = VP_VORTEX_HOME;
   const char *td = getenv("VORTEX_TOOLDIR");
   if (!td || !*td) td = VP_TOOLDIR;
   if (!*vh || !*td) {
      mesa_logw("vortexpipe: VORTEX_HOME / TOOLDIR not configured");
      return false;
   }

   /* Build dir holding libvortex2.a -- VORTEX_HOME/build by default. */
   char bd_buf[512];
   const char *bd = getenv("VORTEX_BUILD");
   if (!bd || !*bd) {
      snprintf(bd_buf, sizeof bd_buf, "%s/build", vh);
      bd = bd_buf;
   }

   char dir[] = "/tmp/vortexpipe.XXXXXX";
   if (!mkdtemp(dir)) {
      mesa_logw("vortexpipe: mkdtemp failed");
      return false;
   }

   char p_ll[256], p_elf[256], p_bin[256];
   snprintf(p_ll,  sizeof p_ll,  "%s/kernel.ll",   dir);
   snprintf(p_elf, sizeof p_elf, "%s/kernel.elf",  dir);
   snprintf(p_bin, sizeof p_bin, "%s/kernel.vxbin", dir);

   bool ok = write_text(p_ll, llvm_ir);
   char *cmd = NULL;

   /* compile + link the IR into a Vortex KMU kernel ELF */
   if (ok) {
      /* Device flags mirror the canonical Vortex kernel toolchain
       * invocation in tests/regression/common.mk: the llvm-vortex
       * clang with +xvortex (the Vortex ISA extension) and +zicond,
       * --sysroot / --gcc-toolchain pointing at the riscv32 GNU
       * toolchain, and -disable-loop-idiom-all. The Vortex
       * branch-divergence pass is part of the +xvortex backend and is
       * left at its default (enabled): it is what lowers divergent
       * SIMT control flow into correct masked execution, so we must
       * never pass -mllvm -vortex-branch-divergence=0. */
      if (asprintf(&cmd,
            "%s/llvm-vortex/bin/clang --target=riscv32-unknown-elf "
            "--sysroot=%s/riscv32-gnu-toolchain/riscv32-unknown-elf "
            "--gcc-toolchain=%s/riscv32-gnu-toolchain "
            "-march=rv32imaf -mabi=ilp32f "
            "-Xclang -target-feature -Xclang +xvortex "
            "-Xclang -target-feature -Xclang +zicond "
            "-mllvm -disable-loop-idiom-all "
            "-Wno-unused-command-line-argument -Wno-override-module "
            "-O3 -mcmodel=medany -nostartfiles -nostdlib "
            "-fdata-sections -ffunction-sections -fuse-ld=lld "
            "%s "
            "-Wl,-Bstatic,--gc-sections,-T,%s/sw/kernel/scripts/link32.ld,"
            "--defsym=STARTUP_ADDR=0x80000000 "
            "%s/sw/kernel/libvortex2.a "
            "-L%s/libc32/lib -lm -lc "
            "%s/libcrt32/lib/baremetal/libclang_rt.builtins-riscv32.a "
            "-o %s 2>%s/clang.log",
            td, td, td, p_ll, vh, bd, td, td, p_elf, dir) < 0) {
         cmd = NULL;
         ok = false;
      }
      if (ok && system(cmd) != 0) {
         mesa_logw("vortexpipe: device clang/link failed (kept %s/clang.log)",
                   dir);
         ok = false;
      }
      free(cmd);
      cmd = NULL;
   }

   /* package the ELF into a .vxbin */
   if (ok) {
      if (asprintf(&cmd,
            "OBJCOPY=%s/llvm-vortex/bin/llvm-objcopy python3 "
            "%s/sw/kernel/scripts/vxbin.py %s %s 2>%s/vxbin.log",
            td, vh, p_elf, p_bin, dir) < 0) {
         cmd = NULL;
         ok = false;
      }
      if (ok && system(cmd) != 0) {
         mesa_logw("vortexpipe: vxbin.py failed (kept %s/vxbin.log)", dir);
         ok = false;
      }
      free(cmd);
      cmd = NULL;
   }

   if (ok) {
      *out_blob = read_blob(p_bin, out_size);
      ok = (*out_blob != NULL);
   }

   /* clean up on success; keep the temp dir for debugging on failure */
   if (ok) {
      if (asprintf(&cmd, "rm -rf %s", dir) >= 0) {
         system(cmd);
         free(cmd);
      }
   }
   return ok;
}

void
vp_free_blob(void *blob)
{
   free(blob);
}
