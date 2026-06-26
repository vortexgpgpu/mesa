/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_compile -- Phase 2 #4b: LLVM IR -> Vortex .vxbin.
 *
 * Drives the existing Vortex device toolchain on the kernel module
 * vp_nir_to_llvm emits: llvm-vortex's clang compiles + links the IR
 * (riscv32 or riscv64, +xvortex) against the KMU device kernel
 * library (libvortex2.a) and the baremetal libc/compiler-rt, then
 * vxbin.py packages the ELF into a .vxbin. First cut: fork/exec the
 * tools via system(); moving in-process is a later optimization
 * (§3 n.4).
 *
 * Paths come from VORTEX_HOME (the Vortex tree) and TOOLDIR (the
 * $HOME/tools install root) -- the meson-baked defaults below,
 * overridable by the VORTEX_HOME / VORTEX_TOOLDIR env vars. The
 * Vortex out-of-tree build dir (where libvortex2.a is built) is
 * VORTEX_HOME/build by convention; VORTEX_BUILD overrides it for
 * trees configured under a differently named build dir (e.g. the
 * CI build32/build64).
 *
 * The target XLEN is picked from the MESA_VORTEX_XLEN env var
 * (default "32"; "64" selects riscv64). Mesa-namespaced so it does
 * not collide with anything the linked Vortex runtime reads; mirrors
 * pocl_vortex/POCL_VORTEX_XLEN in shape (one ICD, env-var dispatch).
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

/* Target XLEN from $MESA_VORTEX_XLEN (default "32"). "64" selects
 * riscv64, anything else selects riscv32. Mesa-namespaced so it
 * doesn't collide with the linked Vortex runtime. */
bool
vp_xlen_is_64(void)
{
   const char *s = getenv("MESA_VORTEX_XLEN");
   return s && strcmp(s, "64") == 0;
}

bool
vp_compile_vxbin(const char *llvm_ir, unsigned long long startup_addr,
                 bool link_gfx_sw, void **out_blob, size_t *out_size)
{
   *out_blob = NULL;
   *out_size = 0;

   const char *vh = getenv("VORTEX_HOME");
   if (!vh || !*vh) vh = VP_VORTEX_HOME;
   const char *td = getenv("VORTEX_TOOLDIR");
   if (!td || !*td) td = VP_TOOLDIR;
   if (!*vh || !*td) {
      mesa_loge("vortexpipe: VORTEX_HOME / TOOLDIR not configured");
      return false;
   }
   const bool is64 = vp_xlen_is_64();
   const char *target  = is64 ? "riscv64-unknown-elf" : "riscv32-unknown-elf";
   const char *gnu_dir = is64 ? "riscv64-gnu-toolchain" : "riscv32-gnu-toolchain";
   const char *march   = is64 ? "rv64imafd" : "rv32imaf";
   const char *mabi    = is64 ? "lp64d"     : "ilp32f";
   const char *linker  = is64 ? "link64.ld" : "link32.ld";
   const char *libc    = is64 ? "libc64"    : "libc32";
   const char *libcrt  = is64 ? "libcrt64"  : "libcrt32";
   const char *crt_a   = is64 ? "libclang_rt.builtins-riscv64.a"
                              : "libclang_rt.builtins-riscv32.a";

   /* Build dir holding libvortex2.a -- VORTEX_HOME/build by default. */
   char bd_buf[512];
   const char *bd = getenv("VORTEX_BUILD");
   if (!bd || !*bd) {
      snprintf(bd_buf, sizeof bd_buf, "%s/build", vh);
      bd = bd_buf;
   }

   char dir[] = "/tmp/vortexpipe.XXXXXX";
   if (!mkdtemp(dir)) {
      mesa_loge("vortexpipe: mkdtemp failed");
      return false;
   }

   char p_ll[256], p_elf[256], p_bin[256];
   snprintf(p_ll,  sizeof p_ll,  "%s/kernel.ll",   dir);
   snprintf(p_elf, sizeof p_elf, "%s/kernel.elf",  dir);
   snprintf(p_bin, sizeof p_bin, "%s/kernel.vxbin", dir);

   bool ok = write_text(p_ll, llvm_ir);
   char *cmd = NULL;

   /* §5: when a unit is routed to software, co-compile the gfx_sw_abi C ABI into
    * the kernel (compiled from the SSOT C++ headers, so the divergent OM merge is
    * processed by the Vortex divergence pass over the whole kernel; --gc-sections
    * drops unused entry points). VX_types.h is the generated copy under $bd/sw. */
   char gfx_seg[1024] = "";
   if (ok && link_gfx_sw) {
      /* -DNDEBUG: the shared tex/OM headers (and cocogfx) use assert(); the
       * baremetal link has no __assert_func, so compile assertions out. */
      snprintf(gfx_seg, sizeof gfx_seg,
         "-std=c++17 -DNDEBUG -D__VORTEX__ -DGFX_SW_DIVERGENCE_OK "
         "-mllvm -vortex-divergence-max-bbs=512 "
         "-I%s/sw/gfx -I%s/sw/common -I%s/third_party -I%s/sw %s/sw/gfx/gfx_sw_abi.cpp",
         vh, vh, vh, bd, vh);
   }

   /* compile + link the IR into a Vortex KMU kernel ELF */
   if (ok) {
      /* Device flags mirror the canonical Vortex kernel toolchain
       * invocation in tests/regression/common.mk: the llvm-vortex
       * clang with +xvortex (the Vortex ISA extension) and +zicond,
       * --sysroot / --gcc-toolchain pointing at the GNU toolchain
       * matching the target XLEN, and -disable-loop-idiom-all. The
       * Vortex branch-divergence pass is part of the +xvortex backend
       * and is left at its default (enabled): it is what lowers
       * divergent SIMT control flow into correct masked execution, so
       * we must never pass -mllvm -vortex-branch-divergence=0. */
      if (asprintf(&cmd,
            "%s/llvm-vortex/bin/clang --target=%s "
            "--sysroot=%s/%s/%s "
            "--gcc-toolchain=%s/%s "
            "-march=%s -mabi=%s "
            "-Xclang -target-feature -Xclang +xvortex "
            "-Xclang -target-feature -Xclang +zicond "
            "-mllvm -disable-loop-idiom-all "
            "-Wno-unused-command-line-argument -Wno-override-module "
            "-O3 -mcmodel=medany -nostartfiles -nostdlib "
            "-fdata-sections -ffunction-sections -fuse-ld=lld "
            "%s %s "
            "-Wl,-Bstatic,--gc-sections,-T,%s/sw/kernel/scripts/%s,"
            "--defsym=STARTUP_ADDR=0x%llx "
            "%s/sw/kernel/libvortex2.a "
            "-L%s/%s/lib -lm -lc "
            "%s/%s/lib/baremetal/%s "
            "-o %s 2>%s/clang.log",
            td, target,
            td, gnu_dir, target,
            td, gnu_dir,
            march, mabi,
            p_ll, gfx_seg,
            vh, linker,
            startup_addr,
            bd,
            td, libc,
            td, libcrt, crt_a,
            p_elf, dir) < 0) {
         cmd = NULL;
         ok = false;
      }
      if (ok && system(cmd) != 0) {
         mesa_loge("vortexpipe: device clang/link failed (kept %s/clang.log)",
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
         mesa_loge("vortexpipe: vxbin.py failed (kept %s/vxbin.log)", dir);
         ok = false;
      }
      free(cmd);
      cmd = NULL;
   }

   if (ok) {
      *out_blob = read_blob(p_bin, out_size);
      ok = (*out_blob != NULL);
   }

   /* clean up on success; keep the temp dir for debugging on failure or
    * when VORTEXPIPE_KEEP_TMP is set (lets a developer inspect kernel
    * ELF / .vxbin contents, symbol tables, etc.). */
   if (ok && !getenv("VORTEXPIPE_KEEP_TMP")) {
      if (asprintf(&cmd, "rm -rf %s", dir) >= 0) {
         system(cmd);
         free(cmd);
      }
   } else if (ok) {
      fprintf(stderr, "vortexpipe: VORTEXPIPE_KEEP_TMP set; kept %s\n", dir);
   }
   return ok;
}

void
vp_free_blob(void *blob)
{
   free(blob);
}
