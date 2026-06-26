/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_compile -- Phase 2 #4b: drive LLVM IR -> Vortex .vxbin.
 */

#ifndef VP_COMPILE_H
#define VP_COMPILE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile LLVM-IR text (a Vortex KMU kernel module, as produced by
 * vp_nir_to_llvm) into a Vortex .vxbin image linked at `startup_addr`.
 * On success returns true and stores a malloc'd blob + size in
 * *out_blob / *out_size; release it with vp_free_blob().
 *
 * `startup_addr` is the device VMA the kernel image is linked at. Stages
 * that co-reside in one device-orchestrated draw must use distinct bases so
 * their images don't overlap: VP_STARTUP_FS (0x80000000) for the fragment
 * shader / compute, VP_STARTUP_VS (0x80400000) for the vertex shader (the
 * embedded front end already occupies 0x80200000). */
#define VP_STARTUP_FS  0x80000000ull
#define VP_STARTUP_VS  0x80400000ull
/* link_gfx_sw: co-compile the gfx_v2 §5 software-fallback C ABI
 * ($VORTEX_HOME/sw/gfx/gfx_sw_abi.cpp) into the kernel so the FS's
 * gfx_tex_sample_sw / gfx_om_fragment_sw calls resolve and inline (the divergent
 * OM merge needs the Vortex divergence pass over the whole kernel). Set when any
 * unit is routed to software; ignored otherwise (--gc-sections drops it). */
bool vp_compile_vxbin(const char *llvm_ir, unsigned long long startup_addr,
                      bool link_gfx_sw, void **out_blob, size_t *out_size);

void vp_free_blob(void *blob);

/* True when $MESA_VORTEX_XLEN selects rv64 (otherwise rv32). Single
 * source of truth for the target XLEN -- both vp_compile_vxbin and
 * the LLVM codegen in vp_nir_to_llvm consult this. Mesa-namespaced
 * env var so it doesn't collide with the linked Vortex runtime. */
bool vp_xlen_is_64(void);

#ifdef __cplusplus
}
#endif

#endif /* VP_COMPILE_H */
