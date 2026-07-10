# gfx_v2 on-device front-end kernel (setup → binning)

The SIMT front end that replaces the host `graphics::Binning()` in vortexpipe's
draw path: on-device triangle clip + setup and parallel bin-sort binning,
producing RASTER's resident primbuf + tilebuf. Two kernel entries —
`setup_k` (stages 0–2) and `binning_k` (stages 3–8) — sequenced by the CP.

## Ownership: the driver builds, the Vortex SDK owns the source

The kernel sources are **single-sourced in the Vortex SDK**, not copied here —
the same sources the SimX graphics tests validate against:

| Source | Location (consumed via `$VORTEX_HOME`) |
|--------|----------------------------------------|
| `pipe_frontend.h` (setup_k / binning_k stage logic) | `sw/gfx/` |
| `setup_math.h` (clip + Q15.16 triangle setup) | `sw/gfx/` |
| `pipe_abi.h`, `setup_types.h` (bin-grid / RT macros) | `sw/gfx/` |
| `gfx_frontend_abi.h` (host/device front-end ABI) | `sw/common/` |
| on-wire RASTER ABI `vx_gfx_abi.h`, runtime `vx_spawn2.h`, `libvortex2.a` | SDK (hardware contract) |

This directory owns only the **build recipe** — the compile unit
(`gfx_frontend_kernel.cpp`, a single `#include "pipe_frontend.h"`) and the
`Makefile` that turns it into `gfx_frontend.vxbin`, the way a GPU driver owns
the build of its device kernels while the ISA/runtime headers come from the
platform SDK. Mirrors how `vp_compile.c` consumes the SDK for NIR shaders.

## Build

```
make            # -> gfx_frontend32.vxbin (XLEN=32; XLEN=64 -> gfx_frontend64.vxbin)
```

Honors the same env as the driver: `VORTEX_HOME` (SDK source: `sw/gfx`,
`sw/common`, `sw/kernel`), `VORTEX_BUILD` (generated `VX_types.h`,
`libvortex2.a`), `VORTEX_TOOLDIR` (llvm-vortex + GNU toolchain). The clang
invocation mirrors `vp_compile.c` (`+xvortex`/`+zicond`, `link32.ld`,
`libvortex2.a`, `vxbin.py`).
