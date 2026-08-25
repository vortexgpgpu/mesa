// gfx_v2 on-device setup -> binning front end — vortexpipe device kernel.
//
// Single translation unit that materialises the front-end kernel entries
// (setup_k stages 0-2, binning_k stages 3-8) for vortexpipe to launch in place
// of the host graphics::Binning(), plus the pass-end multisample resolve. The
// stage logic lives in gfx_frontend_k.h and gfx_resolve_k.h; this file is the
// compile unit the Vortex kernel toolchain turns into a .vxbin (see Makefile).
// Device-only (pulls vx_spawn2.h via those headers).
//
// The resolve rides in this module rather than one of its own because
// vortexpipe already keeps this one resident across the pass, and a second
// module would have to be linked at a third base address to co-reside with it
// and the per-draw fragment shader.

#include "gfx_frontend_k.h"
#include "gfx_resolve_k.h"
