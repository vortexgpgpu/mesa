// gfx_v2 on-device setup -> binning front end — vortexpipe device kernel.
//
// Single translation unit that materialises the two front-end kernel entries
// (setup_k stages 0-2, binning_k stages 3-8) for vortexpipe to launch in place
// of the host graphics::Binning(). The stage logic lives in pipe_frontend.h;
// this file is the compile unit the Vortex kernel toolchain turns into a
// .vxbin (see Makefile). Device-only (pulls vx_spawn2.h via pipe_frontend.h).

#include "pipe_frontend.h"
