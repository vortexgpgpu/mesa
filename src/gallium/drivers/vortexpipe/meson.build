# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT

# Vortex GPGPU integration — consumed exclusively through the vortex
# install tree ($VORTEX_PATH) via pkg-config, the same way the CUDA /
# ROCm / oneAPI SDKs are integrated. The source tree ($VORTEX_HOME)
# and build tree ($VORTEX_BUILD_DIR) are invisible here.
#
# Discovery: vortex's `make install` ships vortex-runtime.pc to
# $VORTEX_PATH/lib/pkgconfig/. Set PKG_CONFIG_PATH there (or pass
# --pkg-config-path to meson) and `dependency('vortex-runtime')` finds:
#   -I$VORTEX_PATH/runtime/include    (graphics.h, vortex2.h, ...)
#   -I$VORTEX_PATH/kernel/include     (vx_graphics.h on-wire ABI)
#   -L$VORTEX_PATH/runtime/lib -lvortex
#
# libvortex.so is the backend-agnostic vortex2.h API + graphics::Binning
# layer; at run time it dlopens a libvortex-<backend>.so sibling
# (libvortex-simx.so / libvortex-xrt.so / ...) which in turn finds
# libsimx.so / libxrtsim.so etc. via an $ORIGIN rpath. So only
# $VORTEX_PATH/runtime/lib needs to be on LD_LIBRARY_PATH at run time.
dep_vortex = dependency('vortex-runtime')

# ── gfx_v2 on-device front-end kernel: build from SDK source + embed ──
# vortexpipe builds its device front end (setup_k + binning_k) from the
# Vortex SDK source tree the way a GPU driver builds its firmware, then
# embeds the resulting .vxbin so vp_launch loads it from memory with no
# runtime toolchain dependency. This is the one place the driver needs the
# SDK *source* tree (sw/gfx, sw/common): the front-end kernel sources are
# not shipped in the install prefix the rest of the driver consumes.
vp_vortex_home = get_option('vortex-home')
if vp_vortex_home == ''
  error('vortexpipe requires -Dvortex-home=<Vortex SDK source tree> to ' +
        'build the gfx_v2 front-end kernel (its sources are not in the ' +
        'install prefix)')
endif

# make builds the .vxbin out-of-tree into the custom_target's @OUTDIR@;
# xxd.py then embeds it as a C byte array (-b: binary, no trailing nul).
gfx_frontend_vxbin = custom_target(
  'gfx_frontend.vxbin',
  output  : 'gfx_frontend.vxbin',
  command : [
    find_program('make'),
    '-C', meson.current_source_dir() / 'kernels' / 'gfx_frontend',
    # absolute: make runs with -C (its CWD differs from ninja's), so a
    # build-root-relative @OUTDIR@ would not resolve. current_build_dir()
    # is this target's output directory (where meson expects the .vxbin).
    'OUTDIR=' + meson.current_build_dir(),
    'VORTEX_HOME=' + vp_vortex_home,
    'VORTEX_TOOLDIR=' + get_option('vortex-tooldir'),
    'all',
  ],
  # Rebuild the blob when the SDK kernel sources (sw/gfx, sw/common) or the
  # build recipe change — meson does not see make's own prerequisites.
  depend_files : files(
      'kernels/gfx_frontend/gfx_frontend_kernel.cpp',
      'kernels/gfx_frontend/Makefile',
    ) + [
      vp_vortex_home / 'sw' / 'gfx' / 'pipe_frontend.h',
      vp_vortex_home / 'sw' / 'gfx' / 'pipe_abi.h',
      vp_vortex_home / 'sw' / 'gfx' / 'setup_math.h',
      vp_vortex_home / 'sw' / 'gfx' / 'setup_types.h',
      vp_vortex_home / 'sw' / 'common' / 'gfx_frontend_abi.h',
    ],
  console : true,
)

gfx_frontend_vxbin_h = custom_target(
  'gfx_frontend_vxbin.h',
  input   : [files_xxd, gfx_frontend_vxbin],
  output  : 'gfx_frontend_vxbin.h',
  command : [prog_python, '@INPUT@', '@OUTPUT@', '-b', '-n', 'gfx_frontend_vxbin'],
)

files_vortexpipe = files(
  'vp_public.h',
  'vp_private.h',
  'vp_screen.c',
  'vp_context.c',
  'vp_nir_to_llvm.h',
  'vp_nir_to_llvm.c',
  'vp_nir_lower_ray_tracing_to_rtu.c',
  'vp_compile.h',
  'vp_compile.c',
  'vp_launch.h',
  'vp_launch.c',
  'vp_raster.h',
  'vp_raster.cpp',
  'vp_gfx_frontend.h',
  'vp_gfx_frontend.c',
)

# Toolchain paths for vp_compile.c (.vxbin generation). vortex-tooldir
# points at the LLVM-vortex / kernel-build toolchain prefix; vortex-path
# is the install root (used to find the device-side kernel/include).
#
# -I<vortex-home>/sw/common gives vp_raster.cpp the gfx_v2 front-end ABI
# (gfx_frontend_abi.h: pipe_arg_t, setup_vertex_t, PIPE_STAGE_*) it shares
# with the embedded device kernel; that header is not in the install prefix.
vp_def_args = [
  '-DVP_VORTEX_PATH="' + get_option('vortex-path') + '"',
  '-DVP_TOOLDIR="'     + get_option('vortex-tooldir') + '"',
  '-I' + vp_vortex_home / 'sw' / 'common',
]

libvortexpipe = static_library(
  'vortexpipe',
  [files_vortexpipe, gfx_frontend_vxbin_h],
  # include_directories('..') exposes drivers/ so vp_screen.c can
  # reach "llvmpipe/lp_public.h".
  include_directories : [
    inc_gallium_aux, inc_gallium, inc_include, inc_src,
    include_directories('..'),
  ],
  c_args   : [c_msvc_compat_args]   + vp_def_args,
  cpp_args : [cpp_msvc_compat_args] + vp_def_args,
  gnu_symbol_visibility : 'hidden',
  # dep_vortex: graphics.h, vortex2.h, libvortex.so (via pkg-config)
  # dep_llvm:   vp_nir_to_llvm.c emits IR via the LLVM C API
  dependencies : [dep_vortex, idep_nir, idep_mesautil, dep_llvm],
)

driver_vortexpipe = declare_dependency(
  compile_args : '-DGALLIUM_VORTEXPIPE',
  link_with : libvortexpipe,
  dependencies : [driver_llvmpipe, dep_vortex],
)
