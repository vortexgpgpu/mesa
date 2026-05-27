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

files_vortexpipe = files(
  'vp_public.h',
  'vp_private.h',
  'vp_screen.c',
  'vp_context.c',
  'vp_nir_to_llvm.h',
  'vp_nir_to_llvm.c',
  'vp_compile.h',
  'vp_compile.c',
  'vp_launch.h',
  'vp_launch.c',
  'vp_raster.h',
  'vp_raster.cpp',
)

# Toolchain paths for vp_compile.c (.vxbin generation). vortex-tooldir
# points at the LLVM-vortex / kernel-build toolchain prefix; vortex-path
# is the install root (used to find the device-side kernel/include).
vp_def_args = [
  '-DVP_VORTEX_PATH="' + get_option('vortex-path') + '"',
  '-DVP_TOOLDIR="'     + get_option('vortex-tooldir') + '"',
]

libvortexpipe = static_library(
  'vortexpipe',
  files_vortexpipe,
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
