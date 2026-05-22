# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT

vortex_home = get_option('vortex-runtime')
if vortex_home == ''
  error('vortexpipe requires -D vortex-runtime=<path-to-vortex-tree> ' +
        '(VORTEX_HOME); see meson.options.')
endif

# Vortex runtime: vortex2.h headers from the source tree, the
# libvortex.so API library from its out-of-tree build dir.
# libvortex.so is the backend-agnostic vortex2.h API layer; at run
# time it dlopens a libvortex-<backend>.so sibling (libvortex-simx.so)
# which in turn finds libsimx.so via an $ORIGIN rpath. So only
# build/sw/runtime needs to be on LD_LIBRARY_PATH at run time.
dep_vortex_rt = cc.find_library(
  'vortex',
  dirs : [join_paths(vortex_home, 'build/sw/runtime')],
)

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

# The Vortex `graphics` library (triangle setup + tile binning) is
# compiled straight into vortexpipe -- vp_raster.cpp calls
# graphics::Binning(), the same host-side setup the draw3d/raster
# regression tests use. cocogfx (third_party/) is header-only.
files_vortex_gfx = [
  join_paths(vortex_home, 'sw/common/graphics.cpp'),
  join_paths(vortex_home, 'sw/common/gfxutil.cpp'),
]

# Header search paths shared by the C and C++ sources: vortex2.h, the
# Vortex graphics headers, cocogfx, and the generated VX_config.h.
vp_inc_args = [
  '-I' + join_paths(vortex_home, 'sw/runtime/include'),
  '-I' + join_paths(vortex_home, 'sw/common'),
  '-I' + join_paths(vortex_home, 'third_party'),
  '-I' + join_paths(vortex_home, 'build/sw'),
]

# baked-in toolchain paths for vp_compile.c (.vxbin generation)
vp_def_args = [
  '-DVP_VORTEX_HOME="' + vortex_home + '"',
  '-DVP_TOOLDIR="' + get_option('vortex-tooldir') + '"',
]

libvortexpipe = static_library(
  'vortexpipe',
  files_vortexpipe + files_vortex_gfx,
  # include_directories('..') exposes drivers/ so vp_screen.c can
  # reach "llvmpipe/lp_public.h".
  include_directories : [
    inc_gallium_aux, inc_gallium, inc_include, inc_src,
    include_directories('..'),
  ],
  c_args   : [c_msvc_compat_args]   + vp_inc_args + vp_def_args,
  cpp_args : [cpp_msvc_compat_args] + vp_inc_args + vp_def_args,
  gnu_symbol_visibility : 'hidden',
  # the vendored Vortex graphics sources are not -Werror clean under
  # Mesa's warning flags; keep them from failing the build.
  override_options : ['werror=false'],
  # dep_llvm: vp_nir_to_llvm.c emits IR via the LLVM C API.
  dependencies : [idep_nir, idep_mesautil, dep_llvm],
)

driver_vortexpipe = declare_dependency(
  compile_args : '-DGALLIUM_VORTEXPIPE',
  link_with : libvortexpipe,
  dependencies : [driver_llvmpipe, dep_vortex_rt],
)
