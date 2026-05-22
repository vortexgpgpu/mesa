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
  'vp_screen.c',
)

libvortexpipe = static_library(
  'vortexpipe',
  files_vortexpipe,
  # include_directories('..') exposes drivers/ so vp_screen.c can
  # reach "llvmpipe/lp_public.h".
  include_directories : [
    inc_gallium_aux, inc_gallium, inc_include, inc_src,
    include_directories('..'),
  ],
  c_args : [
    c_msvc_compat_args,
    '-I' + join_paths(vortex_home, 'sw/runtime/include'),
    # generated VX_config.h (produced by the Vortex out-of-tree build)
    '-I' + join_paths(vortex_home, 'build/sw'),
  ],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir, idep_mesautil],
)

driver_vortexpipe = declare_dependency(
  compile_args : '-DGALLIUM_VORTEXPIPE',
  link_with : libvortexpipe,
  dependencies : [driver_llvmpipe, dep_vortex_rt],
)
