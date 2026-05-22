# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT

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
  c_args : [c_msvc_compat_args],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir, idep_mesautil],
)

driver_vortexpipe = declare_dependency(
  compile_args : '-DGALLIUM_VORTEXPIPE',
  link_with : libvortexpipe,
  dependencies : [driver_llvmpipe],
)
