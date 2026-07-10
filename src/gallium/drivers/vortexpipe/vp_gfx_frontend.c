/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_gfx_frontend -- the embedded gfx_v2 on-device front-end kernel blob.
 */

#include "vp_gfx_frontend.h"

/* Generated at build time by xxd.py (-b) from gfx_frontend<xlen>.vxbin:
 *   static const char gfx_frontend_vxbin<xlen>[] = { 0x.., ... };
 * Binary mode emits no trailing nul, so sizeof() is the exact byte count. */
#include "gfx_frontend_vxbin32.h"
#include "gfx_frontend_vxbin64.h"

const void *
vp_gfx_frontend_vxbin(bool is64, size_t *size_out)
{
   if (is64) {
      *size_out = sizeof(gfx_frontend_vxbin64);
      return gfx_frontend_vxbin64;
   }
   *size_out = sizeof(gfx_frontend_vxbin32);
   return gfx_frontend_vxbin32;
}
