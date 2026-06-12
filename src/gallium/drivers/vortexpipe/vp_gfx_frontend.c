/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_gfx_frontend -- the embedded gfx_v2 on-device front-end kernel blob.
 */

#include "vp_gfx_frontend.h"

/* Generated at build time by xxd.py (-b) from gfx_frontend.vxbin:
 *   static const char gfx_frontend_vxbin[] = { 0x.., ... };
 * Binary mode emits no trailing nul, so sizeof() is the exact byte count. */
#include "gfx_frontend_vxbin.h"

const void *
vp_gfx_frontend_vxbin(size_t *size_out)
{
   *size_out = sizeof(gfx_frontend_vxbin);
   return gfx_frontend_vxbin;
}
