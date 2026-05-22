/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vortexpipe -- Vortex GPU Gallium driver.
 *
 * vortexpipe_create_screen() builds an llvmpipe pipe_screen, opens
 * the Vortex device, and patches the screen's context_create /
 * destroy entry points so vortexpipe can intercept context
 * creation (and, through it, the compute hooks -- see vp_context.c).
 * Everything not overridden continues to run on llvmpipe; the
 * Vortex compute path is filled in by the later Phase 2 increments.
 * See docs/proposals/vulkan_support_proposal.md in the Vortex tree.
 */

#include <stdint.h>

#include "vp_public.h"
#include "vp_private.h"
#include "llvmpipe/lp_public.h"
#include "util/u_memory.h"
#include "util/log.h"

static void
vp_screen_destroy(struct pipe_screen *screen)
{
   struct vp_screen *vps = vp_reg_get(screen);
   void (*lp_destroy)(struct pipe_screen *) = vps->lp_screen_destroy;

   if (vps->dev)
      vx_device_release(vps->dev);
   vp_reg_del(screen);
   FREE(vps);
   lp_destroy(screen);
}

struct pipe_screen *
vortexpipe_create_screen(struct sw_winsys *winsys)
{
   struct pipe_screen *screen = llvmpipe_create_screen(winsys);
   if (!screen)
      return NULL;

   struct vp_screen *vps = CALLOC_STRUCT(vp_screen);
   if (!vps) {
      mesa_logw("vortexpipe: out of memory; running as plain llvmpipe");
      return screen;
   }

   /* Open the Vortex device once, held for the screen's lifetime. */
   uint32_t ndev = 0;
   if (vx_device_count(&ndev) == VX_SUCCESS && ndev > 0 &&
       vx_device_open(0, &vps->dev) == VX_SUCCESS) {
      vp_dbg("vortexpipe: opened Vortex device 0 of %u", ndev);
   } else {
      mesa_logw("vortexpipe: no Vortex device; compute falls back to llvmpipe");
      vps->dev = NULL;
   }

   /* Patch the entry points vortexpipe intercepts; record originals. */
   vps->lp_context_create = screen->context_create;
   vps->lp_screen_destroy = screen->destroy;
   vp_reg_put(screen, vps);

   screen->context_create = vp_context_create;
   screen->destroy        = vp_screen_destroy;

   vp_dbg("vortexpipe: screen ready (llvmpipe base, Vortex hooks armed)");
   return screen;
}
