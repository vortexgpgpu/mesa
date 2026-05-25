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
#include <stdio.h>
#include <string.h>

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

   if (vps->dev) {
      /* Dump GPU perf counters at device teardown, matching pocl_vortex's
       * pocl-vortex.c:373 pattern. Output is gated by $VORTEX_PROFILING
       * inside the runtime; a no-op when unset. */
      vx_device_dump_perf(vps->dev, stdout);
      vx_device_release(vps->dev);
   }
   vp_reg_del(screen);
   FREE(vps);
   lp_destroy(screen);
}

/* Vortex-distinct device name so Vulkan apps (and the test harness)
 * can tell vortexpipe from plain llvmpipe via VkPhysicalDeviceProperties
 * .deviceName. When the Vortex device opened successfully, prefix
 * "vortexpipe" — otherwise fall through to llvmpipe's name so callers
 * see honest "I'm just CPU" without the marketing prefix. */
static const char *
vp_screen_get_name(struct pipe_screen *screen)
{
   struct vp_screen *vps = vp_reg_get(screen);
   if (!vps || !vps->dev)
      return vps->lp_screen_get_name(screen);   /* no Vortex — be honest */
   /* Buffer is screen-resident so the returned pointer remains valid as
    * long as Vulkan holds the screen. */
   if (vps->name_str[0] == '\0') {
      const char *base = vps->lp_screen_get_name(screen);
      snprintf(vps->name_str, sizeof(vps->name_str),
               "vortexpipe (Vortex on %s)", base ? base : "llvmpipe");
   }
   return vps->name_str;
}

struct pipe_screen *
vortexpipe_create_screen(struct sw_winsys *winsys)
{
   struct pipe_screen *screen = llvmpipe_create_screen(winsys);
   if (!screen)
      return NULL;

   struct vp_screen *vps = CALLOC_STRUCT(vp_screen);
   if (!vps) {
      mesa_loge("vortexpipe: out of memory; running as plain llvmpipe");
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
   vps->lp_context_create  = screen->context_create;
   vps->lp_screen_destroy  = screen->destroy;
   vps->lp_screen_get_name = screen->get_name;
   vp_reg_put(screen, vps);

   screen->context_create = vp_context_create;
   screen->destroy        = vp_screen_destroy;
   screen->get_name       = vp_screen_get_name;

   vp_dbg("vortexpipe: screen ready (llvmpipe base, Vortex hooks armed)");
   return screen;
}
