/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vortexpipe -- Vortex GPU Gallium driver.
 *
 * Phase 1 skeleton + Phase 2 device-integration breadcrumb.
 * vortexpipe_create_screen() still delegates the pipe_screen to
 * llvmpipe; the Phase 2 increments add the Vortex compute path
 * (create_compute_state / launch_grid) on top. The vx_device_*
 * calls below confirm the driver links and drives the Vortex
 * runtime -- see docs/proposals/vulkan_support_proposal.md in the
 * Vortex source tree.
 */

#include <stdint.h>

#include "vp_public.h"
#include "llvmpipe/lp_public.h"
#include "util/log.h"

#include "vortex2.h"

struct pipe_screen *
vortexpipe_create_screen(struct sw_winsys *winsys)
{
   /* Phase 2 integration breadcrumb: confirm vortexpipe can link
    * and drive the Vortex runtime (libvortex-simx.so). */
   uint32_t ndev = 0;
   vx_result_t r = vx_device_count(&ndev);
   if (r == VX_SUCCESS && ndev > 0) {
      vx_device_h dev = NULL;
      if (vx_device_open(0, &dev) == VX_SUCCESS) {
         mesa_logi("vortexpipe: opened Vortex device 0 of %u "
                   "(runtime link OK)", ndev);
         vx_device_release(dev);
      } else {
         mesa_logw("vortexpipe: vx_device_open(0) failed");
      }
   } else {
      mesa_logw("vortexpipe: vx_device_count -> %u (%s)",
                ndev, vx_result_string(r));
   }

   /* Phase 1/2: the pipe_screen still delegates to llvmpipe. The
    * Vortex compute path is wired in the subsequent Phase 2
    * increments (create_compute_state / launch_grid). */
   return llvmpipe_create_screen(winsys);
}
