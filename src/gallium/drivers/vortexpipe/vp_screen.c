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

/* NIR finalize hook. lavapipe leaves Vulkan ray queries intact when we
 * advertise driver_ray_queries (RTU present); lower them here, before
 * llvmpipe's own finalize_nir runs, so both the vortexpipe .vxbin path
 * and the llvmpipe base see the lowered (vortex_rt_*) form. */
static char *
vp_finalize_nir(struct pipe_screen *screen, struct nir_shader *nir)
{
   struct vp_screen *vps = vp_reg_get(screen);
   if (vps && vps->has_rtu)
      vp_nir_lower_ray_tracing_to_rtu(nir);
   return vps->lp_finalize_nir(screen, nir);
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

      /* Cache the per-device caps the launch + draw paths need. Query
       * up-front so every vp_launch_grid / vp_raster_draw can refuse
       * fast (without re-querying) when the workload exceeds them. */
      uint64_t nt = 0, nw = 0, isa = 0;
      if (vx_device_query(vps->dev, VX_CAPS_NUM_THREADS, &nt) == VX_SUCCESS &&
          vx_device_query(vps->dev, VX_CAPS_NUM_WARPS,   &nw) == VX_SUCCESS &&
          vx_device_query(vps->dev, VX_CAPS_ISA_FLAGS,   &isa) == VX_SUCCESS) {
         vps->hw_num_threads    = (uint32_t)nt;
         vps->hw_num_warps      = (uint32_t)nw;
         vps->hw_max_block_size = (uint32_t)(nt * nw);
         vps->hw_isa_flags      = isa;
         vps->has_tex    = (isa & VX_ISA_EXT_TEX)    != 0;
         vps->has_raster = (isa & VX_ISA_EXT_RASTER) != 0;
         vps->has_om     = (isa & VX_ISA_EXT_OM)     != 0;
         vps->has_rtu    = (isa & VX_ISA_EXT_RTU)    != 0;
         vp_dbg("vortexpipe: caps: threads=%u, warps=%u, max_block=%u, "
                "tex=%d, raster=%d, om=%d, rtu=%d",
                vps->hw_num_threads, vps->hw_num_warps, vps->hw_max_block_size,
                vps->has_tex, vps->has_raster, vps->has_om, vps->has_rtu);
      } else {
         mesa_logw("vortexpipe: failed to query device caps; treating as bare");
         /* Leave caps fields zero — every cap-gated path then refuses. */
      }
   } else {
      mesa_logw("vortexpipe: no Vortex device; compute falls back to llvmpipe");
      vps->dev = NULL;
   }

   /* Patch the entry points vortexpipe intercepts; record originals. */
   vps->lp_context_create  = screen->context_create;
   vps->lp_screen_destroy  = screen->destroy;
   vps->lp_screen_get_name = screen->get_name;
   vps->lp_finalize_nir    = screen->finalize_nir;
   vp_reg_put(screen, vps);

   screen->context_create = vp_context_create;
   screen->destroy        = vp_screen_destroy;
   screen->get_name       = vp_screen_get_name;
   screen->finalize_nir   = vp_finalize_nir;

   /* Clamp llvmpipe's compute caps to the Vortex hardware cap so well-
    * behaved Vulkan apps that read maxComputeWorkGroupSize /
    * maxComputeWorkGroupInvocations pick a workgroup size that fits one
    * Vortex CTA. Without this clamp, the inherited llvmpipe value (1024)
    * lets apps declare local_size_x=64 and beyond, which the launch_grid
    * cap check then refuses — fine, but Vulkan-clean apps should never
    * see the refusal in the first place. */
   if (vps->dev && vps->hw_max_block_size != 0) {
      struct pipe_compute_caps *caps =
         (struct pipe_compute_caps *)&screen->compute_caps;
      const uint32_t hw_max = vps->hw_max_block_size;
      if (caps->max_block_size[0] > hw_max) caps->max_block_size[0] = hw_max;
      if (caps->max_block_size[1] > hw_max) caps->max_block_size[1] = hw_max;
      if (caps->max_block_size[2] > hw_max) caps->max_block_size[2] = hw_max;
      if (caps->max_threads_per_block > hw_max)
         caps->max_threads_per_block = hw_max;

      /* A subgroup is a Vortex warp. The inherited llvmpipe number is its own SIMD
       * width (lp_native_vector_width / 32) and has never described this hardware;
       * a shader that asks for the subgroup size, or whose quad ops are lowered
       * against it, would be reasoning about the wrong machine.
       *
       * subgroup_sizes is a BITMASK of supported sizes (bit b => size 1<<b). A warp
       * width is a power of two, so assigning it sets exactly the one bit that says
       * "the only subgroup size is NT". */
      caps->subgroup_sizes = vps->hw_num_threads;
      caps->max_subgroups  = vps->hw_num_warps;
   }

   /* Tell the lavapipe frontend to leave Vulkan ray queries intact for
    * vortexpipe to lower (vp_nir_lower_ray_tracing_to_rtu) instead of
    * expanding them to a software BVH walk. */
   ((struct pipe_caps *)&screen->caps)->driver_ray_queries = vps->has_rtu;

   vp_dbg("vortexpipe: screen ready (llvmpipe base, Vortex hooks armed)");
   return screen;
}
