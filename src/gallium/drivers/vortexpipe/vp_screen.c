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
#include "nir.h"
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

   /* Vortex emits one flat kernel. The ray-tracing megashader (RTU or software
    * traversal) keeps each pipeline stage (raygen, closest-hit, miss, ...) as a
    * separate nir_function invoked by nir_call; inline them into the entrypoint.
    * The calls are direct and acyclic — trace-ray recursion is a resume-loop in
    * the entrypoint, not a nested call. Not gated on the RTU: the software
    * traversal megashader has the same structure and must also run flat. */
   if (exec_list_length(&nir->functions) > 1) {
      /* nir_inline_functions with driver_functions set only inlines functions
       * flagged should_inline (or small ones); the RT stage functions are
       * large, so force them. */
      nir_foreach_function(func, nir) {
         if (!func->is_entrypoint)
            func->should_inline = true;
      }
      NIR_PASS(_, nir, nir_lower_returns);
      NIR_PASS(_, nir, nir_inline_functions);
      NIR_PASS(_, nir, nir_opt_copy_prop_vars);
      nir_remove_non_entrypoints(nir);
   }
   /* The megashader keeps its shader-call stack, hit attributes and payload in
    * scratch; promote indexable scratch to SSA/vars where possible (after
    * inlining, so callee scratch is covered) to keep the shader on device. */
   NIR_PASS(_, nir, nir_lower_scratch_to_var);

   return vps->lp_finalize_nir(screen, nir);
}

/* What the device can carry, per usage. A format outside a set would be
 * reinterpreted rather than converted -- wrong pixels with no error, or worse --
 * so it is reported unsupported and Vulkan skips it.
 *
 * The four sets differ, and answering one list for every bind flag is what let
 * a 16-bit colour attachment through while refusing a float texture that works.
 * They are kept apart deliberately. */

/* Textures the sampler path carries: the fixed-function formats, everything the
 * host decodes to A8R8G8B8 on upload, the float formats that upload verbatim and
 * sample through the float texel path, and the depth formats.
 *
 * sRGB is absent on purpose: the host decode converts sRGB to linear at 8-bit
 * output precision, where Vulkan wants the conversion at higher precision ahead
 * of filtering. It belongs on the device's native SRGB8A8 decode, not here. */
static bool
vp_format_sampled(enum pipe_format format)
{
   switch (format) {
   /* 32-bit colour, decoded to the sampler's A8R8G8B8 source format */
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   /* packed and narrow colour, likewise host-decoded */
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   /* 8-bit integer carriers, packed from their bytes (vp_is_int8_rgba) */
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   /* float, uploaded verbatim and sampled as floats (vp_vx_tex_format) */
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   /* depth, including the combined formats whose depth aspect converts to D32F */
   case PIPE_FORMAT_Z16_UNORM:
   case PIPE_FORMAT_Z32_FLOAT:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
      return true;
   default:
      return false;
   }
}

/* Colour attachments. The output merger packs A8R8G8B8 and the pass-end transfer
 * moves four bytes per texel, so a narrower attachment is not merely converted
 * wrongly -- the transfer walks past the end of each row. 32-bit only. */
static bool
vp_format_render_target(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
      return true;
   default:
      return false;
   }
}

/* Depth/stencil attachments the output merger tests and writes. */
static bool
vp_format_depth_stencil(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
   case PIPE_FORMAT_Z16_UNORM:
   case PIPE_FORMAT_Z32_FLOAT:
      return true;
   default:
      return false;
   }
}

/* Storage images: the layouts the fragment translator emits image_load and
 * image_store for. Anything else fails the translation loudly rather than
 * reading wrong texels, but advertising it would still be a promise the driver
 * cannot keep. */
static bool
vp_format_shader_image(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return true;
   default:
      return false;
   }
}

/* Every requested bind must be one the device can honour for this format. */
static bool
vp_device_format_supported(enum pipe_format format, unsigned usage)
{
   if ((usage & PIPE_BIND_SAMPLER_VIEW) && !vp_format_sampled(format))
      return false;
   if ((usage & PIPE_BIND_RENDER_TARGET) && !vp_format_render_target(format))
      return false;
   if ((usage & PIPE_BIND_DEPTH_STENCIL) && !vp_format_depth_stencil(format))
      return false;
   if ((usage & PIPE_BIND_SHADER_IMAGE) && !vp_format_shader_image(format))
      return false;
   return true;   /* buffer / vertex / transfer use is format-agnostic here */
}

static bool
vp_screen_is_format_supported(struct pipe_screen *screen,
                              enum pipe_format format,
                              enum pipe_texture_target target,
                              unsigned sample_count,
                              unsigned storage_sample_count,
                              unsigned usage)
{
   struct vp_screen *vps = vp_reg_get(screen);
   if (!vps->lp_is_format_supported(screen, format, target, sample_count,
                                    storage_sample_count, usage))
      return false;
   /* Without a Vortex device this screen is plain llvmpipe; keep its answer. */
   if (!vps->dev)
      return true;
   /* The device path is single-sample (see the multisample features reported
    * off in lvp_device.c). */
   if (sample_count > 1 || storage_sample_count > 1)
      return false;
   return vp_device_format_supported(format, usage);
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
   vps->lp_context_create      = screen->context_create;
   vps->lp_screen_destroy      = screen->destroy;
   vps->lp_screen_get_name     = screen->get_name;
   vps->lp_finalize_nir        = screen->finalize_nir;
   vps->lp_is_format_supported = screen->is_format_supported;
   vp_reg_put(screen, vps);

   screen->context_create      = vp_context_create;
   screen->destroy             = vp_screen_destroy;
   screen->get_name            = vp_screen_get_name;
   screen->finalize_nir        = vp_finalize_nir;
   screen->is_format_supported = vp_screen_is_format_supported;

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

   /* shaderFloat64 must match the device (lvp derives it from caps.doubles).
    * Only advertise doubles when the Vortex FPU has the RISC-V D extension; an
    * F-only (FLEN=32) build would otherwise inherit llvmpipe's doubles=1 and
    * run fp64 shaders on a device that cannot execute them (silent wrong
    * results, e.g. zero_initialize_workgroup_memory.types.f64* returning
    * garbage). Without D, honestly report no fp64 so such tests are skipped. */
   if (vps->dev && !(vps->hw_isa_flags & VX_ISA_STD_D))
      ((struct pipe_caps *)&screen->caps)->doubles = 0;

   vp_dbg("vortexpipe: screen ready (llvmpipe base, Vortex hooks armed)");
   return screen;
}
