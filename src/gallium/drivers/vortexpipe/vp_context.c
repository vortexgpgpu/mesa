/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vortexpipe context interception.
 *
 * vp_context_create() wraps llvmpipe's context: it records the
 * llvmpipe originals and installs vortexpipe overrides for the
 * compute hooks:
 *   - create_compute_state : NIR -> SPIR-V -> .vxbin
 *   - launch_grid          : vx_enqueue_launch
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vp_private.h"
#include "vp_nir_to_llvm.h"
#include "vp_compile.h"
#include "vp_launch.h"
#include "vp_raster.h"
#include "gfx_frontend_abi.h" /* SETUP_CULL_* device face-cull modes */

#include "pipe/p_state.h"     /* full struct pipe_compute_state */
#include "util/format/u_formats.h"  /* PIPE_FORMAT_* */
#include "util/blend.h"             /* PIPE_BLENDFACTOR_*, PIPE_BLEND_* */
#include "util/hash_table.h"
#include "util/simple_mtx.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"   /* pipe_buffer_map / unmap */
#include "util/log.h"

#include "nir.h"
#include "nir_builder.h"
#include "compiler/glsl_types.h"

/* ---- verbose tracing ------------------------------------------------ */

void
vp_dbg(const char *fmt, ...)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("VORTEXPIPE_DEBUG") != NULL;
   if (!enabled)
      return;

   va_list ap;
   va_start(ap, fmt);
   mesa_log_v(MESA_LOG_INFO, "MESA", fmt, ap);
   va_end(ap);
}

/* ---- pointer-keyed side registry ------------------------------------ */

static simple_mtx_t vp_reg_mtx = SIMPLE_MTX_INITIALIZER;
static struct hash_table *vp_reg;          /* void* key -> vp_screen|vp_context */

void
vp_reg_put(const void *key, void *data)
{
   simple_mtx_lock(&vp_reg_mtx);
   if (!vp_reg)
      vp_reg = _mesa_pointer_hash_table_create(NULL);
   _mesa_hash_table_insert(vp_reg, key, data);
   simple_mtx_unlock(&vp_reg_mtx);
}

void *
vp_reg_get(const void *key)
{
   void *data = NULL;
   simple_mtx_lock(&vp_reg_mtx);
   if (vp_reg) {
      struct hash_entry *e = _mesa_hash_table_search(vp_reg, key);
      if (e)
         data = e->data;
   }
   simple_mtx_unlock(&vp_reg_mtx);
   return data;
}

void
vp_reg_del(const void *key)
{
   simple_mtx_lock(&vp_reg_mtx);
   if (vp_reg) {
      struct hash_entry *e = _mesa_hash_table_search(vp_reg, key);
      if (e)
         _mesa_hash_table_remove(vp_reg, e);
   }
   simple_mtx_unlock(&vp_reg_mtx);
}

/* ---- compute-hook overrides ----------------------------------------- */

/* create_compute_state returns a struct vp_cso* (llvmpipe's cso +
 * the compiled Vortex .vxbin); bind/delete unwrap it. */
static void *
vp_create_compute_state(struct pipe_context *pipe,
                        const struct pipe_compute_state *state)
{
   struct vp_context *vp = vp_reg_get(pipe);

   struct vp_cso *cso = CALLOC_STRUCT(vp_cso);
   if (!cso)
      return NULL;   /* OOM -- vkCreateComputePipelines fails cleanly */
   cso->lp_cso = vp->lp_create_compute_state(pipe, state);

   /* Translate NIR -> LLVM IR -> Vortex .vxbin and retain it. */
   if (state->ir_type == PIPE_SHADER_IR_NIR) {
      char *ir = NULL;
      struct nir_shader *cs_nir = (struct nir_shader *)state->prog;
      /* VK_KHR_zero_initialize_workgroup_memory: a `shared` var with a null
       * initializer must read back as zero. Vortex LMEM is not cleared between
       * dispatches, so emit the standard cooperative zeroing pass (store_shared
       * of zeros strided by local_invocation_index + a workgroup barrier). Align
       * the region up to the 16B store granularity; the LMEM allocation below
       * picks up the aligned size so the zeroing never runs past it. */
      if (cs_nir->info.zero_initialize_shared_memory && cs_nir->info.shared_size > 0) {
         cs_nir->info.shared_size = (cs_nir->info.shared_size + 15u) & ~15u;
         NIR_PASS(_, cs_nir, nir_zero_initialize_shared_memory,
                  cs_nir->info.shared_size, /*chunk_size=*/16);
      }
      /* shared-memory size -> the launch's local-memory allocation. */
      const struct shader_info *si = &cs_nir->info;
      cso->lmem_size = si->shared_size;
      /* A workgroup with no shared memory, no control barrier and a fixed
       * size has independent invocations, so it can be re-tiled to fit the
       * device CTA cap without changing results (the only cross-workgroup
       * invariant a ray-tracing megashader reads is the global invocation id). */
      cso->workgroup_shape_invariant =
         (si->shared_size == 0) && !si->uses_control_barrier &&
         !si->workgroup_size_variable;
      /* the set-0 descriptors the kernel reaches -> launch relocation. */
      vp_scan_descriptors((struct nir_shader *)state->prog,
                          cso->descs, &cso->num_descs);
      /* raw const-index shader-buffer slots (RT trace-ray command buffer) ->
       * SBT shader-record pointer relocation at launch. */
      cso->trace_cmd_slots =
         vp_scan_trace_cmd_slots((struct nir_shader *)state->prog);
      if (vp_nir_to_llvm((struct nir_shader *)state->prog, &ir, NULL, NULL)) {
         if (vp_compile_vxbin(ir, VP_STARTUP_FS, false, &cso->vxbin, &cso->vxbin_size))
            vp_dbg("vortexpipe: compiled shader -> %zu-byte .vxbin",
                      cso->vxbin_size);
         else
            /* The toolchain ran but failed — hard error (broken install,
             * clang/link error, vxbin.py error). Programs run on llvmpipe
             * but the test harness needs to see this; logw masked it. */
            mesa_loge("vortexpipe: .vxbin compile failed; "
                      "shader runs on llvmpipe");
         vp_free_ir(ir);
      } else {
         /* NIR->LLVM returned "not translatable yet" — this is a
          * feature-coverage gap, not a runtime failure. Stay as logw. */
         mesa_logw("vortexpipe: NIR->LLVM unavailable; "
                   "shader runs on llvmpipe");
      }
   }
   return cso;
}

static void
vp_bind_compute_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_cso     *cso = p;
   vp->cur_cso = cso;
   vp->lp_bind_compute_state(pipe, cso ? cso->lp_cso : NULL);
}

static void
vp_delete_compute_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_cso     *cso = p;
   if (vp->cur_cso == cso)
      vp->cur_cso = NULL;
   vp->lp_delete_compute_state(pipe, cso->lp_cso);
   vp_free_blob(cso->vxbin);
   FREE(cso);
}

/* True when MESA_VORTEX_STRICT=1: any fallback from Vortex to llvmpipe
 * is an error and the fallback is refused. The test harness sets this
 * so silent CPU execution of a Vortex test fails the test instead of
 * green-lighting it. Cached because getenv is process-stable. */
static bool
vp_strict_mode(void)
{
   static bool init   = false;
   static bool strict = false;
   if (!init) {
      const char *s = getenv("MESA_VORTEX_STRICT");
      strict = (s && s[0] != '0' && s[0] != '\0');
      init   = true;
   }
   return strict;
}

/* Re-tile a workgroup that exceeds the device CTA cap (num_threads ×
 * num_warps) into several device CTAs. Valid only for shape-invariant
 * shaders (see vp_cso.workgroup_shape_invariant): the transform preserves
 * every axis's global-invocation-id range — new_grid[i] * new_block[i] ==
 * grid[i] * block[i] — so a kernel that reads only the global id (the
 * ray-tracing megashader's LaunchIDEXT) sees identical work. Per axis the
 * new block is the largest divisor of the original block that fits the
 * remaining thread budget; power-of-two blocks (all Vulkan ray-tracing
 * workgroups) land on divisors that keep hardware warps whole. Returns
 * false if the result would still exceed the cap. */
static bool
vp_retile_workgroup(const uint32_t grid[3], const uint32_t block[3],
                    uint32_t cap, uint32_t out_grid[3], uint32_t out_block[3])
{
   uint32_t budget = cap;
   for (int i = 0; i < 3; i++) {
      uint32_t b = block[i] ? block[i] : 1u;
      uint32_t best = 1;
      for (uint32_t d = 1; d <= b && d <= budget; d++) {
         if (b % d == 0)
            best = d;
      }
      out_block[i] = best;
      out_grid[i]  = grid[i] * (b / best);
      budget /= best;
      if (budget == 0)
         budget = 1;
   }
   uint32_t prod = out_block[0] * out_block[1] * out_block[2];
   return prod != 0 && prod <= cap;
}

static void
vp_launch_grid(struct pipe_context *pipe, const struct pipe_grid_info *info)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_cso     *cso = vp->cur_cso;
   bool ran_on_vortex = false;

   /* Vortex's CTA dispatcher executes one workgroup as ONE CTA — the
    * KMU's block_size DCR is sized to address one CTA's threads
    * (CTA_TID_WIDTH+1 bits, max = num_threads × num_warps). A shader
    * whose local_size exceeds that cap truncates in the DCR write and
    * the CTA dispatcher fires warps with tmask=0, which the LSU then
    * asserts on (`invalid request mask`). Reject up-front. */
   struct vp_screen *vps = vp_reg_get(pipe->screen);
   uint32_t block_size = info->block[0] * info->block[1] * info->block[2];

   /* Empty dispatch: any zero grid or block dimension means zero workgroups /
    * zero invocations, which is defined to do nothing. Return before touching
    * the device — dispatching a zero-extent grid would otherwise fire warps
    * that write garbage. (Indirect dispatch reads its extent from a buffer at
    * device time, so it is not shape-checked here.) */
   if (!info->indirect) {
      uint32_t grid_size = info->grid[0] * info->grid[1] * info->grid[2];
      if (grid_size == 0 || block_size == 0) {
         vp_dbg("vortexpipe: launch_grid: empty dispatch grid=[%u,%u,%u] "
                "block=[%u,%u,%u] — no-op",
                info->grid[0], info->grid[1], info->grid[2],
                info->block[0], info->block[1], info->block[2]);
         return;
      }
   }

   /* The grid/block actually dispatched. When a shape-invariant workgroup
    * exceeds the device CTA cap, re-tile it into device-sized CTAs in place
    * of the app's oversized one; otherwise dispatch the app's shape. */
   const uint32_t *eff_grid  = info->grid;
   const uint32_t *eff_block = info->block;
   uint32_t rt_grid[3], rt_block[3];

   if (vps && vps->hw_max_block_size != 0 &&
       block_size > vps->hw_max_block_size) {
      if (cso && cso->workgroup_shape_invariant &&
          vp_retile_workgroup(info->grid, info->block, vps->hw_max_block_size,
                              rt_grid, rt_block)) {
         vp_dbg("vortexpipe: launch_grid: re-tiled workgroup %ux%ux%u -> "
                "block %ux%ux%u grid %ux%ux%u to fit cap %u",
                info->block[0], info->block[1], info->block[2],
                rt_block[0], rt_block[1], rt_block[2],
                rt_grid[0], rt_grid[1], rt_grid[2], vps->hw_max_block_size);
         eff_grid  = rt_grid;
         eff_block = rt_block;
      } else {
         mesa_logw("vortexpipe: launch_grid: workgroup size %u (%ux%ux%u) "
                   "exceeds device cap %u (%u threads × %u warps) and is not "
                   "re-tileable; fallback to llvmpipe",
                   block_size,
                   info->block[0], info->block[1], info->block[2],
                   vps->hw_max_block_size,
                   vps->hw_num_threads, vps->hw_num_warps);
         goto fallback;
      }
   }

   /* Indirect dispatch: the workgroup counts are not in info->grid but in a
    * device buffer at indirect_offset (three uint32 x/y/z). Read them and
    * dispatch that grid; a zero count is an empty (no-op) dispatch. Without
    * this the stale info->grid is dispatched (wrong workgroup count). */
   uint32_t ind_grid[3];
   if (info->indirect) {
      struct pipe_transfer *ixfer = NULL;
      const uint32_t *counts = (const uint32_t *)pipe_buffer_map_range(
         pipe, info->indirect, info->indirect_offset,
         3 * sizeof(uint32_t), PIPE_MAP_READ, &ixfer);
      if (!counts)
         goto fallback;
      ind_grid[0] = counts[0];
      ind_grid[1] = counts[1];
      ind_grid[2] = counts[2];
      pipe_buffer_unmap(pipe, ixfer);
      if (ind_grid[0] == 0 || ind_grid[1] == 0 || ind_grid[2] == 0) {
         vp_dbg("vortexpipe: launch_grid: empty indirect dispatch — no-op");
         return;
      }
      eff_grid = ind_grid;
   }

   /* Run on Vortex when the kernel compiled and we have set 0's
    * descriptor buffer (constant-buffer index 1) plus the descriptor
    * table vp_create_compute_state scanned out of the NIR. */
   if (vp->dev && cso && cso->vxbin && cso->num_descs && vp->cbuf[1]) {
      /* The device descriptor buffer must span every descriptor the
       * kernel reaches: the highest offset plus one struct lp_descriptor. */
      uint32_t desc_bytes = 0;
      for (uint32_t i = 0; i < cso->num_descs; i++) {
         uint32_t end = cso->descs[i].offset + VP_DESC_STRIDE;
         if (end > desc_bytes)
            desc_bytes = end;
      }
      struct pipe_transfer *xfer = NULL;
      void *desc_host = pipe_buffer_map(pipe, vp->cbuf[1],
                                        PIPE_MAP_READ, &xfer);
      if (desc_host) {
         /* Map any raw shader-buffer slots (e.g. the RT trace-ray command
          * buffer at slot 0) so vp_launch can relocate them into the arg block.
          * The maps stay valid across vp_launch (it vx_queue_finish()es before
          * returning, so the async uploads complete before we unmap). */
         struct vp_ssbo ssbos[VP_MAX_SSBO];
         struct pipe_transfer *sxfer[VP_MAX_SSBO] = { 0 };
         uint32_t num_ssbos = 0;
         for (unsigned s = 0; s < VP_MAX_SSBO; s++) {
            if (!vp->sbuf[s] || !vp->sbuf_sz[s])
               continue;
            void *h = pipe_buffer_map(pipe, vp->sbuf[s], PIPE_MAP_READ, &sxfer[s]);
            if (!h)
               continue;
            ssbos[num_ssbos].host = (uint8_t *)h + vp->sbuf_off[s];
            ssbos[num_ssbos].size = vp->sbuf_sz[s];
            ssbos[num_ssbos].slot = s;
            ssbos[num_ssbos].trace_cmd = (cso->trace_cmd_slots >> s) & 1u;
            num_ssbos++;
         }

         vp_dbg("vortexpipe: launch_grid -> Vortex grid=[%u,%u,%u] "
                "block=[%u,%u,%u] descs=%u ssbos=%u",
                eff_grid[0], eff_grid[1], eff_grid[2],
                eff_block[0], eff_block[1], eff_block[2],
                cso->num_descs, num_ssbos);
         ran_on_vortex = vp_launch(vp->dev, cso->vxbin, cso->vxbin_size,
                                   (uint8_t *)desc_host + vp->cbuf_off[1],
                                   desc_bytes, cso->descs, cso->num_descs,
                                   ssbos, num_ssbos,
                                   eff_grid, eff_block, info->grid_base,
                                   cso->lmem_size,
                                   vps && vps->has_rtu);
         for (unsigned s = 0; s < VP_MAX_SSBO; s++)
            if (sxfer[s])
               pipe_buffer_unmap(pipe, sxfer[s]);
         pipe_buffer_unmap(pipe, xfer);
      }
   }

   if (ran_on_vortex) {
      vp->launches_device++;
      vp_dbg("vortexpipe: launch_grid ran on Vortex");
      return;
   }

fallback:
   vp->launches_cpu++;
   if (vp_strict_mode()) {
      /* Refuse the silent llvmpipe fallback: the test harness catches
       * mesa_loge and fails the test, instead of green-lighting CPU
       * execution of a Vortex test. The launch becomes a no-op so the
       * application's validation step sees missing data and fails. */
      mesa_loge("vortexpipe: launch_grid: Vortex path unavailable, "
                "STRICT mode refuses llvmpipe fallback");
   } else {
      mesa_logw("vortexpipe: launch_grid: falling back to llvmpipe "
                "(set MESA_VORTEX_STRICT=1 to fail instead)");
      vp->lp_launch_grid(pipe, info);
   }
}

/* Buffer-binding interception -- launch_grid needs lavapipe's
 * descriptor buffer (bound as a compute constant buffer) to find
 * the SSBO data. Capture the compute constant buffers; forward. */
static void
vp_set_constant_buffer(struct pipe_context *pipe, enum pipe_shader_type shader,
                       unsigned index, bool take_ownership,
                       const struct pipe_constant_buffer *cb)
{
   struct vp_context *vp = vp_reg_get(pipe);
   if (shader == PIPE_SHADER_COMPUTE && index < 8) {
      vp->cbuf[index]     = cb ? cb->buffer : NULL;
      vp->cbuf_off[index] = cb ? cb->buffer_offset : 0u;
   }
   /* Capture the fragment stage's constant buffers so the draw path can
    * upload them + build the resident FS descriptor table (push constants at
    * index 0, descriptor set-0 blob at index 1, UBOs at their bound index). */
   if (shader == PIPE_SHADER_FRAGMENT && index < 8) {
      vp->fs_cbuf[index]     = cb ? cb->buffer : NULL;
      vp->fs_cbuf_off[index] = cb ? cb->buffer_offset : 0u;
      vp->fs_cbuf_sz[index]  = cb ? cb->buffer_size : 0u;
   }
   vp->lp_set_constant_buffer(pipe, shader, index, take_ownership, cb);
}

static void
vp_set_shader_buffers(struct pipe_context *pipe, enum pipe_shader_type shader,
                      unsigned start, unsigned count,
                      const struct pipe_shader_buffer *bufs,
                      unsigned writable_mask)
{
   struct vp_context *vp = vp_reg_get(pipe);
   /* Capture raw compute shader buffers so launch_grid can relocate them into
    * the kernel arg block (arg[VP_ARG_SSBO_BASE + slot]). lavapipe binds the RT
    * trace-ray command buffer here at slot 0; the megashader reads it as
    * load_ssbo(imm 0, off), so the device address must be supplied here. */
   if (shader == PIPE_SHADER_COMPUTE) {
      for (unsigned i = 0; i < count; i++) {
         unsigned slot = start + i;
         if (slot >= VP_MAX_SSBO)
            continue;
         const struct pipe_shader_buffer *b = bufs ? &bufs[i] : NULL;
         vp->sbuf[slot]     = b ? b->buffer : NULL;
         vp->sbuf_off[slot] = b ? b->buffer_offset : 0u;
         vp->sbuf_sz[slot]  = b ? b->buffer_size : 0u;
      }
   }
   vp->lp_set_shader_buffers(pipe, shader, start, count, bufs, writable_mask);
}

/* ---- graphics: vertex-shader hooks ------------------------ */

/* create_vs_state returns a struct vp_cso* (llvmpipe's cso + the
 * compiled Vortex vertex-shader kernel); bind/delete unwrap it. */
static void *
vp_create_vs_state(struct pipe_context *pipe,
                   const struct pipe_shader_state *state)
{
   struct vp_context *vp = vp_reg_get(pipe);

   struct vp_cso *cso = CALLOC_STRUCT(vp_cso);
   if (!cso)
      return NULL;   /* OOM -- vkCreateGraphicsPipelines fails cleanly */
   cso->lp_cso = vp->lp_create_vs_state(pipe, state);

   /* Translate the vertex shader NIR -> LLVM IR -> Vortex .vxbin. */
   if (state->type == PIPE_SHADER_IR_NIR) {
      char *ir = NULL;
      if (vp_nir_to_llvm((struct nir_shader *)state->ir.nir, &ir,
                         &cso->vs_layout, NULL)) {
         /* VS links at a distinct base so it co-resides with the FS
          * (0x80000000) + front end (0x80200000) in one OP_DRAW. */
         if (vp_compile_vxbin(ir, VP_STARTUP_VS, false, &cso->vxbin, &cso->vxbin_size))
            vp_dbg("vortexpipe: compiled vertex shader -> %zu-byte .vxbin",
                   cso->vxbin_size);
         else
            /* Toolchain ran but failed — hard error, same rationale as
             * the compute path above. */
            mesa_loge("vortexpipe: VS .vxbin compile failed; "
                      "vertex stage runs on llvmpipe");
         vp_free_ir(ir);
      } else {
         mesa_logw("vortexpipe: VS NIR->LLVM unavailable; "
                   "vertex stage runs on llvmpipe");
      }
   }
   return cso;
}

/* Residency: release a CSO's device-resident module (its vxbin loaded
 * onto the device). Frees the stage's fixed device address so a different
 * same-stage shader can take it. Safe to call when nothing is resident. */
static void
vp_cso_evict_module(struct vp_cso *cso)
{
   if (!cso)
      return;
   if (cso->vx_kernel) { vx_kernel_release(cso->vx_kernel); cso->vx_kernel = NULL; }
   if (cso->vx_module) { vx_module_release(cso->vx_module); cso->vx_module = NULL; }
}

static void
vp_bind_vs_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_cso     *cso = p;
   /* The VS device address is fixed (VP_STARTUP_VS); evict the previously
    * resident VS so the newly-bound one can load there on the next draw. */
   if (vp->cur_vs && vp->cur_vs != cso)
      vp_cso_evict_module(vp->cur_vs);
   vp->cur_vs = cso;
   vp->lp_bind_vs_state(pipe, cso ? cso->lp_cso : NULL);
}

static void
vp_delete_vs_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_cso     *cso = p;
   if (vp->cur_vs == cso)
      vp->cur_vs = NULL;
   vp_cso_evict_module(cso);
   vp->lp_delete_vs_state(pipe, cso->lp_cso);
   vp_free_blob(cso->vxbin);
   FREE(cso);
}

/* ---- graphics: fragment-shader hooks --------------------- */

/* Per-unit HW-vs-SW routing for a fragment shader, from device caps +
 * the VORTEXPIPE_FORCE_SW knob. A unit absent from the device routes to its
 * SIMT software path (never llvmpipe). All three units
 * (TEX/OM/RASTER) are wired here. SW raster implies SW OM (the one-thread-per-
 * tile kernel merges over the LSU; it has no FF frag window to feed vx_om4). */
static struct vp_sw_routing
vp_fs_routing(struct pipe_context *pipe)
{
   struct vp_screen *vps = vp_reg_get(pipe->screen);
   struct vp_sw_routing r = { false, false, false };
   const char *force = getenv("VORTEXPIPE_FORCE_SW");
   bool force_all    = force && strstr(force, "all");
   bool force_tex    = force && (strstr(force, "tex")    || force_all);
   bool force_om     = force && (strstr(force, "om")     || force_all);
   bool force_raster = force && (strstr(force, "raster") || force_all);
   r.sw_tex    = (vps && !vps->has_tex)    || force_tex;
   r.sw_om     = (vps && !vps->has_om)     || force_om;
   r.sw_raster = (vps && !vps->has_raster) || force_raster;
   if (r.sw_raster)
      r.sw_om = true;   /* SW raster has no FF window → must merge in software */
   return r;
}

/* Count the fragment shader's colour outputs (max render-target index
 * + 1). >1 means the draw targets multiple render targets, which the SW OM MRT
 * fallback handles (the FF vx_om4 unit is single-RT). */
static unsigned
vp_fs_num_color_outputs(struct nir_shader *nir)
{
   unsigned n = 0;
   nir_foreach_shader_out_variable(var, nir) {
      unsigned loc = var->data.location;
      if (loc == FRAG_RESULT_COLOR) {
         if (n < 1) n = 1;                    /* broadcast colour -> RT0 */
      } else if (loc >= FRAG_RESULT_DATA0) {
         unsigned rt = loc - FRAG_RESULT_DATA0;
         if (rt + 1 > n) n = rt + 1;
      }
   }
   return n ? n : 1;
}

/* The driver JIT-compiles the fragment shader at pipeline creation,
 * the same NIR -> LLVM -> .vxbin path the vertex/compute stages use
 * (a real GPU driver compiles every stage; nothing is prebuilt). */
static void *
vp_create_fs_state(struct pipe_context *pipe,
                   const struct pipe_shader_state *state)
{
   struct vp_context *vp = vp_reg_get(pipe);

   struct vp_cso *cso = CALLOC_STRUCT(vp_cso);
   if (!cso)
      return NULL;
   cso->lp_cso = vp->lp_create_fs_state(pipe, state);
   cso->fs_routing = vp_fs_routing(pipe);
   cso->fs_num_color = 1;

   if (state->type == PIPE_SHADER_IR_NIR) {
      char *ir = NULL;
      /* A >1-RT fragment shader must merge in software — the FF OM unit
       * is single-attachment. Force SW OM here so the compiled kernel calls the
       * MRT fallback AND the draw path (fs_routing.sw_om) programs it to match. */
      cso->fs_num_color =
         vp_fs_num_color_outputs((struct nir_shader *)state->ir.nir);
      if (cso->fs_num_color > 1)
         cso->fs_routing.sw_om = true;
      /* The set-0 descriptors the FS reaches (SSBO/UBO/AS) — recorded for
       * the descriptor-blob relocation the SSBO path needs (follow-up). */
      vp_scan_descriptors((struct nir_shader *)state->ir.nir,
                          cso->descs, &cso->num_descs);
      /* Locate the sampled texture's descriptor so the draw can pick the right
       * texture per draw (multi-texture); false ⇒ fall back to cur_tex. */
      cso->has_tex_desc = vp_scan_tex_descriptor(
         (struct nir_shader *)state->ir.nir,
         &cso->tex_desc_cbuf, &cso->tex_desc_offset);
      if (vp_nir_to_llvm((struct nir_shader *)state->ir.nir, &ir, NULL,
                         &cso->fs_routing)) {
         /* Co-compile the gfx_sw ABI (gfx_tex_sample_sw / gfx_om_fragment_sw)
          * whenever the FS could call it: a routed-to-SW unit, or a HW-TEX shader
          * that samples a texture (a mipmapped sampler routes to the SW sampler at
          * draw time via the emit_tex runtime branch). --gc-sections drops it if
          * unused, so a non-mipmapped textured FS pays only compile time. */
         bool uses_sw = cso->fs_routing.sw_tex || cso->fs_routing.sw_om ||
                        cso->has_tex_desc;
         if (vp_compile_vxbin(ir, VP_STARTUP_FS, uses_sw, &cso->vxbin, &cso->vxbin_size))
            vp_dbg("vortexpipe: compiled fragment shader -> %zu-byte .vxbin%s",
                   cso->vxbin_size, uses_sw ? " (SW units)" : "");
         else
            mesa_loge("vortexpipe: FS .vxbin compile failed");
         vp_free_ir(ir);
      } else {
         mesa_logw("vortexpipe: FS NIR->LLVM unavailable; "
                   "fragment stage runs on llvmpipe");
      }
   }
   return cso;
}

static void
vp_bind_fs_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_cso     *cso = p;
   /* FS device address is fixed (VP_STARTUP_FS); evict the previously resident
    * FS so the newly-bound one can load there on the next draw. */
   if (vp->cur_fs && vp->cur_fs != cso)
      vp_cso_evict_module(vp->cur_fs);
   vp->cur_fs = cso;
   vp->lp_bind_fs_state(pipe, cso ? cso->lp_cso : NULL);
}

static void
vp_delete_fs_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_cso     *cso = p;
   if (vp->cur_fs == cso)
      vp->cur_fs = NULL;
   vp_cso_evict_module(cso);
   vp->lp_delete_fs_state(pipe, cso->lp_cso);
   vp_free_blob(cso->vxbin);
   FREE(cso);
}

/* ---- graphics: texture-sampler state --------------------- */

/* VX TEX encodings (VX_types.h) -- filter + wrap. */
#define VX_TEX_FILTER_POINT     0
#define VX_TEX_FILTER_BILINEAR  1
#define VX_TEX_WRAP_CLAMP       0
#define VX_TEX_WRAP_REPEAT      1
#define VX_TEX_WRAP_MIRROR      2

static uint32_t
vp_vx_filter(unsigned f)
{
   return (f == PIPE_TEX_FILTER_LINEAR) ? VX_TEX_FILTER_BILINEAR
                                        : VX_TEX_FILTER_POINT;
}

static uint32_t
vp_vx_wrap(unsigned w)
{
   switch (w) {
   case PIPE_TEX_WRAP_REPEAT:        return VX_TEX_WRAP_REPEAT;
   case PIPE_TEX_WRAP_MIRROR_REPEAT: return VX_TEX_WRAP_MIRROR;
   default:                          return VX_TEX_WRAP_CLAMP;
   }
}

/* Capture the bound texture + sampler for the Vortex TEX unit.
 *
 * lavapipe does not bind app textures through set_sampler_views (it
 * keeps descriptors in a buffer the shader dereferences); the one
 * place the underlying pipe_sampler_view / pipe_sampler_state surface
 * is create_texture_handle. lavapipe calls it twice per resource: an
 * image view passes the view (sampler state NULL), a sampler passes
 * the state (view NULL). gfx-v1 drives a single TEX stage, so we
 * capture the latest texture and the latest sampler. */
static uint64_t
vp_create_texture_handle(struct pipe_context *pipe,
                         struct pipe_sampler_view *view,
                         const struct pipe_sampler_state *state)
{
   struct vp_context *vp = vp_reg_get(pipe);
   if (view && view->texture) {
      vp->cur_tex = view->texture;
      vp_dbg("vortexpipe: TEX texture captured (%ux%u)",
             vp->cur_tex->width0, vp->cur_tex->height0);
      /* Record this texture's level-0 host base so a draw can match its FS tex
       * descriptor (lp_jit_texture.base) back to the resource — the per-draw
       * selection that lets >1 bound texture disambiguate. Dedup by resource. */
      struct pipe_resource *res = view->texture;
      bool known = false;
      for (unsigned i = 0; i < vp->txh_count; i++)
         if (vp->txh_res[i] == res) { known = true; break; }
      if (!known && vp->txh_count < VP_MAX_TEX_HANDLES) {
         struct pipe_transfer *xfer = NULL;
         const void *base = pipe_texture_map(pipe, res, 0, 0, PIPE_MAP_READ,
                                             0, 0, res->width0, res->height0, &xfer);
         if (base) {
            vp->txh_base[vp->txh_count] = base;
            vp->txh_res[vp->txh_count]  = res;
            vp->txh_count++;
            pipe_texture_unmap(pipe, xfer);
         }
      }
   }
   if (state) {
      vp->cur_sampler_store.filter     = vp_vx_filter(state->mag_img_filter);
      vp->cur_sampler_store.min_filter = vp_vx_filter(state->min_img_filter);
      vp->cur_sampler_store.wrap_u = vp_vx_wrap(state->wrap_s);
      vp->cur_sampler_store.wrap_v = vp_vx_wrap(state->wrap_t);
      /* Vulkan has no "disable mipmapping" flag; a non-mipmapped NEAREST/LINEAR
       * sampler is expressed as max_lod == 0.25, which clamps the LOD so only the
       * base level is ever read. A level >= 1 is reachable only when max_lod > 0.5
       * (nearest-mip selects ceil(lod + 0.5) - 1), so that is the mipmap-enable
       * test — min_mip_filter is always NEAREST here and cannot distinguish them. */
      vp->cur_sampler_store.mip_enable = (state->max_lod > 0.5f);
      vp->cur_sampler_store.mip_linear =
         (state->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR);
      vp->cur_sampler = &vp->cur_sampler_store;
      vp_dbg("vortexpipe: TEX sampler captured mag=%u min=%u wrap=%u,%u mip_enable=%u mip_linear=%u",
             vp->cur_sampler->filter, vp->cur_sampler->min_filter,
             vp->cur_sampler->wrap_u, vp->cur_sampler->wrap_v,
             vp->cur_sampler->mip_enable, vp->cur_sampler->mip_linear);
   }
   return vp->lp_create_texture_handle(pipe, view, state);
}

/* Pick the texture the FS actually samples this draw. lavapipe keeps sampled
 * images in per-set descriptor blobs the shader dereferences (not through
 * set_sampler_views), so a shader with >1 bound texture would otherwise all
 * resolve to the last handle create_texture_handle captured. Read
 * lp_jit_texture.base (offset 0 of the sampled image's lp_descriptor) from the
 * bound blob at the FS's (cbuf_index, offset) and match it to a recorded
 * resource, overriding cur_tex. On any miss (no bindless handle, unmapped blob,
 * unrecorded base) cur_tex is left as captured — the single-texture path. */
static void
vp_resolve_tex_from_desc(struct pipe_context *pipe, struct vp_context *vp,
                         const struct vp_cso *fs)
{
   if (!fs || !fs->has_tex_desc)
      return;
   unsigned ci = fs->tex_desc_cbuf;
   if (ci >= 8 || !vp->fs_cbuf[ci])
      return;
   if (fs->tex_desc_offset + sizeof(const void *) > vp->fs_cbuf_sz[ci])
      return;
   struct pipe_transfer *xfer = NULL;
   const uint8_t *blob = pipe_buffer_map(pipe, vp->fs_cbuf[ci], PIPE_MAP_READ, &xfer);
   if (!blob)
      return;
   const void *base = NULL;
   memcpy(&base, blob + vp->fs_cbuf_off[ci] + fs->tex_desc_offset, sizeof base);
   pipe_buffer_unmap(pipe, xfer);
   for (unsigned i = 0; i < vp->txh_count; i++)
      if (vp->txh_base[i] == base) { vp->cur_tex = vp->txh_res[i]; break; }
}

/* ---- graphics: vertex input ---------------------------------------- *
 * The VS runs as a Vortex compute kernel that fetches each thread's
 * vertex attributes from device memory, so vortexpipe captures the
 * vertex-elements layout + the bound vertex buffers and hands them to
 * vp_launch_vs. The velems cso is registered in the pointer registry
 * (keyed by the llvmpipe cso) like the depth/blend csos -- csos made
 * by util_blitter before the hooks armed pass straight through. */
static void *
vp_create_vertex_elements_state(struct pipe_context *pipe, unsigned num,
                                const struct pipe_vertex_element *elements)
{
   struct vp_context *vp = vp_reg_get(pipe);
   void *lp_cso = vp->lp_create_vertex_elements_state(pipe, num, elements);

   struct vp_velems_cso *cso = CALLOC_STRUCT(vp_velems_cso);
   if (cso && num <= VP_MAX_ATTR) {
      cso->num = num;
      for (unsigned i = 0; i < num; i++) {
         cso->src_offset[i]      = elements[i].src_offset;
         cso->src_stride[i]      = elements[i].src_stride;
         cso->buffer_index[i]    = elements[i].vertex_buffer_index;
         cso->instance_divisor[i] = elements[i].instance_divisor;
      }
      vp_reg_put(lp_cso, cso);
   } else {
      /* >VP_MAX_ATTR attributes: leave unregistered -> the draw sees
       * cur_velems == NULL and falls back to llvmpipe. */
      FREE(cso);
   }
   return lp_cso;
}

static void
vp_bind_vertex_elements_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp = vp_reg_get(pipe);
   vp->cur_velems = p ? vp_reg_get(p) : NULL;
   vp->lp_bind_vertex_elements_state(pipe, p);
}

static void
vp_delete_vertex_elements_state(struct pipe_context *pipe, void *p)
{
   struct vp_context    *vp  = vp_reg_get(pipe);
   struct vp_velems_cso *cso = p ? vp_reg_get(p) : NULL;
   if (cso) {
      if (vp->cur_velems == cso)
         vp->cur_velems = NULL;
      vp_reg_del(p);
      FREE(cso);
   }
   vp->lp_delete_vertex_elements_state(pipe, p);
}

static void
vp_set_vertex_buffers(struct pipe_context *pipe, unsigned count,
                      const struct pipe_vertex_buffer *buffers)
{
   struct vp_context *vp = vp_reg_get(pipe);
   vp->num_vbufs = 0;
   if (buffers && count <= VP_MAX_ATTR) {
      for (unsigned i = 0; i < count; i++)
         vp->vbufs[i] = buffers[i];
      vp->num_vbufs = count;
   }
   vp->lp_set_vertex_buffers(pipe, count, buffers);
}

/* ---- graphics: framebuffer interception ------------------ */

/* Residency: defined below, used by set_framebuffer_state to flush the
 * outgoing render pass's resident colour back before the framebuffer changes. */
static void vp_fb_invalidate(struct pipe_context *pipe, struct vp_context *vp);

/* Capture colour attachment 0 and the depth/stencil attachment so the
 * Vortex raster path can round-trip them; everything else stays with
 * llvmpipe. */
static void
vp_set_framebuffer_state(struct pipe_context *pipe,
                         const struct pipe_framebuffer_state *fb)
{
   struct vp_context *vp = vp_reg_get(pipe);
   /* end of the old render pass: flush its resident colour back + drop the
    * resident buffers (the new framebuffer re-initialises on its first draw). */
   vp_fb_invalidate(pipe, vp);
   vp->fb_color  = (fb && fb->nr_cbufs > 0 && fb->cbufs[0])
                      ? fb->cbufs[0]->texture : NULL;
   vp->fb_depth  = (fb && fb->zsbuf) ? fb->zsbuf->texture : NULL;
   vp->fb_width  = fb ? fb->width  : 0;
   vp->fb_height = fb ? fb->height : 0;
   /* Capture every bound colour attachment (fb_cbufs[0] == fb_color). */
   vp->fb_nr_cbufs = 0;
   for (unsigned i = 0; i < GFX_OM_MAX_RT; i++)
      vp->fb_cbufs[i] = NULL;
   if (fb) {
      unsigned n = fb->nr_cbufs < GFX_OM_MAX_RT ? fb->nr_cbufs : GFX_OM_MAX_RT;
      for (unsigned i = 0; i < n; i++)
         vp->fb_cbufs[i] = fb->cbufs[i] ? fb->cbufs[i]->texture : NULL;
      vp->fb_nr_cbufs = n;
   }
   vp->lp_set_framebuffer_state(pipe, fb);
}

/* Copy a render-target attachment between a tight w*h 32-bpp host
 * buffer and the (tiled) Gallium resource. The Vortex raster path
 * round-trips colour + depth this way: read the llvmpipe-cleared
 * attachment, run the raster/OM path, write the result back. */
static bool
vp_resource_rw(struct pipe_context *pipe, struct pipe_resource *res,
               unsigned w, unsigned h, void *host, bool write)
{
   if (!res)
      return false;
   struct pipe_transfer *xfer = NULL;
   uint8_t *map = pipe_texture_map(pipe, res, 0, 0,
                                   write ? PIPE_MAP_WRITE : PIPE_MAP_READ,
                                   0, 0, w, h, &xfer);
   if (!map)
      return false;
   for (unsigned y = 0; y < h; y++) {
      uint8_t *m = map + (size_t)y * xfer->stride;
      uint8_t *t = (uint8_t *)host + (size_t)y * w * 4;
      if (write)
         memcpy(m, t, (size_t)w * 4);
      else
         memcpy(t, m, (size_t)w * 4);
   }
   pipe_texture_unmap(pipe, xfer);
   return true;
}

static bool
vp_fb_color_read(struct pipe_context *pipe, struct vp_context *vp, void *dst)
{
   return vp_resource_rw(pipe, vp->fb_color, vp->fb_width, vp->fb_height,
                         dst, false);
}

/* VX OM encodings (VX_types.h) -- depth-compare function and blend.
 * Used by the residency depth-clear heuristic below and the OM-state
 * translation further down. */
#define VX_OM_DEPTH_FUNC_ALWAYS              0
#define VX_OM_DEPTH_FUNC_NEVER               1
#define VX_OM_DEPTH_FUNC_LESS                2
#define VX_OM_DEPTH_FUNC_LEQUAL              3
#define VX_OM_DEPTH_FUNC_EQUAL               4
#define VX_OM_DEPTH_FUNC_GEQUAL              5
#define VX_OM_DEPTH_FUNC_GREATER             6
#define VX_OM_DEPTH_FUNC_NOTEQUAL            7
#define VX_OM_BLEND_MODE_ADD                 0
#define VX_OM_BLEND_MODE_SUB                 1
#define VX_OM_BLEND_MODE_REV_SUB             2
#define VX_OM_BLEND_MODE_MIN                 3
#define VX_OM_BLEND_MODE_MAX                 4
#define VX_OM_BLEND_FUNC_ZERO                0
#define VX_OM_BLEND_FUNC_ONE                 1
#define VX_OM_BLEND_FUNC_SRC_RGB             2
#define VX_OM_BLEND_FUNC_ONE_MINUS_SRC_RGB   3
#define VX_OM_BLEND_FUNC_DST_RGB             4
#define VX_OM_BLEND_FUNC_ONE_MINUS_DST_RGB   5
#define VX_OM_BLEND_FUNC_SRC_A               6
#define VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A     7
#define VX_OM_BLEND_FUNC_DST_A               8
#define VX_OM_BLEND_FUNC_ONE_MINUS_DST_A     9
#define VX_OM_BLEND_FUNC_CONST_RGB           10
#define VX_OM_BLEND_FUNC_ONE_MINUS_CONST_RGB 11

/* ---- framebuffer + texture residency ------------------------------ *
 * The colour + depth attachments are kept device-resident across the draws of
 * a render pass: cleared/initialised ONCE (not per draw — so depth + colour
 * accumulate correctly across draws), rendered into in place, and copied back
 * to the colour resource only at present / framebuffer-change / llvmpipe
 * fallback. Textures upload once and stay resident, keyed by the bound
 * resource. This removes the per-draw framebuffer round-trip + texture upload. */

/* Allocate a device buffer and (optionally) upload `bytes` from host `src`. */
static bool
vp_dev_upload(vx_device_h dev, const void *src, size_t bytes,
              vx_buffer_h *out_buf, uint64_t *out_addr)
{
   vx_buffer_h b = NULL;
   if (vx_buffer_create(dev, bytes ? bytes : 1, 0, &b) != VX_SUCCESS)
      return false;
   if (vx_buffer_address(b, out_addr) != VX_SUCCESS) {
      vx_buffer_release(b);
      return false;
   }
   bool ok = true;
   if (src && bytes) {
      vx_queue_h q = NULL;
      vx_queue_info_t qi = { sizeof(qi), NULL, VX_QUEUE_PRIORITY_NORMAL, 0 };
      if (vx_queue_create(dev, &qi, &q) != VX_SUCCESS) {
         vx_buffer_release(b);
         return false;
      }
      ok = vx_enqueue_write(q, b, 0, src, bytes, 0, NULL, NULL) == VX_SUCCESS
        && vx_queue_finish(q, VX_TIMEOUT_INFINITE) == VX_SUCCESS;
      vx_queue_release(q);
   }
   if (!ok) { vx_buffer_release(b); return false; }
   *out_buf = b;
   return true;
}

/* Copy the resident colour buffer back to the framebuffer's colour resource so
 * the host / a present / an llvmpipe read observes the rendered result. */
static void
vp_fb_sync_out(struct pipe_context *pipe, struct vp_context *vp)
{
   if (!vp->rfb_dirty || !vp->rcb || !vp->rfb_res)
      return;
   const uint32_t bytes = vp->rfb_w * vp->rfb_h * 4;
   void *host = malloc(bytes);
   if (host &&
       vp_buffer_readback(vp->dev, vp->rcb, host, bytes)) {
      vp_resource_rw(pipe, vp->rfb_res, vp->rfb_w, vp->rfb_h, host, true);
      /* Write each extra colour attachment (1..) back to its resource. */
      for (unsigned k = 1; k < vp->rmrt_nr; k++) {
         if (!vp->rcb_extra[k] || !vp->rmrt_res[k])
            continue;
         if (vp_buffer_readback(vp->dev, vp->rcb_extra[k], host, bytes))
            vp_resource_rw(pipe, vp->rmrt_res[k], vp->rfb_w, vp->rfb_h, host, true);
      }
   }
   free(host);
   vp->rfb_dirty = false;
}

/* Flush + drop the resident colour/depth buffers (framebuffer change, fallback,
 * teardown): the colour resource becomes authoritative again. */
static void
vp_fb_invalidate(struct pipe_context *pipe, struct vp_context *vp)
{
   vp_fb_sync_out(pipe, vp);
   if (vp->rcb) { vx_buffer_release(vp->rcb); vp->rcb = NULL; }
   if (vp->rzb) { vx_buffer_release(vp->rzb); vp->rzb = NULL; }
   /* Drop the extra colour attachments. */
   for (unsigned k = 0; k < GFX_OM_MAX_RT; k++) {
      if (vp->rcb_extra[k]) { vx_buffer_release(vp->rcb_extra[k]); vp->rcb_extra[k] = NULL; }
      vp->rmrt_res[k] = NULL;
   }
   vp->rmrt_nr = 0;
   vp->rfb_res = NULL;
   vp->rfb_w = vp->rfb_h = 0;
}

/* Ensure the resident colour + depth buffers exist for the bound framebuffer at
 * (w,h); (re)allocate + initialise (colour from the resource's clear, depth to
 * the far value) once per pass. Returns the device addresses, or false. */
static bool
vp_fb_ensure(struct pipe_context *pipe, struct vp_context *vp,
             uint32_t w, uint32_t h, const struct vp_om_params *om,
             uint64_t *color_dev, uint64_t *depth_dev)
{
   if (vp->rcb && vp->rfb_res == vp->fb_color &&
       vp->rfb_w == w && vp->rfb_h == h) {
      /* reuse the resident pass buffers (preserve colour + depth across draws) */
      if (vx_buffer_address(vp->rcb, color_dev) != VX_SUCCESS) return false;
      if (vx_buffer_address(vp->rzb, depth_dev) != VX_SUCCESS) return false;
      return true;
   }

   vp_fb_invalidate(pipe, vp);

   const uint32_t bytes = w * h * 4;
   /* colour init: capture the attachment's current contents (the render pass's
    * loadOp=CLEAR already cleared the resource via llvmpipe). */
   void *cinit = malloc(bytes);
   bool cok = cinit && vp_fb_color_read(pipe, vp, cinit);
   if (cok)
      cok = vp_dev_upload(vp->dev, cinit, bytes, &vp->rcb, color_dev);
   free(cinit);
   if (!cok) return false;

   /* depth clear: far value (GREATER/GEQUAL clear to 0, else max), once. */
   uint8_t zfill = (om->depth_func == VX_OM_DEPTH_FUNC_GREATER ||
                    om->depth_func == VX_OM_DEPTH_FUNC_GEQUAL) ? 0x00 : 0xFF;
   void *zinit = malloc(bytes);
   bool zok = zinit != NULL;
   if (zok) {
      memset(zinit, zfill, bytes);
      zok = vp_dev_upload(vp->dev, zinit, bytes, &vp->rzb, depth_dev);
   }
   free(zinit);
   if (!zok) { vx_buffer_release(vp->rcb); vp->rcb = NULL; return false; }

   vp->rfb_res = vp->fb_color;
   vp->rfb_w = w; vp->rfb_h = h;
   vp->rfb_dirty = false;
   return true;
}

/* Ensure the resident device buffers for colour attachments 1.. exist
 * (RT0 + depth are handled by vp_fb_ensure, which the caller runs first). Each
 * extra attachment is initialised once per pass from its resource's cleared
 * contents; color_dev[k] returns its device base (color_dev[0] is left for the
 * caller to fill from vp->rcb). Returns false on allocation failure. */
static bool
vp_fb_ensure_mrt(struct pipe_context *pipe, struct vp_context *vp,
                 uint32_t w, uint32_t h, unsigned num,
                 uint64_t color_dev[GFX_OM_MAX_RT])
{
   if (num > GFX_OM_MAX_RT) num = GFX_OM_MAX_RT;
   const uint32_t bytes = w * h * 4;

   bool have = (vp->rmrt_nr == num);
   for (unsigned k = 1; k < num && have; k++)
      if (!vp->rcb_extra[k] || vp->rmrt_res[k] != vp->fb_cbufs[k])
         have = false;

   if (!have) {
      for (unsigned k = 1; k < GFX_OM_MAX_RT; k++) {
         if (vp->rcb_extra[k]) { vx_buffer_release(vp->rcb_extra[k]); vp->rcb_extra[k] = NULL; }
         vp->rmrt_res[k] = NULL;
      }
      for (unsigned k = 1; k < num; k++) {
         uint64_t dev = 0;
         void *cinit = malloc(bytes);
         bool ok = cinit && vp->fb_cbufs[k] &&
                   vp_resource_rw(pipe, vp->fb_cbufs[k], w, h, cinit, false);
         if (ok)
            ok = vp_dev_upload(vp->dev, cinit, bytes, &vp->rcb_extra[k], &dev);
         free(cinit);
         if (!ok) { vp->rmrt_nr = 0; return false; }
         vp->rmrt_res[k] = vp->fb_cbufs[k];
      }
      vp->rmrt_nr = num;
   }

   for (unsigned k = 1; k < num; k++)
      if (vx_buffer_address(vp->rcb_extra[k], &color_dev[k]) != VX_SUCCESS)
         return false;
   return true;
}

/* Ensure the bound texture is uploaded + resident, keyed by its resource. The
 * mip-0 image is read back to a tight A8R8G8B8 host buffer and uploaded once;
 * a re-bind of the same resource reuses it. Returns the device address. */
static bool
vp_tex_ensure(struct pipe_context *pipe, struct vp_context *vp,
              struct pipe_resource *res, uint32_t w, uint32_t h,
              uint64_t *tex_dev, uint32_t mip_off[VX_TEX_LOD_MAX + 1])
{
   if (vp->rtex_buf && vp->rtex_res == res &&
       vp->rtex_w == w && vp->rtex_h == h) {
      memcpy(mip_off, vp->rtex_mipoff, sizeof(vp->rtex_mipoff));
      return vx_buffer_address(vp->rtex_buf, tex_dev) == VX_SUCCESS;
   }

   if (vp->rtex_buf) { vx_buffer_release(vp->rtex_buf); vp->rtex_buf = NULL; }
   vp->rtex_res = NULL;

   /* Upload the whole mip chain contiguously so the TEX unit can address any
    * level the shader selects: level l lives at texel offset off_texels[l] and
    * has dims max(1, w>>l) x max(1, h>>l). mip_off carries the byte offsets;
    * levels past the resource's last are clamped to the smallest real level so
    * an over-large computed LOD still lands on valid texels. A single-level
    * texture yields off 0 for every LOD (byte-identical to the old upload). */
   const uint32_t last = (res->last_level < (uint32_t)VX_TEX_LOD_MAX)
                       ? res->last_level : (uint32_t)VX_TEX_LOD_MAX;
   uint32_t off_texels[VX_TEX_LOD_MAX + 1] = { 0 };
   uint32_t total = 0;
   for (uint32_t l = 0; l <= last; l++) {
      uint32_t wl = (w >> l) ? (w >> l) : 1u;
      uint32_t hl = (h >> l) ? (h >> l) : 1u;
      off_texels[l] = total;
      total += wl * hl;
   }

   uint32_t *texbuf = malloc((size_t)total * 4);
   bool ok = texbuf != NULL;
   for (uint32_t l = 0; ok && l <= last; l++) {
      uint32_t wl = (w >> l) ? (w >> l) : 1u;
      uint32_t hl = (h >> l) ? (h >> l) : 1u;
      struct pipe_transfer *xfer = NULL;
      uint8_t *map = pipe_texture_map(pipe, res, l, 0, PIPE_MAP_READ,
                                      0, 0, wl, hl, &xfer);
      if (!map) { ok = false; break; }
      uint32_t *dst = texbuf + off_texels[l];
      for (uint32_t y = 0; y < hl; y++) {
         uint8_t *m = map + (size_t)y * xfer->stride;
         for (uint32_t x = 0; x < wl; x++) {
            uint32_t r = m[x * 4 + 0], g = m[x * 4 + 1];
            uint32_t b = m[x * 4 + 2], a = m[x * 4 + 3];
            dst[(size_t)y * wl + x] = (a << 24) | (r << 16) | (g << 8) | b;
         }
      }
      pipe_texture_unmap(pipe, xfer);
   }

   for (uint32_t l = 0; l <= (uint32_t)VX_TEX_LOD_MAX; l++)
      mip_off[l] = off_texels[l <= last ? l : last] * 4u;

   if (ok)
      ok = vp_dev_upload(vp->dev, texbuf, (size_t)total * 4,
                         &vp->rtex_buf, tex_dev);
   free(texbuf);
   if (!ok) return false;
   vp->rtex_res = res;
   vp->rtex_w = w; vp->rtex_h = h;
   memcpy(vp->rtex_mipoff, mip_off, sizeof(vp->rtex_mipoff));
   return true;
}

/* ---- graphics: output-merger state ----------------------- */

static uint32_t
vp_vx_depth_func(unsigned pf)
{
   switch (pf) {
   case PIPE_FUNC_NEVER:    return VX_OM_DEPTH_FUNC_NEVER;
   case PIPE_FUNC_LESS:     return VX_OM_DEPTH_FUNC_LESS;
   case PIPE_FUNC_EQUAL:    return VX_OM_DEPTH_FUNC_EQUAL;
   case PIPE_FUNC_LEQUAL:   return VX_OM_DEPTH_FUNC_LEQUAL;
   case PIPE_FUNC_GREATER:  return VX_OM_DEPTH_FUNC_GREATER;
   case PIPE_FUNC_NOTEQUAL: return VX_OM_DEPTH_FUNC_NOTEQUAL;
   case PIPE_FUNC_GEQUAL:   return VX_OM_DEPTH_FUNC_GEQUAL;
   default:                 return VX_OM_DEPTH_FUNC_ALWAYS;
   }
}

static uint32_t
vp_vx_blend_factor(unsigned bf)
{
   switch (bf) {
   case PIPE_BLENDFACTOR_ZERO:            return VX_OM_BLEND_FUNC_ZERO;
   case PIPE_BLENDFACTOR_ONE:             return VX_OM_BLEND_FUNC_ONE;
   case PIPE_BLENDFACTOR_SRC_COLOR:       return VX_OM_BLEND_FUNC_SRC_RGB;
   case PIPE_BLENDFACTOR_INV_SRC_COLOR:   return VX_OM_BLEND_FUNC_ONE_MINUS_SRC_RGB;
   case PIPE_BLENDFACTOR_DST_COLOR:       return VX_OM_BLEND_FUNC_DST_RGB;
   case PIPE_BLENDFACTOR_INV_DST_COLOR:   return VX_OM_BLEND_FUNC_ONE_MINUS_DST_RGB;
   case PIPE_BLENDFACTOR_SRC_ALPHA:       return VX_OM_BLEND_FUNC_SRC_A;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA:   return VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A;
   case PIPE_BLENDFACTOR_DST_ALPHA:       return VX_OM_BLEND_FUNC_DST_A;
   case PIPE_BLENDFACTOR_INV_DST_ALPHA:   return VX_OM_BLEND_FUNC_ONE_MINUS_DST_A;
   case PIPE_BLENDFACTOR_CONST_COLOR:     return VX_OM_BLEND_FUNC_CONST_RGB;
   case PIPE_BLENDFACTOR_INV_CONST_COLOR: return VX_OM_BLEND_FUNC_ONE_MINUS_CONST_RGB;
   default:                               return VX_OM_BLEND_FUNC_ONE;
   }
}

static uint32_t
vp_vx_blend_mode(unsigned bm)
{
   switch (bm) {
   case PIPE_BLEND_SUBTRACT:         return VX_OM_BLEND_MODE_SUB;
   case PIPE_BLEND_REVERSE_SUBTRACT: return VX_OM_BLEND_MODE_REV_SUB;
   case PIPE_BLEND_MIN:              return VX_OM_BLEND_MODE_MIN;
   case PIPE_BLEND_MAX:              return VX_OM_BLEND_MODE_MAX;
   default:                          return VX_OM_BLEND_MODE_ADD;
   }
}

/* depth-stencil-alpha state: capture depth test / func / writemask
 * into a vp_dsa_cso registered under the llvmpipe cso. Stencil + the
 * alpha test are gfx-v1 deferred (left disabled). create returns
 * llvmpipe's cso so blitter csos made before our hooks pass through. */
static void *
vp_create_dsa_state(struct pipe_context *pipe,
                    const struct pipe_depth_stencil_alpha_state *s)
{
   struct vp_context *vp = vp_reg_get(pipe);
   void *lp_cso = vp->lp_create_dsa_state(pipe, s);

   struct vp_dsa_cso *cso = CALLOC_STRUCT(vp_dsa_cso);
   if (cso) {
      cso->depth_test  = s->depth_enabled;
      cso->depth_write = s->depth_writemask;
      cso->depth_func  = vp_vx_depth_func(s->depth_func);
      vp_reg_put(lp_cso, cso);
   }
   return lp_cso;
}

static void
vp_bind_dsa_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp = vp_reg_get(pipe);
   vp->cur_dsa = p ? vp_reg_get(p) : NULL;   /* NULL for blitter csos */
   vp->lp_bind_dsa_state(pipe, p);
   if (vp->cur_dsa)
      vp_dbg("vortexpipe: OM depth test=%d func=%u write=%d",
             vp->cur_dsa->depth_test, vp->cur_dsa->depth_func,
             vp->cur_dsa->depth_write);
}

static void
vp_delete_dsa_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_dsa_cso *cso = p ? vp_reg_get(p) : NULL;
   if (cso) {
      if (vp->cur_dsa == cso)
         vp->cur_dsa = NULL;
      vp_reg_del(p);
      FREE(cso);
   }
   vp->lp_delete_dsa_state(pipe, p);
}

/* blend state: capture render-target 0's blend equation / factors and
 * the colour write mask, translated to the packed OM DCR words. */
static void *
vp_create_blend_state(struct pipe_context *pipe,
                      const struct pipe_blend_state *s)
{
   struct vp_context *vp = vp_reg_get(pipe);
   void *lp_cso = vp->lp_create_blend_state(pipe, s);

   struct vp_blend_cso *cso = CALLOC_STRUCT(vp_blend_cso);
   if (cso) {
      const struct pipe_rt_blend_state *rt = &s->rt[0];
      cso->blend_enable = rt->blend_enable;
      cso->colormask    = rt->colormask;
      if (rt->blend_enable) {
         uint32_t m = vp_vx_blend_mode(rt->rgb_func);
         cso->blend_mode = (m << 16) | (m << 0);
         cso->blend_func =
            (vp_vx_blend_factor(rt->alpha_dst_factor) << 24) |
            (vp_vx_blend_factor(rt->rgb_dst_factor)   << 16) |
            (vp_vx_blend_factor(rt->alpha_src_factor) << 8)  |
            (vp_vx_blend_factor(rt->rgb_src_factor)   << 0);
      } else {
         /* passthrough: dst = src*ONE + dst*ZERO */
         cso->blend_mode = (VX_OM_BLEND_MODE_ADD << 16) | VX_OM_BLEND_MODE_ADD;
         cso->blend_func =
            (VX_OM_BLEND_FUNC_ZERO << 24) | (VX_OM_BLEND_FUNC_ZERO << 16) |
            (VX_OM_BLEND_FUNC_ONE  << 8)  | (VX_OM_BLEND_FUNC_ONE  << 0);
      }
      /* Per-attachment blend/write-mask. With independent blend off,
       * every attachment uses RT0's equation (Vulkan default); with it on, each
       * carries its own. Slot 0 always equals the scalar fields above. */
      for (unsigned i = 0; i < GFX_OM_MAX_RT; i++) {
         const struct pipe_rt_blend_state *r =
            s->independent_blend_enable ? &s->rt[i] : &s->rt[0];
         cso->rt_colormask[i] = r->colormask;
         if (r->blend_enable) {
            uint32_t m = vp_vx_blend_mode(r->rgb_func);
            cso->rt_blend_mode[i] = (m << 16) | (m << 0);
            cso->rt_blend_func[i] =
               (vp_vx_blend_factor(r->alpha_dst_factor) << 24) |
               (vp_vx_blend_factor(r->rgb_dst_factor)   << 16) |
               (vp_vx_blend_factor(r->alpha_src_factor) << 8)  |
               (vp_vx_blend_factor(r->rgb_src_factor)   << 0);
         } else {
            cso->rt_blend_mode[i] = (VX_OM_BLEND_MODE_ADD << 16) | VX_OM_BLEND_MODE_ADD;
            cso->rt_blend_func[i] =
               (VX_OM_BLEND_FUNC_ZERO << 24) | (VX_OM_BLEND_FUNC_ZERO << 16) |
               (VX_OM_BLEND_FUNC_ONE  << 8)  | (VX_OM_BLEND_FUNC_ONE  << 0);
         }
      }
      vp_reg_put(lp_cso, cso);
   }
   return lp_cso;
}

static void
vp_bind_blend_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp = vp_reg_get(pipe);
   vp->cur_blend = p ? vp_reg_get(p) : NULL;   /* NULL for blitter csos */
   vp->lp_bind_blend_state(pipe, p);
   if (vp->cur_blend)
      vp_dbg("vortexpipe: OM blend enable=%d mode=%#x func=%#x mask=%#x",
             vp->cur_blend->blend_enable, vp->cur_blend->blend_mode,
             vp->cur_blend->blend_func, vp->cur_blend->colormask);
}

static void
vp_delete_blend_state(struct pipe_context *pipe, void *p)
{
   struct vp_context   *vp  = vp_reg_get(pipe);
   struct vp_blend_cso *cso = p ? vp_reg_get(p) : NULL;
   if (cso) {
      if (vp->cur_blend == cso)
         vp->cur_blend = NULL;
      vp_reg_del(p);
      FREE(cso);
   }
   vp->lp_delete_blend_state(pipe, p);
}

/* ---- graphics: rasterizer state (face cull) ------------------------ *
 * The device front end culls by the signed-area sign of each triangle
 * in screen space (gfx_setup.h EdgeEquation), so vortexpipe captures the
 * face-cull mode + front-face winding here and hands the pair to the
 * draw as a device SETUP_CULL_* mode. Registered under the llvmpipe cso
 * like the depth/blend csos, so blitter csos made before our hooks armed
 * pass straight through (cur_rast == NULL -> two-sided default). */
static void *
vp_create_rasterizer_state(struct pipe_context *pipe,
                           const struct pipe_rasterizer_state *s)
{
   struct vp_context *vp = vp_reg_get(pipe);
   void *lp_cso = vp->lp_create_rasterizer_state(pipe, s);

   struct vp_rast_cso *cso = CALLOC_STRUCT(vp_rast_cso);
   if (cso) {
      cso->cull_face = s->cull_face;
      cso->front_ccw = s->front_ccw;
      vp_reg_put(lp_cso, cso);
   }
   return lp_cso;
}

static void
vp_bind_rasterizer_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp = vp_reg_get(pipe);
   vp->cur_rast = p ? vp_reg_get(p) : NULL;   /* NULL for blitter csos */
   vp->lp_bind_rasterizer_state(pipe, p);
   if (vp->cur_rast)
      vp_dbg("vortexpipe: raster cull_face=%u front_ccw=%d",
             vp->cur_rast->cull_face, vp->cur_rast->front_ccw);
}

static void
vp_delete_rasterizer_state(struct pipe_context *pipe, void *p)
{
   struct vp_context  *vp  = vp_reg_get(pipe);
   struct vp_rast_cso *cso = p ? vp_reg_get(p) : NULL;
   if (cso) {
      /* cur_rast is a raw non-owning pointer. Gallium unbinds a state before
       * deleting it, so cur_rast should never still reference cso here; clear it
       * defensively (and log) so a stale binding can never dangle into a freed
       * cso on the draw path. */
      if (vp->cur_rast == cso) {
         vp_dbg("vortexpipe: deleting still-bound rasterizer cso %p", (void *)cso);
         vp->cur_rast = NULL;
      }
      vp_reg_del(p);
      FREE(cso);
   }
   vp->lp_delete_rasterizer_state(pipe, p);
}

/* ---- graphics: viewport state ------------------------------------- *
 * The device front end applies the viewport transform itself (perspective
 * divide + scale/bias in gfx_setup.h ClipToScreen/ClipToHDC), so vortexpipe
 * must carry the app's bound VkViewport through — otherwise the front end's
 * hardwired full-fb y-down transform ignores a negative-height (y-flip) or
 * offset/scaled viewport and mirrors / mis-places the triangle, which also
 * flips the signed-area sign the face cull reads. Gallium already reduces the
 * VkViewport to window-space scale/translate here; capture slot 0 (gfx-v1 is
 * single-viewport) and forward it at draw time. */
static void
vp_set_viewport_states(struct pipe_context *pipe, unsigned start_slot,
                       unsigned num_viewports,
                       const struct pipe_viewport_state *vps)
{
   struct vp_context *vp = vp_reg_get(pipe);
   if (vps && start_slot == 0 && num_viewports >= 1) {
      vp->vp_scale_x = vps[0].scale[0];
      vp->vp_trans_x = vps[0].translate[0];
      vp->vp_scale_y = vps[0].scale[1];
      vp->vp_trans_y = vps[0].translate[1];
      vp->vp_valid   = true;
      vp_dbg("vortexpipe: viewport scale=(%g,%g) translate=(%g,%g)",
             vp->vp_scale_x, vp->vp_scale_y, vp->vp_trans_x, vp->vp_trans_y);
   }
   vp->lp_set_viewport_states(pipe, start_slot, num_viewports, vps);
}

/* Translate the bound Gallium rasterizer state to the device SETUP_CULL_*
 * face-cull mode. The device culls on the signed-area sign in screen
 * space, where a triangle's winding matches the Vulkan framebuffer area
 * sign: vortexpipe runs the VS output (clip-space gl_Position) through the
 * app's captured viewport transform (vp_set_viewport_states -> the device
 * setup's scale/bias), so the front-end det sign equals the Vulkan area sign
 * — including a y-flip, which the negative sy sign-flips correctly. det>0 is
 * the front (positive-area) winding, det<0 the back; front_ccw selects which
 * of the two Vulkan calls "front".
 * PIPE_FACE_FRONT_AND_BACK (cull everything) is handled by the caller. */
static uint32_t
vp_cull_mode(const struct vp_rast_cso *r)
{
   if (!r || r->cull_face == PIPE_FACE_NONE)
      return SETUP_CULL_NONE;
   bool cull_back = (r->cull_face == PIPE_FACE_BACK);
   if (!r->front_ccw)
      cull_back = !cull_back;   /* CW front face swaps the device winding sense */
   return cull_back ? SETUP_CULL_BACK : SETUP_CULL_FRONT;
}

/* A passthrough vertex shader: copies N input attributes straight to
 * the matching output slots. vortexpipe runs the real VS on Vortex,
 * then draws the transformed vertices through this trivial VS so
 * llvmpipe's draw module still does clip / viewport / rasterization.
 * Built once and cached on the context. */
static void *
vp_get_passthrough_vs(struct vp_context *vp, struct pipe_context *pipe,
                      const struct vp_vs_layout *layout)
{
   if (vp->passthrough_vs)
      return vp->passthrough_vs;

   const struct nir_shader_compiler_options *opts =
      pipe->screen->get_compiler_options(pipe->screen, PIPE_SHADER_IR_NIR,
                                         PIPE_SHADER_VERTEX);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, opts,
                                                  "vp_passthrough_vs");

   /* attribute/slot 0 is gl_Position; slots 1.. are generic varyings. */
   unsigned n = 1 + layout->num_varyings;
   for (unsigned i = 0; i < n; i++) {
      nir_variable *iv = nir_variable_create(b.shader, nir_var_shader_in,
                                             glsl_vec4_type(), "vp_in");
      iv->data.location = VERT_ATTRIB_GENERIC0 + i;
      iv->data.driver_location = i;

      nir_variable *ov = nir_variable_create(b.shader, nir_var_shader_out,
                                             glsl_vec4_type(), "vp_out");
      ov->data.location = (i == 0) ? VARYING_SLOT_POS
                                   : layout->varying_loc[i - 1];
      ov->data.driver_location = i;

      nir_store_var(&b, ov, nir_load_var(&b, iv), 0xf);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));

   struct pipe_shader_state pss = { 0 };
   pss.type   = PIPE_SHADER_IR_NIR;
   pss.ir.nir = b.shader;
   vp->passthrough_vs = vp->lp_create_vs_state(pipe, &pss);
   return vp->passthrough_vs;
}

/* Vertex-elements describing the Vortex VS output record: one padded
 * vec4 per slot (gl_Position, then the varyings). Cached. */
static void *
vp_get_velems(struct vp_context *vp, struct pipe_context *pipe,
              const struct vp_vs_layout *layout)
{
   if (vp->velems)
      return vp->velems;

   unsigned n = 1 + layout->num_varyings;
   struct pipe_vertex_element ve[1 + VP_VS_MAX_VARYINGS];
   memset(ve, 0, sizeof ve);
   for (unsigned i = 0; i < n; i++) {
      ve[i].src_offset         = i * 16u;
      ve[i].vertex_buffer_index = 0;
      ve[i].src_format         = PIPE_FORMAT_R32G32B32A32_FLOAT;
      ve[i].src_stride         = layout->stride;
   }
   vp->velems = pipe->create_vertex_elements_state(pipe, n, ve);
   return vp->velems;
}

/* Fallback: draw the Vortex-transformed vertices through a
 * passthrough VS so llvmpipe does clip / raster / fragment / OM. */
static bool
vp_draw_passthrough(struct vp_context *vp, struct pipe_context *pipe,
                    const struct pipe_draw_info *info, unsigned drawid_offset,
                    void *xverts, uint32_t count)
{
   void *pvs    = vp_get_passthrough_vs(vp, pipe, &vp->cur_vs->vs_layout);
   void *velems = pvs ? vp_get_velems(vp, pipe, &vp->cur_vs->vs_layout) : NULL;
   if (!velems)
      return false;

   struct pipe_vertex_buffer vb = { 0 };
   vb.is_user_buffer = true;
   vb.buffer.user    = xverts;
   vp->lp_bind_vs_state(pipe, pvs);
   pipe->bind_vertex_elements_state(pipe, velems);
   pipe->set_vertex_buffers(pipe, 1, &vb);

   struct pipe_draw_start_count_bias d = {
      .start = 0, .count = count, .index_bias = 0,
   };
   vp->lp_draw_vbo(pipe, info, drawid_offset, NULL, &d, 1);
   vp->lp_bind_vs_state(pipe, vp->cur_vs->lp_cso);   /* restore the app VS */
   return true;
}

/* Fill `vin` with the bound vertex-buffer geometry so the VS kernel
 * can fetch per-vertex attributes. gfx-v1 supports a single
 * resource-backed interleaved vertex buffer; returns false (the draw
 * then falls back to llvmpipe) for anything else. On success *xfer
 * holds the buffer mapping the caller must unmap. */
static bool
vp_gather_vertex_input(struct pipe_context *pipe, struct vp_context *vp,
                       const struct pipe_draw_start_count_bias *draws,
                       bool indexed,
                       struct vp_vertex_input *vin,
                       struct pipe_transfer **xfer)
{
   struct vp_velems_cso *ve = vp->cur_velems;
   if (!ve || ve->num == 0 || ve->num > VP_MAX_ATTR || vp->num_vbufs == 0)
      return false;

   /* Multiple vertex bindings are supported as long as they all reference the
    * same underlying resource — the common case where an app binds one buffer at
    * several binding points (e.g. deqp's separate position + texcoord bindings
    * into a single vertex buffer). One upload of that resource then serves every
    * attribute; each binding's own buffer_offset is folded into the per-attribute
    * offset below. Distinct resources per binding would need multiple device
    * uploads and are not yet supported. */
   const struct pipe_vertex_buffer *vb0 = &vp->vbufs[0];
   if (vb0->is_user_buffer || !vb0->buffer.resource)
      return false;
   struct pipe_resource *res = vb0->buffer.resource;
   for (unsigned i = 0; i < ve->num; i++) {
      unsigned bi = ve->buffer_index[i];
      if (bi >= vp->num_vbufs || vp->vbufs[bi].is_user_buffer ||
          vp->vbufs[bi].buffer.resource != res)
         return false;                    /* single shared resource only */
   }

   uint8_t *map = pipe_buffer_map(pipe, res, PIPE_MAP_READ, xfer);
   if (!map)
      return false;

   vin->data        = map;
   vin->size        = res->width0;
   vin->base_offset = 0;   /* each binding's buffer_offset folded per-attribute */
   vin->num_attrs   = ve->num;
   for (unsigned i = 0; i < ve->num; i++) {
      /* the velems index is the VS input driver_location. attr_offset is the
       * attribute's absolute byte offset within the shared resource: this
       * binding's buffer_offset + the attribute's own offset. Fold the draw's
       * first-vertex offset in ONLY for a non-indexed draw, where the device
       * VS's vid is the 0-based draw position; for an INDEXED draw the vid is the
       * index value (already the absolute vertex) and draws[0].start offsets the
       * INDEX buffer (in vp_gather_index_u32), so folding it here too would
       * double-offset the attribute fetch whenever start != 0. */
      unsigned bi = ve->buffer_index[i];
      vin->attr_loc[i]    = i;
      vin->attr_offset[i] = vp->vbufs[bi].buffer_offset + ve->src_offset[i]
                          + (indexed ? 0u : draws[0].start * ve->src_stride[i]);
      vin->attr_stride[i] = ve->src_stride[i];
   }
   vp_dbg("vortexpipe: vertex-input: %u attrs, vbuf %u bytes",
          vin->num_attrs, vin->size);
   return true;
}

/* Upload the draw's index buffer to the device as a flat u32-per-vertex array
 * (widening 16-bit indices and folding in the base-vertex bias), so the VS can
 * resolve gl_VertexIndex / vertex-attribute fetch on device. Returns the device
 * address + owning buffer (caller releases). False -> caller drops to llvmpipe. */
static bool
vp_gather_index_u32(struct pipe_context *pipe, struct vp_context *vp,
                    const struct pipe_draw_info *info,
                    const struct pipe_draw_start_count_bias *draws,
                    uint64_t *out_dev, vx_buffer_h *out_buf)
{
   const unsigned isz   = info->index_size;     /* 2 or 4 */
   const unsigned count = draws[0].count;
   struct pipe_transfer *xfer = NULL;
   const uint8_t *src = NULL;

   if (info->has_user_indices) {
      src = (const uint8_t *)info->index.user;
   } else if (info->index.resource) {
      src = (const uint8_t *)pipe_buffer_map(pipe, info->index.resource,
                                             PIPE_MAP_READ, &xfer);
   }
   if (!src)
      return false;
   src += (size_t)draws[0].start * isz;

   uint32_t *u32 = (uint32_t *)malloc((size_t)count * 4u);
   if (!u32) {
      if (xfer) pipe_buffer_unmap(pipe, xfer);
      return false;
   }
   const int32_t bias = draws[0].index_bias;    /* base vertex */
   if (isz == 4) {
      const uint32_t *s = (const uint32_t *)src;
      for (unsigned i = 0; i < count; i++) u32[i] = s[i] + (uint32_t)bias;
   } else {
      const uint16_t *s = (const uint16_t *)src;
      for (unsigned i = 0; i < count; i++) u32[i] = (uint32_t)s[i] + (uint32_t)bias;
   }

   bool ok = vp_dev_upload(vp->dev, u32, (size_t)count * 4u, out_buf, out_dev);
   free(u32);
   if (xfer) pipe_buffer_unmap(pipe, xfer);
   return ok;
}

/* Topology translation: expand a triangle strip/fan into a flat u32
 * triangle-LIST index array (source vertex indices in list order) and upload it
 * to the device, so the strip/fan renders through the existing indexed
 * triangle-list device path (the front end is list-native). Handles both
 * indexed sources (indices come from the draw's index buffer) and non-indexed
 * sources (identity: source vertex = strip position). `*out_tris` is the emitted
 * triangle count; the device draw runs with count = 3*(*out_tris). Returns the
 * device address + owning buffer (caller releases). False -> drop to llvmpipe. */
static bool
vp_gather_topology_u32(struct pipe_context *pipe, struct vp_context *vp,
                       const struct pipe_draw_info *info,
                       const struct pipe_draw_start_count_bias *draws,
                       uint64_t *out_dev, vx_buffer_h *out_buf, uint32_t *out_tris)
{
   const unsigned count = draws[0].count;
   if (count < 3) { *out_tris = 0; return false; }
   const unsigned tris = count - 2;

   const bool indexed = info->index_size == 2 || info->index_size == 4;
   const int32_t bias = draws[0].index_bias;   /* base vertex */

   /* Map a strip position -> source vertex index (post-bias). */
   struct pipe_transfer *xfer = NULL;
   const uint8_t *isrc = NULL;
   if (indexed) {
      if (info->has_user_indices) {
         isrc = (const uint8_t *)info->index.user;
      } else if (info->index.resource) {
         isrc = (const uint8_t *)pipe_buffer_map(pipe, info->index.resource,
                                                 PIPE_MAP_READ, &xfer);
      }
      if (!isrc) return false;
      isrc += (size_t)draws[0].start * info->index_size;
   }
   /* Non-indexed: the device VS fetches attributes at base+vid*stride where the
    * base already folds draws[0].start (vp_gather_vertex_input), so the vid is
    * the 0-based strip position. Indexed: vid is the source index value + base
    * vertex, matching the indexed triangle-list path. */
   #define VP_SRC_VERT(pos)                                                    \
      (indexed ? (info->index_size == 4                                        \
                     ? ((const uint32_t *)isrc)[pos] + (uint32_t)bias          \
                     : (uint32_t)((const uint16_t *)isrc)[pos] + (uint32_t)bias)\
               : (uint32_t)(pos))

   uint32_t *u32 = (uint32_t *)malloc((size_t)tris * 3u * 4u);
   if (!u32) {
      if (xfer) pipe_buffer_unmap(pipe, xfer);
      return false;
   }

   uint32_t w = 0;
   if (info->mode == MESA_PRIM_TRIANGLE_STRIP) {
      for (unsigned i = 0; i < tris; i++) {
         /* alternate winding so every triangle keeps the strip's facing */
         if (i & 1u) {
            u32[w++] = VP_SRC_VERT(i + 1);
            u32[w++] = VP_SRC_VERT(i);
            u32[w++] = VP_SRC_VERT(i + 2);
         } else {
            u32[w++] = VP_SRC_VERT(i);
            u32[w++] = VP_SRC_VERT(i + 1);
            u32[w++] = VP_SRC_VERT(i + 2);
         }
      }
   } else { /* MESA_PRIM_TRIANGLE_FAN */
      for (unsigned i = 0; i < tris; i++) {
         u32[w++] = VP_SRC_VERT(0);
         u32[w++] = VP_SRC_VERT(i + 1);
         u32[w++] = VP_SRC_VERT(i + 2);
      }
   }
   #undef VP_SRC_VERT

   bool ok = vp_dev_upload(vp->dev, u32, (size_t)tris * 3u * 4u, out_buf, out_dev);
   free(u32);
   if (xfer) pipe_buffer_unmap(pipe, xfer);
   if (ok) *out_tris = tris;
   return ok;
}

static void
vp_draw_vbo(struct pipe_context *pipe,
            const struct pipe_draw_info *info,
            unsigned drawid_offset,
            const struct pipe_draw_indirect_info *indirect,
            const struct pipe_draw_start_count_bias *draws,
            unsigned num_draws)
{
   struct vp_context *vp = vp_reg_get(pipe);
   struct vp_cso     *vs = vp->cur_vs;
   struct vp_cso     *fs = vp->cur_fs;

   /* Run on Vortex for a simple direct, non-instanced single draw with a
    * translated VS. Indexed draws are supported on the hardware path: the
    * index buffer is uploaded (widened to u32) and resolved per-vertex in the
    * VS (arg slot 2), so the i-th VS thread renders index_buf[i]. Everything
    * else falls back wholly to llvmpipe. */
   bool indexed = info->index_size == 2 || info->index_size == 4;
   /* Triangle strips/fans are translated on the host into a triangle-LIST
    * index array (vp_gather_topology_u32) and run through the list-native front
    * end. Lines/points still fall back to llvmpipe (SW-raster work). */
   bool tristrip = info->mode == MESA_PRIM_TRIANGLE_STRIP ||
                   info->mode == MESA_PRIM_TRIANGLE_FAN;
   /* Instancing: a multi-instance draw runs on the device (the VS resolves
    * gl_InstanceIndex and the whole pipeline runs over instance_count ×
    * verts-per-instance vertices). Instance-RATE vertex attributes (a non-zero
    * divisor) are not fetched on device yet, so a draw that binds one falls back
    * to llvmpipe. Single-instance draws never inspect the divisor (fast path). */
   bool instance_rate_attr = false;
   if (info->instance_count > 1 && vp->cur_velems) {
      for (unsigned i = 0; i < vp->cur_velems->num; i++)
         if (vp->cur_velems->instance_divisor[i] != 0) { instance_rate_attr = true; break; }
   }
   bool simple =
      vp->dev && vs && vs->vxbin && vs->vs_layout.stride &&
      !indirect && num_draws == 1 &&
      (info->index_size == 0 || indexed) &&
      !info->primitive_restart && info->instance_count >= 1 &&
      !instance_rate_attr &&
      (info->mode == MESA_PRIM_TRIANGLES || tristrip) &&
      draws[0].count > 0;

   if (simple) {
      uint32_t count  = draws[0].count;
      uint32_t stride = vs->vs_layout.stride;
      /* the device draw is indexed whenever it consumes an index buffer: a
       * genuinely indexed source, or a strip/fan we translate to an index list. */
      bool dev_indexed = indexed || tristrip;

      /* Gather the vertex-buffer geometry if the VS fetches inputs;
       * if it needs them and we can't supply them, fall back wholly. */
      struct vp_vertex_input vin = { 0 };
      struct pipe_transfer  *vxfer = NULL;
      bool vin_ok = !vs->vs_layout.needs_vertex_input ||
                    vp_gather_vertex_input(pipe, vp, draws, indexed, &vin, &vxfer);

      /* Indexed draw: upload the index buffer (widened to u32) so the VS can
       * resolve the per-vertex index on device. Strips/fans are translated to a
       * triangle-LIST index array here (count becomes 3*tris). If the buffer
       * can't be gathered, drop the whole draw to llvmpipe. */
      uint64_t    index_dev = 0;
      vx_buffer_h ibuf      = NULL;
      bool index_ok;
      if (tristrip) {
         uint32_t tris = 0;
         index_ok = vp_gather_topology_u32(pipe, vp, info, draws,
                                           &index_dev, &ibuf, &tris);
         if (index_ok)
            count = tris * 3u;
      } else {
         index_ok = !indexed ||
                    vp_gather_index_u32(pipe, vp, info, draws, &index_dev, &ibuf);
      }
      if (!index_ok) {
         if (vxfer) pipe_buffer_unmap(pipe, vxfer);
         goto llvmpipe;
      }

      /* Decide the hardware-raster path up front (caps only, no VS needed) so
       * the supported path can fold the VS into the device draw. The hardware
       * RASTER + OM + (optional) TEX path needs the matching ISA extensions —
       * a compute-only build traps on the first vx_rast/vx_om/vx_tex — so gate
       * on the cached caps; VORTEXPIPE_SW_RASTER forces the llvmpipe fallback. */
      static int sw_raster = -1;
      if (sw_raster < 0)
         sw_raster = getenv("VORTEXPIPE_SW_RASTER") != NULL;
      struct vp_screen *vps = vp_reg_get(pipe->screen);
      /* A unit absent from the device runs in software (the FS was compiled
       * for it) rather than dropping the whole draw to llvmpipe. RASTER, OM and
       * TEX may each be HW or SW; the device path is taken as long as every unit
       * the draw needs is satisfied HW-or-SW. */
      bool fs_sw_om     = fs && fs->fs_routing.sw_om;
      bool fs_sw_raster = fs && fs->fs_routing.sw_raster;
      bool gfx_hw = vps && (vps->has_raster || fs_sw_raster)
                        && (vps->has_om     || fs_sw_om);
      bool tex_needed = vp->cur_tex != NULL;
      /* A sampler on a TEX-less device no longer drops to llvmpipe — the FS was
       * compiled to sample in software (fs_routing.sw_tex), so the device path
       * runs HW raster + (HW/SW) OM + SW TEX. Only skip if the FS was NOT built
       * for SW texturing (e.g. caps changed under a cached shader). */
      bool fs_sw_tex = fs && fs->fs_routing.sw_tex;
      if (gfx_hw && tex_needed && !vps->has_tex && !fs_sw_tex) {
         mesa_logw("vortexpipe: draw_vbo: device lacks TEX extension and FS not "
                   "compiled for SW texturing — skipping hardware RASTER+OM path");
         gfx_hw = false;
      }
      /* NPOT textures cannot be addressed by the FF TEX unit (power-of-two
       * only); they are sampled by the SW multiply-addressing path, which needs
       * an FS compiled for sw_tex (FF vx_tex4 iff POT, else SW). A
       * bound NPOT texture on a HW-TEX FS therefore drops to llvmpipe; on a
       * sw_tex FS the device path samples it in software. */
      if (gfx_hw && tex_needed && !fs_sw_tex && vp->cur_tex) {
         uint32_t tw = vp->cur_tex->width0, th = vp->cur_tex->height0;
         bool pot = tw && th && !(tw & (tw - 1)) && !(th & (th - 1));
         if (!pot) {
            mesa_logw("vortexpipe: draw_vbo: NPOT texture %ux%u needs the SW "
                      "sampler (FS compiled for HW TEX) — skipping hardware path",
                      tw, th);
            gfx_hw = false;
         }
      }
      /* Viewport transform for the device front end. Gallium already reduced
       * the app's VkViewport to window-space scale/translate (captured in
       * set_viewport_states); forward slot 0 so the device setup maps
       * clip->screen exactly as the app intends. A y-flip (scale_y<0) then
       * flips the signed-area face-cull sign correctly. Unset => the default
       * full-framebuffer y-down transform.
       *
       * The device bbox-clamp + scissor are still full-framebuffer, so only a
       * viewport whose rect equals the full framebuffer [0,W]x[0,H] (in either
       * Y direction — the y-flip case) renders correctly; an offset / partial /
       * scaled viewport would raster outside its intended rect. Those fall out
       * of the HW path and take the VS-on-Vortex + llvmpipe fallback (which
       * applies the viewport itself) — or, under MESA_VORTEX_STRICT, fail
       * rather than mis-render. Follow-up: plumb a viewport-derived scissor
       * + bbox clamp so offset/partial viewports run on the device too. */
      float vp_sx = 0.5f * (float)vp->fb_width,  vp_tx = 0.5f * (float)vp->fb_width;
      float vp_sy = 0.5f * (float)vp->fb_height, vp_ty = 0.5f * (float)vp->fb_height;
      bool vp_full_fb = true;
      if (vp->vp_valid) {
         vp_sx = vp->vp_scale_x; vp_tx = vp->vp_trans_x;
         vp_sy = vp->vp_scale_y; vp_ty = vp->vp_trans_y;
         const float W = (float)vp->fb_width, H = (float)vp->fb_height;
         const float eps = 0.5f;
         float asx = vp_sx < 0 ? -vp_sx : vp_sx, asy = vp_sy < 0 ? -vp_sy : vp_sy;
         float dtx = vp_tx - 0.5f * W, dty = vp_ty - 0.5f * H;
         dtx = dtx < 0 ? -dtx : dtx; dty = dty < 0 ? -dty : dty;
         /* rect = bias ± |scale|; full-fb iff it spans exactly [0,W]x[0,H]. */
         vp_full_fb = (asx > 0.5f * W - eps) && (asx < 0.5f * W + eps) && dtx < eps
                   && (asy > 0.5f * H - eps) && (asy < 0.5f * H + eps) && dty < eps;
         if (!vp_full_fb)
            mesa_logw("vortexpipe: draw_vbo: non-full-framebuffer viewport "
                      "(scale=(%g,%g) translate=(%g,%g), fb=%ux%u) unsupported "
                      "on the device front end (W3) — using the llvmpipe path",
                      vp_sx, vp_sy, vp_tx, vp_ty, vp->fb_width, vp->fb_height);
      }

      bool hw_path = vin_ok && !sw_raster && gfx_hw && fs && fs->vxbin &&
                     vp->fb_color && vp->fb_width && vp->fb_height && vp_full_fb;

      /* Vortex hardware raster + OM path: the VS is folded into the draw —
       * vp_raster_draw runs it as stage 0, so the whole VS→setup→bin→FF→FS
       * draw is one device-orchestrated OP_DRAW with NO host round-trip (no
       * separate host-blocking VS launch). */
      if (hw_path) {
         uint32_t w = vp->fb_width, h = vp->fb_height;

         /* Face cull. PIPE_FACE_FRONT_AND_BACK culls every triangle, so the
          * draw produces nothing — skip the device dispatch (the colour
          * buffer stays at the render-pass clear) rather than emit a device
          * "cull all" mode the front end lacks. */
         uint32_t cull_mode = vp_cull_mode(vp->cur_rast);
         if (vp->cur_rast && vp->cur_rast->cull_face == PIPE_FACE_FRONT_AND_BACK) {
            vp_dbg("vortexpipe: draw_vbo -> cull FRONT_AND_BACK, nothing drawn");
            if (vxfer) pipe_buffer_unmap(pipe, vxfer);
            if (ibuf) vx_buffer_release(ibuf);
            return;
         }

         /* gather the OM state from the bound depth/blend csos */
         struct vp_om_params om = { 0 };
         if (vp->cur_dsa) {
            om.depth_test  = vp->cur_dsa->depth_test;
            om.depth_func  = vp->cur_dsa->depth_func;
            om.depth_write = vp->cur_dsa->depth_write;
         }
         if (vp->cur_blend) {
            om.blend_mode = vp->cur_blend->blend_mode;
            om.blend_func = vp->cur_blend->blend_func;
            om.colormask  = vp->cur_blend->colormask;
         } else {
            /* no blend cso bound: passthrough (src*ONE + dst*ZERO) */
            om.blend_mode = (VX_OM_BLEND_MODE_ADD << 16)
                          | VX_OM_BLEND_MODE_ADD;
            om.blend_func = (VX_OM_BLEND_FUNC_ZERO << 24)
                          | (VX_OM_BLEND_FUNC_ZERO << 16)
                          | (VX_OM_BLEND_FUNC_ONE  << 8)
                          | (VX_OM_BLEND_FUNC_ONE  << 0);
            om.colormask  = 0xf;
         }

         /* gather the bound texture for TEX stage 0, if any. POT textures use
          * the FF vx_tex4 unit; NPOT textures reach this point only on an FS
          * compiled for sw_tex (gated above), where the SW sampler addresses
          * them via width/height. Either way carry the real integer dims. */
         struct vp_tex_params tex = { 0 };
         bool tex_used = false;
         uint32_t tw = 0, th = 0;
         /* Select the per-draw sampled texture from the FS descriptor (multi-
          * texture); no-op when the FS has no bindless handle. */
         vp_resolve_tex_from_desc(pipe, vp, fs);
         if (vp->cur_tex) {
            tw = vp->cur_tex->width0;
            th = vp->cur_tex->height0;
            if (tw && th) {
               tex.width  = tw;
               tex.height = th;
               tex.filter = vp->cur_sampler ? vp->cur_sampler->filter
                                            : VX_TEX_FILTER_POINT;
               tex.wrap_u = vp->cur_sampler ? vp->cur_sampler->wrap_u
                                            : VX_TEX_WRAP_CLAMP;
               tex.wrap_v = vp->cur_sampler ? vp->cur_sampler->wrap_v
                                            : VX_TEX_WRAP_CLAMP;
               tex.min_filter = vp->cur_sampler ? vp->cur_sampler->min_filter
                                                : VX_TEX_FILTER_POINT;
               tex.mip_enable = vp->cur_sampler ? vp->cur_sampler->mip_enable
                                                : false;
               tex.mip_linear = vp->cur_sampler ? vp->cur_sampler->mip_linear
                                                : false;
               tex_used = true;
            }
         }

         /* Resident colour/depth (cleared/initialised once per pass) + resident
          * texture: the OM renders straight into the device buffers, no per-draw
          * framebuffer round-trip or texture re-upload. */
         uint64_t color_dev = 0, depth_dev = 0, tex_dev = 0;
         bool drew = vp_fb_ensure(pipe, vp, w, h, &om, &color_dev, &depth_dev);
         if (drew && tex_used)
            drew = vp_tex_ensure(pipe, vp, vp->cur_tex, tw, th, &tex_dev,
                                 tex.mip_off);

         /* A draw whose FS writes >1 colour output AND targets >1 bound
          * attachment renders each attachment into its own resident buffer and
          * merges through gfx_om_fragment_mrt_sw (SW OM). num_color is the min of
          * what the shader writes and what the framebuffer binds. */
         struct vp_mrt_params mrt;
         memset(&mrt, 0, sizeof mrt);
         unsigned num_color = fs ? fs->fs_num_color : 1;
         if (num_color > GFX_OM_MAX_RT) num_color = GFX_OM_MAX_RT;
         /* The FS kernel's MRT branch is compiled for exactly its colour-output
          * count; if the framebuffer binds fewer attachments the device kernel
          * would dereference an unbound RT. Vulkan discards the surplus outputs,
          * so run this draw on llvmpipe rather than launch a mismatched kernel. */
         if (num_color > 1 && (unsigned)vp->fb_nr_cbufs < num_color) {
            if (vxfer) pipe_buffer_unmap(pipe, vxfer);
            if (ibuf)  vx_buffer_release(ibuf);
            goto llvmpipe;
         }
         bool use_mrt = drew && num_color > 1;
         if (use_mrt) {
            mrt.num_color    = num_color;
            mrt.color_dev[0] = color_dev;   /* RT0 shares the resident colour buf */
            drew = vp_fb_ensure_mrt(pipe, vp, w, h, num_color, mrt.color_dev);
            for (unsigned k = 0; k < num_color; k++) {
               mrt.pitch[k] = w * 4;
               if (vp->cur_blend) {
                  mrt.blend_mode[k] = vp->cur_blend->rt_blend_mode[k];
                  mrt.blend_func[k] = vp->cur_blend->rt_blend_func[k];
                  mrt.colormask[k]  = vp->cur_blend->rt_colormask[k];
               } else {
                  mrt.blend_mode[k] = (VX_OM_BLEND_MODE_ADD << 16) | VX_OM_BLEND_MODE_ADD;
                  mrt.blend_func[k] = (VX_OM_BLEND_FUNC_ZERO << 24)
                                    | (VX_OM_BLEND_FUNC_ZERO << 16)
                                    | (VX_OM_BLEND_FUNC_ONE  << 8)
                                    | (VX_OM_BLEND_FUNC_ONE  << 0);
                  mrt.colormask[k]  = 0xf;
               }
            }
         }
         /* Gather the fragment stage's bound constant buffers (push
          * constants at index 0, descriptor set-0 blob at 1, UBOs at their
          * bound index) so vp_raster_draw can upload them + build the resident
          * FS descriptor table. Mapped only across the synchronous draw. */
         struct vp_fs_consts fs_consts;
         memset(&fs_consts, 0, sizeof(fs_consts));
         fs_consts.descs     = fs->descs;
         fs_consts.num_descs = fs->num_descs;
         struct pipe_transfer *cbxfer[GFX_FS_DESC_SLOTS] = { NULL };
         for (unsigned i = 0; i < GFX_FS_DESC_SLOTS; i++) {
            if (!vp->fs_cbuf[i] || !vp->fs_cbuf_sz[i])
               continue;
            void *m = pipe_buffer_map(pipe, vp->fs_cbuf[i], PIPE_MAP_READ,
                                      &cbxfer[i]);
            if (!m) { cbxfer[i] = NULL; continue; }
            fs_consts.data[i] = (const uint8_t *)m + vp->fs_cbuf_off[i];
            fs_consts.size[i] = vp->fs_cbuf_sz[i];
         }
         if (drew)
            drew = vp_raster_draw(vp->dev, vp->raster_pool,
                                  vs->vxbin, vs->vxbin_size,
                                  &vs->vx_module, &vs->vx_kernel,
                                  fs->vxbin, fs->vxbin_size,
                                  &fs->vx_module, &fs->vx_kernel,
                                  count, info->instance_count, info->start_instance,
                                  &vs->vs_layout,
                                  vs->vs_layout.needs_vertex_input ? &vin : NULL,
                                  index_dev,
                                  color_dev, depth_dev, w, h, &om,
                                  tex_used ? tex_dev : 0, tex_used ? &tex : NULL,
                                  cull_mode, fs_sw_tex, fs_sw_om, fs_sw_raster,
                                  vp_sx, vp_tx, vp_sy, vp_ty,
                                  &fs_consts, use_mrt ? &mrt : NULL);
         for (unsigned i = 0; i < GFX_FS_DESC_SLOTS; i++)
            if (cbxfer[i]) pipe_buffer_unmap(pipe, cbxfer[i]);
         if (drew) {
            vp->rfb_dirty = true;   /* device colour ahead of the resource */
            vp->draws_device++;
            vp_dbg("vortexpipe: draw_vbo -> Vortex VS+RASTER+OM (one OP_DRAW) "
                   "(%u verts, %ux%u, depth_test=%d, textured=%d)",
                   count, w, h, om.depth_test, tex_used);
            if (vxfer) pipe_buffer_unmap(pipe, vxfer);
            if (ibuf) vx_buffer_release(ibuf);
            return;
         }
         /* hw path failed → fall through to VS-on-Vortex + llvmpipe raster */
      }

      /* Taking an llvmpipe path: flush any resident hw renders back to the
       * colour resource (so llvmpipe composites on top) and drop the resident
       * buffers (the next hw draw re-initialises from the resource). */
      vp_fb_invalidate(pipe, vp);

      /* fallback: run the VS on Vortex as a standalone launch, read its output
       * back to a host vertex buffer and rasterize on llvmpipe (unsupported
       * state / no hardware raster). Indexed draws skip this path — the
       * standalone VS launch does not resolve the index buffer — and drop
       * straight to the llvmpipe indexed draw. Instanced draws also skip it (the
       * standalone launch runs one instance only); llvmpipe handles instancing. */
      if (vin_ok && !dev_indexed && info->instance_count == 1) {
         vx_buffer_h vsbuf  = NULL;
         uint64_t    vsaddr = 0;
         if (vp_launch_vs(vp->dev, vs->vxbin, vs->vxbin_size,
                          count, count * stride,
                          vs->vs_layout.needs_vertex_input ? &vin : NULL,
                          &vsbuf, &vsaddr)) {
            void *xverts = malloc((size_t)count * stride);
            if (xverts &&
                vp_buffer_readback(vp->dev, vsbuf, xverts,
                                   (uint32_t)((size_t)count * stride)) &&
                vp_draw_passthrough(vp, pipe, info, drawid_offset,
                                    xverts, count)) {
               vp_dbg("vortexpipe: draw_vbo ran the %u-vertex VS on Vortex "
                      "(llvmpipe raster)", count);
               free(xverts);
               vx_buffer_release(vsbuf);
               if (vxfer) pipe_buffer_unmap(pipe, vxfer);
               return;
            }
            free(xverts);
            if (vsbuf) vx_buffer_release(vsbuf);
         }
      }
      if (vxfer)
         pipe_buffer_unmap(pipe, vxfer);
      if (ibuf) vx_buffer_release(ibuf);
   }

llvmpipe:
   /* inherit-and-accelerate fallback */
   vp->draws_cpu++;
   if (vp_strict_mode()) {
      mesa_loge("vortexpipe: draw_vbo: Vortex path unavailable, "
                "STRICT mode refuses llvmpipe fallback");
      return;
   }
   /* sync + drop any resident hw renders before llvmpipe draws into the FB. */
   vp_fb_invalidate(pipe, vp);
   vp->lp_draw_vbo(pipe, info, drawid_offset, indirect, draws, num_draws);
}

/* present / sync point: copy any resident hw renders back to the colour
 * resource before llvmpipe flushes, so a present / readback observes them. */
static void
vp_flush(struct pipe_context *pipe, struct pipe_fence_handle **fence,
         unsigned flags)
{
   struct vp_context *vp = vp_reg_get(pipe);
   vp_fb_sync_out(pipe, vp);
   vp->lp_flush(pipe, fence, flags);
}

static void
vp_context_destroy(struct pipe_context *pipe)
{
   struct vp_context *vp = vp_reg_get(pipe);
   void (*lp_destroy)(struct pipe_context *) = vp->lp_context_destroy;

   /* Work placement for this context, in one machine-readable line the smoke
    * harness greps: <on device>/<total> for each of dispatch and draw. */
   if (getenv("VORTEXPIPE_ATTRIB"))
      fprintf(stderr, "vortexpipe-attrib: launches=%u/%u draws=%u/%u\n",
              vp->launches_device, vp->launches_device + vp->launches_cpu,
              vp->draws_device, vp->draws_device + vp->draws_cpu);

   /* release the cached draw-integration objects (need a
    * live pipe) */
   if (vp->passthrough_vs)
      vp->lp_delete_vs_state(pipe, vp->passthrough_vs);
   if (vp->velems)
      pipe->delete_vertex_elements_state(pipe, vp->velems);

   /* Residency: flush + release the resident framebuffer + texture. */
   vp_fb_invalidate(pipe, vp);
   if (vp->rtex_buf) { vx_buffer_release(vp->rtex_buf); vp->rtex_buf = NULL; }

   /* release the persistent front-end pool's device buffers (the screen
    * still holds the device open until its own teardown). */
   vp_raster_pool_destroy(vp->raster_pool);

   /* llvmpipe_destroy tears down util_blitter, which calls back into
    * our delete_*_state hooks -- and those need `vp`. So destroy the
    * llvmpipe context first, then drop vortexpipe's own state. */
   lp_destroy(pipe);
   vp_reg_del(pipe);
   FREE(vp);
}

/* ---- context creation ----------------------------------------------- */

struct pipe_context *
vp_context_create(struct pipe_screen *screen, void *priv, unsigned flags)
{
   struct vp_screen *vps = vp_reg_get(screen);
   struct pipe_context *pipe = vps->lp_context_create(screen, priv, flags);
   if (!pipe)
      return NULL;

   struct vp_context *vp = CALLOC_STRUCT(vp_context);
   if (!vp)
      return pipe;          /* degrade to plain llvmpipe context */

   vp->dev                     = vps->dev;
   vp->raster_pool             = vp_raster_pool_create();
   vp->lp_create_compute_state = pipe->create_compute_state;
   vp->lp_bind_compute_state   = pipe->bind_compute_state;
   vp->lp_delete_compute_state = pipe->delete_compute_state;
   vp->lp_launch_grid          = pipe->launch_grid;
   vp->lp_set_constant_buffer  = pipe->set_constant_buffer;
   vp->lp_set_shader_buffers   = pipe->set_shader_buffers;
   vp->lp_create_vs_state      = pipe->create_vs_state;
   vp->lp_bind_vs_state        = pipe->bind_vs_state;
   vp->lp_delete_vs_state      = pipe->delete_vs_state;
   vp->lp_draw_vbo             = pipe->draw_vbo;
   vp->lp_create_fs_state      = pipe->create_fs_state;
   vp->lp_bind_fs_state        = pipe->bind_fs_state;
   vp->lp_delete_fs_state      = pipe->delete_fs_state;
   vp->lp_set_framebuffer_state = pipe->set_framebuffer_state;
   vp->lp_create_dsa_state     = pipe->create_depth_stencil_alpha_state;
   vp->lp_bind_dsa_state       = pipe->bind_depth_stencil_alpha_state;
   vp->lp_delete_dsa_state     = pipe->delete_depth_stencil_alpha_state;
   vp->lp_create_blend_state   = pipe->create_blend_state;
   vp->lp_bind_blend_state     = pipe->bind_blend_state;
   vp->lp_delete_blend_state   = pipe->delete_blend_state;
   vp->lp_create_rasterizer_state = pipe->create_rasterizer_state;
   vp->lp_bind_rasterizer_state   = pipe->bind_rasterizer_state;
   vp->lp_delete_rasterizer_state = pipe->delete_rasterizer_state;
   vp->lp_set_viewport_states  = pipe->set_viewport_states;
   vp->lp_create_texture_handle = pipe->create_texture_handle;
   vp->lp_create_vertex_elements_state = pipe->create_vertex_elements_state;
   vp->lp_bind_vertex_elements_state   = pipe->bind_vertex_elements_state;
   vp->lp_delete_vertex_elements_state = pipe->delete_vertex_elements_state;
   vp->lp_set_vertex_buffers   = pipe->set_vertex_buffers;
   vp->lp_flush                = pipe->flush;
   vp->lp_context_destroy      = pipe->destroy;
   vp_reg_put(pipe, vp);

   pipe->create_compute_state = vp_create_compute_state;
   pipe->bind_compute_state   = vp_bind_compute_state;
   pipe->delete_compute_state = vp_delete_compute_state;
   pipe->launch_grid          = vp_launch_grid;
   pipe->set_constant_buffer  = vp_set_constant_buffer;
   pipe->set_shader_buffers   = vp_set_shader_buffers;
   pipe->create_vs_state      = vp_create_vs_state;
   pipe->bind_vs_state        = vp_bind_vs_state;
   pipe->delete_vs_state      = vp_delete_vs_state;
   pipe->draw_vbo             = vp_draw_vbo;
   pipe->create_fs_state      = vp_create_fs_state;
   pipe->bind_fs_state        = vp_bind_fs_state;
   pipe->delete_fs_state      = vp_delete_fs_state;
   pipe->set_framebuffer_state = vp_set_framebuffer_state;
   pipe->flush                 = vp_flush;
   pipe->create_depth_stencil_alpha_state = vp_create_dsa_state;
   pipe->bind_depth_stencil_alpha_state   = vp_bind_dsa_state;
   pipe->delete_depth_stencil_alpha_state = vp_delete_dsa_state;
   pipe->create_blend_state   = vp_create_blend_state;
   pipe->bind_blend_state     = vp_bind_blend_state;
   pipe->delete_blend_state   = vp_delete_blend_state;
   pipe->create_rasterizer_state = vp_create_rasterizer_state;
   pipe->bind_rasterizer_state   = vp_bind_rasterizer_state;
   pipe->delete_rasterizer_state = vp_delete_rasterizer_state;
   pipe->set_viewport_states   = vp_set_viewport_states;
   pipe->create_texture_handle = vp_create_texture_handle;
   pipe->create_vertex_elements_state = vp_create_vertex_elements_state;
   pipe->bind_vertex_elements_state   = vp_bind_vertex_elements_state;
   pipe->delete_vertex_elements_state = vp_delete_vertex_elements_state;
   pipe->set_vertex_buffers   = vp_set_vertex_buffers;
   pipe->destroy              = vp_context_destroy;

   vp_dbg("vortexpipe: context created -- compute hooks intercepted");
   return pipe;
}
