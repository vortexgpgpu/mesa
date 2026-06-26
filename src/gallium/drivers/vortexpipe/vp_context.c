/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vortexpipe context interception.
 *
 * vp_context_create() wraps llvmpipe's context: it records the
 * llvmpipe originals and installs vortexpipe overrides for the
 * compute hooks. For Phase 2 increment #1 the overrides simply
 * forward to llvmpipe -- they establish the interception point.
 * Subsequent increments fill them in:
 *   - create_compute_state : NIR -> SPIR-V -> .vxbin   (#2, #3)
 *   - launch_grid          : vx_enqueue_launch          (#4)
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vp_private.h"
#include "vp_nir_to_llvm.h"
#include "vp_compile.h"
#include "vp_launch.h"
#include "vp_raster.h"

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
      /* shared-memory size -> the launch's local-memory allocation. */
      cso->lmem_size = ((struct nir_shader *)state->prog)->info.shared_size;
      /* the set-0 descriptors the kernel reaches -> launch relocation. */
      vp_scan_descriptors((struct nir_shader *)state->prog,
                          cso->descs, &cso->num_descs);
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
   if (vps && vps->hw_max_block_size != 0 &&
       block_size > vps->hw_max_block_size) {
      mesa_logw("vortexpipe: launch_grid: workgroup size %u (%ux%ux%u) "
                "exceeds device cap %u (%u threads × %u warps); "
                "fallback to llvmpipe",
                block_size,
                info->block[0], info->block[1], info->block[2],
                vps->hw_max_block_size,
                vps->hw_num_threads, vps->hw_num_warps);
      goto fallback;
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
         vp_dbg("vortexpipe: launch_grid -> Vortex grid=[%u,%u,%u] "
                "block=[%u,%u,%u] descs=%u",
                info->grid[0], info->grid[1], info->grid[2],
                info->block[0], info->block[1], info->block[2],
                cso->num_descs);
         ran_on_vortex = vp_launch(vp->dev, cso->vxbin, cso->vxbin_size,
                                   (uint8_t *)desc_host + vp->cbuf_off[1],
                                   desc_bytes, cso->descs, cso->num_descs,
                                   info->grid, info->block, cso->lmem_size,
                                   vps && vps->has_rtu);
         pipe_buffer_unmap(pipe, xfer);
      }
   }

   if (ran_on_vortex) {
      vp_dbg("vortexpipe: launch_grid ran on Vortex");
      return;
   }

fallback:
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
   vp->lp_set_constant_buffer(pipe, shader, index, take_ownership, cb);
}

static void
vp_set_shader_buffers(struct pipe_context *pipe, enum pipe_shader_type shader,
                      unsigned start, unsigned count,
                      const struct pipe_shader_buffer *bufs,
                      unsigned writable_mask)
{
   struct vp_context *vp = vp_reg_get(pipe);
   vp->lp_set_shader_buffers(pipe, shader, start, count, bufs, writable_mask);
}

/* ---- graphics: vertex-shader hooks (Phase 3) ------------------------ */

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

/* §6.6 residency: release a CSO's device-resident module (its vxbin loaded
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

/* ---- graphics: fragment-shader hooks (Phase 4) --------------------- */

/* §5 per-unit HW-vs-SW routing for a fragment shader, from device caps +
 * the VORTEXPIPE_FORCE_SW knob. A unit absent from the device routes to its
 * SIMT software path (never llvmpipe — charter pillar 4). TEX is wired here;
 * OM/RASTER follow in later steps. */
static struct vp_sw_routing
vp_fs_routing(struct pipe_context *pipe)
{
   struct vp_screen *vps = vp_reg_get(pipe->screen);
   struct vp_sw_routing r = { false, false, false };
   const char *force = getenv("VORTEXPIPE_FORCE_SW");
   bool force_tex = force && (strstr(force, "tex") || strstr(force, "all"));
   r.sw_tex = (vps && !vps->has_tex) || force_tex;
   return r;
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

   if (state->type == PIPE_SHADER_IR_NIR) {
      char *ir = NULL;
      if (vp_nir_to_llvm((struct nir_shader *)state->ir.nir, &ir, NULL,
                         &cso->fs_routing)) {
         bool uses_sw = cso->fs_routing.sw_tex || cso->fs_routing.sw_om;
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

/* ---- graphics: texture-sampler state (Phase 6) --------------------- */

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
   }
   if (state) {
      vp->cur_sampler_store.filter = vp_vx_filter(state->mag_img_filter);
      vp->cur_sampler_store.wrap_u = vp_vx_wrap(state->wrap_s);
      vp->cur_sampler_store.wrap_v = vp_vx_wrap(state->wrap_t);
      vp->cur_sampler = &vp->cur_sampler_store;
      vp_dbg("vortexpipe: TEX sampler captured filter=%u wrap=%u,%u",
             vp->cur_sampler->filter, vp->cur_sampler->wrap_u,
             vp->cur_sampler->wrap_v);
   }
   return vp->lp_create_texture_handle(pipe, view, state);
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
         cso->src_offset[i]   = elements[i].src_offset;
         cso->src_stride[i]   = elements[i].src_stride;
         cso->buffer_index[i] = elements[i].vertex_buffer_index;
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

/* ---- graphics: framebuffer interception (Phase 4) ------------------ */

/* §6.6 residency: defined below, used by set_framebuffer_state to flush the
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

/* ---- §6.6 framebuffer + texture residency ------------------------------ *
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

/* Ensure the bound texture is uploaded + resident, keyed by its resource. The
 * mip-0 image is read back to a tight A8R8G8B8 host buffer and uploaded once;
 * a re-bind of the same resource reuses it. Returns the device address. */
static bool
vp_tex_ensure(struct pipe_context *pipe, struct vp_context *vp,
              struct pipe_resource *res, uint32_t w, uint32_t h,
              uint64_t *tex_dev)
{
   if (vp->rtex_buf && vp->rtex_res == res &&
       vp->rtex_w == w && vp->rtex_h == h)
      return vx_buffer_address(vp->rtex_buf, tex_dev) == VX_SUCCESS;

   if (vp->rtex_buf) { vx_buffer_release(vp->rtex_buf); vp->rtex_buf = NULL; }
   vp->rtex_res = NULL;

   const uint32_t texel_count = w * h;
   uint8_t *px = malloc((size_t)texel_count * 4);
   uint32_t *texbuf = malloc((size_t)texel_count * 4);
   bool ok = px && texbuf && vp_resource_rw(pipe, res, w, h, px, false);
   if (ok) {
      for (uint32_t i = 0; i < texel_count; i++) {
         uint32_t r = px[i * 4 + 0], g = px[i * 4 + 1];
         uint32_t b = px[i * 4 + 2], a = px[i * 4 + 3];
         texbuf[i] = (a << 24) | (r << 16) | (g << 8) | b;
      }
      ok = vp_dev_upload(vp->dev, texbuf, (size_t)texel_count * 4,
                         &vp->rtex_buf, tex_dev);
   }
   free(px);
   free(texbuf);
   if (!ok) return false;
   vp->rtex_res = res;
   vp->rtex_w = w; vp->rtex_h = h;
   return true;
}

/* ---- graphics: output-merger state (Phase 5) ----------------------- */

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

/* Phase 3 fallback: draw the Vortex-transformed vertices through a
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
                       struct vp_vertex_input *vin,
                       struct pipe_transfer **xfer)
{
   struct vp_velems_cso *ve = vp->cur_velems;
   if (!ve || ve->num == 0 || ve->num > VP_MAX_ATTR || vp->num_vbufs == 0)
      return false;
   for (unsigned i = 0; i < ve->num; i++)
      if (ve->buffer_index[i] != 0)        /* single vertex buffer only */
         return false;

   const struct pipe_vertex_buffer *vb = &vp->vbufs[0];
   if (vb->is_user_buffer || !vb->buffer.resource)
      return false;

   struct pipe_resource *res = vb->buffer.resource;
   uint8_t *map = pipe_buffer_map(pipe, res, PIPE_MAP_READ, xfer);
   if (!map)
      return false;

   vin->data        = map;
   vin->size        = res->width0;
   vin->base_offset = vb->buffer_offset;
   vin->num_attrs   = ve->num;
   for (unsigned i = 0; i < ve->num; i++) {
      /* the velems index is the VS input driver_location; fold the
       * draw's first-vertex offset into the attribute base. */
      vin->attr_loc[i]    = i;
      vin->attr_offset[i] = ve->src_offset[i]
                          + draws[0].start * ve->src_stride[i];
      vin->attr_stride[i] = ve->src_stride[i];
   }
   vp_dbg("vortexpipe: vertex-input: %u attrs, vbuf %u bytes",
          vin->num_attrs, vin->size);
   return true;
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

   /* Run on Vortex only for a simple direct, non-indexed, non-
    * instanced single draw with a translated VS; everything else
    * falls back wholly to llvmpipe (§4.5). */
   bool simple =
      vp->dev && vs && vs->vxbin && vs->vs_layout.stride &&
      !indirect && num_draws == 1 && info->index_size == 0 &&
      !info->primitive_restart && info->instance_count == 1 &&
      draws[0].count > 0;

   if (simple) {
      uint32_t count  = draws[0].count;
      uint32_t stride = vs->vs_layout.stride;

      /* Gather the vertex-buffer geometry if the VS fetches inputs;
       * if it needs them and we can't supply them, fall back wholly. */
      struct vp_vertex_input vin = { 0 };
      struct pipe_transfer  *vxfer = NULL;
      bool vin_ok = !vs->vs_layout.needs_vertex_input ||
                    vp_gather_vertex_input(pipe, vp, draws, &vin, &vxfer);

      /* Decide the hardware-raster path up front (caps only, no VS needed) so
       * the supported path can fold the VS into the device draw. The hardware
       * RASTER + OM + (optional) TEX path needs the matching ISA extensions —
       * a compute-only build traps on the first vx_rast/vx_om/vx_tex — so gate
       * on the cached caps; VORTEXPIPE_SW_RASTER forces the llvmpipe fallback. */
      static int sw_raster = -1;
      if (sw_raster < 0)
         sw_raster = getenv("VORTEXPIPE_SW_RASTER") != NULL;
      struct vp_screen *vps = vp_reg_get(pipe->screen);
      bool gfx_hw = vps && vps->has_raster && vps->has_om;
      bool tex_needed = vp->cur_tex != NULL;
      /* §5: a sampler on a TEX-less device no longer drops the whole draw to
       * llvmpipe — the FS was compiled to sample in software (fs_routing.sw_tex),
       * so the device path runs HW raster+OM + SW TEX. Only skip if the FS was
       * NOT built for SW texturing (e.g. caps changed under a cached shader). */
      bool fs_sw_tex = fs && fs->fs_routing.sw_tex;
      if (gfx_hw && tex_needed && !vps->has_tex && !fs_sw_tex) {
         mesa_logw("vortexpipe: draw_vbo: device lacks TEX extension and FS not "
                   "compiled for SW texturing — skipping hardware RASTER+OM path");
         gfx_hw = false;
      }
      bool hw_path = vin_ok && !sw_raster && gfx_hw && fs && fs->vxbin &&
                     vp->fb_color && vp->fb_width && vp->fb_height;

      /* Vortex hardware raster + OM path: the VS is folded into the draw —
       * vp_raster_draw runs it as stage 0, so the whole VS→setup→bin→FF→FS
       * draw is one device-orchestrated OP_DRAW with NO host round-trip (no
       * separate host-blocking VS launch). */
      if (hw_path) {
         uint32_t w = vp->fb_width, h = vp->fb_height;

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

         /* gather the bound texture for TEX stage 0, if any (power-of-two 2D). */
         struct vp_tex_params tex = { 0 };
         bool tex_used = false;
         uint32_t tw = 0, th = 0;
         if (vp->cur_tex) {
            tw = vp->cur_tex->width0;
            th = vp->cur_tex->height0;
            if (tw && th && !(tw & (tw - 1)) && !(th & (th - 1))) {
               tex.width  = tw;
               tex.height = th;
               tex.filter = vp->cur_sampler ? vp->cur_sampler->filter
                                            : VX_TEX_FILTER_POINT;
               tex.wrap_u = vp->cur_sampler ? vp->cur_sampler->wrap_u
                                            : VX_TEX_WRAP_CLAMP;
               tex.wrap_v = vp->cur_sampler ? vp->cur_sampler->wrap_v
                                            : VX_TEX_WRAP_CLAMP;
               tex_used = true;
            } else {
               mesa_logw("vortexpipe: draw_vbo: non-power-of-two "
                         "texture %ux%u unsupported (gfx-v1)", tw, th);
            }
         }

         /* Resident colour/depth (cleared/initialised once per pass) + resident
          * texture: the OM renders straight into the device buffers, no per-draw
          * framebuffer round-trip or texture re-upload. */
         uint64_t color_dev = 0, depth_dev = 0, tex_dev = 0;
         bool drew = vp_fb_ensure(pipe, vp, w, h, &om, &color_dev, &depth_dev);
         if (drew && tex_used)
            drew = vp_tex_ensure(pipe, vp, vp->cur_tex, tw, th, &tex_dev);
         if (drew)
            drew = vp_raster_draw(vp->dev, vp->raster_pool,
                                  vs->vxbin, vs->vxbin_size,
                                  &vs->vx_module, &vs->vx_kernel,
                                  fs->vxbin, fs->vxbin_size,
                                  &fs->vx_module, &fs->vx_kernel,
                                  count, &vs->vs_layout,
                                  vs->vs_layout.needs_vertex_input ? &vin : NULL,
                                  color_dev, depth_dev, w, h, &om,
                                  tex_used ? tex_dev : 0, tex_used ? &tex : NULL,
                                  fs_sw_tex);
         if (drew) {
            vp->rfb_dirty = true;   /* device colour ahead of the resource */
            vp_dbg("vortexpipe: draw_vbo -> Vortex VS+RASTER+OM (one OP_DRAW) "
                   "(%u verts, %ux%u, depth_test=%d, textured=%d)",
                   count, w, h, om.depth_test, tex_used);
            if (vxfer) pipe_buffer_unmap(pipe, vxfer);
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
       * state / no hardware raster). */
      if (vin_ok) {
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
   }

   /* inherit-and-accelerate fallback */
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

   /* release the cached Phase 3 draw-integration objects (need a
    * live pipe) */
   if (vp->passthrough_vs)
      vp->lp_delete_vs_state(pipe, vp->passthrough_vs);
   if (vp->velems)
      pipe->delete_vertex_elements_state(pipe, vp->velems);

   /* §6.6 residency: flush + release the resident framebuffer + texture. */
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
   pipe->create_texture_handle = vp_create_texture_handle;
   pipe->create_vertex_elements_state = vp_create_vertex_elements_state;
   pipe->bind_vertex_elements_state   = vp_bind_vertex_elements_state;
   pipe->delete_vertex_elements_state = vp_delete_vertex_elements_state;
   pipe->set_vertex_buffers   = vp_set_vertex_buffers;
   pipe->destroy              = vp_context_destroy;

   vp_dbg("vortexpipe: context created -- compute hooks intercepted");
   return pipe;
}
