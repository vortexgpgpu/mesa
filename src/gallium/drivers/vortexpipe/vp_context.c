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
      if (vp_nir_to_llvm((struct nir_shader *)state->prog, &ir, NULL)) {
         if (vp_compile_vxbin(ir, &cso->vxbin, &cso->vxbin_size))
            vp_dbg("vortexpipe: compiled shader -> %zu-byte .vxbin",
                      cso->vxbin_size);
         else
            mesa_logw("vortexpipe: .vxbin compile failed; "
                      "shader runs on llvmpipe");
         vp_free_ir(ir);
      } else {
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

static void
vp_launch_grid(struct pipe_context *pipe, const struct pipe_grid_info *info)
{
   struct vp_context *vp = vp_reg_get(pipe);
   bool ran_on_vortex = false;

   /* Run on Vortex when we have a compiled kernel and lavapipe's
    * descriptor buffer for set 0 (constant-buffer index 1). */
   if (vp->dev && vp->cur_cso && vp->cur_cso->vxbin && vp->cbuf[1]) {
      struct pipe_transfer *xfer = NULL;
      void *desc_host = pipe_buffer_map(pipe, vp->cbuf[1],
                                        PIPE_MAP_READ, &xfer);
      if (desc_host) {
         /* lp_descriptor[0]: a struct lp_jit_buffer -- offset 0 is the
          * host data pointer, offset 8 the buffer size in bytes. */
         const uint8_t *d = (const uint8_t *)desc_host + vp->cbuf_off[1];
         uint64_t ssbo_host  = *(const uint64_t *)(d + 0);
         uint32_t ssbo_bytes = *(const uint32_t *)(d + 8);
         pipe_buffer_unmap(pipe, xfer);
         if (ssbo_host && ssbo_bytes) {
            vp_dbg("vortexpipe: launch_grid -> Vortex "
                      "grid=[%u,%u,%u] block=[%u,%u,%u] ssbo=%uB",
                      info->grid[0], info->grid[1], info->grid[2],
                      info->block[0], info->block[1], info->block[2],
                      ssbo_bytes);
            ran_on_vortex = vp_launch(vp->dev,
                                      vp->cur_cso->vxbin,
                                      vp->cur_cso->vxbin_size,
                                      (void *)(uintptr_t)ssbo_host,
                                      ssbo_bytes, info->grid, info->block);
         }
      }
   }

   if (ran_on_vortex)
      vp_dbg("vortexpipe: launch_grid ran on Vortex");
   else
      vp->lp_launch_grid(pipe, info);   /* inherit-and-accelerate fallback */
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
                         &cso->vs_layout)) {
         if (vp_compile_vxbin(ir, &cso->vxbin, &cso->vxbin_size))
            vp_dbg("vortexpipe: compiled vertex shader -> %zu-byte .vxbin",
                   cso->vxbin_size);
         else
            mesa_logw("vortexpipe: VS .vxbin compile failed; "
                      "vertex stage runs on llvmpipe");
         vp_free_ir(ir);
      } else {
         mesa_logw("vortexpipe: VS NIR->LLVM unavailable; "
                   "vertex stage runs on llvmpipe");
      }
   }
   return cso;
}

static void
vp_bind_vs_state(struct pipe_context *pipe, void *p)
{
   struct vp_context *vp  = vp_reg_get(pipe);
   struct vp_cso     *cso = p;
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
   vp->lp_delete_vs_state(pipe, cso->lp_cso);
   vp_free_blob(cso->vxbin);
   FREE(cso);
}

/* ---- graphics: fragment-shader hooks (Phase 4) --------------------- */

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

   if (state->type == PIPE_SHADER_IR_NIR) {
      char *ir = NULL;
      if (vp_nir_to_llvm((struct nir_shader *)state->ir.nir, &ir, NULL)) {
         if (vp_compile_vxbin(ir, &cso->vxbin, &cso->vxbin_size))
            vp_dbg("vortexpipe: compiled fragment shader -> %zu-byte .vxbin",
                   cso->vxbin_size);
         else
            mesa_logw("vortexpipe: FS .vxbin compile failed");
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
   vp->lp_delete_fs_state(pipe, cso->lp_cso);
   vp_free_blob(cso->vxbin);
   FREE(cso);
}

/* ---- graphics: framebuffer interception (Phase 4) ------------------ */

/* Capture colour attachment 0 so the Vortex raster path can round-trip
 * it; everything else stays with llvmpipe. */
static void
vp_set_framebuffer_state(struct pipe_context *pipe,
                         const struct pipe_framebuffer_state *fb)
{
   struct vp_context *vp = vp_reg_get(pipe);
   vp->fb_color  = (fb && fb->nr_cbufs > 0 && fb->cbufs[0])
                      ? fb->cbufs[0]->texture : NULL;
   vp->fb_width  = fb ? fb->width  : 0;
   vp->fb_height = fb ? fb->height : 0;
   vp->lp_set_framebuffer_state(pipe, fb);
}

/* Copy colour attachment 0 into a tight w*h R8G8B8A8 host buffer. The
 * Vortex fragment kernel renders into a linear buffer, so the draw
 * path round-trips: read the (llvmpipe-cleared) attachment, run the
 * raster path, write the result back with vp_fb_color_write. */
static bool
vp_fb_color_read(struct pipe_context *pipe, struct vp_context *vp, void *dst)
{
   if (!vp->fb_color)
      return false;
   struct pipe_transfer *xfer = NULL;
   uint8_t *map = pipe_texture_map(pipe, vp->fb_color, 0, 0, PIPE_MAP_READ,
                                   0, 0, vp->fb_width, vp->fb_height, &xfer);
   if (!map)
      return false;
   for (unsigned y = 0; y < vp->fb_height; y++)
      memcpy((uint8_t *)dst + (size_t)y * vp->fb_width * 4,
             map + (size_t)y * xfer->stride, (size_t)vp->fb_width * 4);
   pipe_texture_unmap(pipe, xfer);
   return true;
}

static bool
vp_fb_color_write(struct pipe_context *pipe, struct vp_context *vp,
                  const void *src)
{
   if (!vp->fb_color)
      return false;
   struct pipe_transfer *xfer = NULL;
   uint8_t *map = pipe_texture_map(pipe, vp->fb_color, 0, 0, PIPE_MAP_WRITE,
                                   0, 0, vp->fb_width, vp->fb_height, &xfer);
   if (!map)
      return false;
   for (unsigned y = 0; y < vp->fb_height; y++)
      memcpy(map + (size_t)y * xfer->stride,
             (const uint8_t *)src + (size_t)y * vp->fb_width * 4,
             (size_t)vp->fb_width * 4);
   pipe_texture_unmap(pipe, xfer);
   return true;
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
      void    *xverts = malloc((size_t)count * stride);

      if (xverts &&
          vp_launch_vs(vp->dev, vs->vxbin, vs->vxbin_size,
                       xverts, count * stride, count)) {
         /* VORTEXPIPE_SW_RASTER forces the Phase 3 llvmpipe-raster
          * path; otherwise rasterize on the Vortex RASTER unit. */
         static int sw_raster = -1;
         if (sw_raster < 0)
            sw_raster = getenv("VORTEXPIPE_SW_RASTER") != NULL;

         /* Vortex hardware raster path: round-trip the colour
          * attachment through the RASTER unit + fragment kernel. */
         if (!sw_raster && fs && fs->vxbin &&
             vp->fb_color && vp->fb_width && vp->fb_height) {
            uint32_t w = vp->fb_width, h = vp->fb_height;
            void *cbuf = malloc((size_t)w * h * 4);
            if (cbuf && vp_fb_color_read(pipe, vp, cbuf) &&
                vp_raster_draw(vp->dev, fs->vxbin, fs->vxbin_size,
                               xverts, count, &vs->vs_layout, cbuf, w, h) &&
                vp_fb_color_write(pipe, vp, cbuf)) {
               vp_dbg("vortexpipe: draw_vbo -> Vortex RASTER unit "
                      "(%u verts, %ux%u)", count, w, h);
               free(cbuf);
               free(xverts);
               return;
            }
            free(cbuf);
         }

         /* fallback: VS on Vortex, rasterization on llvmpipe */
         if (vp_draw_passthrough(vp, pipe, info, drawid_offset,
                                 xverts, count)) {
            vp_dbg("vortexpipe: draw_vbo ran the %u-vertex VS on Vortex "
                   "(llvmpipe raster)", count);
            free(xverts);
            return;
         }
      }
      free(xverts);
   }

   /* inherit-and-accelerate fallback */
   vp->lp_draw_vbo(pipe, info, drawid_offset, indirect, draws, num_draws);
}

static void
vp_context_destroy(struct pipe_context *pipe)
{
   struct vp_context *vp = vp_reg_get(pipe);
   void (*lp_destroy)(struct pipe_context *) = vp->lp_context_destroy;

   /* release the cached Phase 3 draw-integration objects */
   if (vp->passthrough_vs)
      vp->lp_delete_vs_state(pipe, vp->passthrough_vs);
   if (vp->velems)
      pipe->delete_vertex_elements_state(pipe, vp->velems);

   vp_reg_del(pipe);
   FREE(vp);
   lp_destroy(pipe);
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
   pipe->destroy              = vp_context_destroy;

   vp_dbg("vortexpipe: context created -- compute hooks intercepted");
   return pipe;
}
