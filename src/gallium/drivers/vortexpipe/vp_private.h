/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vortexpipe internal state.
 *
 * vortexpipe is a thin driver layered on llvmpipe: it owns the
 * llvmpipe pipe_screen / pipe_context and overrides just the few
 * entry points it specializes (context creation + the compute
 * hooks). Rather than a full decorator -- which would mean ~140
 * forwarding thunks for the pipe_screen/pipe_context vtables --
 * vortexpipe patches the entry points it overrides in place and
 * keeps its side state in a pointer-keyed registry. Patching the
 * vtable is legitimate here: vortexpipe created (and therefore
 * owns) the llvmpipe base.
 */

#ifndef VP_PRIVATE_H
#define VP_PRIVATE_H

#include "pipe/p_screen.h"
#include "pipe/p_context.h"
#include "pipe/p_state.h"      /* struct pipe_vertex_buffer (stored by value) */

#include "vortex2.h"
#include "vp_nir_to_llvm.h"      /* struct vp_vs_layout */

/* Per-screen vortexpipe state, keyed by the llvmpipe pipe_screen *. */
struct vp_screen {
   vx_device_h dev;                       /* Vortex device, or NULL */
   struct pipe_context *(*lp_context_create)(struct pipe_screen *,
                                             void *priv, unsigned flags);
   void (*lp_screen_destroy)(struct pipe_screen *);
   const char *(*lp_screen_get_name)(struct pipe_screen *);
   /* Lazy-formatted device-name override returned by vp_screen_get_name;
    * cached so the pointer Vulkan reads stays valid for the screen's life.
    * Sized to comfortably hold "vortexpipe (Vortex on <llvmpipe name>)". */
   char name_str[128];

   /* Device caps cached at screen open. Used by the launch + draw paths
    * to refuse workloads the hardware can't run as one CTA, and to gate
    * the fixed-function graphics paths on the corresponding ISA bits.
    *
    * hw_max_block_size = hw_num_threads * hw_num_warps. KMU's block_size
    * DCR is sized to address one CTA's threads (CTA_TID_WIDTH+1 bits);
    * larger values truncate, producing empty-tmask warps that crash the
    * RTL with a `invalid request mask` LSU assertion. SimX silently
    * sub-dispatches and so isn't a useful oracle here.
    *
    * has_tex / has_raster / has_om gate vp_raster_draw and the texture
    * upload path on the ISA extensions actually present. A device built
    * without them must take the llvmpipe fallback rather than emitting
    * vx_tex / vx_rast / vx_om instructions that will trap on decode. */
   uint32_t hw_num_threads;
   uint32_t hw_num_warps;
   uint32_t hw_max_block_size;
   uint64_t hw_isa_flags;
   bool     has_tex;
   bool     has_raster;
   bool     has_om;
};

/* A compiled compute state: llvmpipe's cso plus the Vortex kernel
 * image. vortexpipe's create_compute_state returns one of these
 * (not the raw llvmpipe cso); bind/delete unwrap it. The descriptor
 * table (vp_desc, from vp_nir_to_llvm.h) is discovered by scanning
 * the NIR and drives vp_launch_grid's descriptor-buffer relocation. */
struct vp_cso {
   void  *lp_cso;          /* llvmpipe's shader-state object */
   void  *vxbin;           /* compiled Vortex kernel image, or NULL */
   size_t vxbin_size;
   unsigned lmem_size;     /* compute: shared-memory bytes (nir shared_size) */
   struct vp_vs_layout vs_layout;  /* vertex shaders: output record layout */
   struct vp_desc descs[VP_MAX_DESCS];  /* set-0 descriptors the kernel uses */
   unsigned        num_descs;
};

/* Captured + VX-encoded output-merger state (Phase 5). create_*_state
 * translates the Gallium depth/blend state to the Vortex OM encoding
 * once and registers it (keyed by the llvmpipe cso, via vp_reg);
 * create returns llvmpipe's cso unchanged so csos made before
 * vortexpipe's hooks were armed (util_blitter's) pass straight
 * through. Stencil is left disabled for gfx-v1. */
struct vp_dsa_cso {
   bool     depth_test;
   bool     depth_write;
   uint32_t depth_func;        /* VX_OM_DEPTH_FUNC_* */
};

struct vp_blend_cso {
   bool     blend_enable;
   uint32_t blend_mode;        /* VX_DCR_OM_BLEND_MODE packed word */
   uint32_t blend_func;        /* VX_DCR_OM_BLEND_FUNC packed word */
   uint32_t colormask;         /* VX_DCR_OM_CBUF_WRITEMASK (RGBA bits) */
};

/* Captured texture-sampler state (Phase 6), VX TEX-encoded. lavapipe
 * routes both image views and samplers through create_texture_handle,
 * so vortexpipe captures the sampler's filter/wrap there. */
struct vp_sampler_cso {
   uint32_t filter;            /* VX_TEX_FILTER_* */
   uint32_t wrap_u;            /* VX_TEX_WRAP_* */
   uint32_t wrap_v;
};

/* Captured vertex-input layout. The VS kernel fetches one thread's
 * vertex attributes from device memory, so vortexpipe needs the
 * per-attribute byte offset + stride; like the depth/blend csos it is
 * registered (keyed by the llvmpipe cso, via vp_reg). gfx-v1 supports
 * a single interleaved vertex buffer with 32-bit (float) components --
 * what tests/vulkan/draw3d feeds. */
#define VP_MAX_ATTR 8
struct vp_velems_cso {
   unsigned num;
   uint32_t src_offset[VP_MAX_ATTR];   /* attribute byte offset in a vertex */
   uint32_t src_stride[VP_MAX_ATTR];   /* bytes between consecutive vertices */
   uint8_t  buffer_index[VP_MAX_ATTR]; /* which bound vertex buffer */
};

/* Per-context vortexpipe state, keyed by the llvmpipe pipe_context *. */
struct vp_context {
   vx_device_h dev;                       /* borrowed from vp_screen */
   struct vp_cso *cur_cso;                /* bound compute state */
   struct vp_cso *cur_vs;                 /* bound vertex shader (Phase 3) */
   struct vp_cso *cur_fs;                 /* bound fragment shader (Phase 4) */
   /* compute constant buffers, by index -- lavapipe binds the
    * descriptor buffer for descriptor set N at index N+1. */
   struct pipe_resource *cbuf[8];
   unsigned              cbuf_off[8];
   void *(*lp_create_compute_state)(struct pipe_context *,
                                    const struct pipe_compute_state *);
   void  (*lp_bind_compute_state)(struct pipe_context *, void *);
   void  (*lp_delete_compute_state)(struct pipe_context *, void *);
   void  (*lp_launch_grid)(struct pipe_context *,
                           const struct pipe_grid_info *);
   void  (*lp_set_constant_buffer)(struct pipe_context *,
                                   enum pipe_shader_type, unsigned, bool,
                                   const struct pipe_constant_buffer *);
   void  (*lp_set_shader_buffers)(struct pipe_context *,
                                  enum pipe_shader_type, unsigned, unsigned,
                                  const struct pipe_shader_buffer *,
                                  unsigned);
   /* graphics (Phase 3): vertex-shader state + the draw entry. */
   void *(*lp_create_vs_state)(struct pipe_context *,
                               const struct pipe_shader_state *);
   void  (*lp_bind_vs_state)(struct pipe_context *, void *);
   void  (*lp_delete_vs_state)(struct pipe_context *, void *);
   void  (*lp_draw_vbo)(struct pipe_context *,
                        const struct pipe_draw_info *,
                        unsigned drawid_offset,
                        const struct pipe_draw_indirect_info *,
                        const struct pipe_draw_start_count_bias *,
                        unsigned num_draws);
   void *(*lp_create_fs_state)(struct pipe_context *,
                               const struct pipe_shader_state *);
   void  (*lp_bind_fs_state)(struct pipe_context *, void *);
   void  (*lp_delete_fs_state)(struct pipe_context *, void *);
   /* Phase 3 draw integration: a cached passthrough VS + vertex-
    * elements state that feed the Vortex-transformed vertices into
    * llvmpipe's rasterizer (see vp_draw_vbo). */
   void *passthrough_vs;
   void *velems;
   /* Phase 4/5: the bound render targets (set_framebuffer_state). */
   void (*lp_set_framebuffer_state)(struct pipe_context *,
                                    const struct pipe_framebuffer_state *);
   struct pipe_resource *fb_color;
   struct pipe_resource *fb_depth;       /* depth/stencil attachment, or NULL */
   unsigned              fb_width, fb_height;
   /* Phase 5: output-merger state (depth-stencil-alpha + blend). */
   struct vp_dsa_cso   *cur_dsa;
   struct vp_blend_cso *cur_blend;
   void *(*lp_create_dsa_state)(struct pipe_context *,
                                const struct pipe_depth_stencil_alpha_state *);
   void  (*lp_bind_dsa_state)(struct pipe_context *, void *);
   void  (*lp_delete_dsa_state)(struct pipe_context *, void *);
   void *(*lp_create_blend_state)(struct pipe_context *,
                                  const struct pipe_blend_state *);
   void  (*lp_bind_blend_state)(struct pipe_context *, void *);
   void  (*lp_delete_blend_state)(struct pipe_context *, void *);
   /* Phase 6: texture + sampler, captured from create_texture_handle.
    * lavapipe routes an image view (texture, state==NULL) and a
    * sampler (state, view==NULL) through that one entry point; gfx-v1
    * binds the latest of each to TEX stage 0. cur_sampler points at
    * cur_sampler_store once a sampler has been seen. */
   struct pipe_resource  *cur_tex;        /* bound FS texture, or NULL */
   struct vp_sampler_cso *cur_sampler;    /* &cur_sampler_store, or NULL */
   struct vp_sampler_cso  cur_sampler_store;
   uint64_t (*lp_create_texture_handle)(struct pipe_context *,
                                        struct pipe_sampler_view *,
                                        const struct pipe_sampler_state *);
   /* Vertex input: the bound vertex-elements layout + vertex buffers,
    * so the VS kernel can fetch per-vertex attributes from device
    * memory (see vp_launch_vs / the VS load_input path). */
   struct vp_velems_cso     *cur_velems;
   struct pipe_vertex_buffer vbufs[VP_MAX_ATTR];
   unsigned                  num_vbufs;
   void *(*lp_create_vertex_elements_state)(struct pipe_context *, unsigned,
                                            const struct pipe_vertex_element *);
   void  (*lp_bind_vertex_elements_state)(struct pipe_context *, void *);
   void  (*lp_delete_vertex_elements_state)(struct pipe_context *, void *);
   void  (*lp_set_vertex_buffers)(struct pipe_context *, unsigned,
                                  const struct pipe_vertex_buffer *);
   void  (*lp_context_destroy)(struct pipe_context *);
};

/* Pointer-keyed side registry (see file header). */
void  vp_reg_put(const void *key, void *data);
void *vp_reg_get(const void *key);
void  vp_reg_del(const void *key);

/* Installed by vp_screen.c onto the llvmpipe screen's context_create. */
struct pipe_context *vp_context_create(struct pipe_screen *screen,
                                       void *priv, unsigned flags);

/* Verbose driver tracing -- emits only when $VORTEXPIPE_DEBUG is set.
 * Errors/fallbacks use mesa_logw directly and are always shown. */
void vp_dbg(const char *fmt, ...)
#ifdef __GNUC__
   __attribute__((__format__(__printf__, 1, 2)))
#endif
   ;

#endif /* VP_PRIVATE_H */
