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

#include "vortex2.h"
#include "vp_nir_to_llvm.h"      /* struct vp_vs_layout */

/* Per-screen vortexpipe state, keyed by the llvmpipe pipe_screen *. */
struct vp_screen {
   vx_device_h dev;                       /* Vortex device, or NULL */
   struct pipe_context *(*lp_context_create)(struct pipe_screen *,
                                             void *priv, unsigned flags);
   void (*lp_screen_destroy)(struct pipe_screen *);
};

/* A compiled compute state: llvmpipe's cso plus the Vortex kernel
 * image. vortexpipe's create_compute_state returns one of these
 * (not the raw llvmpipe cso); bind/delete unwrap it. */
struct vp_cso {
   void  *lp_cso;          /* llvmpipe's shader-state object */
   void  *vxbin;           /* compiled Vortex kernel image, or NULL */
   size_t vxbin_size;
   struct vp_vs_layout vs_layout;  /* vertex shaders: output record layout */
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
