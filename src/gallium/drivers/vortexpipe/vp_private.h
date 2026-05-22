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
   void  *lp_cso;          /* llvmpipe's compute-state object */
   void  *vxbin;           /* compiled Vortex kernel image, or NULL */
   size_t vxbin_size;
};

/* Per-context vortexpipe state, keyed by the llvmpipe pipe_context *. */
struct vp_context {
   vx_device_h dev;                       /* borrowed from vp_screen */
   struct vp_cso *cur_cso;                /* bound compute state */
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
