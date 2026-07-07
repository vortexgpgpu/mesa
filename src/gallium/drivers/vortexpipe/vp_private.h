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
#include "gfx_fs_desc_abi.h"     /* GFX_OM_MAX_RT */

/* Per-screen vortexpipe state, keyed by the llvmpipe pipe_screen *. */
struct vp_screen {
   vx_device_h dev;                       /* Vortex device, or NULL */
   struct pipe_context *(*lp_context_create)(struct pipe_screen *,
                                             void *priv, unsigned flags);
   void (*lp_screen_destroy)(struct pipe_screen *);
   const char *(*lp_screen_get_name)(struct pipe_screen *);
   char *(*lp_finalize_nir)(struct pipe_screen *, struct nir_shader *nir);
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
    * larger values truncate, producing empty-tmask warps that the
    * device rejects (an invalid-request-mask fault).
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
   bool     has_rtu;       /* VX_ISA_EXT_RTU: HW ray-tracing unit present.
                            * Gates the HW RT path (ray-query / trace-ray
                            * lowered to vx_rt_ ops) vs lavapipe's SW BVH
                            * walk. */
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
   /* Residency: the vxbin loaded onto the device once and reused across
    * draws (compile-once + upload-resident-once — no per-draw module reload,
    * no /tmp round-trip). Lazily loaded by vp_raster_draw, released on delete.
    * Same-stage CSOs share one device address, so binding a different VS/FS
    * evicts the previously-resident one (vp_bind_vs_state / vp_bind_fs_state). */
   vx_module_h vx_module;
   vx_kernel_h vx_kernel;
   /* Per-unit SW routing baked into a fragment shader at compile time (from
    * device caps + VORTEXPIPE_FORCE_SW). The draw path reads this to build +
    * pass the resident SW descriptors the FS was compiled to expect. */
   struct vp_sw_routing fs_routing;
   /* Number of colour outputs the fragment shader writes (RT count).
    * >1 forces the SW-OM MRT path. 0/1 = single RT. */
   unsigned fs_num_color;
};

/* Captured + VX-encoded output-merger state. create_*_state
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
   uint32_t blend_mode;        /* VX_DCR_OM_BLEND_MODE packed word (RT0) */
   uint32_t blend_func;        /* VX_DCR_OM_BLEND_FUNC packed word (RT0) */
   uint32_t colormask;         /* VX_DCR_OM_CBUF_WRITEMASK (RGBA bits, RT0) */
   /* Per-attachment blend/write-mask. Slot 0 mirrors the scalar fields
    * above (single-RT path unchanged); slots 1.. carry the independent-blend
    * state (or a copy of RT0 when independent blend is off). */
   uint32_t rt_blend_mode[GFX_OM_MAX_RT];
   uint32_t rt_blend_func[GFX_OM_MAX_RT];
   uint32_t rt_colormask[GFX_OM_MAX_RT];
};

/* Captured rasterizer state: the face-cull inputs the device front end
 * needs. cull_face is the Gallium PIPE_FACE_* mask and front_ccw the
 * front-face winding; vp_draw_vbo turns the pair into the device
 * SETUP_CULL_* mode (see vp_cull_mode). Only these two fields are used —
 * everything else in pipe_rasterizer_state stays with llvmpipe. */
struct vp_rast_cso {
   unsigned cull_face;         /* PIPE_FACE_{NONE,FRONT,BACK,FRONT_AND_BACK} */
   bool     front_ccw;         /* front face is counter-clockwise */
};

/* Captured texture-sampler state, VX TEX-encoded. lavapipe
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
   /* Instance-rate divisor per attribute (0 = per-vertex). The device VS
    * fetches attributes per-vertex only; an instanced draw that binds an
    * instance-rate attribute falls back to llvmpipe (gl_InstanceIndex-driven
    * instancing IS on-device). */
   uint32_t instance_divisor[VP_MAX_ATTR];
};

/* Per-context vortexpipe state, keyed by the llvmpipe pipe_context *. */
struct vp_context {
   vx_device_h dev;                       /* borrowed from vp_screen */
   struct vp_cso *cur_cso;                /* bound compute state */
   struct vp_cso *cur_vs;                 /* bound vertex shader */
   struct vp_cso *cur_fs;                 /* bound fragment shader */
   /* compute constant buffers, by index -- lavapipe binds the
    * descriptor buffer for descriptor set N at index N+1. */
   struct pipe_resource *cbuf[8];
   unsigned              cbuf_off[8];
   /* Fragment-stage constant buffers, by index (0 = push constants,
    * 1 = descriptor set-0 blob, …). The draw path uploads each bound one
    * to device memory + builds the resident FS descriptor table. */
   struct pipe_resource *fs_cbuf[8];
   unsigned              fs_cbuf_off[8];
   unsigned              fs_cbuf_sz[8];
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
   /* graphics: vertex-shader state + the draw entry. */
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
   /* Draw integration: a cached passthrough VS + vertex-
    * elements state that feed the Vortex-transformed vertices into
    * llvmpipe's rasterizer (see vp_draw_vbo). */
   void *passthrough_vs;
   void *velems;
   /* The bound render targets (set_framebuffer_state). */
   void (*lp_set_framebuffer_state)(struct pipe_context *,
                                    const struct pipe_framebuffer_state *);
   struct pipe_resource *fb_color;
   struct pipe_resource *fb_depth;       /* depth/stencil attachment, or NULL */
   unsigned              fb_width, fb_height;
   /* All bound colour attachments (fb_color == fb_cbufs[0]). A draw to
    * >1 attachment renders each into its own resident device buffer and writes
    * each back at sync. */
   struct pipe_resource *fb_cbufs[GFX_OM_MAX_RT];
   unsigned              fb_nr_cbufs;
   /* Persistent front-end working set, reused across the frame's draws. */
   struct vp_raster_pool *raster_pool;
   /* Framebuffer residency: the colour + depth attachments kept
    * device-resident across the draws of a render pass. rcb/rzb shadow
    * fb_color (rfb_res) at rfb_w x rfb_h: cleared/initialised ONCE per pass
    * (not per draw — preserving depth + colour across draws), then synced back
    * to the colour resource at flush / framebuffer-change / llvmpipe fallback.
    * rfb_dirty = the device buffers hold renders not yet in the resource. */
   vx_buffer_h           rcb, rzb;
   struct pipe_resource *rfb_res;
   unsigned              rfb_w, rfb_h;
   bool                  rfb_dirty;
   /* Extra resident colour buffers for attachments 1.. (RT0 uses rcb).
    * rmrt_res[k] is the framebuffer resource each is synced back to. */
   vx_buffer_h           rcb_extra[GFX_OM_MAX_RT];
   struct pipe_resource *rmrt_res[GFX_OM_MAX_RT];
   unsigned              rmrt_nr;
   void (*lp_flush)(struct pipe_context *, struct pipe_fence_handle **, unsigned);
   /* Texture residency: the converted + uploaded TEX-stage-0 texels kept
    * device-resident across draws, keyed by the bound sampler resource. */
   vx_buffer_h           rtex_buf;
   struct pipe_resource *rtex_res;
   unsigned              rtex_w, rtex_h;
   /* Output-merger state (depth-stencil-alpha + blend). */
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
   /* Rasterizer state: face cull + front-face winding for the device
    * front end (see vp_rast_cso). Captured at create time (the winding /
    * cull mask are only in pipe_rasterizer_state there); everything else
    * passes through to llvmpipe unchanged. */
   struct vp_rast_cso *cur_rast;
   void *(*lp_create_rasterizer_state)(struct pipe_context *,
                                       const struct pipe_rasterizer_state *);
   void  (*lp_bind_rasterizer_state)(struct pipe_context *, void *);
   void  (*lp_delete_rasterizer_state)(struct pipe_context *, void *);
   /* Viewport transform (screen = ndc*scale + bias) captured from the app's
    * bound VkViewport. The device front end otherwise hardwires a full-fb
    * y-down viewport, so a y-flip (negative-height) or offset/scaled viewport
    * would mirror / mis-place the screen-space triangle and flip the signed-area
    * face-cull sense. vp_draw_vbo forwards this to the device setup; only slot 0
    * is tracked (gfx-v1 is single-viewport). vp_valid is false until the app
    * sets one, in which case the default full-fb transform is used. */
   bool  vp_valid;
   float vp_scale_x, vp_trans_x;
   float vp_scale_y, vp_trans_y;
   void  (*lp_set_viewport_states)(struct pipe_context *, unsigned, unsigned,
                                   const struct pipe_viewport_state *);
   /* Texture + sampler, captured from create_texture_handle.
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
