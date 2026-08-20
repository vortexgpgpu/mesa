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

#include "util/hash_table.h"
#include "util/simple_mtx.h"
#include "pipe/p_screen.h"
#include "pipe/p_context.h"
#include "pipe/p_state.h"      /* struct pipe_vertex_buffer (stored by value) */

#include "vortex2.h"
#include "VX_types.h"            /* VX_TEX_LOD_MAX */
#include "vp_nir_to_llvm.h"      /* struct vp_vs_layout */
#include "gfx_fs_desc_abi.h"     /* GFX_OM_MAX_RT */

/* vp_raster.cpp is C++; the declarations below are defined in C translation
 * units, so their linkage has to be spelled out or the C++ caller looks for a
 * mangled symbol that does not exist. */
#ifdef __cplusplus
extern "C" {
#endif

/* Per-screen vortexpipe state, keyed by the llvmpipe pipe_screen *. */
struct vp_screen {
   vx_device_h dev;                       /* Vortex device, or NULL */
   struct pipe_context *(*lp_context_create)(struct pipe_screen *,
                                             void *priv, unsigned flags);
   void (*lp_screen_destroy)(struct pipe_screen *);
   const char *(*lp_screen_get_name)(struct pipe_screen *);
   char *(*lp_finalize_nir)(struct pipe_screen *, struct nir_shader *nir);
   bool (*lp_is_format_supported)(struct pipe_screen *, enum pipe_format,
                                  enum pipe_texture_target, unsigned sample_count,
                                  unsigned storage_sample_count, unsigned usage);
   /* Lazy-formatted device-name override returned by vp_screen_get_name;
    * cached so the pointer Vulkan reads stays valid for the screen's life.
    * Sized to comfortably hold "vortexpipe (Vortex on <llvmpipe name>)". */
   char name_str[128];

   /* Device-side copies of finalized compute shaders, keyed by the NIR
    * llvmpipe kept. Subgroup lowering bakes in a width, and the two consumers
    * of a finalized shader need different ones: llvmpipe executes at its own
    * vector width, the device at a warp. One shader object cannot carry both,
    * so the device gets its own copy lowered for the warp. */
   struct hash_table *dev_nir;
   simple_mtx_t       dev_nir_lock;

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

   /* Device buffers that outlive the dispatch that first needed them, keyed by
    * the host allocation they mirror. Without this a resource is allocated,
    * uploaded and freed per dispatch -- per draw on the graphics path -- so an
    * unchanged vertex buffer is re-uploaded every frame and a resource's device
    * address moves between dispatches.
    *
    * Entries are removed from resource_destroy. A host allocation can be freed
    * and a later one can land at the same address, so a table keyed on a raw
    * pointer without eviction would serve a stale device buffer for a different
    * resource -- this driver has already been bitten by pointer recycling
    * aliasing a driver-side cache. */
   struct vp_resident *resident;
   unsigned            n_resident;
   unsigned            resident_cap;
   simple_mtx_t        resident_lock;
   void (*lp_resource_destroy)(struct pipe_screen *, struct pipe_resource *);
};

/* One host allocation's device-resident mirror. `host_base`/`size` describe the
 * host range it stands for, so a descriptor pointing anywhere inside that range
 * resolves to this entry at the matching offset. */
struct vp_resident {
   const uint8_t *host_base;
   uint32_t       size;
   vx_buffer_h    buf;
   uint64_t       dev_addr;
   /* Whether the host bytes have changed since the device copy was last
    * written. Starts true and is cleared only by a completed upload, so every
    * path that has not been taught to invalidate errs towards re-uploading
    * rather than towards serving stale data. */
   bool           dirty;
};

/* The host bytes `pres` owns, for either a buffer or a texture. False when the
 * range cannot be determined, and the resource must then be treated as holding
 * nothing resident. */
bool vp_resource_host_range(struct pipe_resource *pres, const uint8_t **out_base,
                            uint32_t *out_size);

/* Device address for `bytes` of host memory at `host`, allocating and recording
 * a resident buffer on first use. Returns 0 on failure. `out_buf` receives the
 * device buffer the range lives in (owned by the screen, not the caller) and
 * `out_off` the byte offset of `host` within it. */
uint64_t vp_screen_resident_addr(struct pipe_screen *screen, const void *host,
                                 uint32_t bytes, vx_buffer_h *out_buf,
                                 uint32_t *out_off, bool *out_dirty);

/* Record that the device copy of `bytes` at `host` now matches the host bytes.
 * Called after an upload completes, never before -- a range marked clean that
 * was not actually written is served stale on the next dispatch. */
void vp_screen_resident_clean(struct pipe_screen *screen, const void *host,
                              uint32_t bytes);

/* Mark every resident range overlapping [host, host+bytes) as needing a fresh
 * upload. */
void vp_screen_resident_dirty(struct pipe_screen *screen, const void *host,
                              uint32_t bytes);

/* Mark every resident range dirty. Used when work runs on llvmpipe: it writes
 * host allocations directly, and nothing reports which ones, so the only sound
 * answer is to distrust all of them. */
void vp_screen_resident_dirty_all(struct pipe_screen *screen);

/* True when the device carries the ray-tracing unit, which decides whether an
 * acceleration structure is transcoded to the RTU scene format or copied. */
bool vp_screen_has_rtu(struct pipe_screen *screen);

/* Acceleration-structure relocation, shared by the dispatch and the draw. A
 * BVH holds absolute links to the structures below it, so it cannot be handed
 * to the device as host bytes or as a plain copy -- the links are rewritten as
 * each one is brought across. The context owns every buffer and staging blob
 * that takes, all of which the asynchronous uploads still read, so vp_as_end
 * belongs after vx_queue_finish and nowhere earlier. */
struct vp_as_ctx;
struct vp_as_ctx *vp_as_begin(vx_device_h dev, vx_queue_h q, bool has_rtu);
uint64_t vp_as_relocate(struct vp_as_ctx *c, uint64_t tlas_host);
bool vp_as_ok(const struct vp_as_ctx *c);
void vp_as_end(struct vp_as_ctx *c);

/* A compiled compute state: llvmpipe's cso plus the Vortex kernel
 * image. vortexpipe's create_compute_state returns one of these
 * (not the raw llvmpipe cso); bind/delete unwrap it. The descriptor
 * table (vp_desc, from vp_nir_to_llvm.h) is discovered by scanning
 * the NIR and drives vp_launch_grid's descriptor-buffer relocation. */
/* What makes two compilations of one fragment shader differ. Routing decides
 * which kernel wrapper the translator emits and whether the merge is a call or
 * an OM-aperture store, so it cannot be a runtime argument; the sample count
 * will select the multisample fragment path the same way. Everything else the
 * translator reads is a property of the NIR and so is invariant across
 * variants. Keep this minimal -- every dimension multiplies compile time and
 * device residency churn. */
struct vp_fs_variant_key {
   struct vp_sw_routing routing;
   unsigned             samples;   /* 1, or the pass' sample count */
   /* One bit per render target: set when that attachment's format wants its
    * blue channel in the low byte. A mask rather than a flag because the set
    * may mix formats that disagree, and it is still one scalar to compare. */
   uint8_t              bgra_mask;
};

/* One compiled fragment shader: its image, and the device residency handles
 * that image holds while it occupies the fixed FS address. */
struct vp_fs_variant {
   struct vp_fs_variant_key key;
   void       *vxbin;
   size_t      vxbin_size;
   vx_module_h vx_module;
   vx_kernel_h vx_kernel;
};

/* Routing is fixed per shader (caps + env + NIR) and the channel order is fixed
 * by the render pass the pipeline was created against, so the only live
 * dimension is the sample count, and Vulkan offers few of those. A small fixed
 * array needs no recency bookkeeping. */
#define VP_MAX_FS_VARIANTS 4

struct vp_cso {
   void  *lp_cso;          /* llvmpipe's shader-state object */
   void  *vxbin;           /* compiled Vortex kernel image, or NULL */
   size_t vxbin_size;
   unsigned lmem_size;     /* compute: shared-memory bytes (nir shared_size) */
   /* compute: bitmask of set_shader_buffers slots the kernel reads as a
    * const-index load_ssbo(imm slot, off) -- the RT trace-ray command buffer,
    * whose embedded SBT shader-record pointers need device relocation. */
   uint32_t trace_cmd_slots;
   struct vp_vs_layout vs_layout;  /* vertex shaders: output record layout */
   struct vp_desc descs[VP_MAX_DESCS];  /* set-0 descriptors the kernel uses */
   unsigned        num_descs;
   /* Fragment TEX-stage-0 texture descriptor location: the (cbuf_index, byte
    * offset) of the sampled image's lp_descriptor within its descriptor-set
    * blob, from the tex instruction's nir_tex_src_texture_handle. Lets the draw
    * pick the *actually sampled* texture per draw (lp_jit_texture.base at +0),
    * instead of the last-created handle. has_tex_desc=false ⇒ fall back to the
    * captured cur_tex (single-texture shaders / no bindless handle). */
   bool     has_tex_desc;
   unsigned tex_desc_cbuf;
   unsigned tex_desc_offset;
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
   /* Fragment shaders only. The NIR is cloned here so a variant can be
    * translated after pipeline creation -- llvmpipe owns and frees the original
    * (lp_state_fs.c), so its lifetime is not ours to rely on. NULL for a TGSI
    * shader, and for the vertex and compute stages, which are single-variant
    * and keep using the vxbin/vx_module fields above. */
   struct nir_shader *fs_nir;
   struct vp_fs_variant fs_variants[VP_MAX_FS_VARIANTS];
   unsigned             num_fs_variants;
   /* Which variant currently holds the fixed FS device address, or -1. Only one
    * image can be resident there, so switching variants must evict this one
    * first -- the allocator rejects an overlapping reservation outright. */
   int                  fs_resident;
   /* Number of colour outputs the fragment shader writes (RT count).
    * >1 forces the SW-OM MRT path. 0/1 = single RT. */
   unsigned fs_num_color;
   /* The shader supplies gl_FragDepth, which rules out early-Z. */
   bool     fs_writes_depth;
};

/* Captured + VX-encoded output-merger state. create_*_state
 * translates the Gallium depth/blend state to the Vortex OM encoding
 * once and registers it (keyed by the llvmpipe cso, via vp_reg);
 * create returns llvmpipe's cso unchanged so csos made before
 * vortexpipe's hooks were armed (util_blitter's) pass straight
 * through. */
struct vp_dsa_cso {
   bool     depth_test;
   bool     depth_write;
   uint32_t depth_func;        /* VX_OM_DEPTH_FUNC_* */
   /* Two-sided stencil, indexed [0]=front [1]=back. The reference value is not
    * here: Gallium delivers it as dynamic state through set_stencil_ref. */
   uint32_t stencil_func[2];        /* VX_OM_DEPTH_FUNC_* */
   uint32_t stencil_fail[2];        /* VX_OM_STENCIL_OP_* */
   uint32_t stencil_zfail[2];
   uint32_t stencil_zpass[2];
   uint32_t stencil_mask[2];        /* value (compare) mask */
   uint32_t stencil_writemask[2];
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
   uint32_t logic_op;          /* VX_OM_LOGIC_OP_* */
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
   uint32_t filter;            /* VX_TEX_FILTER_* -- the mag tap */
   uint32_t min_filter;        /* VX_TEX_FILTER_* -- the min (minification) tap */
   uint32_t wrap_u;            /* VX_TEX_WRAP_* */
   uint32_t wrap_v;
   uint32_t wrap_w;            /* VX_TEX_WRAP_* for the 3D depth (r) axis */
   bool     mip_enable;        /* sampler reaches a non-base level (max_lod > 0.5) */
   bool     mip_linear;        /* mip mode is LINEAR (trilinear) rather than nearest */
   bool     compare_enable;    /* PIPE_TEX_COMPARE_R_TO_TEXTURE (sampler2DShadow) */
   uint32_t compare_func;      /* raw PIPE_FUNC_* (mapped to VX at draw time) */
   uint32_t min_lod;           /* sampler LOD clamp lower bound, Q(VX_TEX_LOD_FRAC_BITS) */
   uint32_t max_lod;           /* sampler LOD clamp upper bound, Q(VX_TEX_LOD_FRAC_BITS) */
   int32_t  lod_bias;          /* sampler LOD bias, signed Q(VX_TEX_LOD_FRAC_BITS) */
   uint32_t border;            /* CLAMP_TO_BORDER colour, ARGB8888 */
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
   /* llvmpipe entry points vortexpipe interposes on to invalidate the device's
    * copy of a resource the host just wrote. */
   void (*lp_buffer_subdata)(struct pipe_context *, struct pipe_resource *,
                             unsigned usage, unsigned offset, unsigned size,
                             const void *data);
   void (*lp_texture_subdata)(struct pipe_context *, struct pipe_resource *,
                              unsigned level, unsigned usage,
                              const struct pipe_box *, const void *data,
                              unsigned stride, uintptr_t layer_stride);
   void (*lp_buffer_unmap)(struct pipe_context *, struct pipe_transfer *);
   void (*lp_texture_unmap)(struct pipe_context *, struct pipe_transfer *);
   void (*lp_clear_buffer)(struct pipe_context *, struct pipe_resource *,
                           unsigned offset, unsigned size,
                           const void *clear_value, int clear_value_size);
   void (*lp_clear_texture)(struct pipe_context *, struct pipe_resource *,
                            unsigned level, const struct pipe_box *,
                            const void *data);
   void (*lp_clear_render_target)(struct pipe_context *, struct pipe_surface *,
                                  const union pipe_color_union *,
                                  unsigned dstx, unsigned dsty,
                                  unsigned width, unsigned height,
                                  bool render_condition_enabled);
   void (*lp_clear_depth_stencil)(struct pipe_context *, struct pipe_surface *,
                                  unsigned clear_flags, double depth,
                                  unsigned stencil,
                                  unsigned dstx, unsigned dsty,
                                  unsigned width, unsigned height,
                                  bool render_condition_enabled);
   void (*lp_resource_copy_region)(struct pipe_context *,
                                   struct pipe_resource *dst, unsigned dst_level,
                                   unsigned dstx, unsigned dsty, unsigned dstz,
                                   struct pipe_resource *src, unsigned src_level,
                                   const struct pipe_box *src_box);
   void (*lp_blit)(struct pipe_context *, const struct pipe_blit_info *);
   struct vp_cso *cur_cso;                /* bound compute state */
   struct vp_cso *cur_vs;                 /* bound vertex shader */
   struct vp_cso *cur_fs;                 /* bound fragment shader */
   /* Holder of the VP_STARTUP_FS device address. Compute kernels and fragment
    * shaders link at that one base, so exactly one image occupies it and the
    * loader refuses an overlapping claim. Naming the holder here keeps eviction
    * independent of which CSO is bound: the claimant releases whoever holds the
    * address, rather than guessing that it is cur_fs. NULL when free. */
   struct vp_cso *startup_fs_owner;
   bool           startup_fs_is_compute;  /* holder is a compute module */
   /* Colour attachment channel order for the bound framebuffer, decided once
    * per bind. fb_color_ok is false when an attachment has an order the device
    * cannot produce, and sends the whole render pass to llvmpipe rather than
    * permuting its colours. fb_color_bgra_mask carries one bit per attachment:
    * set means that target's format wants its blue channel in the low byte,
    * which is the order the fragment variant is compiled to pack it in.
    *
    * The bit follows the format's encoder, not the attachment's name -- R8 and
    * RG8 set it because the merger reads their channels from the high lanes,
    * not because those attachments are blue-first. */
   bool           fb_color_ok;
   uint8_t        fb_color_bgra_mask;
   /* Each attachment's storage format, as the merger encodes it, and its texel
    * width. Anything but A8R8G8B8 forces the software merger -- the
    * fixed-function one has no colour-format register -- and the width is what
    * every buffer size, pitch and readback below is measured in. Per attachment
    * because a multi-target framebuffer may mix them: every render target gets
    * its own merger state, so each can be stored in its own format. Entries
    * past fb_nr_cbufs hold the pass-through description. */
   uint32_t       fb_color_format[GFX_OM_MAX_RT];   /* VX_OM_COLOR_FORMAT_* */
   uint32_t       fb_color_bpp[GFX_OM_MAX_RT];      /* bytes per texel */
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
   /* Vertex-stage constant buffers, by index, with the same meaning per index
    * as the fragment set above. The vertex stage overlays its own meanings on
    * arg-block slots 0-4, so it cannot reach these the way compute does; the
    * draw path uploads them and builds a VS descriptor table passed in
    * VP_ARG_VS_DESC. */
   struct pipe_resource *vs_cbuf[8];
   unsigned              vs_cbuf_off[8];
   unsigned              vs_cbuf_sz[8];
   /* Raw compute shader buffers bound via set_shader_buffers (distinct from
    * the descriptor-set SSBOs, which live in the set-0 blob at cbuf[1]).
    * lavapipe binds internal buffers here — e.g. the RT trace-ray command
    * buffer at slot 0 — and the kernel reads them as load_ssbo(imm slot). */
   struct pipe_resource *sbuf[VP_MAX_SSBO];
   unsigned              sbuf_off[VP_MAX_SSBO];
   unsigned              sbuf_sz[VP_MAX_SSBO];
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
   unsigned              fb_samples;     /* samples per pixel; 1 = single-sample */
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
   unsigned              rfb_s;   /* samples the resident buffers were sized for */
   /* Texel width the resident colour buffers were sized for. Carried separately
    * from fb_color_bpp because the flush back to the resource runs as part of
    * binding the *next* framebuffer, when fb_color_bpp already describes it. */
   unsigned              rfb_bpp;
   bool                  rfb_dirty;
   /* Extra resident colour buffers for attachments 1.. (RT0 uses rcb).
    * rmrt_res[k] is the framebuffer resource each is synced back to, and
    * rmrt_bpp[k] the texel width it was sized for -- carried for the same
    * reason as rfb_bpp, and per attachment because the set may mix widths. */
   vx_buffer_h           rcb_extra[GFX_OM_MAX_RT];
   struct pipe_resource *rmrt_res[GFX_OM_MAX_RT];
   unsigned              rmrt_bpp[GFX_OM_MAX_RT];
   unsigned              rmrt_nr;
   void (*lp_flush)(struct pipe_context *, struct pipe_fence_handle **, unsigned);
   /* Texture residency: the converted + uploaded TEX-stage-0 texels kept
    * device-resident across draws, keyed by the bound sampler resource. */
   vx_buffer_h           rtex_buf;
   struct pipe_resource *rtex_res;
   unsigned              rtex_w, rtex_h;
   /* Per-LOD byte offset into rtex_buf (mip 0 at [0]); the whole mip chain is
    * uploaded contiguously so the TEX unit can address any selected level. */
   uint32_t              rtex_mipoff[VX_TEX_LOD_MAX + 1];
   uint32_t              rtex_layer_stride;   /* bytes per array layer (0 = single 2D) */
   /* Sampled-texture identity map: every texture that gets a bindless handle
    * (create_texture_handle) is recorded here as (resource, level-0 host base).
    * A draw reads its FS tex descriptor's lp_jit_texture.base and matches it to
    * pick the sampled resource — so >1 bound texture selects correctly instead
    * of always sampling the last handle created. */
#define VP_MAX_TEX_HANDLES 32
   const void           *txh_base[VP_MAX_TEX_HANDLES];
   struct pipe_resource *txh_res[VP_MAX_TEX_HANDLES];
   unsigned              txh_target[VP_MAX_TEX_HANDLES];
   unsigned              txh_layers[VP_MAX_TEX_HANDLES];
   unsigned              txh_count;
   /* Output-merger state (depth-stencil-alpha + blend). */
   struct vp_dsa_cso   *cur_dsa;
   struct vp_blend_cso *cur_blend;
   uint32_t             cur_stencil_ref[2];   /* set_stencil_ref, front/back */
   uint32_t             cur_blend_color;      /* set_blend_color, ARGB8888 */
   void *(*lp_create_dsa_state)(struct pipe_context *,
                                const struct pipe_depth_stencil_alpha_state *);
   void  (*lp_bind_dsa_state)(struct pipe_context *, void *);
   void  (*lp_delete_dsa_state)(struct pipe_context *, void *);
   void *(*lp_create_blend_state)(struct pipe_context *,
                                  const struct pipe_blend_state *);
   void  (*lp_bind_blend_state)(struct pipe_context *, void *);
   void  (*lp_delete_blend_state)(struct pipe_context *, void *);
   void  (*lp_set_blend_color)(struct pipe_context *,
                               const struct pipe_blend_color *);
   void  (*lp_set_stencil_ref)(struct pipe_context *,
                               const struct pipe_stencil_ref);
   /* Rasterizer state: face cull + front-face winding for the device
    * front end (see vp_rast_cso). Captured at create time (the winding /
    * cull mask are only in pipe_rasterizer_state there); everything else
    * passes through to llvmpipe unchanged. */
   struct vp_rast_cso *cur_rast;
   void *(*lp_create_rasterizer_state)(struct pipe_context *,
                                       const struct pipe_rasterizer_state *);
   void  (*lp_bind_rasterizer_state)(struct pipe_context *, void *);
   void  (*lp_delete_rasterizer_state)(struct pipe_context *, void *);
   /* Work-placement tally, reported at context teardown when
    * VORTEXPIPE_ATTRIB=1. A passing test proves correctness, not that the
    * device ran anything: these separate "ran on Vortex" from "ran on the CPU
    * through the llvmpipe fallback", so a run can be shown to be real offload. */
   unsigned launches_device, launches_cpu;
   unsigned draws_device, draws_cpu;
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
   float vp_min_z, vp_max_z;   /* depth range (VkViewport minDepth..maxDepth) */
   void  (*lp_set_viewport_states)(struct pipe_context *, unsigned, unsigned,
                                   const struct pipe_viewport_state *);
   /* Texture + sampler, captured from create_texture_handle.
    * lavapipe routes an image view (texture, state==NULL) and a
    * sampler (state, view==NULL) through that one entry point; gfx-v1
    * binds the latest of each to TEX stage 0. cur_sampler points at
    * cur_sampler_store once a sampler has been seen. */
   struct pipe_resource  *cur_tex;        /* bound FS texture, or NULL */
   unsigned               cur_tex_first_level; /* bound view's baseMipLevel */
   uint32_t               cur_tex_swizzle;     /* bound view's packed component map */
   unsigned               cur_tex_target;      /* bound view's target (PIPE_TEXTURE_*) */
   unsigned               cur_tex_layers;      /* bound view's array-layer count */
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

/* Hand over the device-side copy of a finalized compute shader, or NULL if the
 * screen kept none. The caller owns it and must ralloc_free it. */
struct nir_shader *vp_screen_take_dev_nir(struct pipe_screen *screen,
                                          struct nir_shader *finalized);

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

#ifdef __cplusplus
}
#endif

#endif /* VP_PRIVATE_H */
