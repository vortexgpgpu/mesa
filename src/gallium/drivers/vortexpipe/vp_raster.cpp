/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_raster -- gfx_v2 on-device draw orchestration.
 *
 * The whole binning front end runs on the device: the embedded setup_k +
 * binning_k kernels (sw/gfx, built into libvortexpipe) do triangle clip +
 * setup and parallel bin-sort over device-resident memory, producing
 * RASTER's primbuf + dense tilebuf with no host graphics::Binning(). The
 * host only converts VS output to the front-end vertex record, sizes the
 * dense tile grid from the framebuffer, and sequences the launches.
 *
 * vp_raster_draw then programs the RASTER and OM DCRs from the
 * device-produced buffers and dispatches the fragment-shader kernel, which
 * polls vx_rast() for quads and submits shaded fragments to the OM unit via
 * vx_om() for depth-test, blend and colour + depth writeback.
 */

#include "vp_raster.h"
#include "vp_launch.h"           /* struct vp_vertex_input (folded-in VS stage) */
#include "vp_compile.h"          /* VP_STARTUP_VS link base */

#include <vector>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "graphics.h"            /* vortex::graphics::rast_prim_t / rast_tile_header_t */
#include "gfx_frontend_abi.h"    /* pipe_arg_t, setup_vertex_t, PIPE_STAGE_*, SETUP_* */
#include "gfx_sw_abi.h"          /* gfx_sw_texstate_t — SW sampler descriptor */
#include "gallivm/lp_bld_jit_types.h"  /* struct lp_jit_buffer (descriptor layout) */

/* The FS descriptor relocation reads a UBO/SSBO descriptor's data pointer at
 * lp_jit_buffer +0 (8 bytes) and its element count at +8 (u32). Pin that layout
 * so a future llvmpipe change to struct lp_jit_buffer fails the build here rather
 * than silently corrupting every relocated UBO/SSBO base address. */
static_assert(offsetof(struct lp_jit_buffer, u) == 0,
              "lp_jit_buffer data pointer must be at offset 0");
static_assert(sizeof(((struct lp_jit_buffer *)0)->u) == 8,
              "lp_jit_buffer data pointer must be 8 bytes");
static_assert(offsetof(struct lp_jit_buffer, num_elements) == 8,
              "lp_jit_buffer num_elements must be at offset 8");
static_assert(sizeof(((struct lp_jit_buffer *)0)->num_elements) == 4,
              "lp_jit_buffer num_elements must be a u32");
#include "vp_gfx_frontend.h"     /* embedded setup_k + binning_k .vxbin */
#include "VX_types.h"            /* VX_DCR_RASTER_*, VX_DCR_OM_*, VX_OM_* */
#include "util/log.h"

namespace graphics = vortex::graphics;

/* The on-device front end bins at the coarse 128 px bin log
 * (VX_config.h VX_CFG_RASTER_BIN_LOGSIZE) and the RASTER unit descends
 * bin -> block -> quad. Must match the front-end kernel's PIPE_BIN_LOG
 * (sw/gfx/pipe_abi.h). */
#ifndef RASTER_BIN_LOGSIZE
#define RASTER_BIN_LOGSIZE 7
#endif

/* sizeof(graphics::rast_prim_t): vec3e_t edges[3] (36B) + rast_attribs_t
 * {z,r,g,b,a,u,v,rhw,w0..w5} (14 planes x 12B = 168B) = 204B. The w0..w5 planes
 * extend the varying interpolation past six scalars (samplerCube textureGrad
 * carries 9); this stride programs the RASTER PBUF_STRIDE DCR, so it must match
 * the device rast_prim_t exactly or the unit reads primitive N>0 from the wrong
 * offset (its edges alias the prior prim's attribute planes). The RASTER unit
 * reads only the leading edges, so the extra planes cost stride only, not HW.
 * Keep in sync with VP_RAST_PRIM_STRIDE in vp_nir_to_llvm.c. */
#define VP_RAST_PRIM_STRIDE 204

/* Folded-in VS stage arg layout (mirrors vp_launch.c): arg slot 0 = VS output
 * buffer device address, slot 1 = attribute table indexed by VS input
 * driver_location with { device base, stride } entries. VP_ARG_SLOTS is the
 * shared kernel-ABI slot count from vp_nir_to_llvm.h. */
#define VP_ATTR_TABLE_LOCS  8

/* Hard error from the Vortex runtime — log as mesa_loge so the host /
 * test harness can detect it. Soft "feature not implemented" notices
 * remain mesa_logw; runtime API failures must not. */
#define VP_CHECK(call, what)                                            \
   do {                                                                 \
      vx_result_t _r = (call);                                          \
      if (_r != VX_SUCCESS) {                                           \
         mesa_loge("vortexpipe: raster: %s failed (%s)", (what),        \
                   vx_result_string(_r));                               \
         goto done;                                                     \
      }                                                                 \
   } while (0)

/* exact log2 of a power-of-two texture dimension; the bound keeps the shift
 * count in range, since a dimension above 2^31 would otherwise shift by 32 and
 * spin here forever */
static inline uint32_t
vp_log2u(uint32_t n)
{
   uint32_t l = 0;
   while (l < 32 && (1u << l) < n)
      l++;
   return l;
}

/* ---- persistent front-end working set -------------------------- *
 * The 17 binning buffers laid out once and reused across the frame's draws
 * instead of allocated per draw. prim + tilebuf (the RASTER AXI master's
 * inputs) are pinned over VX_MEM_PHYS; the rest is device-resident scratch.
 * The pool grows monotonically when a draw needs more capacity. addr caches
 * the device addresses (its count fields are unused — set per draw). */
struct vp_raster_pool {
   vx_buffer_h bufs[17];
   pipe_arg_t  addr;
   uint32_t    cap_tris;
   uint32_t    cap_bins;
   uint32_t    cap_keys;
   uint32_t    cap_T;
   /* Residency: the embedded front end (expand/setup/binning) loaded onto
    * the device once and reused across every draw (it never changes), instead
    * of reloaded per draw. */
   vx_module_h fe_module;
   vx_kernel_h k_expand, k_setup, k_binning;
   /* SW sampler / output-merger: small resident descriptors the FS reads
    * (tex via arg[1], om via arg[2]) when the unit is routed to software;
    * created once, rewritten per draw. */
   vx_buffer_h texstate_buf;
   uint64_t    texstate_dev;
   vx_buffer_h omstate_buf;
   uint64_t    omstate_dev;
};

extern "C" struct vp_raster_pool *
vp_raster_pool_create(void)
{
   return new (std::nothrow) vp_raster_pool{};
}

extern "C" void
vp_raster_pool_destroy(struct vp_raster_pool *pool)
{
   if (!pool)
      return;
   for (vx_buffer_h b : pool->bufs)
      if (b) vx_buffer_release(b);
   if (pool->texstate_buf) vx_buffer_release(pool->texstate_buf);
   if (pool->omstate_buf)  vx_buffer_release(pool->omstate_buf);
   if (pool->k_binning) vx_kernel_release(pool->k_binning);
   if (pool->k_setup)   vx_kernel_release(pool->k_setup);
   if (pool->k_expand)  vx_kernel_release(pool->k_expand);
   if (pool->fe_module) vx_module_release(pool->fe_module);
   delete pool;
}

/* Ensure the pool is sized for this draw (grow-only); (re)allocates and
 * re-caches addresses only when the request exceeds the current capacity.
 * Returns false on an allocation failure. */
static bool
vp_pool_ensure(vx_device_h dev, struct vp_raster_pool *pool,
               uint32_t num_tris, uint32_t num_bins,
               uint32_t keys_cap, uint32_t T)
{
   if (pool->bufs[0] &&
       num_tris <= pool->cap_tris && num_bins <= pool->cap_bins &&
       keys_cap <= pool->cap_keys && T == pool->cap_T)
      return true;   /* fits — reuse the resident set */

   for (vx_buffer_h &b : pool->bufs) { if (b) vx_buffer_release(b); b = NULL; }

   const uint32_t NT = num_tris > pool->cap_tris ? num_tris : pool->cap_tris;
   const uint32_t B  = num_bins > pool->cap_bins ? num_bins : pool->cap_bins;
   const uint32_t K  = keys_cap > pool->cap_keys ? keys_cap : pool->cap_keys;
   const uint32_t MS = SETUP_MAX_SUB;
   const uint32_t P_max = NT * MS;
   const size_t PRIM_SZ = sizeof(graphics::rast_prim_t);
   const size_t HDR_SZ  = sizeof(graphics::rast_bin_header_t);
   const size_t BBOX_SZ = sizeof(setup_bbox_t);
   const size_t TILEBUF_SZ = (size_t)B * HDR_SZ + (size_t)K * 4;

   const uint32_t W   = 0;                                          /* scratch */
   const uint32_t RWP = VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS;   /* RASTER-read */
   pipe_arg_t &a = pool->addr;
   struct { uint64_t bytes; uint64_t *addr; uint32_t flags; } spec[] = {
      { (uint64_t)3 * NT * sizeof(setup_vertex_t), &a.verts_addr,     W   },
      { (uint64_t)P_max * PRIM_SZ,                 &a.slot_prim_addr, W   },
      { (uint64_t)P_max * BBOX_SZ,                 &a.slot_bbox_addr, W   },
      { (uint64_t)NT * 4,                          &a.keep_addr,      W   },
      { (uint64_t)(NT + 1) * 4,                    &a.offset_addr,    W   },
      { (uint64_t)T * 4,                           &a.tsum_addr,      W   },
      { (uint64_t)P_max * PRIM_SZ,                 &a.prim_addr,      RWP },
      { (uint64_t)P_max * BBOX_SZ,                 &a.bbox_addr,      W   },
      { (uint64_t)P_max * 4,                       &a.bcount_addr,    W   },
      { (uint64_t)(P_max + 1) * 4,                 &a.boffset_addr,   W   },
      { (uint64_t)K * 4,                           &a.keys_addr,      W   },
      { (uint64_t)T * 4,                           &a.btsum_addr,     W   },
      { (uint64_t)T * B * 4,                       &a.thist_addr,     W   },
      { (uint64_t)B * 4,                           &a.bincount_addr,  W   },
      { (uint64_t)B * 4,                           &a.binbase_addr,   W   },
      { (uint64_t)TILEBUF_SZ,                      &a.tilebuf_addr,   RWP },
      { (uint64_t)3 * 4,                           &a.meta_addr,      W   },
   };
   for (uint32_t i = 0; i < 17; i++) {
      if (vx_buffer_create(dev, spec[i].bytes ? spec[i].bytes : 1,
                           spec[i].flags, &pool->bufs[i]) != VX_SUCCESS) {
         mesa_loge("vortexpipe: raster: front-end pool alloc failed");
         return false;
      }
      if (vx_buffer_address(pool->bufs[i], spec[i].addr) != VX_SUCCESS) {
         mesa_loge("vortexpipe: raster: front-end pool address failed");
         return false;
      }
   }
   pool->cap_tris = NT; pool->cap_bins = B; pool->cap_keys = K; pool->cap_T = T;
   return true;
}

extern "C" bool
vp_raster_draw(vx_device_h dev, struct vp_raster_pool *pool,
               const void *vs_vxbin, size_t vs_vxbin_size,
               vx_module_h *vs_module_io, vx_kernel_h *vs_kernel_io,
               const void *fs_vxbin, size_t fs_vxbin_size,
               vx_module_h *fs_module_io, vx_kernel_h *fs_kernel_io,
               uint32_t vertex_count,
               uint32_t first_vertex,
               uint32_t instance_count, uint32_t first_instance,
               const struct vp_vs_layout *layout,
               const struct vp_vertex_input *vin,
               uint64_t index_dev,
               uint64_t color_dev, uint64_t depth_dev,
               uint32_t width, uint32_t height,
               const struct vp_om_params *om,
               uint64_t tex_dev, const struct vp_tex_params *tex,
               uint32_t cull_mode,
               bool sw_tex, bool sw_om, bool sw_raster,
               float vp_sx, float vp_tx, float vp_sy, float vp_ty,
               float vp_min_z, float vp_max_z,
               const struct vp_fs_consts *fs_consts,
               const struct vp_fs_consts *vs_consts,
               const struct vp_mrt_params *mrt)
{
   /* A draw targeting >1 colour attachment merges in software (the FF
    * OM unit is single-attachment), so force the SW-OM path when MRT is active.
    * The FS was compiled for sw_om (vp_create_fs_state) to match. */
   const bool mrt_active = mrt && mrt->num_color > 1;
   if (mrt_active)
      sw_om = true;
   /* Instancing: the draw runs the whole VS -> expand -> setup -> bin -> raster
    * pipeline over instance_count × verts_per_instance vertices (each instance is
    * just more primitives — the true-GPU shape). verts_per_instance is the
    * per-instance vertex count (draws[0].count for a list, 3*tris for a strip);
    * the VS resolves gl_InstanceIndex from arg slots 3/4 (below). A single
    * instance keeps the pre-instancing counts exactly (fast path). */
   const uint32_t inst_count = instance_count ? instance_count : 1;
   const uint32_t verts_per_instance = vertex_count;
   const uint32_t total_verts = verts_per_instance * inst_count;
   const uint32_t num_tris = total_verts / 3;
   if (num_tris == 0) {
      mesa_logw("vortexpipe: raster: draw has no complete triangles");
      return false;
   }
   if (!pool) {
      mesa_loge("vortexpipe: raster: no front-end pool");
      return false;
   }

   /* The VS runs as stage 0 of this draw (below): its output buffer is sized
    * + allocated here and consumed in-place by expand_k — no separate
    * host-blocking VS launch, no readback, no host vertex expansion. */

   /* ---- dense tile grid, sized from the framebuffer ------------------- *
    * Bins == RASTER tiles, so the grid count (bin_cols*bin_rows) is the
    * RASTER TILE_COUNT — host-known, no num_tiles readback. The front end
    * emits one header per tile (empty tiles get pids_count=0). */
   const uint32_t BIN      = 1u << RASTER_BIN_LOGSIZE;
   const uint32_t bin_cols = (width  + BIN - 1) / BIN;
   const uint32_t bin_rows = (height + BIN - 1) / BIN;
   const uint32_t num_bins = bin_cols * bin_rows;
   const uint32_t MS       = SETUP_MAX_SUB;
   const uint32_t P_max    = num_tris * MS;          /* worst-case kept prims */

   /* keys[] holds one (bin,prim) entry per coarse 128 px bin a prim's bbox
    * covers; the exact count is meta[1], known only on-device. Size worst-case
    * (every kept prim covers every bin) so the launch chain needs no mid-frame
    * readback. The coarse bins make this far smaller than 32 px tiles, and
    * rast_bin_header_t's 32-bit fields lift the old 16-bit cap; tighter /
    * segmented allocation is a follow-up. */
   const uint32_t keys_cap = P_max && num_bins ? P_max * num_bins : 1;

   const size_t PRIM_SZ = sizeof(graphics::rast_prim_t);
   const size_t HDR_SZ  = sizeof(graphics::rast_bin_header_t);
   const size_t BBOX_SZ = sizeof(setup_bbox_t);
   const size_t TILEBUF_SZ = (size_t)num_bins * HDR_SZ + (size_t)keys_cap * 4;

   bool ok = false;
   vx_queue_h  q     = NULL;

   /* The front-end working set + colour/depth/texture buffers are resident
    * (pool- / context-owned); only the per-draw VS-output + vertex-input
    * buffers are owned here. */
   vx_buffer_h vsbuf = NULL, tbuf = NULL, fabuf = NULL;
   vx_buffer_h vbufs_dev[8] = { NULL };   /* one per distinct vertex buffer */
   /* Per-draw FS descriptor table + one device buffer per bound fragment
    * constant buffer (push constants / UBOs / descriptor blob). */
   vx_buffer_h fs_desc_buf = NULL;
   vx_buffer_h fs_cbuf_bufs[GFX_FS_DESC_SLOTS] = { NULL };
   vx_buffer_h fs_res_bufs[VP_MAX_DESCS] = { NULL };   /* UBO/SSBO resources */
   uint8_t    *fs_desc_stage[GFX_FS_DESC_SLOTS] = { NULL }; /* per-set relocated blob */
   /* The same set for the vertex stage. The VS overlays its own meanings on
    * arg-block slots 0-4, so it reaches its constant buffers only through its
    * own table, handed to it in VP_ARG_VS_DESC. */
   vx_buffer_h vs_desc_buf = NULL;
   vx_buffer_h vs_cbuf_bufs[GFX_FS_DESC_SLOTS] = { NULL };
   vx_buffer_h vs_res_bufs[VP_MAX_DESCS] = { NULL };
   uint8_t    *vs_desc_stage[GFX_FS_DESC_SLOTS] = { NULL };
   vx_buffer_h mrtbuf = NULL;                          /* gfx_sw_omcolor_t[] */

   /* kernels resolved from the residency caches below (the modules persist
    * across draws — caller-owned VS/FS slots + the pool's front end). */
   vx_kernel_h k_vs = NULL, kbuf = NULL, k_expand = NULL, k_setup = NULL,
               k_binning = NULL;

   {
      vx_queue_info_t qi = { sizeof(qi), NULL, VX_QUEUE_PRIORITY_NORMAL, 0 };
      VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");

      /* Residency: load each code module onto the device ONCE and reuse it
       * across draws. The FS (0x80000000), VS (VP_STARTUP_VS=0x80400000) and the
       * embedded front end (0x80200000) co-reside in three distinct slots, so
       * one OP_DRAW sequences VS -> expand -> setup -> binning -> raster -> FS
       * with no host module swap. The VS/FS modules cache in caller-owned CSO
       * slots (*_module_io); the front end caches on the pool. compile-once,
       * upload-resident-once — no /tmp round-trip, no per-draw reload. */
      if (*fs_module_io == NULL) {
         VP_CHECK(vx_module_load_bytes(dev, fs_vxbin, fs_vxbin_size, fs_module_io),
                  "vx_module_load_bytes(fs)");
         VP_CHECK(vx_module_get_kernel(*fs_module_io, "main", fs_kernel_io),
                  "vx_module_get_kernel(fs)");
      }
      kbuf = *fs_kernel_io;

      if (*vs_module_io == NULL) {
         VP_CHECK(vx_module_load_bytes(dev, vs_vxbin, vs_vxbin_size, vs_module_io),
                  "vx_module_load_bytes(vs)");
         VP_CHECK(vx_module_get_kernel(*vs_module_io, "main", vs_kernel_io),
                  "vx_module_get_kernel(vs)");
      }
      k_vs = *vs_kernel_io;

      if (pool->fe_module == NULL) {
         size_t fe_size = 0;
         const void *fe_bytes = vp_gfx_frontend_vxbin(vp_xlen_is_64(), &fe_size);
         VP_CHECK(vx_module_load_bytes(dev, fe_bytes, fe_size, &pool->fe_module),
                  "vx_module_load_bytes(frontend)");
         VP_CHECK(vx_module_get_kernel(pool->fe_module, "expand_k", &pool->k_expand),
                  "vx_module_get_kernel(expand_k)");
         VP_CHECK(vx_module_get_kernel(pool->fe_module, "setup_k", &pool->k_setup),
                  "vx_module_get_kernel(setup_k)");
         VP_CHECK(vx_module_get_kernel(pool->fe_module, "binning_k", &pool->k_binning),
                  "vx_module_get_kernel(binning_k)");
      }
      k_expand  = pool->k_expand;
      k_setup   = pool->k_setup;
      k_binning = pool->k_binning;

      uint64_t nt = 1, nw = 1, nc = 1;
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt), "vx_device_query(NUM_THREADS)");
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &nw), "vx_device_query(NUM_WARPS)");
      VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,   &nc), "vx_device_query(NUM_CORES)");

      /* VS launch geometry: one thread per vertex, rounded up to fill warps and
       * spread across cores (mirrors the former vp_launch_vs). The output is
       * padded for the trailing CTA's out-of-bounds threads; expand_k reads
       * only the first vertex_count records. */
      const uint32_t vstride  = layout->stride;
      const uint32_t cta_max  = (uint32_t)(nt * nw);
      uint32_t vs_block = (total_verts + (uint32_t)nt - 1u) / (uint32_t)nt
                        * (uint32_t)nt;
      if (vs_block > cta_max) vs_block = cta_max;
      if (vs_block == 0)      vs_block = (uint32_t)nt;
      const uint32_t vs_grid    = (total_verts + vs_block - 1u) / vs_block;
      const uint32_t vs_threads = vs_grid * vs_block;
      const uint32_t vsout_bytes = vs_threads * vstride;
      VP_CHECK(vx_buffer_create(dev, vsout_bytes ? vsout_bytes : 1, 0, &vsbuf),
               "vx_buffer_create(vsout)");
      uint64_t vs_out_dev = 0;
      VP_CHECK(vx_buffer_address(vsbuf, &vs_out_dev), "vx_buffer_address(vsout)");

      /* device-filling launch geometry: T = per-CTA threads, G = grid. */
      uint32_t one = 1, gdim[1] = { 1 }, bdim[1] = { 1 };
      VP_CHECK(vx_device_max_occupancy_grid(dev, 1, &one, gdim, bdim),
               "vx_device_max_occupancy_grid");
      const uint32_t T = bdim[0];
      const uint32_t G = gdim[0];

      /* Lay out (or reuse) the persistent front-end working set, then build
       * the shared pipe_arg_t from the pool's cached device addresses plus
       * this draw's counts. */
      if (!vp_pool_ensure(dev, pool, num_tris, num_bins, keys_cap, T)) {
         mesa_loge("vortexpipe: raster: front-end pool sizing failed");
         goto done;
      }
      pipe_arg_t arg = pool->addr;               /* device addresses */
      arg.num_tris   = num_tris;
      arg.stage      = 0;
      arg.width      = width;
      arg.height     = height;
      arg.bin_stripe = (num_bins + G - 1) / G;   /* bins per CTA (HIST/SCATTER) */
      arg.bin_cols   = bin_cols;
      arg.num_bins   = num_bins;
      arg.cull_mode  = cull_mode;                /* SETUP_CULL_* from the rasterizer state */
      arg.vp_sx = vp_sx; arg.vp_tx = vp_tx;      /* app viewport transform (screen = ndc*scale+bias) */
      arg.vp_sy = vp_sy; arg.vp_ty = vp_ty;      /* negative vp_sy = Vulkan y-flip */
      arg.vp_minz = vp_min_z; arg.vp_maxz = vp_max_z;
      /* Vulkan clip z spans [0,w], so the front end must use the Vulkan near
       * plane and depth map rather than the GL ones the native path feeds it. */
      arg.vp_halfz = 1u;

      /* expand_k arg: on-device vertex assembly expands the resident VS
       * records into the setup_vertex_t[] the front end consumes
       * (arg.verts_addr). */
      expand_arg_t earg{};
      earg.vsrec_addr   = vs_out_dev;        /* the folded-in VS stage's output */
      earg.verts_addr   = arg.verts_addr;
      earg.num_verts    = total_verts;       /* instance_count × verts_per_instance */
      earg.vstride      = layout->stride;
      uint32_t nvary = layout->num_varyings;
      if (nvary > EXPAND_MAX_VARYINGS) nvary = EXPAND_MAX_VARYINGS;
      earg.num_varyings = nvary;
      for (uint32_t i = 0; i < nvary; i++)
         earg.varying_comps[i] = layout->varying_comps[i];

      /* Colour + depth are render-pass-resident (color_dev / depth_dev) and the
       * texture is residency-cached (tex_dev) — all owned + uploaded once by the
       * caller. The OM/TEX units reach them through their DCRs; there is no
       * per-draw colour upload/readback, depth clear, or texture upload here.
       * Only the per-draw VS output + vertex inputs are staged below. */
      uint64_t prim_dev = arg.prim_addr;
      uint64_t tile_dev = arg.tilebuf_addr;

      /* the FS kernel needs the primitive buffer (arg[0]); SW units add the
       * texstate/omstate descriptors (arg[1]/[2]); SW raster adds the tile-walk
       * counts (arg[3..8]); and the resident FS descriptor table
       * (arg[GFX_FS_ARG_DESC]). The HW path reaches colour/depth via the OM DCRs. */
      uint64_t argblk[GFX_FS_ARG_APERTURE + 1] = { 0 };
      argblk[0] = prim_dev;

      /* OM aperture geometry: pad the render target to a power of two on each axis
       * so a fragment export's address is a shift, not a multiply. record_shift 3 =
       * an 8-byte {colour, depth} record. vp_log2u rounds up. */
      const uint32_t ap_xbits = vp_log2u(width);
      const uint32_t ap_ybits = vp_log2u(height);
      const uint32_t ap_shift = 3u;
      argblk[GFX_FS_ARG_APERTURE] =
         GFX_FS_APERTURE_PACK(ap_xbits, ap_ybits, ap_shift);

      /* Build the resident FS descriptor table — one device buffer per bound
       * fragment constant buffer (push constants, UBOs, and the descriptor set
       * blob), then a small i64[GFX_FS_DESC_SLOTS] table of their device base
       * addresses. The fragment kernel reads it via load_ubo/load_push_constant
       * (table[i]+off, direct) and load_ssbo (deref the blob at table[1]). The
       * table is always supplied (zero-filled when descriptor-free) so the FS's
       * table reads never dereference a null arg pointer. Per-dispatch + resident.
       * UBO/SSBO descriptor-pointer relocation is implemented below;
       * only AS descriptors (ray-query-in-FS) remain a follow-up. */
      /* Each descriptor set N's blob is the fragment constant buffer at index N+1
       * (lavapipe binds descriptor set N at index N+1). A UBO/SSBO descriptor's
       * lp_jit_buffer places the data pointer at +0 and the byte size at +8 —
       * pinned by the static_assert above so a llvmpipe layout change fails the
       * build instead of silently corrupting every UBO/SSBO base. */
      const uint32_t VP_JIT_BUF_PTR = 0u, VP_JIT_BUF_SIZE = 8u;
      uint64_t fs_desc_table[GFX_FS_DESC_SLOTS] = { 0 };
      for (uint32_t i = 0; fs_consts && i < GFX_FS_DESC_SLOTS; i++) {
         if (!fs_consts->data[i] || !fs_consts->size[i])
            continue;
         const void *upload = fs_consts->data[i];
         /* Relocate EVERY set's descriptor blob, not just set 0. For each
          * buffer descriptor whose cbuf_index selects THIS blob, upload the
          * resource to device memory + rewrite its lp_jit_buffer.ptr host->device
          * so the FS's load_ubo/load_ssbo dereference resolves on device. Done on
          * a per-set private copy (fs_desc_stage[i]) that outlives vx_queue_finish.
          * AS descriptors are left untouched (ray-query-in-FS is deferred). */
         bool has_desc = false;
         for (uint32_t d = 0; d < fs_consts->num_descs && d < VP_MAX_DESCS; d++)
            if (fs_consts->descs[d].kind == VP_DESC_BUFFER &&
                fs_consts->descs[d].cbuf_index == i) { has_desc = true; break; }
         if (has_desc) {
            fs_desc_stage[i] = (uint8_t *)malloc(fs_consts->size[i]);
            if (!fs_desc_stage[i]) { mesa_loge("vortexpipe: raster: fs desc OOM"); goto done; }
            memcpy(fs_desc_stage[i], fs_consts->data[i], fs_consts->size[i]);
            for (uint32_t d = 0; d < fs_consts->num_descs && d < VP_MAX_DESCS; d++) {
               if (fs_consts->descs[d].kind != VP_DESC_BUFFER ||
                   fs_consts->descs[d].cbuf_index != i)
                  continue;
               uint32_t off = fs_consts->descs[d].offset;
               if (off + VP_JIT_BUF_SIZE + 4u > fs_consts->size[i])
                  continue;
               uint64_t host_ptr = 0; uint32_t num_elems = 0;
               memcpy(&host_ptr,  fs_desc_stage[i] + off + VP_JIT_BUF_PTR,  sizeof host_ptr);
               memcpy(&num_elems, fs_desc_stage[i] + off + VP_JIT_BUF_SIZE, sizeof num_elems);
               /* byte size = num_elements × its unit (UBO: dwords, SSBO: bytes). */
               uint32_t rsz = num_elems * (fs_consts->descs[d].elem_bytes
                                           ? fs_consts->descs[d].elem_bytes : 1u);
               if (!host_ptr || !rsz)
                  continue;
               VP_CHECK(vx_buffer_create(dev, rsz, 0, &fs_res_bufs[d]),
                        "vx_buffer_create(fs_res)");
               uint64_t res_dev = 0;
               VP_CHECK(vx_buffer_address(fs_res_bufs[d], &res_dev),
                        "vx_buffer_address(fs_res)");
               VP_CHECK(vx_enqueue_write(q, fs_res_bufs[d], 0,
                                         (void *)(uintptr_t)host_ptr, rsz, 0, NULL, NULL),
                        "vx_enqueue_write(fs_res)");
               memcpy(fs_desc_stage[i] + off + VP_JIT_BUF_PTR, &res_dev, sizeof res_dev);
            }
            upload = fs_desc_stage[i];
         }
         VP_CHECK(vx_buffer_create(dev, fs_consts->size[i], 0, &fs_cbuf_bufs[i]),
                  "vx_buffer_create(fs_cbuf)");
         uint64_t cb_dev = 0;
         VP_CHECK(vx_buffer_address(fs_cbuf_bufs[i], &cb_dev),
                  "vx_buffer_address(fs_cbuf)");
         VP_CHECK(vx_enqueue_write(q, fs_cbuf_bufs[i], 0, upload,
                                   fs_consts->size[i], 0, NULL, NULL),
                  "vx_enqueue_write(fs_cbuf)");
         fs_desc_table[i] = cb_dev;
      }
      VP_CHECK(vx_buffer_create(dev, sizeof(fs_desc_table), 0, &fs_desc_buf),
               "vx_buffer_create(fs_desc)");
      uint64_t fs_desc_dev = 0;
      VP_CHECK(vx_buffer_address(fs_desc_buf, &fs_desc_dev),
               "vx_buffer_address(fs_desc)");
      VP_CHECK(vx_enqueue_write(q, fs_desc_buf, 0, fs_desc_table,
                                sizeof(fs_desc_table), 0, NULL, NULL),
               "vx_enqueue_write(fs_desc)");
      argblk[GFX_FS_ARG_DESC] = fs_desc_dev;

      /* SW sampler: build the resident texture descriptor and hand the FS its
       * device address (arg[1]). The FS (compiled with sw_tex) then samples via
       * gfx_tex_sample_sw over the LSU instead of the FF TEX unit. The descriptor
       * outlives vx_enqueue_draw / vx_queue_finish (declared in this scope). */
      /* Build the resident descriptor whenever a texture is bound, not only for
       * sw_tex: the HW-tex FS reads logdim from it to compute the mip LOD
       * (vx_tex_auto_lod), since the TEX DCRs are host-write-only. The SW sampler
       * additionally consumes format/filter/wrap/mip_off/base from it. */
      gfx_sw_texstate_t texstate{};
      if (tex_dev && tex) {
         texstate.base   = tex_dev;
         for (uint32_t i = 0; i <= (uint32_t)VX_TEX_LOD_MAX; ++i)
            texstate.mip_off[i] = tex->mip_off[i];
         texstate.logdim = (vp_log2u(tex->height) << 16) | vp_log2u(tex->width);
         /* Depth formats (sampler2DShadow) carry the real VX format so the SW
          * sampler reads/compares raw depth; colour textures stay A8R8G8B8. */
         texstate.format = tex->format ? tex->format : VX_TEX_FORMAT_A8R8G8B8;
         texstate.compare_func = tex->compare_func;
         texstate.swizzle = tex->swizzle;
         texstate.layer_stride = tex->layer_stride;
         texstate.min_lod  = tex->min_lod;
         texstate.max_lod  = tex->max_lod;
         texstate.lod_bias = tex->lod_bias;
         texstate.depth    = tex->depth;
         texstate.wrap_w   = tex->wrap_w;
         /* Descriptor-only filter bits (the FS reads them; they never reach the HW
          * TEX_FILTER DCR below): mag tap (bit0, from tex->filter), min tap (bit3),
          * mip-linear (bit1), mip-enable (bit2), NPOT (bit4) and extended format
          * (bit5). A mipmapped, non-power-of-two OR extended-format sampler routes
          * to the SW sampler, which resolves min-vs-mag per fragment from these,
          * addresses NPOT by width/height, and decodes the extended formats the FF
          * unit has neither a decoder nor a texel stride for. */
         const bool tex_pot = tex->width && tex->height
            && !(tex->width & (tex->width - 1u)) && !(tex->height & (tex->height - 1u));
         texstate.filter = tex->filter
            | (tex->min_filter == VX_TEX_FILTER_BILINEAR ? GFX_SW_TEX_FILTER_MIN_BILINEAR : 0u)
            | (tex->mip_linear ? VX_TEX_FILTER_MIP_LINEAR : 0u)
            | (tex->mip_enable ? GFX_SW_TEX_FILTER_MIP_ENABLE : 0u)
            | (tex_pot ? 0u : GFX_SW_TEX_FILTER_NPOT)
            | (texstate.format > (uint32_t)VX_TEX_FORMAT_FF_MAX
                  ? GFX_SW_TEX_FILTER_EXT_FORMAT : 0u)
            | ((tex->wrap_u == VX_TEX_WRAP_BORDER ||
                tex->wrap_v == VX_TEX_WRAP_BORDER)
                  ? GFX_SW_TEX_FILTER_BORDER : 0u);
         texstate.wrap   = (tex->wrap_v << 16) | tex->wrap_u;
         texstate.border = tex->border;
         /* Carry the mip-0 integer dims so the SW sampler can address NPOT
          * textures (multiply addressing). POT textures leave width==1<<logdim
          * so the sampler keeps the bit-exact shift path. */
         texstate.width  = tex->width;
         texstate.height = tex->height;
         if (!pool->texstate_buf) {
            VP_CHECK(vx_buffer_create(dev, sizeof(gfx_sw_texstate_t), 0,
                                      &pool->texstate_buf), "vx_buffer_create(texstate)");
            VP_CHECK(vx_buffer_address(pool->texstate_buf, &pool->texstate_dev),
                     "vx_buffer_address(texstate)");
         }
         VP_CHECK(vx_enqueue_write(q, pool->texstate_buf, 0, &texstate,
                                   sizeof(texstate), 0, NULL, NULL),
                  "vx_enqueue_write(texstate)");
         argblk[1] = pool->texstate_dev;
      }

      /* SW output-merger: build the resident om descriptor from the same OM
       * state the FF path programs into the OM DCRs (depth/blend/colormask +
       * the resident colour/depth buffers), with the enable flags derived as
       * gfx_sw::resolve_om_state does. The FS (compiled with sw_om) merges each
       * fragment via gfx_om_fragment_sw over the LSU using arg[2]. */
      gfx_sw_omstate_t omstate{};
      if (sw_om) {
         omstate.depth_func      = om->depth_test ? om->depth_func
                                                  : (uint32_t)VX_OM_DEPTH_FUNC_ALWAYS;
         omstate.depth_writemask = om->depth_write ? 1u : 0u;
         for (int f = 0; f < 2; ++f) {
            /* Unpack the same front/back halves the FF DCRs carry, so a draw
             * that routes to the software merger produces identical pixels. */
            const int sh = f * 16;
            omstate.stencil_func[f]      = (om->stencil_func      >> sh) & 0xffff;
            omstate.stencil_zpass[f]     = (om->stencil_zpass     >> sh) & 0xffff;
            omstate.stencil_zfail[f]     = (om->stencil_zfail     >> sh) & 0xffff;
            omstate.stencil_fail[f]      = (om->stencil_fail      >> sh) & 0xffff;
            omstate.stencil_ref[f]       = (om->stencil_ref       >> sh) & 0xffff;
            omstate.stencil_mask[f]      = (om->stencil_mask      >> sh) & 0xffff;
            omstate.stencil_writemask[f] = (om->stencil_writemask >> sh) & 0xffff;
         }
         omstate.blend_mode_rgb = om->blend_mode & 0xffff;
         omstate.blend_mode_a   = om->blend_mode >> 16;
         omstate.blend_src_rgb  =  om->blend_func        & 0xff;
         omstate.blend_src_a    = (om->blend_func >> 8)  & 0xff;
         omstate.blend_dst_rgb  = (om->blend_func >> 16) & 0xff;
         omstate.blend_dst_a    = (om->blend_func >> 24) & 0xff;
         omstate.blend_const    = om->blend_const;
         omstate.logic_op       = om->logic_op;
         omstate.zbuf_base      = depth_dev;
         omstate.cbuf_base      = color_dev;
         /* The merger addresses a pixel's samples contiguously within its row,
          * so the pitch it reads is the multisample row stride, not the pixel
          * row width. At one sample the two are the same. */
         omstate.zbuf_pitch     = width * 4 * om->samples;
         omstate.cbuf_pitch     = width * 4 * om->samples;
         omstate.cbuf_writemask4 = om->colormask;
         /* resolve_om_state (host-side derivation, mirrors gfx_sw.h) */
         omstate.depth_enabled = !((omstate.depth_func == (uint32_t)VX_OM_DEPTH_FUNC_ALWAYS)
                                && !(omstate.depth_writemask & 1u));
         for (int f = 0; f < 2; ++f) {
            omstate.stencil_enabled[f] =
               !((omstate.stencil_func[f]  == (uint32_t)VX_OM_DEPTH_FUNC_ALWAYS)
              && (omstate.stencil_zpass[f] == (uint32_t)VX_OM_STENCIL_OP_KEEP)
              && (omstate.stencil_zfail[f] == (uint32_t)VX_OM_STENCIL_OP_KEEP));
         }
         omstate.blend_enabled = !((omstate.blend_mode_rgb == (uint32_t)VX_OM_BLEND_MODE_ADD)
                                && (omstate.blend_mode_a   == (uint32_t)VX_OM_BLEND_MODE_ADD)
                                && (omstate.blend_src_rgb  == (uint32_t)VX_OM_BLEND_FUNC_ONE)
                                && (omstate.blend_src_a    == (uint32_t)VX_OM_BLEND_FUNC_ONE)
                                && (omstate.blend_dst_rgb  == (uint32_t)VX_OM_BLEND_FUNC_ZERO)
                                && (omstate.blend_dst_a    == (uint32_t)VX_OM_BLEND_FUNC_ZERO));
         uint32_t m4 = om->colormask & 0xf;
         omstate.cbuf_writemask = (((m4 >> 0) & 1) * 0x000000ffu) | (((m4 >> 1) & 1) * 0x0000ff00u)
                                | (((m4 >> 2) & 1) * 0x00ff0000u) | (((m4 >> 3) & 1) * 0xff000000u);
         omstate.color_read  = (m4 != 0xf);
         omstate.color_write = (m4 != 0x0);
         if (!pool->omstate_buf) {
            VP_CHECK(vx_buffer_create(dev, sizeof(gfx_sw_omstate_t), 0,
                                      &pool->omstate_buf), "vx_buffer_create(omstate)");
            VP_CHECK(vx_buffer_address(pool->omstate_buf, &pool->omstate_dev),
                     "vx_buffer_address(omstate)");
         }
         VP_CHECK(vx_enqueue_write(q, pool->omstate_buf, 0, &omstate,
                                   sizeof(omstate), 0, NULL, NULL),
                  "vx_enqueue_write(omstate)");
         argblk[2] = pool->omstate_dev;
      }

      /* Build the resident per-attachment colour descriptor array the FS
       * reads (arg[GFX_FS_ARG_MRT]) when it targets >1 colour attachment. The
       * shared depth/stencil state stays in the omstate above; each attachment
       * carries its own base/pitch/blend/write-mask. The enable flags + expanded
       * mask are derived exactly as gfx_sw::resolve_om_color does. */
      /* Declared at draw scope (not inside the if) so it outlives the async
       * vx_enqueue_write below — the runtime copies it at vx_queue_finish, like
       * omstate/texstate above. A block-local array would dangle by then. */
      gfx_sw_omcolor_t rt[GFX_OM_MAX_RT] = {};
      if (mrt_active) {
         uint32_t nrt = mrt->num_color;
         if (nrt > GFX_OM_MAX_RT) nrt = GFX_OM_MAX_RT;
         for (uint32_t k = 0; k < nrt; ++k) {
            rt[k].cbuf_base      = mrt->color_dev[k];
            rt[k].cbuf_pitch     = mrt->pitch[k];
            rt[k].blend_mode_rgb = mrt->blend_mode[k] & 0xffff;
            rt[k].blend_mode_a   = mrt->blend_mode[k] >> 16;
            rt[k].blend_src_rgb  =  mrt->blend_func[k]        & 0xff;
            rt[k].blend_src_a    = (mrt->blend_func[k] >> 8)  & 0xff;
            rt[k].blend_dst_rgb  = (mrt->blend_func[k] >> 16) & 0xff;
            rt[k].blend_dst_a    = (mrt->blend_func[k] >> 24) & 0xff;
            /* Both are pipeline-wide in Vulkan, so every attachment carries the
             * same values the scalar path uses. */
            rt[k].blend_const    = om->blend_const;
            rt[k].logic_op       = om->logic_op;
            rt[k].cbuf_writemask4 = mrt->colormask[k];
            rt[k].blend_enabled  = !((rt[k].blend_mode_rgb == (uint32_t)VX_OM_BLEND_MODE_ADD)
                                  && (rt[k].blend_mode_a   == (uint32_t)VX_OM_BLEND_MODE_ADD)
                                  && (rt[k].blend_src_rgb  == (uint32_t)VX_OM_BLEND_FUNC_ONE)
                                  && (rt[k].blend_src_a    == (uint32_t)VX_OM_BLEND_FUNC_ONE)
                                  && (rt[k].blend_dst_rgb  == (uint32_t)VX_OM_BLEND_FUNC_ZERO)
                                  && (rt[k].blend_dst_a    == (uint32_t)VX_OM_BLEND_FUNC_ZERO));
            uint32_t m4 = mrt->colormask[k] & 0xf;
            rt[k].cbuf_writemask = (((m4 >> 0) & 1) * 0x000000ffu) | (((m4 >> 1) & 1) * 0x0000ff00u)
                                 | (((m4 >> 2) & 1) * 0x00ff0000u) | (((m4 >> 3) & 1) * 0xff000000u);
            rt[k].color_read  = (m4 != 0xf);
            rt[k].color_write = (m4 != 0x0);
         }
         VP_CHECK(vx_buffer_create(dev, sizeof(gfx_sw_omcolor_t) * nrt, 0, &mrtbuf),
                  "vx_buffer_create(mrt)");
         uint64_t mrt_dev = 0;
         VP_CHECK(vx_buffer_address(mrtbuf, &mrt_dev), "vx_buffer_address(mrt)");
         VP_CHECK(vx_enqueue_write(q, mrtbuf, 0, rt, sizeof(gfx_sw_omcolor_t) * nrt,
                                   0, NULL, NULL), "vx_enqueue_write(mrt)");
         argblk[GFX_FS_ARG_MRT] = mrt_dev;
      }

      /* VS vertex-input upload + attribute table (VS arg slot 1; slot 0 = VS
       * output). The table is indexed by VS input driver_location. Both the
       * table and the VS arg block are declared here so they outlive
       * vx_enqueue_draw (async uploads + submit-time arg copy). */
      uint32_t vs_attr_table[VP_ATTR_TABLE_LOCS * 2] = { 0 };
      uint64_t vs_argblk[VP_ARG_SLOTS] = { 0 };
      vs_argblk[0] = vs_out_dev;
      if (vin && vin->num_attrs) {
         /* Upload each distinct vertex-buffer resource once; an attribute then
          * points at its own buffer's device base + its byte offset. */
         uint64_t buf_dev[8] = { 0 };
         for (uint32_t j = 0; j < vin->num_bufs; j++) {
            VP_CHECK(vx_buffer_create(dev, vin->buf_size[j], 0, &vbufs_dev[j]),
                     "vx_buffer_create(vbuf)");
            VP_CHECK(vx_buffer_address(vbufs_dev[j], &buf_dev[j]),
                     "vx_buffer_address(vbuf)");
            VP_CHECK(vx_enqueue_write(q, vbufs_dev[j], 0, vin->buf_data[j],
                                      vin->buf_size[j], 0, NULL, NULL),
                     "vx_enqueue_write(vbuf)");
         }
         for (uint32_t i = 0; i < vin->num_attrs; i++) {
            uint32_t loc = vin->attr_loc[i];
            if (loc >= VP_ATTR_TABLE_LOCS) continue;
            vs_attr_table[loc * 2 + 0] = (uint32_t)buf_dev[vin->attr_buf[i]]
                                       + vin->attr_offset[i];
            vs_attr_table[loc * 2 + 1] = vin->attr_stride[i];
         }
         VP_CHECK(vx_buffer_create(dev, sizeof(vs_attr_table), 0, &tbuf),
                  "vx_buffer_create(attrtab)");
         uint64_t tbuf_dev = 0;
         VP_CHECK(vx_buffer_address(tbuf, &tbuf_dev), "vx_buffer_address(attrtab)");
         VP_CHECK(vx_enqueue_write(q, tbuf, 0, vs_attr_table, sizeof(vs_attr_table),
                                   0, NULL, NULL), "vx_enqueue_write(attrtab)");
         vs_argblk[1] = tbuf_dev;
      }
      /* slot 2: index buffer (u32 index per vertex); 0 = direct (non-indexed)
       * draw, in which case the VS uses its sequential global id as the vid. */
      vs_argblk[2] = index_dev;
      /* Slots 3/4: verts-per-instance + base instance. verts-per-instance is 0
       * for a plain single-instance draw (base 0) so the VS takes the byte-
       * identical non-instanced path (instance = 0, vid = sequential id). When
       * instancing is active it is the per-instance vertex count, so the VS splits
       * gid into instance = gid/vpi and in-instance vertex gid%vpi and resolves
       * gl_InstanceIndex = base + instance. */
      const bool instanced = (inst_count > 1) || (first_instance != 0);
      vs_argblk[3] = instanced ? verts_per_instance : 0;
      vs_argblk[4] = first_instance;
      /* Slot 5: base vertex. Vulkan defines gl_VertexIndex as firstVertex + i,
       * but the vid the VS computes is the 0-based draw position because the
       * host already folded firstVertex into the attribute base addresses. The
       * shader-visible index therefore needs the base back, and only there --
       * adding it to the vid would offset the attribute fetch twice. Zero for
       * an indexed draw, where the vid is already the absolute index value. */
      vs_argblk[5] = index_dev ? 0u : first_vertex;

      /* slot VP_ARG_VS_DESC: the vertex stage's constant-buffer table, built
       * exactly like the fragment one above -- upload each bound VS constant
       * buffer, relocating any UBO/SSBO descriptor's lp_jit_buffer.ptr
       * host->device so the VS's load_ubo/load_ssbo dereference resolves on
       * device, then a table of their device base addresses. Always uploaded,
       * zero-filled when the VS binds nothing, so the VS's table reads never
       * dereference a null pointer. Slots 0-4 carry vertex meanings, so without
       * this table a VS UBO read resolves against the attribute table and a
       * push-constant read against the VS output buffer. */
      uint64_t vs_desc_table[GFX_FS_DESC_SLOTS] = { 0 };
      for (uint32_t i = 0; vs_consts && i < GFX_FS_DESC_SLOTS; i++) {
         if (!vs_consts->data[i] || !vs_consts->size[i])
            continue;
         const void *upload = vs_consts->data[i];
         bool has_desc = false;
         for (uint32_t d = 0; d < vs_consts->num_descs && d < VP_MAX_DESCS; d++)
            if (vs_consts->descs[d].kind == VP_DESC_BUFFER &&
                vs_consts->descs[d].cbuf_index == i) { has_desc = true; break; }
         if (has_desc) {
            vs_desc_stage[i] = (uint8_t *)malloc(vs_consts->size[i]);
            if (!vs_desc_stage[i]) { mesa_loge("vortexpipe: raster: vs desc OOM"); goto done; }
            memcpy(vs_desc_stage[i], vs_consts->data[i], vs_consts->size[i]);
            for (uint32_t d = 0; d < vs_consts->num_descs && d < VP_MAX_DESCS; d++) {
               if (vs_consts->descs[d].kind != VP_DESC_BUFFER ||
                   vs_consts->descs[d].cbuf_index != i)
                  continue;
               uint32_t off = vs_consts->descs[d].offset;
               if (off + VP_JIT_BUF_SIZE + 4u > vs_consts->size[i])
                  continue;
               uint64_t host_ptr = 0; uint32_t num_elems = 0;
               memcpy(&host_ptr,  vs_desc_stage[i] + off + VP_JIT_BUF_PTR,  sizeof host_ptr);
               memcpy(&num_elems, vs_desc_stage[i] + off + VP_JIT_BUF_SIZE, sizeof num_elems);
               uint32_t rsz = num_elems * (vs_consts->descs[d].elem_bytes
                                           ? vs_consts->descs[d].elem_bytes : 1u);
               if (!host_ptr || !rsz)
                  continue;
               VP_CHECK(vx_buffer_create(dev, rsz, 0, &vs_res_bufs[d]),
                        "vx_buffer_create(vs_res)");
               uint64_t res_dev = 0;
               VP_CHECK(vx_buffer_address(vs_res_bufs[d], &res_dev),
                        "vx_buffer_address(vs_res)");
               VP_CHECK(vx_enqueue_write(q, vs_res_bufs[d], 0,
                                         (void *)(uintptr_t)host_ptr, rsz, 0, NULL, NULL),
                        "vx_enqueue_write(vs_res)");
               memcpy(vs_desc_stage[i] + off + VP_JIT_BUF_PTR, &res_dev, sizeof res_dev);
            }
            upload = vs_desc_stage[i];
         }
         VP_CHECK(vx_buffer_create(dev, vs_consts->size[i], 0, &vs_cbuf_bufs[i]),
                  "vx_buffer_create(vs_cbuf)");
         uint64_t cb_dev = 0;
         VP_CHECK(vx_buffer_address(vs_cbuf_bufs[i], &cb_dev),
                  "vx_buffer_address(vs_cbuf)");
         VP_CHECK(vx_enqueue_write(q, vs_cbuf_bufs[i], 0, upload,
                                   vs_consts->size[i], 0, NULL, NULL),
                  "vx_enqueue_write(vs_cbuf)");
         vs_desc_table[i] = cb_dev;
      }
      VP_CHECK(vx_buffer_create(dev, sizeof(vs_desc_table), 0, &vs_desc_buf),
               "vx_buffer_create(vs_desc)");
      uint64_t vs_desc_dev = 0;
      VP_CHECK(vx_buffer_address(vs_desc_buf, &vs_desc_dev),
               "vx_buffer_address(vs_desc)");
      VP_CHECK(vx_enqueue_write(q, vs_desc_buf, 0, vs_desc_table,
                                sizeof(vs_desc_table), 0, NULL, NULL),
               "vx_enqueue_write(vs_desc)");
      vs_argblk[VP_ARG_VS_DESC] = vs_desc_dev;

      /* FS launch fills every HW lane: block = threads × warps (one CTA/core),
       * grid = cores. (nt/nw/nc queried above for the VS geometry.)
       * Launch descriptors + per-stage args must outlive vx_enqueue_draw
       * (the runtime copies them at submit). */
      const uint32_t NSTAGE = 9;
      const uint32_t sgrid[NSTAGE] = { G, 1, G,  G, 1, G,  G, 1, G };
      pipe_arg_t       kargs[NSTAGE];
      vx_launch_info_t vsli{}, eli{}, li[NSTAGE], fli{};

      /* stage 0: vertex shader (one thread per vertex). */
      vsli.struct_size = sizeof(vsli); vsli.kernel = k_vs;
      vsli.args_host = vs_argblk; vsli.args_size = sizeof(vs_argblk);
      vsli.ndim = 1; vsli.grid_dim[0] = vs_grid; vsli.block_dim[0] = vs_block;

      eli.struct_size = sizeof(eli); eli.kernel = k_expand;
      eli.args_host = &earg; eli.args_size = sizeof(earg);
      eli.ndim = 1; eli.grid_dim[0] = G; eli.block_dim[0] = T;

      for (uint32_t s = 0; s < NSTAGE; ++s) {
         kargs[s] = arg; kargs[s].stage = s;
         li[s] = vx_launch_info_t{};
         li[s].struct_size = sizeof(li[s]);
         li[s].kernel      = (s < PIPE_STAGE_BCOUNT) ? k_setup : k_binning;
         li[s].args_host   = &kargs[s];
         li[s].args_size   = sizeof(pipe_arg_t);
         li[s].ndim        = 1;
         li[s].grid_dim[0]  = sgrid[s];
         li[s].block_dim[0] = T;
      }

      fli.struct_size = sizeof(fli); fli.kernel = kbuf;
      fli.args_host = argblk; fli.args_size = sizeof(argblk);
      fli.ndim = 1;
      if (sw_raster) {
         /* SW raster: the FS is the one-warp-per-tile kernel, not the HW frag-window
          * poll loop. Launch one warp per 8x8 screen tile (matching the FS
          * VP_SW_RAST_TILE_LOG); its lanes walk every prim over that tile in draw
          * order. The dense rast_prim_t[] is still front-end-produced; the RASTER
          * DCRs / FF producer are skipped below. */
         const uint32_t SW_TILE_LOG = 3;            /* 8x8 px (FS-side constant) */
         const uint32_t SW_TILE     = 1u << SW_TILE_LOG;
         const uint32_t nx_sw = (width  + SW_TILE - 1) / SW_TILE;
         const uint32_t ny_sw = (height + SW_TILE - 1) / SW_TILE;
         const uint32_t num_tiles_sw = nx_sw * ny_sw;
         /* One WARP per tile: block = NT threads (= one warp), grid = one CTA per
          * tile. The warp's NT lanes cooperate on the tile's quads -- one lane is one
          * pixel and a quad is four adjacent lanes, so lane L takes corner L&3 of
          * quad L>>2 -- the CudaRaster fine-rasterizer mapping. This keeps the FS
          * kernel's loops uniform-trip across the warp (the divergence-safe shape —
          * the earlier one-thread-per-tile mapping diverged per-lane on the
          * covered-quad count and the SIMT reconvergence dropped fragments). Full
          * occupancy: every warp/core runs, lanes shade in parallel. */
         assert(nt >= 4 && (nt % 4) == 0 &&
                "a pixel quad occupies four adjacent lanes, so a warp must hold whole quads");
         fli.grid_dim[0]  = num_tiles_sw;     /* one warp (CTA) per screen tile */
         fli.block_dim[0] = (uint32_t)nt;     /* NT threads = exactly one warp   */
         /* Prim count = the front end's kept-prim count P (meta[0]), NOT num_tris:
          * clip + face cull make the dense primbuf shorter than the input, and P
          * is known only on-device (this draw is one CP batch, no host round-trip).
          * Hand the FS the meta buffer address; it loads meta[0] at runtime so it
          * walks exactly the kept prims — a culled draw walks zero (never the stale
          * prim slot the reused pool still holds from a prior draw). */
         argblk[3] = arg.meta_addr;
         argblk[4] = nx_sw;
         argblk[5] = num_tiles_sw;
         argblk[6] = SW_TILE_LOG;
         argblk[7] = width;
         argblk[8] = height;
      } else {
         /* Grid-less kick: no host fragment grid. The RASTER fragment work
          * distributor (armed below via VX_DCR_RASTER_FRAG_ENTRY/PARAM) injects
          * one 1-warp fragment CTA per covered-quad wave and seeds its payload
          * into the gfx window; the straight-line FS reads it back with GETWS.
          * A non-zero host grid would instead launch a fixed handful of warps
          * that each run the FS once against an un-seeded window — shading only a
          * few quads and leaving the rest of the primitive at the clear colour. */
         fli.grid_dim[0]  = 0;
         fli.block_dim[0] = (uint32_t)(nt * nw);
      }

      /* HW raster: arm the fragment work distributor. The grid-less FS launch
       * carries no host args, so the injected fragment warps take their arg
       * pointer (a0) from VX_DCR_RASTER_FRAG_PARAM — stage the (now-complete)
       * arg block in a device buffer and resolve the FS entry PC for FRAG_ENTRY.
       * The write is enqueued before the draw so it lands before any warp runs. */
      uint64_t frag_entry = 0, fa_dev = 0;
      if (!sw_raster) {
         VP_CHECK(vx_buffer_create(dev, sizeof(argblk), VX_MEM_READ, &fabuf),
                  "vx_buffer_create(fragargs)");
         VP_CHECK(vx_buffer_address(fabuf, &fa_dev), "vx_buffer_address(fragargs)");
         VP_CHECK(vx_enqueue_write(q, fabuf, 0, argblk, sizeof(argblk), 0, NULL, NULL),
                  "vx_enqueue_write(fragargs)");
         VP_CHECK(vx_kernel_address(kbuf, &frag_entry), "vx_kernel_address(fs)");
      }

      /* The whole draw as ONE CP command batch: expand -> setup ->
       * binning, then RASTER/OM/TEX config, then FS. The CP retires the
       * commands in order, draining each launch before the next (the
       * inter-stage device barrier), so the host is untouched between stages
       * — no per-launch round-trip, no module swap. RASTER's device-produced
       * tile + prim buffers feed it; TILE_COUNT is the dense grid count
       * (host-known, no readback). Stencil disabled for gfx-v1 (ALWAYS/KEEP). */
      std::vector<vx_command_t> cmds;
      auto LAUNCH = [&](const vx_launch_info_t *l) {
         vx_command_t c{}; c.type = VX_COMMAND_LAUNCH; c.data.launch = l;
         cmds.push_back(c);
      };
      auto DCRW = [&](uint32_t a, uint32_t v) {
         vx_command_t c{}; c.type = VX_COMMAND_DCR_WRITE;
         c.data.dcr.addr = a; c.data.dcr.value = v; cmds.push_back(c);
      };

      LAUNCH(&vsli);   /* stage 0: vertex shader (folded into the draw) */
      LAUNCH(&eli);
      for (uint32_t s = 0; s < NSTAGE; ++s) LAUNCH(&li[s]);

      /* FF RASTER producer config — skipped when raster is routed to software
       * (the one-thread-per-tile FS walks the primbuf itself; no FF producer). */
      if (!sw_raster) {
      DCRW(VX_DCR_RASTER_TBUF_ADDR,   (uint32_t)(tile_dev / 64));
      DCRW(VX_DCR_RASTER_TILE_COUNT,  num_bins);
      DCRW(VX_DCR_RASTER_PBUF_ADDR,   (uint32_t)(prim_dev / 64));
      DCRW(VX_DCR_RASTER_PBUF_STRIDE, VP_RAST_PRIM_STRIDE);
      /* Scissor to the viewport's screen rect so an offset/partial viewport
       * rasters only within its rectangle (the device applies the vp scale/bias
       * transform, but the coverage walk must be clamped to the vp rect too). A
       * full-framebuffer viewport yields [0,W]x[0,H] — identical to the old
       * hardcoded scissor. rect = bias ± |scale|, clamped to the framebuffer. */
      float asx = vp_sx < 0 ? -vp_sx : vp_sx, asy = vp_sy < 0 ? -vp_sy : vp_sy;
      int32_t sxmin = (int32_t)(vp_tx - asx + 0.5f), sxmax = (int32_t)(vp_tx + asx + 0.5f);
      int32_t symin = (int32_t)(vp_ty - asy + 0.5f), symax = (int32_t)(vp_ty + asy + 0.5f);
      if (sxmin < 0) {
         sxmin = 0;
      }
      if (sxmax > (int32_t)width) {
         sxmax = (int32_t)width;
      }
      if (symin < 0) {
         symin = 0;
      }
      if (symax > (int32_t)height) {
         symax = (int32_t)height;
      }
      DCRW(VX_DCR_RASTER_SCISSOR_X,   ((uint32_t)sxmax << 16) | (uint32_t)sxmin);
      DCRW(VX_DCR_RASTER_SCISSOR_Y,   ((uint32_t)symax << 16) | (uint32_t)symin);
      /* Arm the fragment work distributor: FS entry PC + device args pointer.
       * The RASTER unit injects a fragment warp per covered quad only
       * when FRAG_ENTRY is non-zero — without this the grid-less FS never runs. */
      DCRW(VX_DCR_RASTER_FRAG_ENTRY_LO, (uint32_t)(frag_entry & 0xffffffffu));
      DCRW(VX_DCR_RASTER_FRAG_ENTRY_HI, (uint32_t)(frag_entry >> 32));
      DCRW(VX_DCR_RASTER_FRAG_PARAM_LO, (uint32_t)(fa_dev & 0xffffffffu));
      DCRW(VX_DCR_RASTER_FRAG_PARAM_HI, (uint32_t)(fa_dev >> 32));
      }

      /* FF OM config — skipped when OM is routed to software (the FS merges via
       * gfx_om_fragment_sw from the resident om descriptor instead). */
      if (!sw_om) {
      DCRW(VX_DCR_OM_CBUF_ADDR,        (uint32_t)(color_dev / 64));
      DCRW(VX_DCR_OM_CBUF_PITCH,       width * 4);
      DCRW(VX_DCR_OM_CBUF_WRITEMASK,   om->colormask);
      DCRW(VX_DCR_OM_ZBUF_ADDR,        (uint32_t)(depth_dev / 64));
      DCRW(VX_DCR_OM_ZBUF_PITCH,       width * 4);
      DCRW(VX_DCR_OM_DEPTH_FUNC,       om->depth_test ? om->depth_func : VX_OM_DEPTH_FUNC_ALWAYS);
      DCRW(VX_DCR_OM_DEPTH_WRITEMASK,  (om->depth_test && om->depth_write) ? 1u : 0u);
      DCRW(VX_DCR_OM_STENCIL_FUNC,      om->stencil_func);
      DCRW(VX_DCR_OM_STENCIL_ZPASS,     om->stencil_zpass);
      DCRW(VX_DCR_OM_STENCIL_ZFAIL,     om->stencil_zfail);
      DCRW(VX_DCR_OM_STENCIL_FAIL,      om->stencil_fail);
      DCRW(VX_DCR_OM_STENCIL_REF,       om->stencil_ref);
      DCRW(VX_DCR_OM_STENCIL_MASK,      om->stencil_mask);
      DCRW(VX_DCR_OM_STENCIL_WRITEMASK, om->stencil_writemask);
      DCRW(VX_DCR_OM_BLEND_MODE,        om->blend_mode);
      DCRW(VX_DCR_OM_BLEND_FUNC,        om->blend_func);
      DCRW(VX_DCR_OM_BLEND_CONST,       om->blend_const);
      DCRW(VX_DCR_OM_LOGIC_OP,          om->logic_op);
      DCRW(VX_DCR_OM_EARLYZ_SAFE,       om->earlyz_safe ? 1u : 0u);

      /* OM aperture: a fragment export is a store into a virtual address range
       * that the cluster's OM ingress peels off the L1->L2 trunk. The address is
       * shift-only (the pitch is padded to a power of two), so the ingress decodes
       * it by bit-slicing rather than dividing. Both sides must agree, so the same
       * geometry is programmed here and handed to the FS in its arg block. */
      DCRW(VX_DCR_OM_APERTURE_XBITS,        ap_xbits);
      DCRW(VX_DCR_OM_APERTURE_YBITS,        ap_ybits);
      DCRW(VX_DCR_OM_APERTURE_RECORD_SHIFT, ap_shift);
      DCRW(VX_DCR_OM_APERTURE_DEPTH_ONLY,   0);
      }

      /* TEX unit (stage 0): the texels are residency-cached at tex_dev (A8R8G8B8
       * for the FF formats this path serves); program the sampler. Untextured
       * draws (tex_dev==0) leave TEX state alone. SW-textured draws (sw_tex)
       * skip the FF TEX config —
       * the FS samples in software from the resident descriptor (argblk[1]). */
      if (tex_dev && tex && !sw_tex) {
         uint32_t logw = vp_log2u(tex->width);
         uint32_t logh = vp_log2u(tex->height);
         DCRW(VX_DCR_TEX_STAGE,        0);
         DCRW(VX_DCR_TEX_LOGDIM,       (logh << 16) | logw);
         DCRW(VX_DCR_TEX_FORMAT,       VX_TEX_FORMAT_A8R8G8B8);
         DCRW(VX_DCR_TEX_FILTER,       tex->filter);
         DCRW(VX_DCR_TEX_WRAP,         (tex->wrap_v << 16) | tex->wrap_u);
         DCRW(VX_DCR_TEX_ADDR,         (uint32_t)(tex_dev / 64));
         /* Per-LOD mip byte offsets: the TEX unit indexes mipoff[selected lod]
          * (VX_tex_core) to reach that level's texels within the resident base. */
         for (uint32_t j = 0; j <= (uint32_t)VX_TEX_LOD_MAX; ++j)
            DCRW(VX_DCR_TEX_MIPOFF_BASE + j, tex->mip_off[j]);
      }

      LAUNCH(&fli);

      /* Submit the whole draw as ONE device-orchestrated command: the CP
       * expands the resident descriptor and sequences expand→setup→bin→
       * FF-config→FS on-device, draining each stage with no host round-trip. */
      VP_CHECK(vx_enqueue_draw(q, cmds.data(), (uint32_t)cmds.size(),
                               0, NULL, NULL), "vx_enqueue_draw");

      /* The render lands in the resident colour buffer (color_dev); the caller
       * syncs it back to the framebuffer resource at present / pass end. No
       * per-draw colour readback. */
      VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");
      ok = true;
   }

done:
   /* the front-end working set, code modules and colour/depth/texture buffers
    * are resident (pool- / context-cached) and persist across draws; only the
    * per-draw VS-output + vertex-input buffers are released here. */
   if (tbuf) vx_buffer_release(tbuf);
   for (vx_buffer_h &b : vbufs_dev) { if (b) vx_buffer_release(b); }
   if (vsbuf) vx_buffer_release(vsbuf);
   if (fabuf) vx_buffer_release(fabuf);
   if (fs_desc_buf) vx_buffer_release(fs_desc_buf);
   for (vx_buffer_h b : fs_cbuf_bufs)
      if (b) vx_buffer_release(b);
   for (vx_buffer_h b : fs_res_bufs)
      if (b) vx_buffer_release(b);
   for (uint8_t *s : fs_desc_stage)
      if (s) free(s);
   if (vs_desc_buf) vx_buffer_release(vs_desc_buf);
   for (vx_buffer_h b : vs_cbuf_bufs)
      if (b) vx_buffer_release(b);
   for (vx_buffer_h b : vs_res_bufs)
      if (b) vx_buffer_release(b);
   for (uint8_t *s : vs_desc_stage)
      if (s) free(s);
   if (mrtbuf) vx_buffer_release(mrtbuf);
   if (q)         vx_queue_release(q);
   return ok;
}
