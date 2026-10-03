/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_launch -- Phase 2 #5: dispatch a compiled kernel on the Vortex
 * device.
 *
 * Builds the kernel argument block the translated kernel expects
 * (vp_nir_to_llvm: kernel_main(ptr arg) reads buffer addresses as
 * arg[i] -- an array of i64 device addresses), copies the SSBO
 * host<->device, and drives the vortex2 runtime: queue -> upload ->
 * vx_enqueue_launch -> read back. add1/vecadd-class single-SSBO
 * kernels (binding 0); the multi-binding descriptor-stride case is
 * a later generalization.
 */

#define _GNU_SOURCE
#include "vp_launch.h"
#include "vp_private.h"        /* vp_screen_resident_addr */
#include "gfx_fs_desc_abi.h"     /* GFX_FS_DESC_SLOTS (VS constant-buffer table) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "util/log.h"

/* The i64 arg-block layout (VP_ARG_SLOTS, VP_ARG_SSBO_BASE) is the kernel ABI
 * defined in vp_nir_to_llvm.h — shared with the NIR->LLVM arg[i] reads. */

/* VP_CHECK wraps a Vortex runtime call: any vx_result_t != VX_SUCCESS is a
 * HARD ERROR — the operation we asked the runtime to perform on the device
 * failed. Log as mesa_loge so the host runtime / test harness can detect
 * the failure (vp_launch returns false on goto done; the test harness can
 * grep "MESA: error" or check stderr). DO NOT downgrade to mesa_logw: the
 * silent-fallback bug that ran tests on llvmpipe started here. */
#define VP_CHECK(call, what)                                            \
   do {                                                                 \
      vx_result_t _r = (call);                                          \
      if (_r != VX_SUCCESS) {                                           \
         mesa_loge("vortexpipe: launch: %s failed (%s)", (what),        \
                   vx_result_string(_r));                               \
         goto done;                                                     \
      }                                                                 \
   } while (0)

/* A buffer descriptor's lp_jit_buffer occupies the first bytes of its
 * struct lp_descriptor slot: the data pointer at +0 (8 bytes), the
 * byte size at +8 (4 bytes). VP_DESC_STRIDE lives in vp_launch.h. */
#define VP_JIT_BUF_PTR   0
#define VP_JIT_BUF_SIZE  8

/* A storage-image descriptor's lp_jit_image occupies the first bytes of its
 * lp_descriptor slot (image union at offset 0): base pointer at +0 (aliases
 * lp_jit_buffer.ptr), height at +12, row_stride at +24, base_offset at +40. */
#define VP_JIT_IMG_BASE        0
#define VP_JIT_IMG_HEIGHT      12
#define VP_JIT_IMG_ROW_STRIDE  24
#define VP_JIT_IMG_BASE_OFFSET 40

/* ---- acceleration-structure (BVH) device copy (Phase 7.6) --------- *
 *
 * lavapipe builds the BVH on the CPU; the Vortex traversal kernel
 * walks it, so it must be copied into Vortex device memory. Layout
 * constants mirror struct lvp_bvh_header / lvp_bvh_instance_node in
 * src/gallium/frontends/lavapipe/lvp_acceleration_structure.h:
 *
 *   lvp_bvh_header { vk_aabb bounds;          // 24 B
 *                    uint32 serialization_size, instance_count,
 *                           leaf_nodes_offset, padding; }
 *
 * A box node's two children are BVH-relative uint32 offsets, so they
 * survive the copy unchanged. Only an instance node's bvh_ptr is an
 * absolute pointer (TLAS -> BLAS) and is relocated to the device copy.
 * serialization_size folds in a fixed serialization header plus eight
 * bytes per instance, so the buffer size is recoverable from it. */
#define VP_BVH_SERIALIZATION_SIZE  24   /* uint32, after vk_aabb bounds */
#define VP_BVH_INSTANCE_COUNT      28   /* uint32 */
#define VP_BVH_LEAF_NODES_OFFSET   32   /* uint32 */
#define VP_BVH_INSTANCE_NODE_SIZE  120  /* sizeof(struct lvp_bvh_instance_node) */
#define VP_ACCEL_SERIALIZATION_HDR 56   /* sizeof(lvp_accel_struct_serialization_header) */
#define VP_BVH_MAX_BYTES           (16u << 20)
#define VP_MAX_BVH                 64

/* Device buffers + host staging blobs created while copying one
 * acceleration structure's BVHs; both must outlive vx_queue_finish. */
struct vp_as_ctx {
   vx_device_h dev;
   vx_queue_h  q;
   vx_buffer_h bufs[VP_MAX_BVH];
   unsigned    n_bufs;
   void       *stages[VP_MAX_BVH];
   unsigned    n_stages;
   bool        has_rtu;   /* transcode AS to RTU scene instead of verbatim copy */
   bool        ok;
};

/* Copy one lavapipe BVH (TLAS or BLAS) into Vortex device memory and
 * return its device address. A TLAS's instance nodes carry absolute
 * bvh_ptr links to their BLASes; those are copied recursively and the
 * link rewritten to the device address. Box-node children are
 * BVH-relative and need no fixup. */
static uint64_t
vp_copy_as(struct vp_as_ctx *c, const void *bvh_host)
{
   const uint8_t *h = bvh_host;
   uint32_t ser = 0, inst = 0, leaf_off = 0;
   memcpy(&ser,      h + VP_BVH_SERIALIZATION_SIZE, sizeof ser);
   memcpy(&inst,     h + VP_BVH_INSTANCE_COUNT,     sizeof inst);
   memcpy(&leaf_off, h + VP_BVH_LEAF_NODES_OFFSET,  sizeof leaf_off);

   if (ser <= VP_ACCEL_SERIALIZATION_HDR + 8u * inst) {
      mesa_loge("vortexpipe: launch: implausible BVH header");
      c->ok = false;
      return 0;
   }
   uint32_t size = ser - VP_ACCEL_SERIALIZATION_HDR - 8u * inst;
   if (size == 0 || size > VP_BVH_MAX_BYTES ||
       c->n_stages >= VP_MAX_BVH || c->n_bufs >= VP_MAX_BVH) {
      mesa_loge("vortexpipe: launch: BVH too large / too many BVHs");
      c->ok = false;
      return 0;
   }

   /* private copy so instance-node bvh_ptr fields can be relocated */
   uint8_t *stage = malloc(size);
   if (!stage) {
      c->ok = false;
      return 0;
   }
   memcpy(stage, bvh_host, size);
   c->stages[c->n_stages++] = stage;

   for (uint32_t i = 0; i < inst; i++) {
      uint8_t  *node = stage + leaf_off + (size_t)i * VP_BVH_INSTANCE_NODE_SIZE;
      uint64_t  blas_host = 0;
      memcpy(&blas_host, node, sizeof blas_host);   /* lvp_bvh_instance_node.bvh_ptr */
      if (blas_host) {
         uint64_t blas_dev = vp_copy_as(c, (const void *)(uintptr_t)blas_host);
         if (!c->ok)
            return 0;
         memcpy(node, &blas_dev, sizeof blas_dev);
      }
   }

   vx_buffer_h b = NULL;
   uint64_t dev_addr = 0;
   if (vx_buffer_create(c->dev, size, 0, &b) != VX_SUCCESS) {
      c->ok = false;
      return 0;
   }
   c->bufs[c->n_bufs++] = b;
   if (vx_buffer_address(b, &dev_addr) != VX_SUCCESS ||
       vx_enqueue_write(c->q, b, 0, stage, size, 0, NULL, NULL) != VX_SUCCESS) {
      c->ok = false;
      return 0;
   }
   return dev_addr;
}

/* ── BVH transcode: lavapipe lvp_bvh → Vortex RTU scene ──────────────
 * The RTU walks its own scene format (sim/simx/rtu/rtu_types.h), not the
 * lvp_bvh layout. We collect the acceleration structure's opaque triangles
 * in world space (instance transforms applied), then build a CW-BVH4 scene
 * (scene_kind=2): a 16-byte header { root_offset, scene_kind, scene_bytes,
 * node_count } followed by 64-byte 4-wide internal nodes (common origin +
 * per-axis exponent, 8-bit quantized child AABBs) and 56-byte leaves
 * { 16-byte header (kind|count, geometry_index, _, prim_base) + 40-byte
 * triangle }. One triangle per leaf; the leaf's prim_base carries the
 * source triangle's gl_PrimitiveID. An empty AS degenerates to an empty
 * TriList (all-miss). Instancing-aware two-level TLAS transcode is a
 * follow-up. */

/* lvp_bvh node layout (lvp_acceleration_structure.h). */
#define LVP_BVH_HEADER_SIZE   40   /* sizeof(struct lvp_bvh_header) */
#define LVP_NODE_TRIANGLE     0
#define LVP_NODE_INTERNAL     1
#define LVP_NODE_INSTANCE     2
#define LVP_NODE_AABB         3
#define LVP_NODE_INVALID      0xFFFFFFFFu
#define LVP_BOX_CHILDREN_OFF  48   /* vk_aabb bounds[2] precede children[2] */
#define LVP_TRI_GEOMFLAGS_OFF 44   /* coords[3][3]+padding+primitive_id      */
#define LVP_INST_SBTFLAGS_OFF 12   /* bvh_ptr + custom_instance_and_mask      */
#define LVP_INST_OTW_OFF      72   /* otw_matrix (object→world, mat3x4)       */
#define LVP_INST_CUSTOM_OFF   8    /* custom_instance_and_mask                */
#define LVP_INST_ID_OFF       68   /* instance_id                             */
#define LVP_INST_NODE_BYTES   120  /* sizeof(struct lvp_bvh_instance_node)    */
#define LVP_INST_WTO_OFF      16   /* wto_matrix (world→object, mat3x4)       */
#define LVP_AABB_PRIM_OFF     24   /* lvp_bvh_aabb_node.primitive_id          */
#define LVP_AABB_GEOMFLAGS_OFF 28  /* lvp_bvh_aabb_node.geometry_id_and_flags */
#define LVP_INSTANCE_TRIANGLE_FACING_CULL_DISABLE (1u << 29)
#define LVP_INSTANCE_TRIANGLE_FLIP_FACING         (1u << 28)

/* Opacity lives in the top bits of two words: the geometry's own flag in a
 * triangle node's geometry_id_and_flags, and the instance's overrides in
 * sbt_offset_and_flags. A hit is opaque when the geometry (or the instance's
 * force) says so AND the instance does not force it non-opaque. Geometry-opaque
 * and instance-force-opaque deliberately occupy the same bit, so ORing the two
 * words and testing both bits is the whole rule -- the form
 * lvp_build_hit_is_opaque uses, kept identical here so the device and the CPU
 * traversal never disagree about which triangles need an any-hit decision. */
#define LVP_GEOMETRY_OPAQUE              (1u << 31)
#define LVP_INSTANCE_FORCE_OPAQUE        (1u << 31)
#define LVP_INSTANCE_NO_FORCE_NOT_OPAQUE (1u << 30)
#define LVP_OPAQUE_BITS \
   (LVP_GEOMETRY_OPAQUE | LVP_INSTANCE_FORCE_OPAQUE | LVP_INSTANCE_NO_FORCE_NOT_OPAQUE)

/* A BLAS walked without an enclosing instance takes the flags a default
 * instance would carry, so geometry opacity alone decides. */
#define LVP_INSTANCE_FLAGS_DEFAULT       LVP_INSTANCE_NO_FORCE_NOT_OPAQUE

/* RTU scene constants (rtu_types.h / rtu_bvh.h). */
#define RTU_SCENE_HDR_BYTES   16
#define RTU_TRI_STRIDE        40
#define RTU_TRI_FLAGS_OFFSET  36
#define RTU_TRI_FLAG_OPAQUE   0x1u
#define RTU_SCENE_KIND_TRILIST 0u
#define RTU_SCENE_KIND_BVH4    2u

/* CW-BVH4 on-disk layout (rtu_bvh.h: VxBvhInternalNode / VxBvhLeafHeader). */
#define RTU_BVH4_NODE_BYTES   64u
#define RTU_BVH4_WIDTH        4u
#define RTU_BVH4_OFF_ORIGIN   4u
#define RTU_BVH4_OFF_EXP      16u
#define RTU_BVH4_OFF_CHILD    20u
#define RTU_BVH4_OFF_QMIN     36u
#define RTU_BVH4_OFF_QMAX     48u
#define RTU_BVH_LEAF_HDR_BYTES 16u
#define RTU_BVH_KIND_INTERNAL  0u
#define RTU_BVH_KIND_LEAF_TRI  1u
#define RTU_BVH_COUNT_SHIFT    8u
#define RTU_BVH_CHILD_LEAF_FLAG 0x80000000u
#define RTU_BVH_KIND_LEAF_INST 2u
#define RTU_BVH_KIND_LEAF_PROC 3u
#define RTU_PROC_AABB_BYTES    24u
#define RTU_BVH_FLAG_OPAQUE    0x1u
#define RTU_BVH_FLAG_PROCEDURAL 0x2u
#define RTU_BVH_INSTANCE_STRIDE     64u
#define RTU_BVH_INSTANCE_BLAS_OFF   48u
#define RTU_BVH_INSTANCE_CUSTOM_OFF 52u
#define RTU_BVH_INSTANCE_ID_OFF     56u
#define RTU_BVH_INSTANCE_CULL_OFF   60u
#define RTU_INST_FLAGS_SHIFT        8u
#define RTU_INST_FLAG_TRI_CULL_DIS  0x1u
#define RTU_INST_FLAG_TRI_FLIP      0x2u
#define RTU_INST_FLAG_FORCE_OPAQUE  0x4u
#define RTU_INST_FLAG_FORCE_NO_OPQ  0x8u
/* Scene offsets are 31-bit (bit 31 of a child word flags a leaf). */
#define VP_RTU_SCENE_MAX_BYTES      0x7fffffffu

/* One primitive a BLAS contributes, in object space: a triangle (v[0..8] =
 * v0, v1, v2), or -- RTU_BVH_FLAG_PROCEDURAL set in `flags` -- a procedural
 * AABB (v[0..2] = min, v[3..5] = max) that yields to the intersection shader. */
struct vp_prim {
   float    v[9];
   uint32_t flags;      /* RTU_BVH_FLAG_OPAQUE | RTU_BVH_FLAG_PROCEDURAL */
   uint32_t prim_id;    /* gl_PrimitiveID */
   uint32_t geom_id;    /* lavapipe's geometry_id_and_flags: gl_GeometryIndexEXT
                         * in bits 0..27, the geometry-opaque flag in bit 31.
                         * The RTU reports it verbatim; readers mask. */
   uint32_t ps;         /* lavapipe parent node << 1 | side (visit-order table) */
};

struct vp_prim_list {
   struct vp_prim *p;
   uint32_t count, cap;
   bool ok;
};

/* One TLAS instance, as lavapipe laid it out. */
struct vp_inst {
   const uint8_t *blas;
   float          otw[12];    /* object -> world, 3x4 row-major */
   uint32_t       custom;     /* gl_InstanceCustomIndexEXT */
   uint32_t       mask;       /* visibility mask */
   uint32_t       flags;      /* RTU_INST_FLAG_* */
   uint32_t       id;         /* gl_InstanceID */
   uint32_t       rank;       /* leaf index in the TLAS visit-order table */
   uint8_t        node[LVP_INST_NODE_BYTES];   /* lavapipe's own record */
};

struct vp_inst_list {
   struct vp_inst *p;
   uint32_t count, cap;
   bool ok;
   bool has_geometry;         /* the walked AS is a BLAS, not a TLAS */
};

#define VP_LIST_PUSH(l, item)                                            \
   do {                                                                  \
      if ((l)->count == (l)->cap) {                                      \
         uint32_t ncap = (l)->cap ? (l)->cap * 2 : 64;                   \
         void *np = realloc((l)->p, (size_t)ncap * sizeof(*(l)->p));     \
         if (!np) { (l)->ok = false; break; }                            \
         (l)->p = np;                                                    \
         (l)->cap = ncap;                                                \
      }                                                                  \
      (l)->p[(l)->count++] = (item);                                     \
   } while (0)

/* world = M(3x4 row-major) * [obj; 1]. */
static void
vp_xform_point(const float M[12], const float p[3], float out[3])
{
   for (int i = 0; i < 3; i++)
      out[i] = M[i*4+0]*p[0] + M[i*4+1]*p[1] + M[i*4+2]*p[2] + M[i*4+3];
}

/* C = A ∘ B (both object→world 3x4; apply B then A). */
static void
vp_xform_compose(const float A[12], const float B[12], float C[12])
{
   for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++)
         C[i*4+j] = A[i*4+0]*B[0*4+j] + A[i*4+1]*B[1*4+j] + A[i*4+2]*B[2*4+j];
      C[i*4+3] = A[i*4+0]*B[0*4+3] + A[i*4+1]*B[1*4+3] + A[i*4+2]*B[2*4+3] + A[i*4+3];
   }
}

/* Node-pointer stack for the lavapipe BVH walks. lavapipe's binary trees can
 * run deeper than any fixed recursion bound on large meshes, so the walks are
 * iterative and every subtree is visited. */
struct vp_ptr_stack {
   uint32_t *p;
   uint32_t  count, cap;
   bool      ok;
};

/* Visit-order table of a lavapipe BVH: its binary tree, which lavapipe walks
 * depth-first, nearer child box first, child[0] on equal entry distances, so
 * of two hits at exactly the same t it keeps the one it reaches first. The RTU
 * walks its own tree in its own order; on an exact-t tie it settles which of
 * the two leaves lavapipe reaches first from these tables (the two child boxes
 * under their lowest common ancestor, tested as lavapipe tests them). Scene
 * layouts (rtu_walker.cpp):
 *  TLAS (few nodes; leaf = instance, ranked in child[0]-first DFS order):
 *   { n_leaves, n_nodes, nodes_off, 64 }
 *   leaf[i] at +16 + i*64: { parent << 1 | side, _, _, _, world->object 3x4 }
 *   node[j] at +nodes_off + j*64: { bounds[2] (min, max), parent << 1 | side
 *     (~0 for the root), depth }
 *  BLAS (one node per triangle, so kept compact):
 *   { n_nodes, _, 64, 32 }, 64 B header
 *   node[j] at +64 + j*32: { own box (min, max) = the parent's bounds[side],
 *     parent << 1 | side, depth }
 *   A triangle leaf carries its own parent << 1 | side in its leaf header, and
 *   its box is fp32 min/max of its vertices -- lavapipe's leaf bound, checked
 *   here for every triangle (a BLAS where it does not hold gets no table). */
#define VP_OTAB_HDR_BYTES   16u
#define VP_OTAB_NODE_BYTES  64u
#define VP_OTAB_TLAS_LEAF   64u
#define VP_OTAB_BLAS_HDR    64u
#define VP_OTAB_BLAS_NODE   32u
#define VP_OTAB_ROOT        0xffffffffu

struct vp_onode {
   float    box[12];     /* lvp_bvh_box_node.bounds[2] */
   uint32_t ps;          /* parent << 1 | side */
   uint32_t depth;
};

struct vp_otab {
   struct { struct vp_onode *p; uint32_t count, cap; bool ok; } nodes;
   struct { uint32_t *p; uint32_t count, cap; bool ok; } leaves;   /* TLAS: ps */
   uint32_t leaf_box_mismatch;   /* BLAS: leaves whose bound != vertex min/max */
};

static void
vp_otab_fini(struct vp_otab *ot)
{
   free(ot->nodes.p);
   free(ot->leaves.p);
}

/* Record internal node `node` reached as (parent << 1 | side) = ps; returns
 * its index. */
static uint32_t
vp_otab_node(struct vp_otab *ot, const uint8_t *node, uint32_t ps)
{
   struct vp_onode n;
   memcpy(n.box, node, sizeof n.box);
   n.ps = ps;
   n.depth = ps == VP_OTAB_ROOT ? 0 : ot->nodes.p[ps >> 1].depth + 1;
   const uint32_t idx = ot->nodes.count;
   VP_LIST_PUSH(&ot->nodes, n);
   return idx;
}

/* Collect a BLAS's primitives in object space. Opacity here is the geometry's
 * own: instance overrides travel in the instance record, where the RTU
 * composes them per instance, so one BLAS serves every instance of it. */
static void
vp_walk_blas(const uint8_t *bvh, uint32_t root, struct vp_prim_list *l,
             struct vp_otab *ot)
{
   /* Stack entries are (ps, node_ptr) pairs. */
   struct vp_ptr_stack st = { .ok = true };
   VP_LIST_PUSH(&st, VP_OTAB_ROOT);
   VP_LIST_PUSH(&st, root);
   while (st.ok && l->ok && ot->nodes.ok && ot->leaves.ok && st.count >= 2) {
      const uint32_t node_ptr = st.p[--st.count];
      const uint32_t ps = st.p[--st.count];
      if (node_ptr == LVP_NODE_INVALID)
         continue;
      const uint8_t *node = bvh + (node_ptr & ~7u);
      switch (node_ptr & 7u) {
      case LVP_NODE_INTERNAL: {
         uint32_t c0, c1;
         memcpy(&c0, node + LVP_BOX_CHILDREN_OFF + 0, 4);
         memcpy(&c1, node + LVP_BOX_CHILDREN_OFF + 4, 4);
         const uint32_t idx = vp_otab_node(ot, node, ps);
         VP_LIST_PUSH(&st, idx << 1 | 1u);
         VP_LIST_PUSH(&st, c1);
         VP_LIST_PUSH(&st, idx << 1);
         VP_LIST_PUSH(&st, c0);
         break;
      }
      case LVP_NODE_TRIANGLE: {
         struct vp_prim pr;
         uint32_t geom_flags = 0;
         memcpy(pr.v, node, 36);                    /* coords[3][3] */
         memcpy(&pr.prim_id, node + 40, 4);
         memcpy(&geom_flags, node + LVP_TRI_GEOMFLAGS_OFF, 4);
         pr.geom_id = geom_flags;
         pr.flags = (geom_flags & LVP_GEOMETRY_OPAQUE) ? RTU_BVH_FLAG_OPAQUE : 0u;
         pr.ps = ps;
         if (ps != VP_OTAB_ROOT) {
            const float *bb = ot->nodes.p[ps >> 1].box + 6 * (ps & 1u);
            for (int a = 0; a < 3; a++) {
               if (bb[a] != fminf(pr.v[a], fminf(pr.v[3 + a], pr.v[6 + a])) ||
                   bb[3 + a] != fmaxf(pr.v[a], fmaxf(pr.v[3 + a], pr.v[6 + a]))) {
                  ot->leaf_box_mismatch++;
                  break;
               }
            }
         }
         VP_LIST_PUSH(l, pr);
         break;
      }
      case LVP_NODE_AABB: {
         struct vp_prim pr;
         uint32_t geom_flags = 0;
         memset(&pr, 0, sizeof pr);
         memcpy(pr.v, node, 24);                    /* vk_aabb {min, max} */
         memcpy(&pr.prim_id, node + LVP_AABB_PRIM_OFF, 4);
         memcpy(&geom_flags, node + LVP_AABB_GEOMFLAGS_OFF, 4);
         pr.geom_id = geom_flags;
         pr.flags = RTU_BVH_FLAG_PROCEDURAL |
                    ((geom_flags & LVP_GEOMETRY_OPAQUE) ? RTU_BVH_FLAG_OPAQUE : 0u);
         pr.ps = ps;
         VP_LIST_PUSH(l, pr);
         break;
      }
      default:
         break;
      }
   }
   if (!st.ok || !ot->nodes.ok || !ot->leaves.ok)
      l->ok = false;
   free(st.p);
}

/* lavapipe's packed instance flags -> the RTU's (VkGeometryInstanceFlagBits). */
static uint32_t
vp_rtu_inst_flags(uint32_t sbt_offset_and_flags)
{
   uint32_t f = 0;
   if (sbt_offset_and_flags & LVP_INSTANCE_TRIANGLE_FACING_CULL_DISABLE)
      f |= RTU_INST_FLAG_TRI_CULL_DIS;
   if (sbt_offset_and_flags & LVP_INSTANCE_TRIANGLE_FLIP_FACING)
      f |= RTU_INST_FLAG_TRI_FLIP;
   if (sbt_offset_and_flags & LVP_INSTANCE_FORCE_OPAQUE)
      f |= RTU_INST_FLAG_FORCE_OPAQUE;
   if (!(sbt_offset_and_flags & LVP_INSTANCE_NO_FORCE_NOT_OPAQUE))
      f |= RTU_INST_FLAG_FORCE_NO_OPQ;
   return f;
}

static void
vp_walk_tlas(const uint8_t *bvh, uint32_t root, struct vp_inst_list *il,
             struct vp_otab *ot)
{
   struct vp_ptr_stack st = { .ok = true };
   VP_LIST_PUSH(&st, VP_OTAB_ROOT);
   VP_LIST_PUSH(&st, root);
   while (st.ok && il->ok && ot->nodes.ok && ot->leaves.ok && st.count >= 2) {
      const uint32_t node_ptr = st.p[--st.count];
      const uint32_t ps = st.p[--st.count];
      if (node_ptr == LVP_NODE_INVALID)
         continue;
      const uint8_t *node = bvh + (node_ptr & ~7u);
      switch (node_ptr & 7u) {
      case LVP_NODE_INTERNAL: {
         uint32_t c0, c1;
         memcpy(&c0, node + LVP_BOX_CHILDREN_OFF + 0, 4);
         memcpy(&c1, node + LVP_BOX_CHILDREN_OFF + 4, 4);
         const uint32_t idx = vp_otab_node(ot, node, ps);
         VP_LIST_PUSH(&st, idx << 1 | 1u);
         VP_LIST_PUSH(&st, c1);
         VP_LIST_PUSH(&st, idx << 1);
         VP_LIST_PUSH(&st, c0);
         break;
      }
      case LVP_NODE_INSTANCE: {
         struct vp_inst in;
         uint64_t blas = 0;
         uint32_t cm = 0, sf = 0;
         memcpy(&blas, node, 8);
         memcpy(&cm, node + LVP_INST_CUSTOM_OFF, 4);
         memcpy(&sf, node + LVP_INST_SBTFLAGS_OFF, 4);
         memcpy(&in.id, node + LVP_INST_ID_OFF, 4);
         memcpy(in.otw, node + LVP_INST_OTW_OFF, 48);
         if (!blas)
            break;
         in.blas = (const uint8_t *)(uintptr_t)blas;
         in.custom = cm & 0xffffffu;
         in.mask = cm >> 24;
         in.flags = vp_rtu_inst_flags(sf);
         memcpy(in.node, node, LVP_INST_NODE_BYTES);
         in.rank = il->count;
         VP_LIST_PUSH(il, in);
         VP_LIST_PUSH(&ot->leaves, ps);
         break;
      }
      default:
         il->has_geometry = true;
         break;
      }
   }
   if (!st.ok || !ot->nodes.ok || !ot->leaves.ok)
      il->ok = false;
   free(st.p);
}

/* ── CW-BVH4 builder (binned SAH, ≤4 children, 1 tri/leaf) ──
 * `order` is a permutation of triangle indices; a build node is a leaf
 * (one triangle) or an internal node fanning out to up to 4 children, formed by
 * splitting its largest range until it has four. */
struct vp_bnode {
   float    mn[3], mx[3];
   uint32_t tri;                    /* leaf: source triangle index (prim id) */
   int      child[RTU_BVH4_WIDTH];  /* internal: build-node indices */
   int      nchild;
   bool     leaf;
};

struct vp_bvh {
   float    (*cmin)[3];             /* per-tri AABB min */
   float    (*cmax)[3];             /* per-tri AABB max */
   float    (*cen)[3];              /* per-tri centroid */
   uint32_t *order;                 /* tri index permutation */
   struct vp_bnode *nodes;          /* pre-sized: no realloc during build */
   uint32_t  n_nodes;
};

#define VP_SAH_BINS 16

static float
vp_half_area(const float mn[3], const float mx[3])
{
   float dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
   return dx * dy + dy * dz + dz * dx;
}

/* Binned-SAH binary split of order[start, start+count): bins the triangle
 * centroids along each axis, takes the plane of least
 * area(L)*n(L) + area(R)*n(R), and partitions order[] in place about it.
 * Returns the left count, 0 < m < count. Linear in `count`, so a whole build is
 * O(n log n) -- and the tree it yields is the one the RTU walks, so its quality
 * is paid for in every ray's traversal. */
static uint32_t
vp_bvh_split(struct vp_bvh *b, uint32_t start, uint32_t count)
{
   float cmn[3] = { 1e30f, 1e30f, 1e30f }, cmx[3] = { -1e30f, -1e30f, -1e30f };
   for (uint32_t i = 0; i < count; i++) {
      const float *c = b->cen[b->order[start + i]];
      for (int a = 0; a < 3; a++) {
         if (c[a] < cmn[a]) cmn[a] = c[a];
         if (c[a] > cmx[a]) cmx[a] = c[a];
      }
   }

   float best_cost = 1e38f;
   int best_axis = -1, best_bin = 0;
   for (int a = 0; a < 3; a++) {
      const float ext = cmx[a] - cmn[a];
      if (!(ext > 0.f))
         continue;
      const float scale = (float)VP_SAH_BINS / ext;
      uint32_t cnt[VP_SAH_BINS] = { 0 };
      float bmn[VP_SAH_BINS][3], bmx[VP_SAH_BINS][3];
      for (int k = 0; k < VP_SAH_BINS; k++)
         for (int d = 0; d < 3; d++) { bmn[k][d] = 1e30f; bmx[k][d] = -1e30f; }
      for (uint32_t i = 0; i < count; i++) {
         uint32_t t = b->order[start + i];
         int k = (int)((b->cen[t][a] - cmn[a]) * scale);
         if (k >= VP_SAH_BINS) k = VP_SAH_BINS - 1;
         cnt[k]++;
         for (int d = 0; d < 3; d++) {
            if (b->cmin[t][d] < bmn[k][d]) bmn[k][d] = b->cmin[t][d];
            if (b->cmax[t][d] > bmx[k][d]) bmx[k][d] = b->cmax[t][d];
         }
      }
      /* Right-to-left suffix sweep, then a left-to-right sweep that prices each
       * plane k (bins < k on the left). */
      float rarea[VP_SAH_BINS];
      uint32_t rcnt[VP_SAH_BINS];
      float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
      uint32_t n = 0;
      for (int k = VP_SAH_BINS - 1; k > 0; k--) {
         n += cnt[k];
         for (int d = 0; d < 3; d++) {
            if (bmn[k][d] < mn[d]) mn[d] = bmn[k][d];
            if (bmx[k][d] > mx[d]) mx[d] = bmx[k][d];
         }
         rarea[k] = n ? vp_half_area(mn, mx) : 0.f;
         rcnt[k] = n;
      }
      for (int d = 0; d < 3; d++) { mn[d] = 1e30f; mx[d] = -1e30f; }
      n = 0;
      for (int k = 1; k < VP_SAH_BINS; k++) {
         n += cnt[k - 1];
         for (int d = 0; d < 3; d++) {
            if (bmn[k - 1][d] < mn[d]) mn[d] = bmn[k - 1][d];
            if (bmx[k - 1][d] > mx[d]) mx[d] = bmx[k - 1][d];
         }
         if (!n || !rcnt[k])
            continue;
         float cost = vp_half_area(mn, mx) * (float)n + rarea[k] * (float)rcnt[k];
         if (cost < best_cost) {
            best_cost = cost;
            best_axis = a;
            best_bin  = k;
         }
      }
   }
   /* Coincident centroids leave no plane to price; any balanced cut is as good. */
   if (best_axis < 0)
      return count / 2;

   const float scale = (float)VP_SAH_BINS / (cmx[best_axis] - cmn[best_axis]);
   uint32_t i = start, j = start + count;
   while (i < j) {
      uint32_t t = b->order[i];
      int k = (int)((b->cen[t][best_axis] - cmn[best_axis]) * scale);
      if (k >= VP_SAH_BINS) k = VP_SAH_BINS - 1;
      if (k < best_bin) {
         i++;
      } else {
         b->order[i] = b->order[--j];
         b->order[j] = t;
      }
   }
   uint32_t m = i - start;
   return (m == 0 || m == count) ? count / 2 : m;
}

/* Build the subtree over order[start, start+count); returns its node id. */
static int
vp_bvh_build(struct vp_bvh *b, uint32_t start, uint32_t count)
{
   int id = (int)b->n_nodes++;
   struct vp_bnode *n = &b->nodes[id];
   memset(n, 0, sizeof(*n));
   if (count == 1) {
      n->leaf = true;
      n->tri  = b->order[start];
      memcpy(n->mn, b->cmin[n->tri], sizeof n->mn);
      memcpy(n->mx, b->cmax[n->tri], sizeof n->mx);
      return id;
   }
   /* Partition into up to RTU_BVH4_WIDTH child ranges: repeatedly SAH-split
    * the largest range that still holds more than one triangle. */
   uint32_t rs[RTU_BVH4_WIDTH], rc[RTU_BVH4_WIDTH];
   int nr = 1;
   rs[0] = start; rc[0] = count;
   while (nr < (int)RTU_BVH4_WIDTH) {
      int best = -1;
      uint32_t bestc = 1;
      for (int i = 0; i < nr; i++)
         if (rc[i] > bestc) { bestc = rc[i]; best = i; }
      if (best < 0) break;
      uint32_t s = rs[best], c = rc[best];
      uint32_t m = vp_bvh_split(b, s, c);
      rs[best] = s;     rc[best] = m;
      rs[nr]   = s + m; rc[nr]   = c - m; nr++;
   }
   int kids[RTU_BVH4_WIDTH], nk = 0;
   for (int i = 0; i < nr; i++)
      if (rc[i] > 0) kids[nk++] = vp_bvh_build(b, rs[i], rc[i]);
   n = &b->nodes[id];   /* nodes[] is pre-sized; the pointer is still valid */
   n->leaf = false;
   n->nchild = nk;
   for (int a = 0; a < 3; a++) { n->mn[a] = 1e30f; n->mx[a] = -1e30f; }
   for (int i = 0; i < nk; i++) {
      n->child[i] = kids[i];
      const struct vp_bnode *c = &b->nodes[kids[i]];
      for (int a = 0; a < 3; a++) {
         if (c->mn[a] < n->mn[a]) n->mn[a] = c->mn[a];
         if (c->mx[a] > n->mx[a]) n->mx[a] = c->mx[a];
      }
   }
   return id;
}


static bool
vp_bvh_init(struct vp_bvh *b, uint32_t n)
{
   memset(b, 0, sizeof *b);
   b->cmin  = malloc((size_t)n * sizeof(*b->cmin));
   b->cmax  = malloc((size_t)n * sizeof(*b->cmax));
   b->cen   = malloc((size_t)n * sizeof(*b->cen));
   b->order = malloc((size_t)n * sizeof(*b->order));
   b->nodes = malloc((size_t)(2 * n) * sizeof(*b->nodes));   /* <= 2N-1 nodes */
   if (!b->cmin || !b->cmax || !b->cen || !b->order || !b->nodes)
      return false;
   for (uint32_t i = 0; i < n; i++)
      b->order[i] = i;
   return true;
}

static void
vp_bvh_fini(struct vp_bvh *b)
{
   free(b->cmin); free(b->cmax); free(b->cen); free(b->order); free(b->nodes);
}

static void
vp_bvh_set_box(struct vp_bvh *b, uint32_t i, const float mn[3], const float mx[3])
{
   for (int a = 0; a < 3; a++) {
      b->cmin[i][a] = mn[a];
      b->cmax[i][a] = mx[a];
      b->cen[i][a]  = 0.5f * (mn[a] + mx[a]);
   }
}

/* Per-axis exponent so the node extent maps into the [0,255] uint8 grid:
 * step = 2^exp ≥ ext/255. */
static void
vp_bvh_choose_exp(const float mn[3], const float mx[3], int exp[3])
{
   for (int a = 0; a < 3; a++) {
      float ext = mx[a] - mn[a];
      if (ext <= 0.f) { exp[a] = -16; continue; }
      int e = (int)ceilf(log2f(ext / 255.0f));
      if (e < -16) e = -16;
      /* ext and log2f round in fp32: grow e until the RTU's decode of q=255
       * (origin + 255*2^e, in fp32) reaches the node's max */
      while (e < 16 && mn[a] + 255.0f * ldexpf(1.0f, e) < mx[a])
         e++;
      if (e >  16) e =  16;
      exp[a] = e;
   }
}

/* Conservative quantization w.r.t. the RTU's fp32 decode origin + q*2^exp:
 * the decoded box must contain [v_lo, v_hi]. The offset is taken exactly in
 * double (fp32 v - origin can round inward), then nudged against the decode. */
static uint8_t
vp_quant(float v, float origin, int exp, bool hi)
{
   const float step = ldexpf(1.0f, exp);
   double t = ((double)v - (double)origin) / (double)step;
   double q = hi ? ceil(t) : floor(t);
   if (q < 0.0)   q = 0.0;
   if (q > 255.0) q = 255.0;
   while (hi && q < 255.0 && origin + (float)q * step < v)
      q += 1.0;
   while (!hi && q > 0.0 && origin + (float)q * step > v)
      q -= 1.0;
   return (uint8_t)q;
}

/* The scene under construction: one buffer every BLAS and the TLAS append to,
 * addressed by byte offset from its base (the RTU's scene-relative offsets). */
struct vp_sbuf {
   uint8_t *buf;
   uint32_t size, cap;
   bool ok;
};

static uint32_t
vp_sbuf_alloc(struct vp_sbuf *sb, uint32_t bytes)
{
   uint32_t off = sb->size;
   if (!sb->ok)
      return 0;
   if ((uint64_t)off + bytes > VP_RTU_SCENE_MAX_BYTES) {
      sb->ok = false;
      return 0;
   }
   if (off + bytes > sb->cap) {
      uint32_t ncap = sb->cap ? sb->cap : 4096;
      while (ncap < off + bytes)
         ncap *= 2;
      uint8_t *nb = realloc(sb->buf, ncap);
      if (!nb) { sb->ok = false; return 0; }
      memset(nb + sb->cap, 0, ncap - sb->cap);
      sb->buf = nb;
      sb->cap = ncap;
   }
   sb->size = off + bytes;
   return off;
}

/* Bytes a table of `bytes` takes appended at `size`, 64-B aligned so no node
 * straddles a line. */
static uint64_t
vp_otab_need(uint32_t size, uint64_t bytes)
{
   return (((uint64_t)size + 63u) & ~(uint64_t)63u) - size + bytes;
}

static uint64_t
vp_otab_tlas_bytes(const struct vp_otab *ot)
{
   const uint64_t nodes_off =
      (VP_OTAB_HDR_BYTES + (uint64_t)ot->leaves.count * VP_OTAB_TLAS_LEAF + 63u) & ~(uint64_t)63u;
   return nodes_off + (uint64_t)ot->nodes.count * VP_OTAB_NODE_BYTES;
}

/* Append the TLAS visit-order table to the scene; returns its offset (0 when
 * it does not fit, which the RTU reads as "no table"). Each leaf carries its
 * instance's world->object matrix, so the RTU can rebuild lavapipe's own
 * object-space ray for a tie inside a BLAS. */
static uint32_t
vp_otab_emit_tlas(struct vp_sbuf *sb, const struct vp_otab *ot,
                  const struct vp_inst *insts)
{
   const uint32_t stride = VP_OTAB_TLAS_LEAF;
   const uint64_t bytes = vp_otab_tlas_bytes(ot);
   if (sb->size + vp_otab_need(sb->size, bytes) > VP_RTU_SCENE_MAX_BYTES)
      return 0;
   const uint32_t nodes_off = (VP_OTAB_HDR_BYTES + ot->leaves.count * stride + 63u) & ~63u;
   vp_sbuf_alloc(sb, ((sb->size + 63u) & ~63u) - sb->size);
   const uint32_t off = vp_sbuf_alloc(sb, (uint32_t)bytes);
   if (!sb->ok)
      return 0;
   uint8_t *p = sb->buf + off;
   const uint32_t hdr[4] = { ot->leaves.count, ot->nodes.count, nodes_off, stride };
   memcpy(p, hdr, sizeof hdr);
   for (uint32_t i = 0; i < ot->leaves.count; i++) {
      uint8_t *lp = p + VP_OTAB_HDR_BYTES + i * stride;
      memcpy(lp, &ot->leaves.p[i], 4);
      memcpy(lp + 16, insts[i].node + LVP_INST_WTO_OFF, 48);
   }
   for (uint32_t j = 0; j < ot->nodes.count; j++) {
      uint8_t *np = p + nodes_off + j * VP_OTAB_NODE_BYTES;
      memcpy(np, ot->nodes.p[j].box, 48);
      memcpy(np + 48, &ot->nodes.p[j].ps, 4);
      memcpy(np + 52, &ot->nodes.p[j].depth, 4);
   }
   return off;
}

/* Pack a BLAS visit-order table (see the layout above) into a host blob, for
 * appending once every BLAS is in the scene and the space left is known. */
static uint8_t *
vp_otab_pack_blas(const struct vp_otab *ot, uint32_t *bytes)
{
   const uint64_t n = ot->nodes.count;
   const uint64_t sz = VP_OTAB_BLAS_HDR + n * VP_OTAB_BLAS_NODE;
   if (sz > VP_RTU_SCENE_MAX_BYTES)
      return NULL;
   uint8_t *p = calloc(1, sz);
   if (!p)
      return NULL;
   const uint32_t hdr[4] = { (uint32_t)n, 0, VP_OTAB_BLAS_HDR, VP_OTAB_BLAS_NODE };
   memcpy(p, hdr, sizeof hdr);
   for (uint32_t j = 0; j < n; j++) {
      const struct vp_onode *nd = &ot->nodes.p[j];
      uint8_t *np = p + VP_OTAB_BLAS_HDR + (size_t)j * VP_OTAB_BLAS_NODE;
      if (nd->ps != VP_OTAB_ROOT)
         memcpy(np, ot->nodes.p[nd->ps >> 1].box + 6 * (nd->ps & 1u), 24);
      memcpy(np + 24, &nd->ps, 4);
      memcpy(np + 28, &nd->depth, 4);
   }
   *bytes = (uint32_t)sz;
   return p;
}

/* Leaf encoders for vp_bvh_emit: a BLAS leaf holds one primitive, a TLAS leaf
 * one instance. */
struct vp_leaf_ops {
   uint32_t (*size)(const void *ctx, uint32_t item);
   void     (*write)(const void *ctx, uint32_t item, uint8_t *dst);
   const void *ctx;
};

/* Append a built tree to the scene; returns its root's offset. Offsets are
 * reserved for every node before any is written, since reserving may move the
 * buffer. */
static uint32_t
vp_bvh_emit(const struct vp_bvh *b, int root, const struct vp_leaf_ops *lo,
            struct vp_sbuf *sb)
{
   uint32_t *off = malloc((size_t)b->n_nodes * sizeof(uint32_t));
   if (!off) { sb->ok = false; return 0; }
   for (uint32_t i = 0; i < b->n_nodes; i++) {
      const struct vp_bnode *n = &b->nodes[i];
      off[i] = vp_sbuf_alloc(sb, n->leaf ? lo->size(lo->ctx, n->tri)
                                         : RTU_BVH4_NODE_BYTES);
   }
   if (!sb->ok) { free(off); return 0; }
   for (uint32_t i = 0; i < b->n_nodes; i++) {
      const struct vp_bnode *n = &b->nodes[i];
      uint8_t *p = sb->buf + off[i];
      if (n->leaf) {
         lo->write(lo->ctx, n->tri, p);
         continue;
      }
      uint32_t kind = RTU_BVH_KIND_INTERNAL | ((uint32_t)n->nchild << RTU_BVH_COUNT_SHIFT);
      memcpy(p, &kind, 4);
      memcpy(p + RTU_BVH4_OFF_ORIGIN, n->mn, 12);
      int exp[3];
      vp_bvh_choose_exp(n->mn, n->mx, exp);
      int8_t *pe = (int8_t *)(p + RTU_BVH4_OFF_EXP);
      pe[0] = (int8_t)exp[0]; pe[1] = (int8_t)exp[1]; pe[2] = (int8_t)exp[2];
      uint8_t *qmin = p + RTU_BVH4_OFF_QMIN;
      uint8_t *qmax = p + RTU_BVH4_OFF_QMAX;
      for (int k = 0; k < n->nchild; k++) {
         const struct vp_bnode *c = &b->nodes[n->child[k]];
         uint32_t cw = off[n->child[k]] | (c->leaf ? RTU_BVH_CHILD_LEAF_FLAG : 0u);
         memcpy(p + RTU_BVH4_OFF_CHILD + 4 * k, &cw, 4);
         for (int a = 0; a < 3; a++) {
            qmin[k * 3 + a] = vp_quant(c->mn[a], n->mn[a], exp[a], false);
            qmax[k * 3 + a] = vp_quant(c->mx[a], n->mn[a], exp[a], true);
         }
      }
   }
   uint32_t r = off[root];
   free(off);
   return r;
}

static uint32_t
vp_prim_leaf_size(const void *ctx, uint32_t i)
{
   const struct vp_prim *pr = &((const struct vp_prim *)ctx)[i];
   return RTU_BVH_LEAF_HDR_BYTES + ((pr->flags & RTU_BVH_FLAG_PROCEDURAL)
                                    ? RTU_PROC_AABB_BYTES : RTU_TRI_STRIDE);
}

static void
vp_prim_leaf_write(const void *ctx, uint32_t i, uint8_t *p)
{
   const struct vp_prim *pr = &((const struct vp_prim *)ctx)[i];
   const bool proc = (pr->flags & RTU_BVH_FLAG_PROCEDURAL) != 0;
   uint32_t hdr[4] = {
      (proc ? RTU_BVH_KIND_LEAF_PROC : RTU_BVH_KIND_LEAF_TRI) | (1u << RTU_BVH_COUNT_SHIFT),
      pr->geom_id,          /* gl_GeometryIndexEXT */
      /* A procedural leaf carries its flags word here; a triangle leaf its
       * place in lavapipe's tree (parent << 1 | side in the BLAS's
       * visit-order table), which the RTU uses to break exact-t ties the way
       * lavapipe's first-visited-wins does. */
      proc ? pr->flags : pr->ps,
      pr->prim_id,          /* prim_base = gl_PrimitiveID */
   };
   memcpy(p, hdr, sizeof hdr);
   if (proc) {
      memcpy(p + RTU_BVH_LEAF_HDR_BYTES, pr->v, 24);
   } else {
      memcpy(p + RTU_BVH_LEAF_HDR_BYTES, pr->v, 36);
      memcpy(p + RTU_BVH_LEAF_HDR_BYTES + RTU_TRI_FLAGS_OFFSET, &pr->flags, 4);
   }
}

/* One distinct BLAS in the scene. */
struct vp_blas_entry {
   const uint8_t *host;
   uint32_t       root;      /* scene offset of its root node */
   uint32_t       otab;      /* scene offset of its visit-order table (0: none) */
   uint8_t       *otab_blob; /* the table, until appended */
   uint32_t       otab_bytes;
   float          mn[3], mx[3];
   bool           empty;
};

/* Build one BLAS into the scene. Its primitives are gathered from lavapipe's
 * BVH and rebuilt with binned SAH into the RTU's CW-BVH4 format. */
static void
vp_emit_blas(const uint8_t *host, struct vp_blas_entry *e, struct vp_sbuf *sb)
{
   struct vp_prim_list l = { .ok = true };
   struct vp_otab ot = { .nodes.ok = true, .leaves.ok = true };
   vp_walk_blas(host, LVP_BVH_HEADER_SIZE | LVP_NODE_INTERNAL, &l, &ot);
   e->host = host;
   e->empty = true;
   if (!l.ok) { sb->ok = false; free(l.p); vp_otab_fini(&ot); return; }
   if (l.count == 0) { free(l.p); vp_otab_fini(&ot); return; }
   e->otab = 0;
   if (ot.leaf_box_mismatch)
      mesa_logw("vortexpipe: BLAS %p: %u triangle bound(s) differ from their "
                "vertices' min/max; its exact-t ties fall back to a static key",
                (const void *)host, ot.leaf_box_mismatch);
   else
      e->otab_blob = vp_otab_pack_blas(&ot, &e->otab_bytes);
   vp_otab_fini(&ot);

   struct vp_bvh b;
   if (!vp_bvh_init(&b, l.count)) { sb->ok = false; vp_bvh_fini(&b); free(l.p); return; }
   for (uint32_t i = 0; i < l.count; i++) {
      const float *v = l.p[i].v;
      float mn[3], mx[3];
      if (l.p[i].flags & RTU_BVH_FLAG_PROCEDURAL) {
         memcpy(mn, v, 12);
         memcpy(mx, v + 3, 12);
      } else {
         for (int a = 0; a < 3; a++) {
            mn[a] = fminf(v[a], fminf(v[3 + a], v[6 + a]));
            mx[a] = fmaxf(v[a], fmaxf(v[3 + a], v[6 + a]));
         }
      }
      vp_bvh_set_box(&b, i, mn, mx);
   }
   int root = vp_bvh_build(&b, 0, l.count);
   struct vp_leaf_ops lo = { vp_prim_leaf_size, vp_prim_leaf_write, l.p };
   e->root = vp_bvh_emit(&b, root, &lo, sb);
   memcpy(e->mn, b.nodes[root].mn, 12);
   memcpy(e->mx, b.nodes[root].mx, 12);
   e->empty = false;
   vp_bvh_fini(&b);
   free(l.p);
}

/* A TLAS leaf: one LEAF_INST record naming its BLAS's root. */
struct vp_tlas_leaf_ctx {
   const struct vp_inst *inst;
   const uint32_t       *blas_root;   /* per instance */
   const uint32_t       *blas_otab;   /* per instance (patched later) */
   uint32_t              tlas_otab;
   const struct vp_sbuf *sb;
   uint32_t             *leaf_off;    /* out: each instance leaf's offset */
};

static uint32_t
vp_inst_leaf_size(const void *ctx, uint32_t i)
{
   (void)ctx; (void)i;
   return RTU_BVH_LEAF_HDR_BYTES + RTU_BVH_INSTANCE_STRIDE;
}

static void
vp_inst_leaf_write(const void *ctx, uint32_t i, uint8_t *p)
{
   const struct vp_tlas_leaf_ctx *c = ctx;
   const struct vp_inst *in = &c->inst[i];
   uint32_t kind = RTU_BVH_KIND_LEAF_INST | (1u << RTU_BVH_COUNT_SHIFT);
   memcpy(p, &kind, 4);
   /* Header words 1..3: the BLAS's visit-order table, the instance's rank in
    * the TLAS's (lavapipe's DFS order, kept by the empty-BLAS compaction), and
    * the TLAS's table. */
   memcpy(p + 4, &c->blas_otab[i], 4);
   c->leaf_off[i] = (uint32_t)(p - c->sb->buf);
   memcpy(p + 8, &in->rank, 4);
   memcpy(p + 12, &c->tlas_otab, 4);
   uint8_t *rec = p + RTU_BVH_LEAF_HDR_BYTES;
   memcpy(rec, in->otw, 48);
   uint32_t cull = (in->mask & 0xffu) | (in->flags << RTU_INST_FLAGS_SHIFT);
   memcpy(rec + RTU_BVH_INSTANCE_BLAS_OFF, &c->blas_root[i], 4);
   memcpy(rec + RTU_BVH_INSTANCE_CUSTOM_OFF, &in->custom, 4);
   memcpy(rec + RTU_BVH_INSTANCE_ID_OFF, &in->id, 4);
   memcpy(rec + RTU_BVH_INSTANCE_CULL_OFF, &cull, 4);
}

/* World-space bounds of an instance: its BLAS root box through its transform. */
static void
vp_inst_bounds(const float otw[12], const float mn[3], const float mx[3],
               float wmn[3], float wmx[3])
{
   for (int a = 0; a < 3; a++) { wmn[a] = 1e30f; wmx[a] = -1e30f; }
   for (int k = 0; k < 8; k++) {
      float p[3] = { (k & 1) ? mx[0] : mn[0], (k & 2) ? mx[1] : mn[1],
                     (k & 4) ? mx[2] : mn[2] };
      float w[3];
      vp_xform_point(otw, p, w);
      for (int a = 0; a < 3; a++) {
         wmn[a] = fminf(wmn[a], w[a]);
         wmx[a] = fmaxf(wmx[a], w[a]);
      }
   }
}

/* Transcode the lavapipe AS at `tlas_host` into a two-level RTU CW-BVH4 scene
 * in device memory: one SAH BVH per distinct BLAS, shared by all its instances,
 * under a TLAS of LEAF_INST records the RTU descends with each instance's
 * transform, custom index, ID, mask and flags -- the instancing model of a
 * hardware ray tracer.
 *
 * The RTU hit record names an instance only by ID, so an instance table sits
 * just below the scene base: entry k, at base - VP_RTU_INST_TABLE_STRIDE*(k+1),
 * is a verbatim copy of lavapipe's lvp_bvh_instance_node for gl_InstanceID k.
 * A shader turns the reported ID into that address and reads the SBT offset,
 * transforms and IDs exactly as it would from lavapipe's own node.
 * Returns the scene's device address (0 on failure). */
static uint64_t
vp_transcode_as(struct vp_as_ctx *c, const void *tlas_host)
{
   struct vp_inst_list il = { .ok = true };
   struct vp_otab tot = { .nodes.ok = true, .leaves.ok = true };
   vp_walk_tlas((const uint8_t *)tlas_host, LVP_BVH_HEADER_SIZE | LVP_NODE_INTERNAL, &il, &tot);
   if (il.has_geometry && il.count == 0) {
      /* The handle names a BLAS: trace it as one identity instance. */
      static const float kIdentity[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };
      struct vp_inst in = { .blas = tlas_host, .mask = 0xffu };
      const uint32_t cm = 0xffu << 24, sf = LVP_INSTANCE_FLAGS_DEFAULT;
      memcpy(in.otw, kIdentity, sizeof kIdentity);
      memcpy(in.node + LVP_INST_CUSTOM_OFF, &cm, 4);
      memcpy(in.node + LVP_INST_SBTFLAGS_OFF, &sf, 4);
      memcpy(in.node + LVP_INST_WTO_OFF, kIdentity, sizeof kIdentity);
      memcpy(in.node + LVP_INST_OTW_OFF, kIdentity, sizeof kIdentity);
      in.rank = 0;
      VP_LIST_PUSH(&il, in);
      VP_LIST_PUSH(&tot.leaves, VP_OTAB_ROOT);
   }
   if (!il.ok || !tot.leaves.ok) { free(il.p); vp_otab_fini(&tot); c->ok = false; return 0; }

   struct vp_sbuf sb = { .ok = true };
   vp_sbuf_alloc(&sb, RTU_SCENE_HDR_BYTES);

   /* Every distinct BLAS once. */
   struct vp_blas_entry *blas = calloc(il.count ? il.count : 1, sizeof *blas);
   uint32_t *inst_blas = calloc(il.count ? il.count : 1, sizeof *inst_blas);
   uint32_t n_blas = 0, max_id = 0;
   if (!blas || !inst_blas) sb.ok = false;
   for (uint32_t i = 0; sb.ok && i < il.count; i++) {
      uint32_t k = 0;
      while (k < n_blas && blas[k].host != il.p[i].blas)
         k++;
      if (k == n_blas)
         vp_emit_blas(il.p[i].blas, &blas[n_blas++], &sb);
      inst_blas[i] = k;
      if (il.p[i].id > max_id) max_id = il.p[i].id;
   }

   /* Before compaction: the TLAS's visit-order table is indexed by every
    * collected instance. It is small; without it ties fall back to a static
    * key. */
   const uint32_t tlas_otab = sb.ok ? vp_otab_emit_tlas(&sb, &tot, il.p) : 0;
   vp_otab_fini(&tot);

   /* The TLAS over the non-empty instances. */
   uint32_t n_live = 0;
   for (uint32_t i = 0; sb.ok && i < il.count; i++) {
      if (blas[inst_blas[i]].empty)
         continue;
      il.p[n_live] = il.p[i];
      inst_blas[n_live] = inst_blas[i];
      n_live++;
   }
   uint32_t hdr[4] = { 0, RTU_SCENE_KIND_TRILIST, 0, 0 };   /* empty: all-miss */
   uint32_t *leaf_off = calloc(n_live ? n_live : 1, sizeof *leaf_off);
   if (!leaf_off) sb.ok = false;
   if (sb.ok && n_live) {
      struct vp_bvh b = { 0 };
      uint32_t *roots = malloc(n_live * sizeof *roots);
      uint32_t *otabs = calloc(n_live, sizeof *otabs);
      if (!roots || !otabs || !vp_bvh_init(&b, n_live)) {
         sb.ok = false;
      } else {
         for (uint32_t i = 0; i < n_live; i++) {
            const struct vp_blas_entry *e = &blas[inst_blas[i]];
            float wmn[3], wmx[3];
            vp_inst_bounds(il.p[i].otw, e->mn, e->mx, wmn, wmx);
            vp_bvh_set_box(&b, i, wmn, wmx);
            roots[i] = e->root;
         }
         int root = vp_bvh_build(&b, 0, n_live);
         struct vp_tlas_leaf_ctx lc = { il.p, roots, otabs, tlas_otab, &sb, leaf_off };
         struct vp_leaf_ops lo = { vp_inst_leaf_size, vp_inst_leaf_write, &lc };
         hdr[0] = vp_bvh_emit(&b, root, &lo, &sb);
         hdr[1] = RTU_SCENE_KIND_BVH4;
         hdr[3] = b.n_nodes;
      }
      vp_bvh_fini(&b);
      free(roots);
      free(otabs);
   }

   /* The BLAS visit-order tables go in a buffer of their own: a BLAS without
    * one only loses lavapipe's tie order (its near-equal-t ties fall back to a
    * static key), so they must not take the scene's own room -- its 31-bit
    * offsets, or the largest buffer the device hands out. The RTU reaches
    * them by 32-bit offsets from the scene base, patched in once both buffers
    * have addresses. */
   const uint32_t n_ids = il.count ? max_id + 1 : 0;
   const uint32_t tbl = n_ids * VP_RTU_INST_TABLE_STRIDE;
   const uint32_t base_bytes = sb.size;
   struct vp_sbuf tb = { .ok = true };
   uint32_t otab_kept = 0, otab_dropped = 0;
   for (uint32_t k = 0; k < n_blas; k++) {
      struct vp_blas_entry *e = &blas[k];
      e->otab = 0;
      if (!e->otab_blob)
         continue;
      const uint32_t before = tb.size;
      if (tb.ok && (uint64_t)vp_otab_need(tb.size, e->otab_bytes) + tb.size
                   <= VP_RTU_SCENE_MAX_BYTES) {
         vp_sbuf_alloc(&tb, ((tb.size + 63u) & ~63u) - tb.size);
         const uint32_t off = vp_sbuf_alloc(&tb, e->otab_bytes);
         if (tb.ok) {
            memcpy(tb.buf + off, e->otab_blob, e->otab_bytes);
            e->otab = off + 1u;   /* +1: offset 0 is a table too; fixed below */
            otab_kept++;
         }
      }
      if (!e->otab) {
         tb.size = tb.ok ? before : tb.size;
         otab_dropped++;
      }
      free(e->otab_blob);
      e->otab_blob = NULL;
   }
   if (!tb.ok) {
      otab_dropped += otab_kept;
      otab_kept = 0;
      for (uint32_t k = 0; k < n_blas; k++)
         blas[k].otab = 0;
   }
   hdr[2] = sb.size;
   if (sb.ok)
      memcpy(sb.buf, hdr, sizeof hdr);

   /* Prefix the instance table; its stride keeps the scene 64-B aligned. */
   uint8_t *img = sb.ok ? calloc(1, (size_t)tbl + sb.size) : NULL;
   if (img) {
      for (uint32_t i = 0; i < n_live; i++)
         memcpy(img + tbl - VP_RTU_INST_TABLE_STRIDE * (il.p[i].id + 1u),
                il.p[i].node, LVP_INST_NODE_BYTES);
      memcpy(img + tbl, sb.buf, sb.size);
   }
   uint32_t img_size = tbl + sb.size;
   vp_dbg("vortexpipe: RTU scene: %u instance(s), %u BLAS, %u bytes; "
          "BLAS visit-order tables: %u bytes for %u BLAS (%u without)",
          n_live, n_blas, base_bytes, tb.size, otab_kept, otab_dropped);
   /* VORTEXPIPE_DUMP_SCENE=<prefix>: write each transcoded scene image to
    * <prefix>.<n>.bin (the scene base is at byte offset `tbl`, logged). */
   const char *dump = getenv("VORTEXPIPE_DUMP_SCENE");
   if (img && dump) {
      static unsigned dump_seq;
      char path[512];
      snprintf(path, sizeof path, "%s.%u.bin", dump, dump_seq++);
      FILE *f = fopen(path, "wb");
      if (f) {
         fwrite(img, 1, img_size, f);
         fclose(f);
         mesa_logi("vortexpipe: dumped RTU scene to %s (base offset %u, %u instance(s))",
                   path, tbl, n_live);
      }
   }
   free(sb.buf);
   if (!img) {
      free(blas); free(inst_blas); free(il.p); free(leaf_off);
      c->ok = false; return 0;
   }

   if (c->n_bufs >= VP_MAX_BVH || c->n_stages >= VP_MAX_BVH) {
      free(blas); free(inst_blas); free(il.p); free(leaf_off);
      free(img); c->ok = false; return 0;
   }
   /* The upload is asynchronous; keep the image alive until the queue
    * finishes (freed by the launch cleanup, like vp_copy_as's stages). */
   c->stages[c->n_stages++] = img;
   vx_buffer_h buf = NULL;
   uint64_t dev_addr = 0;
   if (vx_buffer_create(c->dev, img_size, VX_MEM_READ, &buf) != VX_SUCCESS ||
       vx_buffer_address(buf, &dev_addr) != VX_SUCCESS) {
      free(blas); free(inst_blas); free(il.p); free(leaf_off); free(tb.buf);
      c->ok = false; return 0;
   }
   c->bufs[c->n_bufs++] = buf;

   /* The tables, if they get a buffer the RTU can reach from the scene. */
   if (otab_kept && c->n_bufs < VP_MAX_BVH && c->n_stages < VP_MAX_BVH) {
      vx_buffer_h tbuf = NULL;
      uint64_t taddr = 0;
      const uint64_t scene = dev_addr + tbl;
      if (vx_buffer_create(c->dev, tb.size, VX_MEM_READ, &tbuf) == VX_SUCCESS) {
         c->bufs[c->n_bufs++] = tbuf;
         if (vx_buffer_address(tbuf, &taddr) == VX_SUCCESS && taddr >= scene
             && taddr - scene + tb.size <= 0xffffff00ull) {
            const uint32_t delta = (uint32_t)(taddr - scene);
            for (uint32_t i = 0; i < n_live; i++) {
               const uint32_t o = blas[inst_blas[i]].otab;
               const uint32_t w = o ? delta + (o - 1u) : 0u;
               memcpy(img + tbl + leaf_off[i] + 4, &w, 4);
            }
            if (vx_enqueue_write(c->q, tbuf, 0, tb.buf, tb.size, 0, NULL, NULL)
                == VX_SUCCESS) {
               c->stages[c->n_stages++] = tb.buf;
               tb.buf = NULL;
            } else {
               c->ok = false;
            }
         } else {
            mesa_logw("vortexpipe: RTU scene: visit-order tables unreachable "
                      "from the scene; near-equal-t ties fall back to a static key");
         }
      } else {
         mesa_logw("vortexpipe: RTU scene: no device memory for %u bytes of "
                   "visit-order tables; near-equal-t ties fall back to a static key",
                   tb.size);
      }
   }
   free(tb.buf);
   free(blas); free(inst_blas); free(il.p); free(leaf_off);
   if (!c->ok ||
       vx_enqueue_write(c->q, buf, 0, img, img_size, 0, NULL, NULL) != VX_SUCCESS) {
      c->ok = false; return 0;
   }
   return dev_addr + tbl;
}

/* A draw reaches its descriptors through its own loops and tears them down at
 * its own point, but the relocation itself is the same work as a dispatch's.
 * The context stays opaque to the draw: every device buffer and staging blob
 * the copy allocates has to outlive vx_queue_finish, and handing out the struct
 * would make that lifetime the caller's to reconstruct rather than to observe. */
struct vp_as_ctx *
vp_as_begin(vx_device_h dev, vx_queue_h q, bool has_rtu)
{
   struct vp_as_ctx *c = calloc(1, sizeof *c);
   if (!c) {
      return NULL;
   }
   c->dev     = dev;
   c->q       = q;
   c->has_rtu = has_rtu;
   c->ok      = true;
   return c;
}

/* Device address of the acceleration structure whose host root is `tlas_host`,
 * transcoded to the RTU's scene format where the device has one and copied
 * verbatim otherwise. Returns 0 without recording a failure for a null handle,
 * which is a descriptor the shader never binds rather than an error. */
uint64_t
vp_as_relocate(struct vp_as_ctx *c, uint64_t tlas_host)
{
   if (!c || !c->ok || !tlas_host) {
      return 0;
   }
   return c->has_rtu ? vp_transcode_as(c, (const void *)(uintptr_t)tlas_host)
                     : vp_copy_as(c, (const void *)(uintptr_t)tlas_host);
}

bool
vp_as_ok(const struct vp_as_ctx *c)
{
   return c && c->ok;
}

/* Release everything the relocations allocated. Called after vx_queue_finish,
 * never before: the uploads are asynchronous and read the staging blobs. */
void
vp_as_end(struct vp_as_ctx *c)
{
   if (!c) {
      return;
   }
   for (unsigned i = 0; i < c->n_bufs; i++) {
      if (c->bufs[i]) {
         vx_buffer_release(c->bufs[i]);
      }
   }
   for (unsigned i = 0; i < c->n_stages; i++) {
      free(c->stages[i]);
   }
   free(c);
}

bool
vp_launch(struct pipe_screen *screen, vx_device_h dev,
          const void *vxbin, size_t vxbin_size,
          vx_module_h *module_io, vx_kernel_h *kernel_io,
          const void *desc_host, uint32_t desc_bytes,
          const struct vp_desc *descs, uint32_t num_descs,
          const struct vp_ssbo *ssbos, uint32_t num_ssbos,
          const struct vp_tex_heap *tex_heap,
          const uint32_t grid[3], const uint32_t block[3],
          const uint32_t grid_base[3],
          uint32_t lmem_size, bool has_rtu)
{
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_module_h kmod = NULL;
   vx_kernel_h kbuf = NULL;
   vx_buffer_h dbuf = NULL;
   vx_buffer_h hbuf = NULL;   /* bindless texture heap */
   /* Resident device buffers are owned by the screen and outlive the dispatch,
    * so these are borrowed handles -- released here they would be freed out
    * from under the next dispatch that resolves to the same host range. */
   vx_buffer_h res[VP_MAX_DESCS]      = { 0 };  /* borrowed resident buffer    */
   uint32_t    res_off[VP_MAX_DESCS]  = { 0 };  /* range's offset within it    */
   void       *res_host[VP_MAX_DESCS] = { 0 };  /* its host backing            */
   uint32_t    res_bytes[VP_MAX_DESCS]= { 0 };
   vx_buffer_h sres[VP_MAX_SSBO]      = { 0 };  /* per-slot raw-SSBO device buffer */
   vx_buffer_h sbt_res[VP_MAX_SSBO * 4] = { 0 }; /* relocated SBT shader-record buffers */
   uint32_t    n_sbt = 0;
   uint8_t    *cmd_copy[VP_MAX_SSBO]  = { 0 };  /* mutable trace-command copies */
   struct vp_as_ctx asc = { .ok = true };       /* acceleration-structure BVHs */
   uint8_t    *stage = NULL;

   /* A private copy of the descriptor buffer: vp_launch rewrites the
    * resource pointers inside it to device addresses. It must outlive
    * vx_queue_finish (vx_enqueue_write reads it asynchronously). */
   stage = malloc(desc_bytes);
   if (!stage) {
      mesa_loge("vortexpipe: launch: descriptor staging OOM");
      return false;
   }
   memcpy(stage, desc_host, desc_bytes);

   vx_queue_info_t qi = {
      .struct_size = sizeof(qi), .next = NULL,
      .priority = VX_QUEUE_PRIORITY_NORMAL, .flags = 0,
   };
   VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");
   asc.dev = dev;
   asc.q   = q;
   asc.has_rtu = has_rtu;

   /* Load the kernel image straight from memory — no /tmp round-trip — and
    * leave it resident in the caller's slot, so a repeated dispatch of the same
    * pipeline reuses it. "main" is the public name vxbin.py assigns the single
    * conventional kernel (the C entry is "kernel_main"); match the native
    * runtime. */
   if (*module_io == NULL) {
      VP_CHECK(vx_module_load_bytes(dev, vxbin, vxbin_size, module_io),
               "vx_module_load_bytes");
      VP_CHECK(vx_module_get_kernel(*module_io, "main", kernel_io),
               "vx_module_get_kernel");
   }
   kbuf = *kernel_io;

   /* Relocate each descriptor into the staged descriptor blob:
    *  - VP_DESC_BUFFER: copy the resource into device memory, rewrite
    *    lp_jit_buffer.ptr so load_ssbo/store_ssbo dereference on-device.
    *  - VP_DESC_AS: copy the BVH (TLAS + its BLASes) into device
    *    memory with instance-node links relocated, rewrite the
    *    accel_struct device address.
    *  - VP_DESC_IMAGE: copy the storage image into device memory,
    *    rewrite lp_jit_image.base, read back after the launch. */
   for (uint32_t i = 0; i < num_descs; i++) {
      uint8_t *slot = stage + descs[i].offset;

      if (descs[i].kind == VP_DESC_AS) {
         uint64_t tlas_host = 0;
         memcpy(&tlas_host, slot, sizeof tlas_host);   /* lp_descriptor.accel_struct */
         if (!tlas_host)
            continue;
         /* With the RTU, transcode the AS into the RTU scene format; the
          * kernel's vx_rt_trace consumes that. Otherwise copy verbatim. */
         uint64_t tlas_dev = asc.has_rtu
            ? vp_transcode_as(&asc, (const void *)(uintptr_t)tlas_host)
            : vp_copy_as(&asc, (const void *)(uintptr_t)tlas_host);
         if (!asc.ok) {
            mesa_loge("vortexpipe: launch: acceleration-structure copy failed");
            goto done;
         }
         memcpy(slot, &tlas_dev, sizeof tlas_dev);
         continue;
      }

      if (descs[i].kind == VP_DESC_IMAGE) {
         /* Storage image: copy the host backing to device memory and rewrite
          * lp_jit_image.base, mirroring the buffer path. Upload + readback is
          * the correct full-duplex treatment for an accumulation image, which
          * is both read and written each frame; an image the shader only loads
          * is uploaded and released without the readback. */
         uint64_t host_base = 0;
         uint16_t height    = 0;
         uint32_t row       = 0;
         uint32_t base_off  = 0;
         memcpy(&host_base, slot + VP_JIT_IMG_BASE,        sizeof host_base);
         memcpy(&height,    slot + VP_JIT_IMG_HEIGHT,      sizeof height);
         memcpy(&row,       slot + VP_JIT_IMG_ROW_STRIDE,  sizeof row);
         memcpy(&base_off,  slot + VP_JIT_IMG_BASE_OFFSET, sizeof base_off);
         if (!host_base || !height || !row)
            continue;
         uint32_t isize = base_off + (uint32_t)height * row;
         void *img_upload = (void *)(uintptr_t)host_base;
         bool dirty = true;
         uint64_t dev_addr = vp_screen_resident_addr(screen, img_upload, isize,
                                                     &res[i], &res_off[i], &dirty);
         if (!dev_addr) {
            mesa_loge("vortexpipe: launch: no device memory for image");
            goto done;
         }
         if (dirty) {
            VP_CHECK(vx_enqueue_write(q, res[i], res_off[i], img_upload,
                                      isize, 0, NULL, NULL),
                     "vx_enqueue_write(image)");
            vp_screen_resident_clean(screen, img_upload, isize);
         }
         memcpy(slot + VP_JIT_IMG_BASE, &dev_addr, sizeof dev_addr);
         if (descs[i].writable) {
            res_host[i]  = (void *)(uintptr_t)host_base;
            res_bytes[i] = isize;
         }
         continue;
      }

      /* VP_DESC_BUFFER */
      uint64_t host_ptr = 0;
      uint32_t nelem = 0;
      memcpy(&host_ptr, slot + VP_JIT_BUF_PTR,  sizeof host_ptr);
      memcpy(&nelem,    slot + VP_JIT_BUF_SIZE, sizeof nelem);
      if (!host_ptr || !nelem)
         continue;
      /* lp_jit_buffer.num_elements is a COUNT: bytes for an SSBO (elem_bytes 1),
       * dwords for a UBO (elem_bytes 4, DIV_ROUND_UP(size, sizeof(float))). */
      uint32_t size = nelem * (descs[i].elem_bytes ? descs[i].elem_bytes : 1);
      bool dirty = true;
      uint64_t dev_addr = vp_screen_resident_addr(screen,
                                                  (const void *)(uintptr_t)host_ptr,
                                                  size, &res[i], &res_off[i], &dirty);
      if (!dev_addr) {
         mesa_loge("vortexpipe: launch: no device memory for resource");
         goto done;
      }
      if (dirty) {
         VP_CHECK(vx_enqueue_write(q, res[i], res_off[i], (void *)(uintptr_t)host_ptr,
                                   size, 0, NULL, NULL),
                  "vx_enqueue_write(resource)");
         vp_screen_resident_clean(screen, (const void *)(uintptr_t)host_ptr, size);
      } else {
         /* Reported, not silent: an upload skip that cannot be observed cannot
          * be shown to work, and two residency changes in this driver have
          * already measured zero because nothing said whether they fired. */
         vp_dbg("vortexpipe: launch: resource upload skipped, %u bytes already "
                "resident and clean", size);
      }
      memcpy(slot + VP_JIT_BUF_PTR, &dev_addr, sizeof dev_addr);
      /* Only a descriptor the shader stores to needs its device copy brought
       * back. A UBO cannot be written at all, and an SSBO this shader only
       * loads returns exactly the bytes that were just uploaded. */
      if (descs[i].writable) {
         res_host[i]  = (void *)(uintptr_t)host_ptr;
         res_bytes[i] = size;
      }
   }

   if (tex_heap && tex_heap->count) {
      const uint32_t bytes = tex_heap->count * (uint32_t)sizeof(gfx_sw_texstate_t);
      uint64_t heap_dev = 0;
      VP_CHECK(vx_buffer_create(dev, bytes, 0, &hbuf), "vx_buffer_create(tex heap)");
      VP_CHECK(vx_buffer_address(hbuf, &heap_dev), "vx_buffer_address(tex heap)");
      VP_CHECK(vx_enqueue_write(q, hbuf, 0, tex_heap->entries, bytes, 0, NULL, NULL),
               "vx_enqueue_write(tex heap)");
      for (uint32_t k = 0; k < tex_heap->count; k++) {
         uint64_t a = heap_dev + (uint64_t)k * sizeof(gfx_sw_texstate_t);
         memcpy(stage + tex_heap->slot[k], &a, sizeof a);
      }
   }

   /* upload the relocated descriptor buffer */
   VP_CHECK(vx_buffer_create(dev, desc_bytes, 0, &dbuf),
            "vx_buffer_create(descriptors)");
   uint64_t desc_dev = 0;
   VP_CHECK(vx_buffer_address(dbuf, &desc_dev), "vx_buffer_address(descriptors)");
   VP_CHECK(vx_enqueue_write(q, dbuf, 0, stage, desc_bytes, 0, NULL, NULL),
            "vx_enqueue_write(descriptors)");

   /* arg block: i64[VP_ARG_SLOTS]; slot 1 -> set-0 descriptor buffer.
    * In the current vortex2 API the runtime stages the arg blob into a
    * scratch slot at launch time — we pass it inline via args_host
    * instead of allocating an args buffer. */
   uint64_t argblk[VP_ARG_SLOTS] = { 0 };
   argblk[1] = desc_dev;
   /* vkCmdDispatchBase base offset -> gl_WorkGroupID (added in-shader). */
   if (grid_base) {
      argblk[VP_ARG_GRID_BASE_XY] = (uint64_t)grid_base[0]
                                  | ((uint64_t)grid_base[1] << 32);
      argblk[VP_ARG_GRID_BASE_Z]  = (uint64_t)grid_base[2];
   }

   /* Relocate each raw shader-buffer slot into the device and record its data
    * address in arg[VP_ARG_SSBO_BASE + slot]. Upload-only (input buffers). */
   for (uint32_t s = 0; s < num_ssbos; s++) {
      unsigned slot = ssbos[s].slot;
      if (slot >= VP_MAX_SSBO || !ssbos[s].host || !ssbos[s].size)
         continue;
      /* The RT trace-ray command buffer (VkTraceRaysIndirectCommand2KHR)
       * embeds SBT shader-record *device addresses* that lavapipe fills with
       * HOST pointers -- the megashader dereferences the raygen record on-
       * device to select the raygen (gld==raygen-id) and would read garbage.
       * Copy each present SBT record to device memory and rewrite its pointer
       * in a private copy of the command buffer before upload. */
      const void *upload = ssbos[s].host;
      if (ssbos[s].trace_cmd && ssbos[s].size >= 96) {
         /* {device-address offset, size-field offset} for each SBT region:
          * raygen, miss, hit, callable (offsets per the Vulkan struct). */
         static const struct { uint32_t addr_off, size_off; } sbt[4] = {
            { 0, 8 }, { 16, 24 }, { 40, 48 }, { 64, 72 },
         };
         cmd_copy[slot] = malloc(ssbos[s].size);
         if (!cmd_copy[slot]) {
            mesa_loge("vortexpipe: launch: trace-cmd copy OOM");
            goto done;
         }
         memcpy(cmd_copy[slot], ssbos[s].host, ssbos[s].size);
         for (unsigned r = 0; r < 4; r++) {
            uint64_t haddr;
            uint64_t rsize;
            memcpy(&haddr, cmd_copy[slot] + sbt[r].addr_off, sizeof haddr);
            memcpy(&rsize, cmd_copy[slot] + sbt[r].size_off, sizeof rsize);
            if (!haddr || !rsize || rsize > (1u << 20) ||
                n_sbt >= VP_MAX_SSBO * 4)
               continue;
            vx_buffer_h *rb = &sbt_res[n_sbt++];
            VP_CHECK(vx_buffer_create(dev, (uint32_t)rsize, 0, rb),
                     "vx_buffer_create(sbt)");
            uint64_t rdev = 0;
            VP_CHECK(vx_buffer_address(*rb, &rdev), "vx_buffer_address(sbt)");
            VP_CHECK(vx_enqueue_write(q, *rb, 0, (void *)(uintptr_t)haddr,
                                      (uint32_t)rsize, 0, NULL, NULL),
                     "vx_enqueue_write(sbt)");
            memcpy(cmd_copy[slot] + sbt[r].addr_off, &rdev, sizeof rdev);
         }
         upload = cmd_copy[slot];
      }

      VP_CHECK(vx_buffer_create(dev, ssbos[s].size, 0, &sres[slot]),
               "vx_buffer_create(ssbo)");
      uint64_t sdev = 0;
      VP_CHECK(vx_buffer_address(sres[slot], &sdev), "vx_buffer_address(ssbo)");
      VP_CHECK(vx_enqueue_write(q, sres[slot], 0, (void *)(uintptr_t)upload,
                                ssbos[s].size, 0, NULL, NULL),
               "vx_enqueue_write(ssbo)");
      argblk[VP_ARG_SSBO_BASE + slot] = sdev;
   }

   /* dispatch */
   uint32_t ndim = (grid[2] > 1 || block[2] > 1) ? 3
                 : (grid[1] > 1 || block[1] > 1) ? 2 : 1;
   vx_launch_info_t li = {
      .struct_size = sizeof(li), .next = NULL,
      .kernel = kbuf,
      .args_host = argblk, .args_size = sizeof(argblk),
      .ndim = ndim,
      .grid_dim  = { grid[0],  grid[1],  grid[2]  },
      .block_dim = { block[0], block[1], block[2] },
      .lmem_size = lmem_size,
   };
   VP_CHECK(vx_enqueue_launch(q, &li, 0, NULL, NULL), "vx_enqueue_launch");

   /* copy every written buffer back into its host backing */
   for (uint32_t i = 0; i < num_descs; i++) {
      if (!res[i] || !res_host[i])
         continue;
      VP_CHECK(vx_enqueue_read(q, res_host[i], res[i], res_off[i], res_bytes[i],
                               0, NULL, NULL), "vx_enqueue_read(resource)");
   }
   VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");

   ok = true;

done:
   for (uint32_t i = 0; i < VP_MAX_SSBO; i++)
      if (sres[i]) vx_buffer_release(sres[i]);
   for (uint32_t i = 0; i < n_sbt; i++)
      if (sbt_res[i]) vx_buffer_release(sbt_res[i]);
   for (uint32_t i = 0; i < VP_MAX_SSBO; i++)
      free(cmd_copy[i]);
   for (unsigned i = 0; i < asc.n_bufs; i++)
      if (asc.bufs[i]) vx_buffer_release(asc.bufs[i]);
   for (unsigned i = 0; i < asc.n_stages; i++)
      free(asc.stages[i]);
   if (dbuf) vx_buffer_release(dbuf);
   if (hbuf) vx_buffer_release(hbuf);
   /* kbuf aliases *kernel_io and stays resident; the caller releases it. */
   if (q)    vx_queue_release(q);
   free(stage);
   return ok;
}

/* How many VS input driver_locations the attribute table covers. The entry
 * layout itself is VP_ATTR_ENTRY_WORDS, shared with the fetch lowering. */
#define VP_ATTR_TABLE_LOCS  8

bool
vp_launch_vs(struct pipe_screen *screen, vx_device_h dev,
             const void *vxbin, size_t vxbin_size,
             uint32_t vertex_count, uint32_t out_bytes,
             const struct vp_vertex_input *vin,
             vx_buffer_h *out_buf, uint64_t *out_addr)
{
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_module_h kmod = NULL;
   vx_kernel_h kbuf = NULL;
   vx_buffer_h obuf = NULL, tbuf = NULL, dtbuf = NULL;
   /* Borrowed from the screen's residency table -- one per distinct vertex
    * buffer, resident for the resource's life rather than the draw's. */
   vx_buffer_h vbufs_dev[8] = { NULL };
   char vxpath[512];
   const char *vs_tmpdir = getenv("TMPDIR");
   if (!vs_tmpdir || !*vs_tmpdir)
      vs_tmpdir = "/tmp";
   snprintf(vxpath, sizeof vxpath, "%s/vortexpipe-vs.XXXXXX", vs_tmpdir);
   int  vxfd = -1;

   *out_buf  = NULL;
   *out_addr = 0;

   vxfd = mkstemp(vxpath);
   if (vxfd < 0) {
      mesa_loge("vortexpipe: vs launch: mkstemp failed");
      return false;
   }
   if (write(vxfd, vxbin, vxbin_size) != (ssize_t)vxbin_size) {
      mesa_loge("vortexpipe: vs launch: writing .vxbin failed");
      close(vxfd);
      unlink(vxpath);
      return false;
   }
   close(vxfd);

   vx_queue_info_t qi = {
      .struct_size = sizeof(qi), .next = NULL,
      .priority = VX_QUEUE_PRIORITY_NORMAL, .flags = 0,
   };
   VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");
   VP_CHECK(vx_module_load_file(dev, vxpath, &kmod),
            "vx_module_load_file");
   /* "main" is the public name vxbin.py assigns the single conventional
    * kernel (the C entry is "kernel_main"); match the native runtime. */
   VP_CHECK(vx_module_get_kernel(kmod, "main", &kbuf),
            "vx_module_get_kernel");

   /* Query device geometry so the VS launch maximises warp utilization:
    *   block_dim = round_up(vertex_count, num_threads), capped at the
    *               max CTA size (num_threads × num_warps). This keeps
    *               every active warp's tmask full (no partial trailing
    *               warp) and lets a CTA saturate one whole core.
    *   grid_dim  = ceil(vertex_count / block_dim) so the work spreads
    *               across cores (KMU hands one CTA to each free core).
    *
    * Pre-fix shape was grid=(1,1,1) block=(vertex_count,1,1) which:
    *   - silently truncated vertex_count > max_block_size at the DCR
    *     write (KMU's CTA_TID_WIDTH+1 cap),
    *   - left the trailing warp partially-masked when vertex_count
    *     wasn't a multiple of num_threads (degraded warp util),
    *   - sat all work on one core (the other num_cores-1 idle).
    *
    * Padding policy: the device output buffer is sized to grid × block
    * × stride. Out-of-bounds threads (vid in [vertex_count,
    * grid*block)) write to the pad region; consumers read only the first
    * `vertex_count` records, so the slack is never observed. This avoids
    * needing a bounds-check intrinsic in the VS NIR lowering. */
   uint64_t nt = 0, nw = 0;
   VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt),
            "vx_device_query(NUM_THREADS)");
   VP_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &nw),
            "vx_device_query(NUM_WARPS)");
   const uint32_t num_threads  = (uint32_t)nt;
   const uint32_t num_warps    = (uint32_t)nw;
   const uint32_t cta_size_max = num_threads * num_warps;

   uint32_t block_x = (vertex_count + num_threads - 1u) / num_threads
                     * num_threads;          /* round up to nt multiple */
   if (block_x > cta_size_max) block_x = cta_size_max;
   if (block_x == 0)           block_x = num_threads;
   uint32_t grid_x  = (vertex_count + block_x - 1u) / block_x;
   uint32_t launched_threads = grid_x * block_x;
   uint32_t stride           = (vertex_count > 0) ? out_bytes / vertex_count : 0;
   uint32_t obuf_bytes       = launched_threads * stride;
   if (obuf_bytes < out_bytes) obuf_bytes = out_bytes;

   /* output vertex-record buffer + its device address (padded for the
    * trailing CTA's out-of-bounds threads). */
   VP_CHECK(vx_buffer_create(dev, obuf_bytes, 0, &obuf),
            "vx_buffer_create(out)");
   uint64_t out_dev = 0;
   VP_CHECK(vx_buffer_address(obuf, &out_dev), "vx_buffer_address");

   /* arg block: slot 0 -> output buffer device address,
    *            slot 1 -> vertex attribute table (0 if self-contained),
    *            VP_ARG_VS_COUNT -> vertex invocations the draw really has,
    *            VP_ARG_VS_DESC -> VS constant-buffer table (see below) */
   uint64_t argblk[VP_ARG_SLOTS] = { 0 };
   argblk[0] = out_dev;
   /* The block above is rounded up to fill its warps, so the VS prologue stops
    * every thread whose vertex id is past the draw. It reads the real count
    * from here, and a zero here stops all of them -- the launch appears to
    * succeed and the output buffer comes back exactly as allocated. */
   argblk[VP_ARG_VS_COUNT] = vertex_count;

   /* This standalone path (the llvmpipe raster fallback) has no descriptor
    * state, but the VS prologue always dereferences VP_ARG_VS_DESC to reach
    * its constant buffers, so the slot must hold a real table rather than 0.
    * Upload a zero-filled one: a VS that reads a UBO here still gets wrong
    * data -- this path never supported descriptors -- but it does not fault.
    * Wiring real constant buffers through here is tracked separately. */
   uint64_t vs_desc_table[GFX_FS_DESC_SLOTS] = { 0 };
   VP_CHECK(vx_buffer_create(dev, sizeof(vs_desc_table), 0, &dtbuf),
            "vx_buffer_create(vs_desc)");
   uint64_t vs_desc_dev = 0;
   VP_CHECK(vx_buffer_address(dtbuf, &vs_desc_dev), "vx_buffer_address(vs_desc)");
   VP_CHECK(vx_enqueue_write(q, dtbuf, 0, vs_desc_table, sizeof(vs_desc_table),
                             0, NULL, NULL), "vx_enqueue_write(vs_desc)");
   argblk[VP_ARG_VS_DESC] = vs_desc_dev;

   /* Vertex buffer + attribute table: upload the interleaved vertex
    * buffer, then a table indexed by driver_location holding the
    * device base address + stride of each attribute. The VS kernel
    * fetches input `loc` at table[loc].base + index*table[loc].stride, the
    * index taken from the vertex or, for a non-zero divisor, from
    * instance/divisor.
    *
    * `table` is declared at function scope: vx_enqueue_write is
    * asynchronous (the source is read at vx_queue_finish), so it must
    * outlive the `if` block -- the same lifetime rule as argblk. */
   uint32_t table[VP_ATTR_TABLE_LOCS * VP_ATTR_ENTRY_WORDS] = { 0 };
   if (vin && vin->num_attrs) {
      /* Upload each distinct vertex-buffer resource once; an attribute points at
       * its own buffer's device base + its byte offset. */
      uint64_t buf_dev[8] = { 0 };
      uint32_t buf_off[8] = { 0 };
      for (uint32_t j = 0; j < vin->num_bufs; j++) {
         bool vdirty = true;
         buf_dev[j] = vp_screen_resident_addr(screen, vin->buf_data[j],
                                              vin->buf_size[j],
                                              &vbufs_dev[j], &buf_off[j], &vdirty);
         if (!buf_dev[j]) {
            mesa_loge("vortexpipe: launch_vs: no device memory for vertex buffer");
            goto done;
         }
         if (vdirty) {
            VP_CHECK(vx_enqueue_write(q, vbufs_dev[j], buf_off[j], vin->buf_data[j],
                                      vin->buf_size[j], 0, NULL, NULL),
                     "vx_enqueue_write(vbuf)");
            vp_screen_resident_clean(screen, vin->buf_data[j], vin->buf_size[j]);
         }
      }

      for (uint32_t i = 0; i < vin->num_attrs; i++) {
         uint32_t loc = vin->attr_loc[i];
         if (loc >= VP_ATTR_TABLE_LOCS)
            continue;
         uint32_t *e = &table[loc * VP_ATTR_ENTRY_WORDS];
         e[0] = (uint32_t)buf_dev[vin->attr_buf[i]] + vin->attr_offset[i];
         e[1] = vin->attr_stride[i];
         e[2] = vin->attr_divisor[i];
      }
      VP_CHECK(vx_buffer_create(dev, sizeof(table), 0, &tbuf),
               "vx_buffer_create(attrtab)");
      uint64_t tbuf_dev = 0;
      VP_CHECK(vx_buffer_address(tbuf, &tbuf_dev), "vx_buffer_address(attrtab)");
      VP_CHECK(vx_enqueue_write(q, tbuf, 0, table, sizeof(table),
                                0, NULL, NULL), "vx_enqueue_write(attrtab)");
      argblk[1] = tbuf_dev;
   }
   /* slot 2: index buffer base. The standalone VS launch is the non-indexed
    * fallback path, so it is always 0 (the vid is the sequential global id). */
   argblk[2] = 0;

   /* One thread per vertex, sized to fill warps and saturate cores
    * (see geometry-query comment above). */
   vx_launch_info_t li = {
      .struct_size = sizeof(li), .next = NULL,
      .kernel = kbuf,
      .args_host = argblk, .args_size = sizeof(argblk),
      .ndim = 1,
      .grid_dim  = { grid_x,  1, 1 },
      .block_dim = { block_x, 1, 1 },
      .lmem_size = 0,
   };
   VP_CHECK(vx_enqueue_launch(q, &li, 0, NULL, NULL), "vx_enqueue_launch");
   VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");

   /* leave the transformed vertices resident; the caller consumes them
    * on-device (expand_k) or reads them back only on the fallback path. */
   *out_buf  = obuf;
   *out_addr = out_dev;
   ok = true;

done:
   if (tbuf) vx_buffer_release(tbuf);
   if (dtbuf) vx_buffer_release(dtbuf);
   if (!ok && obuf) vx_buffer_release(obuf);   /* on success the caller owns obuf */
   if (kbuf) vx_kernel_release(kbuf);
   if (kmod) vx_module_release(kmod);
   if (q)    vx_queue_release(q);
   unlink(vxpath);
   return ok;
}

/* Copy a resident device buffer back to host memory (one-shot, own queue). */
bool
vp_buffer_readback(vx_device_h dev, vx_buffer_h buf, void *host, uint32_t bytes)
{
   bool ok = false;
   vx_queue_h q = NULL;
   vx_queue_info_t qi = {
      .struct_size = sizeof(qi), .next = NULL,
      .priority = VX_QUEUE_PRIORITY_NORMAL, .flags = 0,
   };
   VP_CHECK(vx_queue_create(dev, &qi, &q), "vx_queue_create");
   VP_CHECK(vx_enqueue_read(q, host, buf, 0, bytes, 0, NULL, NULL),
            "vx_enqueue_read");
   VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");
   ok = true;
done:
   if (q) vx_queue_release(q);
   return ok;
}
