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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "util/log.h"

/* i64 arg-block slots; slot N == base address of buffer set (N-1).
 * add1 uses slot 1 (descriptor set 0). */
#define VP_ARG_SLOTS 8

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
#define LVP_INST_OTW_OFF      72   /* otw_matrix (object→world, mat3x4)       */

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

struct vp_tri_list {
   float    (*tris)[10];   /* 9 coords + flags(as float bits) per triangle */
   uint32_t  count;
   uint32_t  cap;
   bool      ok;
};

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

static void
vp_tri_push(struct vp_tri_list *tl, const float v0[3], const float v1[3],
            const float v2[3])
{
   if (tl->count == tl->cap) {
      uint32_t ncap = tl->cap ? tl->cap * 2 : 64;
      float (*nt)[10] = realloc(tl->tris, (size_t)ncap * sizeof(*nt));
      if (!nt) { tl->ok = false; return; }
      tl->tris = nt;
      tl->cap = ncap;
   }
   float *t = tl->tris[tl->count++];
   t[0]=v0[0]; t[1]=v0[1]; t[2]=v0[2];
   t[3]=v1[0]; t[4]=v1[1]; t[5]=v1[2];
   t[6]=v2[0]; t[7]=v2[1]; t[8]=v2[2];
   uint32_t flags = RTU_TRI_FLAG_OPAQUE;
   memcpy(&t[9], &flags, 4);
}

/* Walk one node (ptr = offset|type) of the lvp_bvh at base `bvh`, applying
 * transform M, appending world-space triangles to tl. */
static void
vp_walk_node(const uint8_t *bvh, uint32_t node_ptr, const float M[12],
             struct vp_tri_list *tl, int depth)
{
   if (node_ptr == LVP_NODE_INVALID || depth > 64 || !tl->ok)
      return;
   uint32_t type = node_ptr & 7u;
   const uint8_t *node = bvh + (node_ptr & ~7u);
   switch (type) {
   case LVP_NODE_INTERNAL: {
      uint32_t c0, c1;
      memcpy(&c0, node + LVP_BOX_CHILDREN_OFF + 0, 4);
      memcpy(&c1, node + LVP_BOX_CHILDREN_OFF + 4, 4);
      vp_walk_node(bvh, c0, M, tl, depth + 1);
      vp_walk_node(bvh, c1, M, tl, depth + 1);
      break;
   }
   case LVP_NODE_TRIANGLE: {
      float coords[9];
      memcpy(coords, node, 36);           /* coords[3][3] */
      float w0[3], w1[3], w2[3];
      vp_xform_point(M, &coords[0], w0);
      vp_xform_point(M, &coords[3], w1);
      vp_xform_point(M, &coords[6], w2);
      vp_tri_push(tl, w0, w1, w2);
      break;
   }
   case LVP_NODE_INSTANCE: {
      uint64_t blas_host = 0;
      float otw[12];
      memcpy(&blas_host, node, 8);
      memcpy(otw, node + LVP_INST_OTW_OFF, 48);
      if (blas_host) {
         float Mc[12];
         vp_xform_compose(M, otw, Mc);     /* world = M ∘ otw */
         const uint8_t *blas = (const uint8_t *)(uintptr_t)blas_host;
         uint32_t root = LVP_BVH_HEADER_SIZE | LVP_NODE_INTERNAL;
         vp_walk_node(blas, root, Mc, tl, depth + 1);
      }
      break;
   }
   case LVP_NODE_AABB:
   default:
      /* Procedural primitive — not handled on the opaque-triangle path. */
      break;
   }
}

/* ── CW-BVH4 builder (greedy median split, ≤6 children, 1 tri/leaf) ──
 * Mirrors the host builder in tests/raytracing/rt_raycast, widened to 6.
 * `order` is a permutation of triangle indices; a build node is a leaf
 * (one triangle) or an internal node fanning out to up to 6 children. */
struct vp_bnode {
   float    mn[3], mx[3];
   uint32_t tri;                    /* leaf: source triangle index (prim id) */
   int      child[RTU_BVH4_WIDTH];  /* internal: build-node indices */
   int      nchild;
   bool     leaf;
};

struct vp_bvh {
   const float (*tris)[10];         /* tl->tris: 9 coords + flags per tri */
   float    (*cmin)[3];             /* per-tri AABB min */
   float    (*cmax)[3];             /* per-tri AABB max */
   float    (*cen)[3];              /* per-tri centroid */
   uint32_t *order;                 /* tri index permutation */
   struct vp_bnode *nodes;          /* pre-sized: no realloc during build */
   uint32_t  n_nodes;
};

/* Sort order[start, start+count) by centroid on the longest axis; return
 * the median split offset. Insertion sort — scenes the RTU handles are
 * small, and it avoids qsort's non-portable context passing. */
static uint32_t
vp_bvh_split(struct vp_bvh *b, uint32_t start, uint32_t count)
{
   float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
   for (uint32_t i = 0; i < count; i++) {
      const float *c = b->cen[b->order[start + i]];
      for (int a = 0; a < 3; a++) {
         if (c[a] < mn[a]) mn[a] = c[a];
         if (c[a] > mx[a]) mx[a] = c[a];
      }
   }
   int axis = 0;
   float ext = mx[0] - mn[0];
   if (mx[1] - mn[1] > ext) { axis = 1; ext = mx[1] - mn[1]; }
   if (mx[2] - mn[2] > ext) { axis = 2; }
   for (uint32_t i = start + 1; i < start + count; i++) {
      uint32_t v = b->order[i];
      float k = b->cen[v][axis];
      uint32_t j = i;
      while (j > start && b->cen[b->order[j - 1]][axis] > k) {
         b->order[j] = b->order[j - 1];
         j--;
      }
      b->order[j] = v;
   }
   return count / 2;
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
   /* Partition into up to 6 child ranges: repeatedly median-split the
    * largest range that still holds more than one triangle. */
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
      if (e >  16) e =  16;
      exp[a] = e;
   }
}

static uint8_t
vp_quant(float v, float origin, int exp, bool hi)
{
   float t = (v - origin) / ldexpf(1.0f, exp);
   float q = hi ? ceilf(t) : floorf(t);
   if (q < 0.f)   q = 0.f;
   if (q > 255.f) q = 255.f;
   return (uint8_t)q;
}

/* Serialize the build tree into a CW-BVH4 scene buffer (malloc'd; caller
 * frees). Returns NULL on OOM. *out_size receives the byte size. */
static uint8_t *
vp_bvh_serialize(struct vp_bvh *b, int root, const float (*tris)[10],
                 uint32_t *out_size)
{
   const uint32_t leaf_bytes = RTU_BVH_LEAF_HDR_BYTES + RTU_TRI_STRIDE;
   uint32_t *off = malloc((size_t)b->n_nodes * sizeof(uint32_t));
   if (!off) return NULL;
   uint32_t cur = RTU_SCENE_HDR_BYTES;
   for (uint32_t i = 0; i < b->n_nodes; i++) {
      off[i] = cur;
      cur += b->nodes[i].leaf ? leaf_bytes : RTU_BVH4_NODE_BYTES;
   }
   uint8_t *buf = calloc(1, cur);
   if (!buf) { free(off); return NULL; }

   uint32_t *sh = (uint32_t *)buf;
   sh[0] = off[root];                 /* root node offset */
   sh[1] = RTU_SCENE_KIND_BVH4;
   sh[2] = cur;                       /* total scene bytes (prefetch sizing) */
   sh[3] = b->n_nodes;                /* node count */

   for (uint32_t i = 0; i < b->n_nodes; i++) {
      const struct vp_bnode *n = &b->nodes[i];
      uint8_t *p = buf + off[i];
      if (n->leaf) {
         uint32_t *lh = (uint32_t *)p;
         lh[0] = RTU_BVH_KIND_LEAF_TRI | (1u << RTU_BVH_COUNT_SHIFT);
         lh[1] = 0;                    /* geometry_index (single geometry) */
         lh[2] = 0;
         lh[3] = n->tri;               /* prim_base = source gl_PrimitiveID */
         const float *t = tris[n->tri];
         memcpy(p + RTU_BVH_LEAF_HDR_BYTES, t, 36);   /* 9 coords */
         uint32_t flags = RTU_TRI_FLAG_OPAQUE;
         memcpy(p + RTU_BVH_LEAF_HDR_BYTES + RTU_TRI_FLAGS_OFFSET, &flags, 4);
      } else {
         uint32_t *kind = (uint32_t *)p;
         *kind = RTU_BVH_KIND_INTERNAL |
                 ((uint32_t)n->nchild << RTU_BVH_COUNT_SHIFT);
         float *origin = (float *)(p + RTU_BVH4_OFF_ORIGIN);
         origin[0] = n->mn[0]; origin[1] = n->mn[1]; origin[2] = n->mn[2];
         int exp[3];
         vp_bvh_choose_exp(n->mn, n->mx, exp);
         int8_t *pe = (int8_t *)(p + RTU_BVH4_OFF_EXP);
         pe[0] = (int8_t)exp[0]; pe[1] = (int8_t)exp[1]; pe[2] = (int8_t)exp[2];
         uint32_t *child = (uint32_t *)(p + RTU_BVH4_OFF_CHILD);
         uint8_t  *qmin  = p + RTU_BVH4_OFF_QMIN;
         uint8_t  *qmax  = p + RTU_BVH4_OFF_QMAX;
         for (int k = 0; k < n->nchild; k++) {
            const struct vp_bnode *c = &b->nodes[n->child[k]];
            uint32_t coff = off[n->child[k]];
            child[k] = coff | (c->leaf ? RTU_BVH_CHILD_LEAF_FLAG : 0u);
            for (int a = 0; a < 3; a++) {
               qmin[k * 3 + a] = vp_quant(c->mn[a], n->mn[a], exp[a], false);
               qmax[k * 3 + a] = vp_quant(c->mx[a], n->mn[a], exp[a], true);
            }
         }
      }
   }
   free(off);
   *out_size = cur;
   return buf;
}

/* Build a CW-BVH4 scene buffer from the collected world-space triangles.
 * Empty input degenerates to an empty TriList (all-miss). */
static uint8_t *
vp_build_bvh4_scene(const struct vp_tri_list *tl, uint32_t *out_size)
{
   if (tl->count == 0) {
      uint8_t *scene = calloc(1, RTU_SCENE_HDR_BYTES);
      if (!scene) return NULL;
      *out_size = RTU_SCENE_HDR_BYTES;   /* {count=0, kind=TriList, 0, 0} */
      return scene;
   }

   const uint32_t n = tl->count;
   struct vp_bvh b = { .tris = tl->tris };
   b.cmin  = malloc((size_t)n * sizeof(*b.cmin));
   b.cmax  = malloc((size_t)n * sizeof(*b.cmax));
   b.cen   = malloc((size_t)n * sizeof(*b.cen));
   b.order = malloc((size_t)n * sizeof(*b.order));
   b.nodes = malloc((size_t)(2 * n) * sizeof(*b.nodes));   /* ≤ 2N-1 nodes */
   if (!b.cmin || !b.cmax || !b.cen || !b.order || !b.nodes) {
      free(b.cmin); free(b.cmax); free(b.cen); free(b.order); free(b.nodes);
      return NULL;
   }
   for (uint32_t i = 0; i < n; i++) {
      const float *t = tl->tris[i];
      for (int a = 0; a < 3; a++) {
         float v0 = t[a], v1 = t[3 + a], v2 = t[6 + a];
         float lo = v0 < v1 ? v0 : v1; lo = lo < v2 ? lo : v2;
         float hi = v0 > v1 ? v0 : v1; hi = hi > v2 ? hi : v2;
         b.cmin[i][a] = lo;
         b.cmax[i][a] = hi;
         b.cen[i][a]  = 0.5f * (lo + hi);
      }
      b.order[i] = i;
   }
   int root = vp_bvh_build(&b, 0, n);
   uint8_t *scene = vp_bvh_serialize(&b, root, tl->tris, out_size);

   free(b.cmin); free(b.cmax); free(b.cen); free(b.order); free(b.nodes);
   return scene;
}

/* Transcode the lavapipe AS at `tlas_host` into an RTU CW-BVH4 scene in
 * device memory; returns its device address (0 on failure). */
static uint64_t
vp_transcode_as(struct vp_as_ctx *c, const void *tlas_host)
{
   struct vp_tri_list tl = { .ok = true };
   static const float kIdentity[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };
   uint32_t root = LVP_BVH_HEADER_SIZE | LVP_NODE_INTERNAL;
   vp_walk_node((const uint8_t *)tlas_host, root, kIdentity, &tl, 0);
   if (!tl.ok) {
      free(tl.tris);
      c->ok = false;
      return 0;
   }

   uint32_t size = 0;
   uint8_t *scene = vp_build_bvh4_scene(&tl, &size);
   free(tl.tris);
   if (!scene) { c->ok = false; return 0; }

   if (c->n_bufs >= VP_MAX_BVH || c->n_stages >= VP_MAX_BVH) {
      free(scene); c->ok = false; return 0;
   }
   /* The upload is asynchronous; keep `scene` alive until the queue
    * finishes (freed by the launch cleanup, like vp_copy_as's stages). */
   c->stages[c->n_stages++] = scene;
   vx_buffer_h b = NULL;
   uint64_t dev_addr = 0;
   if (vx_buffer_create(c->dev, size, VX_MEM_READ, &b) != VX_SUCCESS) {
      c->ok = false; return 0;
   }
   c->bufs[c->n_bufs++] = b;
   if (vx_buffer_address(b, &dev_addr) != VX_SUCCESS ||
       vx_enqueue_write(c->q, b, 0, scene, size, 0, NULL, NULL) != VX_SUCCESS) {
      c->ok = false; return 0;
   }
   return dev_addr;
}

bool
vp_launch(vx_device_h dev,
          const void *vxbin, size_t vxbin_size,
          const void *desc_host, uint32_t desc_bytes,
          const struct vp_desc *descs, uint32_t num_descs,
          const uint32_t grid[3], const uint32_t block[3],
          uint32_t lmem_size, bool has_rtu)
{
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_module_h kmod = NULL;
   vx_kernel_h kbuf = NULL;
   vx_buffer_h dbuf = NULL;
   vx_buffer_h res[VP_MAX_DESCS]      = { 0 };  /* per-descriptor device buffer */
   void       *res_host[VP_MAX_DESCS] = { 0 };  /* its host backing            */
   uint32_t    res_bytes[VP_MAX_DESCS]= { 0 };
   struct vp_as_ctx asc = { .ok = true };       /* acceleration-structure BVHs */
   uint8_t    *stage = NULL;
   char vxpath[] = "/tmp/vortexpipe-k.XXXXXX";
   int  vxfd = -1;

   /* materialize the .vxbin in a temp file for vx_module_load_file */
   vxfd = mkstemp(vxpath);
   if (vxfd < 0) {
      mesa_loge("vortexpipe: launch: mkstemp failed");
      return false;
   }
   if (write(vxfd, vxbin, vxbin_size) != (ssize_t)vxbin_size) {
      mesa_loge("vortexpipe: launch: writing .vxbin failed");
      close(vxfd);
      unlink(vxpath);
      return false;
   }
   close(vxfd);

   /* A private copy of the descriptor buffer: vp_launch rewrites the
    * resource pointers inside it to device addresses. It must outlive
    * vx_queue_finish (vx_enqueue_write reads it asynchronously). */
   stage = malloc(desc_bytes);
   if (!stage) {
      mesa_loge("vortexpipe: launch: descriptor staging OOM");
      unlink(vxpath);
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

   VP_CHECK(vx_module_load_file(dev, vxpath, &kmod),
            "vx_module_load_file");
   /* "main" is the public name vxbin.py assigns the single conventional
    * kernel (the C entry is "kernel_main"); match the native runtime. */
   VP_CHECK(vx_module_get_kernel(kmod, "main", &kbuf),
            "vx_module_get_kernel");

   /* Relocate each descriptor into the staged descriptor blob:
    *  - VP_DESC_BUFFER: copy the resource into device memory, rewrite
    *    lp_jit_buffer.ptr so load_ssbo/store_ssbo dereference on-device.
    *  - VP_DESC_AS: copy the BVH (TLAS + its BLASes) into device
    *    memory with instance-node links relocated, rewrite the
    *    accel_struct device address. */
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

      /* VP_DESC_BUFFER */
      uint64_t host_ptr = 0;
      uint32_t size = 0;
      memcpy(&host_ptr, slot + VP_JIT_BUF_PTR,  sizeof host_ptr);
      memcpy(&size,     slot + VP_JIT_BUF_SIZE, sizeof size);
      if (!host_ptr || !size)
         continue;
      VP_CHECK(vx_buffer_create(dev, size, 0, &res[i]),
               "vx_buffer_create(resource)");
      uint64_t dev_addr = 0;
      VP_CHECK(vx_buffer_address(res[i], &dev_addr), "vx_buffer_address");
      VP_CHECK(vx_enqueue_write(q, res[i], 0, (void *)(uintptr_t)host_ptr,
                                size, 0, NULL, NULL),
               "vx_enqueue_write(resource)");
      memcpy(slot + VP_JIT_BUF_PTR, &dev_addr, sizeof dev_addr);
      res_host[i]  = (void *)(uintptr_t)host_ptr;
      res_bytes[i] = size;
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

   /* copy every buffer back into its host backing */
   for (uint32_t i = 0; i < num_descs; i++) {
      if (!res[i])
         continue;
      VP_CHECK(vx_enqueue_read(q, res_host[i], res[i], 0, res_bytes[i],
                               0, NULL, NULL), "vx_enqueue_read(resource)");
   }
   VP_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE), "vx_queue_finish");

   ok = true;

done:
   for (uint32_t i = 0; i < VP_MAX_DESCS; i++)
      if (res[i]) vx_buffer_release(res[i]);
   for (unsigned i = 0; i < asc.n_bufs; i++)
      if (asc.bufs[i]) vx_buffer_release(asc.bufs[i]);
   for (unsigned i = 0; i < asc.n_stages; i++)
      free(asc.stages[i]);
   if (dbuf) vx_buffer_release(dbuf);
   if (kbuf) vx_kernel_release(kbuf);
   if (kmod) vx_module_release(kmod);
   if (q)    vx_queue_release(q);
   free(stage);
   unlink(vxpath);
   return ok;
}

/* attribute-table entry the VS kernel reads: { device base, stride }.
 * The table is indexed by VS input driver_location. */
#define VP_ATTR_ENTRY_BYTES 8
#define VP_ATTR_TABLE_LOCS  8

bool
vp_launch_vs(vx_device_h dev,
             const void *vxbin, size_t vxbin_size,
             uint32_t vertex_count, uint32_t out_bytes,
             const struct vp_vertex_input *vin,
             vx_buffer_h *out_buf, uint64_t *out_addr)
{
   bool ok = false;
   vx_queue_h  q    = NULL;
   vx_module_h kmod = NULL;
   vx_kernel_h kbuf = NULL;
   vx_buffer_h obuf = NULL, vbuf = NULL, tbuf = NULL;
   char vxpath[] = "/tmp/vortexpipe-vs.XXXXXX";
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
    *            slot 1 -> vertex attribute table (0 if self-contained) */
   uint64_t argblk[VP_ARG_SLOTS] = { 0 };
   argblk[0] = out_dev;

   /* Vertex buffer + attribute table: upload the interleaved vertex
    * buffer, then a table indexed by driver_location holding the
    * device base address + stride of each attribute. The VS kernel
    * fetches input `loc` of vertex `vid` at table[loc].base +
    * vid*table[loc].stride.
    *
    * `table` is declared at function scope: vx_enqueue_write is
    * asynchronous (the source is read at vx_queue_finish), so it must
    * outlive the `if` block -- the same lifetime rule as argblk. */
   uint32_t table[VP_ATTR_TABLE_LOCS * 2] = { 0 };
   if (vin && vin->num_attrs) {
      VP_CHECK(vx_buffer_create(dev, vin->size, 0, &vbuf),
               "vx_buffer_create(vbuf)");
      uint64_t vbuf_dev = 0;
      VP_CHECK(vx_buffer_address(vbuf, &vbuf_dev), "vx_buffer_address(vbuf)");
      VP_CHECK(vx_enqueue_write(q, vbuf, 0, vin->data, vin->size,
                                0, NULL, NULL), "vx_enqueue_write(vbuf)");

      for (uint32_t i = 0; i < vin->num_attrs; i++) {
         uint32_t loc = vin->attr_loc[i];
         if (loc >= VP_ATTR_TABLE_LOCS)
            continue;
         table[loc * 2 + 0] = (uint32_t)vbuf_dev + vin->base_offset
                            + vin->attr_offset[i];
         table[loc * 2 + 1] = vin->attr_stride[i];
      }
      VP_CHECK(vx_buffer_create(dev, sizeof(table), 0, &tbuf),
               "vx_buffer_create(attrtab)");
      uint64_t tbuf_dev = 0;
      VP_CHECK(vx_buffer_address(tbuf, &tbuf_dev), "vx_buffer_address(attrtab)");
      VP_CHECK(vx_enqueue_write(q, tbuf, 0, table, sizeof(table),
                                0, NULL, NULL), "vx_enqueue_write(attrtab)");
      argblk[1] = tbuf_dev;
   }

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
   if (vbuf) vx_buffer_release(vbuf);
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
