/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_nir_lower_ray_tracing_to_rtu -- lower Vulkan ray queries to the
 * Vortex RTU.
 *
 * lavapipe normally expands rq_* / trace_ray into a software BVH walk
 * (lvp_nir_lower_ray_queries). When the device has the RTU
 * (driver_ray_queries cap set by vortexpipe), lavapipe skips that and
 * leaves the rq_* intrinsics intact; this pass rewrites them into the
 * Vortex RTU vendor intrinsics (vortex_rt_trace2/wait2/get — the ISA v2
 * window ABI), which vp_nir_to_llvm emits as CUSTOM1 .insn ops. The model
 * mirrors Intel's
 * brw_nir_lower_ray_queries.c: rq_initialize stages the ray inputs,
 * rq_proceed fires one synchronous trace+wait, rq_load reads the hit
 * attributes back.
 *
 * Scope: opaque-triangle ray queries (the VK_KHR_ray_query path the
 * raytrace test exercises). The RTU resolves the whole ray in one
 * trace+wait, so rq_proceed returns false (the while(proceed) loop runs
 * its body zero times) and the candidate/any-hit path is a follow-up.
 */

#include "vp_nir_to_llvm.h"

#include "nir.h"
#include "nir_builder.h"

/* RTU register-file slots (VX_types.toml [rtu_slots]) and wait status
 * (rtu_status). Kept local so this file doesn't depend on the Vortex
 * VX_types.h header. */
#define VX_RT_RAY_ORIGIN          0   /* slots 0..2  */
#define VX_RT_RAY_DIRECTION       3   /* slots 3..5  */
#define VX_RT_T_MIN               6
#define VX_RT_T_MAX               7
#define VX_RT_OBJECT_RAY_ORIGIN   8   /* slots 8..10 */
#define VX_RT_OBJECT_RAY_DIRECTION 11 /* slots 11..13 */
#define VX_RT_HIT_T               14
#define VX_RT_HIT_BARY_U          15
#define VX_RT_HIT_BARY_V          16
#define VX_RT_HIT_PRIMITIVE_ID    21
#define VX_RT_HIT_INSTANCE_ID     22
#define VX_RT_HIT_GEOMETRY_INDEX  23
#define VX_RT_HIT_INSTANCE_CUSTOM 24
#define VX_RT_RAY_FLAGS           27
#define VX_RT_CULL_MASK           28

#define VX_RT_STS_DONE_HIT        0
#define VX_RT_STS_DONE_MISS       1

/* Committed RayQueryIntersection types (SPIR-V): None=0, Triangle=1,
 * Generated=2. */
#define RQ_COMMITTED_NONE         0
#define RQ_COMMITTED_TRIANGLE     1

/* Per-ray-query lowering state. The RTU is one-ray-per-lane, so a single
 * shared state covers the common single-query shader. The v2 window ABI passes
 * the ray geometry through the trace2 register window (not the slot file), so
 * the ray inputs are staged here at rq_initialize and consumed at rq_proceed. */
struct rq_state {
   nir_variable *scene;   /* uint: TLAS device address (low 32 bits) */
   nir_variable *flags;   /* uint: ray flags                         */
   nir_variable *cull;    /* uint: cull mask (low byte)              */
   nir_variable *origin;  /* vec3: world ray origin                  */
   nir_variable *dir;     /* vec3: world ray direction               */
   nir_variable *tmin;    /* float                                   */
   nir_variable *tmax;    /* float                                   */
   nir_variable *status;  /* uint: vx_rt_wait2 status                */
};

static nir_def *
rt_get(nir_builder *b, unsigned slot, nir_def *status)
{
   return nir_vortex_rt_get(b, 32, status, .base = slot);
}

static void
lower_initialize(nir_builder *b, nir_intrinsic_instr *in, struct rq_state *st)
{
   /* src = { query, accel, flags, cull_mask, origin(vec3), tmin,
    *         direction(vec3), tmax } */
   nir_def *accel  = in->src[1].ssa;
   nir_def *flags  = in->src[2].ssa;
   nir_def *cull   = in->src[3].ssa;
   nir_def *origin = in->src[4].ssa;
   nir_def *tmin   = in->src[5].ssa;
   nir_def *dir    = in->src[6].ssa;
   nir_def *tmax   = in->src[7].ssa;

   /* Stage the ray inputs; rq_proceed issues them as one trace2 macro-op.
    * The TLAS pointer is the low 32 bits of the acceleration-structure
    * address (the RV32 trace2 config carries one XLEN scene register). */
   nir_def *scene = accel->bit_size == 32 ? accel : nir_u2u32(b, accel);
   nir_store_var(b, st->scene, scene, 0x1);
   nir_store_var(b, st->flags, nir_u2u32(b, flags), 0x1);
   nir_store_var(b, st->cull, nir_iand_imm(b, nir_u2u32(b, cull), 0xff), 0x1);
   nir_store_var(b, st->origin, origin, 0x7);
   nir_store_var(b, st->dir, dir, 0x7);
   nir_store_var(b, st->tmin, tmin, 0x1);
   nir_store_var(b, st->tmax, tmax, 0x1);
   nir_store_var(b, st->status, nir_imm_int(b, VX_RT_STS_DONE_MISS), 0x1);
}

static nir_def *
lower_proceed(nir_builder *b, nir_intrinsic_instr *in, struct rq_state *st)
{
   /* Fire one synchronous ray: the RTU walks the whole BVH in trace2+wait2.
    * `while (rayQueryProceedEXT(rq)) {}` evaluates proceed once, so a single
    * trace happens and the loop body (candidate handling) is skipped for the
    * opaque path. Return false. */
   nir_def *scene  = nir_load_var(b, st->scene);
   nir_def *flags  = nir_load_var(b, st->flags);
   nir_def *cull   = nir_load_var(b, st->cull);
   /* lane-3 config word: ray_flags (low 16) | cull_mask (high 16). */
   nir_def *flags_cull =
      nir_ior(b, nir_iand_imm(b, flags, 0xffff),
                 nir_ishl_imm(b, nir_iand_imm(b, cull, 0xff), 16));
   nir_def *origin = nir_load_var(b, st->origin);
   nir_def *dir    = nir_load_var(b, st->dir);
   nir_def *tmin   = nir_load_var(b, st->tmin);
   nir_def *tmax   = nir_load_var(b, st->tmax);

   nir_def *handle = nir_vortex_rt_trace2(b, 32, scene, flags_cull,
                                          origin, dir, tmin, tmax);
   nir_def *status = nir_vortex_rt_wait2(b, 32, handle);
   nir_store_var(b, st->status, status, 0x1);
   return nir_imm_false(b);
}

static nir_def *
lower_load(nir_builder *b, nir_intrinsic_instr *in, struct rq_state *st)
{
   nir_ray_query_value value = nir_intrinsic_ray_query_value(in);
   nir_def *status = nir_load_var(b, st->status);
   unsigned ncomp  = in->def.num_components;

   switch (value) {
   case nir_ray_query_value_intersection_type: {
      /* committed: a hit is a triangle (opaque), a miss is None.
       * candidate: opaque-only, never a pending candidate -> None. */
      bool committed = nir_intrinsic_committed(in);
      if (!committed)
         return nir_imm_int(b, RQ_COMMITTED_NONE);
      return nir_bcsel(b, nir_ieq_imm(b, status, VX_RT_STS_DONE_HIT),
                       nir_imm_int(b, RQ_COMMITTED_TRIANGLE),
                       nir_imm_int(b, RQ_COMMITTED_NONE));
   }
   case nir_ray_query_value_intersection_t:
      return rt_get(b, VX_RT_HIT_T, status);
   case nir_ray_query_value_intersection_barycentrics:
      return nir_vec2(b, rt_get(b, VX_RT_HIT_BARY_U, status),
                         rt_get(b, VX_RT_HIT_BARY_V, status));
   case nir_ray_query_value_intersection_primitive_index:
      return rt_get(b, VX_RT_HIT_PRIMITIVE_ID, status);
   case nir_ray_query_value_intersection_geometry_index:
      return rt_get(b, VX_RT_HIT_GEOMETRY_INDEX, status);
   case nir_ray_query_value_intersection_instance_id:
      return rt_get(b, VX_RT_HIT_INSTANCE_ID, status);
   case nir_ray_query_value_intersection_instance_custom_index:
      return rt_get(b, VX_RT_HIT_INSTANCE_CUSTOM, status);
   case nir_ray_query_value_intersection_object_ray_origin:
      /* Opaque single-level path: object ray == world ray (the RTU only
       * stages a distinct object ray on a callback yield). Return the
       * staged world ray rather than the unwritten object-ray slots. */
      return nir_load_var(b, st->origin);
   case nir_ray_query_value_intersection_object_ray_direction:
      return nir_load_var(b, st->dir);
   default:
      /* Unhandled query value (front-face, transforms, world ray, …):
       * return zeros of the requested shape for now. */
      return nir_imm_zero(b, ncomp, in->def.bit_size);
   }
}

bool
vp_nir_lower_ray_tracing_to_rtu(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   /* Quick scan: any ray-query intrinsics at all? */
   bool has_rq = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_rq_initialize:
         case nir_intrinsic_rq_proceed:
         case nir_intrinsic_rq_load:
         case nir_intrinsic_rq_terminate:
         case nir_intrinsic_rq_generate_intersection:
         case nir_intrinsic_rq_confirm_intersection:
            has_rq = true;
            break;
         default:
            break;
         }
         if (has_rq)
            break;
      }
      if (has_rq)
         break;
   }
   if (!has_rq)
      return false;

   nir_builder b = nir_builder_create(impl);

   /* One shared query state (RTU is one ray per lane). */
   struct rq_state st = {
      .scene  = nir_local_variable_create(impl, glsl_uint_type(), "rq_scene"),
      .flags  = nir_local_variable_create(impl, glsl_uint_type(), "rq_flags"),
      .cull   = nir_local_variable_create(impl, glsl_uint_type(), "rq_cull"),
      .origin = nir_local_variable_create(impl, glsl_vec_type(3), "rq_origin"),
      .dir    = nir_local_variable_create(impl, glsl_vec_type(3), "rq_dir"),
      .tmin   = nir_local_variable_create(impl, glsl_float_type(), "rq_tmin"),
      .tmax   = nir_local_variable_create(impl, glsl_float_type(), "rq_tmax"),
      .status = nir_local_variable_create(impl, glsl_uint_type(), "rq_status"),
   };

   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);

         switch (in->intrinsic) {
         case nir_intrinsic_rq_initialize:
            lower_initialize(&b, in, &st);
            break;
         case nir_intrinsic_rq_proceed:
            nir_def_rewrite_uses(&in->def, lower_proceed(&b, in, &st));
            break;
         case nir_intrinsic_rq_load:
            nir_def_rewrite_uses(&in->def, lower_load(&b, in, &st));
            break;
         case nir_intrinsic_rq_terminate:
         case nir_intrinsic_rq_generate_intersection:
         case nir_intrinsic_rq_confirm_intersection:
            /* Opaque path: no candidate/any-hit handling yet. Drop. */
            break;
         default:
            continue;
         }
         nir_instr_remove(instr);
      }
   }

   nir_progress(true, impl, nir_metadata_none);

   /* The per-query tlas/status are function_temp variables; lift them to
    * SSA so the minimal vp_nir_to_llvm translator (which doesn't handle
    * local-variable load/store) sees clean SSA values — same cleanup the
    * lavapipe SW lowering does. */
   nir_lower_global_vars_to_local(shader);
   nir_lower_vars_to_ssa(shader);
   nir_opt_dce(shader);
   return true;
}
