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
 * Vortex RTU vendor intrinsics (vortex_rt_wtrace/wait/get — the ISA v2
 * window ABI), which vp_nir_to_llvm emits as CUSTOM1 .insn ops. The model
 * mirrors Intel's
 * brw_nir_lower_ray_queries.c: rq_initialize stages the ray inputs,
 * rq_proceed fires one synchronous trace+wait, rq_load reads the hit
 * attributes back.
 *
 * Scope: opaque-triangle ray queries (the VK_KHR_ray_query path the
 * raytrace test exercises), plus the candidate-return path for non-opaque
 * geometry: rq_proceed issues the ray on its first call and hands back the
 * lane's verdict on every later one, so the while(proceed) loop runs once per
 * candidate and ends on a terminal status.
 */

#include "vp_nir_to_llvm.h"

#include "nir.h"
#include "nir_builder.h"

/* The RTU window slot numbers and wait status codes are ABI, owned by
 * VX_types.toml and generated into VX_types.h. They must never be restated
 * here: a local copy is a second source of truth that silently decays into
 * reads of the wrong slot when the window is renumbered. */
#include "VX_types.h"

/* Committed RayQueryIntersection types (SPIR-V): None=0, Triangle=1,
 * Generated=2. */
#define RQ_COMMITTED_NONE         0
#define RQ_COMMITTED_TRIANGLE     1

/* Candidate RayQueryIntersection types (SPIR-V): Triangle=0, AABB=1. The
 * candidate and committed enumerations are different, so a query's `committed`
 * flag selects which one an intersection type is reported in. */
#define RQ_CANDIDATE_TRIANGLE     0
#define RQ_CANDIDATE_AABB         1

/* Per-ray-query lowering state. The RTU is one-ray-per-lane, so a single
 * shared state covers the common single-query shader. The v2 window ABI passes
 * the ray geometry through the trace register window (not the slot file), so
 * the ray inputs are staged here at rq_initialize and consumed at rq_proceed. */
struct rq_state {
   nir_variable *scene;   /* uint: TLAS device address (low 32 bits) */
   nir_variable *flags;   /* uint: ray flags                         */
   nir_variable *cull;    /* uint: cull mask (low byte)              */
   nir_variable *origin;  /* vec3: world ray origin                  */
   nir_variable *dir;     /* vec3: world ray direction               */
   nir_variable *tmin;    /* float                                   */
   nir_variable *tmax;    /* float                                   */
   nir_variable *status;  /* uint: vx_rt_wait status                */
   nir_variable *handle;  /* uint: trace handle, needed to resume    */
   nir_variable *started; /* uint: 0 before the first proceed        */
   nir_variable *action;  /* uint: verdict for the candidate in hand */
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

   /* Stage the ray inputs; rq_proceed issues them as one trace macro-op.
    * The TLAS pointer is the low 32 bits of the acceleration-structure
    * address (the RV32 trace config carries one XLEN scene register). */
   nir_def *scene = accel->bit_size == 32 ? accel : nir_u2u32(b, accel);
   nir_store_var(b, st->scene, scene, 0x1);
   nir_store_var(b, st->flags, nir_u2u32(b, flags), 0x1);
   nir_store_var(b, st->cull, nir_iand_imm(b, nir_u2u32(b, cull), 0xff), 0x1);
   nir_store_var(b, st->origin, origin, 0x7);
   nir_store_var(b, st->dir, dir, 0x7);
   nir_store_var(b, st->tmin, tmin, 0x1);
   nir_store_var(b, st->tmax, tmax, 0x1);
   nir_store_var(b, st->status, nir_imm_int(b, VX_RT_STS_DONE_MISS), 0x1);
   nir_store_var(b, st->handle, nir_imm_int(b, 0), 0x1);
   nir_store_var(b, st->started, nir_imm_int(b, 0), 0x1);
   /* An unanswered candidate is a rejected one, so the verdict defaults to
    * IGNORE and only an explicit confirm or terminate moves it. */
   nir_store_var(b, st->action, nir_imm_int(b, VX_RT_CB_IGNORE), 0x1);
}

/* True while this lane must keep proceeding: it has a candidate to service
 * (YIELD_ANYHIT / YIELD_PROC) or is still traversing without one (PENDING).
 * A PENDING lane has to stay in the loop -- exiting on it would leave
 * traversal unfinished and read stale hit data -- and whatever verdict the
 * loop body computes for it is discarded by the RTU, which only applies a
 * verdict to a lane that actually holds a candidate. */
static nir_def *
sts_is_yield(nir_builder *b, nir_def *status)
{
   return nir_ior(b, nir_ieq_imm(b, status, VX_RT_STS_YIELD_ANYHIT),
             nir_ior(b, nir_ieq_imm(b, status, VX_RT_STS_YIELD_PROC),
                        nir_ieq_imm(b, status, VX_RT_STS_PENDING)));
}

/* True only when the lane holds a candidate this iteration. */
static nir_def *
sts_has_candidate(nir_builder *b, nir_def *status)
{
   return nir_ior(b, nir_ieq_imm(b, status, VX_RT_STS_YIELD_ANYHIT),
                     nir_ieq_imm(b, status, VX_RT_STS_YIELD_PROC));
}

static nir_def *
lower_proceed(nir_builder *b, nir_intrinsic_instr *in, struct rq_state *st)
{
   /* The first proceed issues the ray; every later one hands back the verdict
    * for the candidate the previous proceed returned. Both end by blocking on
    * the handle, so one wait serves either path. This is the shape the RTU's
    * candidate-return protocol expects and the shape rayQueryProceedEXT has:
    * the loop runs once per candidate and ends on a terminal status. */
   nir_push_if(b, nir_ieq_imm(b, nir_load_var(b, st->started), 0));
   {
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

      nir_store_var(b, st->handle,
                    nir_vortex_rt_wtrace(b, 32, scene, flags_cull,
                                         origin, dir, tmin, tmax), 0x1);
      nir_store_var(b, st->started, nir_imm_int(b, 1), 0x1);
   }
   nir_push_else(b, NULL);
   {
      /* An any-hit lane computes no distance of its own, so the candidate's
       * own t goes back unchanged; it is read against the status that
       * delivered the candidate. */
      nir_def *prev = nir_load_var(b, st->status);
      nir_vortex_rt_continue(b, nir_load_var(b, st->action),
                                rt_get(b, VX_RT_HIT_T, prev),
                                nir_imm_int(b, 0));
   }
   nir_pop_if(b, NULL);

   nir_def *status = nir_vortex_rt_wait(b, 32, nir_load_var(b, st->handle));
   nir_store_var(b, st->status, status, 0x1);
   /* Reset for the iteration about to run: the verdict must be what this
    * iteration decides, never what the previous one did. */
   nir_store_var(b, st->action, nir_imm_int(b, VX_RT_CB_IGNORE), 0x1);
   return sts_is_yield(b, status);
}

static nir_def *
lower_load(nir_builder *b, nir_intrinsic_instr *in, struct rq_state *st)
{
   nir_ray_query_value value = nir_intrinsic_ray_query_value(in);
   nir_def *status = nir_load_var(b, st->status);
   unsigned ncomp  = in->def.num_components;

   switch (value) {
   case nir_ray_query_value_intersection_type: {
      /* Committed: a hit is a triangle, a miss is None. Candidate: whichever
       * kind the yield delivered, and None when the lane is still traversing
       * with nothing in hand -- a shader must not treat a PENDING iteration as
       * a candidate to shade. */
      bool committed = nir_intrinsic_committed(in);
      if (!committed) {
         return nir_bcsel(b, sts_has_candidate(b, status),
                          nir_bcsel(b,
                             nir_ieq_imm(b, status, VX_RT_STS_YIELD_PROC),
                             nir_imm_int(b, RQ_CANDIDATE_AABB),
                             nir_imm_int(b, RQ_CANDIDATE_TRIANGLE)),
                          nir_imm_int(b, RQ_COMMITTED_NONE));
      }
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
      .handle = nir_local_variable_create(impl, glsl_uint_type(), "rq_handle"),
      .started = nir_local_variable_create(impl, glsl_uint_type(), "rq_started"),
      .action = nir_local_variable_create(impl, glsl_uint_type(), "rq_action"),
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
         case nir_intrinsic_rq_confirm_intersection:
            nir_store_var(&b, st.action, nir_imm_int(&b, VX_RT_CB_ACCEPT), 0x1);
            break;
         case nir_intrinsic_rq_terminate:
            nir_store_var(&b, st.action, nir_imm_int(&b, VX_RT_CB_TERMINATE), 0x1);
            break;
         case nir_intrinsic_rq_generate_intersection:
            /* A ray query has no intersection shader to run, so a procedural
             * candidate has no hit to generate. Leaving it IGNOREd rejects it,
             * which is the only answer available here. */
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
