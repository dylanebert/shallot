//! Pose finalize: the per-body pose-advance phase of box3d's soft-step solver, ported op-for-op from
//! `solver.c` (b3FinalizeBodiesTask). It advances the center + rotation from the solved deltas,
//! rebuilds the world inertia tensor and body-origin transform, resets the per-step delta/force
//! accumulators, writes the body's sleep velocity and decides whether it needs a continuous sweep.
//! The arena follows it with shape bounds commit; the kernel enlarges proxies serially.
//!
//! Every arithmetic op maps one-to-one to the C scalar path (no SIMD, no FMA); bit-identical to the
//! the frozen historical oracle vectors; current target evidence belongs to the standalone oracle.

#[cfg(target_arch = "wasm32")]
use crate::body::{
    clear_sim_force_torque, flags::DYNAMIC, read_fin, read_sim, read_state, write_fin_center,
    write_fin_transform_p, write_sim_inv_inertia_world, write_sim_rotation, S2_BODY_ID, S2_CENTER0,
    S2_FLAGS, S2_MIN_EXTENT, S2_ROTATION0, SIM2_STRIDE,
};
#[cfg(target_arch = "wasm32")]
use crate::col::Col;
use crate::math::{maxf, minf, Mat3, Transform, Vec3};

/// Position-correction weight for the sleep-velocity blend (b3FinalizeBodies `positionSleepFactor`).
const POSITION_SLEEP_FACTOR: f32 = 0.5;

/// The continuous-collision safety factor (b3FinalizeBodies `safetyFactor`): a body whose step motion
/// exceeds `SAFETY_FACTOR * minExtent` is a fast-body candidate.
const SAFETY_FACTOR: f32 = 0.5;

// --- shape-AABB refit -----------------------------------------------------------------------
// Pure bounds arithmetic stays independent of the wasm regions for bit-pinned native tests.
// The arena commits the resulting bounds and enlarge flags to the owning shape columns.

/// Shape type codes (`ShapeType`).
pub const TY_CAPSULE: u32 = 0;
pub const TY_COMPOUND: u32 = 1;
pub const TY_HULL: u32 = 3;
pub const TY_SPHERE: u32 = 5;

/// The speculative margin the fat AABB inflates a shape by (`B3_SPECULATIVE_DISTANCE`, `src/standard/physics/common/constants.ts`).
/// `4.0 * 0.005` const-evaluates to the same f32 as the TS `f32(4.0 * f32(0.005))`.
const SPECULATIVE_DISTANCE: f32 = 4.0 * 0.005;

/// Does this shape type use inline convex geometry rather than the non-convex geometry pools?
#[inline]
pub fn is_convex_refit(shape_type: u32) -> bool {
    matches!(shape_type, TY_SPHERE | TY_CAPSULE | TY_HULL)
}

/// b3AABB_Contains: does `a` fully enclose `b`? Mirrors `src/math.ts` `aabb.contains` — `a.lower ≤
/// b.lower` and `b.upper ≤ a.upper` on every axis. Each is `[lower.xyz, upper.xyz]`.
#[inline]
pub fn aabb_contains(a: &[f32; 6], b: &[f32; 6]) -> bool {
    !(a[0] > b[0] || b[3] > a[3] || a[1] > b[1] || b[4] > a[4] || a[2] > b[2] || b[5] > a[5])
}

/// Tight world AABB of a sphere under `xf` (b3ComputeSphereAABB / `computeSphereAABBOut`). `geom` is
/// the shape column payload: center(3) radius(1).
#[inline]
fn sphere_aabb(geom: &[f32], xf: Transform) -> (Vec3, Vec3) {
    let c = xf.q.rotate(Vec3::new(geom[0], geom[1], geom[2])).add(xf.p);
    let r = geom[3];
    (
        Vec3::new(c.x - r, c.y - r, c.z - r),
        Vec3::new(c.x + r, c.y + r, c.z + r),
    )
}

/// Tight world AABB of a capsule under `xf` (b3ComputeCapsuleAABB / `computeCapsuleAABBOut`). `geom` is
/// center1(3) center2(3) radius(1); the min/max are the `b3Min`/`b3Max` ternaries (`minf`/`maxf`).
#[inline]
fn capsule_aabb(geom: &[f32], xf: Transform) -> (Vec3, Vec3) {
    let c1 = xf.q.rotate(Vec3::new(geom[0], geom[1], geom[2])).add(xf.p);
    let c2 = xf.q.rotate(Vec3::new(geom[3], geom[4], geom[5])).add(xf.p);
    let r = geom[6];
    (
        Vec3::new(
            minf(c1.x, c2.x) - r,
            minf(c1.y, c2.y) - r,
            minf(c1.z, c2.z) - r,
        ),
        Vec3::new(
            maxf(c1.x, c2.x) + r,
            maxf(c1.y, c2.y) + r,
            maxf(c1.z, c2.z) + r,
        ),
    )
}

/// Tight world AABB of a hull under `xf` (b3AABB_Transform of the hull's local AABB / `transformOut`).
/// `geom` is the local AABB: lower(3) upper(3) — the only hull field the AABB path reads.
#[inline]
fn hull_aabb(geom: &[f32], xf: Transform) -> (Vec3, Vec3) {
    let lo = Vec3::new(geom[0], geom[1], geom[2]);
    let hi = Vec3::new(geom[3], geom[4], geom[5]);
    let center = xf.q.rotate(hi.add(lo).scale(0.5)).add(xf.p);
    let extent = Mat3::from_quat(xf.q).abs().mul_v(hi.sub(lo).scale(0.5));
    (center.sub(extent), center.add(extent))
}

/// Inflate a tight shape bound by the speculative distance and test its resident fat margin.
pub fn refit_bounds(b: [f32; 6], fat: &[f32; 6]) -> ([f32; 6], bool) {
    let lo = Vec3::new(b[0], b[1], b[2]);
    let hi = Vec3::new(b[3], b[4], b[5]);
    let s = SPECULATIVE_DISTANCE;
    let cand = [lo.x - s, lo.y - s, lo.z - s, hi.x + s, hi.y + s, hi.z + s];
    let escaped = !aabb_contains(fat, &cand);
    (cand, escaped)
}

pub fn convex_bounds(shape_type: u32, geom: &[f32], xf: Transform) -> [f32; 6] {
    let (lo, hi) = match shape_type {
        TY_SPHERE => sphere_aabb(geom, xf),
        TY_CAPSULE => capsule_aabb(geom, xf),
        _ => hull_aabb(geom, xf),
    };
    [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z]
}

/// Advance the bodies in `[start, start+count)` from their solved velocity/position deltas.
///
/// Reads state (velocities + deltas), sim (transform.q + invInertiaLocal), the finalize column
/// (center, localCenter, maxExtent), and the sim2 minExtent + flags. Writes the advanced
/// center/transform.p (finalize column), transform.q + invInertiaWorld (sim), the cleared force/torque
/// (sim), the reset deltas (state), sleepVelocity on the owning body, and the sweep base for
/// non-fast bodies. Fast non-bullets sweep immediately; bullets retain their base for the deferred
/// sweep. `h` is the full-step dt, `inv_dt` its inverse.
///
/// # Safety
/// Resident body records and continuous scratch must be reserved. Only this worker may write
/// bodies in the requested range, and no worker may grow memory while it runs.
#[cfg(target_arch = "wasm32")]
pub unsafe fn finalize(
    world_index: usize,
    worker: usize,
    state_col: Col<f32>,
    sim_col: Col<f32>,
    fin_col: Col<f32>,
    sim2_col: Col<f32>,
    flags_col: Col<u32>,
    start: usize,
    count: usize,
    h: f32,
    inv_dt: f32,
    enable_continuous: bool,
) {
    for i in start..start + count {
        let s = read_state(state_col, i);
        let sim = read_sim(sim_col, i);
        let fin = read_fin(fin_col, i);

        let v = s.linear_velocity;
        let w = s.angular_velocity;
        let q0 = sim.rotation;

        // Velocity of the farthest point accounts for rotation; both arcs are measured in the
        // pre-advance frame, so they read the old rotation.
        let local_omega = q0.inv_rotate(w);
        let local_delta_rotation = q0.inv_rotate(s.delta_rotation.v);

        let center = fin.center.add(s.delta_position); // b3OffsetPos
        let q = s.delta_rotation.mul(q0).normalize();

        let velocity_arc = local_omega.abs().modified_cross(fin.max_extent);
        let max_velocity = v.length() + velocity_arc.length();

        // For small angles |theta| ~= 2 * length(sin(theta/2) * v), hence the 2x on the rotation arc.
        let rotation_arc = local_delta_rotation.abs().modified_cross(fin.max_extent);
        let max_delta_position = s.delta_position.length() + 2.0 * rotation_arc.length();

        // Position correction matters less than true velocity for sleep.
        let sleep_velocity = maxf(
            max_velocity,
            POSITION_SLEEP_FACTOR * inv_dt * max_delta_position,
        );

        let state_offset = i * crate::body::STATE_STRIDE;
        for n in 6..12 {
            state_col.set(state_offset + n, 0.0);
        }
        state_col.set(state_offset + 12, 1.0);

        let transform_p = center.add(q.rotate(fin.local_center).neg());

        write_sim_rotation(sim_col, i, q);
        clear_sim_force_torque(sim_col, i);
        write_fin_center(fin_col, i, center);
        write_fin_transform_p(fin_col, i, transform_p);

        let s2 = i * SIM2_STRIDE;
        let body_id = sim2_col.get(s2 + S2_BODY_ID).to_bits() as usize;
        let body = crate::bodies::record_mut(world_index, body_id);
        body.sleep_velocity = sleep_velocity;
        let transient = crate::body::flags::IS_FAST
            | crate::body::flags::IS_SPEED_CAPPED
            | crate::body::flags::HAD_TIME_OF_IMPACT;
        let sim_flags = sim2_col.get(s2 + S2_FLAGS).to_bits();
        let state_flags = flags_col.get(i * crate::body::STATE_STRIDE);
        body.body_move_index = i as i32;
        body.flags = (body.flags & !transient)
            | ((sim_flags | state_flags)
                & (crate::body::flags::IS_SPEED_CAPPED | crate::body::flags::HAD_TIME_OF_IMPACT));
        sim2_col.set(s2 + S2_FLAGS, f32::from_bits(sim_flags & !transient));
        flags_col.set(i * crate::body::STATE_STRIDE, state_flags & !transient);
        crate::events::write_move(world_index, i);
        let awake = !crate::continuous::sleep_enabled()
            || body.flags & crate::body::flags::ENABLE_SLEEP == 0
            || sleep_velocity > body.sleep_threshold;
        let flags = sim2_col.get(s2 + S2_FLAGS).to_bits() & !crate::continuous::IS_FAST;
        sim2_col.set(s2 + S2_FLAGS, f32::from_bits(flags));
        let mut fast_candidate = false;
        if awake {
            body.sleep_time = 0.0;
            let max_motion = maxf(max_delta_position, max_velocity * h);
            fast_candidate = enable_continuous
                && flags_col.get(i * crate::body::STATE_STRIDE) & DYNAMIC != 0
                && max_motion > SAFETY_FACTOR * sim2_col.get(s2 + S2_MIN_EXTENT);
        }
        if !awake {
            body.sleep_time += h;
        }
        if fast_candidate {
            sim2_col.set(
                s2 + S2_FLAGS,
                f32::from_bits(flags | crate::continuous::IS_FAST),
            );
            if flags & crate::continuous::IS_BULLET != 0 {
                crate::continuous::add_bullet(i);
            } else {
                crate::continuous::solve(world_index, worker, i);
            }
        } else {
            sim2_col.set(s2 + S2_ROTATION0, q.v.x);
            sim2_col.set(s2 + S2_ROTATION0 + 1, q.v.y);
            sim2_col.set(s2 + S2_ROTATION0 + 2, q.v.z);
            sim2_col.set(s2 + S2_ROTATION0 + 3, q.s);
            sim2_col.set(s2 + S2_CENTER0, center.x);
            sim2_col.set(s2 + S2_CENTER0 + 1, center.y);
            sim2_col.set(s2 + S2_CENTER0 + 2, center.z);
        }

        // Continuous collision can clip the rotation before inertia is rebuilt.
        let rotation_matrix = Mat3::from_quat(read_sim(sim_col, i).rotation);
        write_sim_inv_inertia_world(
            sim_col,
            i,
            rotation_matrix
                .mul(sim.inv_inertia_local)
                .mul(rotation_matrix.transpose()),
        );
        crate::arena::mark_finalize_island(world_index, worker, body_id);
        crate::arena::refit_body(world_index, worker, sim_col, fin_col, i);
    }
}

#[cfg(test)]
mod refit_tests {
    use super::*;
    use crate::math::{Quat, Vec3};

    fn refit_convex(
        shape_type: u32,
        geom: &[f32],
        xf: Transform,
        fat: &[f32; 6],
    ) -> ([f32; 6], bool) {
        refit_bounds(convex_bounds(shape_type, geom, xf), fat)
    }

    /// Convex types use inline geometry; the other types use the kernel geometry pools.
    #[test]
    fn convex_refit_partition() {
        assert!(is_convex_refit(0)); // capsule
        assert!(is_convex_refit(3)); // hull
        assert!(is_convex_refit(5)); // sphere
        assert!(!is_convex_refit(1)); // compound
        assert!(!is_convex_refit(2)); // height field
        assert!(!is_convex_refit(4)); // mesh
    }

    /// Gold transform for the capsule/hull cases: translation + the unit quaternion (v=(0.5,-0.5,0.5),
    /// s=0.5), whose rotation matrix has -1 entries — the hull path's `Mat3::abs` is load-bearing under
    /// it (drop the abs and the extent goes negative, moving every bound). All components exactly
    /// representable, so the TS gold script (run against `computeCapsuleAABBOut` / `aabb.transformOut`
    /// + the `computeFatShapeAABBOut` inflate) fed identical bits.
    fn gold_xf() -> Transform {
        Transform {
            p: Vec3::new(1.5, 2.25, -3.75),
            q: Quat {
                v: Vec3::new(0.5, -0.5, 0.5),
                s: 0.5,
            },
        }
    }

    /// The candidate AABB, bit-pinned against the TS-derived gold (T1 lesson: a green fixture gate
    /// pins only the inputs it reaches; pin the bits, not tolerances).
    fn assert_bits(cand: &[f32; 6], expected: &[u32; 6]) {
        let got: [u32; 6] = core::array::from_fn(|i| cand[i].to_bits());
        assert_eq!(&got, expected, "candidate bits diverge from the TS gold");
    }

    /// Capsule refit under a rotated+translated transform: candidate bits against the TS
    /// `computeCapsuleAABBOut` + inflate gold, plus both escape decisions (containment is inclusive —
    /// the candidate's own bounds do not escape; a shrunk fat AABB does).
    #[test]
    fn capsule_refit_matches_ts_gold() {
        // center1(3) center2(3) radius — decimal literals round to the same f32 as Math.fround.
        let geom = [0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.25];
        let expected = [
            0x3f3ae148, 0x3fd70a3e, 0xc08d70a4, 0x3ffc28f6, 0x4047ae14, 0xc05851ec,
        ];
        let wide = [0.0f32, 0.0, -10.0, 10.0, 10.0, 10.0]; // strictly contains the candidate
        let (cand, escaped) = refit_convex(TY_CAPSULE, &geom, gold_xf(), &wide);
        assert_bits(&cand, &expected);
        assert!(!escaped);
        // Containment is inclusive: a fat AABB equal to the candidate does not escape.
        let (cand2, escaped) = refit_convex(TY_CAPSULE, &geom, gold_xf(), &cand);
        assert_eq!(cand2, cand);
        assert!(!escaped);
        // Shrink one face of the fat AABB and the escape fires.
        let mut shrunk = cand;
        shrunk[3] -= 0.5;
        let (_, escaped) = refit_convex(TY_CAPSULE, &geom, gold_xf(), &shrunk);
        assert!(escaped);
    }

    /// Hull refit under the same rotated transform (its -1 rotation entries make `Mat3::abs`
    /// load-bearing): candidate bits against the TS `aabb.transformOut` + inflate gold, plus the
    /// escape decision on each side of containment.
    #[test]
    fn hull_refit_matches_ts_gold() {
        // The hull payload is its local AABB: lower(3) upper(3); slot 7 unused.
        let geom = [-0.3f32, -0.5, -0.7, 0.4, 0.6, 0.2, 0.0];
        let expected = [
            0x3f6147ad, 0x4001eb85, 0xc0823d71, 0x400147ae, 0x403e147b, 0xc0551eb8,
        ];
        let wide = [0.0f32, 0.0, -10.0, 10.0, 10.0, 10.0];
        let (cand, escaped) = refit_convex(TY_HULL, &geom, gold_xf(), &wide);
        assert_bits(&cand, &expected);
        assert!(!escaped);
        let mut shrunk = cand;
        shrunk[0] += 0.5; // raise the lower x face past the candidate's
        let (_, escaped) = refit_convex(TY_HULL, &geom, gold_xf(), &shrunk);
        assert!(escaped);
    }

    /// A unit sphere at the origin under the identity transform: tight [-1,1]³, then the speculative
    /// inflate; the escape test fires exactly when the resident fat AABB no longer contains it.
    #[test]
    fn sphere_refit_candidate_and_escape() {
        let geom = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; // center 0, radius 1
        let xf = Transform {
            p: Vec3::ZERO,
            q: Quat::IDENTITY,
        };
        let s = SPECULATIVE_DISTANCE;
        // A fat AABB wider than the candidate → contained → not escaped.
        let (cand, escaped) =
            refit_convex(TY_SPHERE, &geom, xf, &[-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]);
        assert_eq!(
            cand,
            [-1.0 - s, -1.0 - s, -1.0 - s, 1.0 + s, 1.0 + s, 1.0 + s]
        );
        assert!(!escaped);
        // A fat AABB exactly the tight box → the speculative margin pokes out → escaped.
        let (_, escaped) = refit_convex(TY_SPHERE, &geom, xf, &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        assert!(escaped);
    }
}

// S3 — c_parity: SPECULATIVE_DISTANCE, POSITION_SLEEP_FACTOR, SAFETY_FACTOR parity assertions.
// Canonical table: see `manifold.rs` `c_parity` header comment.

#[cfg(test)]
mod c_parity {
    // Append-only at EOF: Rust bakes panic `file:line` into the data section, so moving this
    // module up-file silently obligates an out-of-scope wasm rebuild. New assertions go at the
    // bottom of this module, never above existing ones.
    // These are private `const`s in this file; read via `super::`.

    #[test]
    fn speculative_distance_finalize() {
        // C: B3_SPECULATIVE_DISTANCE = 4.0f * B3_LINEAR_SLOP (constants.h:73)
        // B3_LINEAR_SLOP = 0.005f * 1.0f, so 4.0f * (0.005f * 1.0f)
        assert_eq!(
            super::SPECULATIVE_DISTANCE.to_bits(),
            (4.0f32 * (0.005f32 * 1.0f32)).to_bits()
        );
    }

    #[test]
    fn position_sleep_factor() {
        // C: positionSleepFactor = 0.5f (solver.c:715, exact k/2^n)
        assert_eq!(super::POSITION_SLEEP_FACTOR.to_bits(), 0.5f32.to_bits());
    }

    #[test]
    fn safety_factor() {
        // C: safetyFactor = 0.5f (solver.c:747, exact k/2^n)
        assert_eq!(super::SAFETY_FACTOR.to_bits(), 0.5f32.to_bits());
    }
}
