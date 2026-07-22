//! Pose finalize: the per-body pose-advance phase of box3d's soft-step solver, ported op-for-op from
//! `solver.c` (b3FinalizeBodiesTask) via the TS port (`solver.ts` `finalizeBodies`). It advances the
//! center + rotation from the solved deltas, rebuilds the world inertia tensor and body-origin
//! transform, resets the per-step delta/force accumulators, and emits the two sleep/continuous
//! decision scalars TS branches on.
//!
//! Only the pure column arithmetic lives here. Everything the C task does that touches TS-owned world
//! state stays TS and interleaves at wiring: move-event emission, the continuous (CCD) sweep, island
//! awake/split bookkeeping, `center0`/`rotation0` + `sleepTime`, the three-object flag reset, and the
//! shape-AABB refit + broadphase enlarge. TS reads the advanced transform/center/inertia back out of
//! the shared columns and runs that tail against `[sleepVelocity, maxMotion]`.
//!
//! Every arithmetic op maps one-to-one to the C scalar path (no SIMD, no FMA); bit-identical to the
//! `DISABLE_SIMD` + `FORCE_OVERFLOW` reference (see `math.rs`).

use crate::body::{
    clear_sim_force_torque, flags::DYNAMIC, read_fin, read_sim, read_state, write_fin_center,
    write_fin_transform_p, write_sim_inv_inertia_world, write_sim_rotation, write_state,
    FIN_OUT_STRIDE, S2_CENTER0, S2_MIN_EXTENT, S2_ROTATION0, SIM2_STRIDE,
};
use crate::col::Col;
use crate::math::{maxf, minf, Mat3, Quat, Transform, Vec3};

/// Position-correction weight for the sleep-velocity blend (b3FinalizeBodies `positionSleepFactor`).
const POSITION_SLEEP_FACTOR: f32 = 0.5;

/// The continuous-collision safety factor (b3FinalizeBodies `safetyFactor`): a body whose step motion
/// exceeds `SAFETY_FACTOR * minExtent` is a fast-body candidate. `0.5` is exact in f32, so `0.5 * x`
/// matches the TS `f32(0.5 * minExtent)` bit-for-bit.
const SAFETY_FACTOR: f32 = 0.5;

// --- shape-AABB refit -----------------------------------------------------------------------
// The pure per-shape half of the refit (T4): the tight AABB compute + speculative inflate + escape
// test, mirroring `src/solver.ts` finalizeBodies op-for-op. Kept here (native, gold/fixture-verified)
// and free of the wasm-only region layers; `arena::refit_block` gathers each shape's transform +
// geometry + resident fat AABB out of the columns and calls in, writing the candidate + escaped flag
// back for TS to commit serially.

/// Shape type codes (`src/types.ts` `ShapeType`) the kernel refits in-kernel — sphere/capsule/hull are
/// convex and transform-only. Mesh(4) / height-field(2) / compound(1) fall back to the TS path.
pub const TY_CAPSULE: u32 = 0;
pub const TY_HULL: u32 = 3;
pub const TY_SPHERE: u32 = 5;

/// The speculative margin the fat AABB inflates a shape by (`B3_SPECULATIVE_DISTANCE`, `src/core.ts`).
/// `4.0 * 0.005` const-evaluates to the same f32 as the TS `f32(4.0 * f32(0.005))`.
const SPECULATIVE_DISTANCE: f32 = 4.0 * 0.005;

/// Does the kernel compute this shape type's refit AABB in-kernel? The convex/fallback partition the
/// finalize refit rests on; the TS side mirrors it (`isConvexRefit`, `src/shapecolumns.ts`).
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
        Vec3::new(minf(c1.x, c2.x) - r, minf(c1.y, c2.y) - r, minf(c1.z, c2.z) - r),
        Vec3::new(maxf(c1.x, c2.x) + r, maxf(c1.y, c2.y) + r, maxf(c1.z, c2.z) + r),
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

/// The speculative-inflated tight world AABB of convex shape `geom` (type `shape_type`) under world
/// transform `xf`, and whether it escaped the shape's resident fat AABB `fat`. Mirrors the TS finalize
/// refit op-for-op: `computeShapeAABBOut` (tight) → `computeFatShapeAABBOut` (inflate by the speculative
/// margin, each bound its own round) → `aabb.contains`. `shape_type` must be convex (caller partitions
/// on [`is_convex_refit`]). Returns the candidate `[lower.xyz, upper.xyz]` + the escaped flag.
pub fn refit_convex(shape_type: u32, geom: &[f32], xf: Transform, fat: &[f32; 6]) -> ([f32; 6], bool) {
    let (lo, hi) = match shape_type {
        TY_SPHERE => sphere_aabb(geom, xf),
        TY_CAPSULE => capsule_aabb(geom, xf),
        _ => hull_aabb(geom, xf), // TY_HULL
    };
    let s = SPECULATIVE_DISTANCE;
    let cand = [lo.x - s, lo.y - s, lo.z - s, hi.x + s, hi.y + s, hi.z + s];
    let escaped = !aabb_contains(fat, &cand);
    (cand, escaped)
}

/// Advance the bodies in `[start, start+count)` from their solved velocity/position deltas.
///
/// Reads state (velocities + deltas), sim (transform.q + invInertiaLocal), the finalize column
/// (center, localCenter, maxExtent), and the sim2 minExtent + flags. Writes the advanced
/// center/transform.p (finalize column), transform.q + invInertiaWorld (sim), the cleared force/torque
/// (sim), the reset deltas (state), `[sleepVelocity, maxMotion]` per body (out column), and — for every
/// non-fast body — the sweep base center0/rotation0 (sim2). `h` is the full-step dt, `inv_dt` its inverse.
///
/// The sweep base fold mirrors b3FinalizeBodiesTask's `sim->center0 = sim->center; sim->rotation0 = q`,
/// which the C task runs for every body that isn't a fast (continuous) candidate. A fast candidate is
/// skipped here and gets its base from the CCD sweep instead (`solveContinuous`, TS-side); the TS tail
/// still writes the base for the sleepy branch (a sleepy fast candidate's one writer). Bit-exact-safe
/// because center0/rotation0 are read only by the CCD sweep and never hashed/dumped. `enable_continuous`
/// is the world's continuous toggle (the TS test's third term): with it off no body is a fast candidate
/// and every dynamic body gets its base here, exactly as the C task does — keeping the predicate
/// identical to TS's unconditionally, so a future runtime `enableContinuous` setter (upstream
/// b3World_EnableContinuous) can't silently diverge the two.
pub fn finalize(
    state_col: Col<f32>,
    sim_col: Col<f32>,
    fin_col: Col<f32>,
    out_col: Col<f32>,
    sim2_col: Col<f32>,
    flags_col: Col<u32>,
    start: usize,
    count: usize,
    h: f32,
    inv_dt: f32,
    enable_continuous: bool,
) {
    for i in start..start + count {
        let mut s = read_state(state_col, i);
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
        let sleep_velocity = maxf(max_velocity, POSITION_SLEEP_FACTOR * inv_dt * max_delta_position);

        s.delta_position = Vec3::ZERO;
        s.delta_rotation = Quat::IDENTITY;

        let transform_p = center.add(q.rotate(fin.local_center).neg());

        // World-space inverse inertia tensor: R * invInertiaLocal * Rᵀ.
        let rotation_matrix = Mat3::from_quat(q);
        let inv_inertia_world = rotation_matrix
            .mul(sim.inv_inertia_local)
            .mul(rotation_matrix.transpose());

        let max_motion = maxf(max_delta_position, max_velocity * h);

        write_state(state_col, i, &s);
        write_sim_rotation(sim_col, i, q);
        write_sim_inv_inertia_world(sim_col, i, inv_inertia_world);
        clear_sim_force_torque(sim_col, i);
        write_fin_center(fin_col, i, center);
        write_fin_transform_p(fin_col, i, transform_p);

        let o = i * FIN_OUT_STRIDE;
        out_col.set(o, sleep_velocity);
        out_col.set(o + 1, max_motion);

        // Sweep base for the next continuous step: center0 = center, rotation0 = q, written for every
        // body that isn't a fast candidate (b3FinalizeBodiesTask's non-fast/sleepy branch). A fast
        // candidate — continuous enabled, dynamic, moving farther than half its smallest extent — is
        // left for the CCD path to set. `min_extent` and the dynamic bit ride the sim2/flags columns.
        let s2 = i * SIM2_STRIDE;
        let min_extent = sim2_col.get(s2 + S2_MIN_EXTENT);
        let is_dynamic = flags_col.get(i) & DYNAMIC != 0;
        let fast_candidate =
            enable_continuous && is_dynamic && max_motion > SAFETY_FACTOR * min_extent;
        if !fast_candidate {
            sim2_col.set(s2 + S2_ROTATION0, q.v.x);
            sim2_col.set(s2 + S2_ROTATION0 + 1, q.v.y);
            sim2_col.set(s2 + S2_ROTATION0 + 2, q.v.z);
            sim2_col.set(s2 + S2_ROTATION0 + 3, q.s);
            sim2_col.set(s2 + S2_CENTER0, center.x);
            sim2_col.set(s2 + S2_CENTER0 + 1, center.y);
            sim2_col.set(s2 + S2_CENTER0 + 2, center.z);
        }
    }
}

#[cfg(test)]
mod refit_tests {
    use super::*;
    use crate::math::{Quat, Vec3};

    /// The convex/fallback partition the finalize refit rests on — sphere/capsule/hull in-kernel, every
    /// other `ShapeType` value in TS. Pins each of the six codes against `src/types.ts`.
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
        let (cand, escaped) = refit_convex(TY_SPHERE, &geom, xf, &[-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]);
        assert_eq!(cand, [-1.0 - s, -1.0 - s, -1.0 - s, 1.0 + s, 1.0 + s, 1.0 + s]);
        assert!(!escaped);
        // A fat AABB exactly the tight box → the speculative margin pokes out → escaped.
        let (_, escaped) = refit_convex(TY_SPHERE, &geom, xf, &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        assert!(escaped);
    }
}
