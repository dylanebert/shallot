//! Contact recycling: the recycle branch of box3d's `b3CollideTask` (`physics_world.c`), ported
//! op-for-op via the TS port (`collide.ts` `tryRecycle` + `manifoldstore.ts` `recycleSeparations`).
//! When a contact's relative pose barely moved since last step, the manifold is reused: the anchors
//! are kept and only each point's separation is advanced for the incremental body rotation/translation
//! (a variation of conservative advancement), skipping the full narrowphase. Post-settle this is the
//! fast path ~every contact takes, so it is the collide phase's dominant cost.
//!
//! Pure column arithmetic + the manifold-pool walk. The caller (collide, wired at 4b.3) supplies the
//! bodies' world transforms/centers/extents (resident body columns) and the cached rotations/relative
//! pose (contact record); this returns whether the contact recycled so the caller can `continue`. The
//! touching-state transitions, contact events, and constraint-graph bookkeeping stay TS.
//!
//! Every op maps one-to-one to the C scalar path (no SIMD, no FMA); bit-identical to the
//! `DISABLE_SIMD` reference (see `math.rs`).

use crate::manifold_abi::{
    read_dir, MANIFOLD_STRIDE, M_NORMAL, M_POINTS, M_POINT_COUNT, P_ANCHOR_A, P_ANCHOR_B,
    P_BASE_SEPARATION, P_PERSISTED, P_SEPARATION, POOL_POINT_STRIDE,
};
use crate::col::Col;
use crate::math::{maxf, minf, Mat3, Quat, Transform, Vec3};

/// cos(7°): the recycle gate's angular-distance threshold (`B3_CONTACT_RECYCLE_ANGULAR_DISTANCE`).
const RECYCLE_ANGULAR_DISTANCE: f32 = 0.99240388;

/// Try to reuse `contact_id`'s manifold instead of re-colliding (`b3CollideTask` recycle branch).
///
/// `xf_a`/`xf_b` are the two bodies' world transforms this step, `cached_rot_a`/`cached_rot_b` +
/// `cached_rel_pose` the pose cached last step, `center_a`/`center_b` the bodies' centers of mass,
/// `max_extent_a`/`max_extent_b` their max extents (already zeroed by the caller for static bodies).
/// On success updates every manifold point's separation (base + the conservative-advancement delta),
/// marks it persisted, and returns true; on the gate failing the manifold is left untouched.
#[allow(clippy::too_many_arguments)]
pub fn try_recycle(
    dir: Col<u32>,
    pool: Col<f32>,
    contact_id: usize,
    manifold_count: usize,
    xf_a: Transform,
    xf_b: Transform,
    cached_rot_a: Quat,
    cached_rot_b: Quat,
    cached_rel_pose: Transform,
    center_a: Vec3,
    center_b: Vec3,
    max_extent_a: Vec3,
    max_extent_b: Vec3,
    recycle_tolerance: f32,
) -> bool {
    let angle_a = xf_a.q.dot(cached_rot_a);
    let angle_b = xf_b.q.dot(cached_rot_b);
    let angular_distance = minf(angle_a * angle_a, angle_b * angle_b);

    let xf = xf_a.inv_mul(xf_b);
    let xfc = cached_rel_pose;
    let max_extent = Vec3::new(
        maxf(max_extent_a.x, max_extent_b.x),
        maxf(max_extent_a.y, max_extent_b.y),
        maxf(max_extent_a.z, max_extent_b.z),
    );

    // b3DistanceSquared(xf.p, xfc.p): dv = xfc.p - xf.p.
    let dv = xfc.p.sub(xf.p);
    let dist_squared = dv.dot(dv);

    if angular_distance > RECYCLE_ANGULAR_DISTANCE
        && dist_squared < recycle_tolerance * recycle_tolerance
    {
        let distance = dist_squared.sqrt();
        let slack = recycle_tolerance - distance;

        // qr: the rotation from the cached relative pose to the current one, ~B's local angular
        // velocity when A is static (physics_world.c derivation).
        let qr = xfc.q.inv_mul(xf.q);
        let arc = qr.v.abs().modified_cross(max_extent);
        let arc_sq = 4.0 * arc.length_sq();

        if arc_sq < slack * slack {
            let dq_a = xf_a.q.mul(cached_rot_a.conjugate());
            let dq_b = xf_b.q.mul(cached_rot_b.conjugate());
            let matrix_a = Mat3::from_quat(dq_a);
            let matrix_b = Mat3::from_quat(dq_b);
            // Minimize round-off: difference the two centers directly.
            let dc = center_b.sub(center_a);

            recycle_separations(dir, pool, contact_id, manifold_count, matrix_a, matrix_b, dc);
            return true;
        }
    }

    false
}

/// The recycle-success separation update (`manifoldstore.ts` `recycleSeparations`): keep every
/// manifold point's anchors, recompute `separation = baseSeparation + dot(dc + (matrixB·anchorB −
/// matrixA·anchorA), normal)`, and mark it persisted.
fn recycle_separations(
    dir: Col<u32>,
    pool: Col<f32>,
    contact_id: usize,
    count: usize,
    matrix_a: Mat3,
    matrix_b: Mat3,
    dc: Vec3,
) {
    let base = read_dir(dir, contact_id).manifold_base;
    for m in 0..count {
        let mo = (base + m) * MANIFOLD_STRIDE;
        let normal = Vec3::new(
            pool.get(mo + M_NORMAL),
            pool.get(mo + M_NORMAL + 1),
            pool.get(mo + M_NORMAL + 2),
        );
        let pc = pool.get(mo + M_POINT_COUNT).to_bits() as usize;
        for p in 0..pc {
            let po = mo + M_POINTS + p * POOL_POINT_STRIDE;
            let anchor_a = Vec3::new(
                pool.get(po + P_ANCHOR_A),
                pool.get(po + P_ANCHOR_A + 1),
                pool.get(po + P_ANCHOR_A + 2),
            );
            let anchor_b = Vec3::new(
                pool.get(po + P_ANCHOR_B),
                pool.get(po + P_ANCHOR_B + 1),
                pool.get(po + P_ANCHOR_B + 2),
            );
            let r_a = matrix_a.mul_v(anchor_a);
            let r_b = matrix_b.mul_v(anchor_b);
            let dp = dc.add(r_b.sub(r_a));
            pool.set(po + P_SEPARATION, pool.get(po + P_BASE_SEPARATION) + dp.dot(normal));
            pool.set(po + P_PERSISTED, f32::from_bits(1));
        }
    }
}
