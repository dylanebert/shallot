//! The convex-contact manifold bridge (b3ComputeConvexManifold, contact.c): map the geometry
//! manifold the six convex pair functions produce (in shape A's frame) into the persistent,
//! center-of-mass-relative manifold the solver consumes, carrying warm-start impulses forward by
//! feature id. The pair functions (manifold.rs) run in the B-to-A frame; this layer rotates the
//! result into world orientation and matches the previous step's points to reuse their impulses.
//!
//! Native gold-verified against a C transcription of the same map + warm-start over the real box3d
//! collide functions (fixtures/convex_manifold_gold.c). The wasm build dispatches it per direct convex
//! contact through `arena::dispatch_convex` (3c.3), over the geometry + persistent manifold columns.

use crate::distance::SimplexCache;
use crate::hull::HullData;
use crate::manifold::{
    collide_capsule_and_sphere, collide_capsules, collide_hull_and_capsule,
    collide_hull_and_sphere, collide_hulls, collide_spheres, make_feature_id, Capsule,
    LocalManifold, SatCache, Sphere,
};
use crate::math::{Mat3, Transform, Vec3};

/// Max points in a persistent convex manifold (B3_MAX_MANIFOLD_POINTS). The geometry buffer is wider
/// (`GEOM_CAPACITY`); the pair functions reduce to at most this before the map runs.
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// Geometry-manifold point capacity handed to the pair functions — box3d's 32 (`B3_MAX_LOCAL_MANIFOLD_
/// POINTS`), matching the kernel `LocalManifold` buffer (manifold.rs) and the TS narrowphase. A face-face
/// clip on a high-vertex hull exceeds 8 before the reduce, so the capacity must match box3d or the
/// manifold truncates differently (fixture divergence).
const GEOM_CAPACITY: usize = 32;

/// The GJK/SAT warm-start caches a convex contact carries between steps (b3ContactCache). The
/// hull-vs-sphere/capsule paths use `simplex_cache`; hull-vs-hull uses `sat_cache`.
pub struct ConvexContactCache {
    pub simplex_cache: SimplexCache,
    pub sat_cache: SatCache,
}

impl ConvexContactCache {
    pub fn empty() -> ConvexContactCache {
        ConvexContactCache {
            simplex_cache: SimplexCache::empty(),
            sat_cache: SatCache::empty(),
        }
    }
}

/// One point of a persistent contact manifold (b3ManifoldPoint). The narrowphase writes the anchors
/// in world orientation relative to each body origin; the solver stage later shifts them to be
/// center-of-mass-relative. `feature_id` + `normal_impulse` are the warm-start state carried forward.
#[derive(Clone, Copy)]
pub struct ManifoldPoint {
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub separation: f32,
    pub base_separation: f32,
    pub normal_impulse: f32,
    pub total_normal_impulse: f32,
    pub normal_velocity: f32,
    pub feature_id: u32,
    pub triangle_index: i32,
    pub persisted: bool,
}

impl ManifoldPoint {
    pub const ZERO: ManifoldPoint = ManifoldPoint {
        anchor_a: Vec3::ZERO,
        anchor_b: Vec3::ZERO,
        separation: 0.0,
        base_separation: 0.0,
        normal_impulse: 0.0,
        total_normal_impulse: 0.0,
        normal_velocity: 0.0,
        feature_id: 0,
        triangle_index: -1,
        persisted: false,
    };
}

/// The persistent contact manifold between two shapes (b3Manifold). 1–4 points; the friction/twist/
/// rolling impulses persist across steps but are untouched by `compute_convex_manifold`.
pub struct Manifold {
    pub points: [ManifoldPoint; MAX_MANIFOLD_POINTS],
    pub normal: Vec3,
    pub point_count: usize,
    pub twist_impulse: f32,
    pub friction_impulse: Vec3,
    pub rolling_impulse: Vec3,
}

impl Manifold {
    pub fn new() -> Manifold {
        Manifold {
            points: [ManifoldPoint::ZERO; MAX_MANIFOLD_POINTS],
            normal: Vec3::ZERO,
            point_count: 0,
            twist_impulse: 0.0,
            friction_impulse: Vec3::ZERO,
            rolling_impulse: Vec3::ZERO,
        }
    }
}

impl Default for Manifold {
    fn default() -> Self {
        Manifold::new()
    }
}

/// A convex shape the narrowphase dispatches on (the convex subset of b3Shape). Shape A is always the
/// primary (higher) type, so only the valid ordered pairs below are reached.
pub enum ConvexShape<'a> {
    Sphere(Sphere),
    Capsule(Capsule),
    Hull(HullData<'a>),
}

/// Compute the convex-convex manifold and map it into `manifold`, carrying warm-start impulses forward
/// by feature id (b3ComputeConvexManifold). `manifold` holds the previous step's points on entry (the
/// warm-start source) and is overwritten with the new manifold. @returns false (and clears the
/// manifold) when the shapes are not touching.
pub fn compute_convex_manifold(
    manifold: &mut Manifold,
    shape_a: &ConvexShape,
    xf_a: Transform,
    shape_b: &ConvexShape,
    xf_b: Transform,
    cache: &mut ConvexContactCache,
) -> bool {
    let mut geom = LocalManifold::new();
    let transform_b_to_a = xf_a.inv_mul(xf_b);

    match (shape_a, shape_b) {
        (ConvexShape::Sphere(a), ConvexShape::Sphere(b)) => {
            collide_spheres(&mut geom, GEOM_CAPACITY, a, b, transform_b_to_a);
        }
        (ConvexShape::Capsule(a), ConvexShape::Sphere(b)) => {
            collide_capsule_and_sphere(&mut geom, GEOM_CAPACITY, a, b, transform_b_to_a);
        }
        (ConvexShape::Capsule(a), ConvexShape::Capsule(b)) => {
            collide_capsules(&mut geom, GEOM_CAPACITY, a, b, transform_b_to_a);
        }
        (ConvexShape::Hull(a), ConvexShape::Sphere(b)) => {
            collide_hull_and_sphere(
                &mut geom,
                GEOM_CAPACITY,
                a,
                b,
                transform_b_to_a,
                &mut cache.simplex_cache,
            );
        }
        (ConvexShape::Hull(a), ConvexShape::Capsule(b)) => {
            collide_hull_and_capsule(
                &mut geom,
                GEOM_CAPACITY,
                a,
                b,
                transform_b_to_a,
                &mut cache.simplex_cache,
            );
        }
        (ConvexShape::Hull(a), ConvexShape::Hull(b)) => {
            collide_hulls(
                &mut geom,
                GEOM_CAPACITY,
                a,
                b,
                transform_b_to_a,
                &mut cache.sat_cache,
            );
        }
        // Shape A is always the primary type, so no other ordered pairing occurs.
        _ => geom.point_count = 0,
    }

    if geom.point_count == 0 {
        manifold.point_count = 0;
        return false;
    }

    // Snapshot the previous points' feature ids + impulses before overwriting them; the warm-start
    // match below reads the snapshot and claims entries there, never the points being reused.
    let old_count = manifold.point_count;
    let mut old_feat = [0u32; MAX_MANIFOLD_POINTS];
    let mut old_imp = [0.0f32; MAX_MANIFOLD_POINTS];
    for j in 0..old_count {
        old_feat[j] = manifold.points[j].feature_id;
        old_imp[j] = manifold.points[j].normal_impulse;
    }

    let n = geom.point_count;
    manifold.point_count = n;

    let matrix_a = Mat3::from_quat(xf_a.q);
    manifold.normal = matrix_a.mul_v(geom.normal);

    // Contact points are computed in frame A; anchorB is offset by the body-origin separation.
    let offset = xf_a.p.sub(xf_b.p);
    for i in 0..n {
        let source = geom.points[i];
        let pt = &mut manifold.points[i];
        pt.anchor_a = matrix_a.mul_v(source.point);
        pt.anchor_b = pt.anchor_a.add(offset);
        pt.separation = source.separation;
        pt.feature_id = make_feature_id(source.pair);
        pt.triangle_index = -1;
        pt.normal_velocity = 0.0;
    }

    // Copy impulses from any matching old point (by feature id) via the snapshot.
    for i in 0..n {
        let pt = &mut manifold.points[i];
        pt.total_normal_impulse = 0.0;
        pt.persisted = false;
        for j in 0..old_count {
            if pt.feature_id == old_feat[j] {
                pt.normal_impulse = old_imp[j];
                pt.persisted = true;
                old_feat[j] = u32::MAX; // claimed
                break;
            }
        }
        if !pt.persisted {
            pt.normal_impulse = 0.0;
        }
    }

    true
}
