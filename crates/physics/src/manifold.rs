//! Convex narrowphase manifold generation, ported op-for-op from box3d's `manifold.c` (shared clip/
//! query helpers) and `convex_manifold.c` (the six sphere/capsule/hull pair functions) (Erin Catto,
//! MIT) via the upstream TS port (`src/manifold.ts`). Results are in shape A's local frame;
//! `transform_b_to_a` places shape B in shape A's frame.
//!
//! The TS port carries a zero-alloc ping-pong buffer strategy for the clip loop; here the arithmetic
//! is identical but storage is plain local `Vec`/arrays (Rust value types remove the shared-reference
//! aliasing hazards the TS strategy had to guard against). Only the arithmetic is contract-bound.

use crate::distance::{shape_distance, DistanceInput, ShapeProxy, SimplexCache};
use crate::hull::HullData;
use crate::math::{
    absf, arbitrary_perp, get_length_and_normalize, is_within_segments, line_distance, maxf, minf,
    point_to_segment_distance, segment_distance, Mat3, Plane, Transform, Vec3, FLT_EPSILON,
    FLT_MAX, FLT_MIN,
};

const LINEAR_SLOP: f32 = 0.005;
const SPECULATIVE_DISTANCE: f32 = 4.0 * LINEAR_SLOP;
const MIN_CAPSULE_LENGTH: f32 = LINEAR_SLOP;
const MAX_CLIP_POINTS: usize = 64;

const SHAPE_A: u8 = 0;
const SHAPE_B: u8 = 1;

/// Sphere primitive (b3Sphere).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

/// Capsule primitive (b3Capsule).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Capsule {
    pub center1: Vec3,
    pub center2: Vec3,
    pub radius: f32,
}

impl Capsule {
    pub(crate) fn points(&self) -> &[Vec3] {
        // repr(C) places the two initialized centers contiguously before the radius.
        unsafe { core::slice::from_raw_parts(core::ptr::from_ref(self).cast::<Vec3>(), 2) }
    }
}

/// Identifies a contact point by the two intersecting edges that produced it (b3FeaturePair).
#[derive(Clone, Copy)]
pub struct FeaturePair {
    pub owner1: u8,
    pub index1: u8,
    pub owner2: u8,
    pub index2: u8,
}

impl FeaturePair {
    pub(crate) const SINGLE: FeaturePair = FeaturePair {
        owner1: 0,
        index1: 0,
        owner2: 0,
        index2: 0,
    };
}

/// A local manifold point in shape A's frame (b3LocalManifoldPoint).
#[derive(Clone, Copy)]
pub struct LocalManifoldPoint {
    pub point: Vec3,
    pub separation: f32,
    pub pair: FeaturePair,
    pub triangle_index: i32,
}

impl LocalManifoldPoint {
    pub const ZERO: LocalManifoldPoint = LocalManifoldPoint {
        point: Vec3::ZERO,
        separation: 0.0,
        pair: FeaturePair::SINGLE,
        triangle_index: 0,
    };
}

/// Local manifold borrowing caller-owned points: 32 for convex contacts, the remaining
/// shared point-pool span for triangle contacts.
pub struct LocalManifold<'a> {
    pub normal: Vec3,
    pub triangle_normal: Vec3,
    pub points: &'a mut [LocalManifoldPoint],
    pub point_count: usize,
    pub triangle_index: i32,
    pub vertex_indices: [u32; 3],
    pub triangle_flags: u32,
    pub feature: u32,
    pub squared_distance: f32,
}

impl<'a> LocalManifold<'a> {
    pub fn new(points: &'a mut [LocalManifoldPoint]) -> Self {
        LocalManifold {
            normal: Vec3::ZERO,
            triangle_normal: Vec3::ZERO,
            points,
            point_count: 0,
            triangle_index: 0,
            vertex_indices: [0; 3],
            triangle_flags: 0,
            feature: 0,
            squared_distance: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
struct SeparatingAxis {
    normal: Vec3,
    separation: f32,
    index_a: i32,
    index_b: i32,
    ty: u32,
}

/// Cached separating-axis feature type (b3SeparatingFeature).
pub mod separating_feature {
    pub const INVALID: u32 = 0;
    pub const FACE_AXIS_A: u32 = 2;
    pub const FACE_AXIS_B: u32 = 3;
    pub const EDGE_PAIR_AXIS: u32 = 4;
    pub const MANUAL_FACE_AXIS_A: u32 = 6;
    pub const MANUAL_FACE_AXIS_B: u32 = 7;
    pub const MANUAL_EDGE_PAIR_AXIS: u32 = 8;
}

/// Separating-axis test cache for temporal acceleration of hull-hull collision (b3SATCache).
#[derive(Clone, Copy)]
pub struct SatCache {
    pub separation: f32,
    pub ty: u32,
    pub index_a: usize,
    pub index_b: usize,
    pub hit: u32,
}

impl SatCache {
    pub fn empty() -> SatCache {
        SatCache {
            separation: 0.0,
            ty: 0,
            index_a: 0,
            index_b: 0,
            hit: 0,
        }
    }

    fn reset(&mut self) {
        self.separation = 0.0;
        self.ty = 0;
        self.index_a = 0;
        self.index_b = 0;
        self.hit = 0;
    }
}

/// A clip-polygon vertex (b3ClipVertex).
#[derive(Clone, Copy)]
pub(crate) struct ClipVertex {
    pub(crate) position: Vec3,
    pub(crate) separation: f32,
    pub(crate) pair: FeaturePair,
}

impl ClipVertex {
    pub(crate) const ZERO: ClipVertex = ClipVertex {
        position: Vec3::ZERO,
        separation: 0.0,
        pair: FeaturePair::SINGLE,
    };
}

// --- shared helpers (manifold.c) ---------------------------------------------------------------

/// b3MakeFeaturePair — pack two features into a pair (each index truncated to uint8).
fn make_feature_pair(owner1: u8, index1: u8, owner2: u8, index2: u8) -> FeaturePair {
    FeaturePair {
        owner1,
        index1,
        owner2,
        index2,
    }
}

/// b3MakeFeatureId — pack a feature pair into a uint32 id for warm-start matching.
pub fn make_feature_id(pair: FeaturePair) -> u32 {
    ((pair.owner1 as u32) << 24)
        | ((pair.index1 as u32) << 16)
        | ((pair.owner2 as u32) << 8)
        | (pair.index2 as u32)
}

/// b3FlipPair — swap owners (and flip each) and indices so the pair is independent of the reference.
pub(crate) fn flip_pair(pair: FeaturePair) -> FeaturePair {
    FeaturePair {
        owner1: 1 - pair.owner2,
        index1: pair.index2,
        owner2: 1 - pair.owner1,
        index2: pair.index1,
    }
}

/// b3FindIncidentFace — the face on `hull` most anti-parallel to `ref_normal`.
pub(crate) fn find_incident_face(hull: &HullData, ref_normal: Vec3, vertex_index: usize) -> usize {
    let edges = &hull.edges;
    let planes = &hull.planes;
    let points = &hull.points;

    let mut min_edge_index = 0usize;
    let mut min_edge_projection = FLT_MAX;

    let vertex = hull.vertices[vertex_index];
    let mut edge_index = vertex.edge as usize;
    let mut edge = edges[edge_index];
    let edge_origin = points[edge.origin as usize];

    loop {
        let twin = edges[edge.twin as usize];
        let twin_origin = points[twin.origin as usize];
        let axis = twin_origin.sub(edge_origin).normalize();
        let edge_projection = absf(axis.dot(ref_normal));
        if edge_projection < min_edge_projection {
            min_edge_index = edge_index;
            min_edge_projection = edge_projection;
        }
        edge_index = twin.next as usize;
        edge = edges[edge_index];
        if edge_index == vertex.edge as usize {
            break;
        }
    }

    let min_edge = edges[min_edge_index];
    let min_face_index1 = min_edge.face as usize;
    let min_plane1 = planes[min_face_index1];
    let min_twin = edges[min_edge.twin as usize];
    let min_face_index2 = min_twin.face as usize;
    let min_plane2 = planes[min_face_index2];

    if min_plane1.normal.dot(ref_normal) < min_plane2.normal.dot(ref_normal) {
        min_face_index1
    } else {
        min_face_index2
    }
}

pub(crate) trait ClipSlot {
    fn put(&mut self, vertex: ClipVertex);
}

impl ClipSlot for ClipVertex {
    fn put(&mut self, vertex: ClipVertex) {
        *self = vertex;
    }
}

impl ClipSlot for core::mem::MaybeUninit<ClipVertex> {
    fn put(&mut self, vertex: ClipVertex) {
        self.write(vertex);
    }
}

/// b3ClipPolygon — Sutherland-Hodgman clip of `polygon` against `clip_plane`, writing the clipped polygon
/// into `out` and returning its length. Intersection points re-own their cut edge to `edge` on shape A.
/// `out` must hold at least `count + 1` slots (the clip grows the polygon by at most one vertex).
pub(crate) fn clip_polygon(
    polygon: &[ClipVertex],
    count: usize,
    clip_plane: Plane,
    edge: u8,
    ref_plane: Plane,
    out: &mut [impl ClipSlot],
) -> usize {
    let mut n = 0;

    let mut vertex1 = polygon[count - 1];
    let mut distance1 = clip_plane.separation(vertex1.position);

    for index in 0..count {
        let vertex2 = polygon[index];
        let distance2 = clip_plane.separation(vertex2.position);

        if distance1 <= 0.0 && distance2 <= 0.0 {
            // Both behind: keep vertex2.
            out[n].put(vertex2);
            n += 1;
        } else if distance1 <= 0.0 && distance2 > 0.0 {
            // Leaving: keep intersection, adjust outgoing edge.
            let fraction = distance1 / (distance1 - distance2);
            let position = vertex1
                .position
                .mul_add(fraction, vertex2.position.sub(vertex1.position));
            let mut pair = vertex2.pair;
            pair.owner2 = SHAPE_A;
            pair.index2 = edge;
            out[n].put(ClipVertex {
                position,
                separation: ref_plane.separation(position),
                pair,
            });
            n += 1;
        } else if distance2 <= 0.0 && distance1 > 0.0 {
            // Entering: keep intersection (adjust incoming edge) then vertex2.
            let fraction = distance1 / (distance1 - distance2);
            let position = vertex1
                .position
                .mul_add(fraction, vertex2.position.sub(vertex1.position));
            let mut pair = vertex1.pair;
            pair.owner1 = SHAPE_A;
            pair.index1 = edge;
            out[n].put(ClipVertex {
                position,
                separation: ref_plane.separation(position),
                pair,
            });
            n += 1;
            out[n].put(vertex2);
            n += 1;
        }

        vertex1 = vertex2;
        distance1 = distance2;
    }

    n
}

// --- convex_manifold.c: Gauss-map / clip helpers -----------------------------------------------

fn is_minkowski_face(a: Vec3, b: Vec3, bxa: Vec3, c: Vec3, d: Vec3, dxc: Vec3) -> bool {
    let cba = c.dot(bxa);
    let dba = d.dot(bxa);
    let adc = a.dot(dxc);
    let bdc = b.dot(dxc);
    cba * dba < 0.0 && adc * bdc < 0.0 && cba * bdc > 0.0
}

/// b3ClipSegment — clip a 2-vertex segment against `pl`, in place. Returns the vertex count.
fn clip_segment(segment: &mut [ClipVertex; 2], pl: Plane) -> usize {
    let vertex1 = segment[0];
    let vertex2 = segment[1];

    let distance1 = pl.separation(vertex1.position);
    let distance2 = pl.separation(vertex2.position);

    let mut vertex_count = 0;
    if distance1 <= 0.0 {
        segment[vertex_count] = vertex1;
        vertex_count += 1;
    }
    if distance2 <= 0.0 {
        segment[vertex_count] = vertex2;
        vertex_count += 1;
    }

    if distance1 * distance2 < 0.0 {
        let t = distance1 / (distance1 - distance2);
        let position = vertex1
            .position
            .scale(1.0 - t)
            .add(vertex2.position.scale(t));
        let src = if distance1 > 0.0 { vertex1 } else { vertex2 };
        segment[vertex_count].position = position;
        segment[vertex_count].pair = src.pair;
        vertex_count += 1;
    }

    vertex_count
}

/// b3ClipSegmentToHullFace — clip a segment against every side plane of the reference face.
fn clip_segment_to_hull_face(
    segment: &mut [ClipVertex; 2],
    hull: &HullData,
    ref_face: usize,
) -> usize {
    let faces = &hull.faces;
    let planes = &hull.planes;
    let edges = &hull.edges;
    let points = &hull.points;

    let ref_plane = planes[ref_face];
    let face = faces[ref_face];
    let mut edge_index = face.edge as usize;

    loop {
        let edge = edges[edge_index];
        let next_edge_index = edge.next as usize;
        let next = edges[next_edge_index];

        let vertex1 = points[edge.origin as usize];
        let vertex2 = points[next.origin as usize];
        let tangent = vertex2.sub(vertex1).normalize();
        let binormal = tangent.cross(ref_plane.normal);

        let point_count = clip_segment(segment, Plane::from_normal_and_point(binormal, vertex1));
        if point_count < 2 {
            return 0;
        }
        edge_index = next_edge_index;
        if edge_index == face.edge as usize {
            break;
        }
    }

    2
}

// --- convex_manifold.c: SAT queries ------------------------------------------------------------

fn query_face_direction_hull_and_capsule(
    hull: &HullData,
    capsule: &Capsule,
    capsule_transform: Transform,
) -> SeparatingAxis {
    let mut max_face_index = 0usize;
    let mut max_vertex_index = 0usize;
    let mut max_face_separation = -FLT_MAX;
    let planes = &hull.planes;

    let capsule_points = [
        capsule_transform.point(capsule.center1),
        capsule_transform.point(capsule.center2),
    ];

    for face_index in 0..hull.face_count {
        let pl = planes[face_index];
        let vertex_index = crate::distance::get_point_support(&capsule_points, 2, pl.normal.neg());
        let support = capsule_points[vertex_index];
        let separation = pl.separation(support);
        if separation > max_face_separation {
            max_vertex_index = vertex_index;
            max_face_index = face_index;
            max_face_separation = separation;
        }
    }

    SeparatingAxis {
        normal: planes[max_face_index].normal,
        separation: max_face_separation,
        index_a: max_face_index as i32,
        index_b: max_vertex_index as i32,
        ty: separating_feature::FACE_AXIS_A,
    }
}

fn query_edge_direction_hull_and_capsule(
    hull: &HullData,
    capsule: &Capsule,
    capsule_transform: Transform,
) -> SeparatingAxis {
    let mut max_normal = Vec3::ZERO;
    let mut max_separation = -FLT_MAX;
    let mut max_index_a: i32 = -1;
    let mut max_index_b: i32 = -1;

    // All computations in local space of the hull.
    let p1 = capsule_transform.point(capsule.center1);
    let q1 = capsule_transform.point(capsule.center2);
    let e1 = q1.sub(p1);

    let edges = &hull.edges;
    let points = &hull.points;
    let planes = &hull.planes;

    let mut index = 0;
    while index < hull.edge_count {
        let edge = edges[index];
        let twin = edges[index + 1];

        let q_b = points[twin.origin as usize];
        let u_b = planes[edge.face as usize].normal;
        let v_b = planes[twin.face as usize].normal;
        let cba = u_b.dot(e1);
        let dba = v_b.dot(e1);
        if cba * dba < 0.0 {
            let squared_tolerance = 0.005f32 * 0.005;
            if maxf(cba * cba, dba * dba) < squared_tolerance * e1.length_sq() {
                index += 2;
                continue;
            }
            let t = cba / (cba - dba);
            let axis = u_b.lerp(v_b, t).normalize();
            let separation = axis.dot(q1.sub(q_b));
            if separation > max_separation {
                max_normal = axis;
                max_separation = separation;
                max_index_a = 0;
                max_index_b = index as i32;
            }
        }
        index += 2;
    }

    SeparatingAxis {
        normal: max_normal,
        separation: max_separation,
        index_a: max_index_a,
        index_b: max_index_b,
        ty: separating_feature::EDGE_PAIR_AXIS,
    }
}

/// b3ReduceManifoldPoints — reduce a clipped point set to at most 4 points via a biased extremum
/// search over `points[0..count]`; writes the survivors into `manifold.points`.
fn reduce_manifold_points(
    manifold: &mut LocalManifold,
    capacity: usize,
    points: &mut [LocalManifoldPoint],
    mut count: usize,
) {
    if capacity < 4 {
        return;
    }

    if count <= 4 {
        for i in 0..count {
            manifold.points[i] = points[i];
        }
        manifold.point_count = count;
        return;
    }

    let normal = manifold.normal;
    let speculative_distance = SPECULATIVE_DISTANCE;
    let tol_sqr = speculative_distance * speculative_distance;

    // A pecking-order bias for contact point consistency across time steps.
    let bias: f32 = 0.95;

    // Step 1: extreme point that is touching.
    let mut best_index: i32 = -1;
    let mut best_score = -FLT_MAX;
    let search_direction = arbitrary_perp(normal);
    for index in 0..count {
        let pt = &points[index];
        if pt.separation > speculative_distance {
            continue;
        }
        // The deeper the better.
        let score = -pt.separation + search_direction.dot(pt.point);
        if bias * score > best_score {
            best_index = index as i32;
            best_score = score;
        }
    }

    if best_index == -1 {
        manifold.point_count = 0;
        return;
    }

    manifold.points[0] = points[best_index as usize];
    manifold.point_count = 1;
    points[best_index as usize] = points[count - 1];
    count -= 1;

    let a = manifold.points[0].point;

    // Step 2: farthest point in 2D.
    best_score = 0.0;
    best_index = -1;
    for index in 0..count {
        let p = points[index].point;
        let d = p.sub(a);
        let v = d.mul_sub(d.dot(normal), normal);
        let distance_squared = v.length_sq();
        let separation = maxf(0.0, -points[index].separation);
        let score = distance_squared + 4.0 * separation * separation;
        if bias * score > best_score {
            best_score = score;
            best_index = index as i32;
        }
    }

    if best_score < tol_sqr {
        return;
    }

    manifold.points[1] = points[best_index as usize];
    manifold.point_count = 2;
    points[best_index as usize] = points[count - 1];
    count -= 1;

    let b = manifold.points[1].point;

    // Step 3: point with the maximum triangular area.
    best_score = tol_sqr;
    best_index = -1;
    let mut best_signed_area = 0.0;
    let ba = b.sub(a);
    for index in 0..count {
        let p = points[index].point;
        let signed_area = normal.dot(ba.cross(p.sub(a)));
        let score = absf(signed_area);
        if bias * score >= best_score {
            best_score = score;
            best_index = index as i32;
            best_signed_area = signed_area;
        }
    }

    if best_index == -1 {
        return;
    }

    manifold.points[2] = points[best_index as usize];
    manifold.point_count = 3;
    points[best_index as usize] = points[count - 1];
    count -= 1;

    let c = manifold.points[2].point;

    // Step 4: point adding the most area outside the current triangle.
    best_score = tol_sqr;
    best_index = -1;
    let sign: f32 = if best_signed_area < 0.0 { -1.0 } else { 1.0 };
    for index in 0..count {
        let p = points[index].point;
        let u1 = sign * normal.dot(p.sub(a).cross(ba));
        let u2 = sign * normal.dot(p.sub(b).cross(c.sub(b)));
        let u3 = sign * normal.dot(p.sub(c).cross(a.sub(c)));
        let score = maxf(u1, maxf(u2, u3));
        if bias * score > best_score {
            best_score = score;
            best_index = index as i32;
        }
    }

    if best_index != -1 {
        manifold.points[manifold.point_count] = points[best_index as usize];
        manifold.point_count += 1;
    }
}

// --- sphere / capsule pair collision -----------------------------------------------------------

/// b3CollideSpheres — one-point manifold for two spheres, in frame A.
pub fn collide_spheres(
    manifold: &mut LocalManifold,
    capacity: usize,
    sphere_a: &Sphere,
    sphere_b: &Sphere,
    transform_b_to_a: Transform,
) {
    if capacity == 0 {
        return;
    }

    let center1 = sphere_a.center;
    let center2 = transform_b_to_a.point(sphere_b.center);

    let total_radius = sphere_a.radius + sphere_b.radius;
    let offset = center2.sub(center1);
    let distance_sq = offset.length_sq();

    if distance_sq > total_radius * total_radius {
        return;
    }

    let mut normal = Vec3::new(0.0, 1.0, 0.0);
    let distance = distance_sq.sqrt();
    if distance * distance > 1000.0 * FLT_MIN {
        normal = offset.scale(1.0 / distance);
    }

    // Contact at the midpoint: 0.5 * (((c1 + rA*n) + c2) - rB*n).
    let point = center1
        .mul_add(sphere_a.radius, normal)
        .add(center2)
        .mul_sub(sphere_b.radius, normal)
        .scale(0.5);

    manifold.normal = normal;
    manifold.point_count = 1;

    let pt = &mut manifold.points[0];
    pt.point = point;
    pt.separation = distance - total_radius;
    pt.pair = FeaturePair::SINGLE;
}

/// b3CollideCapsuleAndSphere — one-point manifold for a capsule (A) and sphere (B), in frame A.
pub fn collide_capsule_and_sphere(
    manifold: &mut LocalManifold,
    capacity: usize,
    capsule_a: &Capsule,
    sphere_b: &Sphere,
    transform_b_to_a: Transform,
) {
    manifold.point_count = 0;

    if capacity < 1 {
        return;
    }

    let center = transform_b_to_a.point(sphere_b.center);
    let center1 = capsule_a.center1;
    let center2 = capsule_a.center2;

    let total_radius = sphere_b.radius + capsule_a.radius;

    let closest_point = point_to_segment_distance(center1, center2, center);
    let offset = center.sub(closest_point);
    let distance_sq = offset.length_sq();

    if distance_sq > total_radius * total_radius {
        return;
    }

    let mut normal = Vec3::new(0.0, 1.0, 0.0);
    let distance = distance_sq.sqrt();
    if distance * distance > 1000.0 * FLT_MIN {
        normal = offset.scale(1.0 / distance);
    }

    // Contact at the midpoint: 0.5 * (((center - sB*n) + closestPoint) + cA*n).
    let point = center
        .mul_sub(sphere_b.radius, normal)
        .add(closest_point)
        .mul_add(capsule_a.radius, normal)
        .scale(0.5);

    manifold.normal = normal;
    manifold.point_count = 1;

    let pt = &mut manifold.points[0];
    pt.point = point;
    pt.separation = distance - total_radius;
    pt.pair = FeaturePair::SINGLE;
}

/// b3CollideHullAndSphere — one-point manifold for a hull (A) and sphere (B), in frame A.
pub fn collide_hull_and_sphere(
    manifold: &mut LocalManifold,
    capacity: usize,
    hull_a: &HullData,
    sphere_b: &Sphere,
    transform_b_to_a: Transform,
    cache: &mut SimplexCache,
) {
    manifold.point_count = 0;

    if capacity == 0 {
        return;
    }

    let center = transform_b_to_a.point(sphere_b.center);
    let speculative_distance = SPECULATIVE_DISTANCE;

    let center_pts = [center];
    let distance_input = DistanceInput {
        proxy_a: ShapeProxy {
            points: &hull_a.points,
            count: hull_a.vertex_count,
            radius: 0.0,
        },
        proxy_b: ShapeProxy {
            points: &center_pts,
            count: 1,
            radius: 0.0,
        },
        transform: Transform::IDENTITY,
        use_radii: false,
    };

    let radius_a = 0.0;
    let radius_b = sphere_b.radius;
    let radius = radius_a + radius_b;

    let distance_output = shape_distance(&distance_input, cache);

    if distance_output.distance > radius + speculative_distance {
        *cache = SimplexCache::empty();
        return;
    }

    if distance_output.distance > 100.0 * FLT_EPSILON {
        // Shallow penetration.
        let normal = distance_output
            .point_b
            .sub(distance_output.point_a)
            .normalize();
        let c_a = center.mul_add(
            radius_a - center.sub(distance_output.point_a).dot(normal),
            normal,
        );
        let c_b = center.mul_sub(radius_b, normal);
        let point = c_a.lerp(c_b, 0.5);

        manifold.normal = normal;
        manifold.point_count = 1;

        let pt = &mut manifold.points[0];
        pt.point = point;
        pt.separation = distance_output.distance - radius;
        pt.pair = FeaturePair::SINGLE;
    } else {
        // Deep penetration: pick the hull face the sphere center is least behind.
        let mut best_index = 0usize;
        let mut best_distance = -FLT_MAX;
        let planes = &hull_a.planes;

        for index in 0..hull_a.face_count {
            let distance = planes[index].separation(center);
            if distance > best_distance {
                best_index = index;
                best_distance = distance;
            }
        }

        let normal = planes[best_index].normal;
        let c_a = center.mul_add(
            radius_a - center.sub(distance_output.point_a).dot(normal),
            normal,
        );
        let c_b = center.mul_sub(radius_b, normal);
        let point = c_a.lerp(c_b, 0.5);

        manifold.normal = normal;
        manifold.point_count = 1;

        let pt = &mut manifold.points[0];
        pt.point = point;
        pt.separation = best_distance - radius;
        pt.pair = FeaturePair::SINGLE;
    }
}

/// b3CollideCapsules — up to two-point manifold for two capsules, in frame A.
pub fn collide_capsules(
    manifold: &mut LocalManifold,
    capacity: usize,
    capsule_a: &Capsule,
    capsule_b: &Capsule,
    transform_b_to_a: Transform,
) {
    manifold.point_count = 0;

    if capacity < 2 {
        return;
    }

    let center_a1 = capsule_a.center1;
    let center_a2 = capsule_a.center2;
    let center_b1 = transform_b_to_a.point(capsule_b.center1);
    let center_b2 = transform_b_to_a.point(capsule_b.center2);

    let radius = capsule_a.radius + capsule_b.radius;
    let max_distance = radius + SPECULATIVE_DISTANCE;

    let result = segment_distance(center_a1, center_a2, center_b1, center_b2);
    let offset = result.point2.sub(result.point1);
    let distance_squared = offset.length_sq();
    let linear_slop = LINEAR_SLOP;
    let min_distance = 0.01 * linear_slop;

    if distance_squared > max_distance * max_distance
        || distance_squared < min_distance * min_distance
    {
        return;
    }

    let segment_a = center_a2.sub(center_a1);
    let (edge_a, edge_a_len) = get_length_and_normalize(segment_a);
    if edge_a_len < MIN_CAPSULE_LENGTH {
        return;
    }

    let segment_b = center_b2.sub(center_b1);
    let (edge_b, edge_b_len) = get_length_and_normalize(segment_b);
    if edge_b_len < MIN_CAPSULE_LENGTH {
        return;
    }

    // Parallel edges: |eA x eB| = sin(alpha).
    let alpha_tol: f32 = 0.05;
    let alpha_tol_sqr = alpha_tol * alpha_tol;
    let axis = edge_a.cross(edge_b);

    if axis.length_sq() < alpha_tol_sqr {
        // Clip segment B against the side planes of segment A.
        let planes_a0 = Plane {
            normal: edge_a.neg(),
            offset: -edge_a.dot(capsule_a.center1),
        };
        let planes_a1 = Plane {
            normal: edge_a,
            offset: edge_a.dot(capsule_a.center2),
        };

        let mut vertices_b: [ClipVertex; 2] = [
            ClipVertex {
                position: center_b1,
                separation: 0.0,
                pair: make_feature_pair(SHAPE_A, 0, SHAPE_A, 0),
            },
            ClipVertex {
                position: center_b2,
                separation: 0.0,
                pair: make_feature_pair(SHAPE_A, 1, SHAPE_A, 1),
            },
        ];

        let mut point_count = clip_segment(&mut vertices_b, planes_a0);
        if point_count == 2 {
            point_count = clip_segment(&mut vertices_b, planes_a1);
        }

        if point_count == 2 {
            let closest_point1 =
                point_to_segment_distance(center_a1, center_a2, vertices_b[0].position);
            let closest_point2 =
                point_to_segment_distance(center_a1, center_a2, vertices_b[1].position);

            let distance1 = closest_point1.distance(vertices_b[0].position);
            let distance2 = closest_point2.distance(vertices_b[1].position);
            if distance1 <= radius && distance2 <= radius {
                if distance1 < min_distance || distance2 < min_distance {
                    // Avoid divide by zero.
                    return;
                }

                let normal1 = vertices_b[0]
                    .position
                    .sub(closest_point1)
                    .scale(1.0 / distance1);
                let normal2 = vertices_b[1]
                    .position
                    .sub(closest_point2)
                    .scale(1.0 / distance2);
                let normal = normal1.add(normal2).normalize();
                let radius_a = capsule_a.radius;
                let radius_b = capsule_b.radius;

                // Contact at the midpoint: 0.5 * (((vB.pos + rA*nK) + cP) - rB*n).
                let point1 = vertices_b[0]
                    .position
                    .mul_add(radius_a, normal1)
                    .add(closest_point1)
                    .mul_sub(radius_b, normal)
                    .scale(0.5);
                let point2 = vertices_b[1]
                    .position
                    .mul_add(radius_a, normal2)
                    .add(closest_point2)
                    .mul_sub(radius_b, normal)
                    .scale(0.5);

                manifold.normal = normal;
                manifold.point_count = 2;

                let pair0 = vertices_b[0].pair;
                let pair1 = vertices_b[1].pair;

                manifold.points[0].point = point1;
                manifold.points[0].separation = distance1 - radius;
                manifold.points[0].pair = pair0;
                manifold.points[1].point = point2;
                manifold.points[1].separation = distance2 - radius;
                manifold.points[1].pair = pair1;

                return;
            }
        }
    }

    let (normal, distance) = get_length_and_normalize(offset);
    // Contact at the midpoint 0.5 * (((p1 + rA*n) + p2) - rB*n).
    let point = result
        .point1
        .mul_add(capsule_a.radius, normal)
        .add(result.point2)
        .mul_sub(capsule_b.radius, normal)
        .scale(0.5);

    manifold.normal = normal;
    manifold.point_count = 1;

    let pt = &mut manifold.points[0];
    pt.point = point;
    pt.separation = distance - radius;
    pt.pair = FeaturePair::SINGLE;
}

// --- hull / capsule collision ------------------------------------------------------------------

fn build_hull_face_and_capsule_contact(
    manifold: &mut LocalManifold,
    hull_a: &HullData,
    capsule_b: &Capsule,
    transform_b_to_a: Transform,
    query: SeparatingAxis,
) -> bool {
    let planes = &hull_a.planes;

    let ref_face = query.index_a as usize;
    let ref_plane = planes[ref_face];

    let mut segment_b: [ClipVertex; 2] = [
        ClipVertex {
            position: transform_b_to_a.point(capsule_b.center1),
            separation: 0.0,
            pair: make_feature_pair(SHAPE_A, 0, SHAPE_A, 0),
        },
        ClipVertex {
            position: transform_b_to_a.point(capsule_b.center2),
            separation: 0.0,
            pair: make_feature_pair(SHAPE_A, 1, SHAPE_A, 1),
        },
    ];

    let point_count = clip_segment_to_hull_face(&mut segment_b, hull_a, ref_face);
    if point_count < 2 {
        return false;
    }

    let distance1 = ref_plane.separation(segment_b[0].position);
    let distance2 = ref_plane.separation(segment_b[1].position);
    let speculative_distance = SPECULATIVE_DISTANCE;

    if distance1 <= speculative_distance || distance2 <= speculative_distance {
        let normal = ref_plane.normal;
        let point1 = segment_b[0]
            .position
            .mul_sub(0.5 * (distance1 + capsule_b.radius), normal);
        let point2 = segment_b[1]
            .position
            .mul_sub(0.5 * (distance2 + capsule_b.radius), normal);

        manifold.normal = normal;
        manifold.point_count = 2;

        manifold.points[0].point = point1;
        manifold.points[0].separation = distance1 - capsule_b.radius;
        manifold.points[0].pair = segment_b[0].pair;
        manifold.points[1].point = point2;
        manifold.points[1].separation = distance2 - capsule_b.radius;
        manifold.points[1].pair = segment_b[1].pair;

        return true;
    }

    false
}

fn build_hull_and_capsule_edge_contact(
    manifold: &mut LocalManifold,
    capacity: usize,
    hull_a: &HullData,
    capsule_b: &Capsule,
    transform_b_to_a: Transform,
    query: SeparatingAxis,
) -> bool {
    if capacity < 1 {
        return false;
    }

    let pc = transform_b_to_a.point(capsule_b.center1);
    let qc = transform_b_to_a.point(capsule_b.center2);
    let ec = qc.sub(pc);

    let edges = &hull_a.edges;
    let points = &hull_a.points;

    let edge2 = edges[query.index_b as usize];
    let twin2 = edges[edge2.twin as usize];
    let ph = points[edge2.origin as usize];
    let qh = points[twin2.origin as usize];
    let eh = qh.sub(ph);

    let normal = query.normal;

    let result = line_distance(ph, eh, pc, ec);
    if !is_within_segments(&result) {
        // Closest point beyond end points.
        return false;
    }

    let point = result
        .point1
        .mul_sub(capsule_b.radius, normal)
        .add(result.point2)
        .scale(0.5);

    let separation = normal.dot(result.point2.sub(result.point1));

    manifold.normal = normal;
    manifold.point_count = 1;

    let pt = &mut manifold.points[0];
    pt.point = point;
    pt.separation = separation - capsule_b.radius;
    pt.pair = make_feature_pair(SHAPE_A, query.index_a as u8, SHAPE_B, query.index_b as u8);
    true
}

/// b3CollideHullAndCapsule — up to two-point manifold for a hull (A) and capsule (B), in frame A.
pub fn collide_hull_and_capsule(
    manifold: &mut LocalManifold,
    capacity: usize,
    hull_a: &HullData,
    capsule_b: &Capsule,
    transform_b_to_a: Transform,
    cache: &mut SimplexCache,
) {
    manifold.point_count = 0;

    if capacity < 2 {
        return;
    }

    let cap_pts = [capsule_b.center1, capsule_b.center2];
    let distance_input = DistanceInput {
        proxy_a: ShapeProxy {
            points: &hull_a.points,
            count: hull_a.vertex_count,
            radius: 0.0,
        },
        proxy_b: ShapeProxy {
            points: &cap_pts,
            count: 2,
            radius: 0.0,
        },
        transform: transform_b_to_a,
        use_radii: false,
    };

    let distance_output = shape_distance(&distance_input, cache);
    let speculative_distance = SPECULATIVE_DISTANCE;

    if distance_output.distance > capsule_b.radius + speculative_distance {
        *cache = SimplexCache::empty();
        return;
    }

    if distance_output.distance > 100.0 * FLT_EPSILON {
        let planes = &hull_a.planes;

        // Shallow penetration.
        let delta = distance_output.normal;
        let ref_face = hull_a.support_face(delta);
        let ref_plane = planes[ref_face];

        // Try two contact points if the closest-points difference is nearly parallel to the face.
        let k_tolerance: f32 = 0.998;
        if absf(ref_plane.normal.dot(delta)) > k_tolerance {
            let mut vertices_b: [ClipVertex; 2] = [
                ClipVertex {
                    position: transform_b_to_a.point(capsule_b.center1),
                    separation: 0.0,
                    pair: make_feature_pair(SHAPE_A, 0, SHAPE_A, 0),
                },
                ClipVertex {
                    position: transform_b_to_a.point(capsule_b.center2),
                    separation: 0.0,
                    pair: make_feature_pair(SHAPE_A, 1, SHAPE_A, 1),
                },
            ];

            let point_count = clip_segment_to_hull_face(&mut vertices_b, hull_a, ref_face);

            if point_count == 2 {
                let distance1 = ref_plane.separation(vertices_b[0].position);
                let distance2 = ref_plane.separation(vertices_b[1].position);
                if distance1 <= capsule_b.radius + speculative_distance
                    || distance2 <= capsule_b.radius + speculative_distance
                {
                    let normal = ref_plane.normal;
                    let point1 = vertices_b[0]
                        .position
                        .mul_sub(0.5 * (capsule_b.radius + distance1), normal);
                    let point2 = vertices_b[1]
                        .position
                        .mul_sub(0.5 * (capsule_b.radius + distance2), normal);

                    manifold.normal = normal;
                    manifold.point_count = 2;

                    manifold.points[0].point = point1;
                    manifold.points[0].separation = distance1 - capsule_b.radius;
                    manifold.points[0].pair = vertices_b[0].pair;
                    manifold.points[1].point = point2;
                    manifold.points[1].separation = distance2 - capsule_b.radius;
                    manifold.points[1].pair = vertices_b[1].pair;

                    return;
                }
            }
        }

        // Create contact from closest points.
        let point = distance_output
            .point_a
            .mul_sub(capsule_b.radius, delta)
            .add(distance_output.point_b)
            .scale(0.5);

        manifold.normal = delta;
        manifold.point_count = 1;

        let pt = &mut manifold.points[0];
        pt.point = point;
        pt.separation = distance_output.distance - capsule_b.radius;
        pt.pair = FeaturePair::SINGLE;
        return;
    }

    // Deep penetration.
    let face_query = query_face_direction_hull_and_capsule(hull_a, capsule_b, transform_b_to_a);
    if face_query.separation > capsule_b.radius {
        return;
    }

    let edge_query = query_edge_direction_hull_and_capsule(hull_a, capsule_b, transform_b_to_a);
    if edge_query.separation > capsule_b.radius {
        return;
    }

    // Create face contact.
    let mut face_separation = face_query.separation - capsule_b.radius;
    build_hull_face_and_capsule_contact(manifold, hull_a, capsule_b, transform_b_to_a, face_query);
    if manifold.point_count > 1 {
        face_separation = minf(manifold.points[0].separation, manifold.points[1].separation);
    }

    if edge_query.index_a == -1 {
        return;
    }

    // Create edge contact if face contact fails or edge contact is significantly better.
    let edge_separation = edge_query.separation - capsule_b.radius;
    if manifold.point_count == 0 || edge_separation > face_separation + LINEAR_SLOP {
        build_hull_and_capsule_edge_contact(
            manifold,
            capacity,
            hull_a,
            capsule_b,
            transform_b_to_a,
            edge_query,
        );
    }
}

// --- hull / hull collision ---------------------------------------------------------------------

/// b3BuildPolygon — the incident face of `hull` transformed into frame A as a clip polygon, written into
/// `out` (at least `MAX_CLIP_POINTS` slots); returns its length.
fn build_polygon(
    transform: Transform,
    hull: &HullData,
    inc_face: usize,
    ref_plane: Plane,
    out: &mut [core::mem::MaybeUninit<ClipVertex>],
) -> usize {
    let faces = &hull.faces;
    let edges = &hull.edges;
    let points = &hull.points;

    let face = faces[inc_face];
    let mut edge_index = face.edge as usize;

    let matrix = Mat3::from_quat(transform.q);
    let mut n = 0;

    loop {
        let edge = edges[edge_index];
        let next_edge_index = edge.next as usize;
        let next = edges[next_edge_index];

        let position = matrix.mul_v(points[next.origin as usize]).add(transform.p);
        out[n].write(ClipVertex {
            position,
            separation: ref_plane.separation(position),
            pair: FeaturePair {
                owner1: SHAPE_B,
                index1: edge_index as u8,
                owner2: SHAPE_B,
                index2: next_edge_index as u8,
            },
        });
        n += 1;

        edge_index = next_edge_index;
        if edge_index == face.edge as usize || n >= MAX_CLIP_POINTS {
            break;
        }
    }

    n
}

fn build_face_a_contact(
    manifold: &mut LocalManifold,
    capacity: usize,
    hull_a: &HullData,
    hull_b: &HullData,
    transform_b_to_a: Transform,
    query: SeparatingAxis,
    cache: &mut SatCache,
) -> bool {
    debug_assert_eq!(query.ty, separating_feature::FACE_AXIS_A);
    let faces_a = &hull_a.faces;
    let edges_a = &hull_a.edges;
    let planes_a = &hull_a.planes;
    let points_a = &hull_a.points;

    // Reference face.
    let ref_face = query.index_a as usize;
    let ref_plane = planes_a[ref_face];

    // Find incident face.
    let ref_normal_in_b = transform_b_to_a.q.inv_rotate(ref_plane.normal);
    let inc_face = find_incident_face(hull_b, ref_normal_in_b, query.index_b as usize);

    let mut buffer1 = [core::mem::MaybeUninit::uninit(); MAX_CLIP_POINTS];
    let mut buffer2 = [core::mem::MaybeUninit::uninit(); MAX_CLIP_POINTS];
    let mut input = &mut buffer1;
    let mut scratch = &mut buffer2;
    let mut point_count = build_polygon(transform_b_to_a, hull_b, inc_face, ref_plane, input);

    // Clip incident face against side planes of the reference face.
    let face = faces_a[ref_face];
    let mut edge_index = face.edge as usize;

    loop {
        let edge = edges_a[edge_index];
        let next_edge_index = edge.next as usize;
        let next = edges_a[next_edge_index];
        let vertex1 = points_a[edge.origin as usize];
        let vertex2 = points_a[next.origin as usize];
        let tangent = vertex2.sub(vertex1).normalize();
        let binormal = tangent.cross(ref_plane.normal);

        let clip_plane = Plane::from_normal_and_point(binormal, vertex1);

        point_count = clip_polygon(
            // build_polygon and clip_polygon initialize exactly their returned prefixes.
            unsafe { core::slice::from_raw_parts(input.as_ptr().cast(), point_count) },
            point_count,
            clip_plane,
            edge_index as u8,
            ref_plane,
            scratch,
        );
        core::mem::swap(&mut input, &mut scratch);

        if point_count < 3 {
            cache.reset();
            return false;
        }

        edge_index = next_edge_index;
        if edge_index == face.edge as usize {
            break;
        }
    }

    point_count = point_count.min(MAX_CLIP_POINTS);

    let mut min_separation = FLT_MAX;

    manifold.normal = ref_plane.normal;

    let mut reduce_points = [core::mem::MaybeUninit::uninit(); MAX_CLIP_POINTS];
    for i in 0..point_count {
        let clip_point = unsafe { input[i].assume_init() };
        // Half-way point keeps positions stable when swapping the reference face from A to B.
        reduce_points[i].write(LocalManifoldPoint {
            point: clip_point
                .position
                .mul_sub(0.5 * clip_point.separation, ref_plane.normal),
            separation: clip_point.separation,
            pair: clip_point.pair,
            triangle_index: 0,
        });
        min_separation = minf(min_separation, clip_point.separation);
    }

    if min_separation >= SPECULATIVE_DISTANCE {
        cache.reset();
        return false;
    }

    // Only the prefix written by the loop above becomes initialized point records.
    let reduce_points =
        unsafe { core::slice::from_raw_parts_mut(reduce_points.as_mut_ptr().cast(), point_count) };
    reduce_manifold_points(manifold, capacity, reduce_points, point_count);

    cache.separation = min_separation;
    cache.ty = separating_feature::FACE_AXIS_A;
    cache.index_a = (query.index_a & 0xff) as usize;
    cache.index_b = (query.index_b & 0xff) as usize;

    true
}

fn build_face_b_contact(
    manifold: &mut LocalManifold,
    capacity: usize,
    hull_a: &HullData,
    hull_b: &HullData,
    transform_b_to_a: Transform,
    query: SeparatingAxis,
    cache: &mut SatCache,
) -> bool {
    debug_assert_eq!(query.ty, separating_feature::FACE_AXIS_B);
    let transform_a_to_b = transform_b_to_a.invert();
    let touching = build_face_a_contact(
        manifold,
        capacity,
        hull_b,
        hull_a,
        transform_a_to_b,
        SeparatingAxis {
            normal: query.normal.neg(),
            index_a: query.index_b,
            index_b: query.index_a,
            ty: separating_feature::FACE_AXIS_A,
            ..query
        },
        cache,
    );
    if !touching {
        return false;
    }

    // Results are in frame B; transform them into frame A.
    let matrix = Mat3::from_quat(transform_b_to_a.q);

    // Flip normal so it points from A to B, even though B owns the reference face.
    manifold.normal = matrix.mul_v(manifold.normal).neg();
    for i in 0..manifold.point_count {
        manifold.points[i].point = matrix
            .mul_v(manifold.points[i].point)
            .add(transform_b_to_a.p);
        manifold.points[i].pair = flip_pair(manifold.points[i].pair);
    }

    cache.ty = separating_feature::FACE_AXIS_B;
    cache.index_a = (query.index_a & 0xff) as usize;
    cache.index_b = (query.index_b & 0xff) as usize;
    true
}

fn build_edge_contact(
    manifold: &mut LocalManifold,
    hull_a: &HullData,
    hull_b: &HullData,
    transform_b_to_a: Transform,
    query: SeparatingAxis,
    cache: &mut SatCache,
) -> bool {
    let edges_a = &hull_a.edges;
    let points_a = &hull_a.points;
    let edges_b = &hull_b.edges;
    let points_b = &hull_b.points;

    let edge_a = edges_a[query.index_a as usize];
    let twin_a = edges_a[edge_a.twin as usize];
    let p_a = points_a[edge_a.origin as usize];
    let q_a = points_a[twin_a.origin as usize];
    let e_a = q_a.sub(p_a);

    let edge_b = edges_b[query.index_b as usize];
    let twin_b = edges_b[edge_b.twin as usize];
    let p_b = transform_b_to_a.point(points_b[edge_b.origin as usize]);
    let q_b = transform_b_to_a.point(points_b[twin_b.origin as usize]);
    let e_b = q_b.sub(p_b);

    let normal = query.normal;
    let result = line_distance(p_a, e_a, p_b, e_b);

    if !is_within_segments(&result) {
        cache.reset();
        return false;
    }

    // This can slide off the end from caching.
    let separation = normal.dot(result.point2.sub(result.point1));
    let point = result.point1.add(result.point2).scale(0.5);

    manifold.normal = normal;
    manifold.point_count = 1;

    manifold.points[0].point = point;
    manifold.points[0].separation = separation;
    manifold.points[0].pair =
        make_feature_pair(SHAPE_A, query.index_a as u8, SHAPE_B, query.index_b as u8);

    cache.separation = separation;
    cache.ty = separating_feature::EDGE_PAIR_AXIS;
    cache.index_a = (query.index_a & 0xff) as usize;
    cache.index_b = (query.index_b & 0xff) as usize;

    true
}

#[derive(Clone, Copy)]
struct AxisQuery {
    face_a: SeparatingAxis,
    face_b: SeparatingAxis,
    edge: SeparatingAxis,
    separated: u32,
}

fn hull_aabb_center_extents(hull: &HullData) -> (Vec3, Vec3) {
    let [lower, upper] = hull.bounds;
    (lower.add(upper).scale(0.5), upper.sub(lower).scale(0.5))
}

const NV: usize = 128 + 4;
const NF: usize = 128 + 4;
const NE: usize = 128 + 4;

struct SatScratch<const N: usize> {
    values: [core::mem::MaybeUninit<f32>; N],
    initialized: usize,
}

impl<const N: usize> SatScratch<N> {
    fn new() -> Self {
        Self {
            values: [core::mem::MaybeUninit::uninit(); N],
            initialized: 0,
        }
    }
    fn set(&mut self, i: usize, value: f32) {
        assert!(i <= self.initialized);
        self.values[i].write(value);
        self.initialized = self.initialized.max(i + 1);
    }
    fn get(&self, i: usize) -> f32 {
        assert!(i < self.initialized);
        // Only the sequentially written prefix is readable.
        unsafe { self.values[i].assume_init() }
    }
    fn load(&self, i: usize) -> crate::simd::FloatW {
        assert!(i + 4 <= self.initialized);
        // No reference includes the unwritten suffix.
        unsafe {
            crate::simd::FloatW::load(core::slice::from_raw_parts(
                self.values.as_ptr().add(i).cast(),
                4,
            ))
        }
    }
}

struct HullSoa3<const N: usize> {
    x: SatScratch<N>,
    y: SatScratch<N>,
    z: SatScratch<N>,
}

impl<const N: usize> HullSoa3<N> {
    fn new() -> Self {
        Self {
            x: SatScratch::new(),
            y: SatScratch::new(),
            z: SatScratch::new(),
        }
    }
    fn set(&mut self, i: usize, v: Vec3) {
        self.x.set(i, v.x);
        self.y.set(i, v.y);
        self.z.set(i, v.z);
    }
    fn get(&self, i: usize) -> Vec3 {
        Vec3::new(self.x.get(i), self.y.get(i), self.z.get(i))
    }
    fn load(&self, i: usize) -> [crate::simd::FloatW; 3] {
        [self.x.load(i), self.y.load(i), self.z.load(i)]
    }
}

// b3Dot3W's association, x + (y + z), on every target (simd.h NEON, SSE2 and scalar).
#[inline]
fn dot_wide(a: [crate::simd::FloatW; 3], b: [crate::simd::FloatW; 3]) -> crate::simd::FloatW {
    a[0].mul(b[0]).add(a[1].mul(b[1]).add(a[2].mul(b[2])))
}

#[inline]
fn splat3(v: Vec3) -> [crate::simd::FloatW; 3] {
    use crate::simd::FloatW;
    [FloatW::splat(v.x), FloatW::splat(v.y), FloatW::splat(v.z)]
}

fn negative_transform_from_soa<const N: usize>(
    matrix: &Mat3,
    translation: Vec3,
    input: &[f32],
    point: bool,
    out: &mut HullSoa3<N>,
) {
    use crate::simd::FloatW;
    let n = input.len() / 3;
    let rows = [
        splat3(Vec3::new(matrix.cx.x, matrix.cy.x, matrix.cz.x)),
        splat3(Vec3::new(matrix.cx.y, matrix.cy.y, matrix.cz.y)),
        splat3(Vec3::new(matrix.cx.z, matrix.cy.z, matrix.cz.z)),
    ];
    let t = splat3(translation);
    for i in (0..n).step_by(4) {
        let v = [
            FloatW::load(&input[i..]),
            FloatW::load(&input[n + i..]),
            FloatW::load(&input[2 * n + i..]),
        ];
        for (axis, values) in [&mut out.x, &mut out.y, &mut out.z].into_iter().enumerate() {
            let mut value = dot_wide(rows[axis], v);
            if point {
                value = value.add(t[axis]);
            }
            for (lane, value) in value.neg().to_array().into_iter().enumerate() {
                values.set(i + lane, value);
            }
        }
    }
}

fn compute_separating_axis(
    hull_a: &HullData,
    hull_b: &HullData,
    transform_b_to_a: Transform,
    early_return: bool,
) -> AxisQuery {
    let rotation = Mat3::from_quat(transform_b_to_a.q);
    let inverse_rotation = rotation.transpose();
    let (center_b, extent_b) = hull_aabb_center_extents(hull_b);
    let mut result = AxisQuery {
        face_a: SeparatingAxis {
            normal: Vec3::ZERO,
            separation: -f32::INFINITY,
            index_a: 0,
            index_b: 0,
            ty: separating_feature::FACE_AXIS_A,
        },
        face_b: SeparatingAxis {
            normal: Vec3::ZERO,
            separation: -f32::INFINITY,
            index_a: 0,
            index_b: 0,
            ty: separating_feature::FACE_AXIS_B,
        },
        edge: SeparatingAxis {
            normal: Vec3::ZERO,
            separation: -f32::INFINITY,
            index_a: -1,
            index_b: -1,
            ty: separating_feature::EDGE_PAIR_AXIS,
        },
        separated: separating_feature::INVALID,
    };

    for i in 0..hull_a.face_count {
        let plane = hull_a.planes[i];
        let direction = inverse_rotation.mul_v(plane.normal).neg();
        let plane_separation = plane.normal.dot(transform_b_to_a.p) - plane.offset;
        let bias = direction.dot(center_b) + 1.0625 * direction.abs().dot(extent_b);
        let vertex = hull_b.support_vertex_wide(direction, bias);
        let p = hull_b.points[vertex];
        let support = direction.dot(p);
        let separation = plane_separation - support;
        if separation > result.face_a.separation {
            result.face_a = SeparatingAxis {
                normal: plane.normal,
                separation,
                index_a: i as i32,
                index_b: vertex as i32,
                ty: separating_feature::FACE_AXIS_A,
            };
            if early_return && separation > SPECULATIVE_DISTANCE {
                result.separated = separating_feature::FACE_AXIS_A;
                return result;
            }
        }
    }

    let (center_a, extent_a) = hull_aabb_center_extents(hull_a);
    for i in 0..hull_b.face_count {
        let plane = hull_b.planes[i];
        let direction = rotation.mul_v(plane.normal).neg();
        let plane_separation = direction.dot(transform_b_to_a.p) - plane.offset;
        let bias = direction.dot(center_a) + 1.0625 * direction.abs().dot(extent_a);
        let vertex = hull_a.support_vertex_wide(direction, bias);
        let support = direction.dot(hull_a.points[vertex]);
        let separation = plane_separation - support;
        if separation > result.face_b.separation {
            result.face_b = SeparatingAxis {
                normal: rotation.mul_v(plane.normal).neg(),
                separation,
                index_a: vertex as i32,
                index_b: i as i32,
                ty: separating_feature::FACE_AXIS_B,
            };
            if early_return && separation > SPECULATIVE_DISTANCE {
                result.separated = separating_feature::FACE_AXIS_B;
                return result;
            }
        }
    }

    use crate::simd::FloatW;
    let mut b_normals = HullSoa3::<NF>::new();
    let mut b_points = HullSoa3::<NV>::new();
    negative_transform_from_soa(
        &rotation,
        transform_b_to_a.p,
        &hull_b.soa_normals,
        false,
        &mut b_normals,
    );
    negative_transform_from_soa(
        &rotation,
        transform_b_to_a.p,
        &hull_b.soa_points,
        true,
        &mut b_points,
    );
    let mut b_c = HullSoa3::<NE>::new();
    let mut b_d = HullSoa3::<NE>::new();
    let mut b_v0 = HullSoa3::<NE>::new();
    let mut b_dc = HullSoa3::<NE>::new();
    let nb = hull_b.edge_count / 2;
    for j in 0..nb {
        let edge = hull_b.edges[2 * j];
        let twin = hull_b.edges[2 * j + 1];
        let v0 = b_points.get(edge.origin as usize);
        b_c.set(j, b_normals.get(edge.face as usize));
        b_d.set(j, b_normals.get(twin.face as usize));
        b_v0.set(j, v0);
        b_dc.set(j, b_points.get(twin.origin as usize).sub(v0));
    }

    let mut a_n0 = HullSoa3::<NE>::new();
    let mut a_n1 = HullSoa3::<NE>::new();
    let mut a_dir = HullSoa3::<NE>::new();
    let mut a_v0 = HullSoa3::<NE>::new();
    let mut a_tol = SatScratch::<NE>::new();
    let na = hull_a.edge_count / 2;
    let squared_tol = 0.005 * 0.005;
    for i in 0..na {
        let edge = hull_a.edges[2 * i];
        let twin = hull_a.edges[2 * i + 1];
        let v0 = hull_a.points[edge.origin as usize];
        let dir = hull_a.points[twin.origin as usize].sub(v0);
        a_n0.set(i, hull_a.planes[edge.face as usize].normal);
        a_n1.set(i, hull_a.planes[twin.face as usize].normal);
        a_dir.set(i, dir);
        a_v0.set(i, v0);
        // b3ComputeSeparatingAxis computes this scalar sum left-to-right, not b3Dot3W.
        a_tol.set(
            i,
            squared_tol * (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z),
        );
    }
    for i in na..na + 4 {
        a_n0.set(i, Vec3::ZERO);
        a_n1.set(i, Vec3::ZERO);
        a_dir.set(i, Vec3::ZERO);
        a_v0.set(i, Vec3::ZERO);
        a_tol.set(i, 0.0);
    }
    let zero = FloatW::zero();
    let eps = FloatW::splat(-0.0001);
    let inf = FloatW::splat(f32::INFINITY);
    for j in 0..nb {
        let c = splat3(b_c.get(j));
        let d = splat3(b_d.get(j));
        let dc = splat3(b_dc.get(j));
        let bv0 = splat3(b_v0.get(j));
        for i in (0..na).step_by(4) {
            let dir = a_dir.load(i);
            let cba = dot_wide(c, dir);
            let dba = dot_wide(d, dir);
            let adc = dot_wide(a_n0.load(i), dc);
            let bdc = dot_wide(a_n1.load(i), dc);
            let max_cd = cba.mul(cba).max(dba.mul(dba));
            let mask = cba
                .mul(dba)
                .less_than(eps)
                .and(adc.mul(bdc).less_than(eps))
                .and(cba.mul(bdc).less_than(eps))
                .and(max_cd.greater_than(a_tol.load(i)));
            if !mask.any_true() {
                continue;
            }
            let t = zero.sub(cba).div(dba.sub(cba));
            let mut normal = core::array::from_fn::<_, 3, _>(|k| c[k].mul_add(t, d[k].sub(c[k])));
            let inv = FloatW::splat(1.0).div(dot_wide(normal, normal).sqrt());
            normal = normal.map(|n| n.mul(inv));
            let av0 = a_v0.load(i);
            let support = dot_wide(core::array::from_fn(|k| av0[k].add(bv0[k])), normal);
            let separation = zero.sub(FloatW::blend(inf, support, mask));
            if !separation
                .greater_than(FloatW::splat(result.edge.separation))
                .any_true()
            {
                continue;
            }
            let s = separation.to_array();
            let n = normal.map(|v| v.to_array());
            // Lane order preserves Box3D's ties and first early return; zero tail fails the Gauss mask.
            for lane in 0..4 {
                if s[lane] > result.edge.separation {
                    result.edge = SeparatingAxis {
                        normal: Vec3::new(n[0][lane], n[1][lane], n[2][lane]),
                        separation: s[lane],
                        index_a: (2 * (i + lane)) as i32,
                        index_b: (2 * j) as i32,
                        ty: separating_feature::EDGE_PAIR_AXIS,
                    };
                    if early_return && s[lane] > SPECULATIVE_DISTANCE {
                        result.separated = separating_feature::EDGE_PAIR_AXIS;
                        return result;
                    }
                }
            }
        }
    }
    result
}

/// b3CollideHulls — up to four-point manifold for two convex hulls, in frame A, with SAT cache.
pub fn collide_hulls(
    manifold: &mut LocalManifold,
    capacity: usize,
    hull_a: &HullData,
    hull_b: &HullData,
    transform_b_to_a: Transform,
    cache: &mut SatCache,
) {
    manifold.point_count = 0;

    if capacity < 4 {
        return;
    }

    let speculative_distance = SPECULATIVE_DISTANCE;
    let linear_slop = LINEAR_SLOP;
    let edges_a = &hull_a.edges;
    let planes_a = &hull_a.planes;
    let points_a = &hull_a.points;
    let edges_b = &hull_b.edges;
    let planes_b = &hull_b.planes;
    let points_b = &hull_b.points;

    cache.hit = 0;

    // Attempt to use the cache to speed up collision.
    match cache.ty {
        separating_feature::INVALID => {}

        separating_feature::FACE_AXIS_A => {
            let pl = planes_a[cache.index_a];
            let search_direction_in_b = transform_b_to_a.q.inv_rotate(pl.normal).neg();
            let vertex_index = hull_b.support_vertex(search_direction_in_b);
            let support = transform_b_to_a.point(points_b[vertex_index]);
            let separation = pl.separation(support);

            if separation >= speculative_distance {
                cache.hit = 1;
                return;
            }

            let face_query = SeparatingAxis {
                normal: pl.normal,
                separation: 0.0,
                index_a: cache.index_a as i32,
                index_b: vertex_index as i32,
                ty: separating_feature::FACE_AXIS_A,
            };
            let mut local_cache = SatCache::empty();
            let touching = build_face_a_contact(
                manifold,
                capacity,
                hull_a,
                hull_b,
                transform_b_to_a,
                face_query,
                &mut local_cache,
            );
            if touching && absf(cache.separation - local_cache.separation) < linear_slop {
                cache.hit = 1;
                return;
            }
        }

        separating_feature::FACE_AXIS_B => {
            let pl = planes_b[cache.index_b];
            let search_direction_in_a = transform_b_to_a.q.rotate(pl.normal).neg();
            let vertex_index = hull_a.support_vertex(search_direction_in_a);
            let support = transform_b_to_a.inv_point(points_a[vertex_index]);
            let separation = pl.separation(support);

            if separation >= speculative_distance {
                cache.hit = 1;
                return;
            }

            let face_query = SeparatingAxis {
                normal: pl.normal.neg(),
                separation: 0.0,
                index_a: vertex_index as i32,
                index_b: cache.index_b as i32,
                ty: separating_feature::FACE_AXIS_B,
            };
            let mut local_cache = SatCache::empty();
            let touching = build_face_b_contact(
                manifold,
                capacity,
                hull_a,
                hull_b,
                transform_b_to_a,
                face_query,
                &mut local_cache,
            );
            if touching && absf(cache.separation - local_cache.separation) < linear_slop {
                cache.hit = 1;
                return;
            }
        }

        separating_feature::EDGE_PAIR_AXIS => {
            let index1 = cache.index_a;
            let edge1 = edges_a[index1];
            let twin1 = edges_a[index1 + 1];

            let p1 = points_a[edge1.origin as usize];
            let q1 = points_a[twin1.origin as usize];
            let e1 = q1.sub(p1);

            let u1 = planes_a[edge1.face as usize].normal;
            let v1 = planes_a[twin1.face as usize].normal;

            let index2 = cache.index_b;
            let edge2 = edges_b[index2];
            let twin2 = edges_b[index2 + 1];

            let p2 = transform_b_to_a.point(points_b[edge2.origin as usize]);
            let q2 = transform_b_to_a.point(points_b[twin2.origin as usize]);
            let e2 = q2.sub(p2);

            let u2 = transform_b_to_a
                .q
                .rotate(planes_b[edge2.face as usize].normal);
            let v2 = transform_b_to_a
                .q
                .rotate(planes_b[twin2.face as usize].normal);

            let is_mink = is_minkowski_face(u1, v1, e1, u2.neg(), v2.neg(), e2);
            if is_mink {
                let cba = u2.dot(e1);
                let dba = v2.dot(e1);
                {
                    let squared_tolerance = 0.005f32 * 0.005;
                    if maxf(cba * cba, dba * dba) >= squared_tolerance * e1.length_sq() {
                        let t = cba / (cba - dba);
                        let normal = u2.lerp(v2, t).normalize();
                        let separation = normal.dot(q1.sub(q2));
                        if separation > speculative_distance {
                            // Box3D counts a cached edge pair still separated as a hit and keeps the cache.
                            cache.hit = 1;
                            return;
                        }
                        let edge_query = SeparatingAxis {
                            normal: normal.neg(),
                            index_a: cache.index_a as i32,
                            index_b: cache.index_b as i32,
                            ty: separating_feature::EDGE_PAIR_AXIS,
                            separation,
                        };
                        let mut local_cache = SatCache::empty();
                        let touching = build_edge_contact(
                            manifold,
                            hull_a,
                            hull_b,
                            transform_b_to_a,
                            edge_query,
                            &mut local_cache,
                        );
                        if touching && absf(cache.separation - local_cache.separation) < linear_slop
                        {
                            cache.hit = 1;
                            return;
                        }
                    }
                }
            }
        }

        // Manual axes are for testing.
        separating_feature::MANUAL_FACE_AXIS_A => {
            let face_query_a =
                compute_separating_axis(hull_a, hull_b, transform_b_to_a, false).face_a;
            build_face_a_contact(
                manifold,
                capacity,
                hull_a,
                hull_b,
                transform_b_to_a,
                face_query_a,
                cache,
            );
            return;
        }

        separating_feature::MANUAL_FACE_AXIS_B => {
            let face_query_b =
                compute_separating_axis(hull_a, hull_b, transform_b_to_a, false).face_b;
            build_face_b_contact(
                manifold,
                capacity,
                hull_a,
                hull_b,
                transform_b_to_a,
                face_query_b,
                cache,
            );
            return;
        }

        separating_feature::MANUAL_EDGE_PAIR_AXIS => {
            let edge_query = compute_separating_axis(hull_a, hull_b, transform_b_to_a, false).edge;
            if edge_query.index_a != -1 {
                build_edge_contact(
                    manifold,
                    hull_a,
                    hull_b,
                    transform_b_to_a,
                    edge_query,
                    cache,
                );
            }
            return;
        }

        _ => {}
    }

    manifold.point_count = 0;
    cache.reset();

    let axis_query = compute_separating_axis(hull_a, hull_b, transform_b_to_a, true);
    if axis_query.separated != separating_feature::INVALID {
        cache.ty = axis_query.separated;
        match axis_query.separated {
            separating_feature::FACE_AXIS_A => {
                cache.separation = axis_query.face_a.separation;
                cache.index_a = (axis_query.face_a.index_a & 0xff) as usize;
                cache.index_b = (axis_query.face_a.index_b & 0xff) as usize;
            }
            separating_feature::FACE_AXIS_B => {
                cache.separation = axis_query.face_b.separation;
                cache.index_a = (axis_query.face_b.index_a & 0xff) as usize;
                cache.index_b = (axis_query.face_b.index_b & 0xff) as usize;
            }
            _ => {
                cache.separation = axis_query.edge.separation;
                cache.index_a = (axis_query.edge.index_a & 0xff) as usize;
                cache.index_b = (axis_query.edge.index_b & 0xff) as usize;
            }
        }
        return;
    }

    let face_query = if axis_query.face_a.separation > axis_query.face_b.separation {
        axis_query.face_a
    } else {
        // The active target chooses face B on an exact tie.
        axis_query.face_b
    };
    if axis_query.face_a.separation > axis_query.face_b.separation {
        build_face_a_contact(
            manifold,
            capacity,
            hull_a,
            hull_b,
            transform_b_to_a,
            face_query,
            cache,
        );
    } else {
        build_face_b_contact(
            manifold,
            capacity,
            hull_a,
            hull_b,
            transform_b_to_a,
            face_query,
            cache,
        );
    }

    if axis_query.edge.index_a == -1 {
        return;
    }

    let clipped_face_separation = cache.separation;
    if manifold.point_count == 0
        || axis_query.edge.separation > clipped_face_separation + linear_slop
    {
        let mut edge_points = [LocalManifoldPoint::ZERO; 32];
        let mut edge_manifold = LocalManifold::new(&mut edge_points);
        let mut edge_cache = SatCache::empty();
        build_edge_contact(
            &mut edge_manifold,
            hull_a,
            hull_b,
            transform_b_to_a,
            axis_query.edge,
            &mut edge_cache,
        );
        if edge_manifold.point_count == 1 {
            manifold.normal = edge_manifold.normal;
            manifold.triangle_normal = edge_manifold.triangle_normal;
            manifold.triangle_index = edge_manifold.triangle_index;
            manifold.vertex_indices = edge_manifold.vertex_indices;
            manifold.triangle_flags = edge_manifold.triangle_flags;
            manifold.point_count = edge_manifold.point_count;
            manifold.feature = edge_manifold.feature;
            manifold.squared_distance = edge_manifold.squared_distance;
            manifold.points[0] = edge_manifold.points[0];
            *cache = edge_cache;
        }
    }
}

// S3 — cross-language constant parity table (reference mechanism).
//
// Every non-exact float literal in `crates/physics/src/**` that reaches f32 arithmetic, checked against its
// C reference twin in the Box3D source. A row carries an assertion exactly when its subject is
// a readable source item (a module-level `const`/`static`); function-local `let`s and inline
// literals cannot be read from a test module and are marked "not assertable" in the table with
// the value verified by hand against C. Exactness (k/2^n) is an annotation on the row, never an
// exclusion criterion — exact literals like POSITION_SLEEP_FACTOR (0.5) and FLT_EPSILON (2^-23)
// are asserted when they are readable source items. No Rust literal diverges from its C
// reference — Rust f32 literals are already f32 (no double-rounding, unlike JS where `0.01` is
// f64 until `Math.fround`). The class of bugs S2 fixed in TypeScript is absent here by literal
// choice and design boundary, not by construction: `math.rs`'s `rint_even`/`remainderf` is a
// live f64 path (reproducing libm), and a non-exact literal there reproduces rule 1a — it does
// not today only because every literal in that path is 0.5/1.0/2.0/0.0. That literal choice
// inside `math.rs`'s f64 path is a static fact no trigger and no sweep watches —
// No TypeScript scan reaches Rust. The triggers below
// watch the *design boundary moving*, not that static fact: trigger (a) fires if a new f64 path
// appears outside `math.rs`; trigger (b) fires if overlap/separation symbols enter Rust code.
// Reopen if either trigger moves (run from ; `grep -v ':[0-9]*:\s*//'` strips comment-only
// lines so the trigger cannot match its own documentation or this table's prose):
//   (a) grep -rn 'f64' crates/physics/src/*.rs | grep -v ':[0-9]*:\s*//' | grep -v 'math.rs'
//   (b) grep -rnE 'SeparationFunction|separation_function|make_separation|overlap_capsule|overlap_hull|overlap_sphere|shape_overlap|test_overlap|OVERLAP_SLOP|kToleranceSquared|k_tolerance_squared' crates/physics/src/*.rs | grep -v ':[0-9]*:\s*//'
// Both read 0 today. Probe before trusting a zero: drop `grep -v 'math.rs'` from (a) — the
// f64 hits in math.rs's rint_even/remainderf confirm the pipeline finds code symbols; add
// `fat_overlap` to (b)'s alternation — the AABB hits in arena.rs confirm the same.
//
// Assertions for module-level consts in other files live in that file's own `c_parity` module
// (appended at end-of-file), reading `super::THE_CONST` directly. This module holds only the
// constants defined in `manifold.rs` itself plus the `pub const`s in `math.rs`.
//
// Enumeration command (run from `crates/physics/src/`):
//   for f in *.rs; do sed '/#\[cfg(test)\]/,$d' "$f" | grep -v '^\s*//' | grep -oE '[0-9]+\.?[0-9]*' | sort -u; done | sort -u
// Excluded as noise: comment-only lines, test-module code, integer literals in array sizes /
// struct field indices. Exact k/2^n literals are annotated in the table, not excluded from it.
//
// Legend (per-row, not one blanket claim):
//   †g = function-local `let` or inline literal (no test can read it directly), but its enclosing
//        function IS gated bit-exactly by `tests/math_gold.rs` against C-minted gold vectors:
//        atan2 half-pi/pi consts (math.rs) — `atan2_sweep` (1930 rows); arbitrary_perp a/b
//        (math.rs) — `vec_quat_matrix_transform_cases` dispatch "arbitraryPerp".
//   †r = function-local `let` or inline literal (no test can read it directly), and its enclosing
//        function IS reached by a `manifold_gold.rs` scene (manifold output asserted bit-exactly),
//        but the literal itself is not directly asserted:
//        min_distance, k_tolerance (manifold), bias, alpha_tol — `collide_capsules`/
//        `edge_edge_separation`/`reduce_manifold_points` via `capsules_bit_exact`,
//        `hull_capsule_bit_exact`, `hulls_bit_exact`;
//        k_tolerance (hull-caps), k_rel_edge_tol — `collide_hull_and_capsule`/`collide_hulls`
//        via `hull_capsule_bit_exact`, `hulls_bit_exact`.
//   No † row is category (c) (not reached) — every enclosing function is exercised by a gold scene.
//
// ┌────────────────────────────┬───────────────────────────────┬───────────────────────────────┬──────────────────────────────────────────────┐
// │ Constant                   │ TS (value, f32 bits)           │ Rust (value, f32 bits)        │ C reference (value, file:line symbol)         │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ LINEAR_SLOP                │ 0.005, 0x3BA3D70A             │ 0.005, 0x3BA3D70A             │ 0.005f * b3GetLengthUnitsPerMeter()          │
// │ manifold.rs:18             │ constants.ts (LINEAR_SLOP),   │ manifold.rs:18               │ constants.h:53 (B3_LINEAR_SLOP)              │
// │                            │ manifold.ts (LINEAR_SLOP)     │                              │ core.c:39 (b3_lengthUnitsPerMeter = 1.0f)    │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ SPECULATIVE_DISTANCE       │ 0.02, 0x3CA3D70A              │ 0.02, 0x3CA3D70A              │ 4.0f * B3_LINEAR_SLOP                        │
// │ manifold.rs:19             │ constants.ts (SPECULATIVE_DISTANCE),│ manifold.rs:19               │ constants.h:73 (B3_SPECULATIVE_DISTANCE)      │
// │                            │ manifold.ts (SPECULATIVE_DISTANCE)│                           │                                              │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ SPECULATIVE_DISTANCE       │ 0.02, 0x3CA3D70A              │ 0.02, 0x3CA3D70A              │ 4.0f * B3_LINEAR_SLOP                        │
// │ finalize.rs:47             │ constants.ts (SPECULATIVE_DISTANCE)│ finalize.rs:47               │ constants.h:73 (B3_SPECULATIVE_DISTANCE)      │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ MIN_CAPSULE_LENGTH         │ 0.005, 0x3BA3D70A             │ 0.005, 0x3BA3D70A             │ B3_LINEAR_SLOP                               │
// │ manifold.rs:20             │ manifold.ts (MIN_CAPSULE_LENGTH)│ manifold.rs:20               │ constants.h:55 (B3_MIN_CAPSULE_LENGTH)       │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ min_distance (0.01*slop) †r│ 5e-5, 0x3851B717              │ not assertable (fn-local)     │ 0.01f * B3_LINEAR_SLOP                        │
// │ manifold.rs:982            │ manifold.ts (minDistance)     │ manifold.rs:982               │ convex_manifold.c:691 (minDistance)           │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ k_tolerance (manifold)  †r │ 0.005, 0x3BA3D70A             │ not assertable (fn-local)     │ 0.005f                                        │
// │ manifold.rs:212            │ manifold.ts (kTolerance)      │ manifold.rs:212              │ manifold.c:26 (kTolerance)                    │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ bias (clip)             †r │ 0.95, 0x3F733333              │ not assertable (fn-local)     │ 0.95f                                         │
// │ manifold.rs:648            │ manifold.ts (bias)            │ manifold.rs:648              │ convex_manifold.c:338 (bias)                  │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ alpha_tol                †r │ 0.05, 0x3D4CCCCD             │ not assertable (fn-local)     │ 0.05f                                         │
// │ manifold.rs:1003           │ manifold.ts (alphaTol)        │ manifold.rs:1003             │ convex_manifold.c:716 (alphaTol)              │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ k_tolerance (hull-caps) †r │ 0.998, 0x3F7F7CEE            │ not assertable (fn-local)     │ 0.998f                                        │
// │ manifold.rs:1309           │ manifold.ts (kTolerance)      │ manifold.rs:1309             │ convex_manifold.c:974 (kTolerance)            │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ k_rel_edge_tol (×2)     †r │ 0.9, 0x3F666666              │ not assertable (fn-local)     │ 0.90f                                         │
// │ manifold.rs:1402,1929      │ manifold.ts (kRelEdgeTolerance)│ manifold.rs:1402,1929        │ convex_manifold.c:1062,1572 (kRelEdgeTol)     │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ PI                         │ 3.1415927, 0x40490FDB        │ 3.1415927, 0x40490FDB         │ 3.14159265359f                                │
// │ math.rs:19                 │ math.ts (PI)                  │ math.rs:19                   │ math_functions.h:21 (B3_PI)                   │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ FLT_EPSILON                │ 1.1920929e-7, 0x34000000     │ 1.1920929e-7, 0x34000000      │ FLT_EPSILON (<float.h>, exact 2^-23)          │
// │ math.rs:21                 │ math.ts (FLT_EPSILON)         │ math.rs:21                   │ (C standard library)                          │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ FLT_MIN                    │ 1.1754944e-38, 0x00800000   │ 1.1754944e-38, 0x00800000     │ FLT_MIN (<float.h>, exact 2^-126)             │
// │ math.rs:22                 │ math.ts (FLT_MIN)             │ math.rs:22                   │ (C standard library)                          │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ ATAN_P0                    │ 0.024840284, 0x3CCB7DDA     │ 0.024840284, 0x3CCB7DDA       │ 0.024840285f                                  │
// │ math.rs:112                │ math.ts (ATAN_P0)             │ math.rs:112                  │ math_functions.c:188 (atan coeff)             │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ ATAN_P1                    │ 0.18681417, 0x3E3F4C37      │ 0.18681417, 0x3E3F4C37        │ 0.18681418f                                   │
// │ math.rs:113                │ math.ts (ATAN_P1)             │ math.rs:113                  │ math_functions.c:188 (atan coeff)             │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ ATAN_P2                    │ -0.09409795, 0xBDC0B66D     │ -0.09409795, 0xBDC0B66D       │ -0.094097948f                                 │
// │ math.rs:114                │ math.ts (ATAN_P2)             │ math.rs:114                  │ math_functions.c:189 (atan coeff)             │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ ATAN_P3                    │ -0.33213073, 0xBEAA0D0A     │ -0.33213073, 0xBEAA0D0A       │ -0.33213072f                                  │
// │ math.rs:115                │ math.ts (ATAN_P3)             │ math.rs:115                  │ math_functions.c:189 (atan coeff)             │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ atan2 half-pi const     †g │ 1.5707964, 0x3FC90FDB       │ not assertable (fn-local)     │ 1.57079637f                                   │
// │ math.rs:138                │ math.ts (atan2 inline 1.57079637)│ math.rs:138                  │ math_functions.c:196 (half-pi const)          │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ atan2 pi const          †g │ 3.1415927, 0x40490FDB       │ not assertable (fn-local)     │ 3.14159274f                                   │
// │ math.rs:141                │ math.ts (atan2 inline 3.14159274)│ math.rs:141                  │ math_functions.c:201 (pi const)               │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ arbitrary_perp a        †g │ 0.67, 0x3F2B851F            │ not assertable (fn-local)     │ 0.67f                                         │
// │ math.rs:350                │ math.ts (arbitraryPerp a)     │ math.rs:350                  │ math_internal.h:147 (a)                       │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ arbitrary_perp b        †g │ -0.42, 0xBED70A3D           │ not assertable (fn-local)     │ -0.42f                                        │
// │ math.rs:351                │ math.ts (arbitraryPerp b)     │ math.rs:351                  │ math_internal.h:148 (b)                       │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ MAX_ROTATION               │ 0.7853982, 0x3F490FDB       │ 0.7853982, 0x3F490FDB         │ 0.25f * B3_PI                                 │
// │ integrate.rs:19            │ absent (kernel-only phase)    │ integrate.rs:19              │ constants.h:70 (B3_MAX_ROTATION)              │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ RECYCLE_ANGULAR_DISTANCE   │ 0.99240386, 0x3F7E0E2E      │ 0.99240386, 0x3F7E0E2E        │ 0.99240388f                                   │
// │ recycle.rs:24              │ constants.ts (CONTACT_RECYCLE_ANGULAR_DISTANCE)│ recycle.rs:24                │ constants.h:86 (B3_CONTACT_RECYCLE_ANG_DIST)  │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ POSITION_SLEEP_FACTOR      │ 0.5, 0x3F000000             │ 0.5, 0x3F000000              │ 0.5f (exact k/2^n)                            │
// │ finalize.rs:25             │ absent (kernel-only phase)    │ finalize.rs:25               │ solver.c:715 (positionSleepFactor)            │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ SAFETY_FACTOR              │ 0.5, 0x3F000000             │ 0.5, 0x3F000000              │ 0.5f (exact k/2^n)                            │
// │ finalize.rs:30             │ solver.ts (safetyFactor)      │ finalize.rs:30               │ solver.c:747 (safetyFactor)                   │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ OVERLAP_SLOP               │ 0.00050000002, 0x3A03126F   │ *absent*                      │ 0.1f * B3_LINEAR_SLOP                         │
// │ (TS constants.ts:23)       │ constants.ts (OVERLAP_SLOP)   │ (see below)                  │ constants.h:60 (B3_OVERLAP_SLOP)              │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ kToleranceSquared (0.05²)  │ 0.0025000002, 0x3B23D70B    │ not asserted                  │ 0.05f * 0.05f                                 │
// │ (TS distance.ts:1154)      │ distance.ts (kToleranceSquared)│ toi.rs (inline, see below)    │ distance.c:1307 (kToleranceSquared)           │
// ├────────────────────────────┼───────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────┤
// │ kToleranceSquared (0.005²) │ 2.5e-5, 0x37D1B717          │ not asserted                  │ 0.005f * 0.005f                               │
// │ (TS distance.ts:1249)      │ distance.ts (kToleranceSquared)│ toi.rs (inline, see below)    │ distance.c:1436 (kToleranceSquared)           │
// └────────────────────────────┴───────────────────────────────┴───────────────────────────────┴──────────────────────────────────────────────┘
//
// Reconciliation (ungated prose, not a checked claim): 15 assertions + 13 rows without assertions = 28 rows.

// Other constants — reachability readings:
//
// OVERLAP_SLOP (C B3_OVERLAP_SLOP = 0.1f * B3_LINEAR_SLOP, constants.h:60, bits 0x3A03126F):
//   Used in C overlap-query predicates: b3OverlapCapsule (capsule.c:84), b3OverlapHull
//   (hull.c:2439), b3OverlapSphere (sphere.c:47). Each calls b3ShapeDistance and compares the
//   result to B3_OVERLAP_SLOP. The Rust kernel does not port these overlap-query predicates —
//   grep for 'shape_overlap|test_overlap|fn overlap' in crates/physics/src/** returns nothing. The
//   kernel boundary rule (physics.md) puts graph/world mutation, contact events, and the public
//   API on the TS side; overlap queries are public-API predicates. No Rust code path needs this
//   constant.
//
// kToleranceSquared (C 0.05f * 0.05f at distance.c:1307, bits 0x3B23D70B; 0.005f * 0.005f at
//   distance.c:1436, bits 0x37D1B717): Kernel toi.rs uses these expressions in Function::new for the
//   conservative-advancement separation function. This module's assertions do not cover them.

#[cfg(test)]
mod c_parity {
    // Append-only at EOF: Rust bakes panic `file:line` into the data section, so moving this
    // module up-file silently obligates an out-of-scope wasm rebuild. New assertions go at the
    // bottom of this module, never above existing ones.
    use crate::math;

    // ── manifold.rs constants ──

    #[test]
    fn linear_slop() {
        // C: B3_LINEAR_SLOP = 0.005f * b3GetLengthUnitsPerMeter() (constants.h:53)
        // b3GetLengthUnitsPerMeter() defaults to 1.0f (core.c:39)
        assert_eq!(super::LINEAR_SLOP.to_bits(), (0.005f32 * 1.0f32).to_bits());
    }

    #[test]
    fn speculative_distance_manifold() {
        // C: B3_SPECULATIVE_DISTANCE = 4.0f * B3_LINEAR_SLOP (constants.h:73)
        // B3_LINEAR_SLOP = 0.005f * 1.0f, so 4.0f * (0.005f * 1.0f)
        assert_eq!(
            super::SPECULATIVE_DISTANCE.to_bits(),
            (4.0f32 * (0.005f32 * 1.0f32)).to_bits()
        );
    }

    #[test]
    fn min_capsule_length() {
        // C: B3_MIN_CAPSULE_LENGTH = B3_LINEAR_SLOP (constants.h:55)
        assert_eq!(
            super::MIN_CAPSULE_LENGTH.to_bits(),
            (0.005f32 * 1.0f32).to_bits()
        );
    }

    // ── math.rs pub consts (readable from this module via `math::`) ──

    #[test]
    fn pi() {
        // C: B3_PI = 3.14159265359f (math_functions.h:21)
        // Rust: PI = 3.141_592_653_59 (math.rs:19)
        assert_eq!(math::PI.to_bits(), 3.14159265359f32.to_bits());
    }

    #[test]
    fn flt_epsilon() {
        // C: FLT_EPSILON from <float.h> = 1.1920928955078125e-7 (2^-23)
        // Rust: FLT_EPSILON = 1.192_092_895_507_812_5e-7 (math.rs:21)
        assert_eq!(
            math::FLT_EPSILON.to_bits(),
            1.1920928955078125e-7f32.to_bits()
        );
    }

    #[test]
    fn flt_min() {
        // C: FLT_MIN from <float.h> = 1.1754943508222875e-38 (2^-126)
        // Rust: FLT_MIN = 1.175_494_350_822_287_5e-38 (math.rs:22)
        assert_eq!(math::FLT_MIN.to_bits(), 1.1754943508222875e-38f32.to_bits());
    }
}
