//! Convex narrowphase manifold generation, ported op-for-op from box3d's `manifold.c` (shared clip/
//! query helpers) and `convex_manifold.c` (the six sphere/capsule/hull pair functions) (Erin Catto,
//! MIT) via the tumble.js TS port (`src/manifold.ts`). Results are in shape A's local frame;
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
#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

/// Capsule primitive (b3Capsule).
#[derive(Clone, Copy)]
pub struct Capsule {
    pub center1: Vec3,
    pub center2: Vec3,
    pub radius: f32,
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
    const SINGLE: FeaturePair = FeaturePair {
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
    const ZERO: LocalManifoldPoint = LocalManifoldPoint {
        point: Vec3::ZERO,
        separation: 0.0,
        pair: FeaturePair::SINGLE,
        triangle_index: 0,
    };
}

/// A local manifold with a fixed 32-slot point buffer (b3LocalManifold, convex subset — box3d's
/// `B3_MAX_LOCAL_MANIFOLD_POINTS`). The convex pair functions write into `points[0..point_count]` in
/// shape A's frame; a face-face clip on a high-vertex hull can exceed 8 points before the reduce to 4,
/// so the buffer must match box3d's capacity or the manifold truncates differently (fixture divergence).
pub struct LocalManifold {
    pub normal: Vec3,
    pub points: [LocalManifoldPoint; 32],
    pub point_count: usize,
}

impl LocalManifold {
    pub fn new() -> LocalManifold {
        LocalManifold {
            normal: Vec3::ZERO,
            points: [LocalManifoldPoint::ZERO; 32],
            point_count: 0,
        }
    }
}

impl Default for LocalManifold {
    fn default() -> Self {
        LocalManifold::new()
    }
}

/// Result of a face-direction SAT query (b3FaceQuery).
#[derive(Clone, Copy)]
struct FaceQuery {
    separation: f32,
    face_index: usize,
    vertex_index: usize,
}

/// Result of an edge-direction SAT query (b3EdgeQuery).
#[derive(Clone, Copy)]
struct EdgeQuery {
    separation: f32,
    index_a: i32,
    index_b: i32,
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
struct ClipVertex {
    position: Vec3,
    separation: f32,
    pair: FeaturePair,
}

impl ClipVertex {
    const ZERO: ClipVertex = ClipVertex {
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
fn flip_pair(pair: FeaturePair) -> FeaturePair {
    FeaturePair {
        owner1: 1 - pair.owner2,
        index1: pair.index2,
        owner2: 1 - pair.owner1,
        index2: pair.index1,
    }
}

/// b3EdgeEdgeSeparation — separation along the cross product of two edges, oriented outward.
fn edge_edge_separation(p1: Vec3, e1: Vec3, c1: Vec3, p2: Vec3, e2: Vec3, c2: Vec3) -> f32 {
    let u = e1.cross(e2);
    let length = u.length();

    // Skip near-parallel edges: |e1 x e2| = sin(alpha) * |e1| * |e2|.
    let k_tolerance: f32 = 0.005;
    if length < k_tolerance * (e1.length_sq() * e2.length_sq()).sqrt() {
        return -FLT_MAX;
    }
    if length * length < 1000.0 * FLT_MIN {
        return -FLT_MAX;
    }

    let mut n = u.scale(1.0 / length);

    // Orient n away from the shape with the most significant sign value.
    let sign1 = n.dot(p1.sub(c1));
    let sign2 = n.dot(p2.sub(c2));
    if absf(sign1) > absf(sign2) {
        if sign1 < 0.0 {
            n = n.neg();
        }
    } else if sign2 > 0.0 {
        n = n.neg();
    }

    n.dot(p2.sub(p1))
}

/// b3FindIncidentFace — the face on `hull` most anti-parallel to `ref_normal`.
fn find_incident_face(hull: &HullData, ref_normal: Vec3, vertex_index: usize) -> usize {
    let edges = &hull.edges;
    let planes = &hull.planes;
    let points = &hull.points;

    let mut min_edge_index = 0usize;
    let mut min_edge_projection = FLT_MAX;

    let vertex = hull.vertices[vertex_index];
    let mut edge_index = vertex.edge;
    let mut edge = edges[edge_index];
    let edge_origin = points[edge.origin];

    loop {
        let twin = edges[edge.twin];
        let twin_origin = points[twin.origin];
        let axis = twin_origin.sub(edge_origin).normalize();
        let edge_projection = absf(axis.dot(ref_normal));
        if edge_projection < min_edge_projection {
            min_edge_index = edge_index;
            min_edge_projection = edge_projection;
        }
        edge_index = twin.next;
        edge = edges[edge_index];
        if edge_index == vertex.edge {
            break;
        }
    }

    let min_edge = edges[min_edge_index];
    let min_face_index1 = min_edge.face;
    let min_plane1 = planes[min_face_index1];
    let min_twin = edges[min_edge.twin];
    let min_face_index2 = min_twin.face;
    let min_plane2 = planes[min_face_index2];

    if min_plane1.normal.dot(ref_normal) < min_plane2.normal.dot(ref_normal) {
        min_face_index1
    } else {
        min_face_index2
    }
}

/// b3ClipPolygon — Sutherland-Hodgman clip of `polygon` against `clip_plane`, writing the clipped polygon
/// into `out` and returning its length. Intersection points re-own their cut edge to `edge` on shape A.
/// `out` must hold at least `count + 1` slots (the clip grows the polygon by at most one vertex).
fn clip_polygon(
    polygon: &[ClipVertex],
    count: usize,
    clip_plane: Plane,
    edge: u8,
    ref_plane: Plane,
    out: &mut [ClipVertex],
) -> usize {
    let mut n = 0;

    let mut vertex1 = polygon[count - 1];
    let mut distance1 = clip_plane.separation(vertex1.position);

    for index in 0..count {
        let vertex2 = polygon[index];
        let distance2 = clip_plane.separation(vertex2.position);

        if distance1 <= 0.0 && distance2 <= 0.0 {
            // Both behind: keep vertex2.
            out[n] = vertex2;
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
            out[n] = ClipVertex {
                position,
                separation: ref_plane.separation(position),
                pair,
            };
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
            out[n] = ClipVertex {
                position,
                separation: ref_plane.separation(position),
                pair,
            };
            n += 1;
            out[n] = vertex2;
            n += 1;
        }

        vertex1 = vertex2;
        distance1 = distance2;
    }

    n
}

// --- convex_manifold.c: Gauss-map / clip helpers -----------------------------------------------

fn is_minkowski_face_isolated(a: Vec3, b: Vec3, n: Vec3) -> bool {
    let an = a.dot(n);
    let bn = b.dot(n);
    an * bn <= 0.0
}

fn is_minkowski_face(a: Vec3, b: Vec3, bxa: Vec3, c: Vec3, d: Vec3, dxc: Vec3) -> bool {
    let cba = c.dot(bxa);
    let dba = d.dot(bxa);
    let adc = a.dot(dxc);
    let bdc = b.dot(dxc);
    cba * dba < 0.0 && adc * bdc < 0.0 && cba * bdc > 0.0
}

/// b3ClipSegment — clip a 2-vertex segment against `pl`, in place. Returns the vertex count.
fn clip_segment(segment: &mut [ClipVertex; 3], pl: Plane) -> usize {
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
        segment[vertex_count] = ClipVertex {
            position,
            separation: 0.0,
            pair: src.pair,
        };
        vertex_count += 1;
    }

    vertex_count
}

/// b3ClipSegmentToHullFace — clip a segment against every side plane of the reference face.
fn clip_segment_to_hull_face(
    segment: &mut [ClipVertex; 3],
    hull: &HullData,
    ref_face: usize,
) -> usize {
    let faces = &hull.faces;
    let planes = &hull.planes;
    let edges = &hull.edges;
    let points = &hull.points;

    let ref_plane = planes[ref_face];
    let face = faces[ref_face];
    let mut edge_index = face.edge;

    loop {
        let edge = edges[edge_index];
        let next_edge_index = edge.next;
        let next = edges[next_edge_index];

        let vertex1 = points[edge.origin];
        let vertex2 = points[next.origin];
        let tangent = vertex2.sub(vertex1).normalize();
        let binormal = tangent.cross(ref_plane.normal);

        let point_count = clip_segment(segment, Plane::from_normal_and_point(binormal, vertex1));
        if point_count < 2 {
            return 0;
        }
        edge_index = next_edge_index;
        if edge_index == face.edge {
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
) -> FaceQuery {
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

    FaceQuery {
        separation: max_face_separation,
        face_index: max_face_index & 0xff,
        vertex_index: max_vertex_index & 0xff,
    }
}

fn query_face_directions(
    hull_a: &HullData,
    hull_b: &HullData,
    relative_transform: Transform,
) -> FaceQuery {
    // All computations in local space of the second hull.
    let transform = relative_transform.invert();
    let planes_a = &hull_a.planes;
    let points_b = &hull_b.points;

    let mut max_face_index = 0usize;
    let mut max_vertex_index = 0usize;
    let mut max_face_separation = -FLT_MAX;

    for face_index in 0..hull_a.face_count {
        let pl = planes_a[face_index].transform(transform);
        let vertex_index = hull_b.support_vertex(pl.normal.neg());
        let support = points_b[vertex_index];
        let separation = pl.separation(support);
        if separation > max_face_separation {
            max_face_index = face_index;
            max_vertex_index = vertex_index;
            max_face_separation = separation;
        }
    }

    FaceQuery {
        separation: max_face_separation,
        face_index: max_face_index & 0xff,
        vertex_index: max_vertex_index & 0xff,
    }
}

fn query_edge_direction_hull_and_capsule(
    hull: &HullData,
    capsule: &Capsule,
    capsule_transform: Transform,
) -> EdgeQuery {
    let mut max_separation = -FLT_MAX;
    let mut max_index1: i32 = -1;
    let mut max_index2: i32 = -1;

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

        let p2 = points[edge.origin];
        let q2 = points[twin.origin];
        let e2 = q2.sub(p2);

        let u2 = planes[edge.face].normal;
        let v2 = planes[twin.face].normal;

        if is_minkowski_face_isolated(u2, v2, e1) {
            let c1 = q1.add(p1).scale(0.5);
            let c2 = hull.center;
            let separation = edge_edge_separation(q1, e1, c1, q2, e2, c2);
            if separation > max_separation {
                max_separation = separation;
                max_index1 = 0;
                max_index2 = index as i32;
            }
        }
        index += 2;
    }

    EdgeQuery {
        separation: max_separation,
        index_a: max_index1 & 0xff,
        index_b: max_index2 & 0xff,
    }
}

fn query_edge_directions(
    hull_a: &HullData,
    hull_b: &HullData,
    transform_b_to_a: Transform,
) -> EdgeQuery {
    let mut max_separation = -FLT_MAX;
    let mut max_index_a: i32 = -1;
    let mut max_index_b: i32 = -1;

    let edges_a = &hull_a.edges;
    let points_a = &hull_a.points;
    let planes_a = &hull_a.planes;
    let edges_b = &hull_b.edges;
    let points_b = &hull_b.points;
    let planes_b = &hull_b.planes;

    // Work in frame A.
    let matrix = Mat3::from_quat(transform_b_to_a.q);

    let mut index_b = 0;
    while index_b < hull_b.edge_count {
        let edge_b = edges_b[index_b];
        let twin_b = edges_b[index_b + 1];

        let mut q_b = points_b[twin_b.origin];
        let e_b = matrix.mul_v(q_b.sub(points_b[edge_b.origin]));
        q_b = matrix.mul_v(q_b).add(transform_b_to_a.p);

        let u_b = matrix.mul_v(planes_b[edge_b.face].normal);
        let v_b = matrix.mul_v(planes_b[twin_b.face].normal);

        let mut index_a = 0;
        while index_a < hull_a.edge_count {
            let edge_a = edges_a[index_a];
            let twin_a = edges_a[index_a + 1];

            let q_a = points_a[twin_a.origin];
            let e_a = q_a.sub(points_a[edge_a.origin]);
            let u_a = planes_a[edge_a.face].normal;
            let v_a = planes_a[twin_a.face].normal;

            let cba = u_b.dot(e_a);
            let dba = v_b.dot(e_a);
            let adc = -u_a.dot(e_b);
            let bdc = -v_a.dot(e_b);
            let is_mink = cba * dba < 0.0 && adc * bdc < 0.0 && cba * bdc > 0.0;

            if is_mink {
                let center_a = hull_a.center;
                let center_b = transform_b_to_a.point(hull_b.center);
                let separation = edge_edge_separation(q_a, e_a, center_a, q_b, e_b, center_b);
                if separation > max_separation {
                    max_separation = separation;
                    max_index_a = index_a as i32;
                    max_index_b = index_b as i32;
                }
            }
            index_a += 2;
        }
        index_b += 2;
    }

    EdgeQuery {
        separation: max_separation,
        index_a: max_index_a,
        index_b: max_index_b,
    }
}

/// b3ReduceManifoldPoints — reduce a clipped point set to at most 4 points via a biased extremum
/// search over `points[0..count]`; writes the survivors into `manifold.points`.
fn reduce_manifold_points(
    manifold: &mut LocalManifold,
    capacity: usize,
    points: &[LocalManifoldPoint],
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

    // Swap-remove over an index array (not `points`) to mirror the TS pool-preserving indirection.
    let mut idx = [0usize; MAX_CLIP_POINTS];
    for i in 0..count {
        idx[i] = i;
    }

    // Step 1: extreme point that is touching.
    let mut best_index: i32 = -1;
    let mut best_score = -FLT_MAX;
    let search_direction = arbitrary_perp(normal);
    for index in 0..count {
        let pt = &points[idx[index]];
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

    manifold.points[0] = points[idx[best_index as usize]];
    manifold.point_count = 1;
    idx[best_index as usize] = idx[count - 1];
    count -= 1;

    let a = manifold.points[0].point;

    // Step 2: farthest point in 2D.
    best_score = 0.0;
    best_index = -1;
    for index in 0..count {
        let p = points[idx[index]].point;
        let d = p.sub(a);
        let v = d.mul_sub(d.dot(normal), normal);
        let distance_squared = v.length_sq();
        let separation = maxf(0.0, -points[idx[index]].separation);
        let score = distance_squared + 4.0 * separation * separation;
        if bias * score > best_score {
            best_score = score;
            best_index = index as i32;
        }
    }

    if best_score < tol_sqr {
        return;
    }

    manifold.points[1] = points[idx[best_index as usize]];
    manifold.point_count = 2;
    idx[best_index as usize] = idx[count - 1];
    count -= 1;

    let b = manifold.points[1].point;

    // Step 3: point with the maximum triangular area.
    best_score = tol_sqr;
    best_index = -1;
    let mut best_signed_area = 0.0;
    let ba = b.sub(a);
    for index in 0..count {
        let p = points[idx[index]].point;
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

    manifold.points[2] = points[idx[best_index as usize]];
    manifold.point_count = 3;
    idx[best_index as usize] = idx[count - 1];
    count -= 1;

    let c = manifold.points[2].point;

    // Step 4: point adding the most area outside the current triangle.
    best_score = tol_sqr;
    best_index = -1;
    let sign: f32 = if best_signed_area < 0.0 { -1.0 } else { 1.0 };
    for index in 0..count {
        let p = points[idx[index]].point;
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
        manifold.points[manifold.point_count] = points[idx[best_index as usize]];
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
    manifold.point_count = 0;
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

        let mut vertices_b: [ClipVertex; 3] = [
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
            ClipVertex {
                position: Vec3::ZERO,
                separation: 0.0,
                pair: FeaturePair::SINGLE,
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

                manifold.points[0] = LocalManifoldPoint {
                    point: point1,
                    separation: distance1 - radius,
                    pair: pair0,
                    triangle_index: 0,
                };
                manifold.points[1] = LocalManifoldPoint {
                    point: point2,
                    separation: distance2 - radius,
                    pair: pair1,
                    triangle_index: 0,
                };

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
    query: FaceQuery,
) -> bool {
    let planes = &hull_a.planes;

    let ref_face = query.face_index;
    let ref_plane = planes[ref_face];

    let mut segment_b: [ClipVertex; 3] = [
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
        ClipVertex {
            position: Vec3::ZERO,
            separation: 0.0,
            pair: FeaturePair::SINGLE,
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

        manifold.points[0] = LocalManifoldPoint {
            point: point1,
            separation: distance1 - capsule_b.radius,
            pair: segment_b[0].pair,
            triangle_index: 0,
        };
        manifold.points[1] = LocalManifoldPoint {
            point: point2,
            separation: distance2 - capsule_b.radius,
            pair: segment_b[1].pair,
            triangle_index: 0,
        };

        return true;
    }

    false
}

fn deepest_point_separation(manifold: &LocalManifold) -> f32 {
    let mut min_separation = FLT_MAX;
    for i in 0..manifold.point_count {
        min_separation = minf(min_separation, manifold.points[i].separation);
    }
    min_separation
}

fn build_hull_and_capsule_edge_contact(
    manifold: &mut LocalManifold,
    capacity: usize,
    hull_a: &HullData,
    capsule_b: &Capsule,
    transform_b_to_a: Transform,
    query: EdgeQuery,
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
    let twin2 = edges[edge2.twin];
    let ch = hull_a.center;
    let ph = points[edge2.origin];
    let qh = points[twin2.origin];
    let eh = qh.sub(ph);

    let mut normal = ec.cross(eh);
    normal = normal.normalize();

    // Normal should point outward from hull.
    if normal.dot(ph.sub(ch)) < 0.0 {
        normal = normal.neg();
    }

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
            let mut vertices_b: [ClipVertex; 3] = [
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
                ClipVertex {
                    position: Vec3::ZERO,
                    separation: 0.0,
                    pair: FeaturePair::SINGLE,
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

                    manifold.points[0] = LocalManifoldPoint {
                        point: point1,
                        separation: distance1 - capsule_b.radius,
                        pair: vertices_b[0].pair,
                        triangle_index: 0,
                    };
                    manifold.points[1] = LocalManifoldPoint {
                        point: point2,
                        separation: distance2 - capsule_b.radius,
                        pair: vertices_b[1].pair,
                        triangle_index: 0,
                    };

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
        face_separation = deepest_point_separation(manifold);
    }

    // Create edge contact if face contact fails or edge contact is significantly better.
    let k_rel_edge_tolerance: f32 = 0.9;
    let k_abs_tolerance = 0.5 * LINEAR_SLOP;
    let edge_separation = edge_query.separation - capsule_b.radius;
    if manifold.point_count == 0
        || edge_separation > k_rel_edge_tolerance * face_separation + k_abs_tolerance
    {
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
    out: &mut [ClipVertex],
) -> usize {
    let faces = &hull.faces;
    let edges = &hull.edges;
    let points = &hull.points;

    let face = faces[inc_face];
    let mut edge_index = face.edge;

    let matrix = Mat3::from_quat(transform.q);
    let mut n = 0;

    loop {
        let edge = edges[edge_index];
        let next_edge_index = edge.next;
        let next = edges[next_edge_index];

        let position = matrix.mul_v(points[next.origin]).add(transform.p);
        out[n] = ClipVertex {
            position,
            separation: ref_plane.separation(position),
            pair: FeaturePair {
                owner1: SHAPE_B,
                index1: edge_index as u8,
                owner2: SHAPE_B,
                index2: next_edge_index as u8,
            },
        };
        n += 1;

        edge_index = next_edge_index;
        if edge_index == face.edge || n >= MAX_CLIP_POINTS {
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
    query: FaceQuery,
    cache: &mut SatCache,
) -> bool {
    let faces_a = &hull_a.faces;
    let edges_a = &hull_a.edges;
    let planes_a = &hull_a.planes;
    let points_a = &hull_a.points;

    // Reference face.
    let ref_face = query.face_index;
    let ref_plane = planes_a[ref_face];

    // Find incident face.
    let ref_normal_in_b = transform_b_to_a.q.inv_rotate(ref_plane.normal);
    let inc_face = find_incident_face(hull_b, ref_normal_in_b, query.vertex_index);

    // Build clip polygon from incident face in frame A, then clip it against the reference face's side
    // planes. `input` holds the running polygon; each clip writes into `scratch` and is copied back
    // (the polygon grows by at most one vertex per plane and is bounded by MAX_CLIP_POINTS).
    let mut input = [ClipVertex::ZERO; MAX_CLIP_POINTS];
    let mut scratch = [ClipVertex::ZERO; MAX_CLIP_POINTS];
    let mut point_count = build_polygon(transform_b_to_a, hull_b, inc_face, ref_plane, &mut input);

    // Clip incident face against side planes of the reference face.
    let face = faces_a[ref_face];
    let mut edge_index = face.edge;

    loop {
        let edge = edges_a[edge_index];
        let next_edge_index = edge.next;
        let next = edges_a[next_edge_index];
        let vertex1 = points_a[edge.origin];
        let vertex2 = points_a[next.origin];
        let tangent = vertex2.sub(vertex1).normalize();
        let binormal = tangent.cross(ref_plane.normal);

        let clip_plane = Plane::from_normal_and_point(binormal, vertex1);

        point_count = clip_polygon(
            &input,
            point_count,
            clip_plane,
            edge_index as u8,
            ref_plane,
            &mut scratch,
        );
        input[..point_count].copy_from_slice(&scratch[..point_count]);

        if point_count < 3 {
            cache.reset();
            return false;
        }

        edge_index = next_edge_index;
        if edge_index == face.edge {
            break;
        }
    }

    point_count = point_count.min(MAX_CLIP_POINTS);

    let mut min_separation = FLT_MAX;

    manifold.normal = ref_plane.normal;

    let mut reduce_points = [LocalManifoldPoint::ZERO; MAX_CLIP_POINTS];
    for i in 0..point_count {
        let clip_point = input[i];
        // Half-way point keeps positions stable when swapping the reference face from A to B.
        reduce_points[i] = LocalManifoldPoint {
            point: clip_point
                .position
                .mul_sub(0.5 * clip_point.separation, ref_plane.normal),
            separation: clip_point.separation,
            pair: clip_point.pair,
            triangle_index: 0,
        };
        min_separation = minf(min_separation, clip_point.separation);
    }

    if min_separation >= SPECULATIVE_DISTANCE {
        cache.reset();
        return false;
    }

    reduce_manifold_points(manifold, capacity, &reduce_points, point_count);

    cache.separation = min_separation;
    cache.ty = separating_feature::FACE_AXIS_A;
    cache.index_a = query.face_index & 0xff;
    cache.index_b = query.vertex_index & 0xff;

    true
}

fn build_face_b_contact(
    manifold: &mut LocalManifold,
    capacity: usize,
    hull_a: &HullData,
    hull_b: &HullData,
    transform_b_to_a: Transform,
    query: FaceQuery,
    cache: &mut SatCache,
) -> bool {
    let transform_a_to_b = transform_b_to_a.invert();
    let touching = build_face_a_contact(
        manifold,
        capacity,
        hull_b,
        hull_a,
        transform_a_to_b,
        query,
        cache,
    );
    if !touching {
        return false;
    }

    // Results are in frame B; transform them into frame A.
    let matrix = Mat3::from_quat(transform_b_to_a.q);

    // Flip normal so it points from A to B, even though B owns the reference face.
    manifold.normal = matrix.mul_v(manifold.normal).neg();
    cache.ty = separating_feature::FACE_AXIS_B;
    cache.index_a = query.vertex_index & 0xff;
    cache.index_b = query.face_index & 0xff;

    for i in 0..manifold.point_count {
        manifold.points[i].point = matrix
            .mul_v(manifold.points[i].point)
            .add(transform_b_to_a.p);
        manifold.points[i].pair = flip_pair(manifold.points[i].pair);
    }

    true
}

fn build_edge_contact(
    manifold: &mut LocalManifold,
    hull_a: &HullData,
    hull_b: &HullData,
    transform_b_to_a: Transform,
    query: EdgeQuery,
    cache: &mut SatCache,
) -> bool {
    let edges_a = &hull_a.edges;
    let points_a = &hull_a.points;
    let edges_b = &hull_b.edges;
    let points_b = &hull_b.points;

    let edge_a = edges_a[query.index_a as usize];
    let twin_a = edges_a[edge_a.twin];
    let center_a = hull_a.center;
    let p_a = points_a[edge_a.origin];
    let q_a = points_a[twin_a.origin];
    let e_a = q_a.sub(p_a);

    let edge_b = edges_b[query.index_b as usize];
    let twin_b = edges_b[edge_b.twin];
    let p_b = transform_b_to_a.point(points_b[edge_b.origin]);
    let q_b = transform_b_to_a.point(points_b[twin_b.origin]);
    let e_b = q_b.sub(p_b);

    let mut normal = e_a.cross(e_b);
    normal = normal.normalize();

    if normal.dot(p_a.sub(center_a)) < 0.0 {
        normal = normal.neg();
    }

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

    manifold.points[0] = LocalManifoldPoint {
        point,
        separation,
        pair: make_feature_pair(SHAPE_A, query.index_a as u8, SHAPE_B, query.index_b as u8),
        triangle_index: 0,
    };

    cache.separation = separation;
    cache.ty = separating_feature::EDGE_PAIR_AXIS;
    cache.index_a = (query.index_a & 0xff) as usize;
    cache.index_b = (query.index_b & 0xff) as usize;

    true
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

    // Attempt to use the cache to speed up collision.
    match cache.ty {
        separating_feature::INVALID => {
            cache.reset();
        }

        separating_feature::FACE_AXIS_A => {
            let pl = planes_a[cache.index_a];
            let search_direction_in_b = transform_b_to_a.q.inv_rotate(pl.normal).neg();
            let vertex_index = hull_b.support_vertex(search_direction_in_b);
            let support = transform_b_to_a.point(points_b[vertex_index]);
            let separation = pl.separation(support);

            if separation >= speculative_distance {
                return;
            }

            let face_query = FaceQuery {
                separation: 0.0,
                face_index: cache.index_a,
                vertex_index,
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
                return;
            }

            let face_query = FaceQuery {
                separation: 0.0,
                face_index: cache.index_b,
                vertex_index,
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
                return;
            }
        }

        separating_feature::EDGE_PAIR_AXIS => {
            let index1 = cache.index_a;
            let edge1 = edges_a[index1];
            let twin1 = edges_a[index1 + 1];

            let p1 = points_a[edge1.origin];
            let q1 = points_a[twin1.origin];
            let e1 = q1.sub(p1);

            let u1 = planes_a[edge1.face].normal;
            let v1 = planes_a[twin1.face].normal;

            let index2 = cache.index_b;
            let edge2 = edges_b[index2];
            let twin2 = edges_b[index2 + 1];

            let p2 = transform_b_to_a.point(points_b[edge2.origin]);
            let q2 = transform_b_to_a.point(points_b[twin2.origin]);
            let e2 = q2.sub(p2);

            let u2 = transform_b_to_a.q.rotate(planes_b[edge2.face].normal);
            let v2 = transform_b_to_a.q.rotate(planes_b[twin2.face].normal);

            let is_mink = is_minkowski_face(u1, v1, e1, u2.neg(), v2.neg(), e2);
            if is_mink {
                let c1 = hull_a.center;
                let c2 = transform_b_to_a.point(hull_b.center);

                let separation = edge_edge_separation(p1, e1, c1, p2, e2, c2);
                if separation > speculative_distance {
                    return;
                }

                let edge_query = EdgeQuery {
                    index_a: cache.index_a as i32,
                    index_b: cache.index_b as i32,
                    separation: 0.0,
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
                if touching && absf(cache.separation - local_cache.separation) < linear_slop {
                    return;
                }
            }
        }

        // Manual axes are for testing.
        separating_feature::MANUAL_FACE_AXIS_A => {
            let face_query_a = query_face_directions(hull_a, hull_b, transform_b_to_a);
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
            let face_query_b = query_face_directions(hull_b, hull_a, transform_b_to_a.invert());
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
            let edge_query = query_edge_directions(hull_a, hull_b, transform_b_to_a);
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

    // Find axis of minimum penetration.
    let face_query_a = query_face_directions(hull_a, hull_b, transform_b_to_a);
    if face_query_a.separation > speculative_distance {
        cache.separation = face_query_a.separation;
        cache.ty = separating_feature::FACE_AXIS_A;
        cache.index_a = face_query_a.face_index & 0xff;
        cache.index_b = face_query_a.vertex_index & 0xff;
        return;
    }

    let face_query_b = query_face_directions(hull_b, hull_a, transform_b_to_a.invert());
    if face_query_b.separation > speculative_distance {
        cache.separation = face_query_b.separation;
        cache.ty = separating_feature::FACE_AXIS_B;
        cache.index_a = face_query_b.vertex_index & 0xff;
        cache.index_b = face_query_b.face_index & 0xff;
        return;
    }

    let edge_query = query_edge_directions(hull_a, hull_b, transform_b_to_a);
    if edge_query.separation > speculative_distance {
        cache.separation = edge_query.separation;
        cache.ty = separating_feature::EDGE_PAIR_AXIS;
        cache.index_a = (edge_query.index_a & 0xff) as usize;
        cache.index_b = (edge_query.index_b & 0xff) as usize;
        return;
    }

    // Always build a face contact (e.g. Jenga problem).
    let face_separation_a = face_query_a.separation;
    let face_separation_b = face_query_b.separation;

    if face_separation_b > face_separation_a + 0.5 * linear_slop {
        build_face_b_contact(
            manifold,
            capacity,
            hull_a,
            hull_b,
            transform_b_to_a,
            face_query_b,
            cache,
        );
    } else {
        build_face_a_contact(
            manifold,
            capacity,
            hull_a,
            hull_b,
            transform_b_to_a,
            face_query_a,
            cache,
        );
    }

    if edge_query.index_a == -1 {
        // No valid edge pairs (all edges parallel).
        return;
    }

    let clipped_face_separation = cache.separation;

    // Create edge contact if face contact fails or edge contact is significantly better.
    let k_rel_edge_tolerance: f32 = 0.9;
    let k_abs_tolerance = 0.5 * linear_slop;

    if manifold.point_count == 0
        || edge_query.separation > k_rel_edge_tolerance * clipped_face_separation + k_abs_tolerance
    {
        let mut edge_manifold = LocalManifold::new();
        edge_manifold.point_count = 0;

        build_edge_contact(
            &mut edge_manifold,
            hull_a,
            hull_b,
            transform_b_to_a,
            edge_query,
            cache,
        );

        if edge_manifold.point_count == 1 {
            // Copy the edge manifold out, preserving the caller's point buffer.
            manifold.normal = edge_manifold.normal;
            manifold.point_count = edge_manifold.point_count;
            manifold.points[0] = edge_manifold.points[0];
        }
    }
}
