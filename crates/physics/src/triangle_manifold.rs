//! Triangle narrowphase from Box3D triangle_manifold.c (Erin Catto, MIT).
use crate::distance::{shape_distance, DistanceInput, ShapeProxy, SimplexCache};
use crate::hull::HullData;
use crate::manifold::{
    clip_polygon, find_incident_face, flip_pair, Capsule, ClipVertex, FeaturePair, LocalManifold,
    SatCache, Sphere,
};
use crate::math::{line_distance, maxf, minf, Plane, Transform, Vec3};

const SLOP: f32 = 0.005;
const SPECULATIVE: f32 = 4.0 * SLOP;
const FACE: u32 = 1;
const EDGES: [u32; 3] = [3, 4, 5];
const FEATURES: [u32; 8] = [0, 6, 7, 3, 8, 5, 4, 1];
const SINGLE: FeaturePair = FeaturePair {
    owner1: 0,
    index1: 0,
    owner2: 0,
    index2: 0,
};

struct TriangleData {
    points: [Vec3; 3],
    edges: [Vec3; 3],
    plane: Plane,
    flags: u32,
}

#[derive(Clone, Copy)]
struct Axis {
    normal: Vec3,
    separation: f32,
    a: usize,
    b: usize,
    ty: u32,
}
impl Axis {
    fn save(self, cache: &mut SatCache) {
        cache.separation = self.separation;
        cache.ty = self.ty;
        cache.index_a = self.a;
        cache.index_b = self.b;
    }
}
fn triangle_support(tri: &[Vec3; 3], direction: Vec3) -> usize {
    let mut index = 0;
    let mut distance = tri[0].dot(direction);
    for i in 1..3 {
        let d = tri[i].dot(direction);
        if d > distance {
            distance = d;
            index = i;
        }
    }
    index
}
fn triangle_axis(plane: Plane, hull: &HullData) -> Axis {
    let b = hull.support_vertex(plane.normal.neg());
    Axis {
        normal: plane.normal,
        separation: plane.separation(hull.points[b]),
        a: 0,
        b,
        ty: 2,
    }
}
fn hull_axis(tri: &[Vec3; 3], hull: &HullData) -> Axis {
    let mut result = Axis {
        normal: Vec3::ZERO,
        separation: f32::NEG_INFINITY,
        a: usize::MAX,
        b: usize::MAX,
        ty: 3,
    };
    for b in 0..hull.face_count {
        let plane = hull.planes[b];
        let a = triangle_support(tri, plane.normal.neg());
        let separation = plane.separation(tri[a]);
        if separation > result.separation {
            result = Axis {
                normal: plane.normal.neg(),
                separation,
                a,
                b,
                ty: 3,
            };
        }
    }
    result
}
fn edge_axis(
    tri: &[Vec3; 3],
    edges: &[Vec3; 3],
    plane: Plane,
    hull: &HullData,
    a: usize,
    b: usize,
) -> Option<Axis> {
    let e = hull.edges[b];
    let twin = hull.edges[b + 1];
    let p = hull.points[e.origin as usize];
    let edge = hull.points[twin.origin as usize].sub(p);
    let n1 = hull.planes[e.face as usize].normal;
    let n2 = hull.planes[twin.face as usize].normal;
    let cab = n1.dot(edges[a]);
    let dab = n2.dot(edges[a]);
    let bcd = plane.normal.dot(edge);
    if cab * dab >= 0.0
        || cab * bcd <= 0.0
        || maxf(cab * cab, dab * dab) < SLOP * SLOP * edges[a].length_sq()
    {
        return None;
    }
    let axis = n1.lerp(n2, cab / (cab - dab)).normalize();
    Some(Axis {
        normal: axis.neg(),
        separation: axis.dot(tri[a].sub(p)),
        a,
        b,
        ty: 4,
    })
}
fn edges_axis(triangle: &TriangleData, hull: &HullData) -> Axis {
    // Box3D retains the flags at this boundary but does not yet filter edge axes.
    let _flags = triangle.flags;
    let tri = &triangle.points;
    let edges = &triangle.edges;
    let plane = triangle.plane;
    let mut result = Axis {
        normal: Vec3::ZERO,
        separation: f32::NEG_INFINITY,
        a: usize::MAX,
        b: usize::MAX,
        ty: 4,
    };
    for b in (0..hull.edge_count).step_by(2) {
        for a in 0..3 {
            if let Some(query) = edge_axis(tri, edges, plane, hull, a, b) {
                if query.separation > result.separation {
                    result = query;
                }
            }
        }
    }
    result
}
fn hull_face_contact(
    m: &mut LocalManifold,
    capacity: usize,
    tri: &[Vec3; 3],
    hull: &HullData,
    query: Axis,
    cache: &mut SatCache,
    speculative: bool,
) -> f32 {
    m.point_count = 0;
    let plane = hull.planes[query.b];
    let mut buffer1 = [core::mem::MaybeUninit::<ClipVertex>::uninit(); 64];
    let mut buffer2 = [core::mem::MaybeUninit::<ClipVertex>::uninit(); 64];
    let mut input = &mut buffer1;
    let mut output = &mut buffer2;
    for i in 0..3 {
        input[i].write(ClipVertex {
            position: tri[i],
            separation: plane.separation(tri[i]),
            pair: FeaturePair {
                owner1: 1,
                index1: ((i + 2) % 3) as u8,
                owner2: 1,
                index2: i as u8,
            },
        });
    }
    let mut count = 3;
    let first = hull.faces[query.b].edge as usize;
    let mut edge = first;
    loop {
        let e = hull.edges[edge];
        let next = hull.edges[e.next as usize];
        let v1 = hull.points[e.origin as usize];
        let v2 = hull.points[next.origin as usize];
        let side = v2.sub(v1).normalize().cross(plane.normal);
        count = clip_polygon(
            // The seed loop and clip_polygon initialize exactly the returned prefix.
            unsafe { core::slice::from_raw_parts(input.as_ptr().cast(), count) },
            count,
            Plane::from_normal_and_point(side, v1),
            edge as u8,
            plane,
            output,
        );
        if count < 3 {
            *cache = SatCache::empty();
            return query.separation;
        }
        core::mem::swap(&mut input, &mut output);
        edge = e.next as usize;
        if edge == first {
            break;
        }
    }
    let mut min_separation = f32::MAX;
    let mut final_count = 0;
    for slot in &input[..count.min(capacity)] {
        // Clipping initializes exactly the active prefix.
        let p = unsafe { slot.assume_init_ref() };
        min_separation = minf(min_separation, p.separation);
        if !speculative && p.separation > 0.0 {
            continue;
        }
        m.points[final_count].point = p.position.mul_sub(p.separation, plane.normal);
        m.points[final_count].separation = p.separation;
        m.points[final_count].pair = flip_pair(p.pair);
        final_count += 1;
    }
    if min_separation > if speculative { SPECULATIVE } else { 0.0 } {
        *cache = SatCache::empty();
        return min_separation;
    }
    m.point_count = final_count;
    m.normal = plane.normal.neg();
    m.feature = 2;
    Axis {
        separation: min_separation,
        ..query
    }
    .save(cache);
    min_separation
}
fn triangle_face_contact(
    m: &mut LocalManifold,
    capacity: usize,
    tri: &[Vec3; 3],
    edges: &[Vec3; 3],
    plane: Plane,
    hull: &HullData,
    query: Axis,
    cache: &mut SatCache,
    speculative: bool,
) -> f32 {
    let face = find_incident_face(hull, plane.normal, query.b);
    let mut buffer1 = [core::mem::MaybeUninit::<ClipVertex>::uninit(); 64];
    let mut buffer2 = [core::mem::MaybeUninit::<ClipVertex>::uninit(); 64];
    let mut input = &mut buffer1;
    let mut output = &mut buffer2;
    let first = hull.faces[face].edge as usize;
    let mut edge = first;
    let mut count = 0;
    loop {
        let e = hull.edges[edge];
        let p = hull.points[hull.edges[e.next as usize].origin as usize];
        input[count].write(ClipVertex {
            position: p,
            separation: plane.separation(p),
            pair: FeaturePair {
                owner1: 1,
                index1: edge as u8,
                owner2: 1,
                index2: e.next,
            },
        });
        count += 1;
        edge = e.next as usize;
        if edge == first || count == 64 {
            break;
        }
    }
    for i in 0..3 {
        if count == 0 {
            break;
        }
        let side = edges[i].cross(plane.normal).normalize();
        count = clip_polygon(
            // The seed loop and clip_polygon initialize exactly the returned prefix.
            unsafe { core::slice::from_raw_parts(input.as_ptr().cast(), count) },
            count,
            Plane::from_normal_and_point(side, tri[i]),
            i as u8,
            plane,
            output,
        );
        core::mem::swap(&mut input, &mut output);
    }
    if count == 0 {
        *cache = SatCache::empty();
        return f32::MAX;
    }
    let mut min_separation = f32::MAX;
    let mut final_count = 0;
    for slot in &input[..count.min(capacity)] {
        // Clipping initializes exactly the active prefix.
        let p = unsafe { slot.assume_init_ref() };
        min_separation = minf(min_separation, p.separation);
        if !speculative && p.separation > 0.0 {
            continue;
        }
        m.points[final_count].point = p.position;
        m.points[final_count].separation = p.separation;
        m.points[final_count].pair = p.pair;
        final_count += 1;
    }
    if min_separation >= if speculative { SPECULATIVE } else { 0.0 } {
        *cache = SatCache::empty();
        return min_separation;
    }
    m.point_count = final_count;
    m.normal = plane.normal;
    m.feature = FACE;
    Axis {
        separation: min_separation,
        ..query
    }
    .save(cache);
    min_separation
}
fn hull_edge_contact(
    m: &mut LocalManifold,
    capacity: usize,
    tri: &[Vec3; 3],
    edges: &[Vec3; 3],
    hull: &HullData,
    query: Axis,
    cache: &mut SatCache,
) {
    let e = hull.edges[query.b];
    let p = hull.points[e.origin as usize];
    let edge = hull.points[hull.edges[e.twin as usize].origin as usize].sub(p);
    let result = line_distance(tri[query.a], edges[query.a], p, edge);
    if capacity == 0
        || result.fraction1 < 0.0
        || result.fraction1 > 1.0
        || result.fraction2 < 0.0
        || result.fraction2 > 1.0
    {
        *cache = SatCache::empty();
        return;
    }
    let separation = query.normal.dot(p.sub(tri[query.a]));
    m.points[0].point = result.point1.add(result.point2).scale(0.5);
    m.points[0].separation = separation;
    m.points[0].pair = FeaturePair {
        owner1: 0,
        index1: query.a as u8,
        owner2: 1,
        index2: query.b as u8,
    };
    m.normal = query.normal;
    m.point_count = 1;
    m.feature = EDGES[query.a];
    Axis {
        separation,
        ..query
    }
    .save(cache);
}

pub fn collide_hull_and_triangle(
    m: &mut LocalManifold,
    capacity: usize,
    hull: &HullData,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    triangle_flags: u32,
    cache: &mut SatCache,
    speculative: bool,
) {
    m.point_count = 0;
    m.feature = 0;
    if capacity < 4 {
        return;
    }
    let tri = [v1, v2, v3];
    let plane = Plane::from_points(v1, v2, v3);
    let offset = plane.separation(hull.center);
    if cache.ty == 1 {
        if (cache.separation - offset).abs() < SLOP {
            return;
        }
        cache.ty = 0;
    }
    if offset < -SLOP {
        cache.ty = 1;
        cache.separation = offset;
        return;
    }
    let triangle = TriangleData {
        points: tri,
        edges: [v2.sub(v1), v3.sub(v2), v1.sub(v3)],
        plane,
        flags: triangle_flags,
    };
    let edges = triangle.edges;
    let distance = if speculative { SPECULATIVE } else { 0.0 };
    cache.hit = 1;
    match cache.ty {
        2 => {
            let query = triangle_axis(plane, hull);
            if query.separation > distance {
                return;
            }
            let mut local = *cache;
            let separation = triangle_face_contact(
                m,
                capacity,
                &tri,
                &edges,
                plane,
                hull,
                query,
                &mut local,
                speculative,
            );
            if m.point_count > 0 && (cache.separation - separation).abs() < SLOP {
                return;
            }
            m.point_count = 0;
            *cache = SatCache::empty();
        }
        3 => {
            let p = hull.planes[cache.index_b];
            let a = triangle_support(&tri, p.normal.neg());
            let separation = p.separation(tri[a]);
            if separation > distance {
                return;
            }
            if separation >= -2.0 * SLOP {
                let query = Axis {
                    normal: p.normal.neg(),
                    separation,
                    a,
                    b: cache.index_b,
                    ty: 3,
                };
                let mut local = *cache;
                let clipped =
                    hull_face_contact(m, capacity, &tri, hull, query, &mut local, speculative);
                if m.point_count > 0 && (cache.separation - clipped).abs() < SLOP {
                    return;
                }
            }
            m.point_count = 0;
            *cache = SatCache::empty();
        }
        4 => {
            if let Some(query) = edge_axis(&tri, &edges, plane, hull, cache.index_a, cache.index_b)
            {
                if query.separation > distance {
                    return;
                }
                if (cache.separation - query.separation).abs() < SLOP {
                    let mut local = *cache;
                    hull_edge_contact(m, capacity, &tri, &edges, hull, query, &mut local);
                    if m.point_count > 0 {
                        return;
                    }
                }
            }
            *cache = SatCache::empty();
        }
        6 => {
            triangle_face_contact(
                m,
                capacity,
                &tri,
                &edges,
                plane,
                hull,
                triangle_axis(plane, hull),
                cache,
                speculative,
            );
            return;
        }
        7 => {
            hull_face_contact(
                m,
                capacity,
                &tri,
                hull,
                hull_axis(&tri, hull),
                cache,
                speculative,
            );
            return;
        }
        8 => {
            let query = edges_axis(&triangle, hull);
            if query.a != usize::MAX {
                hull_edge_contact(m, capacity, &tri, &edges, hull, query, cache);
            }
            return;
        }
        _ => {}
    }
    cache.hit = 0;
    let a = triangle_axis(plane, hull);
    if a.separation > distance {
        a.save(cache);
        return;
    }
    let b = hull_axis(&tri, hull);
    if b.separation > distance {
        b.save(cache);
        return;
    }
    let e = edges_axis(&triangle, hull);
    if e.separation > distance {
        e.save(cache);
        return;
    }
    let pushing_down = b.normal.dot(plane.normal) < -0.25;
    let clipped = if b.separation >= a.separation && !pushing_down {
        hull_face_contact(m, capacity, &tri, hull, b, cache, speculative)
    } else {
        triangle_face_contact(
            m,
            capacity,
            &tri,
            &edges,
            plane,
            hull,
            a,
            cache,
            speculative,
        )
    };
    if e.a != usize::MAX
        && ((m.point_count == 0 && e.separation > maxf(a.separation, b.separation))
            || (m.point_count == 1 && e.separation > clipped + SLOP))
    {
        m.point_count = 0;
        hull_edge_contact(m, capacity, &tri, &edges, hull, e, cache);
    }
    if m.point_count == 0 {
        let mut simplex = SimplexCache::empty();
        let output = shape_distance(
            &DistanceInput {
                proxy_a: ShapeProxy {
                    points: &tri,
                    count: 3,
                    radius: 0.0,
                },
                proxy_b: ShapeProxy {
                    points: hull.points,
                    count: hull.vertex_count,
                    radius: 0.0,
                },
                transform: Transform::IDENTITY,
                use_radii: false,
            },
            &mut simplex,
        );
        if output.distance > 0.0 {
            let mut mask = 0;
            for i in 0..simplex.count {
                mask |= 1 << simplex.index_a[i];
            }
            m.point_count = 1;
            m.feature = FEATURES[mask];
            m.normal = output.normal;
            m.points[0].point = output.point_b;
            m.points[0].separation = output.distance;
            m.points[0].pair = SINGLE;
        }
        *cache = SatCache::empty();
    }
}

fn closest_point(a: Vec3, b: Vec3, c: Vec3, q: Vec3) -> (Vec3, u32) {
    let ab = b.sub(a);
    let ac = c.sub(a);
    let aq = q.sub(a);
    let d1 = ab.dot(aq);
    let d2 = ac.dot(aq);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, 6);
    }
    let bq = q.sub(b);
    let d3 = ab.dot(bq);
    let d4 = ac.dot(bq);
    if d3 > 0.0 && d4 <= d3 {
        return (b, 7);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a.mul_add(d1 / (d1 - d3), ab), 3);
    }
    let cq = q.sub(c);
    let d5 = ab.dot(cq);
    let d6 = ac.dot(cq);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, 8);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a.mul_add(d2 / (d2 - d6), ac), 5);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 >= d3 && d5 >= d6 {
        return (b.mul_add((d4 - d3) / ((d4 - d3) + (d5 - d6)), c.sub(b)), 4);
    }
    let denom = (va + vb) + vc;
    (a.mul_add(vb / denom, ab).mul_add(vc / denom, ac), FACE)
}

pub fn collide_sphere_and_triangle(
    m: &mut LocalManifold,
    capacity: usize,
    sphere: &Sphere,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
) {
    m.point_count = 0;
    if capacity == 0 {
        return;
    }
    let plane = Plane::from_points(v1, v2, v3);
    if plane.separation(sphere.center) < 0.0 {
        return;
    }
    let (closest, feature) = closest_point(v1, v2, v3, sphere.center);
    let squared = closest.sub(sphere.center).length_sq();
    let max_distance = sphere.radius + SPECULATIVE;
    if squared > max_distance * max_distance {
        return;
    }
    let distance = squared.sqrt();
    let normal = if distance * distance > 1000.0 * f32::MIN_POSITIVE {
        sphere.center.sub(closest).scale(1.0 / distance)
    } else {
        v2.sub(v1).cross(v3.sub(v1)).normalize()
    };
    m.normal = normal;
    m.point_count = 1;
    m.feature = feature;
    m.squared_distance = squared;
    m.points[0].point = sphere
        .center
        .sub(normal.scale(sphere.radius))
        .add(closest)
        .scale(0.5);
    m.points[0].separation = distance - sphere.radius;
    m.points[0].pair = SINGLE;
}

#[derive(Clone, Copy)]
struct Clip {
    position: Vec3,
    pair: FeaturePair,
}
fn segment(c: &Capsule) -> [Clip; 2] {
    [
        Clip {
            position: c.center1,
            pair: SINGLE,
        },
        Clip {
            position: c.center2,
            pair: FeaturePair {
                index1: 1,
                index2: 1,
                ..SINGLE
            },
        },
    ]
}
fn clip_segment(s: &mut [Clip; 2], triangle: &[Vec3; 3], plane: Plane) -> bool {
    let mut v1 = triangle[2];
    for &v2 in triangle {
        let tangent = v2.sub(v1).normalize();
        let clip = Plane::from_normal_and_point(tangent.cross(plane.normal), v1);
        let [p1, p2] = *s;
        let d1 = clip.separation(p1.position);
        let d2 = clip.separation(p2.position);
        let mut count = 0;
        if d1 <= 0.0 {
            s[count] = p1;
            count += 1;
        }
        if d2 <= 0.0 {
            s[count] = p2;
            count += 1;
        }
        if (d1 > 0.0) != (d2 > 0.0) {
            s[count] = Clip {
                position: p1.position.lerp(p2.position, d1 / (d1 - d2)),
                pair: if d1 > 0.0 { p1.pair } else { p2.pair },
            };
            count += 1;
        }
        if count != 2 {
            return false;
        }
        v1 = v2;
    }
    true
}
fn face_contact(m: &mut LocalManifold, s: &[Clip; 2], plane: Plane, radius: f32) {
    m.normal = plane.normal;
    m.feature = FACE;
    m.point_count = 2;
    for i in 0..2 {
        let distance = plane.separation(s[i].position);
        m.points[i].point = s[i]
            .position
            .mul_sub(0.5 * (distance + radius), plane.normal);
        m.points[i].separation = distance - radius;
        m.points[i].pair = s[i].pair;
    }
}

pub fn collide_capsule_and_triangle(
    m: &mut LocalManifold,
    capacity: usize,
    c: &Capsule,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    cache: &mut SimplexCache,
) {
    m.point_count = 0;
    if capacity < 2 {
        return;
    }
    let triangle = [v1, v2, v3];
    let plane = Plane::from_points(v1, v2, v3);
    if plane.separation(c.center1.lerp(c.center2, 0.5)) < 0.0 {
        return;
    }
    let capsule = [c.center1, c.center2];
    let output = shape_distance(
        &DistanceInput {
            proxy_a: ShapeProxy {
                points: &triangle,
                count: 3,
                radius: 0.0,
            },
            proxy_b: ShapeProxy {
                points: &capsule,
                count: 2,
                radius: 0.0,
            },
            transform: Transform::IDENTITY,
            use_radii: false,
        },
        cache,
    );
    if output.distance > c.radius + SPECULATIVE {
        return;
    }
    if output.distance > 100.0 * f32::EPSILON {
        let delta = output.point_b.sub(output.point_a).normalize();
        if plane.normal.dot(delta).abs() > 0.2 {
            let mut s = segment(c);
            if clip_segment(&mut s, &triangle, plane) {
                face_contact(m, &s, plane, c.radius);
                return;
            }
        }
        m.normal = delta;
        m.point_count = 1;
        let mut mask = 0;
        for i in 0..cache.count {
            mask |= 1 << cache.index_a[i];
        }
        m.feature = FEATURES[mask];
        m.points[0].point = output
            .point_a
            .lerp(output.point_b.mul_sub(c.radius, delta), 0.5);
        m.points[0].separation = output.distance - c.radius;
        m.points[0].pair = SINGLE;
        return;
    }
    let face_separation = minf(plane.separation(c.center1), plane.separation(c.center2));
    if face_separation > c.radius {
        return;
    }
    let edge = c.center2.sub(c.center1);
    let a = edge.dot(plane.normal);
    let mut best = -f32::MAX;
    let mut index = None;
    let mut normal = Vec3::ZERO;
    let mut edge_index = 2;
    let mut start = v3;
    for (i, &end) in triangle.iter().enumerate() {
        let side = end.sub(start).cross(plane.normal).normalize();
        let b = edge.dot(side);
        if a * a + b * b >= (SLOP * SLOP) * edge.length_sq() {
            let axis = if a * b <= 0.0 {
                side.lerp(plane.normal, b / (b - a))
            } else {
                side.lerp(plane.normal.neg(), b / (a + b))
            }
            .normalize();
            let separation = axis.dot(c.center1.sub(start));
            if separation > best {
                best = separation;
                index = Some(edge_index);
                normal = axis;
            }
        }
        start = end;
        edge_index = i;
    }
    if best > c.radius {
        return;
    }
    let mut clipped_separation = face_separation - c.radius;
    let mut s = segment(c);
    if clip_segment(&mut s, &triangle, plane) {
        let d1 = plane.separation(s[0].position);
        let d2 = plane.separation(s[1].position);
        if !(d1 > SPECULATIVE + c.radius && d2 > SPECULATIVE + c.radius) {
            face_contact(m, &s, plane, c.radius);
            clipped_separation = minf(m.points[0].separation, m.points[1].separation);
        }
    }
    let Some(i) = index else {
        return;
    };
    if m.point_count == 0 || best - c.radius > clipped_separation + SLOP {
        let start = triangle[i];
        let triangle_edge = triangle[(i + 1) % 3].sub(start);
        let side = triangle_edge.cross(plane.normal).normalize();
        let b = edge.dot(side);
        if a * a + b * b < (SLOP * SLOP) * edge.length_sq() {
            return;
        }
        let result = line_distance(start, triangle_edge, c.center1, edge);
        if result.fraction1 < 0.0
            || result.fraction1 > 1.0
            || result.fraction2 < 0.0
            || result.fraction2 > 1.0
        {
            return;
        }
        m.normal = normal;
        m.point_count = 1;
        m.feature = EDGES[i];
        m.points[0].point = result
            .point1
            .lerp(result.point2.mul_sub(c.radius, normal), 0.5);
        m.points[0].separation = normal.dot(c.center1.sub(start)) - c.radius;
        m.points[0].pair = FeaturePair {
            owner1: 0,
            index1: i as u8,
            owner2: 1,
            index2: 0,
        };
    }
}
