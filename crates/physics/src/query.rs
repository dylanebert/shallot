//! Per-shape queries ported from Box3D sphere.c, capsule.c and hull.c.

use crate::distance::{
    shape_cast, shape_distance, CastOutput, DistanceInput, ShapeCastPairInput, ShapeProxy,
    SimplexCache,
};
use crate::height_query::{
    collide_mover_height, overlap_height, ray_cast_height, shape_cast_height, HeightField,
};
use crate::hull::HullData;
use crate::manifold::{Capsule, Sphere};
use crate::math::{
    clampf, get_length_and_normalize, point_to_segment_distance, segment_distance, Plane,
    Transform, Vec3, FLT_EPSILON,
};
use crate::mesh_query::{collide_mover_mesh, overlap_mesh, ray_cast_mesh, shape_cast_mesh, Mesh};

#[derive(Clone, Copy)]
pub struct RayCastInput {
    pub origin: Vec3,
    pub translation: Vec3,
    pub max_fraction: f32,
}

pub struct ShapeCastInput<'a> {
    pub proxy: ShapeProxy<'a>,
    pub translation: Vec3,
    pub max_fraction: f32,
    pub can_encroach: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlaneResult {
    pub plane: Plane,
    pub point: Vec3,
    pub triangle_index: i32,
    pub child_index: i32,
    pub material_index: i32,
}

impl PlaneResult {
    pub const ZERO: Self = Self {
        plane: Plane {
            normal: Vec3::ZERO,
            offset: 0.0,
        },
        point: Vec3::ZERO,
        triangle_index: 0,
        child_index: 0,
        material_index: 0,
    };
}

pub enum Shape<'a> {
    Sphere(&'a Sphere),
    Capsule(&'a Capsule),
    Hull(HullData<'a>),
    Mesh(Mesh<'a>),
    Height(HeightField<'a>),
    #[cfg(target_arch = "wasm32")]
    Compound(crate::compound_query::Compound<'a>),
}

pub fn ray_cast_shape(shape: &Shape, transform: Transform, input: &RayCastInput) -> CastOutput {
    let local = RayCastInput {
        origin: transform.inv_point(input.origin),
        translation: transform.q.inv_rotate(input.translation),
        max_fraction: input.max_fraction,
    };
    let mut out = ray_cast_local(shape, &local);
    out.point = transform.point(out.point);
    out.normal = transform.q.rotate(out.normal);
    out
}

pub(crate) fn ray_cast_local(shape: &Shape, input: &RayCastInput) -> CastOutput {
    match shape {
        Shape::Sphere(s) => ray_cast_sphere(s, input),
        Shape::Capsule(s) => ray_cast_capsule(s, input),
        Shape::Hull(s) => ray_cast_hull(s, input),
        Shape::Mesh(s) => ray_cast_mesh(*s, input),
        Shape::Height(s) => ray_cast_height(*s, input),
        #[cfg(target_arch = "wasm32")]
        Shape::Compound(s) => crate::compound_query::ray_cast_compound(*s, input),
    }
}

pub fn shape_cast_shape(shape: &Shape, transform: Transform, input: &ShapeCastInput) -> CastOutput {
    let mut points = [core::mem::MaybeUninit::<Vec3>::uninit(); 128];
    let count = input.proxy.count.min(128);
    for (i, p) in points[..count].iter_mut().enumerate() {
        p.write(transform.inv_point(input.proxy.points[i]));
    }
    // Only the transformed prefix is initialized and exposed to the proxy.
    let points = unsafe { core::slice::from_raw_parts(points.as_ptr().cast::<Vec3>(), count) };
    let local = ShapeCastInput {
        proxy: ShapeProxy {
            points,
            count,
            radius: input.proxy.radius,
        },
        translation: transform.q.inv_rotate(input.translation),
        max_fraction: input.max_fraction,
        can_encroach: input.can_encroach,
    };
    let mut out = shape_cast_local(shape, &local);
    out.point = transform.point(out.point);
    out.normal = transform.q.rotate(out.normal);
    out
}

pub(crate) fn shape_cast_local(shape: &Shape, input: &ShapeCastInput) -> CastOutput {
    match shape {
        Shape::Sphere(s) => shape_cast_convex(
            ShapeProxy {
                points: core::slice::from_ref(&s.center),
                count: 1,
                radius: s.radius,
            },
            input,
        ),
        Shape::Capsule(s) => shape_cast_convex(
            ShapeProxy {
                points: s.points(),
                count: 2,
                radius: s.radius,
            },
            input,
        ),
        Shape::Hull(s) => shape_cast_convex(
            ShapeProxy {
                points: s.points,
                count: s.vertex_count,
                radius: 0.0,
            },
            input,
        ),
        Shape::Mesh(s) => shape_cast_mesh(*s, input),
        Shape::Height(s) => shape_cast_height(*s, input),
        #[cfg(target_arch = "wasm32")]
        Shape::Compound(s) => crate::compound_query::shape_cast_compound(*s, input),
    }
}

pub fn overlap_shape(shape: &Shape, transform: Transform, proxy: ShapeProxy) -> bool {
    match shape {
        Shape::Sphere(s) => overlap_convex(
            ShapeProxy {
                points: core::slice::from_ref(&s.center),
                count: 1,
                radius: s.radius,
            },
            transform,
            proxy,
        ),
        Shape::Capsule(s) => overlap_convex(
            ShapeProxy {
                points: s.points(),
                count: 2,
                radius: s.radius,
            },
            transform,
            proxy,
        ),
        Shape::Hull(s) => overlap_convex(
            ShapeProxy {
                points: s.points,
                count: s.vertex_count,
                radius: 0.0,
            },
            transform,
            proxy,
        ),
        Shape::Mesh(s) => overlap_mesh(*s, transform, proxy),
        Shape::Height(s) => overlap_height(*s, transform, proxy),
        #[cfg(target_arch = "wasm32")]
        Shape::Compound(s) => crate::compound_query::overlap_compound(*s, transform, proxy),
    }
}

pub fn collide_mover(
    planes: &mut [PlaneResult],
    shape: &Shape,
    transform: Transform,
    mover: &Capsule,
    material_count: i32,
) -> usize {
    if planes.is_empty() {
        return 0;
    }
    let local = Capsule {
        center1: transform.inv_point(mover.center1),
        center2: transform.inv_point(mover.center2),
        radius: mover.radius,
    };
    let count = collide_mover_local(planes, shape, &local);
    for p in &mut planes[..count] {
        p.plane.normal = transform.q.rotate(p.plane.normal);
        p.point = transform.point(p.point);
        p.material_index = p.material_index.max(0).min(material_count - 1);
    }
    count
}

pub(crate) fn collide_mover_local(
    planes: &mut [PlaneResult],
    shape: &Shape,
    mover: &Capsule,
) -> usize {
    if planes.is_empty() {
        return 0;
    }
    let result = match shape {
        Shape::Sphere(s) => collide_mover_sphere(s, mover),
        Shape::Capsule(s) => collide_mover_capsule(s, mover),
        Shape::Hull(s) => collide_mover_hull(s, mover),
        Shape::Mesh(s) => return collide_mover_mesh(planes, *s, mover),
        Shape::Height(s) => return collide_mover_height(planes, *s, mover),
        #[cfg(target_arch = "wasm32")]
        Shape::Compound(s) => {
            return crate::compound_query::collide_mover_compound(planes, *s, mover)
        }
    };
    if let Some(result) = result {
        planes[0] = result;
        1
    } else {
        0
    }
}

pub fn overlap_convex(shape: ShapeProxy, transform: Transform, proxy: ShapeProxy) -> bool {
    let input = DistanceInput {
        proxy_a: shape,
        proxy_b: proxy,
        transform: transform.inv_mul(Transform::IDENTITY),
        use_radii: true,
    };
    shape_distance(&input, &mut SimplexCache::empty()).distance < 0.1 * 0.005
}

pub fn shape_cast_convex(shape: ShapeProxy, input: &ShapeCastInput) -> CastOutput {
    shape_cast(&ShapeCastPairInput {
        proxy_a: shape,
        proxy_b: input.proxy,
        transform: Transform::IDENTITY,
        translation_b: input.translation,
        max_fraction: input.max_fraction,
        can_encroach: input.can_encroach,
    })
}

pub fn ray_cast_sphere(shape: &Sphere, input: &RayCastInput) -> CastOutput {
    let mut output = CastOutput {
        triangle_index: 0,
        ..CastOutput::MISS
    };
    let s = input.origin.sub(shape.center);
    let rr = shape.radius * shape.radius;
    let (d, length) = get_length_and_normalize(input.translation);
    if length == 0.0 {
        if s.length_sq() < rr {
            output.point = input.origin;
            output.hit = true;
        }
        return output;
    }
    let t = -s.dot(d);
    let c = s.mul_add(t, d);
    let cc = c.dot(c);
    if cc > rr {
        return output;
    }
    let h = (rr - cc).sqrt();
    let fraction = t - h;
    if fraction < 0.0 || input.max_fraction * length < fraction {
        if s.length_sq() < rr {
            output.point = input.origin;
            output.hit = true;
        }
        return output;
    }
    let hit_point = s.mul_add(fraction, d);
    output.fraction = fraction / length;
    if output.fraction > input.max_fraction {
        output.fraction = input.max_fraction;
    }
    output.normal = hit_point.normalize();
    output.point = shape.center.mul_add(shape.radius, output.normal);
    output.hit = true;
    output
}

pub fn ray_cast_capsule(shape: &Capsule, input: &RayCastInput) -> CastOutput {
    let c1 = shape.center1;
    let c2 = shape.center2;
    let r = shape.radius;
    let mut output = CastOutput {
        triangle_index: 0,
        ..CastOutput::MISS
    };
    let d = c2.sub(c1);
    let tol = 0.01 * 0.005;
    let length_sq = d.length_sq();
    if length_sq < tol * tol {
        return ray_cast_sphere(
            &Sphere {
                center: c1.add(c2).scale(0.5),
                radius: r,
            },
            input,
        );
    }
    let s = input.origin.sub(c1);
    let length = length_sq.sqrt();
    let axis = d.scale(1.0 / length);
    let u = s.dot(axis);
    let c = axis.scale(u);
    let sc = s.sub(c);
    let sc2 = sc.length_sq();
    if sc2 < r * r {
        let cp = axis.scale(clampf(u, 0.0, length));
        if s.sub(cp).length_sq() < r * r {
            output.hit = true;
            output.point = input.origin;
            return output;
        }
        return ray_cast_sphere(
            &Sphere {
                center: c1.add(cp),
                radius: r,
            },
            input,
        );
    }
    let dr = input.translation;
    let (ray_axis, ray_length) = get_length_and_normalize(dr);
    if ray_length == 0.0 {
        return output;
    }
    let v = u + input.max_fraction * dr.dot(axis);
    if (u < -r && v < -r) || (length + r < u && length + r < v) {
        return output;
    }
    let a12 = axis.dot(ray_axis);
    let det = 1.0 - a12 * a12;
    let tr = if det < FLT_EPSILON {
        let perp = ray_axis.mul_sub(a12, axis);
        let perp2 = perp.length_sq();
        let beta = sc.dot(perp);
        let gamma = sc2 - r * r;
        let disc = beta * beta - perp2 * gamma;
        if beta >= 0.0 || disc < 0.0 {
            return output;
        }
        gamma / (-beta + disc.sqrt())
    } else {
        let inv_det = 1.0 / det;
        let sa2 = s.dot(ray_axis);
        let t1 = (u - a12 * sa2) * inv_det;
        let t2 = (a12 * u - sa2) * inv_det;
        let p1 = axis.scale(t1);
        let p2 = s.mul_add(t2, ray_axis);
        let g2 = p2.sub(p1).length_sq();
        if g2 > r * r {
            return output;
        }
        let h = ((r * r - g2) * inv_det).sqrt();
        t2 - h
    };
    if tr < 0.0 || input.max_fraction * ray_length < tr {
        return output;
    }
    let tc = u + tr * a12;
    if tc < 0.0 {
        return ray_cast_sphere(
            &Sphere {
                center: c1,
                radius: r,
            },
            input,
        );
    }
    if length < tc {
        return ray_cast_sphere(
            &Sphere {
                center: c2,
                radius: r,
            },
            input,
        );
    }
    let p = s.mul_add(tr, ray_axis);
    output.point = c1.add(p);
    output.normal = p.mul_sub(tc, axis).normalize();
    output.fraction = clampf(tr / ray_length, 0.0, input.max_fraction);
    output.hit = true;
    output
}

pub fn ray_cast_hull(shape: &HullData, input: &RayCastInput) -> CastOutput {
    let mut output = CastOutput {
        triangle_index: 0,
        ..CastOutput::MISS
    };
    let mut lower = 0.0;
    let mut upper = input.max_fraction;
    let mut best_face = None;
    for (index, plane) in shape.planes[..shape.face_count].iter().enumerate() {
        let distance = plane.offset - plane.normal.dot(input.origin);
        let denominator = plane.normal.dot(input.translation);
        if denominator == 0.0 {
            if distance < 0.0 {
                return output;
            }
        } else {
            let fraction = distance / denominator;
            if denominator < 0.0 {
                if fraction > lower {
                    best_face = Some(index);
                    lower = fraction;
                }
            } else if fraction < upper {
                upper = fraction;
            }
            if upper < lower {
                return output;
            }
        }
    }
    if let Some(index) = best_face {
        output.point = input.origin.add(input.translation.scale(lower));
        output.normal = shape.planes[index].normal;
        output.fraction = lower;
    } else {
        output.point = input.origin;
    }
    output.hit = true;
    output
}

fn round_mover_plane(
    point: Vec3,
    closest: Vec3,
    total_radius: f32,
    mover: &Capsule,
) -> Option<PlaneResult> {
    let (mut normal, mut distance) = get_length_and_normalize(closest.sub(point));
    if distance > total_radius {
        return None;
    }
    if distance < 0.005 {
        let (axis, length) = get_length_and_normalize(mover.center2.sub(mover.center1));
        normal = if length > 0.005 {
            axis.perp()
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        distance = 0.0;
    }
    Some(PlaneResult {
        plane: Plane {
            normal,
            offset: total_radius - distance,
        },
        point,
        triangle_index: 0,
        child_index: 0,
        material_index: 0,
    })
}

pub fn collide_mover_sphere(shape: &Sphere, mover: &Capsule) -> Option<PlaneResult> {
    let closest = point_to_segment_distance(mover.center1, mover.center2, shape.center);
    round_mover_plane(shape.center, closest, mover.radius + shape.radius, mover)
}

pub fn collide_mover_capsule(shape: &Capsule, mover: &Capsule) -> Option<PlaneResult> {
    let approach = segment_distance(shape.center1, shape.center2, mover.center1, mover.center2);
    round_mover_plane(
        approach.point1,
        approach.point2,
        mover.radius + shape.radius,
        mover,
    )
}

pub fn collide_mover_hull(shape: &HullData, mover: &Capsule) -> Option<PlaneResult> {
    let input = DistanceInput {
        proxy_a: ShapeProxy {
            points: shape.points,
            count: shape.vertex_count,
            radius: 0.0,
        },
        proxy_b: ShapeProxy {
            points: mover.points(),
            count: 2,
            radius: mover.radius,
        },
        transform: Transform::IDENTITY,
        use_radii: false,
    };
    let output = shape_distance(&input, &mut SimplexCache::empty());
    // Box3D refuses deep hull overlap to keep behavior consistent with a hull converted to a mesh.
    if output.distance == 0.0 || output.distance > mover.radius {
        return None;
    }
    Some(PlaneResult {
        plane: Plane {
            normal: output.normal,
            offset: mover.radius - output.distance,
        },
        point: output.point_a,
        triangle_index: 0,
        child_index: 0,
        material_index: 0,
    })
}
