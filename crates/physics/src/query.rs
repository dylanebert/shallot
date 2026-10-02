//! Per-shape queries ported from Box3D sphere.c, capsule.c and hull.c.

use crate::distance::{
    shape_cast, shape_distance, CastOutput, DistanceInput, ShapeCastPairInput, ShapeProxy,
    SimplexCache,
};
use crate::hull::HullData;
use crate::manifold::{Capsule, Sphere};
use crate::math::{
    clampf, get_length_and_normalize, point_to_segment_distance, segment_distance, Plane,
    Transform, Vec3, FLT_EPSILON,
};

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

#[derive(Clone, Copy)]
pub struct PlaneResult {
    pub plane: Plane,
    pub point: Vec3,
    pub triangle_index: i32,
    pub child_index: i32,
    pub material_index: i32,
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
    let points = [mover.center1, mover.center2];
    let input = DistanceInput {
        proxy_a: ShapeProxy {
            points: shape.points,
            count: shape.vertex_count,
            radius: 0.0,
        },
        proxy_b: ShapeProxy {
            points: &points,
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
