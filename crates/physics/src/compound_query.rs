//! Box3D compound.c queries over the uploaded inner tree and child records.
use crate::distance::{CastOutput, ShapeProxy};
use crate::manifold::Capsule;
use crate::math::{Mat3, Quat, Transform, Vec3};
use crate::mesh_query::{bounds_overlap, bounds_ray_overlap, max, min, proxy_bounds};
use crate::query::{
    collide_mover_local, overlap_shape, ray_cast_shape, shape_cast_local, PlaneResult,
    RayCastInput, Shape, ShapeCastInput,
};
#[derive(Clone, Copy)]
pub struct Compound<'a> {
    pub root: i32,
    pub nodes: &'a [u32],
    pub children: &'a [u32],
}
fn vec(r: &[u32], i: usize) -> Vec3 {
    Vec3::new(
        f32::from_bits(r[i]),
        f32::from_bits(r[i + 1]),
        f32::from_bits(r[i + 2]),
    )
}
pub(crate) fn child(c: Compound, index: usize) -> (Shape<'static>, Transform, [i32; 4]) {
    let r = &c.children[index * 19..index * 19 + 19];
    let xf = Transform {
        p: vec(r, 1),
        q: Quat {
            v: vec(r, 4),
            s: f32::from_bits(r[7]),
        },
    };
    let shape = unsafe { crate::query_abi::geometry(r[0], &r[12..]) };
    (
        shape,
        xf,
        [r[8] as i32, r[9] as i32, r[10] as i32, r[11] as i32],
    )
}
fn bounds(c: Compound, index: usize) -> (Vec3, Vec3) {
    let r = &c.nodes[index * 12..];
    (vec(r, 0), vec(r, 3))
}
fn cast(
    c: Compound,
    lower: Vec3,
    upper: Vec3,
    translation: Vec3,
    mut fraction: f32,
    mut visit: impl FnMut(usize, f32) -> f32,
) {
    if c.root < 0 {
        return;
    }
    let start = lower.add(upper).scale(0.5);
    let extent = upper.sub(start);
    let mut stack = [0usize; 1024];
    let mut count = 1;
    stack[0] = c.root as usize;
    while count > 0 {
        count -= 1;
        let index = stack[count];
        let r = &c.nodes[index * 12..];
        let (lo, hi) = bounds(c, index);
        let t = translation.scale(fraction);
        if !bounds_overlap(lo, hi, min(lower, lower.add(t)), max(upper, upper.add(t)))
            || !bounds_ray_overlap(lo.sub(extent), hi.add(extent), start, translation)
        {
            continue;
        }
        if r[11] & 4 != 0 {
            let value = visit(r[8] as usize, fraction);
            if value == 0.0 {
                return;
            }
            if value > 0.0 && value <= fraction {
                fraction = value;
            }
        } else if count < 1023 {
            let a = r[8] as usize;
            let b = r[9] as usize;
            let (a1, a2) = bounds(c, a);
            let (b1, b2) = bounds(c, b);
            let (far, near) = if a1.add(a2).scale(0.5).sub(start).length_sq()
                < b1.add(b2).scale(0.5).sub(start).length_sq()
            {
                (b, a)
            } else {
                (a, b)
            };
            stack[count] = far;
            stack[count + 1] = near;
            count += 2;
        }
    }
}
pub(crate) fn query(c: Compound, lower: Vec3, upper: Vec3, visit: impl FnMut(i32, u32) -> bool) {
    crate::tree::query(
        c.nodes,
        c.root,
        c.nodes.len() / 12,
        [lower.x, lower.y, lower.z],
        [upper.x, upper.y, upper.z],
        u32::MAX,
        u32::MAX,
        false,
        &mut [0; 1024],
        visit,
    );
}
pub fn ray_cast_compound(c: Compound, input: &RayCastInput) -> CastOutput {
    let mut result = CastOutput {
        triangle_index: 0,
        ..CastOutput::MISS
    };
    cast(
        c,
        input.origin,
        input.origin,
        input.translation,
        input.max_fraction,
        |index, fraction| {
            let (shape, xf, materials) = child(c, index);
            let mut out = ray_cast_shape(
                &shape,
                xf,
                &RayCastInput {
                    max_fraction: fraction,
                    ..*input
                },
            );
            out.material_index = materials[out.material_index.min(3) as usize];
            if out.hit {
                out.child_index = index as i32;
                result = out;
                out.fraction
            } else {
                fraction
            }
        },
    );
    result
}
pub fn shape_cast_compound(c: Compound, input: &ShapeCastInput) -> CastOutput {
    let mut result = CastOutput {
        triangle_index: 0,
        ..CastOutput::MISS
    };
    if input.proxy.count == 0 {
        return result;
    }
    let (lower, upper) = proxy_bounds(input.proxy);
    cast(
        c,
        lower,
        upper,
        input.translation,
        input.max_fraction,
        |index, fraction| {
            let (shape, xf, materials) = child(c, index);
            // Box3D uses the inverse transform's rotation matrix here, rather than inverse-point
            // quaternion math used by b3ShapeCastShape. Preserve that arithmetic path.
            let inv = xf.invert();
            let matrix = Mat3::from_quat(inv.q);
            let mut points = [Vec3::ZERO; 128];
            let count = input.proxy.count.min(128);
            for (i, p) in points[..count].iter_mut().enumerate() {
                *p = matrix.mul_v(input.proxy.points[i]).add(inv.p);
            }
            let local = ShapeCastInput {
                proxy: ShapeProxy {
                    points: &points,
                    count,
                    radius: input.proxy.radius,
                },
                translation: matrix.mul_v(input.translation),
                max_fraction: fraction,
                can_encroach: input.can_encroach,
            };
            let mut out = shape_cast_local(&shape, &local);
            out.material_index = materials[out.material_index.min(3) as usize];
            if out.hit {
                out.point = xf.point(out.point);
                out.normal = xf.q.rotate(out.normal);
                out.child_index = index as i32;
                result = out;
                out.fraction
            } else {
                fraction
            }
        },
    );
    result
}
pub fn overlap_compound(c: Compound, transform: Transform, proxy: ShapeProxy) -> bool {
    let (lower, upper) = proxy_bounds(proxy);
    let mut overlap = false;
    query(c, lower, upper, |_, index| {
        let (shape, xf, _) = child(c, index as usize);
        overlap = overlap_shape(&shape, transform.mul(xf), proxy);
        !overlap
    });
    overlap
}
pub fn collide_mover_compound(planes: &mut [PlaneResult], c: Compound, mover: &Capsule) -> usize {
    if planes.is_empty() {
        return 0;
    }
    let points = [mover.center1, mover.center2];
    let (lower, upper) = proxy_bounds(ShapeProxy {
        points: &points,
        count: 2,
        radius: mover.radius,
    });
    let mut count = 0;
    query(c, lower, upper, |_, index| {
        let (shape, xf, materials) = child(c, index as usize);
        let local = Capsule {
            center1: xf.inv_point(mover.center1),
            center2: xf.inv_point(mover.center2),
            radius: mover.radius,
        };
        let n = collide_mover_local(&mut planes[count..], &shape, &local);
        for p in &mut planes[count..count + n] {
            p.plane.normal = xf.q.rotate(p.plane.normal);
            p.point = xf.point(p.point);
            p.child_index = index as i32;
            p.material_index = materials[p.material_index.min(3) as usize];
        }
        count += n;
        count < planes.len()
    });
    count
}
