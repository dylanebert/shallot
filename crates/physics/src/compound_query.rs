//! Box3D compound.c queries over the uploaded inner tree and child records.
use crate::distance::{CastOutput, ShapeProxy};
use crate::manifold::Capsule;
use crate::math::{Mat3, Quat, Transform, Vec3};
use crate::mesh_query::proxy_bounds;
use crate::query::{
    collide_mover_local, overlap_shape, ray_cast_shape, shape_cast_local, PlaneResult,
    RayCastInput, Shape, ShapeCastInput,
};
#[derive(Clone, Copy)]
pub struct Compound<'a> {
    pub root: i32,
    pub nodes: &'a [u32],
    data: &'a [u32],
}
impl<'a> Compound<'a> {
    pub unsafe fn from_pointer(pointer: *const u32) -> Self {
        let words = *pointer.add(2) as usize / 4;
        let data = core::slice::from_raw_parts(pointer, words);
        let capacity = *pointer.add(9) as usize;
        let node_offset = *pointer.add(3) as usize / 4;
        Self {
            root: *pointer.add(7) as i32,
            nodes: core::slice::from_raw_parts(pointer.add(node_offset), capacity * 12),
            data,
        }
    }
}
fn vec(r: &[u32], i: usize) -> Vec3 {
    Vec3::new(
        f32::from_bits(r[i]),
        f32::from_bits(r[i + 1]),
        f32::from_bits(r[i + 2]),
    )
}
fn section(c: Compound, offset_word: usize, count_word: usize, stride: usize) -> &'static [u32] {
    let offset = c.data[offset_word] as usize / 4;
    let count = c.data[count_word] as usize * stride;
    unsafe { core::slice::from_raw_parts(c.data.as_ptr().add(offset), count) }
}
pub(crate) unsafe fn child_material_index(
    pointer: *const u32,
    index: usize,
    triangle: usize,
) -> u32 {
    let c = Compound::from_pointer(pointer);
    let capsules = c.data[20] as usize;
    let hulls = c.data[22] as usize;
    let meshes = c.data[25] as usize;
    if index < capsules {
        section(c, 19, 20, 8)[index * 8 + 7]
    } else if index < capsules + hulls {
        section(c, 21, 22, 9)[(index - capsules) * 9 + 8]
    } else if index < capsules + hulls + meshes {
        let r = &section(c, 24, 25, 15)[(index - capsules - hulls) * 15..];
        let mesh = c
            .data
            .as_ptr()
            .cast::<u8>()
            .add(r[10] as usize)
            .cast::<u32>();
        let triangle_material = crate::geo::mesh_view(mesh, vec(r, 7)).materials[triangle] as usize;
        r[11 + triangle_material.min(3)]
    } else {
        section(c, 27, 28, 5)[(index - capsules - hulls - meshes) * 5 + 4]
    }
}
pub(crate) unsafe fn child_words(pointer: *const u32, index: usize) -> [u32; 19] {
    let c = Compound::from_pointer(pointer);
    let capsules = c.data[20] as usize;
    let hulls = c.data[22] as usize;
    let meshes = c.data[25] as usize;
    let mut out = [0u32; 19];
    out[7] = 1.0f32.to_bits();
    let set_transform = |out: &mut [u32; 19], p: Vec3, q: Quat| {
        out[1..4].copy_from_slice(&[p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]);
        out[4..7].copy_from_slice(&[q.v.x.to_bits(), q.v.y.to_bits(), q.v.z.to_bits()]);
        out[7] = q.s.to_bits();
    };
    if index < capsules {
        let r = &section(c, 19, 20, 8)[index * 8..];
        out[0] = 0;
        out[12..19].copy_from_slice(&[r[0], r[1], r[2], r[3], r[4], r[5], r[6]]);
        out[8] = r[7];
    } else if index < capsules + hulls {
        let r = &section(c, 21, 22, 9)[(index - capsules) * 9..];
        out[0] = 3;
        set_transform(
            &mut out,
            vec(r, 0),
            Quat {
                v: vec(r, 3),
                s: f32::from_bits(r[6]),
            },
        );
        out[8] = r[8];
        out[12] = c.data.as_ptr().cast::<u8>().add(r[7] as usize) as u32;
        out[13] = *((out[12] as *const u32).add(12));
    } else if index < capsules + hulls + meshes {
        let r = &section(c, 24, 25, 15)[(index - capsules - hulls) * 15..];
        out[0] = 4;
        set_transform(
            &mut out,
            vec(r, 0),
            Quat {
                v: vec(r, 3),
                s: f32::from_bits(r[6]),
            },
        );
        out[8..12].copy_from_slice(&r[11..15]);
        out[12] = c.data.as_ptr().cast::<u8>().add(r[10] as usize) as u32;
        out[13..16].copy_from_slice(&r[7..10]);
    } else {
        let r = &section(c, 27, 28, 5)[(index - capsules - hulls - meshes) * 5..];
        out[0] = 5;
        out[12..16].copy_from_slice(&r[..4]);
        out[8] = r[4];
    }
    out
}
pub(crate) fn child(c: Compound, index: usize) -> (Shape<'static>, Transform, [i32; 4]) {
    let capsules = c.data[20] as usize;
    let hulls = c.data[22] as usize;
    let meshes = c.data[25] as usize;
    let (shape, xf, materials) = unsafe {
        if index < capsules {
            let r = &section(c, 19, 20, 8)[index * 8..];
            (
                Shape::Capsule(&*r.as_ptr().cast::<Capsule>()),
                Transform::IDENTITY,
                [r[7] as i32, 0, 0, 0],
            )
        } else if index < capsules + hulls {
            let r = &section(c, 21, 22, 9)[(index - capsules) * 9..];
            let xf = Transform {
                p: vec(r, 0),
                q: Quat {
                    v: vec(r, 3),
                    s: f32::from_bits(r[6]),
                },
            };
            let ptr = c
                .data
                .as_ptr()
                .cast::<u8>()
                .add(r[7] as usize)
                .cast::<u32>();
            (
                Shape::Hull(crate::geo::hull_view(ptr as usize)),
                xf,
                [r[8] as i32, 0, 0, 0],
            )
        } else if index < capsules + hulls + meshes {
            let r = &section(c, 24, 25, 15)[(index - capsules - hulls) * 15..];
            let xf = Transform {
                p: vec(r, 0),
                q: Quat {
                    v: vec(r, 3),
                    s: f32::from_bits(r[6]),
                },
            };
            let scale = vec(r, 7);
            let ptr = c
                .data
                .as_ptr()
                .cast::<u8>()
                .add(r[10] as usize)
                .cast::<u32>();
            (
                Shape::Mesh(crate::geo::mesh_view(ptr, scale)),
                xf,
                [r[11] as i32, r[12] as i32, r[13] as i32, r[14] as i32],
            )
        } else {
            let r = &section(c, 27, 28, 5)[(index - capsules - hulls - meshes) * 5..];
            (
                Shape::Sphere(&*r.as_ptr().cast::<crate::manifold::Sphere>()),
                Transform::IDENTITY,
                [r[4] as i32, 0, 0, 0],
            )
        }
    };
    (shape, xf, materials)
}
fn cast(
    c: Compound,
    lower: Vec3,
    upper: Vec3,
    translation: Vec3,
    fraction: f32,
    box_cast: bool,
    mut visit: impl FnMut(usize, f32) -> f32,
) {
    if c.root < 0 {
        return;
    }
    crate::tree::cast::<16, _>(
        c.nodes,
        c.root,
        c.data[8] as usize,
        lower,
        upper,
        translation,
        fraction,
        u32::MAX,
        u32::MAX,
        false,
        box_cast,
        |fraction, _, index| visit(index as usize, fraction),
    );
}
pub(crate) fn query(c: Compound, lower: Vec3, upper: Vec3, visit: impl FnMut(i32, u32) -> bool) {
    if c.root < 0 {
        return;
    }
    let mut stack = [0i32; crate::tree::STACK_SIZE];
    crate::tree::query_packed::<16, _>(
        c.nodes,
        c.root,
        c.data[8] as usize,
        [lower.x, lower.y, lower.z],
        [upper.x, upper.y, upper.z],
        u32::MAX,
        u32::MAX,
        false,
        &mut stack,
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
        false,
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
            if out.hit {
                out.material_index = materials[out.material_index.min(3) as usize];
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
        true,
        |index, fraction| {
            let (shape, xf, materials) = child(c, index);
            // Box3D uses the inverse transform's rotation matrix here, rather than inverse-point
            // quaternion math used by b3ShapeCastShape. Preserve that arithmetic path.
            let inv = xf.invert();
            let matrix = Mat3::from_quat(inv.q);
            let mut points = [core::mem::MaybeUninit::<Vec3>::uninit(); 128];
            let count = input.proxy.count.min(128);
            for (i, p) in points[..count].iter_mut().enumerate() {
                p.write(matrix.mul_v(input.proxy.points[i]).add(inv.p));
            }
            // The inverse-matrix path initializes exactly the proxy's active prefix.
            let points =
                unsafe { core::slice::from_raw_parts(points.as_ptr().cast::<Vec3>(), count) };
            let local = ShapeCastInput {
                proxy: ShapeProxy {
                    points,
                    count,
                    radius: input.proxy.radius,
                },
                translation: matrix.mul_v(input.translation),
                max_fraction: fraction,
                can_encroach: input.can_encroach,
            };
            let mut out = shape_cast_local(&shape, &local);
            if out.hit {
                out.material_index = materials[out.material_index.min(3) as usize];
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
