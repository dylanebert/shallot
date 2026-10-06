//! Per-shape query ABI. Inputs live in a fixed scratch record; mover outputs use caller-owned memory.
use crate::compound_query::Compound;
use crate::distance::{CastOutput, ShapeProxy};
use crate::manifold::{Capsule, Sphere};
use crate::math::{Quat, Transform, Vec3};
use crate::query::*;
use crate::shapes::{SHAPE_STRIDE, S_GEOM, S_GEO_REFERENCE, S_MATERIAL_COUNT, S_TYPE};
// transform(7), count/radius(2), translation(3), fraction/encroach(2), points(128*3).
static mut INPUT: [f32; 398] = [0.0; 398];
static mut OUTPUT: [f32; 12] = [0.0; 12];
#[export_name = "shapeQueryInputPtr"]
pub extern "C" fn input_ptr() -> *mut f32 {
    &raw mut INPUT as *mut f32
}
#[export_name = "shapeQueryOutputPtr"]
pub extern "C" fn output_ptr() -> *const f32 {
    &raw const OUTPUT as *const f32
}
fn vec(r: &[u32], i: usize) -> Vec3 {
    Vec3::new(
        f32::from_bits(r[i]),
        f32::from_bits(r[i + 1]),
        f32::from_bits(r[i + 2]),
    )
}
pub(crate) unsafe fn geometry(kind: u32, r: &[u32]) -> Shape<'static> {
    match kind {
        0 => Shape::Capsule(Capsule {
            center1: vec(r, 0),
            center2: vec(r, 3),
            radius: f32::from_bits(r[6]),
        }),
        5 => Shape::Sphere(Sphere {
            center: vec(r, 0),
            radius: f32::from_bits(r[3]),
        }),
        3 => Shape::Hull(crate::geo::hull_view(r[0] as usize)),
        4 => Shape::Mesh(crate::geo::mesh_view(r[0] as usize, vec(r, 1))),
        2 => Shape::Height(crate::geo::height_view(r[0] as usize)),
        1 => {
            let p = crate::geo::extra_ptr(r[0] as usize);
            Shape::Compound(Compound {
                root: *p as i32,
                nodes: core::slice::from_raw_parts(
                    crate::geo::extra_ptr(*p.add(3) as usize),
                    *p.add(1) as usize * 12,
                ),
                children: core::slice::from_raw_parts(
                    crate::geo::extra_ptr(*p.add(4) as usize),
                    *p.add(2) as usize * 19,
                ),
            })
        }
        _ => unreachable!(),
    }
}
pub(crate) unsafe fn shape(world: usize, id: usize) -> (Shape<'static>, i32) {
    crate::shapes::shape_set_active_world(world as u32);
    active_shape(id)
}
pub(crate) unsafe fn active_shape(id: usize) -> (Shape<'static>, i32) {
    let col = crate::shapes::col();
    let o = id * SHAPE_STRIDE;
    let kind = col.get(o + S_TYPE);
    let mut r = [0; 7];
    for (i, x) in r.iter_mut().enumerate() {
        *x = col.get(o + S_GEOM + i);
    }
    if kind == 3 || kind == 2 || kind == 1 {
        r[0] = col.get(o + S_GEO_REFERENCE);
    }
    if kind == 4 {
        r = [col.get(o + S_GEO_REFERENCE), r[0], r[1], r[2], 0, 0, 0];
    }
    (geometry(kind, &r), col.get(o + S_MATERIAL_COUNT) as i32)
}
pub(crate) unsafe fn input() -> (&'static [f32], Transform, ShapeProxy<'static>) {
    let r = core::slice::from_raw_parts(&raw const INPUT as *const f32, 398);
    let xf = Transform {
        p: Vec3::new(r[0], r[1], r[2]),
        q: Quat {
            v: Vec3::new(r[3], r[4], r[5]),
            s: r[6],
        },
    };
    let points = core::slice::from_raw_parts(r.as_ptr().add(14) as *const Vec3, 128);
    (
        r,
        xf,
        ShapeProxy {
            points,
            count: (r[7] as usize).min(128),
            radius: r[8],
        },
    )
}
pub(crate) unsafe fn output(out: CastOutput) {
    let p = &raw mut OUTPUT as *mut f32;
    let values = [
        u32::from(out.hit) as f32,
        out.fraction,
        out.point.x,
        out.point.y,
        out.point.z,
        out.normal.x,
        out.normal.y,
        out.normal.z,
        out.iterations as f32,
        out.triangle_index as f32,
        out.child_index as f32,
        out.material_index as f32,
    ];
    core::ptr::copy_nonoverlapping(values.as_ptr(), p, 12);
}
#[export_name = "shapeQueryCompound"]
pub unsafe extern "C" fn query_compound(
    world: usize,
    id: usize,
    lx: f32,
    ly: f32,
    lz: f32,
    ux: f32,
    uy: f32,
    uz: f32,
) {
    let (Shape::Compound(compound), _) = shape(world, id) else {
        unreachable!()
    };
    crate::compound_query::query(
        compound,
        Vec3::new(lx, ly, lz),
        Vec3::new(ux, uy, uz),
        |_, child| crate::world_query::callback(0, child as usize, core::ptr::null(), 0) != 0.0,
    );
}
#[export_name = "shapeQueryRay"]
pub extern "C" fn ray(world: usize, id: usize, local: u32) {
    unsafe {
        let (shape, _) = shape(world, id);
        let (r, xf, proxy) = input();
        let input = RayCastInput {
            origin: proxy.points[0],
            translation: Vec3::new(r[9], r[10], r[11]),
            max_fraction: r[12],
        };
        output(if local != 0 {
            ray_cast_local(&shape, &input)
        } else {
            ray_cast_shape(&shape, xf, &input)
        });
    }
}
#[export_name = "shapeQueryCast"]
pub extern "C" fn cast(world: usize, id: usize, local: u32) {
    unsafe {
        let (shape, _) = shape(world, id);
        let (r, xf, proxy) = input();
        let input = ShapeCastInput {
            proxy,
            translation: Vec3::new(r[9], r[10], r[11]),
            max_fraction: r[12],
            can_encroach: r[13] != 0.0,
        };
        output(if local != 0 {
            shape_cast_local(&shape, &input)
        } else {
            shape_cast_shape(&shape, xf, &input)
        });
    }
}
#[export_name = "shapeQueryOverlap"]
pub extern "C" fn overlap(world: usize, id: usize) -> u32 {
    unsafe {
        let (shape, _) = shape(world, id);
        let (_, xf, proxy) = input();
        u32::from(overlap_shape(&shape, xf, proxy))
    }
}
/// # Safety
/// `planes` must address capacity writable PlaneResult records (40 bytes each) in linear memory.
#[export_name = "shapeQueryMover"]
pub unsafe extern "C" fn mover(
    world: usize,
    id: usize,
    planes: *mut PlaneResult,
    capacity: usize,
    local: u32,
) -> usize {
    if capacity == 0 {
        return 0;
    }
    let (shape, materials) = shape(world, id);
    let (_, xf, proxy) = input();
    let mover = Capsule {
        center1: proxy.points[0],
        center2: proxy.points[1],
        radius: proxy.radius,
    };
    let planes = core::slice::from_raw_parts_mut(planes, capacity);
    if local != 0 {
        collide_mover_local(planes, &shape, &mover)
    } else {
        collide_mover(planes, &shape, xf, &mover, materials)
    }
}
