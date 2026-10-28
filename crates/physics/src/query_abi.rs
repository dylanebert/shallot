//! Per-shape query ABI. Inputs live in a fixed scratch record; mover outputs use caller-owned memory.
use crate::compound_query::Compound;
use crate::distance::{CastOutput, ShapeProxy};
use crate::finalize::{TY_CAPSULE, TY_COMPOUND, TY_HEIGHT_FIELD, TY_HULL, TY_MESH, TY_SPHERE};
use crate::manifold::{Capsule, Sphere};
use crate::math::{Quat, Transform, Vec3};
use crate::query::*;

/// Finds a touching pair in body A's contact adjacency list. Results retain the solver's A/B order.
#[export_name = "contactQuery"]
pub unsafe extern "C" fn contact_query(world: usize, shape_a: usize, shape_b: usize) -> usize {
    use crate::manifold_abi::*;
    let shapes = crate::shapes::col(world);
    let body = shapes.get(shape_a * SHAPE_STRIDE + crate::shapes::S_QUERY_BODY) as usize;
    let dir = crate::manifolds::dir_col(world);
    let mut key = crate::bodies::record(world, body).head_contact_key;
    while key != -1 {
        let offset = (key >> 1) as usize * DIR_STRIDE;
        let a = dir.get(offset + DIR_SHAPE_A) as usize;
        let b = dir.get(offset + DIR_SHAPE_B) as usize;
        if ((a == shape_a && b == shape_b) || (a == shape_b && b == shape_a))
            && dir.get(offset + DIR_FLAGS) & contact_flags::TOUCHING != 0
        {
            let result = &raw mut crate::world_query::RESULT as *mut u32;
            result.write(a as u32);
            result.add(1).write(b as u32);
            result.add(2).write(dir.get(offset + DIR_MANIFOLD_COUNT));
            return dir.get(offset + DIR_MANIFOLD_BASE) as usize;
        }
        key = dir.get(offset + DIR_EDGE_A + 2 + 3 * (key & 1) as usize) as i32;
    }
    0
}
use crate::shapes::{SHAPE_STRIDE, S_GEOM, S_MATERIAL_COUNT, S_TYPE};
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
        // Resident records use the repr(C) primitive layouts and u32/f32 alignment.
        TY_CAPSULE => Shape::Capsule(&*r.as_ptr().cast::<Capsule>()),
        TY_SPHERE => Shape::Sphere(&*r.as_ptr().cast::<Sphere>()),
        TY_HULL => Shape::Hull(crate::geo::hull_view(r[0] as usize)),
        TY_MESH => Shape::Mesh(crate::geo::mesh_view(r[0] as *const u32, vec(r, 1))),
        TY_HEIGHT_FIELD => Shape::Height(crate::geo::height_view(r[0] as *const u32)),
        TY_COMPOUND => Shape::Compound(Compound::from_pointer(r[0] as *const u32)),
        _ => unreachable!(),
    }
}
pub(crate) unsafe fn shape(world: usize, id: usize) -> (Shape<'static>, i32) {
    active_shape(world, id)
}
pub(crate) unsafe fn active_shape(world_index: usize, id: usize) -> (Shape<'static>, i32) {
    let col = crate::shapes::col(world_index);
    let o = id * SHAPE_STRIDE;
    let kind = col.get(o + S_TYPE);
    let r = &crate::shapes::col_slice(world_index)[o + S_GEOM..o + S_GEOM + 7];
    (geometry(kind, r), col.get(o + S_MATERIAL_COUNT) as i32)
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
    unsafe { query_compound_in_world(world, id, lx, ly, lz, ux, uy, uz) }
}

pub unsafe extern "C" fn query_compound_in_world(
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
    ray_in_world(world, id, local)
}

pub extern "C" fn ray_in_world(world: usize, id: usize, local: u32) {
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
    cast_in_world(world, id, local)
}

pub extern "C" fn cast_in_world(world: usize, id: usize, local: u32) {
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
    overlap_in_world(world, id)
}

pub extern "C" fn overlap_in_world(world: usize, id: usize) -> u32 {
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
    unsafe { mover_in_world(world, id, planes, capacity, local) }
}

pub unsafe extern "C" fn mover_in_world(
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
