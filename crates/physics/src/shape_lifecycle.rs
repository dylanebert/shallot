//! Nongeometry shape lifecycle in Box3D's body/shape list and proxy order.
use crate::{bodies, body, regions, shapes};

#[export_name = "shapeCapsuleType"]
pub extern "C" fn capsule_type(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> u32 {
    // b3CreateCapsuleShape collapses a capsule shorter than linear slop to a sphere.
    let x = bx - ax;
    let y = by - ay;
    let z = bz - az;
    let slop = 0.005f32;
    if (x * x + y * y) + z * z <= slop * slop {
        5
    } else {
        0
    }
}
#[export_name = "shapeLink"]
pub unsafe extern "C" fn link(world: usize, id: usize, body_id: usize) {
    regions::select(world as u32);
    let body = bodies::record_mut(world, body_id);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    u.set(o + 29, body_id as u32);
    u.set(o + shapes::S_PREV, u32::MAX);
    u.set(o + shapes::S_NEXT, body.head_shape_id as u32);
    if body.head_shape_id != -1 {
        u.set(
            body.head_shape_id as usize * shapes::SHAPE_STRIDE + shapes::S_PREV,
            id as u32,
        );
    }
    body.head_shape_id = id as i32;
    body.shape_count += 1;
    bodies::column(body_id, 5, body::SIM2_STRIDE)
        .set(body::S2_HEAD_SHAPE, f32::from_bits(id as u32));
}
#[export_name = "shapeUnlink"]
pub unsafe extern "C" fn unlink(world: usize, id: usize) {
    regions::select(world as u32);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    let body_id = u.get(o + 29) as usize;
    let prev = u.get(o + shapes::S_PREV);
    let next = u.get(o + shapes::S_NEXT);
    if prev != u32::MAX {
        u.set(prev as usize * shapes::SHAPE_STRIDE + shapes::S_NEXT, next);
    }
    if next != u32::MAX {
        u.set(next as usize * shapes::SHAPE_STRIDE + shapes::S_PREV, prev);
    }
    let body = bodies::record_mut(world, body_id);
    if body.head_shape_id == id as i32 {
        body.head_shape_id = next as i32;
    }
    body.shape_count -= 1;
    bodies::column(body_id, 5, body::SIM2_STRIDE).set(
        body::S2_HEAD_SHAPE,
        f32::from_bits(body.head_shape_id as u32),
    );
}
/** Unlink one shape for the host to release its still-host-owned geometry and sensor payloads. */
#[export_name = "shapeBodyTake"]
pub unsafe extern "C" fn body_take(world: usize, body: usize) -> i32 {
    regions::select(world as u32);
    let shape = bodies::record(world, body).head_shape_id;
    if shape != -1 {
        unlink(world, shape as usize);
    }
    shape
}
#[export_name = "shapeBodyAllowsType"]
pub unsafe extern "C" fn body_allows_type(world: usize, body: usize, kind: u32) -> bool {
    regions::select(world as u32);
    if kind == 0 {
        return true;
    }
    let mut shape = bodies::record(world, body).head_shape_id;
    while shape != -1 {
        let o = shape as usize * shapes::SHAPE_STRIDE;
        let u = shapes::col();
        let kind = u.get(o + shapes::S_TYPE);
        if kind == 1 || kind == 2 {
            return false;
        }
        shape = u.get(o + shapes::S_NEXT) as i32;
    }
    true
}
#[export_name = "shapeQueryPose"]
pub unsafe extern "C" fn query_pose(world: usize, shape: usize, body_id: usize) {
    regions::select(world as u32);
    let record = bodies::record(world, body_id);
    let o = shape * shapes::SHAPE_STRIDE;
    let columns = shapes::col();
    columns.set(
        o + 32,
        if record.set_index == 2 {
            record.local_index as u32 + 1
        } else {
            0
        },
    );
    if record.set_index == 2 {
        return;
    }
    let sim = bodies::column(body_id, 1, body::SIM_STRIDE);
    let fin = bodies::column(body_id, 2, body::FIN_STRIDE);
    let sim2 = bodies::column(body_id, 5, body::SIM2_STRIDE);
    columns.set(o + 42, sim2.get(body::S2_FLAGS).to_bits());
    for lane in 0..6 {
        columns.set(o + 44 + lane, fin.get(lane).to_bits());
    }
    for lane in 0..3 {
        columns.set(o + 18 + lane, fin.get(9 + lane).to_bits());
    }
    for lane in 0..4 {
        columns.set(o + 21 + lane, sim.get(28 + lane).to_bits());
    }
}
#[export_name = "shapeSyncBody"]
pub unsafe extern "C" fn sync_body(world: usize, id: usize) {
    regions::select(world as u32);
    let mut shape = bodies::record(world, id).head_shape_id;
    while shape != -1 {
        query_pose(world, shape as usize, id);
        shape = shapes::col().get(shape as usize * shapes::SHAPE_STRIDE + shapes::S_NEXT) as i32;
    }
}
#[export_name = "shapeSyncBodyBounds"]
pub unsafe extern "C" fn sync_body_bounds(world: usize, body_id: usize) {
    regions::select(world as u32);
    let mut shape = bodies::record(world, body_id).head_shape_id;
    while shape != -1 {
        body_record_bounds(world, body_id, shape as usize);
        query_pose(world, shape as usize, body_id);
        shape = shapes::col().get(shape as usize * shapes::SHAPE_STRIDE + shapes::S_NEXT) as i32;
    }
}
unsafe fn body_record_bounds(world: usize, id: usize, shape: usize) {
    regions::select(world as u32);
    let pose = bodies::geometry(id).0;
    let tight = crate::continuous::bounds(shape, pose);
    let fat = crate::fataabb::col();
    let mut previous = [0.0; 6];
    for lane in 0..6 {
        previous[lane] = fat.get(shape * 6 + lane);
    }
    let (bounds, escaped) = crate::finalize::refit_bounds(tight, &previous);
    let f = shapes::col_f();
    let o = shape * shapes::SHAPE_STRIDE;
    for lane in 0..6 {
        f.set(o + 34 + lane, bounds[lane]);
    }
    if escaped {
        let margin = f.get(o + 40);
        let mut enlarged = bounds;
        for lane in 0..3 {
            enlarged[lane] -= margin;
            enlarged[lane + 3] += margin;
        }
        for lane in 0..6 {
            fat.set(shape * 6 + lane, enlarged[lane]);
        }
        let key = shapes::col().get(o + shapes::S_PROXY_KEY);
        if key != u32::MAX {
            crate::broad::move_proxy(
                key,
                enlarged[0],
                enlarged[1],
                enlarged[2],
                enlarged[3],
                enlarged[4],
                enlarged[5],
            );
        }
    }
}
#[export_name = "shapeDestroyProxy"]
pub unsafe extern "C" fn destroy_proxy(world: usize, id: usize) {
    regions::select(world as u32);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE + shapes::S_PROXY_KEY;
    let key = u.get(o);
    if key != u32::MAX {
        crate::broad::destroy_proxy(key);
        u.set(o, u32::MAX);
    }
}
#[export_name = "shapeCreateProxy"]
pub unsafe extern "C" fn create_proxy(world: usize, id: usize, force: bool) {
    regions::select(world as u32);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    let body_id = u.get(o + 29) as usize;
    let tight = crate::continuous::bounds(id, bodies::geometry(body_id).0);
    create_proxy_bounds(
        world, id, force, tight[0], tight[1], tight[2], tight[3], tight[4], tight[5],
    );
}
#[export_name = "shapeCreateProxyBounds"]
pub unsafe extern "C" fn create_proxy_bounds(
    world: usize,
    id: usize,
    force: bool,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    regions::select(world as u32);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    let body_id = u.get(o + 29) as usize;
    let body_type = bodies::record(world, body_id).body_type as usize;
    let (bounds, _) = crate::finalize::refit_bounds([lx, ly, lz, hx, hy, hz], &[0.0; 6]);
    let f = shapes::col_f();
    let fat = crate::fataabb::col();
    let margin = if body_type == 0 { 0.02 } else { f.get(o + 40) };
    let mut enlarged = bounds;
    for lane in 0..3 {
        enlarged[lane] -= margin;
        enlarged[lane + 3] += margin;
    }
    for lane in 0..6 {
        f.set(o + 34 + lane, bounds[lane]);
        fat.set(id * 6 + lane, enlarged[lane]);
    }
    let key = crate::broad::create_proxy(
        body_type,
        enlarged[0],
        enlarged[1],
        enlarged[2],
        enlarged[3],
        enlarged[4],
        enlarged[5],
        u.get(o + 25),
        u.get(o + 26),
        id as u32,
        force as u32,
    );
    u.set(o + shapes::S_PROXY_KEY, key);
    query_pose(world, id, body_id);
}
// Borrowed geometry report for the synchronous authoring call, not world or shape storage.
static mut GEOMETRY_INPUT: [f32; 4] = [0.0; 4];
#[export_name = "shapeGeometryInputPtr"]
pub extern "C" fn geometry_input_ptr() -> *mut f32 {
    (&raw mut GEOMETRY_INPUT) as *mut f32
}
#[export_name = "shapeFinishGeometry"]
pub unsafe extern "C" fn finish_geometry(world: usize, id: usize) {
    regions::select(world as u32);
    let f = shapes::col_f();
    let o = id * shapes::SHAPE_STRIDE;
    f.set(o + 65, GEOMETRY_INPUT[0]);
    f.set(o + 66, GEOMETRY_INPUT[1]);
    f.set(o + 67, GEOMETRY_INPUT[2]);
    f.set(o + 40, GEOMETRY_INPUT[3]);
}
#[export_name = "shapeFilterWrite"]
pub unsafe extern "C" fn filter_write(
    world: usize,
    id: usize,
    category_hi: u32,
    category_lo: u32,
    mask_hi: u32,
    mask_lo: u32,
    group: i32,
) {
    regions::select(world as u32);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    u.set(o + 25, category_hi);
    u.set(o + 26, category_lo);
    u.set(o + 27, mask_hi);
    u.set(o + 28, mask_lo);
    u.set(o + 31, group as u32);
}
#[export_name = "shapeContactNext"]
pub unsafe extern "C" fn contact_next(world: usize, id: usize, mut key: i32) -> i32 {
    use crate::manifold_abi::*;
    regions::select(world as u32);
    if key == -2 {
        key = bodies::record(
            world,
            shapes::col().get(id * shapes::SHAPE_STRIDE + 29) as usize,
        )
        .head_contact_key;
    }
    let d = crate::manifolds::dir_col();
    while key != -1 {
        let contact = (key >> 1) as usize;
        let o = contact * DIR_STRIDE;
        if d.get(o + DIR_SHAPE_A) as usize == id || d.get(o + DIR_SHAPE_B) as usize == id {
            return key;
        }
        key = d.get(o + DIR_EDGE_A + 2 + 3 * (key & 1) as usize) as i32;
    }
    -1
}
#[export_name = "shapeAttachSensor"]
pub unsafe extern "C" fn attach_sensor(world: usize, id: usize, index: i32) {
    regions::select(world as u32);
    shapes::col().set(id * shapes::SHAPE_STRIDE + 41, index as u32);
}
#[export_name = "shapeBodyProxies"]
pub unsafe extern "C" fn body_proxies(world: usize, id: usize, mode: u32) {
    regions::select(world as u32);
    let mut shape = bodies::record(world, id).head_shape_id;
    while shape != -1 {
        if mode != 1 {
            destroy_proxy(world, shape as usize);
        }
        if mode != 0 {
            create_proxy(world, shape as usize, true);
        }
        shape = shapes::col().get(shape as usize * shapes::SHAPE_STRIDE + shapes::S_NEXT) as i32;
    }
}
