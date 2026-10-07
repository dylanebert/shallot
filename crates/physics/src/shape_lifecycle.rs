//! Nongeometry shape lifecycle in Box3D's body/shape list and proxy order.
use crate::{bodies, regions, shapes};

#[export_name = "shapeLink"]
pub unsafe extern "C" fn link(world: usize, id: usize, body_id: usize) {
    regions::select(world as u32);
    let body = bodies::record_mut(world, body_id);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    u.set(o + 1, body_id as u32);
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
}
#[export_name = "shapeUnlink"]
pub unsafe extern "C" fn unlink(world: usize, id: usize) {
    regions::select(world as u32);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    let body_id = u.get(o + 1) as usize;
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
#[export_name = "shapeSyncBodyBounds"]
pub unsafe extern "C" fn sync_body_bounds(world: usize, body_id: usize) {
    regions::select(world as u32);
    let mut shape = bodies::record(world, body_id).head_shape_id;
    while shape != -1 {
        body_record_bounds(world, body_id, shape as usize);
        shape = shapes::col().get(shape as usize * shapes::SHAPE_STRIDE + shapes::S_NEXT) as i32;
    }
}
unsafe fn body_record_bounds(world: usize, id: usize, shape: usize) {
    regions::select(world as u32);
    let pose = bodies::geometry(id).0;
    let tight = crate::continuous::bounds(shape, pose);
    let fat = shapes::col_f();
    let fat_offset = shape * shapes::SHAPE_STRIDE + shapes::S_FAT_AABB;
    let mut previous = [0.0; 6];
    for lane in 0..6 {
        previous[lane] = fat.get(fat_offset + lane);
    }
    let (bounds, escaped) = crate::finalize::refit_bounds(tight, &previous);
    let f = shapes::col_f();
    let o = shape * shapes::SHAPE_STRIDE;
    for lane in 0..6 {
        f.set(o + 10 + lane, bounds[lane]);
    }
    if escaped {
        let margin = f.get(o + 9);
        let mut enlarged = bounds;
        for lane in 0..3 {
            enlarged[lane] -= margin;
            enlarged[lane + 3] += margin;
        }
        for lane in 0..6 {
            fat.set(fat_offset + lane, enlarged[lane]);
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
    let body_id = u.get(o + 1) as usize;
    let tight = crate::continuous::bounds(id, bodies::geometry(body_id).0);
    create_proxy_bounds(
        world, id, force, tight[0], tight[1], tight[2], tight[3], tight[4], tight[5],
    );
}
#[export_name = "shapeCreateProxyTransform"]
pub unsafe extern "C" fn create_proxy_transform(
    world: usize,
    id: usize,
    force: bool,
    x: f32,
    y: f32,
    z: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qs: f32,
) {
    regions::select(world as u32);
    let pose = crate::math::Transform {
        p: crate::math::Vec3::new(x, y, z),
        q: crate::math::Quat {
            v: crate::math::Vec3::new(qx, qy, qz),
            s: qs,
        },
    };
    let tight = crate::shape_geometry::bounds(id, pose);
    create_proxy_bounds(
        world, id, force, tight[0], tight[1], tight[2], tight[3], tight[4], tight[5],
    );
}
unsafe fn create_proxy_bounds(
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
    let body_id = u.get(o + 1) as usize;
    let body_type = bodies::record(world, body_id).body_type as usize;
    let enlarged = write_bounds(id, body_type, [lx, ly, lz, hx, hy, hz]);
    let key = crate::broad::create_proxy(
        body_type,
        enlarged[0],
        enlarged[1],
        enlarged[2],
        enlarged[3],
        enlarged[4],
        enlarged[5],
        u.get(o + 39),
        u.get(o + 38),
        id as u32,
        force as u32,
    );
    u.set(o + shapes::S_PROXY_KEY, key);
}
unsafe fn write_bounds(id: usize, body_type: usize, tight: [f32; 6]) -> [f32; 6] {
    let o = id * shapes::SHAPE_STRIDE;
    let (bounds, _) = crate::finalize::refit_bounds(tight, &[0.0; 6]);
    let f = shapes::col_f();
    let fat = shapes::col_f();
    let margin = if body_type == 0 { 0.02 } else { f.get(o + 9) };
    let mut enlarged = bounds;
    for lane in 0..3 {
        enlarged[lane] -= margin;
        enlarged[lane + 3] += margin;
    }
    for lane in 0..6 {
        f.set(o + 10 + lane, bounds[lane]);
        fat.set(o + shapes::S_FAT_AABB + lane, enlarged[lane]);
    }
    enlarged
}
pub unsafe fn release_geometry(world: usize, id: usize) {
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    let kind = u.get(o + shapes::S_TYPE);
    let pointer = u.get(o + shapes::S_GEO_REFERENCE) as usize;
    if kind == 3 {
        crate::hull_database::remove(world, pointer);
    } else if matches!(kind, 1 | 2 | 4) {
        crate::geometry_database::remove(world, kind, pointer);
    }
}
pub unsafe fn destroy_internal(world: usize, id: usize, wake: bool) {
    let u = shapes::col();
    unlink(world, id);
    destroy_proxy(world, id);
    destroy_contacts(world, id, wake);
    if u.get(id * shapes::SHAPE_STRIDE + 4) != u32::MAX {
        crate::sensor::destroy(world, id);
    }
    release_geometry(world, id);
    shapes::shape_destroy(world as u32, id as u32);
}
#[export_name = "shapeDestroyWorld"]
pub unsafe extern "C" fn destroy(world: usize, id: usize, update_mass: bool) {
    regions::select(world as u32);
    let body = shapes::col().get(id * shapes::SHAPE_STRIDE + 1) as usize;
    destroy_internal(world, id, true);
    if update_mass {
        crate::body_record::runtime::update_mass(world, body);
    }
}
#[export_name = "shapeFinishCreate"]
pub unsafe extern "C" fn finish_create(
    world: usize,
    id: usize,
    force: bool,
    sensor: bool,
    update_mass: bool,
) {
    regions::select(world as u32);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE;
    let body = u.get(o + 1) as usize;
    if bodies::record(world, body).set_index != 1 {
        create_proxy(world, id, force && u.get(o + shapes::S_TYPE) != 1);
    }
    link(world, id, body);
    if sensor {
        crate::sensor::create(world, id);
    }
    if update_mass {
        crate::body_record::runtime::update_mass(world, body);
    }
}
unsafe fn destroy_contacts(world: usize, id: usize, wake: bool) {
    let mut key = contact_next(world, id, -2);
    while key != -1 {
        let contact = (key >> 1) as usize;
        let next = crate::manifolds::dir_col().get(
            contact * crate::manifold_abi::DIR_STRIDE
                + crate::manifold_abi::DIR_EDGE_A
                + 2
                + 3 * (key & 1) as usize,
        ) as i32;
        key = contact_next(world, id, next);
        crate::physics_world::destroy_contact(contact, wake);
    }
}
#[export_name = "shapeSetFlag"]
pub unsafe extern "C" fn set_flag(world: usize, id: usize, flag: u32, enabled: bool) {
    regions::select(world as u32);
    let u = shapes::col();
    let o = id * shapes::SHAPE_STRIDE + shapes::S_FLAGS;
    let bits = flag << 16;
    let old = u.get(o);
    u.set(o, if enabled { old | bits } else { old & !bits });
}
#[export_name = "shapeSetFilter64"]
pub unsafe extern "C" fn set_filter64(
    world: usize,
    id: usize,
    category: u64,
    mask: u64,
    group: i32,
) {
    set_filter(
        world,
        id,
        (category >> 32) as u32,
        category as u32,
        (mask >> 32) as u32,
        mask as u32,
        group,
    );
}
#[export_name = "shapeFilterWrite64"]
pub unsafe extern "C" fn filter_write64(
    world: usize,
    id: usize,
    category: u64,
    mask: u64,
    group: i32,
) {
    filter_write(
        world,
        id,
        (category >> 32) as u32,
        category as u32,
        (mask >> 32) as u32,
        mask as u32,
        group,
    );
}
#[export_name = "shapeSetFilter"]
pub unsafe extern "C" fn set_filter(
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
    if u.get(o + 39) == category_hi
        && u.get(o + 38) == category_lo
        && u.get(o + 41) == mask_hi
        && u.get(o + 40) == mask_lo
        && u.get(o + 42) as i32 == group
    {
        return;
    }
    filter_write(world, id, category_hi, category_lo, mask_hi, mask_lo, group);
    destroy_contacts(world, id, true);
    // shape.c compares category bits after assigning the filter, so invokeContacts recreates the proxy.
    let proxy = u.get(o + shapes::S_PROXY_KEY);
    let body = u.get(o + 1) as usize;
    if proxy != u32::MAX {
        destroy_proxy(world, id);
        create_proxy(world, id, true);
    } else {
        let tight = crate::continuous::bounds(id, bodies::geometry(body).0);
        write_bounds(id, bodies::record(world, body).body_type as usize, tight);
    }
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
    u.set(o + 39, category_hi);
    u.set(o + 38, category_lo);
    u.set(o + 41, mask_hi);
    u.set(o + 40, mask_lo);
    u.set(o + 42, group as u32);
}
#[export_name = "shapeContactNext"]
pub unsafe extern "C" fn contact_next(world: usize, id: usize, mut key: i32) -> i32 {
    use crate::manifold_abi::*;
    regions::select(world as u32);
    if key == -2 {
        key = bodies::record(
            world,
            shapes::col().get(id * shapes::SHAPE_STRIDE + 1) as usize,
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
    shapes::col().set(id * shapes::SHAPE_STRIDE + 4, index as u32);
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
