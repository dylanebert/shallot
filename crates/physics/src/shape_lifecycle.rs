//! Nongeometry shape lifecycle in Box3D's body/shape list and proxy order.
use crate::{bodies, shapes};

#[export_name = "shapeLink"]
pub unsafe extern "C" fn link(world: usize, id: usize, body_id: usize) {
    crate::regions::select(world as u32);
    unsafe { link_in_world(world, id, body_id) }
}

pub unsafe extern "C" fn link_in_world(world: usize, id: usize, body_id: usize) {
    let body = bodies::record_mut(world, body_id);
    let u = shapes::col(world);
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
    crate::regions::select(world as u32);
    unsafe { unlink_in_world(world, id) }
}

pub unsafe extern "C" fn unlink_in_world(world: usize, id: usize) {
    let u = shapes::col(world);
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
    crate::regions::select(world as u32);
    unsafe { body_take_in_world(world, body) }
}

pub unsafe extern "C" fn body_take_in_world(world: usize, body: usize) -> i32 {
    let shape = bodies::record(world, body).head_shape_id;
    if shape != -1 {
        unlink_in_world(world, shape as usize);
    }
    shape
}
#[export_name = "shapeBodyAllowsType"]
pub unsafe extern "C" fn body_allows_type(world: usize, body: usize, kind: u32) -> bool {
    crate::regions::select(world as u32);
    unsafe { body_allows_type_in_world(world, body, kind) }
}

pub unsafe extern "C" fn body_allows_type_in_world(world: usize, body: usize, kind: u32) -> bool {
    if kind == 0 {
        return true;
    }
    let mut shape = bodies::record(world, body).head_shape_id;
    while shape != -1 {
        let o = shape as usize * shapes::SHAPE_STRIDE;
        let u = shapes::col(world);
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
    crate::regions::select(world as u32);
    unsafe { sync_body_bounds_in_world(world, body_id) }
}

pub unsafe extern "C" fn sync_body_bounds_in_world(world: usize, body_id: usize) {
    let mut shape = bodies::record(world, body_id).head_shape_id;
    while shape != -1 {
        body_record_bounds(world, body_id, shape as usize);
        shape =
            shapes::col(world).get(shape as usize * shapes::SHAPE_STRIDE + shapes::S_NEXT) as i32;
    }
}
unsafe fn body_record_bounds(world: usize, id: usize, shape: usize) {
    let pose = bodies::geometry(world, id).0;
    let tight = crate::continuous::bounds(world, shape, pose);
    let fat = shapes::col_f(world);
    let fat_offset = shape * shapes::SHAPE_STRIDE + shapes::S_FAT_AABB;
    let mut previous = [0.0; 6];
    for lane in 0..6 {
        previous[lane] = fat.get(fat_offset + lane);
    }
    let (bounds, escaped) = crate::finalize::refit_bounds(tight, &previous);
    let f = shapes::col_f(world);
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
        let key = shapes::col(world).get(o + shapes::S_PROXY_KEY);
        if key != u32::MAX {
            crate::broad::move_proxy_in_world(
                world,
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
    crate::regions::select(world as u32);
    unsafe { destroy_proxy_in_world(world, id) }
}

pub unsafe extern "C" fn destroy_proxy_in_world(world: usize, id: usize) {
    let u = shapes::col(world);
    let o = id * shapes::SHAPE_STRIDE + shapes::S_PROXY_KEY;
    let key = u.get(o);
    if key != u32::MAX {
        crate::broad::destroy_proxy_in_world(world, key);
        u.set(o, u32::MAX);
    }
}
#[export_name = "shapeCreateProxy"]
pub unsafe extern "C" fn create_proxy(world: usize, id: usize, force: bool) {
    crate::regions::select(world as u32);
    unsafe { create_proxy_in_world(world, id, force) }
}

pub unsafe extern "C" fn create_proxy_in_world(world: usize, id: usize, force: bool) {
    let u = shapes::col(world);
    let o = id * shapes::SHAPE_STRIDE;
    let body_id = u.get(o + 1) as usize;
    let tight = crate::continuous::bounds(world, id, bodies::geometry(world, body_id).0);
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
    crate::regions::select(world as u32);
    unsafe { create_proxy_transform_in_world(world, id, force, x, y, z, qx, qy, qz, qs) }
}

pub unsafe extern "C" fn create_proxy_transform_in_world(
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
    let pose = crate::math::Transform {
        p: crate::math::Vec3::new(x, y, z),
        q: crate::math::Quat {
            v: crate::math::Vec3::new(qx, qy, qz),
            s: qs,
        },
    };
    let tight = crate::shape_geometry::bounds(world, id, pose);
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
    let u = shapes::col(world);
    let o = id * shapes::SHAPE_STRIDE;
    let body_id = u.get(o + 1) as usize;
    let body_type = bodies::record(world, body_id).body_type as usize;
    let enlarged = write_bounds(world, id, body_type, [lx, ly, lz, hx, hy, hz]);
    let key = crate::broad::create_proxy_in_world(
        world,
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
unsafe fn write_bounds(
    world_index: usize,
    id: usize,
    body_type: usize,
    tight: [f32; 6],
) -> [f32; 6] {
    let o = id * shapes::SHAPE_STRIDE;
    let (bounds, _) = crate::finalize::refit_bounds(tight, &[0.0; 6]);
    let f = shapes::col_f(world_index);
    let fat = shapes::col_f(world_index);
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
    let u = shapes::col(world);
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
    let u = shapes::col(world);
    unlink_in_world(world, id);
    destroy_proxy_in_world(world, id);
    destroy_contacts(world, id, wake);
    if u.get(id * shapes::SHAPE_STRIDE + 4) != u32::MAX {
        crate::sensor::destroy_in_world(world, id);
    }
    release_geometry(world, id);
    shapes::shape_destroy(world as u32, id as u32);
}
#[export_name = "shapeDestroyWorld"]
pub unsafe extern "C" fn destroy(world: usize, id: usize, update_mass: bool) {
    crate::regions::select(world as u32);
    unsafe { destroy_in_world(world, id, update_mass) }
}

pub unsafe extern "C" fn destroy_in_world(world: usize, id: usize, update_mass: bool) {
    let body = shapes::col(world).get(id * shapes::SHAPE_STRIDE + 1) as usize;
    destroy_internal(world, id, true);
    if update_mass {
        crate::body_record::runtime::update_mass_in_world(world, body);
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
    crate::regions::select(world as u32);
    unsafe { finish_create_in_world(world, id, force, sensor, update_mass) }
}

pub unsafe extern "C" fn finish_create_in_world(
    world: usize,
    id: usize,
    force: bool,
    sensor: bool,
    update_mass: bool,
) {
    let u = shapes::col(world);
    let o = id * shapes::SHAPE_STRIDE;
    let body = u.get(o + 1) as usize;
    if bodies::record(world, body).set_index != 1 {
        create_proxy_in_world(world, id, force && u.get(o + shapes::S_TYPE) != 1);
    }
    link_in_world(world, id, body);
    if sensor {
        crate::sensor::create_in_world(world, id);
    }
    if update_mass {
        crate::body_record::runtime::update_mass_in_world(world, body);
    }
}
unsafe fn destroy_contacts(world: usize, id: usize, wake: bool) {
    let mut key = contact_next_in_world(world, id, -2);
    while key != -1 {
        let contact = (key >> 1) as usize;
        let next = crate::manifolds::dir_col(world).get(
            contact * crate::manifold_abi::DIR_STRIDE
                + crate::manifold_abi::DIR_EDGE_A
                + 2
                + 3 * (key & 1) as usize,
        ) as i32;
        key = contact_next_in_world(world, id, next);
        crate::contact_lifecycle::destroy(world, contact, wake);
    }
}
#[export_name = "shapeSetFlag"]
pub unsafe extern "C" fn set_flag(world: usize, id: usize, flag: u32, enabled: bool) {
    crate::regions::select(world as u32);
    unsafe { set_flag_in_world(world, id, flag, enabled) }
}

pub unsafe extern "C" fn set_flag_in_world(world: usize, id: usize, flag: u32, enabled: bool) {
    let u = shapes::col(world);
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
    unsafe { set_filter64_in_world(world, id, category, mask, group) }
}

pub unsafe extern "C" fn set_filter64_in_world(
    world: usize,
    id: usize,
    category: u64,
    mask: u64,
    group: i32,
) {
    set_filter_in_world(
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
    unsafe { filter_write64_in_world(world, id, category, mask, group) }
}

pub unsafe extern "C" fn filter_write64_in_world(
    world: usize,
    id: usize,
    category: u64,
    mask: u64,
    group: i32,
) {
    filter_write_in_world(
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
    crate::regions::select(world as u32);
    unsafe { set_filter_in_world(world, id, category_hi, category_lo, mask_hi, mask_lo, group) }
}

pub unsafe extern "C" fn set_filter_in_world(
    world: usize,
    id: usize,
    category_hi: u32,
    category_lo: u32,
    mask_hi: u32,
    mask_lo: u32,
    group: i32,
) {
    let u = shapes::col(world);
    let o = id * shapes::SHAPE_STRIDE;
    if u.get(o + 39) == category_hi
        && u.get(o + 38) == category_lo
        && u.get(o + 41) == mask_hi
        && u.get(o + 40) == mask_lo
        && u.get(o + 42) as i32 == group
    {
        return;
    }
    filter_write_in_world(world, id, category_hi, category_lo, mask_hi, mask_lo, group);
    destroy_contacts(world, id, true);
    // shape.c compares category bits after assigning the filter, so invokeContacts recreates the proxy.
    let proxy = u.get(o + shapes::S_PROXY_KEY);
    let body = u.get(o + 1) as usize;
    if proxy != u32::MAX {
        destroy_proxy_in_world(world, id);
        create_proxy_in_world(world, id, true);
    } else {
        let tight = crate::continuous::bounds(world, id, bodies::geometry(world, body).0);
        write_bounds(
            world,
            id,
            bodies::record(world, body).body_type as usize,
            tight,
        );
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
    crate::regions::select(world as u32);
    unsafe { filter_write_in_world(world, id, category_hi, category_lo, mask_hi, mask_lo, group) }
}

pub unsafe extern "C" fn filter_write_in_world(
    world: usize,
    id: usize,
    category_hi: u32,
    category_lo: u32,
    mask_hi: u32,
    mask_lo: u32,
    group: i32,
) {
    let u = shapes::col(world);
    let o = id * shapes::SHAPE_STRIDE;
    u.set(o + 39, category_hi);
    u.set(o + 38, category_lo);
    u.set(o + 41, mask_hi);
    u.set(o + 40, mask_lo);
    u.set(o + 42, group as u32);
}
#[export_name = "shapeSetDensity"]
pub unsafe extern "C" fn set_density(world: usize, id: usize, value: f32, update_mass: bool) {
    crate::regions::select(world as u32);
    let offset = id * shapes::SHAPE_STRIDE + shapes::S_DENSITY;
    let values = shapes::col_f(world);
    if values.get(offset) == value {
        return;
    }
    values.set(offset, value);
    if update_mass {
        let body =
            shapes::col(world).get(id * shapes::SHAPE_STRIDE + shapes::S_QUERY_BODY) as usize;
        crate::body_record::runtime::update_mass_in_world(world, body);
    }
}
unsafe fn reset_after_geometry_change(world: usize, id: usize) {
    destroy_contacts(world, id, true);
    let u = shapes::col(world);
    let o = id * shapes::SHAPE_STRIDE;
    if u.get(o + shapes::S_PROXY_KEY) != u32::MAX {
        unsafe {
            destroy_proxy_in_world(world, id);
            create_proxy_in_world(world, id, true);
        }
    } else {
        let body = u.get(o + 1) as usize;
        let tight = crate::continuous::bounds(world, id, bodies::geometry(world, body).0);
        write_bounds(
            world,
            id,
            bodies::record(world, body).body_type as usize,
            tight,
        );
    }
}

#[export_name = "shapeSetSphere"]
pub unsafe extern "C" fn set_sphere(world: usize, id: usize, x: f32, y: f32, z: f32, radius: f32) {
    crate::regions::select(world as u32);
    unsafe {
        release_geometry(world, id);
        let u = shapes::col(world);
        let o = id * shapes::SHAPE_STRIDE;
        u.set(o + shapes::S_TYPE, 5);
        u.set(o + shapes::S_GEO_REFERENCE, 0);
        let f = shapes::col_f(world);
        for (lane, value) in [x, y, z, radius].into_iter().enumerate() {
            f.set(o + 48 + lane, value);
        }
        crate::shape_geometry::finish_geometry_in_world(world, id);
        reset_after_geometry_change(world, id);
    }
}

#[export_name = "shapeSetCapsule"]
pub unsafe extern "C" fn set_capsule(
    world: usize,
    id: usize,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    radius: f32,
) {
    crate::regions::select(world as u32);
    unsafe {
        release_geometry(world, id);
        let u = shapes::col(world);
        let o = id * shapes::SHAPE_STRIDE;
        u.set(o + shapes::S_TYPE, 0);
        u.set(o + shapes::S_GEO_REFERENCE, 0);
        let f = shapes::col_f(world);
        for (lane, value) in [ax, ay, az, bx, by, bz, radius].into_iter().enumerate() {
            f.set(o + 48 + lane, value);
        }
        crate::shape_geometry::finish_geometry_in_world(world, id);
        reset_after_geometry_change(world, id);
    }
}

#[export_name = "shapeSetHull"]
pub unsafe extern "C" fn set_hull(world: usize, id: usize, handle: usize) {
    crate::regions::select(world as u32);
    unsafe {
        release_geometry(world, id);
        let u = shapes::col(world);
        let o = id * shapes::SHAPE_STRIDE;
        u.set(o + shapes::S_TYPE, 3);
        u.set(o + shapes::S_GEO_REFERENCE, handle as u32);
        crate::shape_geometry::finish_geometry_in_world(world, id);
        reset_after_geometry_change(world, id);
    }
}

#[export_name = "shapeSetMesh"]
pub unsafe extern "C" fn set_mesh(world: usize, id: usize, handle: usize, x: f32, y: f32, z: f32) {
    crate::regions::select(world as u32);
    unsafe {
        release_geometry(world, id);
        let u = shapes::col(world);
        let o = id * shapes::SHAPE_STRIDE;
        u.set(o + shapes::S_TYPE, 4);
        u.set(o + shapes::S_GEO_REFERENCE, handle as u32);
        let f = shapes::col_f(world);
        for (lane, value) in [x, y, z].into_iter().enumerate() {
            let sign = if value >= 0.0 { 1.0 } else { -1.0 };
            f.set(o + 49 + lane, sign * crate::math::maxf(value.abs(), 0.01));
        }
        crate::shape_geometry::finish_geometry_in_world(world, id);
        reset_after_geometry_change(world, id);
    }
}

#[export_name = "shapeGetGeometryReference"]
pub unsafe extern "C" fn get_geometry_reference(world: usize, id: usize) -> u32 {
    crate::regions::select(world as u32);
    shapes::col(world).get(id * shapes::SHAPE_STRIDE + shapes::S_GEO_REFERENCE)
}

#[export_name = "shapeSetFriction"]
pub unsafe extern "C" fn set_friction(world: usize, id: usize, value: f32) {
    crate::regions::select(world as u32);
    let material = unsafe { shapes::shape_material_ptr(world as u32, id as u32) } as *mut f32;
    unsafe {
        *material = value;
    }
}
#[export_name = "shapeSetRestitution"]
pub unsafe extern "C" fn set_restitution(world: usize, id: usize, value: f32) {
    crate::regions::select(world as u32);
    let material = unsafe { shapes::shape_material_ptr(world as u32, id as u32) } as *mut f32;
    unsafe {
        *material.add(1) = value;
    }
}
#[export_name = "shapeSetSurfaceMaterial"]
pub unsafe extern "C" fn set_surface_material(
    world: usize,
    id: usize,
    friction: f32,
    restitution: f32,
    rolling: f32,
    x: f32,
    y: f32,
    z: f32,
    low: u32,
    high: u32,
    color: u32,
) {
    crate::regions::select(world as u32);
    unsafe {
        shapes::material_set(
            world,
            id,
            0,
            friction,
            restitution,
            rolling,
            x,
            y,
            z,
            low,
            high,
            color,
        );
    }
}
#[export_name = "shapeSetMeshMaterial"]
pub unsafe extern "C" fn set_mesh_material(
    world: usize,
    id: usize,
    index: usize,
    friction: f32,
    restitution: f32,
    rolling: f32,
    x: f32,
    y: f32,
    z: f32,
    low: u32,
    high: u32,
    color: u32,
) {
    crate::regions::select(world as u32);
    unsafe {
        shapes::material_set(
            world,
            id,
            index,
            friction,
            restitution,
            rolling,
            x,
            y,
            z,
            low,
            high,
            color,
        );
    }
}
#[export_name = "shapeContactNext"]
pub unsafe extern "C" fn contact_next(world: usize, id: usize, key: i32) -> i32 {
    crate::regions::select(world as u32);
    unsafe { contact_next_in_world(world, id, key) }
}

pub unsafe extern "C" fn contact_next_in_world(world: usize, id: usize, mut key: i32) -> i32 {
    use crate::manifold_abi::*;

    if key == -2 {
        key = bodies::record(
            world,
            shapes::col(world).get(id * shapes::SHAPE_STRIDE + 1) as usize,
        )
        .head_contact_key;
    }
    let d = crate::manifolds::dir_col(world);
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
    crate::regions::select(world as u32);
    unsafe { attach_sensor_in_world(world, id, index) }
}

pub unsafe extern "C" fn attach_sensor_in_world(world: usize, id: usize, index: i32) {
    shapes::col(world).set(id * shapes::SHAPE_STRIDE + 4, index as u32);
}
#[export_name = "shapeBodyProxies"]
pub unsafe extern "C" fn body_proxies(world: usize, id: usize, mode: u32) {
    crate::regions::select(world as u32);
    unsafe { body_proxies_in_world(world, id, mode) }
}

pub unsafe extern "C" fn body_proxies_in_world(world: usize, id: usize, mode: u32) {
    let mut shape = bodies::record(world, id).head_shape_id;
    while shape != -1 {
        if mode != 1 {
            destroy_proxy_in_world(world, shape as usize);
        }
        if mode != 0 {
            create_proxy_in_world(world, shape as usize, true);
        }
        shape =
            shapes::col(world).get(shape as usize * shapes::SHAPE_STRIDE + shapes::S_NEXT) as i32;
    }
}
