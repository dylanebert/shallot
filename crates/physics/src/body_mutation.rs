//! body.c cold world mutations.
use crate::{
    bodies, body_record::runtime as body, joint_lifecycle as joint, joint_record as joints,
    shape_lifecycle as shape,
};

#[export_name = "bodyWakeWorld"]
pub unsafe extern "C" fn wake_body(world: usize, id: usize) -> bool {
    crate::regions::select(world as u32);
    unsafe { wake_body_in_world(world, id) }
}

pub unsafe extern "C" fn wake_body_in_world(world: usize, id: usize) -> bool {
    let set = bodies::record(world, id).set_index;
    if set >= 3 {
        crate::solver_set::wake(world as usize, set as usize);
        return true;
    }
    false
}
unsafe fn contacts(world: usize, id: usize, wake: bool) {
    let mut key = bodies::record(world, id).head_contact_key;
    while key != -1 {
        let contact = (key >> 1) as usize;
        let d = crate::manifolds::dir_col(world as usize);
        key = d.get(
            contact * crate::manifold_abi::DIR_STRIDE
                + crate::manifold_abi::DIR_EDGE_A
                + 2
                + 3 * (key & 1) as usize,
        ) as i32;
        crate::physics_world::destroy_contact(world as usize, contact, wake);
    }
}
#[export_name = "bodyDestroyWorld"]
pub unsafe extern "C" fn destroy(world: usize, id: usize) {
    crate::regions::select(world as u32);
    unsafe { destroy_in_world(world, id) }
}

pub unsafe extern "C" fn destroy_in_world(world: usize, id: usize) {
    let mut key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(world as usize, id);
        key = r.edges[(key & 1) as usize].next_key;
        joint::destroy_in_world(world as usize, id, true);
    }
    contacts(world, id, true);
    let mut shape_id = bodies::record(world, id).head_shape_id;
    while shape_id != -1 {
        let s = shape_id as usize;
        let u = crate::shapes::col(world as usize);
        let o = s * crate::shapes::SHAPE_STRIDE;
        shape_id = u.get(o + crate::shapes::S_NEXT) as i32;
        if u.get(o + 4) != u32::MAX {
            crate::sensor::destroy_in_world(world, s);
        }
        shape::destroy_internal(world, s, true);
    }
    body::remove_island(world, id);
    bodies::body_destroy_in_world(world as usize, world as u32, id as u32);
}
#[export_name = "bodySetType"]
pub unsafe extern "C" fn set_type(world: usize, id: usize, kind: i32) {
    crate::regions::select(world as u32);
    unsafe { set_type_in_world(world, id, kind) }
}

pub unsafe extern "C" fn set_type_in_world(world: usize, id: usize, kind: i32) {
    let original = bodies::record(world, id).body_type;
    if original == kind || !shape::body_allows_type_in_world(world, id, kind as u32) {
        return;
    }
    if bodies::record(world, id).set_index == 1 {
        body::change_type(world, id, kind);
        body::update_mass(world, id);
        return;
    }
    contacts(world, id, false);
    wake_body_in_world(world, id);
    let mut key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(world as usize, id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index == 1 {
            continue;
        }
        wake_body_in_world(world, r.edges[0].body_id as usize);
        wake_body_in_world(world, r.edges[1].body_id as usize);
        joint::unlink_record_in_world(world as usize, id);
        joint::transfer_in_world(world as usize, id, 0);
    }
    body::change_type(world, id, kind);
    body::transfer(world, id, if kind == 0 { 0 } else { 2 }, true);
    if original == 0 {
        body::create_island(world, id);
    } else if kind == 0 {
        body::remove_island(world, id);
    }
    key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(world as usize, id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index != 1
            && (bodies::record(world, r.edges[0].body_id as usize).body_type == 2
                || bodies::record(world, r.edges[1].body_id as usize).body_type == 2)
        {
            joint::transfer_in_world(world as usize, id, 2);
        }
    }
    shape::body_proxies_in_world(world, id, 2);
    key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let joint_id = (key >> 1) as usize;
        let r = *joints::record(world as usize, joint_id);
        let other = r.edges[((key & 1) ^ 1) as usize].body_id as usize;
        key = r.edges[(key & 1) as usize].next_key;
        let b = bodies::record(world, other);
        if b.set_index != 1 && (kind == 2 || b.body_type == 2) {
            joint::link_record_in_world(world as usize, joint_id);
        }
    }
    body::sync_flags(world, id);
    body::update_mass(world, id);
}
#[export_name = "bodySetAwake"]
pub unsafe extern "C" fn set_awake(world: usize, id: usize, awake: bool) {
    crate::regions::select(world as u32);
    unsafe { set_awake_in_world(world, id, awake) }
}

pub unsafe extern "C" fn set_awake_in_world(world: usize, id: usize, awake: bool) {
    let r = *bodies::record(world, id);
    if awake {
        wake_body_in_world(world, id);
    } else if r.set_index == 2 && r.island_id != -1 {
        if crate::island::field_in_world(world as usize, r.island_id as usize, 3) > 0 {
            crate::island::split_in_world(world as usize, r.island_id as usize);
        }
        crate::physics_world::try_sleep_island_in_world(
            world as usize,
            bodies::record(world, id).island_id as usize,
        );
    }
}
#[export_name = "bodyDisable"]
pub unsafe extern "C" fn disable(world: usize, id: usize) {
    crate::regions::select(world as u32);
    unsafe { disable_in_world(world, id) }
}

pub unsafe extern "C" fn disable_in_world(world: usize, id: usize) {
    if bodies::record(world, id).set_index == 1 {
        return;
    }
    contacts(world, id, true);
    let mut key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(world as usize, id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index == 1 {
            continue;
        }
        joint::unlink_record_in_world(world as usize, id);
        joint::transfer_in_world(world as usize, id, 1);
    }
    shape::body_proxies_in_world(world, id, 0);
    body::remove_island(world, id);
    body::transfer(world, id, 1, true);
}
#[export_name = "bodyEnable"]
pub unsafe extern "C" fn enable(world: usize, id: usize) {
    crate::regions::select(world as u32);
    unsafe { enable_in_world(world, id) }
}

pub unsafe extern "C" fn enable_in_world(world: usize, id: usize) {
    if bodies::record(world, id).set_index != 1 {
        return;
    }
    let target = if bodies::record(world, id).body_type == 0 {
        0
    } else {
        2
    };
    body::transfer(world, id, target, true);
    shape::body_proxies_in_world(world, id, 1);
    if target != 0 {
        body::create_island(world, id);
    }
    let mut key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(world as usize, id);
        key = r.edges[(key & 1) as usize].next_key;
        let a = bodies::record(world, r.edges[0].body_id as usize).set_index;
        let b = bodies::record(world, r.edges[1].body_id as usize).set_index;
        if a == 1 || b == 1 {
            continue;
        }
        let target = if a == 0 { b } else { a };
        joint::transfer_in_world(world as usize, id, target as usize);
        if target != 0 {
            joint::link_record_in_world(world as usize, id);
        }
    }
}
