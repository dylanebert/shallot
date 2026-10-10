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
        crate::solver_set::wake(world, set as usize);
        return true;
    }
    false
}
unsafe fn contacts(world: usize, id: usize, wake: bool) {
    let mut key = bodies::record(world, id).head_contact_key;
    while key != -1 {
        let contact = (key >> 1) as usize;
        let d = crate::manifolds::dir_col(world);
        key = d.get(
            contact * crate::manifold_abi::DIR_STRIDE
                + crate::manifold_abi::DIR_EDGE_A
                + 2
                + 3 * (key & 1) as usize,
        ) as i32;
        crate::contact_lifecycle::destroy(world, contact, wake);
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
        let r = *joints::record(world, id);
        key = r.edges[(key & 1) as usize].next_key;
        joint::destroy_in_world(world, id, true);
    }
    contacts(world, id, true);
    let mut shape_id = bodies::record(world, id).head_shape_id;
    while shape_id != -1 {
        let s = shape_id as usize;
        let u = crate::shapes::col(world);
        let o = s * crate::shapes::SHAPE_STRIDE;
        shape_id = u.get(o + crate::shapes::S_NEXT) as i32;
        if u.get(o + 4) != u32::MAX {
            crate::sensor::destroy_in_world(world, s);
        }
        shape::destroy_internal(world, s, true);
    }
    body::remove_island(world, id);
    bodies::body_destroy_in_world(world, world as u32, id as u32);
}
#[export_name = "bodyGetProperty"]
pub unsafe extern "C" fn get_property(world: usize, id: usize, property: u32) -> f32 {
    crate::regions::select(world as u32);
    unsafe { get_property_in_world(world, id, property) }
}

pub unsafe fn get_property_in_world(world: usize, id: usize, property: u32) -> f32 {
    let record = bodies::record(world, id);
    match property {
        0..=2 => {
            let sim = bodies::column(world, id, 1, crate::body::SIM_STRIDE);
            sim.get(crate::body::LINEAR_DAMPING + property as usize)
        }
        3 => record.sleep_threshold,
        4 => (u8::from(record.flags & crate::body::flags::ENABLE_SLEEP != 0)) as f32,
        5 => (record.flags & crate::body::flags::ALL_MOTION_LOCKS) as f32,
        6 => (u8::from(record.flags & crate::continuous::IS_BULLET != 0)) as f32,
        7 => (u8::from(record.flags & crate::body::flags::ALLOW_FAST_ROTATION != 0)) as f32,
        8 => (u8::from(record.flags & crate::body::flags::ENABLE_CONTACT_RECYCLING != 0)) as f32,
        9 => (u8::from(record.set_index != 1)) as f32,
        10 => (u8::from(record.set_index == 2)) as f32,
        _ => 0.0,
    }
}

#[export_name = "bodySetProperty"]
pub unsafe extern "C" fn set_property(world: usize, id: usize, property: u32, value: f32) {
    crate::regions::select(world as u32);
    unsafe { set_property_in_world(world, id, property, value) }
}

pub unsafe fn set_property_in_world(world: usize, id: usize, property: u32, value: f32) {
    use crate::body::{self, SIM_STRIDE};
    match property {
        0..=2 => {
            let sim = bodies::column(world, id, 1, SIM_STRIDE);
            sim.set(body::LINEAR_DAMPING + property as usize, value);
        }
        3 => bodies::record_mut(world, id).sleep_threshold = value,
        4 => {
            let record = bodies::record_mut(world, id);
            let enabled = value != 0.0;
            let was_enabled = record.flags & body::flags::ENABLE_SLEEP != 0;
            if enabled == was_enabled {
                return;
            }
            if enabled {
                record.flags |= body::flags::ENABLE_SLEEP;
            } else {
                record.flags &= !body::flags::ENABLE_SLEEP;
            }
            crate::body_record::runtime::sync_flags_in_world(world, id);
            if !enabled {
                wake_body_in_world(world, id);
            }
        }
        5 => {
            let locks = (value as u32) & body::flags::ALL_MOTION_LOCKS;
            let record = bodies::record_mut(world, id);
            let old = record.flags & body::flags::ALL_MOTION_LOCKS;
            if old == locks {
                return;
            }
            let fixed_rotation_before = old & 0x38 == 0x38;
            let fixed_rotation_after = locks & 0x38 == 0x38;
            record.flags = (record.flags & !0x3f) | locks;
            crate::body_record::runtime::sync_flags_in_world(world, id);
            let set_index = record.set_index;
            if set_index == 2 {
                let state = bodies::column(world, id, 0, body::STATE_STRIDE);
                for (bit, lane) in [(1, 0), (2, 1), (4, 2), (8, 3), (16, 4), (32, 5)] {
                    if locks & bit != 0 {
                        state.set(lane, 0.0);
                    }
                }
            }
            if fixed_rotation_before != fixed_rotation_after {
                crate::body_record::runtime::update_mass_in_world(world, id);
            }
        }
        6 => {
            let flag = crate::continuous::IS_BULLET;
            let record = bodies::record_mut(world, id);
            record.flags = if value != 0.0 {
                record.flags | flag
            } else {
                record.flags & !flag
            };
            crate::body_record::runtime::sync_flags_in_world(world, id);
        }
        7 => {
            let flag = body::flags::ALLOW_FAST_ROTATION;
            let record = bodies::record_mut(world, id);
            record.flags = if value != 0.0 {
                record.flags | flag
            } else {
                record.flags & !flag
            };
            crate::body_record::runtime::sync_flags_in_world(world, id);
        }
        8 => {
            let flag = body::flags::ENABLE_CONTACT_RECYCLING;
            let record = bodies::record_mut(world, id);
            record.flags = if value != 0.0 {
                record.flags | flag
            } else {
                record.flags & !flag
            };
            crate::body_record::runtime::sync_flags_in_world(world, id);
        }
        9 => {
            if value != 0.0 {
                enable_in_world(world, id);
            } else {
                disable_in_world(world, id);
            }
        }
        10 => set_awake_in_world(world, id, value != 0.0),
        _ => {}
    }
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
        let r = *joints::record(world, id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index == 1 {
            continue;
        }
        wake_body_in_world(world, r.edges[0].body_id as usize);
        wake_body_in_world(world, r.edges[1].body_id as usize);
        joint::unlink_record_in_world(world, id);
        joint::transfer_in_world(world, id, 0);
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
        let r = *joints::record(world, id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index != 1
            && (bodies::record(world, r.edges[0].body_id as usize).body_type == 2
                || bodies::record(world, r.edges[1].body_id as usize).body_type == 2)
        {
            joint::transfer_in_world(world, id, 2);
        }
    }
    shape::body_proxies_in_world(world, id, 2);
    key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let joint_id = (key >> 1) as usize;
        let r = *joints::record(world, joint_id);
        let other = r.edges[((key & 1) ^ 1) as usize].body_id as usize;
        key = r.edges[(key & 1) as usize].next_key;
        let b = bodies::record(world, other);
        if b.set_index != 1 && (kind == 2 || b.body_type == 2) {
            joint::link_record_in_world(world, joint_id);
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
        if crate::island::field_in_world(world, r.island_id as usize, 3) > 0 {
            crate::island::split_in_world(world, r.island_id as usize);
        }
        crate::physics_world::try_sleep_island_in_world(
            world,
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
        let r = *joints::record(world, id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index == 1 {
            continue;
        }
        joint::unlink_record_in_world(world, id);
        joint::transfer_in_world(world, id, 1);
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
        let r = *joints::record(world, id);
        key = r.edges[(key & 1) as usize].next_key;
        let a = bodies::record(world, r.edges[0].body_id as usize).set_index;
        let b = bodies::record(world, r.edges[1].body_id as usize).set_index;
        if a == 1 || b == 1 {
            continue;
        }
        let target = if a == 0 { b } else { a };
        joint::transfer_in_world(world, id, target as usize);
        if target != 0 {
            joint::link_record_in_world(world, id);
        }
    }
}
