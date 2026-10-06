//! body.c cold world mutations.
use crate::{
    bodies, body_record::runtime as body, joint_lifecycle as joint, joint_record as joints,
    regions, shape_lifecycle as shape,
};

#[export_name = "bodyWakeWorld"]
pub unsafe extern "C" fn wake_body(world: usize, id: usize) -> bool {
    regions::select(world as u32);
    let set = bodies::record(world, id).set_index;
    if set >= 3 {
        crate::solver_set::wake(set as usize);
        return true;
    }
    false
}
unsafe fn contacts(world: usize, id: usize, wake: bool) {
    let mut key = bodies::record(world, id).head_contact_key;
    while key != -1 {
        let contact = (key >> 1) as usize;
        let d = crate::manifolds::dir_col();
        key = d.get(
            contact * crate::manifold_abi::DIR_STRIDE
                + crate::manifold_abi::DIR_EDGE_A
                + 2
                + 3 * (key & 1) as usize,
        ) as i32;
        crate::physics_world::destroy_contact(contact, wake);
    }
}
#[export_name = "bodyDestroyWorld"]
pub unsafe extern "C" fn destroy(world: usize, id: usize) {
    regions::select(world as u32);
    let mut key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(id);
        key = r.edges[(key & 1) as usize].next_key;
        joint::destroy(id, true);
    }
    contacts(world, id, true);
    let mut shape_id = bodies::record(world, id).head_shape_id;
    while shape_id != -1 {
        let s = shape_id as usize;
        let u = crate::shapes::col();
        let o = s * crate::shapes::SHAPE_STRIDE;
        shape_id = u.get(o + crate::shapes::S_NEXT) as i32;
        if u.get(o + 41) != u32::MAX {
            crate::sensor::destroy(world, s);
        }
        shape::destroy_proxy(world, s);
        let kind = u.get(o + crate::shapes::S_TYPE);
        let pointer = u.get(o + crate::shapes::S_GEO_REFERENCE) as usize;
        if kind == 3 {
            crate::hull_database::remove(world, pointer);
        } else if matches!(kind, 1 | 2 | 4) {
            crate::geometry_database::remove(world, kind, pointer);
        }
        crate::shapes::shape_destroy(world as u32, s as u32);
    }
    body::remove_island(world, id);
    let moved = bodies::body_destroy(world as u32, id as u32);
    if moved != u32::MAX {
        shape::sync_body(world, moved as usize);
    }
}
#[export_name = "bodySetType"]
pub unsafe extern "C" fn set_type(world: usize, id: usize, kind: i32) {
    regions::select(world as u32);
    let original = bodies::record(world, id).body_type;
    if original == kind || !shape::body_allows_type(world, id, kind as u32) {
        return;
    }
    if bodies::record(world, id).set_index == 1 {
        body::change_type(world, id, kind);
        body::update_mass(world, id);
        return;
    }
    contacts(world, id, false);
    wake_body(world, id);
    let mut key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index == 1 {
            continue;
        }
        wake_body(world, r.edges[0].body_id as usize);
        wake_body(world, r.edges[1].body_id as usize);
        joint::unlink_record(id);
        joint::transfer(id, 0);
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
        let r = *joints::record(id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index != 1
            && (bodies::record(world, r.edges[0].body_id as usize).body_type == 2
                || bodies::record(world, r.edges[1].body_id as usize).body_type == 2)
        {
            joint::transfer(id, 2);
        }
    }
    shape::body_proxies(world, id, 2);
    key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let joint_id = (key >> 1) as usize;
        let r = *joints::record(joint_id);
        let other = r.edges[((key & 1) ^ 1) as usize].body_id as usize;
        key = r.edges[(key & 1) as usize].next_key;
        let b = bodies::record(world, other);
        if b.set_index != 1 && (kind == 2 || b.body_type == 2) {
            joint::link_record(joint_id);
        }
    }
    body::sync_flags(world, id);
    body::update_mass(world, id);
}
#[export_name = "bodySetAwake"]
pub unsafe extern "C" fn set_awake(world: usize, id: usize, awake: bool) {
    regions::select(world as u32);
    let r = *bodies::record(world, id);
    if awake {
        wake_body(world, id);
    } else if r.set_index == 2 && r.island_id != -1 {
        if crate::island::field(r.island_id as usize, 3) > 0 {
            crate::island::split(r.island_id as usize);
        }
        crate::physics_world::try_sleep_island(bodies::record(world, id).island_id as usize);
    }
}
#[export_name = "bodyDisable"]
pub unsafe extern "C" fn disable(world: usize, id: usize) {
    regions::select(world as u32);
    if bodies::record(world, id).set_index == 1 {
        return;
    }
    contacts(world, id, true);
    let mut key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(id);
        key = r.edges[(key & 1) as usize].next_key;
        if r.set_index == 1 {
            continue;
        }
        joint::unlink_record(id);
        joint::transfer(id, 1);
    }
    shape::body_proxies(world, id, 0);
    body::remove_island(world, id);
    body::transfer(world, id, 1, true);
}
#[export_name = "bodyEnable"]
pub unsafe extern "C" fn enable(world: usize, id: usize) {
    regions::select(world as u32);
    if bodies::record(world, id).set_index != 1 {
        return;
    }
    let target = if bodies::record(world, id).body_type == 0 {
        0
    } else {
        2
    };
    body::transfer(world, id, target, true);
    shape::body_proxies(world, id, 1);
    if target != 0 {
        body::create_island(world, id);
    }
    let mut key = bodies::record(world, id).head_joint_key;
    while key != -1 {
        let id = (key >> 1) as usize;
        let r = *joints::record(id);
        key = r.edges[(key & 1) as usize].next_key;
        let a = bodies::record(world, r.edges[0].body_id as usize).set_index;
        let b = bodies::record(world, r.edges[1].body_id as usize).set_index;
        if a == 1 || b == 1 {
            continue;
        }
        let target = if a == 0 { b } else { a };
        joint::transfer(id, target as usize);
        if target != 0 {
            joint::link_record(id);
        }
    }
}
