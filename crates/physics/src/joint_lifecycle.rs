//! joint.c identity lifecycle.
use crate::{
    bodies, constraint_graph as graph, island, joint_abi::*, joint_record as records, joints,
    regions, solver_set as sets,
};
pub unsafe fn reset(_world: usize) {}
#[export_name = "jointCollectEvents"]
pub unsafe extern "C" fn collect_events() -> usize {
    crate::events::clear_joints(regions::active());
    for id in 0..records::capacity() {
        let r = records::record(id);
        if r.set_index == 2
            && joints::read_float(r.color_index as usize, r.local_index as usize, J_EVENT) != 0.0
        {
            crate::events::joint(regions::active(), id, r.generation as u32);
        }
    }
    crate::events::count(regions::active(), 5)
}
unsafe fn begin() {}
unsafe fn wake_body(id: usize) {
    sets::wake(bodies::record(regions::active(), id).set_index as usize);
}
unsafe fn link(id: usize) {
    let r = *records::record(id);
    let a = r.edges[0].body_id as usize;
    let b = r.edges[1].body_id as usize;
    let world = regions::active();
    let sa = bodies::record(world, a).set_index;
    let sb = bodies::record(world, b).set_index;
    if sa == 2 && sb >= 3 {
        sets::wake(sb as usize);
    } else if sb == 2 && sa >= 3 {
        sets::wake(sa as usize);
    }
    island::link_joint(
        id as i32,
        a as i32,
        b as i32,
        bodies::record(world, a).island_id,
        bodies::record(world, b).island_id,
    );
}
#[export_name = "jointLink"]
pub unsafe extern "C" fn link_record(id: usize) {
    begin();
    link(id);
}
#[export_name = "jointUnlink"]
pub unsafe extern "C" fn unlink_record(id: usize) {
    let r = *records::record(id);
    if r.island_id != -1 {
        island::unlink_joint(id as i32, r.island_id, r.island_index as usize);
    }
}
#[export_name = "jointCreate"]
pub unsafe extern "C" fn create(
    a: usize,
    b: usize,
    joint_type: i32,
    draw_scale: f32,
    collide_connected: bool,
    ax: f32,
    ay: f32,
    az: f32,
    aqx: f32,
    aqy: f32,
    aqz: f32,
    aqs: f32,
    bx: f32,
    by: f32,
    bz: f32,
    bqx: f32,
    bqy: f32,
    bqz: f32,
    bqs: f32,
    force: f32,
    torque: f32,
    hertz: f32,
    damping: f32,
) -> u32 {
    begin();
    let id = records::alloc() as usize;
    let world = regions::active();
    records::record_mut(id).joint_type = joint_type;
    records::record_mut(id).draw_scale = draw_scale;
    records::record_mut(id).collide_connected = collide_connected;
    records::link_bodies(id, a, b);
    let sa = bodies::record(world, a).set_index;
    let sb = bodies::record(world, b).set_index;
    let max = sa.max(sb) as usize;
    let key = if sa == 1 || sb == 1 {
        graph::COLORS + 1
    } else if bodies::get_type(world, a) != 2 && bodies::get_type(world, b) != 2 {
        graph::COLORS
    } else if sa == 2 || sb == 2 {
        if max >= 3 {
            sets::wake(max);
        }
        graph::create_joint(a, b)
    } else {
        graph::COLORS + max
    };
    let index = if key < graph::COLORS {
        joints::count(key) - 1
    } else {
        joints::append(key)
    };
    records::set_location(id, key, index);
    joints::write_word(key, index, J_JOINT_ID, id as u32);
    joints::write_word(key, index, J_BODY_ID_A, a as u32);
    joints::write_word(key, index, J_BODY_ID_B, b as u32);
    if sa >= 3 && sb >= 3 && sa != sb {
        sets::merge(sa as usize, sb as usize);
    }
    let r = records::record(id);
    let key = if r.set_index == 2 {
        r.color_index as usize
    } else {
        graph::COLORS + r.set_index as usize
    };
    let index = r.local_index as usize;
    for (field, value) in [ax, ay, az, aqx, aqy, aqz, aqs].into_iter().enumerate() {
        joints::write_float(key, index, J_LOCAL_FRAME_A + field, value);
    }
    for (field, value) in [bx, by, bz, bqx, bqy, bqz, bqs].into_iter().enumerate() {
        joints::write_float(key, index, J_LOCAL_FRAME_B + field, value);
    }
    joints::write_word(key, index, J_TYPE, joint_type as u32);
    joints::write_float(key, index, J_CONSTRAINT_HERTZ, hertz);
    joints::write_float(key, index, J_CONSTRAINT_DAMPING, damping);
    joints::write_float(key, index, J_FORCE_THRESHOLD, force);
    joints::write_float(key, index, J_TORQUE_THRESHOLD, torque);
    if records::record(id).set_index > 1 {
        link(id);
    }
    id as u32
}
#[export_name = "jointTransfer"]
pub unsafe extern "C" fn transfer(id: usize, target: usize) {
    let r = *records::record(id);
    sets::transfer_joint(
        r.set_index as usize,
        r.color_index as usize,
        r.local_index as usize,
        target,
        r.edges[0].body_id as usize,
        r.edges[1].body_id as usize,
    );
}
#[export_name = "jointDestroy"]
pub unsafe extern "C" fn destroy(id: usize, wake_attached: bool) {
    begin();
    let r = *records::record(id);
    records::unlink_bodies(id);
    unlink_record(id);
    if r.set_index == 2 {
        graph::remove_joint(
            r.edges[0].body_id as usize,
            r.edges[1].body_id as usize,
            r.color_index as usize,
            r.local_index as usize,
        );
    } else {
        joints::remove(graph::COLORS + r.set_index as usize, r.local_index as usize);
    }
    records::free(id as u32);
    if wake_attached {
        wake_body(r.edges[0].body_id as usize);
        wake_body(r.edges[1].body_id as usize);
    }
}
