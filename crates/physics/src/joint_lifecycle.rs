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
#[export_name = "jointSetCollideConnected"]
pub unsafe extern "C" fn set_collide_connected(world: usize, id: usize, collide: bool) {
    regions::select(world as u32);
    let r = *records::record(id);
    if r.collide_connected == collide {
        return;
    }
    records::record_mut(id).collide_connected = collide;
    let a = r.edges[0].body_id as usize;
    let b = r.edges[1].body_id as usize;
    if collide {
        let a = bodies::record(world, a);
        let b = bodies::record(world, b);
        let mut shape = if a.shape_count < b.shape_count {
            a.head_shape_id
        } else {
            b.head_shape_id
        };
        while shape != -1 {
            let o = shape as usize * crate::shapes::SHAPE_STRIDE;
            let u = crate::shapes::col();
            let key = u.get(o + crate::shapes::S_PROXY_KEY);
            if key != u32::MAX {
                crate::broad::buffer_move(key);
            }
            shape = u.get(o + crate::shapes::S_NEXT) as i32;
        }
    } else {
        let a = bodies::record(world, a);
        let b = bodies::record(world, b);
        let (mut key, other) = if a.contact_count < b.contact_count {
            (a.head_contact_key, b.id)
        } else {
            (b.head_contact_key, a.id)
        };
        while key != -1 {
            let id = (key >> 1) as usize;
            let edge = (key & 1) as usize;
            let d = crate::manifolds::dir_col();
            let o = id * crate::manifold_abi::DIR_STRIDE + crate::manifold_abi::DIR_EDGE_A;
            key = d.get(o + 2 + 3 * edge) as i32;
            if d.get(o + 3 * (edge ^ 1)) as i32 == other {
                crate::physics_world::destroy_contact(id, false);
            }
        }
    }
}
#[export_name = "jointWakeBodies"]
pub unsafe extern "C" fn wake_bodies(world: usize, id: usize) {
    regions::select(world as u32);
    let r = *records::record(id);
    wake_body(r.edges[0].body_id as usize);
    wake_body(r.edges[1].body_id as usize);
}
unsafe fn location(id: usize) -> (usize, usize) {
    let r = records::record(id);
    (
        if r.set_index == 2 {
            r.color_index as usize
        } else {
            graph::COLORS + r.set_index as usize
        },
        r.local_index as usize,
    )
}
#[export_name = "jointWriteVec3"]
pub unsafe extern "C" fn write_vec3(id: usize, field: usize, x: f32, y: f32, z: f32) {
    let (key, index) = location(id);
    joints::write_float(key, index, field, x);
    joints::write_float(key, index, field + 1, y);
    joints::write_float(key, index, field + 2, z);
}
#[export_name = "jointWriteQuat"]
pub unsafe extern "C" fn write_quat(id: usize, field: usize, x: f32, y: f32, z: f32, s: f32) {
    write_vec3(id, field, x, y, z);
    let (key, index) = location(id);
    joints::write_float(key, index, field + 3, s);
}
#[export_name = "jointEnable"]
pub unsafe extern "C" fn enable(world: usize, id: usize, bit: u32, enabled: bool) {
    regions::select(world as u32);
    let (key, index) = location(id);
    let (field, impulse, count) = match (records::record(id).joint_type as u32, bit) {
        (TY_DISTANCE, DJ_ENABLE_SPRING | DJ_ENABLE_LIMIT) => (DJ_ENABLE, 0, 0),
        (TY_DISTANCE, DJ_ENABLE_MOTOR) => (DJ_ENABLE, DJ_MOTOR_IMPULSE, 1),
        (TY_REVOLUTE, RJ_ENABLE_SPRING) => (RJ_ENABLE, RJ_SPRING_IMPULSE, 1),
        (TY_REVOLUTE, RJ_ENABLE_MOTOR) => (RJ_ENABLE, RJ_MOTOR_IMPULSE, 1),
        (TY_REVOLUTE, RJ_ENABLE_LIMIT) => (RJ_ENABLE, RJ_LOWER_IMPULSE, 2),
        (TY_PRISMATIC, PJ_ENABLE_SPRING) => (PJ_ENABLE, PJ_SPRING_IMPULSE, 1),
        (TY_PRISMATIC, PJ_ENABLE_MOTOR) => (PJ_ENABLE, PJ_MOTOR_IMPULSE, 1),
        (TY_PRISMATIC, PJ_ENABLE_LIMIT) => (PJ_ENABLE, PJ_LOWER_IMPULSE, 2),
        (TY_SPHERICAL, SJ_ENABLE_SPRING) => (SJ_ENABLE, SJ_SPRING_IMPULSE, 3),
        (TY_SPHERICAL, SJ_ENABLE_MOTOR) => (SJ_ENABLE, SJ_MOTOR_IMPULSE, 3),
        (TY_SPHERICAL, SJ_ENABLE_CONE_LIMIT) => (SJ_ENABLE, SJ_SWING_IMPULSE, 1),
        (TY_SPHERICAL, SJ_ENABLE_TWIST_LIMIT) => (SJ_ENABLE, SJ_LOWER_TWIST_IMPULSE, 2),
        (TY_WHEEL, WHJ_ENABLE_SUSPENSION_SPRING) => (WHJ_ENABLE, WHJ_SUSPENSION_SPRING_IMPULSE, 1),
        (TY_WHEEL, WHJ_ENABLE_SUSPENSION_LIMIT) => (WHJ_ENABLE, WHJ_LOWER_SUSPENSION_IMPULSE, 2),
        (TY_WHEEL, WHJ_ENABLE_SPIN_MOTOR) => (WHJ_ENABLE, WHJ_SPIN_IMPULSE, 1),
        (TY_WHEEL, WHJ_ENABLE_STEERING) => (WHJ_ENABLE, WHJ_ANGULAR_IMPULSE, 2),
        (TY_WHEEL, WHJ_ENABLE_STEERING_LIMIT) => (WHJ_ENABLE, WHJ_LOWER_STEERING_IMPULSE, 2),
        _ => unreachable!(),
    };
    let bits = joints::read_word(key, index, field);
    if (bits & bit != 0) != enabled {
        for lane in 0..count {
            joints::write_float(key, index, impulse + lane, 0.0);
        }
    }
    joints::write_word(
        key,
        index,
        field,
        if enabled { bits | bit } else { bits & !bit },
    );
}
#[export_name = "jointSetLimits"]
pub unsafe extern "C" fn set_limits(world: usize, id: usize, mut lower: f32, mut upper: f32) {
    regions::select(world as u32);
    let (key, index) = location(id);
    let kind = records::record(id).joint_type as u32;
    if kind == TY_DISTANCE {
        lower = crate::math::clampf(
            lower,
            0.005 * crate::math::LENGTH_UNITS_PER_METER,
            crate::math::HUGE,
        );
        upper = crate::math::clampf(
            upper,
            0.005 * crate::math::LENGTH_UNITS_PER_METER,
            crate::math::HUGE,
        );
    }
    if kind != TY_WHEEL {
        let lo = if lower < upper { lower } else { upper };
        let hi = if lower > upper { lower } else { upper };
        lower = lo;
        upper = hi;
    }
    if kind == TY_REVOLUTE || kind == TY_SPHERICAL {
        let bound = 0.99 * crate::math::PI;
        lower = crate::math::clampf(lower, -bound, bound);
        upper = crate::math::clampf(upper, -bound, bound);
    }
    let (lo, hi) = match kind {
        TY_DISTANCE => (DJ_MIN_LENGTH, DJ_MAX_LENGTH),
        TY_REVOLUTE => (RJ_LOWER_ANGLE, RJ_UPPER_ANGLE),
        TY_PRISMATIC => (PJ_LOWER_TRANSLATION, PJ_UPPER_TRANSLATION),
        TY_SPHERICAL => (SJ_LOWER_TWIST_ANGLE, SJ_UPPER_TWIST_ANGLE),
        TY_WHEEL => (WHJ_LOWER_SUSPENSION_LIMIT, WHJ_UPPER_SUSPENSION_LIMIT),
        _ => unreachable!(),
    };
    let changed =
        lower != joints::read_float(key, index, lo) || upper != joints::read_float(key, index, hi);
    joints::write_float(key, index, lo, lower);
    joints::write_float(key, index, hi, upper);
    if kind == TY_DISTANCE {
        for field in [DJ_IMPULSE, DJ_LOWER_IMPULSE, DJ_UPPER_IMPULSE] {
            joints::write_float(key, index, field, 0.0);
        }
    }
    if kind == TY_WHEEL && changed {
        for field in [WHJ_LOWER_SUSPENSION_IMPULSE, WHJ_UPPER_SUSPENSION_IMPULSE] {
            joints::write_float(key, index, field, 0.0);
        }
    }
}
#[export_name = "distanceJointSetLength"]
pub unsafe extern "C" fn set_length(world: usize, id: usize, length: f32) {
    regions::select(world as u32);
    let (key, index) = location(id);
    joints::write_float(
        key,
        index,
        DJ_LENGTH,
        crate::math::clampf(
            length,
            0.005 * crate::math::LENGTH_UNITS_PER_METER,
            crate::math::HUGE,
        ),
    );
    for field in [DJ_IMPULSE, DJ_LOWER_IMPULSE, DJ_UPPER_IMPULSE] {
        joints::write_float(key, index, field, 0.0);
    }
}
#[export_name = "motorJointSetMaxSpring"]
pub unsafe extern "C" fn set_max_spring(world: usize, id: usize, torque: bool, value: f32) {
    regions::select(world as u32);
    let (key, index) = location(id);
    joints::write_float(
        key,
        index,
        if torque {
            MJ_MAX_SPRING_TORQUE
        } else {
            MJ_MAX_SPRING_FORCE
        },
        if 0.0 > value { 0.0 } else { value },
    );
}
#[export_name = "jointTransfer"]
pub unsafe extern "C" fn transfer(id: usize, target: usize) {
    let r = *records::record(id);
    if r.set_index as usize == target {
        return;
    }
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
