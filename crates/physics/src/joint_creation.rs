//! joint.c per-kind payload initialization. JointArray::append zeroes the payload,
//! including impulses and solver scratch, before these definition fields are written.
use crate::{constraint_graph as graph, joint_abi::*, joint_record, joints, math::*, regions};

unsafe fn location(world: usize, id: usize) -> (usize, usize) {
    regions::select(world as u32);
    let r = joint_record::record(id);
    (
        if r.set_index == 2 {
            r.color_index as usize
        } else {
            graph::COLORS + r.set_index as usize
        },
        r.local_index as usize,
    )
}

#[export_name = "jointInitDistance"]
pub unsafe extern "C" fn distance(
    world: usize,
    id: usize,
    length: f32,
    hertz: f32,
    damping: f32,
    lower_force: f32,
    upper_force: f32,
    min_length: f32,
    max_length: f32,
    max_force: f32,
    speed: f32,
    spring: bool,
    limit: bool,
    motor: bool,
) {
    let (key, index) = location(world, id);
    for (field, value) in [
        (DJ_LENGTH, maxf(length, 0.005)),
        (DJ_HERTZ, hertz),
        (DJ_DAMPING_RATIO, damping),
        (DJ_LOWER_SPRING_FORCE, lower_force),
        (DJ_UPPER_SPRING_FORCE, upper_force),
        (DJ_MIN_LENGTH, maxf(min_length, 0.005)),
        (DJ_MAX_LENGTH, maxf(min_length, max_length)),
        (DJ_MAX_MOTOR_FORCE, max_force),
        (DJ_MOTOR_SPEED, speed),
    ] {
        joints::write_float(key, index, field, value);
    }
    joints::write_word(
        key,
        index,
        DJ_ENABLE,
        (u32::from(spring) * DJ_ENABLE_SPRING)
            | (u32::from(limit) * DJ_ENABLE_LIMIT)
            | (u32::from(motor) * DJ_ENABLE_MOTOR),
    );
}

#[export_name = "jointInitRevolute"]
pub unsafe extern "C" fn revolute(
    world: usize,
    id: usize,
    hertz: f32,
    damping: f32,
    target: f32,
    lower: f32,
    upper: f32,
    max_torque: f32,
    speed: f32,
    spring: bool,
    limit: bool,
    motor: bool,
) {
    let (key, index) = location(world, id);
    for (field, value) in [
        (RJ_HERTZ, hertz),
        (RJ_DAMPING_RATIO, damping),
        (RJ_TARGET_ANGLE, clampf(target, -PI, PI)),
        (
            RJ_LOWER_ANGLE,
            clampf(minf(lower, upper), -0.99 * PI, 0.99 * PI),
        ),
        (
            RJ_UPPER_ANGLE,
            clampf(maxf(lower, upper), -0.99 * PI, 0.99 * PI),
        ),
        (RJ_MAX_MOTOR_TORQUE, max_torque),
        (RJ_MOTOR_SPEED, speed),
    ] {
        joints::write_float(key, index, field, value);
    }
    joints::write_word(
        key,
        index,
        RJ_ENABLE,
        (u32::from(spring) * RJ_ENABLE_SPRING)
            | (u32::from(limit) * RJ_ENABLE_LIMIT)
            | (u32::from(motor) * RJ_ENABLE_MOTOR),
    );
}

#[export_name = "jointInitSpherical"]
pub unsafe extern "C" fn spherical(
    world: usize,
    id: usize,
    hertz: f32,
    damping: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qs: f32,
    cone: f32,
    lower: f32,
    upper: f32,
    max_torque: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    spring: bool,
    cone_limit: bool,
    twist_limit: bool,
    motor: bool,
) {
    let (key, index) = location(world, id);
    for (field, value) in [
        (SJ_HERTZ, hertz),
        (SJ_DAMPING_RATIO, damping),
        (SJ_TARGET_ROTATION, qx),
        (SJ_TARGET_ROTATION + 1, qy),
        (SJ_TARGET_ROTATION + 2, qz),
        (SJ_TARGET_ROTATION + 3, qs),
        (SJ_CONE_ANGLE, clampf(cone, 0.0, 0.5 * PI)),
        (
            SJ_LOWER_TWIST_ANGLE,
            clampf(minf(lower, upper), -0.99 * PI, 0.99 * PI),
        ),
        (
            SJ_UPPER_TWIST_ANGLE,
            clampf(maxf(lower, upper), -0.99 * PI, 0.99 * PI),
        ),
        (SJ_MAX_MOTOR_TORQUE, max_torque),
        (SJ_MOTOR_VELOCITY, vx),
        (SJ_MOTOR_VELOCITY + 1, vy),
        (SJ_MOTOR_VELOCITY + 2, vz),
    ] {
        joints::write_float(key, index, field, value);
    }
    joints::write_word(
        key,
        index,
        SJ_ENABLE,
        (u32::from(spring) * SJ_ENABLE_SPRING)
            | (u32::from(cone_limit) * SJ_ENABLE_CONE_LIMIT)
            | (u32::from(twist_limit) * SJ_ENABLE_TWIST_LIMIT)
            | (u32::from(motor) * SJ_ENABLE_MOTOR),
    );
}

#[export_name = "jointInitPrismatic"]
pub unsafe extern "C" fn prismatic(
    world: usize,
    id: usize,
    hertz: f32,
    damping: f32,
    target: f32,
    lower: f32,
    upper: f32,
    max_force: f32,
    speed: f32,
    spring: bool,
    limit: bool,
    motor: bool,
) {
    let (key, index) = location(world, id);
    for (field, value) in [
        (PJ_HERTZ, hertz),
        (PJ_DAMPING_RATIO, damping),
        (PJ_TARGET_TRANSLATION, target),
        (PJ_LOWER_TRANSLATION, lower),
        (PJ_UPPER_TRANSLATION, upper),
        (PJ_MAX_MOTOR_FORCE, max_force),
        (PJ_MOTOR_SPEED, speed),
    ] {
        joints::write_float(key, index, field, value);
    }
    joints::write_word(
        key,
        index,
        PJ_ENABLE,
        (u32::from(spring) * PJ_ENABLE_SPRING)
            | (u32::from(limit) * PJ_ENABLE_LIMIT)
            | (u32::from(motor) * PJ_ENABLE_MOTOR),
    );
}

#[export_name = "jointInitWheel"]
pub unsafe extern "C" fn wheel(
    world: usize,
    id: usize,
    suspension_spring: bool,
    suspension_hertz: f32,
    suspension_damping: f32,
    suspension_limit: bool,
    suspension_lower: f32,
    suspension_upper: f32,
    spin_motor: bool,
    max_spin_torque: f32,
    spin_speed: f32,
    steering: bool,
    steering_hertz: f32,
    steering_damping: f32,
    steering_target: f32,
    max_steering_torque: f32,
    steering_limit: bool,
    steering_lower: f32,
    steering_upper: f32,
) {
    let (key, index) = location(world, id);
    for (field, value) in [
        (WHJ_SUSPENSION_HERTZ, suspension_hertz),
        (WHJ_SUSPENSION_DAMPING_RATIO, suspension_damping),
        (WHJ_LOWER_SUSPENSION_LIMIT, suspension_lower),
        (WHJ_UPPER_SUSPENSION_LIMIT, suspension_upper),
        (WHJ_MAX_SPIN_TORQUE, max_spin_torque),
        (WHJ_SPIN_SPEED, spin_speed),
        (WHJ_STEERING_HERTZ, steering_hertz),
        (WHJ_STEERING_DAMPING_RATIO, steering_damping),
        (WHJ_TARGET_STEERING_ANGLE, steering_target),
        (WHJ_MAX_STEERING_TORQUE, max_steering_torque),
        (WHJ_LOWER_STEERING_LIMIT, steering_lower),
        (WHJ_UPPER_STEERING_LIMIT, steering_upper),
    ] {
        joints::write_float(key, index, field, value);
    }
    joints::write_word(
        key,
        index,
        WHJ_ENABLE,
        (u32::from(suspension_spring) * WHJ_ENABLE_SUSPENSION_SPRING)
            | (u32::from(suspension_limit) * WHJ_ENABLE_SUSPENSION_LIMIT)
            | (u32::from(spin_motor) * WHJ_ENABLE_SPIN_MOTOR)
            | (u32::from(steering) * WHJ_ENABLE_STEERING)
            | (u32::from(steering_limit) * WHJ_ENABLE_STEERING_LIMIT),
    );
}

#[export_name = "jointInitWeld"]
pub unsafe extern "C" fn weld(
    world: usize,
    id: usize,
    linear_hertz: f32,
    linear_damping: f32,
    angular_hertz: f32,
    angular_damping: f32,
) {
    let (key, index) = location(world, id);
    for (field, value) in [
        (WJ_LINEAR_HERTZ, linear_hertz),
        (WJ_LINEAR_DAMPING_RATIO, linear_damping),
        (WJ_ANGULAR_HERTZ, angular_hertz),
        (WJ_ANGULAR_DAMPING_RATIO, angular_damping),
    ] {
        joints::write_float(key, index, field, value);
    }
}

#[export_name = "jointInitMotor"]
pub unsafe extern "C" fn motor(
    world: usize,
    id: usize,
    lx: f32,
    ly: f32,
    lz: f32,
    max_force: f32,
    ax: f32,
    ay: f32,
    az: f32,
    max_torque: f32,
    linear_hertz: f32,
    linear_damping: f32,
    spring_force: f32,
    angular_hertz: f32,
    angular_damping: f32,
    spring_torque: f32,
) {
    let (key, index) = location(world, id);
    for (field, value) in [
        (MJ_LINEAR_VELOCITY, lx),
        (MJ_LINEAR_VELOCITY + 1, ly),
        (MJ_LINEAR_VELOCITY + 2, lz),
        (MJ_MAX_VELOCITY_FORCE, max_force),
        (MJ_ANGULAR_VELOCITY, ax),
        (MJ_ANGULAR_VELOCITY + 1, ay),
        (MJ_ANGULAR_VELOCITY + 2, az),
        (MJ_MAX_VELOCITY_TORQUE, max_torque),
        (MJ_LINEAR_HERTZ, linear_hertz),
        (MJ_LINEAR_DAMPING_RATIO, linear_damping),
        (MJ_MAX_SPRING_FORCE, spring_force),
        (MJ_ANGULAR_HERTZ, angular_hertz),
        (MJ_ANGULAR_DAMPING_RATIO, angular_damping),
        (MJ_MAX_SPRING_TORQUE, spring_torque),
    ] {
        joints::write_float(key, index, field, value);
    }
}

#[export_name = "jointInitParallel"]
pub unsafe extern "C" fn parallel(
    world: usize,
    id: usize,
    hertz: f32,
    damping: f32,
    max_torque: f32,
) {
    let (key, index) = location(world, id);
    for (field, value) in [
        (PLJ_HERTZ, hertz),
        (PLJ_DAMPING_RATIO, damping),
        (PLJ_MAX_TORQUE, max_torque),
    ] {
        joints::write_float(key, index, field, value);
    }
}
