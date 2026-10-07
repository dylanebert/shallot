//! Word offsets for numeric bindings; tagged byte offsets for boolean bindings.
use crate::joint_sim::*;
use core::mem::{offset_of, size_of};

pub const BOOL_FIELD: usize = 1 << 16;
macro_rules! layout {
    ($($name:ident = $value:expr;)*) => {
        $(pub const $name: usize = $value;)*
        const FIELDS: &[(&str, usize)] = &[$((stringify!($name), $name)),*];
    };
}
macro_rules! field {
    ($ty:ty, $field:ident) => {
        (offset_of!(JointSim, payload) + offset_of!($ty, $field)) / 4
    };
}
macro_rules! flag {
    ($ty:ty, $field:ident) => {
        BOOL_FIELD | (offset_of!(JointSim, payload) + offset_of!($ty, $field))
    };
}
layout! {
    JOINT_STRIDE = size_of::<JointSim>() / 4;
    J_JOINT_ID = offset_of!(JointSim, joint_id) / 4;
    J_TYPE = offset_of!(JointSim, joint_type) / 4;
    J_BODY_ID_A = offset_of!(JointSim, body_id_a) / 4;
    J_BODY_ID_B = offset_of!(JointSim, body_id_b) / 4;
    J_LOCAL_FRAME_A = offset_of!(JointSim, local_frame_a) / 4;
    J_LOCAL_FRAME_B = offset_of!(JointSim, local_frame_b) / 4;
    J_INV_MASS_A = offset_of!(JointSim, inv_mass_a) / 4;
    J_INV_MASS_B = offset_of!(JointSim, inv_mass_b) / 4;
    J_INV_IA = offset_of!(JointSim, inv_ia) / 4;
    J_INV_IB = offset_of!(JointSim, inv_ib) / 4;
    J_CONSTRAINT_HERTZ = offset_of!(JointSim, constraint_hertz) / 4;
    J_CONSTRAINT_DAMPING = offset_of!(JointSim, constraint_damping_ratio) / 4;
    J_CONSTRAINT_SOFTNESS = offset_of!(JointSim, constraint_softness) / 4;
    J_FORCE_THRESHOLD = offset_of!(JointSim, force_threshold) / 4;
    J_TORQUE_THRESHOLD = offset_of!(JointSim, torque_threshold) / 4;
    J_PAYLOAD = offset_of!(JointSim, payload) / 4;
    J_FIXED_ROTATION = BOOL_FIELD | offset_of!(JointSim, fixed_rotation);
    DJ_LENGTH = field!(DistanceJoint, length);
    DJ_HERTZ = field!(DistanceJoint, hertz);
    DJ_DAMPING_RATIO = field!(DistanceJoint, damping_ratio);
    DJ_LOWER_SPRING_FORCE = field!(DistanceJoint, lower_spring_force);
    DJ_UPPER_SPRING_FORCE = field!(DistanceJoint, upper_spring_force);
    DJ_MIN_LENGTH = field!(DistanceJoint, min_length);
    DJ_MAX_LENGTH = field!(DistanceJoint, max_length);
    DJ_MAX_MOTOR_FORCE = field!(DistanceJoint, max_motor_force);
    DJ_MOTOR_SPEED = field!(DistanceJoint, motor_speed);
    DJ_IMPULSE = field!(DistanceJoint, impulse);
    DJ_LOWER_IMPULSE = field!(DistanceJoint, lower_impulse);
    DJ_UPPER_IMPULSE = field!(DistanceJoint, upper_impulse);
    DJ_MOTOR_IMPULSE = field!(DistanceJoint, motor_impulse);
    DJ_ANCHOR_A = field!(DistanceJoint, anchor_a);
    DJ_ANCHOR_B = field!(DistanceJoint, anchor_b);
    DJ_DELTA_CENTER = field!(DistanceJoint, delta_center);
    DJ_AXIAL_MASS = field!(DistanceJoint, axial_mass);
    DJ_DIST_SOFTNESS = field!(DistanceJoint, distance_softness);
    DJ_ENABLE = flag!(DistanceJoint, enable_spring);
    WJ_LINEAR_HERTZ = field!(WeldJoint, linear_hertz);
    WJ_LINEAR_DAMPING_RATIO = field!(WeldJoint, linear_damping_ratio);
    WJ_ANGULAR_HERTZ = field!(WeldJoint, angular_hertz);
    WJ_ANGULAR_DAMPING_RATIO = field!(WeldJoint, angular_damping_ratio);
    WJ_LINEAR_IMPULSE = field!(WeldJoint, linear_impulse);
    WJ_ANGULAR_IMPULSE = field!(WeldJoint, angular_impulse);
    WJ_FRAME_A = field!(WeldJoint, frame_a);
    WJ_FRAME_B = field!(WeldJoint, frame_b);
    WJ_DELTA_CENTER = field!(WeldJoint, delta_center);
    WJ_ANGULAR_MASS = field!(WeldJoint, angular_mass);
    WJ_LINEAR_SPRING = field!(WeldJoint, linear_spring);
    WJ_ANGULAR_SPRING = field!(WeldJoint, angular_spring);
    WJ_FIXED_ROTATION = J_FIXED_ROTATION;
    RJ_HERTZ = field!(RevoluteJoint, hertz);
    RJ_DAMPING_RATIO = field!(RevoluteJoint, damping_ratio);
    RJ_MAX_MOTOR_TORQUE = field!(RevoluteJoint, max_motor_torque);
    RJ_MOTOR_SPEED = field!(RevoluteJoint, motor_speed);
    RJ_TARGET_ANGLE = field!(RevoluteJoint, target_angle);
    RJ_LOWER_ANGLE = field!(RevoluteJoint, lower_angle);
    RJ_UPPER_ANGLE = field!(RevoluteJoint, upper_angle);
    RJ_LINEAR_IMPULSE = field!(RevoluteJoint, linear_impulse);
    RJ_PERP_IMPULSE = field!(RevoluteJoint, perp_impulse);
    RJ_SPRING_IMPULSE = field!(RevoluteJoint, spring_impulse);
    RJ_MOTOR_IMPULSE = field!(RevoluteJoint, motor_impulse);
    RJ_LOWER_IMPULSE = field!(RevoluteJoint, lower_impulse);
    RJ_UPPER_IMPULSE = field!(RevoluteJoint, upper_impulse);
    RJ_FRAME_A = field!(RevoluteJoint, frame_a);
    RJ_FRAME_B = field!(RevoluteJoint, frame_b);
    RJ_ROTATION_AXIS_Z = field!(RevoluteJoint, rotation_axis_z);
    RJ_PERP_AXIS_X = field!(RevoluteJoint, perp_axis_x);
    RJ_PERP_AXIS_Y = field!(RevoluteJoint, perp_axis_y);
    RJ_DELTA_CENTER = field!(RevoluteJoint, delta_center);
    RJ_AXIAL_MASS = field!(RevoluteJoint, axial_mass);
    RJ_SPRING_SOFTNESS = field!(RevoluteJoint, spring_softness);
    RJ_FIXED_ROTATION = J_FIXED_ROTATION;
    RJ_ENABLE = flag!(RevoluteJoint, enable_spring);
    PJ_HERTZ = field!(PrismaticJoint, hertz);
    PJ_DAMPING_RATIO = field!(PrismaticJoint, damping_ratio);
    PJ_MAX_MOTOR_FORCE = field!(PrismaticJoint, max_motor_force);
    PJ_MOTOR_SPEED = field!(PrismaticJoint, motor_speed);
    PJ_TARGET_TRANSLATION = field!(PrismaticJoint, target_translation);
    PJ_LOWER_TRANSLATION = field!(PrismaticJoint, lower_translation);
    PJ_UPPER_TRANSLATION = field!(PrismaticJoint, upper_translation);
    PJ_PERP_IMPULSE = field!(PrismaticJoint, perp_impulse);
    PJ_ANGULAR_IMPULSE = field!(PrismaticJoint, angular_impulse);
    PJ_SPRING_IMPULSE = field!(PrismaticJoint, spring_impulse);
    PJ_MOTOR_IMPULSE = field!(PrismaticJoint, motor_impulse);
    PJ_LOWER_IMPULSE = field!(PrismaticJoint, lower_impulse);
    PJ_UPPER_IMPULSE = field!(PrismaticJoint, upper_impulse);
    PJ_FRAME_A = field!(PrismaticJoint, frame_a);
    PJ_FRAME_B = field!(PrismaticJoint, frame_b);
    PJ_JOINT_AXIS = field!(PrismaticJoint, joint_axis);
    PJ_PERP_AXIS_Y = field!(PrismaticJoint, perp_axis_y);
    PJ_PERP_AXIS_Z = field!(PrismaticJoint, perp_axis_z);
    PJ_DELTA_CENTER = field!(PrismaticJoint, delta_center);
    PJ_ROTATION_MASS = field!(PrismaticJoint, rotation_mass);
    PJ_SPRING_SOFTNESS = field!(PrismaticJoint, spring_softness);
    PJ_FIXED_ROTATION = J_FIXED_ROTATION;
    PJ_ENABLE = flag!(PrismaticJoint, enable_spring);
    SJ_HERTZ = field!(SphericalJoint, hertz);
    SJ_DAMPING_RATIO = field!(SphericalJoint, damping_ratio);
    SJ_MAX_MOTOR_TORQUE = field!(SphericalJoint, max_motor_torque);
    SJ_MOTOR_VELOCITY = field!(SphericalJoint, motor_velocity);
    SJ_LOWER_TWIST_ANGLE = field!(SphericalJoint, lower_twist_angle);
    SJ_UPPER_TWIST_ANGLE = field!(SphericalJoint, upper_twist_angle);
    SJ_CONE_ANGLE = field!(SphericalJoint, cone_angle);
    SJ_TARGET_ROTATION = field!(SphericalJoint, target_rotation);
    SJ_LINEAR_IMPULSE = field!(SphericalJoint, linear_impulse);
    SJ_SPRING_IMPULSE = field!(SphericalJoint, spring_impulse);
    SJ_MOTOR_IMPULSE = field!(SphericalJoint, motor_impulse);
    SJ_LOWER_TWIST_IMPULSE = field!(SphericalJoint, lower_twist_impulse);
    SJ_UPPER_TWIST_IMPULSE = field!(SphericalJoint, upper_twist_impulse);
    SJ_SWING_IMPULSE = field!(SphericalJoint, swing_impulse);
    SJ_FRAME_A = field!(SphericalJoint, frame_a);
    SJ_FRAME_B = field!(SphericalJoint, frame_b);
    SJ_DELTA_CENTER = field!(SphericalJoint, delta_center);
    SJ_SWING_AXIS = field!(SphericalJoint, swing_axis);
    SJ_TWIST_JACOBIAN = field!(SphericalJoint, twist_jacobian);
    SJ_ROTATION_MASS = field!(SphericalJoint, rotation_mass);
    SJ_SWING_MASS = field!(SphericalJoint, swing_mass);
    SJ_TWIST_MASS = field!(SphericalJoint, twist_mass);
    SJ_SPRING_SOFTNESS = field!(SphericalJoint, spring_softness);
    SJ_FIXED_ROTATION = J_FIXED_ROTATION;
    SJ_ENABLE = flag!(SphericalJoint, enable_spring);
    WHJ_MAX_SPIN_TORQUE = field!(WheelJoint, max_spin_torque);
    WHJ_SPIN_SPEED = field!(WheelJoint, spin_speed);
    WHJ_LOWER_SUSPENSION_LIMIT = field!(WheelJoint, lower_suspension_limit);
    WHJ_UPPER_SUSPENSION_LIMIT = field!(WheelJoint, upper_suspension_limit);
    WHJ_SUSPENSION_HERTZ = field!(WheelJoint, suspension_hertz);
    WHJ_SUSPENSION_DAMPING_RATIO = field!(WheelJoint, suspension_damping_ratio);
    WHJ_LOWER_STEERING_LIMIT = field!(WheelJoint, lower_steering_limit);
    WHJ_UPPER_STEERING_LIMIT = field!(WheelJoint, upper_steering_limit);
    WHJ_TARGET_STEERING_ANGLE = field!(WheelJoint, target_steering_angle);
    WHJ_MAX_STEERING_TORQUE = field!(WheelJoint, max_steering_torque);
    WHJ_STEERING_HERTZ = field!(WheelJoint, steering_hertz);
    WHJ_STEERING_DAMPING_RATIO = field!(WheelJoint, steering_damping_ratio);
    WHJ_LINEAR_IMPULSE = field!(WheelJoint, linear_impulse);
    WHJ_ANGULAR_IMPULSE = field!(WheelJoint, angular_impulse);
    WHJ_SPIN_IMPULSE = field!(WheelJoint, spin_impulse);
    WHJ_SUSPENSION_SPRING_IMPULSE = field!(WheelJoint, suspension_spring_impulse);
    WHJ_LOWER_SUSPENSION_IMPULSE = field!(WheelJoint, lower_suspension_impulse);
    WHJ_UPPER_SUSPENSION_IMPULSE = field!(WheelJoint, upper_suspension_impulse);
    WHJ_STEERING_SPRING_IMPULSE = field!(WheelJoint, steering_spring_impulse);
    WHJ_LOWER_STEERING_IMPULSE = field!(WheelJoint, lower_steering_impulse);
    WHJ_UPPER_STEERING_IMPULSE = field!(WheelJoint, upper_steering_impulse);
    WHJ_FRAME_A = field!(WheelJoint, frame_a);
    WHJ_FRAME_B = field!(WheelJoint, frame_b);
    WHJ_DELTA_CENTER = field!(WheelJoint, delta_center);
    WHJ_SPIN_MASS = field!(WheelJoint, spin_mass);
    WHJ_SUSPENSION_MASS = field!(WheelJoint, suspension_mass);
    WHJ_STEERING_MASS = field!(WheelJoint, steering_mass);
    WHJ_SUSPENSION_SOFTNESS = field!(WheelJoint, suspension_softness);
    WHJ_STEERING_SOFTNESS = field!(WheelJoint, steering_softness);
    WHJ_FIXED_ROTATION = J_FIXED_ROTATION;
    WHJ_ENABLE = flag!(WheelJoint, enable_spin_motor);
    MJ_LINEAR_VELOCITY = field!(MotorJoint, linear_velocity);
    MJ_ANGULAR_VELOCITY = field!(MotorJoint, angular_velocity);
    MJ_MAX_VELOCITY_FORCE = field!(MotorJoint, max_velocity_force);
    MJ_MAX_VELOCITY_TORQUE = field!(MotorJoint, max_velocity_torque);
    MJ_LINEAR_HERTZ = field!(MotorJoint, linear_hertz);
    MJ_LINEAR_DAMPING_RATIO = field!(MotorJoint, linear_damping_ratio);
    MJ_ANGULAR_HERTZ = field!(MotorJoint, angular_hertz);
    MJ_ANGULAR_DAMPING_RATIO = field!(MotorJoint, angular_damping_ratio);
    MJ_MAX_SPRING_FORCE = field!(MotorJoint, max_spring_force);
    MJ_MAX_SPRING_TORQUE = field!(MotorJoint, max_spring_torque);
    MJ_LINEAR_VELOCITY_IMPULSE = field!(MotorJoint, linear_velocity_impulse);
    MJ_ANGULAR_VELOCITY_IMPULSE = field!(MotorJoint, angular_velocity_impulse);
    MJ_LINEAR_SPRING_IMPULSE = field!(MotorJoint, linear_spring_impulse);
    MJ_ANGULAR_SPRING_IMPULSE = field!(MotorJoint, angular_spring_impulse);
    MJ_FRAME_A = field!(MotorJoint, frame_a);
    MJ_FRAME_B = field!(MotorJoint, frame_b);
    MJ_DELTA_CENTER = field!(MotorJoint, delta_center);
    MJ_LINEAR_SPRING = field!(MotorJoint, linear_spring);
    MJ_ANGULAR_SPRING = field!(MotorJoint, angular_spring);
    MJ_ANGULAR_MASS = field!(MotorJoint, angular_mass);
    PLJ_HERTZ = field!(ParallelJoint, hertz);
    PLJ_DAMPING_RATIO = field!(ParallelJoint, damping_ratio);
    PLJ_MAX_TORQUE = field!(ParallelJoint, max_torque);
    PLJ_PERP_IMPULSE = field!(ParallelJoint, perp_impulse);
    PLJ_QUAT_A = field!(ParallelJoint, quat_a);
    PLJ_QUAT_B = field!(ParallelJoint, quat_b);
    PLJ_PERP_AXIS_X = field!(ParallelJoint, perp_axis_x);
    PLJ_PERP_AXIS_Y = field!(ParallelJoint, perp_axis_y);
    PLJ_SOFTNESS = field!(ParallelJoint, softness);
    PLJ_FIXED_ROTATION = J_FIXED_ROTATION;
}
macro_rules! masks {
    ($($name:ident = $value:expr;)*) => {
        $(pub const $name: u32 = $value;)*
        const MASKS: &[(&str, usize)] = &[$((stringify!($name), $name as usize)),*];
    };
}
masks! {
    DJ_ENABLE_SPRING = 1;
    DJ_ENABLE_LIMIT = 2;
    DJ_ENABLE_MOTOR = 4;
    RJ_ENABLE_SPRING = 1;
    RJ_ENABLE_MOTOR = 2;
    RJ_ENABLE_LIMIT = 4;
    PJ_ENABLE_SPRING = 1;
    PJ_ENABLE_LIMIT = 2;
    PJ_ENABLE_MOTOR = 4;
    SJ_ENABLE_SPRING = 1;
    SJ_ENABLE_MOTOR = 2;
    SJ_ENABLE_CONE_LIMIT = 4;
    SJ_ENABLE_TWIST_LIMIT = 8;
    WHJ_ENABLE_SPIN_MOTOR = 1;
    WHJ_ENABLE_SUSPENSION_SPRING = 2;
    WHJ_ENABLE_SUSPENSION_LIMIT = 4;
    WHJ_ENABLE_STEERING = 8;
    WHJ_ENABLE_STEERING_LIMIT = 16;
}

#[cfg(target_arch = "wasm32")]
#[export_name = "jointLayoutNamePtr"]
pub extern "C" fn name_ptr(index: usize) -> usize {
    FIELDS
        .get(index)
        .or_else(|| MASKS.get(index - FIELDS.len()))
        .map_or(0, |field| field.0.as_ptr() as usize)
}
#[cfg(target_arch = "wasm32")]
#[export_name = "jointLayoutNameLen"]
pub extern "C" fn name_len(index: usize) -> usize {
    if index < FIELDS.len() {
        FIELDS[index].0.len()
    } else {
        MASKS[index - FIELDS.len()].0.len()
    }
}
#[cfg(target_arch = "wasm32")]
#[export_name = "jointLayoutOffset"]
pub extern "C" fn offset(index: usize) -> usize {
    if index < FIELDS.len() {
        FIELDS[index].1
    } else {
        MASKS[index - FIELDS.len()].1
    }
}
