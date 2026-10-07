//! joint.h's joint simulation records and discriminated payload union.
use crate::contact::Softness;
use crate::math::{Mat3, Quat, Transform, Vec2, Vec3};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DistanceJoint {
    pub length: f32,
    pub hertz: f32,
    pub damping_ratio: f32,
    pub lower_spring_force: f32,
    pub upper_spring_force: f32,
    pub min_length: f32,
    pub max_length: f32,
    pub max_motor_force: f32,
    pub motor_speed: f32,
    pub impulse: f32,
    pub lower_impulse: f32,
    pub upper_impulse: f32,
    pub motor_impulse: f32,
    pub index_a: i32,
    pub index_b: i32,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub delta_center: Vec3,
    pub distance_softness: Softness,
    pub axial_mass: f32,
    pub enable_spring: bool,
    pub enable_limit: bool,
    pub enable_motor: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MotorJoint {
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub max_velocity_force: f32,
    pub max_velocity_torque: f32,
    pub linear_hertz: f32,
    pub linear_damping_ratio: f32,
    pub max_spring_force: f32,
    pub angular_hertz: f32,
    pub angular_damping_ratio: f32,
    pub max_spring_torque: f32,
    pub linear_velocity_impulse: Vec3,
    pub angular_velocity_impulse: Vec3,
    pub linear_spring_impulse: Vec3,
    pub angular_spring_impulse: Vec3,
    pub linear_spring: Softness,
    pub angular_spring: Softness,
    pub index_a: i32,
    pub index_b: i32,
    pub frame_a: Transform,
    pub frame_b: Transform,
    pub delta_center: Vec3,
    pub angular_mass: Mat3,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ParallelJoint {
    pub hertz: f32,
    pub damping_ratio: f32,
    pub max_torque: f32,
    pub perp_impulse: Vec2,
    pub perp_axis_x: Vec3,
    pub perp_axis_y: Vec3,
    pub quat_a: Quat,
    pub quat_b: Quat,
    pub index_a: i32,
    pub index_b: i32,
    pub softness: Softness,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PrismaticJoint {
    pub perp_impulse: Vec2,
    pub angular_impulse: Vec3,
    pub spring_impulse: f32,
    pub motor_impulse: f32,
    pub lower_impulse: f32,
    pub upper_impulse: f32,
    pub hertz: f32,
    pub damping_ratio: f32,
    pub max_motor_force: f32,
    pub motor_speed: f32,
    pub target_translation: f32,
    pub lower_translation: f32,
    pub upper_translation: f32,
    pub index_a: i32,
    pub index_b: i32,
    pub frame_a: Transform,
    pub frame_b: Transform,
    pub joint_axis: Vec3,
    pub perp_axis_y: Vec3,
    pub perp_axis_z: Vec3,
    pub delta_center: Vec3,
    pub delta_angle: f32,
    pub rotation_mass: Mat3,
    pub spring_softness: Softness,
    pub enable_spring: bool,
    pub enable_limit: bool,
    pub enable_motor: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RevoluteJoint {
    pub linear_impulse: Vec3,
    pub perp_impulse: Vec2,
    pub spring_impulse: f32,
    pub motor_impulse: f32,
    pub lower_impulse: f32,
    pub upper_impulse: f32,
    pub hertz: f32,
    pub damping_ratio: f32,
    pub max_motor_torque: f32,
    pub motor_speed: f32,
    pub target_angle: f32,
    pub lower_angle: f32,
    pub upper_angle: f32,
    pub index_a: i32,
    pub index_b: i32,
    pub frame_a: Transform,
    pub frame_b: Transform,
    pub rotation_axis_z: Vec3,
    pub perp_axis_x: Vec3,
    pub perp_axis_y: Vec3,
    pub delta_center: Vec3,
    pub delta_angle: f32,
    pub axial_mass: f32,
    pub spring_softness: Softness,
    pub enable_spring: bool,
    pub enable_motor: bool,
    pub enable_limit: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SphericalJoint {
    pub linear_impulse: Vec3,
    pub spring_impulse: Vec3,
    pub motor_impulse: Vec3,
    pub lower_twist_impulse: f32,
    pub upper_twist_impulse: f32,
    pub swing_impulse: f32,
    pub hertz: f32,
    pub damping_ratio: f32,
    pub max_motor_torque: f32,
    pub motor_velocity: Vec3,
    pub lower_twist_angle: f32,
    pub upper_twist_angle: f32,
    pub cone_angle: f32,
    pub target_rotation: Quat,
    pub index_a: i32,
    pub index_b: i32,
    pub frame_a: Transform,
    pub frame_b: Transform,
    pub delta_center: Vec3,
    pub swing_axis: Vec3,
    pub twist_jacobian: Vec3,
    pub rotation_mass: Mat3,
    pub swing_mass: f32,
    pub twist_mass: f32,
    pub spring_softness: Softness,
    pub enable_spring: bool,
    pub enable_motor: bool,
    pub enable_cone_limit: bool,
    pub enable_twist_limit: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeldJoint {
    pub linear_hertz: f32,
    pub linear_damping_ratio: f32,
    pub angular_hertz: f32,
    pub angular_damping_ratio: f32,
    pub linear_spring: Softness,
    pub angular_spring: Softness,
    pub linear_impulse: Vec3,
    pub angular_impulse: Vec3,
    pub index_a: i32,
    pub index_b: i32,
    pub frame_a: Transform,
    pub frame_b: Transform,
    pub delta_center: Vec3,
    pub angular_mass: Mat3,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WheelJoint {
    pub linear_impulse: Vec2,
    pub angular_impulse: Vec2,
    pub spin_impulse: f32,
    pub max_spin_torque: f32,
    pub spin_speed: f32,
    pub suspension_spring_impulse: f32,
    pub lower_suspension_impulse: f32,
    pub upper_suspension_impulse: f32,
    pub lower_suspension_limit: f32,
    pub upper_suspension_limit: f32,
    pub suspension_hertz: f32,
    pub suspension_damping_ratio: f32,
    pub steering_spring_impulse: f32,
    pub lower_steering_impulse: f32,
    pub upper_steering_impulse: f32,
    pub lower_steering_limit: f32,
    pub upper_steering_limit: f32,
    pub target_steering_angle: f32,
    pub max_steering_torque: f32,
    pub steering_hertz: f32,
    pub steering_damping_ratio: f32,
    pub index_a: i32,
    pub index_b: i32,
    pub frame_a: Transform,
    pub frame_b: Transform,
    pub delta_center: Vec3,
    pub spin_mass: f32,
    pub suspension_mass: f32,
    pub steering_mass: f32,
    pub suspension_softness: Softness,
    pub steering_softness: Softness,
    pub enable_spin_motor: bool,
    pub enable_suspension_spring: bool,
    pub enable_suspension_limit: bool,
    pub enable_steering: bool,
    pub enable_steering_limit: bool,
    pub enable_steering_motor: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union JointPayload {
    pub distance: DistanceJoint,
    pub motor: MotorJoint,
    pub parallel: ParallelJoint,
    pub revolute: RevoluteJoint,
    pub spherical: SphericalJoint,
    pub prismatic: PrismaticJoint,
    pub weld: WeldJoint,
    pub wheel: WheelJoint,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct JointSim {
    pub joint_id: i32,
    pub body_id_a: i32,
    pub body_id_b: i32,
    pub joint_type: u32,
    pub local_frame_a: Transform,
    pub local_frame_b: Transform,
    pub inv_mass_a: f32,
    pub inv_mass_b: f32,
    pub inv_ia: Mat3,
    pub inv_ib: Mat3,
    pub constraint_hertz: f32,
    pub constraint_damping_ratio: f32,
    pub constraint_softness: Softness,
    pub force_threshold: f32,
    pub torque_threshold: f32,
    pub fixed_rotation: bool,
    pub payload: JointPayload,
}
