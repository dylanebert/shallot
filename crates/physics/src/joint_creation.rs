//! joint.c creation: resolved base definition and one complete per-kind operation.
use crate::{joint_abi::*, joint_record, math::*, regions::MAX_WORLDS};

#[repr(C)]
struct JointDefinition {
    body_a: u32,
    body_b: u32,
    draw_scale: f32,
    collide_connected: u32,
    frame_a: Transform,
    frame_b: Transform,
    force: f32,
    torque: f32,
    hertz: f32,
    damping: f32,
}
static mut DEFINITIONS: [[u32; 22]; MAX_WORLDS] = [[0; 22]; MAX_WORLDS];

// The host writes a resolved definition before the single creation call; no joint exists yet.
#[export_name = "jointDefinitionPtr"]
pub unsafe extern "C" fn definition_ptr(world: usize) -> usize {
    DEFINITIONS[world].as_mut_ptr() as usize
}

unsafe fn create(world: usize, definition: usize, kind: u32) -> usize {
    let def = &*(definition as *const JointDefinition);
    crate::joint_lifecycle::create_in_world(
        world,
        def.body_a as usize,
        def.body_b as usize,
        kind as i32,
        def.draw_scale,
        def.collide_connected != 0,
        def.frame_a.p.x,
        def.frame_a.p.y,
        def.frame_a.p.z,
        def.frame_a.q.v.x,
        def.frame_a.q.v.y,
        def.frame_a.q.v.z,
        def.frame_a.q.s,
        def.frame_b.p.x,
        def.frame_b.p.y,
        def.frame_b.p.z,
        def.frame_b.q.v.x,
        def.frame_b.q.v.y,
        def.frame_b.q.v.z,
        def.frame_b.q.s,
        def.force,
        def.torque,
        def.hertz,
        def.damping,
    ) as usize
}

macro_rules! create_joint {
    ($export:literal, $name:ident, $kind:expr, ($($arg:ident: $ty:ty),*), |$sim:ident| $initialize:block) => {
        #[export_name = $export]
        pub unsafe extern "C" fn $name(world: usize, definition: usize, $($arg: $ty),*) -> u32 {
            let id = create(world, definition, $kind);
            let $sim = &mut *(joint_record::sim_pointer_in_world(world, id) as *mut crate::joint_sim::JointSim);
            $initialize
            id as u32
        }
    };
}

create_joint!("jointCreateDistance", distance, TY_DISTANCE,
    (length: f32, hertz: f32, damping: f32, lower_force: f32, upper_force: f32,
     min_length: f32, max_length: f32, max_force: f32, speed: f32,
     spring: bool, limit: bool, motor: bool), |sim| {
    let joint = &mut sim.payload.distance;
    core::ptr::write_bytes(joint, 0, 1);
    joint.length = maxf(length, 0.005);
    joint.hertz = hertz;
    joint.damping_ratio = damping;
    joint.lower_spring_force = lower_force;
    joint.upper_spring_force = upper_force;
    joint.min_length = maxf(min_length, 0.005);
    joint.max_length = maxf(min_length, max_length);
    joint.max_motor_force = max_force;
    joint.motor_speed = speed;
    joint.enable_spring = spring;
    joint.enable_limit = limit;
    joint.enable_motor = motor;
});

create_joint!("jointCreateRevolute", revolute, TY_REVOLUTE,
    (hertz: f32, damping: f32, target: f32, lower: f32, upper: f32,
     max_torque: f32, speed: f32, spring: bool, limit: bool, motor: bool), |sim| {
    let joint = &mut sim.payload.revolute;
    core::ptr::write_bytes(joint, 0, 1);
    joint.hertz = hertz;
    joint.damping_ratio = damping;
    joint.target_angle = clampf(target, -PI, PI);
    joint.lower_angle = clampf(minf(lower, upper), -0.99 * PI, 0.99 * PI);
    joint.upper_angle = clampf(maxf(lower, upper), -0.99 * PI, 0.99 * PI);
    joint.max_motor_torque = max_torque;
    joint.motor_speed = speed;
    joint.enable_spring = spring;
    joint.enable_limit = limit;
    joint.enable_motor = motor;
});

create_joint!("jointCreateSpherical", spherical, TY_SPHERICAL,
    (hertz: f32, damping: f32, target_x: f32, target_y: f32, target_z: f32, target_s: f32,
     cone: f32, lower: f32, upper: f32, max_torque: f32, vx: f32, vy: f32, vz: f32,
     spring: bool, cone_limit: bool, twist_limit: bool, motor: bool), |sim| {
    let joint = &mut sim.payload.spherical;
    core::ptr::write_bytes(joint, 0, 1);
    joint.hertz = hertz;
    joint.damping_ratio = damping;
    joint.target_rotation = Quat { v: Vec3::new(target_x, target_y, target_z), s: target_s };
    joint.max_motor_torque = max_torque;
    joint.motor_velocity = Vec3::new(vx, vy, vz);
    joint.cone_angle = clampf(cone, 0.0, 0.5 * PI);
    joint.lower_twist_angle = clampf(minf(lower, upper), -0.99 * PI, 0.99 * PI);
    joint.upper_twist_angle = clampf(maxf(lower, upper), -0.99 * PI, 0.99 * PI);
    joint.enable_spring = spring;
    joint.enable_motor = motor;
    joint.enable_cone_limit = cone_limit;
    joint.enable_twist_limit = twist_limit;
});

create_joint!("jointCreatePrismatic", prismatic, TY_PRISMATIC,
    (hertz: f32, damping: f32, target: f32, lower: f32, upper: f32,
     max_force: f32, speed: f32, spring: bool, limit: bool, motor: bool), |sim| {
    let joint = &mut sim.payload.prismatic;
    core::ptr::write_bytes(joint, 0, 1);
    joint.hertz = hertz;
    joint.damping_ratio = damping;
    joint.target_translation = target;
    joint.lower_translation = lower;
    joint.upper_translation = upper;
    joint.max_motor_force = max_force;
    joint.motor_speed = speed;
    joint.enable_spring = spring;
    joint.enable_limit = limit;
    joint.enable_motor = motor;
});

create_joint!("jointCreateWheel", wheel, TY_WHEEL,
    (suspension_spring: bool, suspension_hertz: f32, suspension_damping: f32,
     suspension_limit: bool, suspension_lower: f32, suspension_upper: f32,
     spin_motor: bool, max_spin_torque: f32, spin_speed: f32,
     steering: bool, steering_hertz: f32, steering_damping: f32,
     steering_target: f32, max_steering_torque: f32, steering_limit: bool,
     steering_lower: f32, steering_upper: f32), |sim| {
    let joint = &mut sim.payload.wheel;
    core::ptr::write_bytes(joint, 0, 1);
    joint.suspension_hertz = suspension_hertz;
    joint.suspension_damping_ratio = suspension_damping;
    joint.lower_suspension_limit = suspension_lower;
    joint.upper_suspension_limit = suspension_upper;
    joint.max_spin_torque = max_spin_torque;
    joint.spin_speed = spin_speed;
    joint.steering_hertz = steering_hertz;
    joint.steering_damping_ratio = steering_damping;
    joint.target_steering_angle = steering_target;
    joint.max_steering_torque = max_steering_torque;
    joint.lower_steering_limit = steering_lower;
    joint.upper_steering_limit = steering_upper;
    joint.enable_suspension_spring = suspension_spring;
    joint.enable_suspension_limit = suspension_limit;
    joint.enable_spin_motor = spin_motor;
    joint.enable_steering = steering;
    joint.enable_steering_limit = steering_limit;
});

create_joint!("jointCreateWeld", weld, TY_WELD,
    (linear_hertz: f32, linear_damping: f32, angular_hertz: f32, angular_damping: f32), |sim| {
    let joint = &mut sim.payload.weld;
    core::ptr::write_bytes(joint, 0, 1);
    joint.linear_hertz = linear_hertz;
    joint.linear_damping_ratio = linear_damping;
    joint.angular_hertz = angular_hertz;
    joint.angular_damping_ratio = angular_damping;
});

create_joint!("jointCreateParallel", parallel, TY_PARALLEL,
    (hertz: f32, damping: f32, max_torque: f32), |sim| {
    let joint = &mut sim.payload.parallel;
    core::ptr::write_bytes(joint, 0, 1);
    joint.hertz = hertz;
    joint.damping_ratio = damping;
    joint.max_torque = max_torque;
});

create_joint!("jointCreateMotor", motor, TY_MOTOR,
    (lx: f32, ly: f32, lz: f32, max_velocity_force: f32,
     ax: f32, ay: f32, az: f32, max_velocity_torque: f32,
     linear_hertz: f32, linear_damping: f32, max_spring_force: f32,
     angular_hertz: f32, angular_damping: f32, max_spring_torque: f32), |sim| {
    let joint = &mut sim.payload.motor;
    core::ptr::write_bytes(joint, 0, 1);
    joint.linear_velocity = Vec3::new(lx, ly, lz);
    joint.angular_velocity = Vec3::new(ax, ay, az);
    joint.max_velocity_force = max_velocity_force;
    joint.max_velocity_torque = max_velocity_torque;
    joint.linear_hertz = linear_hertz;
    joint.linear_damping_ratio = linear_damping;
    joint.max_spring_force = max_spring_force;
    joint.angular_hertz = angular_hertz;
    joint.angular_damping_ratio = angular_damping;
    joint.max_spring_torque = max_spring_torque;
});
