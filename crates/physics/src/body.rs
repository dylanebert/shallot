//! Resident b3BodySim and b3BodyState word layouts (Box3D body.h).
use crate::col::Col;
use crate::math::{Mat3, Quat, Vec3};

pub const STATE_STRIDE: usize = 16;
pub const STATE_LIVE: usize = 13;
pub const STATE_FLAGS: usize = 13;
pub const SIM_STRIDE: usize = 54;
// Thin column bindings address the same resident sim array.
pub const FIN_STRIDE: usize = SIM_STRIDE;
pub const SIM2_STRIDE: usize = SIM_STRIDE;
pub const TRANSFORM_P: usize = 0;
pub const ROTATION: usize = 3;
pub const CENTER: usize = 7;
pub const S2_ROTATION0: usize = 10;
pub const S2_CENTER0: usize = 14;
pub const LOCAL_CENTER: usize = 17;
pub const FORCE: usize = 20;
pub const TORQUE: usize = 23;
pub const INV_MASS: usize = 26;
pub const INV_INERTIA_LOCAL: usize = 27;
pub const INV_INERTIA_WORLD: usize = 36;
pub const S2_MIN_EXTENT: usize = 45;
pub const MAX_EXTENT: usize = 46;
pub const LINEAR_DAMPING: usize = 49;
pub const ANGULAR_DAMPING: usize = 50;
pub const GRAVITY_SCALE: usize = 51;
pub const S2_BODY_ID: usize = 52;
pub const S2_FLAGS: usize = 53;

#[repr(C)]
pub struct BodySim {
    pub transform: crate::math::Transform,
    pub center: Vec3,
    pub rotation0: Quat,
    pub center0: Vec3,
    pub local_center: Vec3,
    pub force: Vec3,
    pub torque: Vec3,
    pub inv_mass: f32,
    pub inv_inertia_local: Mat3,
    pub inv_inertia_world: Mat3,
    pub min_extent: f32,
    pub max_extent: Vec3,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub body_id: i32,
    pub flags: u32,
}

#[repr(C)]
pub struct BodyState {
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub delta_position: Vec3,
    pub delta_rotation: Quat,
    pub flags: u32,
    pub padding: [u32; 2],
}
const _: () = assert!(core::mem::size_of::<BodySim>() == SIM_STRIDE * 4);
const _: () = assert!(core::mem::size_of::<BodyState>() == STATE_STRIDE * 4);
const _: () = {
    use core::mem::offset_of;
    assert!(offset_of!(BodyState, flags) == STATE_FLAGS * 4);
    assert!(offset_of!(BodySim, transform) == TRANSFORM_P * 4);
    assert!(offset_of!(BodySim, center) == CENTER * 4);
    assert!(offset_of!(BodySim, rotation0) == S2_ROTATION0 * 4);
    assert!(offset_of!(BodySim, center0) == S2_CENTER0 * 4);
    assert!(offset_of!(BodySim, local_center) == LOCAL_CENTER * 4);
    assert!(offset_of!(BodySim, force) == FORCE * 4);
    assert!(offset_of!(BodySim, torque) == TORQUE * 4);
    assert!(offset_of!(BodySim, inv_mass) == INV_MASS * 4);
    assert!(offset_of!(BodySim, inv_inertia_local) == INV_INERTIA_LOCAL * 4);
    assert!(offset_of!(BodySim, inv_inertia_world) == INV_INERTIA_WORLD * 4);
    assert!(offset_of!(BodySim, min_extent) == S2_MIN_EXTENT * 4);
    assert!(offset_of!(BodySim, max_extent) == MAX_EXTENT * 4);
    assert!(offset_of!(BodySim, linear_damping) == LINEAR_DAMPING * 4);
    assert!(offset_of!(BodySim, angular_damping) == ANGULAR_DAMPING * 4);
    assert!(offset_of!(BodySim, gravity_scale) == GRAVITY_SCALE * 4);
    assert!(offset_of!(BodySim, body_id) == S2_BODY_ID * 4);
    assert!(offset_of!(BodySim, flags) == S2_FLAGS * 4);
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub delta_position: Vec3,
    pub delta_rotation: Quat,
}
#[inline]
fn vec3(col: Col<f32>, o: usize) -> Vec3 {
    Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2))
}
#[inline]
fn put_vec3(col: Col<f32>, o: usize, v: Vec3) {
    col.set(o, v.x);
    col.set(o + 1, v.y);
    col.set(o + 2, v.z);
}
#[inline]
fn quat(col: Col<f32>, o: usize) -> Quat {
    Quat {
        v: vec3(col, o),
        s: col.get(o + 3),
    }
}
#[inline]
fn put_quat(col: Col<f32>, o: usize, q: Quat) {
    put_vec3(col, o, q.v);
    col.set(o + 3, q.s);
}
#[inline]
pub fn read_state(col: Col<f32>, i: usize) -> State {
    let o = i * STATE_STRIDE;
    State {
        linear_velocity: vec3(col, o),
        angular_velocity: vec3(col, o + 3),
        delta_position: vec3(col, o + 6),
        delta_rotation: quat(col, o + 9),
    }
}
#[inline]
pub fn write_state(col: Col<f32>, i: usize, s: &State) {
    let o = i * STATE_STRIDE;
    put_vec3(col, o, s.linear_velocity);
    put_vec3(col, o + 3, s.angular_velocity);
    put_vec3(col, o + 6, s.delta_position);
    put_quat(col, o + 9, s.delta_rotation);
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimIntegrate {
    pub inv_mass: f32,
    pub gravity_scale: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub force: Vec3,
    pub torque: Vec3,
    pub inv_inertia_local: Mat3,
    pub inv_inertia_world: Mat3,
    pub rotation: Quat,
}
#[inline]
fn mat3(col: Col<f32>, o: usize) -> Mat3 {
    Mat3 {
        cx: vec3(col, o),
        cy: vec3(col, o + 3),
        cz: vec3(col, o + 6),
    }
}
#[inline]
pub fn read_sim(col: Col<f32>, i: usize) -> SimIntegrate {
    let o = i * SIM_STRIDE;
    SimIntegrate {
        inv_mass: col.get(o + INV_MASS),
        gravity_scale: col.get(o + GRAVITY_SCALE),
        linear_damping: col.get(o + LINEAR_DAMPING),
        angular_damping: col.get(o + ANGULAR_DAMPING),
        force: vec3(col, o + FORCE),
        torque: vec3(col, o + TORQUE),
        inv_inertia_local: mat3(col, o + INV_INERTIA_LOCAL),
        inv_inertia_world: mat3(col, o + INV_INERTIA_WORLD),
        rotation: quat(col, o + ROTATION),
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimFinalize {
    pub center: Vec3,
    pub local_center: Vec3,
    pub max_extent: Vec3,
}
#[inline]
pub fn read_fin(col: Col<f32>, i: usize) -> SimFinalize {
    let o = i * SIM_STRIDE;
    SimFinalize {
        center: vec3(col, o + CENTER),
        local_center: vec3(col, o + LOCAL_CENTER),
        max_extent: vec3(col, o + MAX_EXTENT),
    }
}
#[inline]
pub fn write_fin_center(col: Col<f32>, i: usize, center: Vec3) {
    put_vec3(col, i * SIM_STRIDE + CENTER, center);
}
#[inline]
pub fn write_fin_transform_p(col: Col<f32>, i: usize, p: Vec3) {
    put_vec3(col, i * SIM_STRIDE + TRANSFORM_P, p);
}
#[inline]
pub fn write_sim_rotation(col: Col<f32>, i: usize, q: Quat) {
    put_quat(col, i * SIM_STRIDE + ROTATION, q);
}
#[inline]
pub fn write_sim_inv_inertia_world(col: Col<f32>, i: usize, m: Mat3) {
    let o = i * SIM_STRIDE + INV_INERTIA_WORLD;
    put_vec3(col, o, m.cx);
    put_vec3(col, o + 3, m.cy);
    put_vec3(col, o + 6, m.cz);
}
#[inline]
pub fn clear_sim_force_torque(col: Col<f32>, i: usize) {
    let o = i * SIM_STRIDE + FORCE;
    for k in 0..6 {
        col.set(o + k, 0.0);
    }
}
pub mod flags {
    pub const LOCK_LINEAR_X: u32 = 0x0000_0001;
    pub const LOCK_LINEAR_Y: u32 = 0x0000_0002;
    pub const LOCK_LINEAR_Z: u32 = 0x0000_0004;
    pub const LOCK_ANGULAR_X: u32 = 0x0000_0008;
    pub const LOCK_ANGULAR_Y: u32 = 0x0000_0010;
    pub const LOCK_ANGULAR_Z: u32 = 0x0000_0020;
    pub const IS_FAST: u32 = 0x0000_0040;
    pub const HAD_TIME_OF_IMPACT: u32 = 0x0000_0200;
    pub const IS_SPEED_CAPPED: u32 = 0x0000_0100;
    pub const ALLOW_FAST_ROTATION: u32 = 0x0000_0400;
    pub const DYNAMIC: u32 = 0x0000_1000;
    pub const ENABLE_SLEEP: u32 = 0x0000_2000;
}
