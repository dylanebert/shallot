//! Thin numeric and boolean bindings over joint.h's typed simulation records.
use crate::col::Col;
pub use crate::joint_layout::*;
use crate::joint_sim::*;
use crate::math::{Mat3, Quat, Transform, Vec3};
use core::mem::offset_of;

pub const TY_PARALLEL: u32 = 0;
pub const TY_DISTANCE: u32 = 1;
pub const TY_FILTER: u32 = 2;
pub const TY_MOTOR: u32 = 3;
pub const TY_PRISMATIC: u32 = 4;
pub const TY_REVOLUTE: u32 = 5;
pub const TY_SPHERICAL: u32 = 6;
pub const TY_WELD: u32 = 7;
pub const TY_WHEEL: u32 = 8;
pub const NULL_INDEX: u32 = u32::MAX;

#[inline]
pub fn joint_type(col: Col<f32>, slot: usize) -> u32 {
    get(col, slot, J_TYPE).to_bits()
}

#[inline]
fn index_offset(kind: u32) -> usize {
    let offset = match kind {
        TY_DISTANCE => offset_of!(DistanceJoint, index_a),
        TY_MOTOR => offset_of!(MotorJoint, index_a),
        TY_PARALLEL => offset_of!(ParallelJoint, index_a),
        TY_PRISMATIC => offset_of!(PrismaticJoint, index_a),
        TY_REVOLUTE => offset_of!(RevoluteJoint, index_a),
        TY_SPHERICAL => offset_of!(SphericalJoint, index_a),
        TY_WELD => offset_of!(WeldJoint, index_a),
        TY_WHEEL => offset_of!(WheelJoint, index_a),
        _ => return J_PAYLOAD,
    };
    J_PAYLOAD + offset / 4
}

#[inline]
pub fn set_indices(col: Col<f32>, slot: usize, a: u32, b: u32) {
    if joint_type(col, slot) == TY_FILTER {
        return;
    }
    let field = index_offset(joint_type(col, slot));
    set(col, slot, field, f32::from_bits(a));
    set(col, slot, field + 1, f32::from_bits(b));
}

pub struct JointBase {
    pub sim_index_a: u32,
    pub sim_index_b: u32,
    pub inv_mass_a: f32,
    pub inv_mass_b: f32,
    pub inv_ia: Mat3,
    pub inv_ib: Mat3,
    pub local_frame_a: Transform,
    pub local_frame_b: Transform,
}
pub struct JointPose {
    pub qa: Quat,
    pub local_center_a: Vec3,
    pub center_a: Vec3,
    pub qb: Quat,
    pub local_center_b: Vec3,
    pub center_b: Vec3,
}

#[inline]
pub fn read_base<const KIND: u32>(col: Col<f32>, slot: usize) -> JointBase {
    let indices = index_offset(KIND);
    JointBase {
        sim_index_a: get(col, slot, indices).to_bits(),
        sim_index_b: get(col, slot, indices + 1).to_bits(),
        inv_mass_a: get(col, slot, J_INV_MASS_A),
        inv_mass_b: get(col, slot, J_INV_MASS_B),
        inv_ia: get_mat3(col, slot, J_INV_IA),
        inv_ib: get_mat3(col, slot, J_INV_IB),
        local_frame_a: get_transform(col, slot, J_LOCAL_FRAME_A),
        local_frame_b: get_transform(col, slot, J_LOCAL_FRAME_B),
    }
}

#[inline]
fn byte_ptr(col: Col<f32>, slot: usize, field: usize) -> *mut u8 {
    debug_assert!(field & BOOL_FIELD != 0);
    let byte = field & !BOOL_FIELD;
    debug_assert!(slot < col.len() / JOINT_STRIDE && byte < JOINT_STRIDE * 4);
    unsafe { col.ptr().cast::<u8>().add(slot * JOINT_STRIDE * 4 + byte) }
}

#[inline]
pub fn enabled(col: Col<f32>, slot: usize, field: usize, mask: u32) -> bool {
    unsafe { *byte_ptr(col, slot, field).add(mask.trailing_zeros() as usize) != 0 }
}

pub fn read_flags(col: Col<f32>, slot: usize, field: usize) -> u32 {
    let count = flag_count(field);
    let ptr = byte_ptr(col, slot, field);
    let mut flags = 0;
    for i in 0..count {
        flags |= u32::from(unsafe { *ptr.add(i) != 0 }) << i;
    }
    flags
}
pub fn write_flags(col: Col<f32>, slot: usize, field: usize, flags: u32) {
    let count = flag_count(field);
    let ptr = byte_ptr(col, slot, field);
    for i in 0..count {
        unsafe { *ptr.add(i) = u8::from(flags & (1 << i) != 0) };
    }
}
fn flag_count(field: usize) -> usize {
    match field {
        DJ_ENABLE | RJ_ENABLE | PJ_ENABLE => 3,
        SJ_ENABLE => 4,
        WHJ_ENABLE => 5,
        _ => panic!("not a joint enable field"),
    }
}

#[inline]
pub fn get(col: Col<f32>, slot: usize, field: usize) -> f32 {
    if field == J_FIXED_ROTATION {
        return u8::from(unsafe { *byte_ptr(col, slot, field) != 0 }) as f32;
    }
    col.get(slot * JOINT_STRIDE + field)
}
#[inline]
pub fn set(col: Col<f32>, slot: usize, field: usize, value: f32) {
    if field == J_FIXED_ROTATION {
        unsafe { *byte_ptr(col, slot, field) = u8::from(value != 0.0) };
    } else {
        col.set(slot * JOINT_STRIDE + field, value);
    }
}
#[inline]
pub fn get_vec3(col: Col<f32>, slot: usize, field: usize) -> Vec3 {
    Vec3::new(
        get(col, slot, field),
        get(col, slot, field + 1),
        get(col, slot, field + 2),
    )
}
#[inline]
pub fn set_vec3(col: Col<f32>, slot: usize, field: usize, value: Vec3) {
    set(col, slot, field, value.x);
    set(col, slot, field + 1, value.y);
    set(col, slot, field + 2, value.z);
}
#[inline]
pub fn get_quat(col: Col<f32>, slot: usize, field: usize) -> Quat {
    Quat {
        v: get_vec3(col, slot, field),
        s: get(col, slot, field + 3),
    }
}
#[inline]
pub fn set_quat(col: Col<f32>, slot: usize, field: usize, value: Quat) {
    set_vec3(col, slot, field, value.v);
    set(col, slot, field + 3, value.s);
}
#[inline]
pub fn get_mat3(col: Col<f32>, slot: usize, field: usize) -> Mat3 {
    Mat3 {
        cx: get_vec3(col, slot, field),
        cy: get_vec3(col, slot, field + 3),
        cz: get_vec3(col, slot, field + 6),
    }
}
#[inline]
pub fn set_mat3(col: Col<f32>, slot: usize, field: usize, value: Mat3) {
    set_vec3(col, slot, field, value.cx);
    set_vec3(col, slot, field + 3, value.cy);
    set_vec3(col, slot, field + 6, value.cz);
}
#[inline]
pub fn get_transform(col: Col<f32>, slot: usize, field: usize) -> Transform {
    Transform {
        p: get_vec3(col, slot, field),
        q: get_quat(col, slot, field + 3),
    }
}
#[inline]
pub fn set_transform(col: Col<f32>, slot: usize, field: usize, value: Transform) {
    set_vec3(col, slot, field, value.p);
    set_quat(col, slot, field + 3, value.q);
}
