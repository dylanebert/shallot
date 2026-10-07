//! Joint reaction values used by the public accessors and debug draw.
use crate::{
    col::Col,
    joint_abi::*,
    math::{Mat3, Quat, Transform, Vec3},
};

#[cfg(target_arch = "wasm32")]
#[export_name = "jointReaction"]
pub unsafe extern "C" fn run(world: usize, id: usize, inv_h: f32, torque: u32) {
    crate::regions::select(world as u32);
    unsafe { run_in_world(world, id, inv_h, torque) }
}

#[cfg(target_arch = "wasm32")]
pub unsafe extern "C" fn run_in_world(world: usize, id: usize, inv_h: f32, torque: u32) {
    let c = Col::new(
        crate::joint_record::sim_pointer_in_world(world, id) as *mut f32,
        JOINT_STRIDE,
    );
    let pose = |body_id| crate::draw::pose(world, body_id);
    let v = if torque == 0 {
        force(c, inv_h, pose)
    } else {
        constraint_torque(c, inv_h, pose)
    };
    let p = crate::query_abi::output_ptr() as *mut f32;
    *p = v.x;
    *p.add(1) = v.y;
    *p.add(2) = v.z;
}

fn axis(q: Quat, relative: Quat, v: Vec3) -> Vec3 {
    q.rotate(v.scale(relative.s).add(relative.v.cross(v)))
        .scale(0.5)
}

pub fn force(c: Col<f32>, inv_h: f32, pose: impl Fn(usize) -> Transform) -> Vec3 {
    let g = |n| get(c, 0, n);
    let v = |n| get_vec3(c, 0, n);
    match joint_type(c, 0) {
        TY_PARALLEL | TY_FILTER => Vec3::ZERO,
        TY_DISTANCE => {
            let a = pose(g(J_BODY_ID_A).to_bits() as usize);
            let b = pose(g(J_BODY_ID_B).to_bits() as usize);
            let pa = a.point(v(J_LOCAL_FRAME_A));
            let pb = b.point(v(J_LOCAL_FRAME_B));
            pb.sub(pa).normalize().scale(
                (((g(DJ_IMPULSE) + g(DJ_LOWER_IMPULSE)) - g(DJ_UPPER_IMPULSE))
                    + g(DJ_MOTOR_IMPULSE))
                    * inv_h,
            )
        }
        TY_MOTOR => v(MJ_LINEAR_VELOCITY_IMPULSE)
            .add(v(MJ_LINEAR_SPRING_IMPULSE))
            .scale(inv_h),
        TY_PRISMATIC => {
            let a = pose(g(J_BODY_ID_A).to_bits() as usize);
            let impulse = Vec3::new(
                g(PJ_PERP_IMPULSE),
                g(PJ_PERP_IMPULSE + 1),
                ((g(PJ_MOTOR_IMPULSE) + g(PJ_LOWER_IMPULSE)) + g(PJ_UPPER_IMPULSE))
                    + g(PJ_SPRING_IMPULSE),
            );
            a.q.rotate(get_quat(c, 0, J_LOCAL_FRAME_A + 3).rotate(impulse.scale(inv_h)))
        }
        TY_REVOLUTE => v(RJ_LINEAR_IMPULSE).scale(inv_h),
        TY_SPHERICAL => v(SJ_LINEAR_IMPULSE).scale(inv_h),
        TY_WELD => v(WJ_LINEAR_IMPULSE).scale(inv_h),
        TY_WHEEL => {
            let a = pose(g(J_BODY_ID_A).to_bits() as usize);
            let impulse = Vec3::new(
                g(WHJ_LINEAR_IMPULSE),
                g(WHJ_LINEAR_IMPULSE + 1),
                (g(WHJ_LOWER_SUSPENSION_LIMIT) + g(WHJ_UPPER_SUSPENSION_IMPULSE))
                    + g(WHJ_SUSPENSION_SPRING_IMPULSE),
            );
            a.q.rotate(get_quat(c, 0, J_LOCAL_FRAME_A + 3).rotate(impulse.scale(inv_h)))
        }
        _ => unreachable!(),
    }
}

pub fn constraint_torque(c: Col<f32>, inv_h: f32, pose: impl Fn(usize) -> Transform) -> Vec3 {
    let g = |n| get(c, 0, n);
    let v = |n| get_vec3(c, 0, n);
    let q = |n| get_quat(c, 0, n);
    let z = Vec3::new(0.0, 0.0, 1.0);
    let x = Vec3::new(1.0, 0.0, 0.0);
    let y = Vec3::new(0.0, 1.0, 0.0);
    match joint_type(c, 0) {
        TY_PARALLEL => {
            let qa = q(PLJ_QUAT_A);
            let rel = qa.inv_mul(q(PLJ_QUAT_B));
            let px = axis(qa, rel, x);
            let py = axis(qa, rel, y);
            set_vec3(c, 0, PLJ_PERP_AXIS_X, px);
            set_vec3(c, 0, PLJ_PERP_AXIS_Y, py);
            px.scale(g(PLJ_PERP_IMPULSE))
                .add(py.scale(g(PLJ_PERP_IMPULSE + 1)))
                .scale(inv_h)
        }
        TY_DISTANCE | TY_FILTER => Vec3::ZERO,
        TY_MOTOR => v(MJ_ANGULAR_VELOCITY_IMPULSE)
            .add(v(MJ_ANGULAR_SPRING_IMPULSE))
            .scale(inv_h),
        TY_PRISMATIC => {
            let a = pose(g(J_BODY_ID_A).to_bits() as usize);
            a.q.rotate(q(J_LOCAL_FRAME_A + 3).rotate(v(PJ_ANGULAR_IMPULSE).scale(inv_h)))
        }
        TY_REVOLUTE => {
            let a = pose(g(J_BODY_ID_A).to_bits() as usize);
            let world_axis = a.q.rotate(q(J_LOCAL_FRAME_A + 3).rotate(z));
            let qa = q(RJ_FRAME_A + 3);
            let rel = qa.inv_mul(q(RJ_FRAME_B + 3));
            let px = axis(qa, rel, x);
            let py = axis(qa, rel, y);
            set_vec3(c, 0, RJ_PERP_AXIS_X, px);
            set_vec3(c, 0, RJ_PERP_AXIS_Y, py);
            let axial = ((g(RJ_SPRING_IMPULSE) + g(RJ_MOTOR_IMPULSE)) + g(RJ_LOWER_IMPULSE))
                - g(RJ_UPPER_IMPULSE);
            px.scale(g(RJ_PERP_IMPULSE))
                .add(py.scale(g(RJ_PERP_IMPULSE + 1)))
                .mul_add(axial, v(RJ_ROTATION_AXIS_Z))
                .mul_add(axial, world_axis)
                .scale(inv_h)
        }
        TY_SPHERICAL => {
            let a = pose(g(J_BODY_ID_A).to_bits() as usize);
            let b = pose(g(J_BODY_ID_B).to_bits() as usize);
            let qa = a.q.mul(q(J_LOCAL_FRAME_A + 3));
            let qb = b.q.mul(q(J_LOCAL_FRAME_B + 3));
            let cone = qa.rotate(z);
            let twist = qb.rotate(z);
            let swing = cone.cross(twist).normalize();
            v(SJ_SPRING_IMPULSE)
                .add(v(SJ_MOTOR_IMPULSE))
                .mul_add(g(SJ_LOWER_TWIST_IMPULSE) - g(SJ_UPPER_TWIST_IMPULSE), twist)
                .mul_add(g(SJ_SWING_IMPULSE), swing)
                .scale(inv_h)
        }
        TY_WELD => v(WJ_ANGULAR_IMPULSE).scale(inv_h),
        TY_WHEEL => {
            let a = pose(g(J_BODY_ID_A).to_bits() as usize);
            Mat3::from_quat(a.q.mul(q(J_LOCAL_FRAME_A + 3)))
                .cz
                .scale(inv_h * g(WHJ_SPIN_IMPULSE))
        }
        _ => unreachable!(),
    }
}

pub fn reaction(c: Col<f32>, a: Transform, b: Transform, inv_h: f32) -> (Vec3, Vec3) {
    let pose = |id| {
        if id == get(c, 0, J_BODY_ID_A).to_bits() as usize {
            a
        } else {
            b
        }
    };
    (force(c, inv_h, pose), constraint_torque(c, inv_h, pose))
}
