//! Joint reaction values used by the public accessors and debug draw.
#[cfg(target_arch = "wasm32")]
#[export_name = "jointReaction"]
pub unsafe extern "C" fn run(world: usize, id: usize, inv_h: f32, torque: u32) {
    crate::regions::select(world as u32);
    let j = crate::joint_record::record(id);
    let c = Col::new(
        crate::joint_record::sim_pointer(id) as *mut f32,
        JOINT_STRIDE,
    );
    let a = crate::draw::pose(j.edges[0].body_id as usize);
    let b = crate::draw::pose(j.edges[1].body_id as usize);
    if torque != 0 {
        let kind = joint_type(c, 0);
        let (qa, qb, px, py) = match kind {
            0 => (PLJ_QUAT_A, PLJ_QUAT_B, PLJ_PERP_AXIS_X, PLJ_PERP_AXIS_Y),
            5 => (
                RJ_FRAME_A + 3,
                RJ_FRAME_B + 3,
                RJ_PERP_AXIS_X,
                RJ_PERP_AXIS_Y,
            ),
            _ => (0, 0, 0, 0),
        };
        if kind == 0 || kind == 5 {
            let q = get_quat(c, 0, qa);
            let rel = q.inv_mul(get_quat(c, 0, qb));
            set_vec3(c, 0, px, axis(q, rel, Vec3::new(1.0, 0.0, 0.0)));
            set_vec3(c, 0, py, axis(q, rel, Vec3::new(0.0, 1.0, 0.0)));
        }
    }
    let (force, t) = reaction(c, a, b, inv_h);
    let v = if torque == 0 { force } else { t };
    let p = crate::query_abi::output_ptr() as *mut f32;
    *p = v.x;
    *p.add(1) = v.y;
    *p.add(2) = v.z;
}
use crate::{
    col::Col,
    joint_abi::*,
    math::{Mat3, Quat, Transform, Vec3},
};
fn axis(q: Quat, relative: Quat, v: Vec3) -> Vec3 {
    q.rotate(v.scale(relative.s).add(relative.v.cross(v)))
        .scale(0.5)
}
pub fn reaction(c: Col<f32>, a: Transform, b: Transform, inv_h: f32) -> (Vec3, Vec3) {
    let g = |n| c.get(n);
    let v = |n| get_vec3(c, 0, n);
    let q = |n| get_quat(c, 0, n);
    let z = Vec3::new(0.0, 0.0, 1.0);
    let x = Vec3::new(1.0, 0.0, 0.0);
    let y = Vec3::new(0.0, 1.0, 0.0);
    let local = q(J_LOCAL_FRAME_A + 3);
    let rotate = |v: Vec3| a.q.rotate(local.rotate(v));
    let (force, torque) = match joint_type(c, 0) {
        0 => {
            let qa = q(PLJ_QUAT_A);
            let rel = qa.inv_mul(q(PLJ_QUAT_B));
            (
                Vec3::ZERO,
                axis(qa, rel, x)
                    .scale(g(PLJ_PERP_IMPULSE))
                    .add(axis(qa, rel, y).scale(g(PLJ_PERP_IMPULSE + 1))),
            )
        }
        1 => {
            let pa = a.point(v(J_LOCAL_FRAME_A));
            let pb = b.point(v(J_LOCAL_FRAME_B));
            return (
                pb.sub(pa).normalize().scale(
                    (((g(DJ_IMPULSE) + g(DJ_LOWER_IMPULSE)) - g(DJ_UPPER_IMPULSE))
                        + g(DJ_MOTOR_IMPULSE))
                        * inv_h,
                ),
                Vec3::ZERO,
            );
        }
        2 => (Vec3::ZERO, Vec3::ZERO),
        3 => (
            v(MJ_LINEAR_VELOCITY_IMPULSE).add(v(MJ_LINEAR_SPRING_IMPULSE)),
            v(MJ_ANGULAR_VELOCITY_IMPULSE).add(v(MJ_ANGULAR_SPRING_IMPULSE)),
        ),
        4 => {
            return (
                rotate(
                    Vec3::new(
                        g(PJ_PERP_IMPULSE),
                        g(PJ_PERP_IMPULSE + 1),
                        ((g(PJ_MOTOR_IMPULSE) + g(PJ_LOWER_IMPULSE)) + g(PJ_UPPER_IMPULSE))
                            + g(PJ_SPRING_IMPULSE),
                    )
                    .scale(inv_h),
                ),
                rotate(v(PJ_ANGULAR_IMPULSE).scale(inv_h)),
            )
        }
        5 => {
            let qa = q(RJ_FRAME_A + 3);
            let rel = qa.inv_mul(q(RJ_FRAME_B + 3));
            let axial = ((g(RJ_SPRING_IMPULSE) + g(RJ_MOTOR_IMPULSE)) + g(RJ_LOWER_IMPULSE))
                - g(RJ_UPPER_IMPULSE);
            let impulse = axis(qa, rel, x)
                .scale(g(RJ_PERP_IMPULSE))
                .add(axis(qa, rel, y).scale(g(RJ_PERP_IMPULSE + 1)))
                .mul_add(axial, v(RJ_ROTATION_AXIS_Z))
                .mul_add(axial, rotate(z));
            (v(RJ_LINEAR_IMPULSE), impulse)
        }
        6 => {
            let qa = a.q.mul(local);
            let qb = b.q.mul(q(J_LOCAL_FRAME_B + 3));
            let cone = qa.rotate(z);
            let twist = qb.rotate(z);
            let swing = cone.cross(twist).normalize();
            (
                v(SJ_LINEAR_IMPULSE),
                v(SJ_SPRING_IMPULSE)
                    .add(v(SJ_MOTOR_IMPULSE))
                    .mul_add(g(SJ_LOWER_TWIST_IMPULSE) - g(SJ_UPPER_TWIST_IMPULSE), twist)
                    .mul_add(g(SJ_SWING_IMPULSE), swing),
            )
        }
        7 => (v(WJ_LINEAR_IMPULSE), v(WJ_ANGULAR_IMPULSE)),
        8 => {
            return (
                rotate(
                    Vec3::new(
                        g(WHJ_LINEAR_IMPULSE),
                        g(WHJ_LINEAR_IMPULSE + 1),
                        (g(WHJ_LOWER_SUSPENSION_LIMIT) + g(WHJ_UPPER_SUSPENSION_IMPULSE))
                            + g(WHJ_SUSPENSION_SPRING_IMPULSE),
                    )
                    .scale(inv_h),
                ),
                Mat3::from_quat(a.q.mul(local))
                    .cz
                    .scale(inv_h * g(WHJ_SPIN_IMPULSE)),
            )
        }
        _ => unreachable!(),
    };
    (force.scale(inv_h), torque.scale(inv_h))
}
