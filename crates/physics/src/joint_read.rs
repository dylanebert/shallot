//! Public joint rotations and separation errors from Box3D joint.c and *_joint.c.
use crate::{
    col::Col,
    joint_abi::*,
    math::{atan2, maxf, Mat3, Quat, Transform, Vec3},
};

fn angle(q: Quat) -> f32 {
    2.0 * atan2(q.v.length(), q.s)
}
fn relative(c: Col<f32>, a: Transform, b: Transform) -> Quat {
    let qa = a.q.mul(get_quat(c, 0, J_LOCAL_FRAME_A + 3));
    let mut qb = b.q.mul(get_quat(c, 0, J_LOCAL_FRAME_B + 3));
    if qa.dot(qb) < 0.0 {
        qb = Quat {
            v: qb.v.scale(-1.0),
            s: -qb.s,
        };
    }
    qa.inv_mul(qb)
}

pub fn read(c: Col<f32>, a: Transform, b: Transform, operation: u32) -> f32 {
    match operation {
        0 | 2 => relative(c, a, b).get_twist_angle(),
        1 => relative(c, a, b).get_swing_angle(),
        3 => {
            let ma = Mat3::from_quat(a.q.mul(get_quat(c, 0, J_LOCAL_FRAME_A + 3)));
            let mb = Mat3::from_quat(b.q.mul(get_quat(c, 0, J_LOCAL_FRAME_B + 3)));
            atan2(-mb.cz.dot(ma.cy), mb.cz.dot(ma.cz))
        }
        4 => angular_separation(c, a, b),
        5 => linear_separation(c, a, b),
        _ => unreachable!(),
    }
}

fn angular_separation(c: Col<f32>, a: Transform, b: Transform) -> f32 {
    let g = |n| get(c, 0, n);
    let mut q = a.q.inv_mul(b.q);
    match joint_type(c, 0) {
        TY_DISTANCE | TY_MOTOR | TY_FILTER | TY_WHEEL => 0.0,
        TY_PARALLEL => {
            q.v.z = 0.0;
            angle(q)
        }
        TY_PRISMATIC => angle(q),
        TY_REVOLUTE => {
            if enabled(c, 0, RJ_ENABLE, RJ_ENABLE_LIMIT) {
                let twist = q.get_twist_angle();
                if twist < g(RJ_LOWER_ANGLE) || g(RJ_UPPER_ANGLE) < twist {
                    return angle(q);
                }
            }
            q.v.z = 0.0;
            angle(q)
        }
        TY_SPHERICAL => {
            let mut sum = 0.0;
            if enabled(c, 0, SJ_ENABLE, SJ_ENABLE_CONE_LIMIT) {
                sum += maxf(0.0, q.get_swing_angle() - g(SJ_CONE_ANGLE));
            }
            if enabled(c, 0, SJ_ENABLE, SJ_ENABLE_TWIST_LIMIT) {
                let twist = q.get_twist_angle();
                sum += maxf(0.0, g(SJ_LOWER_TWIST_ANGLE) - twist);
                sum += maxf(0.0, twist - g(SJ_UPPER_TWIST_ANGLE));
            }
            sum
        }
        TY_WELD => {
            if g(WJ_ANGULAR_HERTZ) == 0.0 {
                angle(q)
            } else {
                0.0
            }
        }
        _ => unreachable!(),
    }
}

fn linear_separation(c: Col<f32>, a: Transform, b: Transform) -> f32 {
    let g = |n| get(c, 0, n);
    let dp = b
        .point(get_vec3(c, 0, J_LOCAL_FRAME_B))
        .sub(a.point(get_vec3(c, 0, J_LOCAL_FRAME_A)));
    match joint_type(c, 0) {
        TY_PARALLEL | TY_MOTOR | TY_FILTER => 0.0,
        TY_DISTANCE => {
            let length = dp.length();
            if enabled(c, 0, DJ_ENABLE, DJ_ENABLE_SPRING) {
                if enabled(c, 0, DJ_ENABLE, DJ_ENABLE_LIMIT) {
                    if length < g(DJ_MIN_LENGTH) {
                        return g(DJ_MIN_LENGTH) - length;
                    }
                    if length > g(DJ_MAX_LENGTH) {
                        return length - g(DJ_MAX_LENGTH);
                    }
                }
                0.0
            } else {
                (length - g(DJ_LENGTH)).abs()
            }
        }
        TY_REVOLUTE | TY_SPHERICAL => dp.length(),
        TY_WELD => {
            if g(WJ_LINEAR_HERTZ) == 0.0 {
                dp.length()
            } else {
                0.0
            }
        }
        TY_PRISMATIC => axis_separation(
            a,
            dp,
            enabled(c, 0, PJ_ENABLE, PJ_ENABLE_LIMIT),
            g(PJ_LOWER_TRANSLATION),
            g(PJ_UPPER_TRANSLATION),
        ),
        TY_WHEEL => axis_separation(
            a,
            dp,
            enabled(c, 0, WHJ_ENABLE, WHJ_ENABLE_SUSPENSION_LIMIT),
            g(WHJ_LOWER_SUSPENSION_LIMIT),
            g(WHJ_UPPER_SUSPENSION_LIMIT),
        ),
        _ => unreachable!(),
    }
}
fn axis_separation(a: Transform, dp: Vec3, enabled: bool, lower: f32, upper: f32) -> f32 {
    let axis = a.q.rotate(Vec3::new(1.0, 0.0, 0.0));
    let perpendicular = axis.perp().dot(dp).abs();
    let mut limit = 0.0;
    if enabled {
        let translation = axis.dot(dp);
        if translation < lower {
            limit = lower - translation;
        }
        if upper < translation {
            limit = translation - upper;
        }
    }
    (perpendicular * perpendicular + limit * limit).sqrt()
}

/// # Safety
/// The world and joint must be live, with both attached body records resident.
#[cfg(target_arch = "wasm32")]
#[export_name = "jointRead"]
pub unsafe extern "C" fn run(world: usize, id: usize, operation: u32) -> f32 {
    crate::regions::select(world as u32);
    let c = Col::new(
        crate::joint_record::sim_pointer_in_world(world, id) as *mut f32,
        JOINT_STRIDE,
    );
    let a = crate::draw::pose(world, get(c, 0, J_BODY_ID_A).to_bits() as usize);
    let b = crate::draw::pose(world, get(c, 0, J_BODY_ID_B).to_bits() as usize);
    read(c, a, b, operation)
}
