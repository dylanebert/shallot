use shallot_physics::{col, joint_abi, math};

#[path = "../src/joint_draw.rs"]
#[allow(dead_code)]
mod joint_draw;

use col::Col;
use joint_abi::*;
use math::{Quat, Transform, Vec3};
use std::cell::Cell;

#[test]
fn accessors_resolve_only_the_body_poses_their_native_request_reads() {
    for (kind, force_reads, torque_reads) in [
        (TY_PARALLEL, 0, 0),
        (TY_DISTANCE, 2, 0),
        (TY_FILTER, 0, 0),
        (TY_MOTOR, 0, 0),
        (TY_PRISMATIC, 1, 1),
        (TY_REVOLUTE, 0, 1),
        (TY_SPHERICAL, 0, 2),
        (TY_WELD, 0, 0),
        (TY_WHEEL, 1, 1),
    ] {
        let mut memory = vec![0.0; JOINT_STRIDE];
        let c = unsafe { Col::of(&mut memory) };
        set(c, 0, J_TYPE, f32::from_bits(kind));
        set(c, 0, J_BODY_ID_A, f32::from_bits(11));
        set(c, 0, J_BODY_ID_B, f32::from_bits(12));
        set_quat(c, 0, J_LOCAL_FRAME_A + 3, Quat::IDENTITY);
        set_quat(c, 0, J_LOCAL_FRAME_B + 3, Quat::IDENTITY);
        let reads = Cell::new(0);
        let pose = |id| {
            assert!(id == 11 || id == 12);
            reads.set(reads.get() + 1);
            Transform {
                p: Vec3::ZERO,
                q: Quat::IDENTITY,
            }
        };
        joint_draw::force(c, 2.0, pose);
        assert_eq!(reads.get(), force_reads, "force kind {kind}");
        reads.set(0);
        joint_draw::constraint_torque(c, 2.0, pose);
        assert_eq!(reads.get(), torque_reads, "torque kind {kind}");
    }
}

#[test]
fn force_does_not_recompute_torque_axes() {
    for (kind, qa, qb, px, py) in [
        (
            TY_PARALLEL,
            PLJ_QUAT_A,
            PLJ_QUAT_B,
            PLJ_PERP_AXIS_X,
            PLJ_PERP_AXIS_Y,
        ),
        (
            TY_REVOLUTE,
            RJ_FRAME_A + 3,
            RJ_FRAME_B + 3,
            RJ_PERP_AXIS_X,
            RJ_PERP_AXIS_Y,
        ),
    ] {
        let mut memory = vec![0.0; JOINT_STRIDE];
        let c = unsafe { Col::of(&mut memory) };
        set(c, 0, J_TYPE, f32::from_bits(kind));
        set_quat(c, 0, qa, Quat::IDENTITY);
        set_quat(c, 0, qb, Quat::IDENTITY);
        let sentinel = Vec3::new(7.0, 8.0, 9.0);
        set_vec3(c, 0, px, sentinel);
        set_vec3(c, 0, py, sentinel);
        joint_draw::force(c, 2.0, |_| panic!("force does not read poses"));
        assert_eq!(get_vec3(c, 0, px), sentinel);
        assert_eq!(get_vec3(c, 0, py), sentinel);
        joint_draw::constraint_torque(c, 2.0, |_| Transform {
            p: Vec3::ZERO,
            q: Quat::IDENTITY,
        });
        assert_eq!(get_vec3(c, 0, px), Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(get_vec3(c, 0, py), Vec3::new(0.0, 0.5, 0.0));
    }
}
