use core::mem::{offset_of, size_of};
use shallot_physics::col::Col;
use shallot_physics::joint_abi::*;
use shallot_physics::joint_sim::*;

#[test]
fn joint_sim_and_payloads_have_joint_h_sizes_and_index_offsets() {
    // Measured with sizeof/offsetof against Box3D 47d7f7cc joint.h.
    assert_eq!(size_of::<JointSim>(), 444);
    assert_eq!(offset_of!(JointSim, payload), 184);
    assert_eq!(offset_of!(JointSim, fixed_rotation), 180);
    assert_eq!(offset_of!(JointSim, force_threshold), 172);
    assert_eq!(offset_of!(JointSim, constraint_softness), 160);
    assert_eq!(
        (
            size_of::<DistanceJoint>(),
            offset_of!(DistanceJoint, index_a)
        ),
        (116, 52)
    );
    assert_eq!(
        (size_of::<MotorJoint>(), offset_of!(MotorJoint, index_a)),
        (240, 128)
    );
    assert_eq!(
        (
            size_of::<ParallelJoint>(),
            offset_of!(ParallelJoint, index_a)
        ),
        (96, 76)
    );
    assert_eq!(
        (
            size_of::<PrismaticJoint>(),
            offset_of!(PrismaticJoint, index_a)
        ),
        (232, 64)
    );
    assert_eq!(
        (
            size_of::<RevoluteJoint>(),
            offset_of!(RevoluteJoint, index_a)
        ),
        (200, 64)
    );
    assert_eq!(
        (
            size_of::<SphericalJoint>(),
            offset_of!(SphericalJoint, index_a)
        ),
        (260, 100)
    );
    assert_eq!(
        (size_of::<WeldJoint>(), offset_of!(WeldJoint, index_a)),
        (176, 64)
    );
    assert_eq!(
        (size_of::<WheelJoint>(), offset_of!(WheelJoint, index_a)),
        (212, 92)
    );
}

#[test]
fn boolean_bindings_write_native_bytes_without_changing_numeric_fields() {
    let mut memory = vec![0.0; JOINT_STRIDE * 2];
    let col = unsafe { Col::of(&mut memory) };
    set(col, 1, J_JOINT_ID, f32::from_bits(77));
    for (field, count) in [
        (DJ_ENABLE, 3),
        (RJ_ENABLE, 3),
        (PJ_ENABLE, 3),
        (SJ_ENABLE, 4),
        (WHJ_ENABLE, 5),
    ] {
        for flags in 0..(1 << count) {
            write_flags(col, 1, field, flags);
            assert_eq!(read_flags(col, 1, field), flags);
            for i in 0..count {
                assert_eq!(enabled(col, 1, field, 1 << i), flags & (1 << i) != 0);
            }
            assert_eq!(get(col, 1, J_JOINT_ID).to_bits(), 77);
            assert_eq!(get(col, 0, J_JOINT_ID).to_bits(), 0);
        }
    }
    set(col, 1, J_FIXED_ROTATION, 1.0);
    assert_eq!(get(col, 1, J_FIXED_ROTATION), 1.0);
    set(col, 1, J_FIXED_ROTATION, 0.0);
    assert_eq!(get(col, 1, J_FIXED_ROTATION), 0.0);
}
