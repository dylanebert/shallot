//! Box3D mover_joint.c and pogo_joint.c (Erin Catto, MIT).
use super::*;
use crate::joint_abi::*;

pub(super) fn prepare_mover(joints: Col<f32>, slot: usize, pose: JointPose, warm: bool) {
    let base = read_base::<TY_MOVER>(joints, slot);
    set_mat3(joints, slot, J_INV_IA, Mat3::ZERO);
    set_mat3(joints, slot, J_INV_IB, Mat3::ZERO);
    let matrix = Mat3::from_quat(pose.qb.mul(base.local_frame_b.q));
    set_vec3(joints, slot, MVJ_AXIS_X, matrix.cx);
    set_vec3(joints, slot, MVJ_AXIS_Z, matrix.cz);
    let k = base.inv_mass_a + base.inv_mass_b;
    set(
        joints,
        slot,
        MVJ_LINEAR_MASS,
        if k > 0.0 { 1.0 / k } else { 0.0 },
    );
    if !warm {
        set(joints, slot, MVJ_IMPULSE, 0.0);
        set(joints, slot, MVJ_IMPULSE + 1, 0.0);
    }
}

pub(super) fn warm_mover(joints: Col<f32>, slot: usize, states: Col<f32>, flags: Col<u32>) {
    let base = read_base::<TY_MOVER>(joints, slot);
    let a = read_end(states, flags, base.sim_index_a);
    let b = read_end(states, flags, base.sim_index_b);
    let p = blend2(
        get(joints, slot, MVJ_IMPULSE),
        get_vec3(joints, slot, MVJ_AXIS_X),
        get(joints, slot, MVJ_IMPULSE + 1),
        get_vec3(joints, slot, MVJ_AXIS_Z),
    );
    if a.dynamic {
        write_velocity(
            states,
            base.sim_index_a,
            a.state.linear_velocity.mul_sub(base.inv_mass_a, p),
            a.state.angular_velocity,
        );
    }
    if b.dynamic {
        write_velocity(
            states,
            base.sim_index_b,
            b.state.linear_velocity.mul_add(base.inv_mass_b, p),
            b.state.angular_velocity,
        );
    }
}

pub(super) fn solve_mover(
    joints: Col<f32>,
    slot: usize,
    states: Col<f32>,
    flags: Col<u32>,
    h: f32,
) {
    let base = read_base::<TY_MOVER>(joints, slot);
    let a = read_end(states, flags, base.sim_index_a);
    let b = read_end(states, flags, base.sim_index_b);
    let axis_x = get_vec3(joints, slot, MVJ_AXIS_X);
    let axis_z = get_vec3(joints, slot, MVJ_AXIS_Z);
    let vr = b
        .state
        .linear_velocity
        .sub(a.state.linear_velocity)
        .sub(get_vec3(joints, slot, MVJ_LINEAR_VELOCITY));
    let mass = get(joints, slot, MVJ_LINEAR_MASS);
    let old_x = get(joints, slot, MVJ_IMPULSE);
    let old_z = get(joints, slot, MVJ_IMPULSE + 1);
    let mut x = old_x + -mass * vr.dot(axis_x);
    let mut z = old_z + -mass * vr.dot(axis_z);
    let max = h * get(joints, slot, MVJ_MAX_FORCE);
    let length_squared = x * x + z * z;
    if length_squared > max * max {
        let scale = max / length_squared.sqrt();
        x *= scale;
        z *= scale;
    }
    set(joints, slot, MVJ_IMPULSE, x);
    set(joints, slot, MVJ_IMPULSE + 1, z);
    let p = blend2(x - old_x, axis_x, z - old_z, axis_z);
    if a.dynamic {
        write_velocity(
            states,
            base.sim_index_a,
            a.state.linear_velocity.mul_sub(base.inv_mass_a, p),
            a.state.angular_velocity,
        );
    }
    if b.dynamic {
        write_velocity(
            states,
            base.sim_index_b,
            b.state.linear_velocity.mul_add(base.inv_mass_b, p),
            b.state.angular_velocity,
        );
    }
}

pub(super) fn prepare_pogo(joints: Col<f32>, slot: usize, pose: JointPose, warm: bool) {
    let base = read_base::<TY_POGO>(joints, slot);
    let frame_a = Transform {
        q: pose.qa.mul(base.local_frame_a.q),
        p: pose
            .qa
            .rotate(base.local_frame_a.p.sub(pose.local_center_a)),
    };
    let frame_b = Transform {
        q: pose.qb.mul(base.local_frame_b.q),
        p: pose
            .qb
            .rotate(base.local_frame_b.p.sub(pose.local_center_b)),
    };
    set_transform(joints, slot, PGJ_FRAME_A, frame_a);
    set_transform(joints, slot, PGJ_FRAME_B, frame_b);
    set_vec3(
        joints,
        slot,
        PGJ_DELTA_CENTER,
        pose.center_b.sub(pose.center_a),
    );
    let normal = get_vec3(joints, slot, PGJ_NORMAL);
    let rn_a = frame_a.p.cross(normal);
    let rn_b = frame_b.p.cross(normal);
    let k = base.inv_mass_a
        + base.inv_mass_b
        + rn_a.dot(base.inv_ia.mul_v(rn_a))
        + rn_b.dot(base.inv_ib.mul_v(rn_b));
    set(
        joints,
        slot,
        PGJ_LINEAR_MASS,
        if k > 0.0 { 1.0 / k } else { 0.0 },
    );
    if !warm {
        set(joints, slot, PGJ_VELOCITY, 0.0);
        set(joints, slot, PGJ_IMPULSE, 0.0);
    }
}

pub(super) fn warm_pogo(joints: Col<f32>, slot: usize, states: Col<f32>, flags: Col<u32>) {
    let base = read_base::<TY_POGO>(joints, slot);
    if get(joints, slot, PGJ_HERTZ) == 0.0 {
        set(joints, slot, PGJ_IMPULSE, 0.0);
        return;
    }
    let a = read_end(states, flags, base.sim_index_a);
    let b = read_end(states, flags, base.sim_index_b);
    let ra = a
        .state
        .delta_rotation
        .rotate(get_transform(joints, slot, PGJ_FRAME_A).p);
    let rb = b
        .state
        .delta_rotation
        .rotate(get_transform(joints, slot, PGJ_FRAME_B).p);
    let p = get_vec3(joints, slot, PGJ_NORMAL).scale(get(joints, slot, PGJ_IMPULSE));
    if a.dynamic {
        write_velocity(
            states,
            base.sim_index_a,
            a.state.linear_velocity.mul_sub(base.inv_mass_a, p),
            a.state.angular_velocity.sub(base.inv_ia.mul_v(ra.cross(p))),
        );
    }
    if b.dynamic {
        write_velocity(
            states,
            base.sim_index_b,
            b.state.linear_velocity.mul_add(base.inv_mass_b, p),
            b.state.angular_velocity.add(base.inv_ib.mul_v(rb.cross(p))),
        );
    }
}

pub(super) fn solve_pogo(
    joints: Col<f32>,
    slot: usize,
    states: Col<f32>,
    flags: Col<u32>,
    use_bias: bool,
    h: f32,
) {
    let base = read_base::<TY_POGO>(joints, slot);
    if get(joints, slot, PGJ_HERTZ) == 0.0 {
        set(joints, slot, PGJ_IMPULSE, 0.0);
        set(joints, slot, PGJ_VELOCITY, 0.0);
        return;
    }
    let a = read_end(states, flags, base.sim_index_a);
    let b = read_end(states, flags, base.sim_index_b);
    let frame_a = get_transform(joints, slot, PGJ_FRAME_A);
    let frame_b = get_transform(joints, slot, PGJ_FRAME_B);
    let ra = a.state.delta_rotation.rotate(frame_a.p);
    let rb = b.state.delta_rotation.rotate(frame_b.p);
    let mut bias = 0.0;
    if use_bias {
        let d = b
            .state
            .delta_position
            .sub(a.state.delta_position)
            .add(rb.sub(ra))
            .add(get_vec3(joints, slot, PGJ_DELTA_CENTER));
        let axis = frame_b.q.rotate(Vec3::new(0.0, 1.0, 0.0));
        let c = axis.dot(d) - get(joints, slot, PGJ_REST_LENGTH);
        let velocity = crate::math::spring_damper(
            get(joints, slot, PGJ_HERTZ),
            get(joints, slot, PGJ_DAMPING_RATIO),
            c,
            get(joints, slot, PGJ_VELOCITY),
            h,
        );
        set(joints, slot, PGJ_VELOCITY, velocity);
        bias = -velocity;
    }
    let normal = get_vec3(joints, slot, PGJ_NORMAL);
    let vr = b
        .state
        .linear_velocity
        .add(b.state.angular_velocity.cross(rb))
        .sub(
            a.state
                .linear_velocity
                .add(a.state.angular_velocity.cross(ra)),
        );
    let cdot = normal.dot(vr);
    let tension = h * get(joints, slot, PGJ_MAX_TENSION_FORCE);
    let compression = h * get(joints, slot, PGJ_MAX_COMPRESSION_FORCE);
    let old = get(joints, slot, PGJ_IMPULSE);
    let impulse = -get(joints, slot, PGJ_LINEAR_MASS) * (cdot + bias);
    let new = clampf(old + impulse, -tension, compression);
    set(joints, slot, PGJ_IMPULSE, new);
    let p = normal.scale(new - old);
    if a.dynamic {
        write_velocity(
            states,
            base.sim_index_a,
            a.state.linear_velocity.mul_sub(base.inv_mass_a, p),
            a.state.angular_velocity.sub(base.inv_ia.mul_v(ra.cross(p))),
        );
    }
    if b.dynamic {
        write_velocity(
            states,
            base.sim_index_b,
            b.state.linear_velocity.mul_add(base.inv_mass_b, p),
            b.state.angular_velocity.add(base.inv_ib.mul_v(rb.cross(p))),
        );
    }
}
