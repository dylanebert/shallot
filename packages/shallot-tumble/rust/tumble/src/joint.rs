//! Joint constraint solve, ported op-for-op from box3d's `joint.c` dispatch + the per-type files
//! (Erin Catto, MIT) via the tumble.js TS port (`src/joint.ts`, `src/distanceJoint.ts`). Rust `f32`
//! is native IEEE-754 with no FMA contraction, so each TS `f32(...)`-wrapped op maps to one Rust op in
//! the same operand order (see `math.rs`); `b3MinFloat`/`b3MaxFloat`/`b3ClampFloat` are the explicit
//! ternaries, never `f32::min`/`max`.
//!
//! The template stage ports **distance** end-to-end (`joint_abi.rs` header + payload); 3c copies the
//! shape (a `prepare`/`warm_start`/`solve` per type over the header + its payload). The base
//! `prepare`/`warm_start`/`solve` dispatch on `J_TYPE` exactly as box3d's `b3PrepareJoint` etc do.

use crate::body::{flags::DYNAMIC, read_state, State, STATE_STRIDE};
use crate::col::Col;
use crate::contact::Softness;
use crate::joint_abi::{
    get, get_mat3, get_quat, get_transform, get_vec3, joint_type, read_base, read_pose, set,
    set_mat3, set_quat, set_transform, set_vec3, JointBase, DJ_ANCHOR_A, DJ_ANCHOR_B, DJ_AXIAL_MASS,
    DJ_DAMPING_RATIO,
    DJ_DELTA_CENTER, DJ_DIST_SOFTNESS, DJ_ENABLE, DJ_ENABLE_LIMIT, DJ_ENABLE_MOTOR, DJ_ENABLE_SPRING,
    DJ_HERTZ, DJ_IMPULSE, DJ_LENGTH, DJ_LOWER_IMPULSE, DJ_LOWER_SPRING_FORCE, DJ_MAX_LENGTH,
    DJ_MAX_MOTOR_FORCE, DJ_MIN_LENGTH, DJ_MOTOR_IMPULSE, DJ_MOTOR_SPEED, DJ_UPPER_IMPULSE,
    DJ_UPPER_SPRING_FORCE, J_CONSTRAINT_DAMPING, J_CONSTRAINT_HERTZ, J_CONSTRAINT_SOFTNESS,
    MJ_ANGULAR_DAMPING_RATIO, MJ_ANGULAR_HERTZ, MJ_ANGULAR_MASS, MJ_ANGULAR_SPRING,
    MJ_ANGULAR_SPRING_IMPULSE, MJ_ANGULAR_VELOCITY, MJ_ANGULAR_VELOCITY_IMPULSE, MJ_DELTA_CENTER,
    MJ_FRAME_A, MJ_FRAME_B, MJ_LINEAR_DAMPING_RATIO, MJ_LINEAR_HERTZ, MJ_LINEAR_SPRING,
    MJ_LINEAR_SPRING_IMPULSE, MJ_LINEAR_VELOCITY, MJ_LINEAR_VELOCITY_IMPULSE, MJ_MAX_SPRING_FORCE,
    MJ_MAX_SPRING_TORQUE, MJ_MAX_VELOCITY_FORCE, MJ_MAX_VELOCITY_TORQUE,
    NULL_INDEX, PJ_ANGULAR_IMPULSE, PJ_DAMPING_RATIO, PJ_DELTA_CENTER, PJ_ENABLE,
    PJ_ENABLE_LIMIT, PJ_ENABLE_MOTOR, PJ_ENABLE_SPRING, PJ_FIXED_ROTATION, PJ_FRAME_A, PJ_FRAME_B,
    PJ_HERTZ, PJ_JOINT_AXIS, PJ_LOWER_IMPULSE, PJ_LOWER_TRANSLATION, PJ_MAX_MOTOR_FORCE,
    PJ_MOTOR_IMPULSE, PJ_MOTOR_SPEED, PJ_PERP_AXIS_Y, PJ_PERP_AXIS_Z, PJ_PERP_IMPULSE,
    PJ_ROTATION_MASS, PJ_SPRING_IMPULSE, PJ_SPRING_SOFTNESS, PJ_TARGET_TRANSLATION,
    PJ_UPPER_IMPULSE, PJ_UPPER_TRANSLATION, PLJ_DAMPING_RATIO, PLJ_FIXED_ROTATION, PLJ_HERTZ,
    PLJ_MAX_TORQUE, PLJ_PERP_AXIS_X, PLJ_PERP_AXIS_Y, PLJ_PERP_IMPULSE, PLJ_QUAT_A, PLJ_QUAT_B,
    PLJ_SOFTNESS, RJ_AXIAL_MASS, RJ_DAMPING_RATIO, RJ_DELTA_CENTER, RJ_ENABLE, RJ_ENABLE_LIMIT,
    RJ_ENABLE_MOTOR, RJ_ENABLE_SPRING, RJ_FIXED_ROTATION, RJ_FRAME_A, RJ_FRAME_B, RJ_HERTZ,
    RJ_LINEAR_IMPULSE, RJ_LOWER_ANGLE, RJ_LOWER_IMPULSE, RJ_MAX_MOTOR_TORQUE, RJ_MOTOR_IMPULSE,
    RJ_MOTOR_SPEED, RJ_PERP_AXIS_X, RJ_PERP_AXIS_Y, RJ_PERP_IMPULSE, RJ_ROTATION_AXIS_Z,
    RJ_SPRING_IMPULSE, RJ_SPRING_SOFTNESS, RJ_TARGET_ANGLE, RJ_UPPER_ANGLE, RJ_UPPER_IMPULSE,
    SJ_CONE_ANGLE, SJ_DAMPING_RATIO, SJ_DELTA_CENTER, SJ_ENABLE, SJ_ENABLE_CONE_LIMIT,
    SJ_ENABLE_MOTOR, SJ_ENABLE_SPRING, SJ_ENABLE_TWIST_LIMIT, SJ_FIXED_ROTATION, SJ_FRAME_A,
    SJ_FRAME_B, SJ_HERTZ, SJ_LINEAR_IMPULSE, SJ_LOWER_TWIST_ANGLE, SJ_LOWER_TWIST_IMPULSE,
    SJ_MAX_MOTOR_TORQUE, SJ_MOTOR_IMPULSE, SJ_MOTOR_VELOCITY, SJ_ROTATION_MASS, SJ_SPRING_IMPULSE,
    SJ_SPRING_SOFTNESS, SJ_SWING_AXIS, SJ_SWING_IMPULSE, SJ_SWING_MASS, SJ_TARGET_ROTATION,
    SJ_TWIST_JACOBIAN, SJ_TWIST_MASS, SJ_UPPER_TWIST_ANGLE, SJ_UPPER_TWIST_IMPULSE, TY_DISTANCE,
    TY_MOTOR, TY_PARALLEL, TY_PRISMATIC, TY_REVOLUTE, TY_SPHERICAL, TY_WELD, TY_WHEEL,
    WHJ_ANGULAR_IMPULSE,
    WHJ_DELTA_CENTER, WHJ_ENABLE, WHJ_ENABLE_SPIN_MOTOR, WHJ_ENABLE_STEERING,
    WHJ_ENABLE_STEERING_LIMIT, WHJ_ENABLE_SUSPENSION_LIMIT, WHJ_ENABLE_SUSPENSION_SPRING,
    WHJ_FIXED_ROTATION, WHJ_FRAME_A, WHJ_FRAME_B, WHJ_LINEAR_IMPULSE, WHJ_LOWER_STEERING_IMPULSE,
    WHJ_LOWER_STEERING_LIMIT, WHJ_LOWER_SUSPENSION_IMPULSE, WHJ_LOWER_SUSPENSION_LIMIT,
    WHJ_MAX_SPIN_TORQUE, WHJ_MAX_STEERING_TORQUE, WHJ_SPIN_IMPULSE, WHJ_SPIN_MASS, WHJ_SPIN_SPEED,
    WHJ_STEERING_DAMPING_RATIO, WHJ_STEERING_HERTZ, WHJ_STEERING_MASS, WHJ_STEERING_SOFTNESS,
    WHJ_STEERING_SPRING_IMPULSE, WHJ_SUSPENSION_DAMPING_RATIO, WHJ_SUSPENSION_HERTZ,
    WHJ_SUSPENSION_MASS, WHJ_SUSPENSION_SOFTNESS, WHJ_SUSPENSION_SPRING_IMPULSE,
    WHJ_TARGET_STEERING_ANGLE, WHJ_UPPER_STEERING_IMPULSE, WHJ_UPPER_STEERING_LIMIT,
    WHJ_UPPER_SUSPENSION_IMPULSE, WHJ_UPPER_SUSPENSION_LIMIT, WJ_ANGULAR_DAMPING_RATIO,
    WJ_ANGULAR_HERTZ, WJ_ANGULAR_IMPULSE, WJ_ANGULAR_MASS, WJ_ANGULAR_SPRING, WJ_DELTA_CENTER,
    WJ_FIXED_ROTATION, WJ_FRAME_A, WJ_FRAME_B, WJ_LINEAR_DAMPING_RATIO, WJ_LINEAR_HERTZ,
    WJ_LINEAR_IMPULSE, WJ_LINEAR_SPRING,
};
use crate::math::{
    atan2, blend2, blend3, clampf, maxf, minf, Mat2, Mat3, Quat, Transform, Vec2, Vec3, FLT_MIN, PI,
};

/// b3MakeSoft: soft-constraint coefficients from a target frequency, damping, and substep dt. Mirrors
/// `src/softness.ts`; a zero frequency short-circuits to a rigid (zero-bias) response.
fn make_soft(hertz: f32, zeta: f32, h: f32) -> Softness {
    if hertz == 0.0 {
        return Softness {
            bias_rate: 0.0,
            mass_scale: 0.0,
            impulse_scale: 0.0,
        };
    }
    let omega = 2.0 * PI * hertz;
    let a1 = 2.0 * zeta + h * omega;
    let a2 = h * omega * a1;
    let a3 = 1.0 / (1.0 + a2);
    Softness {
        bias_rate: omega / a1,
        mass_scale: a2 * a3,
        impulse_scale: a3,
    }
}

/// The velocity/delta state one joint end reads. A null (static / sleeping) index reads the identity
/// body state (box3d's `b3_identityBodyState`): zero velocity/delta, identity delta-rotation, no
/// `DYNAMIC` flag — so `apply_p` no-ops it and the writeback skips it.
struct End {
    state: State,
    dynamic: bool,
}

#[inline]
fn read_end(state_col: Col<f32>, flags_col: Col<u32>, index: u32) -> End {
    if index == NULL_INDEX {
        End {
            state: State {
                linear_velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
                delta_position: Vec3::ZERO,
                delta_rotation: Quat::IDENTITY,
            },
            dynamic: false,
        }
    } else {
        let i = index as usize;
        End {
            state: read_state(state_col, i),
            dynamic: flags_col.get(i) & DYNAMIC != 0,
        }
    }
}

/// Write a dynamic body's solved linear + angular velocity back to the resident state column (slots
/// 0..5; delta position/rotation are untouched by the solve).
#[inline]
fn write_velocity(state_col: Col<f32>, index: u32, v: Vec3, w: Vec3) {
    let o = index as usize * STATE_STRIDE;
    state_col.set(o, v.x);
    state_col.set(o + 1, v.y);
    state_col.set(o + 2, v.z);
    state_col.set(o + 3, w.x);
    state_col.set(o + 4, w.y);
    state_col.set(o + 5, w.z);
}

// --- base dispatch ----------------------------------------------------------------------------

/// b3PrepareJoint: clamp the base constraint hertz, compute `constraintSoftness`, dispatch to the type.
pub fn prepare(joints: Col<f32>, slot: usize, h: f32, inv_h: f32, enable_warm_starting: bool) {
    let hertz = minf(get(joints, slot, J_CONSTRAINT_HERTZ), 0.25 * inv_h);
    let soft = make_soft(hertz, get(joints, slot, J_CONSTRAINT_DAMPING), h);
    set(joints, slot, J_CONSTRAINT_SOFTNESS, soft.bias_rate);
    set(joints, slot, J_CONSTRAINT_SOFTNESS + 1, soft.mass_scale);
    set(joints, slot, J_CONSTRAINT_SOFTNESS + 2, soft.impulse_scale);

    match joint_type(joints, slot) {
        TY_DISTANCE => prepare_distance(joints, slot, h, enable_warm_starting),
        TY_WELD => prepare_weld(joints, slot, h, enable_warm_starting),
        TY_REVOLUTE => prepare_revolute(joints, slot, h, enable_warm_starting),
        TY_SPHERICAL => prepare_spherical(joints, slot, h, enable_warm_starting),
        TY_PRISMATIC => prepare_prismatic(joints, slot, h, enable_warm_starting),
        TY_WHEEL => prepare_wheel(joints, slot, h, enable_warm_starting),
        TY_MOTOR => prepare_motor(joints, slot, h, enable_warm_starting),
        TY_PARALLEL => prepare_parallel(joints, slot, h, enable_warm_starting),
        _ => {} // TY_FILTER is a no-op (collision filter, no solve); no other awake type remains.
    }
}

/// b3WarmStartJoint.
pub fn warm_start(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    match joint_type(joints, slot) {
        TY_DISTANCE => warm_start_distance(joints, slot, state_col, flags_col),
        TY_WELD => warm_start_weld(joints, slot, state_col, flags_col),
        TY_REVOLUTE => warm_start_revolute(joints, slot, state_col, flags_col),
        TY_SPHERICAL => warm_start_spherical(joints, slot, state_col, flags_col),
        TY_PRISMATIC => warm_start_prismatic(joints, slot, state_col, flags_col),
        TY_WHEEL => warm_start_wheel(joints, slot, state_col, flags_col),
        TY_MOTOR => warm_start_motor(joints, slot, state_col, flags_col),
        TY_PARALLEL => warm_start_parallel(joints, slot, state_col, flags_col),
        _ => {}
    }
}

/// b3SolveJoint.
pub fn solve(
    joints: Col<f32>,
    slot: usize,
    state_col: Col<f32>,
    flags_col: Col<u32>,
    use_bias: bool,
    h: f32,
    inv_h: f32,
) {
    match joint_type(joints, slot) {
        TY_DISTANCE => solve_distance(joints, slot, state_col, flags_col, use_bias, h, inv_h),
        TY_WELD => solve_weld(joints, slot, state_col, flags_col, use_bias),
        TY_REVOLUTE => solve_revolute(joints, slot, state_col, flags_col, use_bias, h, inv_h),
        TY_SPHERICAL => solve_spherical(joints, slot, state_col, flags_col, use_bias, h, inv_h),
        TY_PRISMATIC => solve_prismatic(joints, slot, state_col, flags_col, use_bias, h, inv_h),
        TY_WHEEL => solve_wheel(joints, slot, state_col, flags_col, use_bias, h, inv_h),
        TY_MOTOR => solve_motor(joints, slot, state_col, flags_col, h),
        TY_PARALLEL => solve_parallel(joints, slot, state_col, flags_col, h),
        _ => {}
    }
}

// --- shared payload helpers -------------------------------------------------------------------

#[inline]
fn read_softness(joints: Col<f32>, slot: usize, field: usize) -> Softness {
    Softness {
        bias_rate: get(joints, slot, field),
        mass_scale: get(joints, slot, field + 1),
        impulse_scale: get(joints, slot, field + 2),
    }
}

#[inline]
fn write_softness(joints: Col<f32>, slot: usize, field: usize, s: Softness) {
    set(joints, slot, field, s.bias_rate);
    set(joints, slot, field + 1, s.mass_scale);
    set(joints, slot, field + 2, s.impulse_scale);
}

// --- distance joint ---------------------------------------------------------------------------

/// b3PrepareDistanceJoint (`src/distanceJoint.ts` `prepareDistanceJoint`). The two sim indices +
/// invMass/invInertia are marshaled by TS; the pose fields feed the anchor computation.
fn prepare_distance(joints: Col<f32>, slot: usize, h: f32, enable_warm_starting: bool) {
    let base = read_base(joints, slot);
    let pose = read_pose(joints, slot);

    let anchor_a = pose
        .qa
        .rotate(base.local_frame_a.p.sub(pose.local_center_a));
    let anchor_b = pose
        .qb
        .rotate(base.local_frame_b.p.sub(pose.local_center_b));
    let delta_center = pose.center_b.sub(pose.center_a);

    let separation = anchor_b.sub(anchor_a).add(delta_center);
    let axis = separation.normalize();

    let cr_a = anchor_a.cross(axis);
    let cr_b = anchor_b.cross(axis);
    let k = base.inv_mass_a + base.inv_mass_b
        + cr_a.dot(base.inv_ia.mul_v(cr_a))
        + cr_b.dot(base.inv_ib.mul_v(cr_b));
    let axial_mass = if k > 0.0 { 1.0 / k } else { 0.0 };

    let soft = make_soft(
        get(joints, slot, DJ_HERTZ),
        get(joints, slot, DJ_DAMPING_RATIO),
        h,
    );

    set_vec3(joints, slot, DJ_ANCHOR_A, anchor_a);
    set_vec3(joints, slot, DJ_ANCHOR_B, anchor_b);
    set_vec3(joints, slot, DJ_DELTA_CENTER, delta_center);
    set(joints, slot, DJ_AXIAL_MASS, axial_mass);
    set(joints, slot, DJ_DIST_SOFTNESS, soft.bias_rate);
    set(joints, slot, DJ_DIST_SOFTNESS + 1, soft.mass_scale);
    set(joints, slot, DJ_DIST_SOFTNESS + 2, soft.impulse_scale);

    if !enable_warm_starting {
        set(joints, slot, DJ_IMPULSE, 0.0);
        set(joints, slot, DJ_LOWER_IMPULSE, 0.0);
        set(joints, slot, DJ_UPPER_IMPULSE, 0.0);
        set(joints, slot, DJ_MOTOR_IMPULSE, 0.0);
    }
}

/// b3WarmStartDistanceJoint.
fn warm_start_distance(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let anchor_a = get_vec3(joints, slot, DJ_ANCHOR_A);
    let anchor_b = get_vec3(joints, slot, DJ_ANCHOR_B);
    let delta_center = get_vec3(joints, slot, DJ_DELTA_CENTER);

    let r_a = end_a.state.delta_rotation.rotate(anchor_a);
    let r_b = end_b.state.delta_rotation.rotate(anchor_b);

    let ds = end_b
        .state
        .delta_position
        .sub(end_a.state.delta_position)
        .add(r_b.sub(r_a));
    let separation = delta_center.add(ds);
    let axis = separation.normalize();

    let axial_impulse = get(joints, slot, DJ_IMPULSE) + get(joints, slot, DJ_LOWER_IMPULSE)
        - get(joints, slot, DJ_UPPER_IMPULSE)
        + get(joints, slot, DJ_MOTOR_IMPULSE);
    let p = axis.scale(axial_impulse);

    let mut v_a = end_a.state.linear_velocity;
    let mut w_a = end_a.state.angular_velocity;
    let mut v_b = end_b.state.linear_velocity;
    let mut w_b = end_b.state.angular_velocity;
    if end_a.dynamic {
        v_a = v_a.mul_sub(m_a, p);
        w_a = w_a.sub(i_a.mul_v(r_a.cross(p)));
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        v_b = v_b.mul_add(m_b, p);
        w_b = w_b.add(i_b.mul_v(r_b.cross(p)));
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

/// The relative velocity of the two anchors projected onto the axis (`relVel` in the TS solve).
#[inline]
fn rel_vel(v_a: Vec3, w_a: Vec3, v_b: Vec3, w_b: Vec3, r_a: Vec3, r_b: Vec3, axis: Vec3) -> f32 {
    let vr = v_b.sub(v_a).add(w_b.cross(r_b).sub(w_a.cross(r_a)));
    axis.dot(vr)
}

/// Apply an axial impulse along `axis` to both ends (`applyP` in the TS solve).
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_p(
    v_a: &mut Vec3,
    w_a: &mut Vec3,
    v_b: &mut Vec3,
    w_b: &mut Vec3,
    impulse: f32,
    axis: Vec3,
    r_a: Vec3,
    r_b: Vec3,
    m_a: f32,
    m_b: f32,
    i_a: Mat3,
    i_b: Mat3,
) {
    let p = axis.scale(impulse);
    *v_a = v_a.mul_sub(m_a, p);
    *w_a = w_a.sub(i_a.mul_v(r_a.cross(p)));
    *v_b = v_b.mul_add(m_b, p);
    *w_b = w_b.add(i_b.mul_v(r_b.cross(p)));
}

/// b3SolveDistanceJoint (`solveDistanceJoint`).
fn solve_distance(
    joints: Col<f32>,
    slot: usize,
    state_col: Col<f32>,
    flags_col: Col<u32>,
    use_bias: bool,
    h: f32,
    inv_h: f32,
) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let mut v_a = end_a.state.linear_velocity;
    let mut w_a = end_a.state.angular_velocity;
    let mut v_b = end_b.state.linear_velocity;
    let mut w_b = end_b.state.angular_velocity;

    let anchor_a = get_vec3(joints, slot, DJ_ANCHOR_A);
    let anchor_b = get_vec3(joints, slot, DJ_ANCHOR_B);
    let delta_center = get_vec3(joints, slot, DJ_DELTA_CENTER);
    let axial_mass = get(joints, slot, DJ_AXIAL_MASS);

    let r_a = end_a.state.delta_rotation.rotate(anchor_a);
    let r_b = end_b.state.delta_rotation.rotate(anchor_b);

    let ds = end_b
        .state
        .delta_position
        .sub(end_a.state.delta_position)
        .add(r_b.sub(r_a));
    let separation = delta_center.add(ds);
    let length = separation.length();
    let axis = separation.normalize();

    let enable = get(joints, slot, DJ_ENABLE).to_bits();
    let enable_spring = enable & DJ_ENABLE_SPRING != 0;
    let enable_limit = enable & DJ_ENABLE_LIMIT != 0;
    let enable_motor = enable & DJ_ENABLE_MOTOR != 0;

    let length_joint = get(joints, slot, DJ_LENGTH);
    let min_length = get(joints, slot, DJ_MIN_LENGTH);
    let max_length = get(joints, slot, DJ_MAX_LENGTH);

    let cs_bias = get(joints, slot, J_CONSTRAINT_SOFTNESS);
    let cs_mass = get(joints, slot, J_CONSTRAINT_SOFTNESS + 1);
    let cs_impulse = get(joints, slot, J_CONSTRAINT_SOFTNESS + 2);

    if enable_spring && (min_length < max_length || !enable_limit) {
        // spring
        let hertz = get(joints, slot, DJ_HERTZ);
        if hertz > 0.0 {
            let d_bias = get(joints, slot, DJ_DIST_SOFTNESS);
            let d_mass = get(joints, slot, DJ_DIST_SOFTNESS + 1);
            let d_impulse = get(joints, slot, DJ_DIST_SOFTNESS + 2);
            let cdot = rel_vel(v_a, w_a, v_b, w_b, r_a, r_b, axis);
            let c = length - length_joint;
            let bias = d_bias * c;

            let m = d_mass * axial_mass;
            let old_impulse = get(joints, slot, DJ_IMPULSE);
            let mut impulse = -m * (cdot + bias) - d_impulse * old_impulse;
            let new_impulse = clampf(
                get(joints, slot, DJ_IMPULSE) + impulse,
                get(joints, slot, DJ_LOWER_SPRING_FORCE) * h,
                get(joints, slot, DJ_UPPER_SPRING_FORCE) * h,
            );
            set(joints, slot, DJ_IMPULSE, new_impulse);
            impulse = new_impulse - old_impulse;
            apply_p(
                &mut v_a, &mut w_a, &mut v_b, &mut w_b, impulse, axis, r_a, r_b, m_a, m_b, i_a, i_b,
            );
        }

        if enable_limit {
            // lower limit
            {
                let cdot = rel_vel(v_a, w_a, v_b, w_b, r_a, r_b, axis);
                let c = length - min_length;
                let mut bias = 0.0;
                let mut mass_coeff = 1.0_f32;
                let mut impulse_coeff = 0.0;
                if c > 0.0 {
                    bias = c * inv_h;
                } else if use_bias {
                    bias = cs_bias * c;
                    mass_coeff = cs_mass;
                    impulse_coeff = cs_impulse;
                }

                let lower_impulse = get(joints, slot, DJ_LOWER_IMPULSE);
                let mut impulse =
                    -mass_coeff * axial_mass * (cdot + bias) - impulse_coeff * lower_impulse;
                let new_impulse = maxf(0.0, lower_impulse + impulse);
                impulse = new_impulse - lower_impulse;
                set(joints, slot, DJ_LOWER_IMPULSE, new_impulse);
                apply_p(
                    &mut v_a, &mut w_a, &mut v_b, &mut w_b, impulse, axis, r_a, r_b, m_a, m_b, i_a,
                    i_b,
                );
            }

            // upper limit (impulse sign flipped)
            {
                let vr = v_a
                    .sub(v_b)
                    .add(w_a.cross(r_a).sub(w_b.cross(r_b)));
                let cdot = axis.dot(vr);
                let c = max_length - length;
                let mut bias = 0.0;
                let mut mass_scale = 1.0_f32;
                let mut impulse_scale = 0.0;
                if c > 0.0 {
                    bias = c * inv_h;
                } else if use_bias {
                    bias = cs_bias * c;
                    mass_scale = cs_mass;
                    impulse_scale = cs_impulse;
                }

                let upper_impulse = get(joints, slot, DJ_UPPER_IMPULSE);
                let mut impulse =
                    -mass_scale * axial_mass * (cdot + bias) - impulse_scale * upper_impulse;
                let new_impulse = maxf(0.0, upper_impulse + impulse);
                impulse = new_impulse - upper_impulse;
                set(joints, slot, DJ_UPPER_IMPULSE, new_impulse);

                let p = axis.scale(-impulse);
                v_a = v_a.mul_sub(m_a, p);
                w_a = w_a.sub(i_a.mul_v(r_a.cross(p)));
                v_b = v_b.mul_add(m_b, p);
                w_b = w_b.add(i_b.mul_v(r_b.cross(p)));
            }
        }

        if enable_motor {
            let cdot = rel_vel(v_a, w_a, v_b, w_b, r_a, r_b, axis);
            let mut impulse = axial_mass * (get(joints, slot, DJ_MOTOR_SPEED) - cdot);
            let old_impulse = get(joints, slot, DJ_MOTOR_IMPULSE);
            let max_impulse = h * get(joints, slot, DJ_MAX_MOTOR_FORCE);
            let new_impulse = clampf(old_impulse + impulse, -max_impulse, max_impulse);
            set(joints, slot, DJ_MOTOR_IMPULSE, new_impulse);
            impulse = new_impulse - old_impulse;
            apply_p(
                &mut v_a, &mut w_a, &mut v_b, &mut w_b, impulse, axis, r_a, r_b, m_a, m_b, i_a, i_b,
            );
        }
    } else {
        // rigid constraint
        let cdot = rel_vel(v_a, w_a, v_b, w_b, r_a, r_b, axis);
        let c = length - length_joint;

        let mut bias = 0.0;
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias {
            bias = cs_bias * c;
            mass_scale = cs_mass;
            impulse_scale = cs_impulse;
        }

        let old_impulse = get(joints, slot, DJ_IMPULSE);
        let impulse = -mass_scale * axial_mass * (cdot + bias) - impulse_scale * old_impulse;
        set(joints, slot, DJ_IMPULSE, old_impulse + impulse);
        apply_p(
            &mut v_a, &mut w_a, &mut v_b, &mut w_b, impulse, axis, r_a, r_b, m_a, m_b, i_a, i_b,
        );
    }

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

// --- weld joint -------------------------------------------------------------------------------

/// b3PrepareWeldJoint (`src/weldJoint.ts` `prepareWeldJoint`). Frames are world-space relative to each
/// COM; a zero linear/angular hertz falls the corresponding spring back to the base constraint softness.
fn prepare_weld(joints: Col<f32>, slot: usize, h: f32, enable_warm_starting: bool) {
    let base = read_base(joints, slot);
    let pose = read_pose(joints, slot);

    let frame_a = Transform {
        q: pose.qa.mul(base.local_frame_a.q),
        p: pose.qa.rotate(base.local_frame_a.p.sub(pose.local_center_a)),
    };
    let frame_b = Transform {
        q: pose.qb.mul(base.local_frame_b.q),
        p: pose.qb.rotate(base.local_frame_b.p.sub(pose.local_center_b)),
    };
    let delta_center = pose.center_b.sub(pose.center_a);

    let inv_inertia_sum = base.inv_ia.add(base.inv_ib);
    let fixed_rotation = inv_inertia_sum.det() < 1000.0 * FLT_MIN;
    let angular_mass = inv_inertia_sum.invert();

    let constraint_softness = read_softness(joints, slot, J_CONSTRAINT_SOFTNESS);
    let linear_hertz = get(joints, slot, WJ_LINEAR_HERTZ);
    let linear_spring = if linear_hertz == 0.0 {
        constraint_softness
    } else {
        make_soft(linear_hertz, get(joints, slot, WJ_LINEAR_DAMPING_RATIO), h)
    };
    let angular_hertz = get(joints, slot, WJ_ANGULAR_HERTZ);
    let angular_spring = if angular_hertz == 0.0 {
        constraint_softness
    } else {
        make_soft(angular_hertz, get(joints, slot, WJ_ANGULAR_DAMPING_RATIO), h)
    };

    set_transform(joints, slot, WJ_FRAME_A, frame_a);
    set_transform(joints, slot, WJ_FRAME_B, frame_b);
    set_vec3(joints, slot, WJ_DELTA_CENTER, delta_center);
    set_mat3(joints, slot, WJ_ANGULAR_MASS, angular_mass);
    write_softness(joints, slot, WJ_LINEAR_SPRING, linear_spring);
    write_softness(joints, slot, WJ_ANGULAR_SPRING, angular_spring);
    set(joints, slot, WJ_FIXED_ROTATION, if fixed_rotation { 1.0 } else { 0.0 });

    if !enable_warm_starting {
        set_vec3(joints, slot, WJ_LINEAR_IMPULSE, Vec3::ZERO);
        set_vec3(joints, slot, WJ_ANGULAR_IMPULSE, Vec3::ZERO);
    }
}

/// b3WarmStartWeldJoint.
fn warm_start_weld(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let frame_a = get_transform(joints, slot, WJ_FRAME_A);
    let frame_b = get_transform(joints, slot, WJ_FRAME_B);
    let linear_impulse = get_vec3(joints, slot, WJ_LINEAR_IMPULSE);
    let angular_impulse = get_vec3(joints, slot, WJ_ANGULAR_IMPULSE);

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

    let v_a = end_a.state.linear_velocity.mul_sub(m_a, linear_impulse);
    let w_a = end_a
        .state
        .angular_velocity
        .sub(i_a.mul_v(r_a.cross(linear_impulse).add(angular_impulse)));
    let v_b = end_b.state.linear_velocity.mul_add(m_b, linear_impulse);
    let w_b = end_b
        .state
        .angular_velocity
        .add(i_b.mul_v(r_b.cross(linear_impulse).add(angular_impulse)));

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

/// b3SolveWeldJoint.
fn solve_weld(
    joints: Col<f32>,
    slot: usize,
    state_col: Col<f32>,
    flags_col: Col<u32>,
    use_bias: bool,
) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let mut v_a = end_a.state.linear_velocity;
    let mut w_a = end_a.state.angular_velocity;
    let mut v_b = end_b.state.linear_velocity;
    let mut w_b = end_b.state.angular_velocity;

    let frame_a = get_transform(joints, slot, WJ_FRAME_A);
    let frame_b = get_transform(joints, slot, WJ_FRAME_B);
    let angular_mass = get_mat3(joints, slot, WJ_ANGULAR_MASS);
    let linear_spring = read_softness(joints, slot, WJ_LINEAR_SPRING);
    let angular_spring = read_softness(joints, slot, WJ_ANGULAR_SPRING);
    let linear_hertz = get(joints, slot, WJ_LINEAR_HERTZ);
    let angular_hertz = get(joints, slot, WJ_ANGULAR_HERTZ);
    let delta_center = get_vec3(joints, slot, WJ_DELTA_CENTER);
    let fixed_rotation = get(joints, slot, WJ_FIXED_ROTATION) != 0.0;

    let mut linear_impulse = get_vec3(joints, slot, WJ_LINEAR_IMPULSE);
    let mut angular_impulse = get_vec3(joints, slot, WJ_ANGULAR_IMPULSE);

    let quat_a = end_a.state.delta_rotation.mul(frame_a.q);
    let mut quat_b = end_b.state.delta_rotation.mul(frame_b.q);
    if quat_a.dot(quat_b) < 0.0 {
        quat_b = quat_b.negate();
    }
    let rel_q = quat_a.inv_mul(quat_b);

    // angular constraint
    if !fixed_rotation {
        let mut bias = Vec3::ZERO;
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias || angular_hertz > 0.0 {
            let target_quat = Quat::IDENTITY;
            let delta_rotation = rel_q.delta_to_rotation(target_quat);
            let c = quat_a.rotate(delta_rotation).neg();
            bias = c.scale(angular_spring.bias_rate);
            mass_scale = angular_spring.mass_scale;
            impulse_scale = angular_spring.impulse_scale;
        }

        let cdot = w_b.sub(w_a);
        let impulse = angular_mass
            .mul_v(cdot.add(bias))
            .scale(-mass_scale)
            .mul_sub(impulse_scale, angular_impulse);
        angular_impulse = angular_impulse.add(impulse);

        w_a = w_a.sub(i_a.mul_v(impulse));
        w_b = w_b.add(i_b.mul_v(impulse));
    }

    // linear constraint
    {
        let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
        let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

        let cdot = v_b
            .add(w_b.cross(r_b))
            .sub(v_a.add(w_a.cross(r_a)));

        let mut bias = Vec3::ZERO;
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias || linear_hertz > 0.0 {
            let dc_a = end_a.state.delta_position;
            let dc_b = end_b.state.delta_position;
            let separation = dc_b.sub(dc_a).add(r_b.sub(r_a)).add(delta_center);
            bias = separation.scale(linear_spring.bias_rate);
            mass_scale = linear_spring.mass_scale;
            impulse_scale = linear_spring.impulse_scale;
        }

        let s_a = Mat3::skew(r_a);
        let s_b = Mat3::skew(r_b);
        let k_a = s_a.mul(base.inv_ia.mul(s_a));
        let k_b = s_b.mul(base.inv_ib.mul(s_b));
        let mut k = k_a.add(k_b).neg();
        let mm = m_a + m_b;
        k.cx.x += mm;
        k.cy.y += mm;
        k.cz.z += mm;

        let b = k.solve(cdot.add(bias));

        let impulse = b.scale(-mass_scale).mul_sub(impulse_scale, linear_impulse);
        linear_impulse = linear_impulse.add(impulse);

        v_a = v_a.mul_sub(m_a, impulse);
        w_a = w_a.sub(i_a.mul_v(r_a.cross(impulse)));
        v_b = v_b.mul_add(m_b, impulse);
        w_b = w_b.add(i_b.mul_v(r_b.cross(impulse)));
    }

    set_vec3(joints, slot, WJ_LINEAR_IMPULSE, linear_impulse);
    set_vec3(joints, slot, WJ_ANGULAR_IMPULSE, angular_impulse);

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

// --- revolute joint ---------------------------------------------------------------------------

/// b3PrepareRevoluteJoint (`src/revoluteJoint.ts` `prepareRevoluteJoint`). The hinge axis is body A's
/// local z; the perp axes are the warm-start collinearity basis.
fn prepare_revolute(joints: Col<f32>, slot: usize, h: f32, enable_warm_starting: bool) {
    let base = read_base(joints, slot);
    let pose = read_pose(joints, slot);

    let frame_a = Transform {
        q: pose.qa.mul(base.local_frame_a.q),
        p: pose.qa.rotate(base.local_frame_a.p.sub(pose.local_center_a)),
    };
    let frame_b = Transform {
        q: pose.qb.mul(base.local_frame_b.q),
        p: pose.qb.rotate(base.local_frame_b.p.sub(pose.local_center_b)),
    };
    let delta_center = pose.center_b.sub(pose.center_a);

    let inv_inertia_sum = base.inv_ia.add(base.inv_ib);
    let fixed_rotation = inv_inertia_sum.det() < 1000.0 * FLT_MIN;

    let axis_x = Vec3::new(1.0, 0.0, 0.0);
    let axis_y = Vec3::new(0.0, 1.0, 0.0);
    let axis_z = Vec3::new(0.0, 0.0, 1.0);

    let rotation_axis_z = frame_a.q.rotate(axis_z);
    let k = rotation_axis_z.dot(inv_inertia_sum.mul_v(rotation_axis_z));
    let axial_mass = if k > 0.0 { 1.0 / k } else { 0.0 };

    let rel_q = frame_a.q.inv_mul(frame_b.q);
    let perp_axis_x = frame_a
        .q
        .rotate(axis_x.scale(rel_q.s).add(rel_q.v.cross(axis_x)))
        .scale(0.5);
    let perp_axis_y = frame_a
        .q
        .rotate(axis_y.scale(rel_q.s).add(rel_q.v.cross(axis_y)))
        .scale(0.5);

    let soft = make_soft(
        get(joints, slot, RJ_HERTZ),
        get(joints, slot, RJ_DAMPING_RATIO),
        h,
    );

    set_transform(joints, slot, RJ_FRAME_A, frame_a);
    set_transform(joints, slot, RJ_FRAME_B, frame_b);
    set_vec3(joints, slot, RJ_ROTATION_AXIS_Z, rotation_axis_z);
    set_vec3(joints, slot, RJ_PERP_AXIS_X, perp_axis_x);
    set_vec3(joints, slot, RJ_PERP_AXIS_Y, perp_axis_y);
    set_vec3(joints, slot, RJ_DELTA_CENTER, delta_center);
    set(joints, slot, RJ_AXIAL_MASS, axial_mass);
    write_softness(joints, slot, RJ_SPRING_SOFTNESS, soft);
    set(joints, slot, RJ_FIXED_ROTATION, if fixed_rotation { 1.0 } else { 0.0 });

    if !enable_warm_starting {
        set_vec3(joints, slot, RJ_LINEAR_IMPULSE, Vec3::ZERO);
        set(joints, slot, RJ_PERP_IMPULSE, 0.0);
        set(joints, slot, RJ_PERP_IMPULSE + 1, 0.0);
        set(joints, slot, RJ_MOTOR_IMPULSE, 0.0);
        set(joints, slot, RJ_SPRING_IMPULSE, 0.0);
        set(joints, slot, RJ_LOWER_IMPULSE, 0.0);
        set(joints, slot, RJ_UPPER_IMPULSE, 0.0);
    }
}

/// b3WarmStartRevoluteJoint.
fn warm_start_revolute(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let frame_a = get_transform(joints, slot, RJ_FRAME_A);
    let frame_b = get_transform(joints, slot, RJ_FRAME_B);
    let rotation_axis_z = get_vec3(joints, slot, RJ_ROTATION_AXIS_Z);
    let perp_axis_x = get_vec3(joints, slot, RJ_PERP_AXIS_X);
    let perp_axis_y = get_vec3(joints, slot, RJ_PERP_AXIS_Y);
    let linear_impulse = get_vec3(joints, slot, RJ_LINEAR_IMPULSE);
    let perp_x = get(joints, slot, RJ_PERP_IMPULSE);
    let perp_y = get(joints, slot, RJ_PERP_IMPULSE + 1);
    let spring_impulse = get(joints, slot, RJ_SPRING_IMPULSE);
    let motor_impulse = get(joints, slot, RJ_MOTOR_IMPULSE);
    let lower_impulse = get(joints, slot, RJ_LOWER_IMPULSE);
    let upper_impulse = get(joints, slot, RJ_UPPER_IMPULSE);

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

    let axial_impulse = ((spring_impulse + motor_impulse) + lower_impulse) - upper_impulse;
    let mut angular_impulse = perp_axis_x.scale(perp_x).add(perp_axis_y.scale(perp_y));
    angular_impulse = angular_impulse.mul_add(axial_impulse, rotation_axis_z);

    let v_a = end_a.state.linear_velocity.mul_sub(m_a, linear_impulse);
    let w_a = end_a
        .state
        .angular_velocity
        .sub(i_a.mul_v(r_a.cross(linear_impulse).add(angular_impulse)));
    let v_b = end_b.state.linear_velocity.mul_add(m_b, linear_impulse);
    let w_b = end_b
        .state
        .angular_velocity
        .add(i_b.mul_v(r_b.cross(linear_impulse).add(angular_impulse)));

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

/// b3SolveRevoluteJoint.
#[allow(clippy::too_many_lines)]
fn solve_revolute(
    joints: Col<f32>,
    slot: usize,
    state_col: Col<f32>,
    flags_col: Col<u32>,
    use_bias: bool,
    h: f32,
    inv_h: f32,
) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);
    let mut v_a = end_a.state.linear_velocity;
    let mut w_a = end_a.state.angular_velocity;
    let mut v_b = end_b.state.linear_velocity;
    let mut w_b = end_b.state.angular_velocity;

    let frame_a = get_transform(joints, slot, RJ_FRAME_A);
    let frame_b = get_transform(joints, slot, RJ_FRAME_B);
    let rotation_axis_z = get_vec3(joints, slot, RJ_ROTATION_AXIS_Z);
    let delta_center = get_vec3(joints, slot, RJ_DELTA_CENTER);
    let axial_mass = get(joints, slot, RJ_AXIAL_MASS);
    let spring_soft = read_softness(joints, slot, RJ_SPRING_SOFTNESS);
    let cs = read_softness(joints, slot, J_CONSTRAINT_SOFTNESS);
    let fixed_rotation = get(joints, slot, RJ_FIXED_ROTATION) != 0.0;
    let enable = get(joints, slot, RJ_ENABLE).to_bits();
    let enable_spring = enable & RJ_ENABLE_SPRING != 0;
    let enable_motor = enable & RJ_ENABLE_MOTOR != 0;
    let enable_limit = enable & RJ_ENABLE_LIMIT != 0;
    let target_angle = get(joints, slot, RJ_TARGET_ANGLE);
    let motor_speed = get(joints, slot, RJ_MOTOR_SPEED);
    let max_motor_torque = get(joints, slot, RJ_MAX_MOTOR_TORQUE);
    let lower_angle = get(joints, slot, RJ_LOWER_ANGLE);
    let upper_angle = get(joints, slot, RJ_UPPER_ANGLE);

    let mut linear_impulse = get_vec3(joints, slot, RJ_LINEAR_IMPULSE);
    let mut perp_impulse = Vec2::new(
        get(joints, slot, RJ_PERP_IMPULSE),
        get(joints, slot, RJ_PERP_IMPULSE + 1),
    );
    let mut spring_impulse = get(joints, slot, RJ_SPRING_IMPULSE);
    let mut motor_impulse = get(joints, slot, RJ_MOTOR_IMPULSE);
    let mut lower_impulse = get(joints, slot, RJ_LOWER_IMPULSE);
    let mut upper_impulse = get(joints, slot, RJ_UPPER_IMPULSE);

    let quat_a = end_a.state.delta_rotation.mul(frame_a.q);
    let mut quat_b = end_b.state.delta_rotation.mul(frame_b.q);
    if quat_a.dot(quat_b) < 0.0 {
        quat_b = quat_b.negate();
    }
    let rel_q = quat_a.inv_mul(quat_b);

    // Solve spring
    if enable_spring && !fixed_rotation {
        let angle = rel_q.get_twist_angle();
        let c = angle - target_angle;
        let bias = spring_soft.bias_rate * c;
        let mass_scale = spring_soft.mass_scale;
        let impulse_scale = spring_soft.impulse_scale;
        let cdot = w_b.sub(w_a).dot(rotation_axis_z);

        let delta_impulse =
            -mass_scale * axial_mass * (cdot + bias) - impulse_scale * spring_impulse;
        spring_impulse += delta_impulse;

        w_a = w_a.mul_sub(delta_impulse, i_a.mul_v(rotation_axis_z));
        w_b = w_b.mul_add(delta_impulse, i_b.mul_v(rotation_axis_z));
    }

    if enable_motor && !fixed_rotation {
        let cdot = w_b.sub(w_a).dot(rotation_axis_z) - motor_speed;
        let mut delta_impulse = -axial_mass * cdot;
        let mut new_impulse = motor_impulse + delta_impulse;
        let max_impulse = max_motor_torque * h;
        new_impulse = clampf(new_impulse, -max_impulse, max_impulse);
        delta_impulse = new_impulse - motor_impulse;
        motor_impulse = new_impulse;

        w_a = w_a.mul_sub(delta_impulse, i_a.mul_v(rotation_axis_z));
        w_b = w_b.mul_add(delta_impulse, i_b.mul_v(rotation_axis_z));
    }

    if enable_limit && !fixed_rotation {
        let angle = rel_q.get_twist_angle();
        let axis = rotation_axis_z;

        // Lower limit
        {
            let c = angle - lower_angle;
            let mut bias = 0.0;
            let mut mass_scale = 1.0_f32;
            let mut impulse_scale = 0.0;
            if c > 0.0 {
                bias = c * inv_h;
            } else if use_bias {
                bias = cs.bias_rate * c;
                mass_scale = cs.mass_scale;
                impulse_scale = cs.impulse_scale;
            }
            let cdot = w_b.sub(w_a).dot(axis);
            let old_impulse = lower_impulse;
            let mut delta_impulse =
                -mass_scale * axial_mass * (cdot + bias) - impulse_scale * old_impulse;
            lower_impulse = maxf(old_impulse + delta_impulse, 0.0);
            delta_impulse = lower_impulse - old_impulse;

            w_a = w_a.mul_sub(delta_impulse, i_a.mul_v(axis));
            w_b = w_b.mul_add(delta_impulse, i_b.mul_v(axis));
        }

        // Upper limit
        {
            let c = upper_angle - angle;
            let mut bias = 0.0;
            let mut mass_scale = 1.0_f32;
            let mut impulse_scale = 0.0;
            if c > 0.0 {
                bias = c * inv_h;
            } else if use_bias {
                bias = cs.bias_rate * c;
                mass_scale = cs.mass_scale;
                impulse_scale = cs.impulse_scale;
            }
            // sign flipped on Cdot
            let cdot = w_a.sub(w_b).dot(axis);
            let old_impulse = upper_impulse;
            let mut delta_impulse =
                -mass_scale * axial_mass * (cdot + bias) - impulse_scale * old_impulse;
            upper_impulse = maxf(old_impulse + delta_impulse, 0.0);
            delta_impulse = upper_impulse - old_impulse;

            // sign flipped on applied impulse
            w_a = w_a.mul_add(delta_impulse, i_a.mul_v(axis));
            w_b = w_b.mul_sub(delta_impulse, i_b.mul_v(axis));
        }
    }

    // Collinearity constraint
    if !fixed_rotation {
        let mut bias = Vec2::new(0.0, 0.0);
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias {
            let c = Vec2::new(rel_q.v.x, rel_q.v.y);
            bias = c.scale(cs.bias_rate);
            mass_scale = cs.mass_scale;
            impulse_scale = cs.impulse_scale;
        }

        let axis_x = Vec3::new(1.0, 0.0, 0.0);
        let axis_y = Vec3::new(0.0, 1.0, 0.0);
        let perp_axis_x = quat_a
            .rotate(axis_x.scale(rel_q.s).add(rel_q.v.cross(axis_x)))
            .scale(0.5);
        let perp_axis_y = quat_a
            .rotate(axis_y.scale(rel_q.s).add(rel_q.v.cross(axis_y)))
            .scale(0.5);
        set_vec3(joints, slot, RJ_PERP_AXIS_X, perp_axis_x);
        set_vec3(joints, slot, RJ_PERP_AXIS_Y, perp_axis_y);

        let inv_inertia_sum = i_a.add(i_b);
        let kxx = perp_axis_x.dot(inv_inertia_sum.mul_v(perp_axis_x));
        let kyy = perp_axis_y.dot(inv_inertia_sum.mul_v(perp_axis_y));
        let kxy = perp_axis_x.dot(inv_inertia_sum.mul_v(perp_axis_y));
        let k = Mat2 {
            cx: Vec2::new(kxx, kxy),
            cy: Vec2::new(kxy, kyy),
        };

        let w_rel = w_b.sub(w_a);
        let cdot = Vec2::new(w_rel.dot(perp_axis_x), w_rel.dot(perp_axis_y));
        let old_impulse = perp_impulse;
        let sol = k.solve(cdot.add(bias));
        let delta_impulse = sol.scale(-mass_scale).sub(old_impulse.scale(impulse_scale));
        perp_impulse = perp_impulse.add(delta_impulse);

        let angular_impulse = perp_axis_x
            .scale(delta_impulse.x)
            .add(perp_axis_y.scale(delta_impulse.y));
        w_a = w_a.sub(i_a.mul_v(angular_impulse));
        w_b = w_b.add(i_b.mul_v(angular_impulse));
    }

    // Solve point-to-point constraint
    {
        let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
        let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

        let cdot = v_b
            .add(w_b.cross(r_b))
            .sub(v_a)
            .sub(w_a.cross(r_a));

        let mut bias = Vec3::ZERO;
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias {
            let dc_a = end_a.state.delta_position;
            let dc_b = end_b.state.delta_position;
            let separation = dc_b.sub(dc_a).add(r_b.sub(r_a)).add(delta_center);
            bias = separation.scale(cs.bias_rate);
            mass_scale = cs.mass_scale;
            impulse_scale = cs.impulse_scale;
        }

        let s_a = Mat3::skew(r_a);
        let s_b = Mat3::skew(r_b);
        let k_a = s_a.mul(base.inv_ia.mul(s_a));
        let k_b = s_b.mul(base.inv_ib.mul(s_b));
        let mut k = k_a.add(k_b).neg();
        let mm = m_a + m_b;
        k.cx.x += mm;
        k.cy.y += mm;
        k.cz.z += mm;

        let b = k.solve(cdot.add(bias));
        let impulse = b.scale(-mass_scale).sub(linear_impulse.scale(impulse_scale));
        linear_impulse = linear_impulse.add(impulse);

        v_a = v_a.mul_sub(m_a, impulse);
        w_a = w_a.sub(i_a.mul_v(r_a.cross(impulse)));
        v_b = v_b.mul_add(m_b, impulse);
        w_b = w_b.add(i_b.mul_v(r_b.cross(impulse)));
    }

    set_vec3(joints, slot, RJ_LINEAR_IMPULSE, linear_impulse);
    set(joints, slot, RJ_PERP_IMPULSE, perp_impulse.x);
    set(joints, slot, RJ_PERP_IMPULSE + 1, perp_impulse.y);
    set(joints, slot, RJ_SPRING_IMPULSE, spring_impulse);
    set(joints, slot, RJ_MOTOR_IMPULSE, motor_impulse);
    set(joints, slot, RJ_LOWER_IMPULSE, lower_impulse);
    set(joints, slot, RJ_UPPER_IMPULSE, upper_impulse);

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

// --- spherical joint --------------------------------------------------------------------------

/// b3PrepareSphericalJoint (`src/sphericalJoint.ts` `prepareSphericalJoint`). The cone axis is body A's
/// local z, the twist axis body B's; the swing axis / twist jacobian / masses are zero when their limit
/// is off (matching the serial path's zero-defaulted fields, so the read in warm-start no-ops safely).
fn prepare_spherical(joints: Col<f32>, slot: usize, h: f32, enable_warm_starting: bool) {
    let base = read_base(joints, slot);
    let pose = read_pose(joints, slot);

    let frame_a = Transform {
        q: pose.qa.mul(base.local_frame_a.q),
        p: pose.qa.rotate(base.local_frame_a.p.sub(pose.local_center_a)),
    };
    let frame_b = Transform {
        q: pose.qb.mul(base.local_frame_b.q),
        p: pose.qb.rotate(base.local_frame_b.p.sub(pose.local_center_b)),
    };
    let delta_center = pose.center_b.sub(pose.center_a);

    let inv_inertia_sum = base.inv_ia.add(base.inv_ib);
    let fixed_rotation = inv_inertia_sum.det() < 1000.0 * FLT_MIN;

    let axis_z = Vec3::new(0.0, 0.0, 1.0);
    let cone_axis = frame_a.q.rotate(axis_z);
    let twist_axis = frame_b.q.rotate(axis_z);

    let enable = get(joints, slot, SJ_ENABLE).to_bits();
    let enable_cone = enable & SJ_ENABLE_CONE_LIMIT != 0;
    let enable_twist = enable & SJ_ENABLE_TWIST_LIMIT != 0;

    let mut swing_axis = Vec3::ZERO;
    let mut swing_mass = 0.0;
    if enable_cone {
        swing_axis = cone_axis.cross(twist_axis).normalize();
        let k = swing_axis.dot(inv_inertia_sum.mul_v(swing_axis));
        swing_mass = if k > 0.0 { 1.0 / k } else { 0.0 };
    }

    let mut twist_jacobian = Vec3::ZERO;
    let mut twist_mass = 0.0;
    if enable_twist {
        let rel_q = frame_a.q.inv_mul(frame_b.q);
        let num = rel_q.v.x * rel_q.v.x + rel_q.v.y * rel_q.v.y;
        let den = rel_q.v.z * rel_q.v.z + rel_q.s * rel_q.s;
        let tan_theta_over_2 = (num / den).sqrt();

        let swing_axis_t = cone_axis.cross(twist_axis).normalize();
        let perp_axis = swing_axis_t.cross(cone_axis);
        twist_jacobian = cone_axis.mul_add(tan_theta_over_2, perp_axis);
        let k = twist_jacobian.dot(inv_inertia_sum.mul_v(twist_jacobian));
        twist_mass = if k > 0.0 { 1.0 / k } else { 0.0 };
    }

    let rotation_mass = if fixed_rotation {
        Mat3::ZERO
    } else {
        inv_inertia_sum.invert()
    };

    let soft = make_soft(
        get(joints, slot, SJ_HERTZ),
        get(joints, slot, SJ_DAMPING_RATIO),
        h,
    );

    set_transform(joints, slot, SJ_FRAME_A, frame_a);
    set_transform(joints, slot, SJ_FRAME_B, frame_b);
    set_vec3(joints, slot, SJ_DELTA_CENTER, delta_center);
    set_vec3(joints, slot, SJ_SWING_AXIS, swing_axis);
    set_vec3(joints, slot, SJ_TWIST_JACOBIAN, twist_jacobian);
    set_mat3(joints, slot, SJ_ROTATION_MASS, rotation_mass);
    set(joints, slot, SJ_SWING_MASS, swing_mass);
    set(joints, slot, SJ_TWIST_MASS, twist_mass);
    write_softness(joints, slot, SJ_SPRING_SOFTNESS, soft);
    set(joints, slot, SJ_FIXED_ROTATION, if fixed_rotation { 1.0 } else { 0.0 });

    if !enable_warm_starting {
        set_vec3(joints, slot, SJ_LINEAR_IMPULSE, Vec3::ZERO);
        set_vec3(joints, slot, SJ_MOTOR_IMPULSE, Vec3::ZERO);
        set_vec3(joints, slot, SJ_SPRING_IMPULSE, Vec3::ZERO);
        set(joints, slot, SJ_SWING_IMPULSE, 0.0);
        set(joints, slot, SJ_LOWER_TWIST_IMPULSE, 0.0);
        set(joints, slot, SJ_UPPER_TWIST_IMPULSE, 0.0);
    }
}

/// b3WarmStartSphericalJoint.
fn warm_start_spherical(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let frame_a = get_transform(joints, slot, SJ_FRAME_A);
    let frame_b = get_transform(joints, slot, SJ_FRAME_B);
    let swing_axis = get_vec3(joints, slot, SJ_SWING_AXIS);
    let twist_jacobian = get_vec3(joints, slot, SJ_TWIST_JACOBIAN);
    let linear_impulse = get_vec3(joints, slot, SJ_LINEAR_IMPULSE);
    let spring_impulse = get_vec3(joints, slot, SJ_SPRING_IMPULSE);
    let motor_impulse = get_vec3(joints, slot, SJ_MOTOR_IMPULSE);
    let lower_twist = get(joints, slot, SJ_LOWER_TWIST_IMPULSE);
    let upper_twist = get(joints, slot, SJ_UPPER_TWIST_IMPULSE);
    let swing_impulse = get(joints, slot, SJ_SWING_IMPULSE);

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

    let mut angular_impulse = spring_impulse.add(motor_impulse);
    angular_impulse = angular_impulse.mul_sub(swing_impulse, swing_axis);
    angular_impulse = angular_impulse.mul_add(lower_twist - upper_twist, twist_jacobian);

    let v_a = end_a.state.linear_velocity.mul_sub(m_a, linear_impulse);
    let w_a = end_a
        .state
        .angular_velocity
        .sub(i_a.mul_v(r_a.cross(linear_impulse).add(angular_impulse)));
    let v_b = end_b.state.linear_velocity.mul_add(m_b, linear_impulse);
    let w_b = end_b
        .state
        .angular_velocity
        .add(i_b.mul_v(r_b.cross(linear_impulse).add(angular_impulse)));

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

/// b3SolveSphericalJoint.
#[allow(clippy::too_many_lines)]
fn solve_spherical(
    joints: Col<f32>,
    slot: usize,
    state_col: Col<f32>,
    flags_col: Col<u32>,
    use_bias: bool,
    h: f32,
    inv_h: f32,
) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);
    let mut v_a = end_a.state.linear_velocity;
    let mut w_a = end_a.state.angular_velocity;
    let mut v_b = end_b.state.linear_velocity;
    let mut w_b = end_b.state.angular_velocity;

    let frame_a = get_transform(joints, slot, SJ_FRAME_A);
    let frame_b = get_transform(joints, slot, SJ_FRAME_B);
    let delta_center = get_vec3(joints, slot, SJ_DELTA_CENTER);
    let swing_axis = get_vec3(joints, slot, SJ_SWING_AXIS);
    let twist_jacobian = get_vec3(joints, slot, SJ_TWIST_JACOBIAN);
    let rotation_mass = get_mat3(joints, slot, SJ_ROTATION_MASS);
    let swing_mass = get(joints, slot, SJ_SWING_MASS);
    let twist_mass = get(joints, slot, SJ_TWIST_MASS);
    let spring_soft = read_softness(joints, slot, SJ_SPRING_SOFTNESS);
    let cs = read_softness(joints, slot, J_CONSTRAINT_SOFTNESS);
    let fixed_rotation = get(joints, slot, SJ_FIXED_ROTATION) != 0.0;
    let target_rotation = get_quat(joints, slot, SJ_TARGET_ROTATION);
    let motor_velocity = get_vec3(joints, slot, SJ_MOTOR_VELOCITY);
    let max_motor_torque = get(joints, slot, SJ_MAX_MOTOR_TORQUE);
    let lower_twist_angle = get(joints, slot, SJ_LOWER_TWIST_ANGLE);
    let upper_twist_angle = get(joints, slot, SJ_UPPER_TWIST_ANGLE);
    let cone_angle = get(joints, slot, SJ_CONE_ANGLE);
    let enable = get(joints, slot, SJ_ENABLE).to_bits();
    let enable_spring = enable & SJ_ENABLE_SPRING != 0;
    let enable_motor = enable & SJ_ENABLE_MOTOR != 0;
    let enable_cone = enable & SJ_ENABLE_CONE_LIMIT != 0;
    let enable_twist = enable & SJ_ENABLE_TWIST_LIMIT != 0;

    let mut linear_impulse = get_vec3(joints, slot, SJ_LINEAR_IMPULSE);
    let mut spring_impulse = get_vec3(joints, slot, SJ_SPRING_IMPULSE);
    let mut motor_impulse = get_vec3(joints, slot, SJ_MOTOR_IMPULSE);
    let mut lower_twist_impulse = get(joints, slot, SJ_LOWER_TWIST_IMPULSE);
    let mut upper_twist_impulse = get(joints, slot, SJ_UPPER_TWIST_IMPULSE);
    let mut swing_impulse = get(joints, slot, SJ_SWING_IMPULSE);

    let quat_a = end_a.state.delta_rotation.mul(frame_a.q);
    let quat_b = end_b.state.delta_rotation.mul(frame_b.q);
    let rel_q = quat_a.inv_mul(quat_b);

    // Solve spring
    if enable_spring && !fixed_rotation {
        let delta_rotation = rel_q.delta_to_rotation(target_rotation);
        let c = quat_a.rotate(delta_rotation).neg();
        let bias = c.scale(spring_soft.bias_rate);
        let mass_scale = spring_soft.mass_scale;
        let impulse_scale = spring_soft.impulse_scale;
        let cdot = w_b.sub(w_a);

        let impulse = rotation_mass
            .mul_v(cdot.add(bias))
            .scale(-mass_scale)
            .mul_sub(impulse_scale, spring_impulse);
        spring_impulse = spring_impulse.add(impulse);

        w_a = w_a.sub(i_a.mul_v(impulse));
        w_b = w_b.add(i_b.mul_v(impulse));
    }

    if enable_motor && !fixed_rotation {
        let cdot = w_b.sub(w_a);
        let mut lambda = rotation_mass.mul_v(cdot.sub(motor_velocity)).neg();
        let mut new_impulse = motor_impulse.add(lambda);
        let length = new_impulse.length_sq().sqrt();
        let max_impulse = max_motor_torque * h;
        if length > max_impulse {
            new_impulse = new_impulse.scale(max_impulse / length);
        }
        lambda = new_impulse.sub(motor_impulse);
        motor_impulse = new_impulse;

        w_a = w_a.sub(i_a.mul_v(lambda));
        w_b = w_b.add(i_b.mul_v(lambda));
    }

    if enable_twist && !fixed_rotation {
        let twist_angle = rel_q.get_twist_angle();

        // Lower limit
        {
            let c = twist_angle - lower_twist_angle;
            let mut bias = 0.0;
            let mut mass_scale = 1.0_f32;
            let mut impulse_scale = 0.0;
            if c > 0.0 {
                bias = c * inv_h;
            } else if use_bias {
                bias = cs.bias_rate * c;
                mass_scale = cs.mass_scale;
                impulse_scale = cs.impulse_scale;
            }
            let cdot = w_b.sub(w_a).dot(twist_jacobian);
            let old_impulse = lower_twist_impulse;
            let mut delta_impulse =
                -mass_scale * twist_mass * (cdot + bias) - impulse_scale * old_impulse;
            lower_twist_impulse = maxf(old_impulse + delta_impulse, 0.0);
            delta_impulse = lower_twist_impulse - old_impulse;
            w_a = w_a.mul_sub(delta_impulse, i_a.mul_v(twist_jacobian));
            w_b = w_b.mul_add(delta_impulse, i_b.mul_v(twist_jacobian));
        }

        // Upper limit
        {
            let c = upper_twist_angle - twist_angle;
            let mut bias = 0.0;
            let mut mass_scale = 1.0_f32;
            let mut impulse_scale = 0.0;
            if c > 0.0 {
                bias = c * inv_h;
            } else if use_bias {
                bias = cs.bias_rate * c;
                mass_scale = cs.mass_scale;
                impulse_scale = cs.impulse_scale;
            }
            // sign flipped on Cdot
            let cdot = w_a.sub(w_b).dot(twist_jacobian);
            let old_impulse = upper_twist_impulse;
            let mut delta_impulse =
                -mass_scale * twist_mass * (cdot + bias) - impulse_scale * old_impulse;
            upper_twist_impulse = maxf(old_impulse + delta_impulse, 0.0);
            delta_impulse = upper_twist_impulse - old_impulse;
            // sign flipped on applied impulse
            w_a = w_a.mul_add(delta_impulse, i_a.mul_v(twist_jacobian));
            w_b = w_b.mul_sub(delta_impulse, i_b.mul_v(twist_jacobian));
        }
    }

    if enable_cone && !fixed_rotation {
        let swing_angle = rel_q.get_swing_angle();
        let c = cone_angle - swing_angle;
        let mut bias = 0.0;
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if c > 0.0 {
            bias = c * inv_h;
        } else if use_bias {
            bias = cs.bias_rate * c;
            mass_scale = cs.mass_scale;
            impulse_scale = cs.impulse_scale;
        }
        // sign flipped on Cdot
        let cdot = w_a.sub(w_b).dot(swing_axis);
        let old_impulse = swing_impulse;
        let mut delta_impulse =
            -mass_scale * swing_mass * (cdot + bias) - impulse_scale * old_impulse;
        swing_impulse = maxf(old_impulse + delta_impulse, 0.0);
        delta_impulse = swing_impulse - old_impulse;
        // sign flipped on applied impulse
        w_a = w_a.mul_add(delta_impulse, i_a.mul_v(swing_axis));
        w_b = w_b.mul_sub(delta_impulse, i_b.mul_v(swing_axis));
    }

    // Solve point-to-point constraint
    {
        let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
        let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

        let cdot = v_b
            .add(w_b.cross(r_b))
            .sub(v_a)
            .sub(w_a.cross(r_a));

        let mut bias = Vec3::ZERO;
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias {
            let dc_a = end_a.state.delta_position;
            let dc_b = end_b.state.delta_position;
            let separation = dc_b.sub(dc_a).add(r_b.sub(r_a)).add(delta_center);
            bias = separation.scale(cs.bias_rate);
            mass_scale = cs.mass_scale;
            impulse_scale = cs.impulse_scale;
        }

        let s_a = Mat3::skew(r_a);
        let s_b = Mat3::skew(r_b);
        let k_a = s_a.mul(base.inv_ia.mul(s_a));
        let k_b = s_b.mul(base.inv_ib.mul(s_b));
        let mut k = k_a.add(k_b).neg();
        let mm = m_a + m_b;
        k.cx.x += mm;
        k.cy.y += mm;
        k.cz.z += mm;

        let b = k.solve(cdot.add(bias));
        let impulse = b.scale(-mass_scale).sub(linear_impulse.scale(impulse_scale));
        linear_impulse = linear_impulse.add(impulse);

        v_a = v_a.mul_sub(m_a, impulse);
        w_a = w_a.sub(i_a.mul_v(r_a.cross(impulse)));
        v_b = v_b.mul_add(m_b, impulse);
        w_b = w_b.add(i_b.mul_v(r_b.cross(impulse)));
    }

    set_vec3(joints, slot, SJ_LINEAR_IMPULSE, linear_impulse);
    set_vec3(joints, slot, SJ_SPRING_IMPULSE, spring_impulse);
    set_vec3(joints, slot, SJ_MOTOR_IMPULSE, motor_impulse);
    set(joints, slot, SJ_LOWER_TWIST_IMPULSE, lower_twist_impulse);
    set(joints, slot, SJ_UPPER_TWIST_IMPULSE, upper_twist_impulse);
    set(joints, slot, SJ_SWING_IMPULSE, swing_impulse);

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

// --- prismatic joint --------------------------------------------------------------------------

/// b3PreparePrismaticJoint (`src/prismaticJoint.ts`). The joint axis is body A's local x; perpY/perpZ are
/// its local y/z. `rotationMass` is always inverted (unlike spherical); the axial effective mass is not
/// prepared — it is recomputed fresh each solve step.
fn prepare_prismatic(joints: Col<f32>, slot: usize, h: f32, enable_warm_starting: bool) {
    let base = read_base(joints, slot);
    let pose = read_pose(joints, slot);

    let frame_a = Transform {
        q: pose.qa.mul(base.local_frame_a.q),
        p: pose.qa.rotate(base.local_frame_a.p.sub(pose.local_center_a)),
    };
    let frame_b = Transform {
        q: pose.qb.mul(base.local_frame_b.q),
        p: pose.qb.rotate(base.local_frame_b.p.sub(pose.local_center_b)),
    };
    let delta_center = pose.center_b.sub(pose.center_a);

    let inv_inertia_sum = base.inv_ia.add(base.inv_ib);
    let fixed_rotation = inv_inertia_sum.det() < 1000.0 * FLT_MIN;
    let rotation_mass = inv_inertia_sum.invert();

    let matrix_a = Mat3::from_quat(frame_a.q);
    let joint_axis = matrix_a.cx;
    let perp_axis_y = matrix_a.cy;
    let perp_axis_z = matrix_a.cz;

    let soft = make_soft(
        get(joints, slot, PJ_HERTZ),
        get(joints, slot, PJ_DAMPING_RATIO),
        h,
    );

    set_transform(joints, slot, PJ_FRAME_A, frame_a);
    set_transform(joints, slot, PJ_FRAME_B, frame_b);
    set_vec3(joints, slot, PJ_JOINT_AXIS, joint_axis);
    set_vec3(joints, slot, PJ_PERP_AXIS_Y, perp_axis_y);
    set_vec3(joints, slot, PJ_PERP_AXIS_Z, perp_axis_z);
    set_vec3(joints, slot, PJ_DELTA_CENTER, delta_center);
    set_mat3(joints, slot, PJ_ROTATION_MASS, rotation_mass);
    write_softness(joints, slot, PJ_SPRING_SOFTNESS, soft);
    set(joints, slot, PJ_FIXED_ROTATION, if fixed_rotation { 1.0 } else { 0.0 });

    if !enable_warm_starting {
        set(joints, slot, PJ_PERP_IMPULSE, 0.0);
        set(joints, slot, PJ_PERP_IMPULSE + 1, 0.0);
        set_vec3(joints, slot, PJ_ANGULAR_IMPULSE, Vec3::ZERO);
        set(joints, slot, PJ_MOTOR_IMPULSE, 0.0);
        set(joints, slot, PJ_SPRING_IMPULSE, 0.0);
        set(joints, slot, PJ_LOWER_IMPULSE, 0.0);
        set(joints, slot, PJ_UPPER_IMPULSE, 0.0);
    }
}

/// b3WarmStartPrismaticJoint.
fn warm_start_prismatic(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let frame_a = get_transform(joints, slot, PJ_FRAME_A);
    let frame_b = get_transform(joints, slot, PJ_FRAME_B);
    let delta_center = get_vec3(joints, slot, PJ_DELTA_CENTER);
    let joint_axis0 = get_vec3(joints, slot, PJ_JOINT_AXIS);
    let perp_axis_y0 = get_vec3(joints, slot, PJ_PERP_AXIS_Y);
    let perp_axis_z0 = get_vec3(joints, slot, PJ_PERP_AXIS_Z);
    let perp_x = get(joints, slot, PJ_PERP_IMPULSE);
    let perp_y = get(joints, slot, PJ_PERP_IMPULSE + 1);
    let angular_impulse = get_vec3(joints, slot, PJ_ANGULAR_IMPULSE);
    let spring_impulse = get(joints, slot, PJ_SPRING_IMPULSE);
    let motor_impulse = get(joints, slot, PJ_MOTOR_IMPULSE);
    let lower_impulse = get(joints, slot, PJ_LOWER_IMPULSE);
    let upper_impulse = get(joints, slot, PJ_UPPER_IMPULSE);

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);
    let d = end_b
        .state
        .delta_position
        .sub(end_a.state.delta_position)
        .add(delta_center)
        .add(r_b.sub(r_a));
    let joint_axis = end_a.state.delta_rotation.rotate(joint_axis0);
    let s_ax = r_a.add(d).cross(joint_axis);
    let s_bx = r_b.cross(joint_axis);

    let perp_y_axis = end_a.state.delta_rotation.rotate(perp_axis_y0);
    let perp_z_axis = end_a.state.delta_rotation.rotate(perp_axis_z0);
    let s_ay = r_a.add(d).cross(perp_y_axis);
    let s_by = r_b.cross(perp_y_axis);
    let s_az = r_a.add(d).cross(perp_z_axis);
    let s_bz = r_b.cross(perp_z_axis);

    let axial_impulse = ((spring_impulse + motor_impulse) + lower_impulse) - upper_impulse;

    let p = blend3(axial_impulse, joint_axis, perp_x, perp_y_axis, perp_y, perp_z_axis);
    let l_a = blend3(axial_impulse, s_ax, perp_x, s_ay, perp_y, s_az).add(angular_impulse);
    let l_b = blend3(axial_impulse, s_bx, perp_x, s_by, perp_y, s_bz).add(angular_impulse);

    let v_a = end_a.state.linear_velocity.mul_sub(m_a, p);
    let w_a = end_a.state.angular_velocity.sub(i_a.mul_v(l_a));
    let v_b = end_b.state.linear_velocity.mul_add(m_b, p);
    let w_b = end_b.state.angular_velocity.add(i_b.mul_v(l_b));

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

/// b3SolvePrismaticJoint.
#[allow(clippy::too_many_lines)]
fn solve_prismatic(
    joints: Col<f32>,
    slot: usize,
    state_col: Col<f32>,
    flags_col: Col<u32>,
    use_bias: bool,
    h: f32,
    inv_h: f32,
) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);
    let mut v_a = end_a.state.linear_velocity;
    let mut w_a = end_a.state.angular_velocity;
    let mut v_b = end_b.state.linear_velocity;
    let mut w_b = end_b.state.angular_velocity;

    let frame_a = get_transform(joints, slot, PJ_FRAME_A);
    let frame_b = get_transform(joints, slot, PJ_FRAME_B);
    let delta_center = get_vec3(joints, slot, PJ_DELTA_CENTER);
    let joint_axis0 = get_vec3(joints, slot, PJ_JOINT_AXIS);
    let perp_axis_y0 = get_vec3(joints, slot, PJ_PERP_AXIS_Y);
    let perp_axis_z0 = get_vec3(joints, slot, PJ_PERP_AXIS_Z);
    let rotation_mass = get_mat3(joints, slot, PJ_ROTATION_MASS);
    let spring_soft = read_softness(joints, slot, PJ_SPRING_SOFTNESS);
    let cs = read_softness(joints, slot, J_CONSTRAINT_SOFTNESS);
    let fixed_rotation = get(joints, slot, PJ_FIXED_ROTATION) != 0.0;
    let target_translation = get(joints, slot, PJ_TARGET_TRANSLATION);
    let motor_speed = get(joints, slot, PJ_MOTOR_SPEED);
    let max_motor_force = get(joints, slot, PJ_MAX_MOTOR_FORCE);
    let lower_translation = get(joints, slot, PJ_LOWER_TRANSLATION);
    let upper_translation = get(joints, slot, PJ_UPPER_TRANSLATION);
    let enable = get(joints, slot, PJ_ENABLE).to_bits();
    let enable_spring = enable & PJ_ENABLE_SPRING != 0;
    let enable_motor = enable & PJ_ENABLE_MOTOR != 0;
    let enable_limit = enable & PJ_ENABLE_LIMIT != 0;

    let mut perp_impulse = Vec2::new(
        get(joints, slot, PJ_PERP_IMPULSE),
        get(joints, slot, PJ_PERP_IMPULSE + 1),
    );
    let mut angular_impulse = get_vec3(joints, slot, PJ_ANGULAR_IMPULSE);
    let mut spring_impulse = get(joints, slot, PJ_SPRING_IMPULSE);
    let mut motor_impulse = get(joints, slot, PJ_MOTOR_IMPULSE);
    let mut lower_impulse = get(joints, slot, PJ_LOWER_IMPULSE);
    let mut upper_impulse = get(joints, slot, PJ_UPPER_IMPULSE);

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

    let d = end_b
        .state
        .delta_position
        .sub(end_a.state.delta_position)
        .add(delta_center)
        .add(r_b.sub(r_a));

    let joint_axis = end_a.state.delta_rotation.rotate(joint_axis0);
    let s_ax = r_a.add(d).cross(joint_axis);
    let s_bx = r_b.cross(joint_axis);
    let joint_translation = d.dot(joint_axis);

    // The axial effective mass must be fresh to avoid divergence when the joint is stressed.
    let ka = ((m_a + m_b) + s_ax.dot(i_a.mul_v(s_ax))) + s_bx.dot(i_b.mul_v(s_bx));
    let axial_mass = if ka > 0.0 { 1.0 / ka } else { 0.0 };

    // Solve spring
    if enable_spring && !fixed_rotation {
        let c = joint_translation - target_translation;
        let bias = spring_soft.bias_rate * c;
        let mass_scale = spring_soft.mass_scale;
        let impulse_scale = spring_soft.impulse_scale;

        let v_rel = v_b
            .add(w_b.cross(r_b))
            .sub(v_a)
            .sub(w_a.cross(r_a.add(d)));
        let cdot = v_rel.dot(joint_axis);
        let delta_impulse = -mass_scale * axial_mass * (cdot + bias) - impulse_scale * spring_impulse;
        spring_impulse += delta_impulse;

        let p = joint_axis.scale(delta_impulse);
        let l_a = s_ax.scale(delta_impulse);
        let l_b = s_bx.scale(delta_impulse);
        v_a = v_a.mul_sub(m_a, p);
        w_a = w_a.sub(i_a.mul_v(l_a));
        v_b = v_b.mul_add(m_b, p);
        w_b = w_b.add(i_b.mul_v(l_b));
    }

    if enable_motor && !fixed_rotation {
        let v_rel = v_b
            .add(w_b.cross(r_b))
            .sub(v_a)
            .sub(w_a.cross(r_a.add(d)));
        let cdot = v_rel.dot(joint_axis) - motor_speed;

        let mut delta_impulse = -axial_mass * cdot;
        let mut new_impulse = motor_impulse + delta_impulse;
        let max_impulse = max_motor_force * h;
        new_impulse = clampf(new_impulse, -max_impulse, max_impulse);
        delta_impulse = new_impulse - motor_impulse;
        motor_impulse = new_impulse;

        let p = joint_axis.scale(delta_impulse);
        let l_a = s_ax.scale(delta_impulse);
        let l_b = s_bx.scale(delta_impulse);
        v_a = v_a.mul_sub(m_a, p);
        w_a = w_a.sub(i_a.mul_v(l_a));
        v_b = v_b.mul_add(m_b, p);
        w_b = w_b.add(i_b.mul_v(l_b));
    }

    if enable_limit && !fixed_rotation {
        let speculative_distance = 0.25 * (upper_translation - lower_translation);

        // Lower limit
        {
            let c = joint_translation - lower_translation;
            if c < speculative_distance {
                let mut bias = 0.0;
                let mut mass_scale = 1.0_f32;
                let mut impulse_scale = 0.0;
                if c > 0.0 {
                    bias = c * inv_h;
                } else if use_bias {
                    bias = cs.bias_rate * c;
                    mass_scale = cs.mass_scale;
                    impulse_scale = cs.impulse_scale;
                }
                let v_rel = v_b
                    .add(w_b.cross(r_b))
                    .sub(v_a)
                    .sub(w_a.cross(r_a.add(d)));
                let cdot = v_rel.dot(joint_axis);
                let old_impulse = lower_impulse;
                let mut delta_impulse =
                    -mass_scale * axial_mass * (cdot + bias) - impulse_scale * old_impulse;
                lower_impulse = maxf(old_impulse + delta_impulse, 0.0);
                delta_impulse = lower_impulse - old_impulse;

                let p = joint_axis.scale(delta_impulse);
                let l_a = s_ax.scale(delta_impulse);
                let l_b = s_bx.scale(delta_impulse);
                v_a = v_a.mul_sub(m_a, p);
                w_a = w_a.sub(i_a.mul_v(l_a));
                v_b = v_b.mul_add(m_b, p);
                w_b = w_b.add(i_b.mul_v(l_b));
            } else {
                lower_impulse = 0.0;
            }
        }

        // Upper limit
        {
            let c = upper_translation - joint_translation;
            if c < speculative_distance {
                let mut bias = 0.0;
                let mut mass_scale = 1.0_f32;
                let mut impulse_scale = 0.0;
                if c > 0.0 {
                    bias = c * inv_h;
                } else if use_bias {
                    bias = cs.bias_rate * c;
                    mass_scale = cs.mass_scale;
                    impulse_scale = cs.impulse_scale;
                }
                // sign flipped on Cdot
                let v_rel = v_b
                    .add(w_b.cross(r_b))
                    .sub(v_a)
                    .sub(w_a.cross(r_a.add(d)));
                let cdot = -v_rel.dot(joint_axis);
                let old_impulse = upper_impulse;
                let delta_impulse =
                    -mass_scale * axial_mass * (cdot + bias) - impulse_scale * old_impulse;
                upper_impulse = maxf(old_impulse + delta_impulse, 0.0);

                // sign flipped on applied impulse
                let neg_delta_impulse = old_impulse - upper_impulse;
                let p = joint_axis.scale(neg_delta_impulse);
                let l_a = s_ax.scale(neg_delta_impulse);
                let l_b = s_bx.scale(neg_delta_impulse);
                v_a = v_a.mul_sub(m_a, p);
                w_a = w_a.sub(i_a.mul_v(l_a));
                v_b = v_b.mul_add(m_b, p);
                w_b = w_b.add(i_b.mul_v(l_b));
            } else {
                upper_impulse = 0.0;
            }
        }
    }

    // Rotation constraint
    if !fixed_rotation {
        let mut bias = Vec3::ZERO;
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias {
            let quat_a = end_a.state.delta_rotation.mul(frame_a.q);
            let quat_b = end_b.state.delta_rotation.mul(frame_b.q);
            let rel_q = quat_a.inv_mul(quat_b);
            let delta_rotation = rel_q.delta_to_rotation(Quat::IDENTITY);
            let c = quat_a.rotate(delta_rotation).neg();
            bias = c.scale(cs.bias_rate);
            mass_scale = cs.mass_scale;
            impulse_scale = cs.impulse_scale;
        }

        let cdot = w_b.sub(w_a);
        let impulse = rotation_mass
            .mul_v(cdot.add(bias))
            .scale(-mass_scale)
            .sub(angular_impulse.scale(impulse_scale));
        angular_impulse = angular_impulse.add(impulse);

        w_a = w_a.sub(i_a.mul_v(impulse));
        w_b = w_b.add(i_b.mul_v(impulse));
    }

    // Solve point-to-line constraint
    {
        let perp_y_axis = end_a.state.delta_rotation.rotate(perp_axis_y0);
        let perp_z_axis = end_a.state.delta_rotation.rotate(perp_axis_z0);

        let mut bias = Vec2::new(0.0, 0.0);
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias {
            let c = Vec2::new(perp_y_axis.dot(d), perp_z_axis.dot(d));
            bias = c.scale(cs.bias_rate);
            mass_scale = cs.mass_scale;
            impulse_scale = cs.impulse_scale;
        }

        let v_rel = v_b
            .add(w_b.cross(r_b))
            .sub(v_a)
            .sub(w_a.cross(r_a.add(d)));
        let cdot = Vec2::new(perp_y_axis.dot(v_rel), perp_z_axis.dot(v_rel));

        let s_ay = r_a.add(d).cross(perp_y_axis);
        let s_by = r_b.cross(perp_y_axis);
        let s_az = r_a.add(d).cross(perp_z_axis);
        let s_bz = r_b.cross(perp_z_axis);

        let kyy = ((m_a + m_b) + s_ay.dot(i_a.mul_v(s_ay))) + s_by.dot(i_b.mul_v(s_by));
        let kyz = s_ay.dot(i_a.mul_v(s_az)) + s_by.dot(i_b.mul_v(s_bz));
        let kzz = ((m_a + m_b) + s_az.dot(i_a.mul_v(s_az))) + s_bz.dot(i_b.mul_v(s_bz));

        let k = Mat2 {
            cx: Vec2::new(kyy, kyz),
            cy: Vec2::new(kyz, kzz),
        };

        let old_impulse = perp_impulse;
        let sol = k.solve(cdot.add(bias));
        let delta_impulse = sol.scale(-mass_scale).sub(old_impulse.scale(impulse_scale));
        perp_impulse = old_impulse.add(delta_impulse);

        let p = blend2(delta_impulse.x, perp_y_axis, delta_impulse.y, perp_z_axis);
        v_a = v_a.mul_sub(m_a, p);
        w_a = w_a.sub(i_a.mul_v(blend2(delta_impulse.x, s_ay, delta_impulse.y, s_az)));
        v_b = v_b.mul_add(m_b, p);
        w_b = w_b.add(i_b.mul_v(blend2(delta_impulse.x, s_by, delta_impulse.y, s_bz)));
    }

    set(joints, slot, PJ_PERP_IMPULSE, perp_impulse.x);
    set(joints, slot, PJ_PERP_IMPULSE + 1, perp_impulse.y);
    set_vec3(joints, slot, PJ_ANGULAR_IMPULSE, angular_impulse);
    set(joints, slot, PJ_SPRING_IMPULSE, spring_impulse);
    set(joints, slot, PJ_MOTOR_IMPULSE, motor_impulse);
    set(joints, slot, PJ_LOWER_IMPULSE, lower_impulse);
    set(joints, slot, PJ_UPPER_IMPULSE, upper_impulse);

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

// --- wheel joint ------------------------------------------------------------------------------

/// The steering constraint axis (twist about body A's x), `b3SolveWheelJoint`'s inline block. Pure f32
/// dots, so every recompute is bit-identical.
fn steering_axis(matrix_a: Mat3, matrix_b: Mat3) -> Vec3 {
    let cs = matrix_b.cz.dot(matrix_a.cz);
    let ss = -matrix_b.cz.dot(matrix_a.cy);
    let mut den = cs * cs + ss * ss;
    den = if den > 0.0 { 1.0 / den } else { 0.0 };
    matrix_b
        .cz
        .cross(matrix_a.cy.scale(-cs).sub(matrix_a.cz.scale(ss)))
        .scale(den)
}

/// b3PrepareWheelJoint (`src/wheelJoint.ts`). All three effective masses + both softnesses are written
/// unconditionally, so no scratch needs zeroing when a feature is off.
fn prepare_wheel(joints: Col<f32>, slot: usize, h: f32, enable_warm_starting: bool) {
    let base = read_base(joints, slot);
    let pose = read_pose(joints, slot);

    let frame_a = Transform {
        q: pose.qa.mul(base.local_frame_a.q),
        p: pose.qa.rotate(base.local_frame_a.p.sub(pose.local_center_a)),
    };
    let frame_b = Transform {
        q: pose.qb.mul(base.local_frame_b.q),
        p: pose.qb.rotate(base.local_frame_b.p.sub(pose.local_center_b)),
    };
    let delta_center = pose.center_b.sub(pose.center_a);

    let inv_inertia_sum = base.inv_ia.add(base.inv_ib);
    let fixed_rotation = inv_inertia_sum.det() < 1000.0 * FLT_MIN;

    let r_a = frame_a.p;
    let r_b = frame_b.p;
    let matrix_a = Mat3::from_quat(frame_a.q);
    let matrix_b = Mat3::from_quat(frame_b.q);

    let suspension_axis = matrix_a.cx;
    let r_an = r_a.cross(suspension_axis);
    let r_bn = r_b.cross(suspension_axis);
    let ks = base.inv_mass_a + base.inv_mass_b
        + r_an.dot(base.inv_ia.mul_v(r_an))
        + r_bn.dot(base.inv_ib.mul_v(r_bn));
    let suspension_mass = if ks > 0.0 { 1.0 / ks } else { 0.0 };

    let suspension_soft = make_soft(
        get(joints, slot, WHJ_SUSPENSION_HERTZ),
        get(joints, slot, WHJ_SUSPENSION_DAMPING_RATIO),
        h,
    );
    let steering_soft = make_soft(
        get(joints, slot, WHJ_STEERING_HERTZ),
        get(joints, slot, WHJ_STEERING_DAMPING_RATIO),
        h,
    );

    let spin_axis = matrix_b.cz;
    let kspin = spin_axis.dot(inv_inertia_sum.mul_v(spin_axis));
    let spin_mass = if kspin > 0.0 { 1.0 / kspin } else { 0.0 };

    let s_axis = steering_axis(matrix_a, matrix_b);
    let ksteer = s_axis.dot(inv_inertia_sum.mul_v(s_axis));
    let steering_mass = if ksteer > 0.0 { 1.0 / ksteer } else { 0.0 };

    set_transform(joints, slot, WHJ_FRAME_A, frame_a);
    set_transform(joints, slot, WHJ_FRAME_B, frame_b);
    set_vec3(joints, slot, WHJ_DELTA_CENTER, delta_center);
    set(joints, slot, WHJ_SPIN_MASS, spin_mass);
    set(joints, slot, WHJ_SUSPENSION_MASS, suspension_mass);
    set(joints, slot, WHJ_STEERING_MASS, steering_mass);
    write_softness(joints, slot, WHJ_SUSPENSION_SOFTNESS, suspension_soft);
    write_softness(joints, slot, WHJ_STEERING_SOFTNESS, steering_soft);
    set(joints, slot, WHJ_FIXED_ROTATION, if fixed_rotation { 1.0 } else { 0.0 });

    if !enable_warm_starting {
        set(joints, slot, WHJ_LINEAR_IMPULSE, 0.0);
        set(joints, slot, WHJ_LINEAR_IMPULSE + 1, 0.0);
        set(joints, slot, WHJ_ANGULAR_IMPULSE, 0.0);
        set(joints, slot, WHJ_ANGULAR_IMPULSE + 1, 0.0);
        set(joints, slot, WHJ_SPIN_IMPULSE, 0.0);
        set(joints, slot, WHJ_SUSPENSION_SPRING_IMPULSE, 0.0);
        set(joints, slot, WHJ_LOWER_SUSPENSION_IMPULSE, 0.0);
        set(joints, slot, WHJ_UPPER_SUSPENSION_IMPULSE, 0.0);
        set(joints, slot, WHJ_STEERING_SPRING_IMPULSE, 0.0);
        set(joints, slot, WHJ_LOWER_STEERING_IMPULSE, 0.0);
        set(joints, slot, WHJ_UPPER_STEERING_IMPULSE, 0.0);
    }
}

/// b3WarmStartWheelJoint.
#[allow(clippy::too_many_lines)]
fn warm_start_wheel(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let frame_a = get_transform(joints, slot, WHJ_FRAME_A);
    let frame_b = get_transform(joints, slot, WHJ_FRAME_B);
    let delta_center = get_vec3(joints, slot, WHJ_DELTA_CENTER);

    let linear_impulse_y = get(joints, slot, WHJ_LINEAR_IMPULSE);
    let linear_impulse_z = get(joints, slot, WHJ_LINEAR_IMPULSE + 1);
    let angular_impulse_x = get(joints, slot, WHJ_ANGULAR_IMPULSE);
    let angular_impulse_y = get(joints, slot, WHJ_ANGULAR_IMPULSE + 1);
    let spin_impulse = get(joints, slot, WHJ_SPIN_IMPULSE);
    let suspension_spring_impulse = get(joints, slot, WHJ_SUSPENSION_SPRING_IMPULSE);
    let lower_suspension_impulse = get(joints, slot, WHJ_LOWER_SUSPENSION_IMPULSE);
    let upper_suspension_impulse = get(joints, slot, WHJ_UPPER_SUSPENSION_IMPULSE);
    let steering_spring_impulse = get(joints, slot, WHJ_STEERING_SPRING_IMPULSE);
    let lower_steering_impulse = get(joints, slot, WHJ_LOWER_STEERING_IMPULSE);
    let upper_steering_impulse = get(joints, slot, WHJ_UPPER_STEERING_IMPULSE);
    let enable = get(joints, slot, WHJ_ENABLE).to_bits();
    let enable_steering = enable & WHJ_ENABLE_STEERING != 0;

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);
    let d = end_b
        .state
        .delta_position
        .sub(end_a.state.delta_position)
        .add(delta_center)
        .add(r_b.sub(r_a));

    let quat_a = end_a.state.delta_rotation.mul(frame_a.q);
    let mut quat_b = end_b.state.delta_rotation.mul(frame_b.q);
    if quat_a.dot(quat_b) < 0.0 {
        quat_b = quat_b.negate();
    }
    let matrix_a = Mat3::from_quat(quat_a);
    let matrix_b = Mat3::from_quat(quat_b);

    let s_ax = d.add(r_a).cross(matrix_a.cx);
    let s_bx = r_b.cross(matrix_a.cx);
    let s_ay = d.add(r_a).cross(matrix_a.cy);
    let s_by = r_b.cross(matrix_a.cy);
    let s_az = d.add(r_a).cross(matrix_a.cz);
    let s_bz = r_b.cross(matrix_a.cz);

    let suspension_impulse =
        (suspension_spring_impulse + lower_suspension_impulse) - upper_suspension_impulse;

    let linear_impulse = blend3(
        suspension_impulse,
        matrix_a.cx,
        linear_impulse_y,
        matrix_a.cy,
        linear_impulse_z,
        matrix_a.cz,
    );
    let angular_impulse_a = blend3(
        suspension_impulse,
        s_ax,
        linear_impulse_y,
        s_ay,
        linear_impulse_z,
        s_az,
    );
    let angular_impulse_b = blend3(
        suspension_impulse,
        s_bx,
        linear_impulse_y,
        s_by,
        linear_impulse_z,
        s_bz,
    );
    let mut angular_impulse = matrix_a.cz.scale(spin_impulse);

    let spin_axis = matrix_b.cz;

    if enable_steering {
        let s_axis = steering_axis(matrix_a, matrix_b);
        let perp_axis = spin_axis.cross(matrix_a.cx);
        let steering_impulse =
            (steering_spring_impulse + lower_steering_impulse) - upper_steering_impulse;
        angular_impulse = blend3(
            angular_impulse_x,
            perp_axis,
            spin_impulse,
            spin_axis,
            steering_impulse,
            s_axis,
        );
    } else {
        let rel_q = quat_a.inv_mul(quat_b);
        let axis_x = Vec3::new(1.0, 0.0, 0.0);
        let axis_y = Vec3::new(0.0, 1.0, 0.0);
        let perp_axis_x = quat_a
            .rotate(axis_x.scale(rel_q.s).add(rel_q.v.cross(axis_x)))
            .scale(0.5);
        let perp_axis_y = quat_a
            .rotate(axis_y.scale(rel_q.s).add(rel_q.v.cross(axis_y)))
            .scale(0.5);
        angular_impulse = angular_impulse.add(blend3(
            angular_impulse_x,
            perp_axis_x,
            angular_impulse_y,
            perp_axis_y,
            spin_impulse,
            spin_axis,
        ));
    }

    if end_a.dynamic {
        let v_a = end_a.state.linear_velocity.mul_sub(m_a, linear_impulse);
        let w_a = end_a
            .state
            .angular_velocity
            .sub(i_a.mul_v(angular_impulse_a.add(angular_impulse)));
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        let v_b = end_b.state.linear_velocity.mul_add(m_b, linear_impulse);
        let w_b = end_b
            .state
            .angular_velocity
            .add(i_b.mul_v(angular_impulse_b.add(angular_impulse)));
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

/// b3SolveWheelJoint.
#[allow(clippy::too_many_lines)]
fn solve_wheel(
    joints: Col<f32>,
    slot: usize,
    state_col: Col<f32>,
    flags_col: Col<u32>,
    use_bias: bool,
    h: f32,
    inv_h: f32,
) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);
    let mut v_a = end_a.state.linear_velocity;
    let mut w_a = end_a.state.angular_velocity;
    let mut v_b = end_b.state.linear_velocity;
    let mut w_b = end_b.state.angular_velocity;

    let frame_a = get_transform(joints, slot, WHJ_FRAME_A);
    let frame_b = get_transform(joints, slot, WHJ_FRAME_B);
    let delta_center = get_vec3(joints, slot, WHJ_DELTA_CENTER);
    let spin_mass = get(joints, slot, WHJ_SPIN_MASS);
    let suspension_mass = get(joints, slot, WHJ_SUSPENSION_MASS);
    let steering_mass = get(joints, slot, WHJ_STEERING_MASS);
    let suspension_soft = read_softness(joints, slot, WHJ_SUSPENSION_SOFTNESS);
    let steering_soft = read_softness(joints, slot, WHJ_STEERING_SOFTNESS);
    let cs = read_softness(joints, slot, J_CONSTRAINT_SOFTNESS);
    let fixed_rotation = get(joints, slot, WHJ_FIXED_ROTATION) != 0.0;

    let spin_speed = get(joints, slot, WHJ_SPIN_SPEED);
    let max_spin_torque = get(joints, slot, WHJ_MAX_SPIN_TORQUE);
    let lower_suspension_limit = get(joints, slot, WHJ_LOWER_SUSPENSION_LIMIT);
    let upper_suspension_limit = get(joints, slot, WHJ_UPPER_SUSPENSION_LIMIT);
    let target_steering_angle = get(joints, slot, WHJ_TARGET_STEERING_ANGLE);
    let max_steering_torque = get(joints, slot, WHJ_MAX_STEERING_TORQUE);
    let lower_steering_limit = get(joints, slot, WHJ_LOWER_STEERING_LIMIT);
    let upper_steering_limit = get(joints, slot, WHJ_UPPER_STEERING_LIMIT);
    let enable = get(joints, slot, WHJ_ENABLE).to_bits();
    let enable_spin_motor = enable & WHJ_ENABLE_SPIN_MOTOR != 0;
    let enable_suspension_spring = enable & WHJ_ENABLE_SUSPENSION_SPRING != 0;
    let enable_suspension_limit = enable & WHJ_ENABLE_SUSPENSION_LIMIT != 0;
    let enable_steering = enable & WHJ_ENABLE_STEERING != 0;
    let enable_steering_limit = enable & WHJ_ENABLE_STEERING_LIMIT != 0;

    let mut linear_impulse = Vec2::new(
        get(joints, slot, WHJ_LINEAR_IMPULSE),
        get(joints, slot, WHJ_LINEAR_IMPULSE + 1),
    );
    let mut angular_impulse = Vec2::new(
        get(joints, slot, WHJ_ANGULAR_IMPULSE),
        get(joints, slot, WHJ_ANGULAR_IMPULSE + 1),
    );
    let mut spin_impulse = get(joints, slot, WHJ_SPIN_IMPULSE);
    let mut suspension_spring_impulse = get(joints, slot, WHJ_SUSPENSION_SPRING_IMPULSE);
    let mut lower_suspension_impulse = get(joints, slot, WHJ_LOWER_SUSPENSION_IMPULSE);
    let mut upper_suspension_impulse = get(joints, slot, WHJ_UPPER_SUSPENSION_IMPULSE);
    let mut steering_spring_impulse = get(joints, slot, WHJ_STEERING_SPRING_IMPULSE);
    let mut lower_steering_impulse = get(joints, slot, WHJ_LOWER_STEERING_IMPULSE);
    let mut upper_steering_impulse = get(joints, slot, WHJ_UPPER_STEERING_IMPULSE);

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

    let quat_a = end_a.state.delta_rotation.mul(frame_a.q);
    let mut quat_b = end_b.state.delta_rotation.mul(frame_b.q);
    if quat_a.dot(quat_b) < 0.0 {
        quat_b = quat_b.negate();
    }
    let rel_q = quat_a.inv_mul(quat_b);
    let matrix_a = Mat3::from_quat(quat_a);
    let matrix_b = Mat3::from_quat(quat_b);

    let d = end_b
        .state
        .delta_position
        .sub(end_a.state.delta_position)
        .add(delta_center)
        .add(r_b.sub(r_a));
    let s_ax = d.add(r_a).cross(matrix_a.cx);
    let s_bx = r_b.cross(matrix_a.cx);
    let s_ay = d.add(r_a).cross(matrix_a.cy);
    let s_by = r_b.cross(matrix_a.cy);
    let s_az = d.add(r_a).cross(matrix_a.cz);
    let s_bz = r_b.cross(matrix_a.cz);

    let translation = matrix_a.cx.dot(d);

    let cs_dot = matrix_b.cz.dot(matrix_a.cz);
    let ss = -matrix_b.cz.dot(matrix_a.cy);
    let mut den = cs_dot * cs_dot + ss * ss;
    den = if den > 0.0 { 1.0 / den } else { 0.0 };
    let steer_axis = matrix_b
        .cz
        .cross(matrix_a.cy.scale(-cs_dot).sub(matrix_a.cz.scale(ss)))
        .scale(den);

    // motor constraint
    if enable_spin_motor && !fixed_rotation {
        let spin_axis = matrix_b.cz;
        let cdot = w_b.sub(w_a).dot(spin_axis) - spin_speed;
        let mut impulse = -spin_mass * cdot;
        let old_impulse = spin_impulse;
        let max_impulse = h * max_spin_torque;
        spin_impulse = clampf(spin_impulse + impulse, -max_impulse, max_impulse);
        impulse = spin_impulse - old_impulse;
        w_a = w_a.sub(i_a.mul_v(spin_axis.scale(impulse)));
        w_b = w_b.add(i_b.mul_v(spin_axis.scale(impulse)));
    }

    // suspension spring (a real spring — applied even during relax, no fixedRotation guard)
    if enable_suspension_spring {
        let c = translation;
        let bias = suspension_soft.bias_rate * c;
        let mass_scale = suspension_soft.mass_scale;
        let impulse_scale = suspension_soft.impulse_scale;

        let cdot = matrix_a.cx.dot(v_b.sub(v_a)) + s_bx.dot(w_b) - s_ax.dot(w_a);
        let impulse = -mass_scale * suspension_mass * (cdot + bias)
            - impulse_scale * suspension_spring_impulse;
        suspension_spring_impulse += impulse;

        let linear = matrix_a.cx.scale(impulse);
        let angular_a = s_ax.scale(impulse);
        let angular_b = s_bx.scale(impulse);
        v_a = v_a.mul_sub(m_a, linear);
        w_a = w_a.sub(i_a.mul_v(angular_a));
        v_b = v_b.mul_add(m_b, linear);
        w_b = w_b.add(i_b.mul_v(angular_b));
    }

    // steering
    if enable_steering && !fixed_rotation {
        let steering_angle = atan2(ss, cs_dot);

        {
            // spring — real spring, applied during relax too
            let c = steering_angle - target_steering_angle;
            let bias = steering_soft.bias_rate * c;
            let mass_scale = steering_soft.mass_scale;
            let impulse_scale = steering_soft.impulse_scale;

            let cdot = steer_axis.dot(w_b.sub(w_a));
            let old_impulse = steering_spring_impulse;
            let mut impulse =
                -mass_scale * steering_mass * (cdot + bias) - impulse_scale * old_impulse;
            let max_impulse = h * max_steering_torque;
            steering_spring_impulse = clampf(old_impulse + impulse, -max_impulse, max_impulse);
            impulse = steering_spring_impulse - old_impulse;
            w_a = w_a.sub(i_a.mul_v(steer_axis.scale(impulse)));
            w_b = w_b.add(i_b.mul_v(steer_axis.scale(impulse)));
        }

        if enable_steering_limit {
            // Lower limit
            {
                let c = steering_angle - lower_steering_limit;
                let mut bias = 0.0;
                let mut mass_scale = 1.0_f32;
                let mut impulse_scale = 0.0;
                if c > 0.0 {
                    bias = c * inv_h;
                } else if use_bias {
                    bias = cs.bias_rate * c;
                    mass_scale = cs.mass_scale;
                    impulse_scale = cs.impulse_scale;
                }
                let cdot = steer_axis.dot(w_b.sub(w_a));
                let old_impulse = lower_steering_impulse;
                let mut impulse =
                    -mass_scale * steering_mass * (cdot + bias) - impulse_scale * old_impulse;
                lower_steering_impulse = maxf(old_impulse + impulse, 0.0);
                impulse = lower_steering_impulse - old_impulse;
                w_a = w_a.sub(i_a.mul_v(steer_axis.scale(impulse)));
                w_b = w_b.add(i_b.mul_v(steer_axis.scale(impulse)));
            }

            // Upper limit — signs flipped
            {
                let c = upper_steering_limit - steering_angle;
                let mut bias = 0.0;
                let mut mass_scale = 1.0_f32;
                let mut impulse_scale = 0.0;
                if c > 0.0 {
                    bias = c * inv_h;
                } else if use_bias {
                    bias = cs.bias_rate * c;
                    mass_scale = cs.mass_scale;
                    impulse_scale = cs.impulse_scale;
                }
                let cdot = steer_axis.dot(w_a.sub(w_b));
                let old_impulse = upper_steering_impulse;
                let mut impulse =
                    -mass_scale * steering_mass * (cdot + bias) - impulse_scale * old_impulse;
                upper_steering_impulse = maxf(old_impulse + impulse, 0.0);
                impulse = upper_steering_impulse - old_impulse;
                w_a = w_a.add(i_a.mul_v(steer_axis.scale(impulse)));
                w_b = w_b.sub(i_b.mul_v(steer_axis.scale(impulse)));
            }
        }
    }

    if enable_suspension_limit {
        // Lower limit
        {
            let c = translation - lower_suspension_limit;
            let mut bias = 0.0;
            let mut mass_scale = 1.0_f32;
            let mut impulse_scale = 0.0;
            if c > 0.0 {
                bias = c * inv_h;
            } else if use_bias {
                bias = cs.bias_rate * c;
                mass_scale = cs.mass_scale;
                impulse_scale = cs.impulse_scale;
            }
            let cdot = matrix_a.cx.dot(v_b.sub(v_a)) + s_bx.dot(w_b) - s_ax.dot(w_a);
            let mut impulse = -mass_scale * suspension_mass * (cdot + bias)
                - impulse_scale * lower_suspension_impulse;
            let old_impulse = lower_suspension_impulse;
            lower_suspension_impulse = maxf(old_impulse + impulse, 0.0);
            impulse = lower_suspension_impulse - old_impulse;

            let linear = matrix_a.cx.scale(impulse);
            let angular_a = s_ax.scale(impulse);
            let angular_b = s_bx.scale(impulse);
            v_a = v_a.mul_sub(m_a, linear);
            w_a = w_a.sub(i_a.mul_v(angular_a));
            v_b = v_b.mul_add(m_b, linear);
            w_b = w_b.add(i_b.mul_v(angular_b));
        }

        // Upper limit — signs flipped
        {
            let c = upper_suspension_limit - translation;
            let mut bias = 0.0;
            let mut mass_scale = 1.0_f32;
            let mut impulse_scale = 0.0;
            if c > 0.0 {
                bias = c * inv_h;
            } else if use_bias {
                bias = cs.bias_rate * c;
                mass_scale = cs.mass_scale;
                impulse_scale = cs.impulse_scale;
            }
            let cdot = matrix_a.cx.dot(v_a.sub(v_b)) + s_ax.dot(w_a) - s_bx.dot(w_b);
            let mut impulse = -mass_scale * suspension_mass * (cdot + bias)
                - impulse_scale * upper_suspension_impulse;
            let old_impulse = upper_suspension_impulse;
            upper_suspension_impulse = maxf(old_impulse + impulse, 0.0);
            impulse = upper_suspension_impulse - old_impulse;

            let linear = matrix_a.cx.scale(impulse);
            let angular_a = s_ax.scale(impulse);
            let angular_b = s_bx.scale(impulse);
            v_a = v_a.mul_add(m_a, linear);
            w_a = w_a.add(i_a.mul_v(angular_a));
            v_b = v_b.mul_sub(m_b, linear);
            w_b = w_b.sub(i_b.mul_v(angular_b));
        }
    }

    // Collinearity constraint
    if !fixed_rotation {
        if enable_steering {
            let mut bias = 0.0;
            let mut mass_scale = 1.0_f32;
            let mut impulse_scale = 0.0;
            if use_bias {
                let c = matrix_a.cx.dot(matrix_b.cz);
                bias = cs.bias_rate * c;
                mass_scale = cs.mass_scale;
                impulse_scale = cs.impulse_scale;
            }
            let u = matrix_b.cz.cross(matrix_a.cx);
            let cdot = w_b.sub(w_a).dot(u);
            let inv_inertia_sum = i_a.add(i_b);
            let k = u.dot(inv_inertia_sum.mul_v(u));
            let perp_mass = if k > 0.0 { 1.0 / k } else { 0.0 };

            let delta_impulse =
                -mass_scale * perp_mass * (cdot + bias) - impulse_scale * angular_impulse.x;
            angular_impulse.x += delta_impulse;

            w_a = w_a.mul_sub(delta_impulse, i_a.mul_v(u));
            w_b = w_b.mul_add(delta_impulse, i_b.mul_v(u));
        } else {
            let mut bias = Vec2::new(0.0, 0.0);
            let mut mass_scale = 1.0_f32;
            let mut impulse_scale = 0.0;
            if use_bias {
                let c = Vec2::new(rel_q.v.x, rel_q.v.y);
                bias = Vec2::new(cs.bias_rate * c.x, cs.bias_rate * c.y);
                mass_scale = cs.mass_scale;
                impulse_scale = cs.impulse_scale;
            }
            let axis_x = Vec3::new(1.0, 0.0, 0.0);
            let axis_y = Vec3::new(0.0, 1.0, 0.0);
            let perp_axis_x = quat_a
                .rotate(axis_x.scale(rel_q.s).add(rel_q.v.cross(axis_x)))
                .scale(0.5);
            let perp_axis_y = quat_a
                .rotate(axis_y.scale(rel_q.s).add(rel_q.v.cross(axis_y)))
                .scale(0.5);
            let inv_inertia_sum = i_a.add(i_b);
            let kxx = perp_axis_x.dot(inv_inertia_sum.mul_v(perp_axis_x));
            let kyy = perp_axis_y.dot(inv_inertia_sum.mul_v(perp_axis_y));
            let kxy = perp_axis_x.dot(inv_inertia_sum.mul_v(perp_axis_y));
            let k = Mat2 {
                cx: Vec2::new(kxx, kxy),
                cy: Vec2::new(kxy, kyy),
            };
            let w_rel = w_b.sub(w_a);
            let cdot = Vec2::new(w_rel.dot(perp_axis_x), w_rel.dot(perp_axis_y));
            let old_impulse = angular_impulse;
            let cdot_plus_bias = Vec2::new(cdot.x + bias.x, cdot.y + bias.y);
            let sol = k.solve(cdot_plus_bias);
            let delta_impulse = Vec2::new(
                -mass_scale * sol.x - impulse_scale * old_impulse.x,
                -mass_scale * sol.y - impulse_scale * old_impulse.y,
            );
            angular_impulse = Vec2::new(old_impulse.x + delta_impulse.x, old_impulse.y + delta_impulse.y);

            let ang = blend2(delta_impulse.x, perp_axis_x, delta_impulse.y, perp_axis_y);
            w_a = w_a.sub(i_a.mul_v(ang));
            w_b = w_b.add(i_b.mul_v(ang));
        }
    }

    // Solve point-to-line constraint
    {
        let perp_y = matrix_a.cy;
        let perp_z = matrix_a.cz;

        let mut bias = Vec2::new(0.0, 0.0);
        let mut mass_scale = 1.0_f32;
        let mut impulse_scale = 0.0;
        if use_bias {
            let c = Vec2::new(perp_y.dot(d), perp_z.dot(d));
            bias = Vec2::new(cs.bias_rate * c.x, cs.bias_rate * c.y);
            mass_scale = cs.mass_scale;
            impulse_scale = cs.impulse_scale;
        }

        let v_rel = v_b
            .add(w_b.cross(r_b))
            .sub(v_a)
            .sub(w_a.cross(r_a.add(d)));
        let cdot = Vec2::new(perp_y.dot(v_rel), perp_z.dot(v_rel));

        let kyy = ((m_a + m_b) + s_ay.dot(i_a.mul_v(s_ay))) + s_by.dot(i_b.mul_v(s_by));
        let kyz = s_ay.dot(i_a.mul_v(s_az)) + s_by.dot(i_b.mul_v(s_bz));
        let kzz = ((m_a + m_b) + s_az.dot(i_a.mul_v(s_az))) + s_bz.dot(i_b.mul_v(s_bz));
        let k = Mat2 {
            cx: Vec2::new(kyy, kyz),
            cy: Vec2::new(kyz, kzz),
        };

        let old_impulse = linear_impulse;
        let cdot_plus_bias = Vec2::new(cdot.x + bias.x, cdot.y + bias.y);
        let sol = k.solve(cdot_plus_bias);
        let delta_impulse = Vec2::new(
            -mass_scale * sol.x - impulse_scale * old_impulse.x,
            -mass_scale * sol.y - impulse_scale * old_impulse.y,
        );
        linear_impulse = Vec2::new(old_impulse.x + delta_impulse.x, old_impulse.y + delta_impulse.y);

        let linear = blend2(delta_impulse.x, perp_y, delta_impulse.y, perp_z);
        v_a = v_a.mul_sub(m_a, linear);
        w_a = w_a.sub(i_a.mul_v(blend2(delta_impulse.x, s_ay, delta_impulse.y, s_az)));
        v_b = v_b.mul_add(m_b, linear);
        w_b = w_b.add(i_b.mul_v(blend2(delta_impulse.x, s_by, delta_impulse.y, s_bz)));
    }

    set(joints, slot, WHJ_LINEAR_IMPULSE, linear_impulse.x);
    set(joints, slot, WHJ_LINEAR_IMPULSE + 1, linear_impulse.y);
    set(joints, slot, WHJ_ANGULAR_IMPULSE, angular_impulse.x);
    set(joints, slot, WHJ_ANGULAR_IMPULSE + 1, angular_impulse.y);
    set(joints, slot, WHJ_SPIN_IMPULSE, spin_impulse);
    set(joints, slot, WHJ_SUSPENSION_SPRING_IMPULSE, suspension_spring_impulse);
    set(joints, slot, WHJ_LOWER_SUSPENSION_IMPULSE, lower_suspension_impulse);
    set(joints, slot, WHJ_UPPER_SUSPENSION_IMPULSE, upper_suspension_impulse);
    set(joints, slot, WHJ_STEERING_SPRING_IMPULSE, steering_spring_impulse);
    set(joints, slot, WHJ_LOWER_STEERING_IMPULSE, lower_steering_impulse);
    set(joints, slot, WHJ_UPPER_STEERING_IMPULSE, upper_steering_impulse);

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

// --- motor joint ------------------------------------------------------------------------------

/// K = (1/mA + 1/mB) I − skew(rA) invIA skew(rA) − skew(rB) invIB skew(rB), the point-to-point
/// effective-mass matrix the motor's linear spring + linear velocity sub-solves share (`linearK`).
#[inline]
fn linear_k(m_a: f32, m_b: f32, inv_ia: Mat3, inv_ib: Mat3, r_a: Vec3, r_b: Vec3) -> Mat3 {
    let s_a = Mat3::skew(r_a);
    let s_b = Mat3::skew(r_b);
    let k_a = s_a.mul(inv_ia.mul(s_a));
    let k_b = s_b.mul(inv_ib.mul(s_b));
    let mut k = k_a.add(k_b).neg();
    let mm = m_a + m_b;
    k.cx.x += mm;
    k.cy.y += mm;
    k.cz.z += mm;
    k
}

/// b3PrepareMotorJoint (`src/motorJoint.ts` `prepareMotorJoint`). Frames are world-space relative to
/// each COM; the two springs resolve straight from the type hertz (no base-softness fallback). The C's
/// `fixedRotation` is computed but never read by the motor solve, so it is not stored.
fn prepare_motor(joints: Col<f32>, slot: usize, h: f32, enable_warm_starting: bool) {
    let base = read_base(joints, slot);
    let pose = read_pose(joints, slot);

    let frame_a = Transform {
        q: pose.qa.mul(base.local_frame_a.q),
        p: pose.qa.rotate(base.local_frame_a.p.sub(pose.local_center_a)),
    };
    let frame_b = Transform {
        q: pose.qb.mul(base.local_frame_b.q),
        p: pose.qb.rotate(base.local_frame_b.p.sub(pose.local_center_b)),
    };
    let delta_center = pose.center_b.sub(pose.center_a);

    let linear_spring = make_soft(
        get(joints, slot, MJ_LINEAR_HERTZ),
        get(joints, slot, MJ_LINEAR_DAMPING_RATIO),
        h,
    );
    let angular_spring = make_soft(
        get(joints, slot, MJ_ANGULAR_HERTZ),
        get(joints, slot, MJ_ANGULAR_DAMPING_RATIO),
        h,
    );
    let angular_mass = base.inv_ia.add(base.inv_ib).invert();

    set_transform(joints, slot, MJ_FRAME_A, frame_a);
    set_transform(joints, slot, MJ_FRAME_B, frame_b);
    set_vec3(joints, slot, MJ_DELTA_CENTER, delta_center);
    write_softness(joints, slot, MJ_LINEAR_SPRING, linear_spring);
    write_softness(joints, slot, MJ_ANGULAR_SPRING, angular_spring);
    set_mat3(joints, slot, MJ_ANGULAR_MASS, angular_mass);

    if !enable_warm_starting {
        set_vec3(joints, slot, MJ_LINEAR_VELOCITY_IMPULSE, Vec3::ZERO);
        set_vec3(joints, slot, MJ_ANGULAR_VELOCITY_IMPULSE, Vec3::ZERO);
        set_vec3(joints, slot, MJ_LINEAR_SPRING_IMPULSE, Vec3::ZERO);
        set_vec3(joints, slot, MJ_ANGULAR_SPRING_IMPULSE, Vec3::ZERO);
    }
}

/// b3WarmStartMotorJoint. The combined linear + angular impulses are applied to both ends (the C writes
/// unconditionally through the identity state; only dynamic ends are written back).
fn warm_start_motor(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let frame_a = get_transform(joints, slot, MJ_FRAME_A);
    let frame_b = get_transform(joints, slot, MJ_FRAME_B);

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

    let linear_impulse = get_vec3(joints, slot, MJ_LINEAR_VELOCITY_IMPULSE)
        .add(get_vec3(joints, slot, MJ_LINEAR_SPRING_IMPULSE));
    let angular_impulse = get_vec3(joints, slot, MJ_ANGULAR_VELOCITY_IMPULSE)
        .add(get_vec3(joints, slot, MJ_ANGULAR_SPRING_IMPULSE));

    let v_a = end_a.state.linear_velocity.mul_sub(m_a, linear_impulse);
    let w_a = end_a
        .state
        .angular_velocity
        .sub(i_a.mul_v(r_a.cross(linear_impulse).add(angular_impulse)));
    let v_b = end_b.state.linear_velocity.mul_add(m_b, linear_impulse);
    let w_b = end_b
        .state
        .angular_velocity
        .add(i_b.mul_v(r_b.cross(linear_impulse).add(angular_impulse)));

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

/// b3SolveMotorJoint (`solveMotorJoint`; takes no `useBias`). Four independent sub-solves gated on their
/// max effort: angular spring, angular velocity, linear spring, linear velocity.
fn solve_motor(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>, h: f32) {
    let base = read_base(joints, slot);
    let m_a = base.inv_mass_a;
    let m_b = base.inv_mass_b;
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let mut v_a = end_a.state.linear_velocity;
    let mut w_a = end_a.state.angular_velocity;
    let mut v_b = end_b.state.linear_velocity;
    let mut w_b = end_b.state.angular_velocity;

    let frame_a = get_transform(joints, slot, MJ_FRAME_A);
    let frame_b = get_transform(joints, slot, MJ_FRAME_B);
    let delta_center = get_vec3(joints, slot, MJ_DELTA_CENTER);
    let angular_mass = get_mat3(joints, slot, MJ_ANGULAR_MASS);
    let linear_spring = read_softness(joints, slot, MJ_LINEAR_SPRING);
    let angular_spring = read_softness(joints, slot, MJ_ANGULAR_SPRING);

    let linear_velocity = get_vec3(joints, slot, MJ_LINEAR_VELOCITY);
    let angular_velocity = get_vec3(joints, slot, MJ_ANGULAR_VELOCITY);
    let max_velocity_force = get(joints, slot, MJ_MAX_VELOCITY_FORCE);
    let max_velocity_torque = get(joints, slot, MJ_MAX_VELOCITY_TORQUE);
    let linear_hertz = get(joints, slot, MJ_LINEAR_HERTZ);
    let angular_hertz = get(joints, slot, MJ_ANGULAR_HERTZ);
    let max_spring_force = get(joints, slot, MJ_MAX_SPRING_FORCE);
    let max_spring_torque = get(joints, slot, MJ_MAX_SPRING_TORQUE);

    let quat_a = end_a.state.delta_rotation.mul(frame_a.q);
    let mut quat_b = end_b.state.delta_rotation.mul(frame_b.q);
    if quat_a.dot(quat_b) < 0.0 {
        quat_b = quat_b.negate();
    }
    let rel_q = quat_a.inv_mul(quat_b);

    // angular spring
    if max_spring_torque > 0.0 && angular_hertz > 0.0 {
        let target_quat = Quat::IDENTITY;
        let delta_rotation = rel_q.delta_to_rotation(target_quat);
        let c = quat_a.rotate(delta_rotation).neg();

        let bias = c.scale(angular_spring.bias_rate);
        let mass_scale = angular_spring.mass_scale;
        let impulse_scale = angular_spring.impulse_scale;

        let cdot = w_b.sub(w_a);
        let max_impulse = h * max_spring_torque;
        let old_impulse = get_vec3(joints, slot, MJ_ANGULAR_SPRING_IMPULSE);
        let impulse = angular_mass
            .mul_v(cdot.add(bias))
            .scale(-mass_scale)
            .mul_sub(impulse_scale, old_impulse);
        let mut new_impulse = old_impulse.add(impulse);
        if new_impulse.length_sq() > max_impulse * max_impulse {
            new_impulse = new_impulse.normalize().scale(max_impulse);
        }
        set_vec3(joints, slot, MJ_ANGULAR_SPRING_IMPULSE, new_impulse);
        let applied = new_impulse.sub(old_impulse);

        w_a = w_a.sub(i_a.mul_v(applied));
        w_b = w_b.add(i_b.mul_v(applied));
    }

    // angular velocity
    if max_velocity_torque > 0.0 {
        let cdot = w_b.sub(w_a).sub(angular_velocity);
        let impulse = angular_mass.mul_v(cdot).neg();

        let max_impulse = h * max_velocity_torque;
        let old_impulse = get_vec3(joints, slot, MJ_ANGULAR_VELOCITY_IMPULSE);
        let mut new_impulse = old_impulse.add(impulse);
        if new_impulse.length_sq() > max_impulse * max_impulse {
            new_impulse = new_impulse.normalize().scale(max_impulse);
        }
        set_vec3(joints, slot, MJ_ANGULAR_VELOCITY_IMPULSE, new_impulse);
        let applied = new_impulse.sub(old_impulse);

        w_a = w_a.sub(i_a.mul_v(applied));
        w_b = w_b.add(i_b.mul_v(applied));
    }

    let r_a = end_a.state.delta_rotation.rotate(frame_a.p);
    let r_b = end_b.state.delta_rotation.rotate(frame_b.p);

    // linear spring
    if max_spring_force > 0.0 && linear_hertz > 0.0 {
        let dc_a = end_a.state.delta_position;
        let dc_b = end_b.state.delta_position;
        let c = dc_b.sub(dc_a).add(r_b.sub(r_a)).add(delta_center);

        let bias = c.scale(linear_spring.bias_rate);
        let mass_scale = linear_spring.mass_scale;
        let impulse_scale = linear_spring.impulse_scale;

        let cdot = v_b
            .add(w_b.cross(r_b))
            .sub(v_a.add(w_a.cross(r_a)));

        let k = linear_k(m_a, m_b, base.inv_ia, base.inv_ib, r_a, r_b);
        let b = k.solve(cdot.add(bias));

        let old_impulse = get_vec3(joints, slot, MJ_LINEAR_SPRING_IMPULSE);
        let impulse = b.scale(-mass_scale).mul_sub(impulse_scale, old_impulse);
        let max_impulse = h * max_spring_force;
        let mut new_impulse = old_impulse.add(impulse);
        if new_impulse.length_sq() > max_impulse * max_impulse {
            new_impulse = new_impulse.normalize().scale(max_impulse);
        }
        set_vec3(joints, slot, MJ_LINEAR_SPRING_IMPULSE, new_impulse);
        let applied = new_impulse.sub(old_impulse);

        v_a = v_a.mul_sub(m_a, applied);
        w_a = w_a.sub(i_a.mul_v(r_a.cross(applied)));
        v_b = v_b.mul_add(m_b, applied);
        w_b = w_b.add(i_b.mul_v(r_b.cross(applied)));
    }

    // linear velocity
    if max_velocity_force > 0.0 {
        let cdot = v_b
            .add(w_b.cross(r_b))
            .sub(v_a.add(w_a.cross(r_a)))
            .sub(linear_velocity);

        let k = linear_k(m_a, m_b, base.inv_ia, base.inv_ib, r_a, r_b);
        let b = k.solve(cdot);
        let impulse = b.neg();

        let old_impulse = get_vec3(joints, slot, MJ_LINEAR_VELOCITY_IMPULSE);
        let max_impulse = h * max_velocity_force;
        let mut new_impulse = old_impulse.add(impulse);
        if new_impulse.length_sq() > max_impulse * max_impulse {
            new_impulse = new_impulse.normalize().scale(max_impulse);
        }
        set_vec3(joints, slot, MJ_LINEAR_VELOCITY_IMPULSE, new_impulse);
        let applied = new_impulse.sub(old_impulse);

        v_a = v_a.mul_sub(m_a, applied);
        w_a = w_a.sub(i_a.mul_v(r_a.cross(applied)));
        v_b = v_b.mul_add(m_b, applied);
        w_b = w_b.add(i_b.mul_v(r_b.cross(applied)));
    }

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, v_a, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, v_b, w_b);
    }
}

// --- parallel joint ---------------------------------------------------------------------------

/// The two perpendicular collinearity axes in world space, from the relative rotation `rel_q` of the
/// two joint frames — half the rotated imaginary parts (`perpAxes` in `src/parallelJoint.ts`).
#[inline]
fn perp_axes(q_a: Quat, rel_q: Quat) -> (Vec3, Vec3) {
    let axis_x = Vec3::new(1.0, 0.0, 0.0);
    let axis_y = Vec3::new(0.0, 1.0, 0.0);
    let x = q_a
        .rotate(axis_x.scale(rel_q.s).add(rel_q.v.cross(axis_x)))
        .scale(0.5);
    let y = q_a
        .rotate(axis_y.scale(rel_q.s).add(rel_q.v.cross(axis_y)))
        .scale(0.5);
    (x, y)
}

/// b3PrepareParallelJoint (`src/parallelJoint.ts` `prepareParallelJoint`). Unlike motor, `fixedRotation`
/// IS read by the solve, so it is stored.
fn prepare_parallel(joints: Col<f32>, slot: usize, h: f32, enable_warm_starting: bool) {
    let base = read_base(joints, slot);
    let pose = read_pose(joints, slot);

    let fixed_rotation = base.inv_ia.add(base.inv_ib).det() < 1000.0 * FLT_MIN;

    let quat_a = pose.qa.mul(base.local_frame_a.q);
    let quat_b = pose.qb.mul(base.local_frame_b.q);
    let rel_q = quat_a.inv_mul(quat_b);
    let (perp_x, perp_y) = perp_axes(quat_a, rel_q);

    let soft = make_soft(
        get(joints, slot, PLJ_HERTZ),
        get(joints, slot, PLJ_DAMPING_RATIO),
        h,
    );

    set_quat(joints, slot, PLJ_QUAT_A, quat_a);
    set_quat(joints, slot, PLJ_QUAT_B, quat_b);
    set_vec3(joints, slot, PLJ_PERP_AXIS_X, perp_x);
    set_vec3(joints, slot, PLJ_PERP_AXIS_Y, perp_y);
    write_softness(joints, slot, PLJ_SOFTNESS, soft);
    set(joints, slot, PLJ_FIXED_ROTATION, if fixed_rotation { 1.0 } else { 0.0 });

    if !enable_warm_starting {
        set(joints, slot, PLJ_PERP_IMPULSE, 0.0);
        set(joints, slot, PLJ_PERP_IMPULSE + 1, 0.0);
    }
}

/// b3WarmStartParallelJoint. Angular-only — the linear velocities pass through untouched.
fn warm_start_parallel(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>) {
    let base = read_base(joints, slot);
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let perp_x = get_vec3(joints, slot, PLJ_PERP_AXIS_X);
    let perp_y = get_vec3(joints, slot, PLJ_PERP_AXIS_Y);
    let perp_impulse_x = get(joints, slot, PLJ_PERP_IMPULSE);
    let perp_impulse_y = get(joints, slot, PLJ_PERP_IMPULSE + 1);

    let angular_impulse = blend2(perp_impulse_x, perp_x, perp_impulse_y, perp_y);

    let w_a = end_a.state.angular_velocity.sub(i_a.mul_v(angular_impulse));
    let w_b = end_b.state.angular_velocity.add(i_b.mul_v(angular_impulse));

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, end_a.state.linear_velocity, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, end_b.state.linear_velocity, w_b);
    }
}

/// b3SolveParallelJoint (`solveParallelJoint`). Takes no `useBias` — a pure soft constraint, solved
/// identically in the bias and relax passes.
fn solve_parallel(joints: Col<f32>, slot: usize, state_col: Col<f32>, flags_col: Col<u32>, h: f32) {
    let base = read_base(joints, slot);
    let i_a = base.inv_ia;
    let i_b = base.inv_ib;

    let end_a = read_end(state_col, flags_col, base.sim_index_a);
    let end_b = read_end(state_col, flags_col, base.sim_index_b);

    let mut w_a = end_a.state.angular_velocity;
    let mut w_b = end_b.state.angular_velocity;

    let fixed_rotation = get(joints, slot, PLJ_FIXED_ROTATION) != 0.0;
    let max_torque = get(joints, slot, PLJ_MAX_TORQUE);
    let soft = read_softness(joints, slot, PLJ_SOFTNESS);
    let joint_quat_a = get_quat(joints, slot, PLJ_QUAT_A);
    let joint_quat_b = get_quat(joints, slot, PLJ_QUAT_B);

    let quat_a = end_a.state.delta_rotation.mul(joint_quat_a);
    let mut quat_b = end_b.state.delta_rotation.mul(joint_quat_b);
    if quat_a.dot(quat_b) < 0.0 {
        quat_b = quat_b.negate();
    }
    let rel_q = quat_a.inv_mul(quat_b);

    if !fixed_rotation && max_torque > 0.0 {
        let bias = Vec2::new(soft.bias_rate * rel_q.v.x, soft.bias_rate * rel_q.v.y);
        let mass_scale = soft.mass_scale;
        let impulse_scale = soft.impulse_scale;

        let (perp_x, perp_y) = perp_axes(quat_a, rel_q);
        // Store the recomputed axes: the next substep's warm start reads them (b3SolveParallelJoint
        // writes joint->perpAxisX/Y; `src/parallelJoint.ts` mirrors it). Omitting this left warm start
        // on the stale prepared axes — the MT-vs-serial parallel-joint divergence.
        set_vec3(joints, slot, PLJ_PERP_AXIS_X, perp_x);
        set_vec3(joints, slot, PLJ_PERP_AXIS_Y, perp_y);

        let inv_inertia_sum = i_a.add(i_b);
        let kxx = perp_x.dot(inv_inertia_sum.mul_v(perp_x));
        let kyy = perp_y.dot(inv_inertia_sum.mul_v(perp_y));
        let kxy = perp_x.dot(inv_inertia_sum.mul_v(perp_y));
        let k = Mat2 {
            cx: Vec2::new(kxx, kxy),
            cy: Vec2::new(kxy, kyy),
        };

        let w_rel = w_b.sub(w_a);
        let cdot = Vec2::new(w_rel.dot(perp_x), w_rel.dot(perp_y));

        let max_impulse = h * max_torque;
        let old_impulse = Vec2::new(
            get(joints, slot, PLJ_PERP_IMPULSE),
            get(joints, slot, PLJ_PERP_IMPULSE + 1),
        );
        let cdot_plus_bias = Vec2::new(cdot.x + bias.x, cdot.y + bias.y);
        let sol = k.solve(cdot_plus_bias);
        let delta0 = Vec2::new(
            -mass_scale * sol.x - impulse_scale * old_impulse.x,
            -mass_scale * sol.y - impulse_scale * old_impulse.y,
        );
        let mut perp_impulse = Vec2::new(old_impulse.x + delta0.x, old_impulse.y + delta0.y);
        let len_sq = perp_impulse.x * perp_impulse.x + perp_impulse.y * perp_impulse.y;
        if len_sq > max_impulse * max_impulse {
            let s = max_impulse / len_sq.sqrt();
            perp_impulse = Vec2::new(s * perp_impulse.x, s * perp_impulse.y);
        }
        set(joints, slot, PLJ_PERP_IMPULSE, perp_impulse.x);
        set(joints, slot, PLJ_PERP_IMPULSE + 1, perp_impulse.y);

        let delta = perp_impulse.sub(old_impulse);
        let angular_impulse = blend2(delta.x, perp_x, delta.y, perp_y);
        w_a = w_a.sub(i_a.mul_v(angular_impulse));
        w_b = w_b.add(i_b.mul_v(angular_impulse));
    }

    if end_a.dynamic {
        write_velocity(state_col, base.sim_index_a, end_a.state.linear_velocity, w_a);
    }
    if end_b.dynamic {
        write_velocity(state_col, base.sim_index_b, end_b.state.linear_velocity, w_b);
    }
}

/// Silence the unused-`JointBase`-field lint on native builds where only some fields are read per phase.
#[allow(dead_code)]
fn _base_fields(b: &JointBase) -> (u32, u32) {
    (b.sim_index_a, b.sim_index_b)
}
