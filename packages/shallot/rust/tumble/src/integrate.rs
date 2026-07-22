//! Body integration: the two per-body phases of box3d's soft-step solver, ported op-for-op from
//! `solver.c` (b3IntegrateVelocitiesTask / b3IntegratePositionsTask) via the TS port (`solver.ts`).
//! Both read/write the shared body-state columns (`body.rs`); they are scalar per body (box3d runs
//! integrate serially even in the SIMD build — only the contact constraints go wide).
//!
//! Every arithmetic op maps one-to-one to the C, same operand order, no FMA contraction — bit-exact
//! with the `DISABLE_SIMD` + `FORCE_OVERFLOW` reference (see `math.rs`).

use crate::body::flags::{
    ALLOW_FAST_ROTATION, IS_SPEED_CAPPED, LOCK_ANGULAR_X, LOCK_ANGULAR_Y, LOCK_ANGULAR_Z,
    LOCK_LINEAR_X, LOCK_LINEAR_Y, LOCK_LINEAR_Z,
};
use crate::body::{read_sim, read_state, write_state};
use crate::col::Col;
use crate::math::{blend2, Mat3, Vec3, PI};

/// Maximum rotation per substep, a quarter turn (B3_MAX_ROTATION). `0.25` and `PI` are f32-exact so
/// the product is bit-identical to the C constant.
const MAX_ROTATION: f32 = 0.25 * PI;

/// Apply gravity, damping, and gyroscopic torque to the bodies in `[start, start+count)`
/// (b3IntegrateVelocities). Reads the sim column, reads/writes the velocity fields of the state column.
pub fn integrate_velocities(
    state_col: Col<f32>,
    sim_col: Col<f32>,
    start: usize,
    count: usize,
    gravity: Vec3,
    h: f32,
) {
    for i in start..start + count {
        let sim = read_sim(sim_col, i);
        let mut s = read_state(state_col, i);

        let mut v = s.linear_velocity;
        let mut w = s.angular_velocity;

        // Pade approximation of exponential damping: v2 = v1 / (1 + c * h).
        let linear_damping = 1.0 / (1.0 + h * sim.linear_damping);
        let angular_damping = 1.0 / (1.0 + h * sim.angular_damping);

        // Gravity scale is zero for kinematic bodies.
        let gravity_scale = if sim.inv_mass > 0.0 {
            sim.gravity_scale
        } else {
            0.0
        };

        let linear_velocity_delta = blend2(h * sim.inv_mass, sim.force, h * gravity_scale, gravity);
        v = linear_velocity_delta.mul_add(linear_damping, v);

        let angular_velocity_delta = sim.inv_inertia_world.mul_v(sim.torque).scale(h);
        w = angular_velocity_delta.mul_add(angular_damping, w);

        // Gyroscopic torque via one Newton-Raphson step in local coordinates.
        {
            let q = s.delta_rotation.mul(sim.rotation);
            let inertia_local = sim.inv_inertia_local.invert();

            let omega1 = q.inv_rotate(w);
            let mut omega2 = omega1;

            // Symmetric inertia tensor: 6 unique entries (column-major).
            let i00 = inertia_local.cx.x;
            let i01 = inertia_local.cy.x;
            let i02 = inertia_local.cz.x;
            let i11 = inertia_local.cy.y;
            let i12 = inertia_local.cz.y;
            let i22 = inertia_local.cz.z;

            // The gyro loop runs exactly one iteration (b3IntegrateVelocities `gyroIteration < 1`).
            let w1 = omega2.x;
            let w2 = omega2.y;
            let w3 = omega2.z;

            // Iw = I * omega2 (shared between residual and Jacobian).
            let iw1 = i00 * w1 + i01 * w2 + i02 * w3;
            let iw2 = i01 * w1 + i11 * w2 + i12 * w3;
            let iw3 = i02 * w1 + i12 * w2 + i22 * w3;

            // Residual: b = I*(omega2 - omega1) + h * (omega2 × I*omega2).
            let dw = omega2.sub(omega1);
            let b = Vec3::new(
                i00 * dw.x + i01 * dw.y + i02 * dw.z + h * (w2 * iw3 - w3 * iw2),
                i01 * dw.x + i11 * dw.y + i12 * dw.z + h * (w3 * iw1 - w1 * iw3),
                i02 * dw.x + i12 * dw.y + i22 * dw.z + h * (w1 * iw2 - w2 * iw1),
            );

            // Jacobian J = I + h * (skew(omega2) * I - skew(I*omega2)).
            let j = Mat3 {
                cx: Vec3::new(
                    i00 + h * (w2 * i02 - w3 * i01),
                    i01 + h * (w3 * i00 - w1 * i02 - iw3),
                    i02 + h * (w1 * i01 - w2 * i00 + iw2),
                ),
                cy: Vec3::new(
                    i01 + h * (w2 * i12 - w3 * i11 + iw3),
                    i11 + h * (w3 * i01 - w1 * i12),
                    i12 + h * (w1 * i11 - w2 * i01 - iw1),
                ),
                cz: Vec3::new(
                    i02 + h * (w2 * i22 - w3 * i12 - iw2),
                    i12 + h * (w3 * i02 - w1 * i22 + iw1),
                    i22 + h * (w1 * i12 - w2 * i02),
                ),
            };

            omega2 = omega2.sub(j.solve(b));

            w = q.rotate(omega2);
        }

        s.linear_velocity = v;
        s.angular_velocity = w;
        write_state(state_col, i, &s);
    }
}

/// Advance the bodies in `[start, start+count)` from their velocities, applying motion locks and speed
/// caps (b3IntegratePositions). Reads/writes the state column and the parallel flags column.
pub fn integrate_positions(
    state_col: Col<f32>,
    flags_col: Col<u32>,
    start: usize,
    count: usize,
    h: f32,
    max_linear_velocity: f32,
    inv_dt: f32,
) {
    let max_linear_speed = max_linear_velocity;
    let max_angular_speed = MAX_ROTATION * inv_dt;
    let max_linear_speed_squared = max_linear_speed * max_linear_speed;
    let max_angular_speed_squared = max_angular_speed * max_angular_speed;

    for i in start..start + count {
        let mut s = read_state(state_col, i);
        let mut flags = flags_col.get(i);

        // Motion locks — a constraint applied last, zeroing the locked components.
        let mut v = Vec3::new(
            if flags & LOCK_LINEAR_X != 0 {
                0.0
            } else {
                s.linear_velocity.x
            },
            if flags & LOCK_LINEAR_Y != 0 {
                0.0
            } else {
                s.linear_velocity.y
            },
            if flags & LOCK_LINEAR_Z != 0 {
                0.0
            } else {
                s.linear_velocity.z
            },
        );
        let mut w = Vec3::new(
            if flags & LOCK_ANGULAR_X != 0 {
                0.0
            } else {
                s.angular_velocity.x
            },
            if flags & LOCK_ANGULAR_Y != 0 {
                0.0
            } else {
                s.angular_velocity.y
            },
            if flags & LOCK_ANGULAR_Z != 0 {
                0.0
            } else {
                s.angular_velocity.z
            },
        );

        if v.dot(v) > max_linear_speed_squared {
            let ratio = max_linear_speed / v.length();
            v = v.scale(ratio);
            flags |= IS_SPEED_CAPPED;
        }

        if w.dot(w) > max_angular_speed_squared && (flags & ALLOW_FAST_ROTATION) == 0 {
            let ratio = max_angular_speed / w.length();
            w = w.scale(ratio);
            flags |= IS_SPEED_CAPPED;
        }

        s.linear_velocity = v;
        s.angular_velocity = w;
        s.delta_position = s.delta_position.mul_add(h, v);
        s.delta_rotation = s.delta_rotation.integrate_rotation(w.scale(h));

        write_state(state_col, i, &s);
        flags_col.set(i, flags);
    }
}
