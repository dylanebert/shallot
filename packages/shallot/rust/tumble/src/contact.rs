//! The scalar (mesh/overflow) contact solver, ported op-for-op from box3d's `contact_solver.c`
//! `_Mesh` functions via the bit-exact TS port (`src/contactsolver.ts`). Prepare builds the transient
//! constraints from the touching manifolds; warm-start seeds velocities from the prior step's
//! impulses; solve/relax run the substep TGS iterations; restitution applies bounce; store writes the
//! final impulses back to the manifolds.
//!
//! Two column groups:
//!   - **persistent** — the column-resident contact-manifold store (`manifold_abi`), keyed by
//!     contactId: a directory record (material row + body indices + block descriptor + hit flag) and
//!     the pool manifolds. `prepare` gathers each record's contact through a per-color slot index
//!     (`slot[c] = contactId`, plus its transient-column cursors); `store` writes the solved impulses
//!     back into the same pool records (next step's warm start reads them there).
//!   - **transient** — the `b3ContactConstraint` / `b3ManifoldConstraint` / `b3ManifoldConstraintPoint`
//!     the solver consumes. Prepare builds it each step, per-step-sequential; the later phases
//!     read/mutate it.
//!
//! Every arithmetic op maps one-to-one to the C scalar path (no SIMD, no FMA); Rust f32 is native
//! IEEE with no contraction, so the result is bit-identical to the `DISABLE_SIMD` reference (math.rs).

use crate::body::flags as body_flags;
use crate::body::{read_sim, read_state, STATE_STRIDE};
use crate::manifold_abi::{
    read_dir, set_hit, M_FRICTION, M_NORMAL, M_POINTS, M_POINT_COUNT, M_ROLLING, M_TWIST,
    MANIFOLD_STRIDE, P_ANCHOR_A, P_ANCHOR_B, P_NORMAL_IMPULSE, P_NORMAL_VELOCITY, P_SEPARATION,
    P_TOTAL_NORMAL_IMPULSE, POOL_POINT_STRIDE, SLOT_CONTACT, SLOT_MANIFOLD_START, SLOT_POINT_START,
    SLOT_STRIDE,
};
use crate::col::Col;
use crate::math::{blend2, clampf, maxf, Mat2, Mat3, Quat, Vec2, Vec3, FLT_EPSILON};

/// Sentinel body index for a static body (no solver state), mirroring box3d's `B3_NULL_INDEX`.
pub const NULL_INDEX: u32 = u32::MAX;

/// Soft-constraint coefficients (b3Softness), a per-step scalar the solver reads from the context.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Softness {
    pub bias_rate: f32,
    pub mass_scale: f32,
    pub impulse_scale: f32,
}

// --- transient contact-constraint columns ---------------------------------------------------

/// f32 stride of the transient ContactConstraint: invMassA(1) invMassB(1) invIA(9) invIB(9)
/// rollingMass(9) softness(biasRate,massScale,impulseScale = 3) friction(1) restitution(1)
/// rollingResistance(1).
pub const CC_STRIDE: usize = 35;
/// u32 stride of the transient ContactConstraint meta: indexA indexB manifoldCount manifoldStart.
pub const CC_META_STRIDE: usize = 4;

/// f32 stride of the transient ManifoldConstraint: normal(3) tangent1(3) tangent2(3) tangentMass(4)
/// frictionImpulse(2) twistMass(1) twistImpulse(1) rollingImpulse(3) tangentVelocity1(1)
/// tangentVelocity2(1) originA(3) originB(3).
pub const MC_STRIDE: usize = 28;
/// u32 stride of the transient ManifoldConstraint meta: pointCount pointStart.
pub const MC_META_STRIDE: usize = 2;

/// f32 stride of the transient ManifoldConstraintPoint: rA(3) rB(3) baseSeparation(1)
/// normalImpulse(1) totalNormalImpulse(1) normalMass(1) relativeVelocity(1) leverArm(1).
pub const MCP_STRIDE: usize = 12;

// --- column read/write helpers --------------------------------------------------------------

#[inline]
fn v3(col: Col<f32>, o: usize) -> Vec3 {
    Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2))
}

#[inline]
fn write_v3(col: Col<f32>, o: usize, v: Vec3) {
    col.set(o, v.x);
    col.set(o + 1, v.y);
    col.set(o + 2, v.z);
}

#[inline]
fn write_mat3(col: Col<f32>, o: usize, m: Mat3) {
    write_v3(col, o, m.cx);
    write_v3(col, o + 3, m.cy);
    write_v3(col, o + 6, m.cz);
}

#[inline]
fn read_mat3(col: Col<f32>, o: usize) -> Mat3 {
    Mat3 {
        cx: v3(col, o),
        cy: v3(col, o + 3),
        cz: v3(col, o + 6),
    }
}

/// All the columns the contact solver phases operate over. Body columns are shared with the integrate
/// phases (`body.rs`); the persistent store columns (`slot`/`dir`/`pool`) are the narrowphase → solver
/// handoff; the transient constraint columns are the solver's own. Every one is shared-mutable
/// (`col.rs`): a stage's blocks hold the whole column and write disjoint records of it, concurrently.
#[derive(Clone, Copy)]
pub struct Columns<'a> {
    pub state: Col<'a, f32>,
    pub flags: Col<'a, u32>,
    pub sim: Col<'a, f32>,
    /// Per scalar solver-record slot (`SLOT_STRIDE` u32): contactId, transient `mc` base, transient
    /// `mcp` base. Maps a per-color scalar record to its persistent contact + transient-column slice.
    pub slot: Col<'a, u32>,
    /// Persistent contact directory (`DIR_STRIDE` u32/contactId), material + body indices + block
    /// descriptor + hit flag. Read for the gather; `store` writes the per-contact hit flag.
    pub dir: Col<'a, u32>,
    /// Persistent manifold pool (`MANIFOLD_STRIDE` f32/record). `prepare` reads the manifolds;
    /// `store` writes the solved impulses back into them (next step's warm start).
    pub pool: Col<'a, f32>,
    pub cc: Col<'a, f32>,
    pub cc_meta: Col<'a, u32>,
    pub mc: Col<'a, f32>,
    pub mc_meta: Col<'a, u32>,
    pub mcp: Col<'a, f32>,
}

// --- prepare --------------------------------------------------------------------------------

/// Build the transient constraints for the scalar records in `[start, start+count)`
/// (b3PrepareContacts_Mesh). Gathers each record's contact through the slot index → the persistent
/// directory + pool, plus body state/sim; writes the transient columns (per-step-sequential, indexed
/// by the slot's `mc`/`mcp` bases).
pub fn prepare(
    cols: &Columns,
    start: usize,
    count: usize,
    contact_softness: Softness,
    static_softness: Softness,
    warm_start_scale: f32,
) {
    const CONTACT_STATIC_FLAG: u32 = 0x0000_0008;

    for c in start..start + count {
        let so = c * SLOT_STRIDE;
        let contact_id = cols.slot.get(so + SLOT_CONTACT) as usize;
        let manifold_start = cols.slot.get(so + SLOT_MANIFOLD_START) as usize;
        let mut point_cursor = cols.slot.get(so + SLOT_POINT_START) as usize;

        let d = read_dir(cols.dir, contact_id);
        let index_a = d.index_a;
        let index_b = d.index_b;
        let manifold_count = d.manifold_count;
        let manifold_base = d.manifold_base;
        let friction = d.friction;
        let restitution = d.restitution;
        let rolling_resistance = d.rolling_resistance;
        let tangent_velocity = d.tangent_velocity;
        let contact_flags = d.flags;

        // Body A / B data (mass, inverse world inertia, velocities) — zeroed for a static body.
        let (m_a, i_a, v_a, w_a) = body_terms(cols.sim, cols.state, index_a);
        let (m_b, i_b, v_b, w_b) = body_terms(cols.sim, cols.state, index_b);

        // ContactConstraint record.
        let is_static = (contact_flags & CONTACT_STATIC_FLAG) != 0;
        let softness = if is_static {
            static_softness
        } else {
            contact_softness
        };
        let rolling_mass = i_a.add(i_b).invert();

        let cco = c * CC_STRIDE;
        cols.cc.set(cco, m_a);
        cols.cc.set(cco + 1, m_b);
        write_mat3(cols.cc, cco + 2, i_a);
        write_mat3(cols.cc, cco + 11, i_b);
        write_mat3(cols.cc, cco + 20, rolling_mass);
        cols.cc.set(cco + 29, softness.bias_rate);
        cols.cc.set(cco + 30, softness.mass_scale);
        cols.cc.set(cco + 31, softness.impulse_scale);
        cols.cc.set(cco + 32, friction);
        cols.cc.set(cco + 33, restitution);
        cols.cc.set(cco + 34, rolling_resistance);

        let ccm = c * CC_META_STRIDE;
        cols.cc_meta.set(ccm, index_a);
        cols.cc_meta.set(ccm + 1, index_b);
        cols.cc_meta.set(ccm + 2, manifold_count as u32);
        cols.cc_meta.set(ccm + 3, manifold_start as u32);

        for mi in 0..manifold_count {
            let m = manifold_start + mi; // transient manifold record
            let mpo = (manifold_base + mi) * MANIFOLD_STRIDE; // persistent pool record
            let point_count = cols.pool.get(mpo + M_POINT_COUNT).to_bits() as usize;
            let point_start = point_cursor; // transient point base for this manifold
            point_cursor += point_count;
            let normal = v3(cols.pool, mpo + M_NORMAL);
            let tangent1 = normal.perp();
            let tangent2 = tangent1.cross(normal);

            let mut center_a = Vec3::ZERO;
            let mut center_b = Vec3::ZERO;

            for pi in 0..point_count {
                let pp = mpo + M_POINTS + pi * POOL_POINT_STRIDE; // pool point record
                let p = point_start + pi; // transient point record
                let r_a = v3(cols.pool, pp + P_ANCHOR_A);
                let r_b = v3(cols.pool, pp + P_ANCHOR_B);
                let separation = cols.pool.get(pp + P_SEPARATION);
                let mp_normal_impulse = cols.pool.get(pp + P_NORMAL_IMPULSE);

                let rn_a = r_a.cross(normal);
                let rn_b = r_b.cross(normal);
                let k_normal = m_a + m_b + rn_a.dot(i_a.mul_v(rn_a)) + rn_b.dot(i_b.mul_v(rn_b));

                let vr_a = v_a.add(w_a.cross(r_a));
                let vr_b = v_b.add(w_b.cross(r_b));

                let base_separation = separation - r_b.sub(r_a).dot(normal);
                let normal_mass = if k_normal > 0.0 { 1.0 / k_normal } else { 0.0 };
                let relative_velocity = normal.dot(vr_b.sub(vr_a));

                let po = p * MCP_STRIDE;
                write_v3(cols.mcp, po, r_a);
                write_v3(cols.mcp, po + 3, r_b);
                cols.mcp.set(po + 6, base_separation);
                cols.mcp.set(po + 7, warm_start_scale * mp_normal_impulse);
                cols.mcp.set(po + 8, 0.0); // totalNormalImpulse
                cols.mcp.set(po + 9, normal_mass);
                cols.mcp.set(po + 10, relative_velocity);
                cols.mcp.set(po + 11, 0.0); // leverArm, filled below

                center_a = center_a.add(r_a);
                center_b = center_b.add(r_b);
            }

            let inv_count = 1.0 / point_count as f32;
            center_a = center_a.scale(inv_count);
            center_b = center_b.scale(inv_count);

            for p in point_start..point_start + point_count {
                let po = p * MCP_STRIDE;
                let r_a = v3(cols.mcp, po);
                cols.mcp.set(po + 11, r_a.distance(center_a));
            }

            let rt_a1 = center_a.cross(tangent1);
            let rt_a2 = center_a.cross(tangent2);
            let rt_b1 = center_b.cross(tangent1);
            let rt_b2 = center_b.cross(tangent2);

            let kxx = m_a + m_b + rt_a1.dot(i_a.mul_v(rt_a1)) + rt_b1.dot(i_b.mul_v(rt_b1));
            let kyy = m_a + m_b + rt_a2.dot(i_a.mul_v(rt_a2)) + rt_b2.dot(i_b.mul_v(rt_b2));
            let kxy = rt_a1.dot(i_a.mul_v(rt_a2)) + rt_b1.dot(i_b.mul_v(rt_b2));
            let k = Mat2 {
                cx: Vec2::new(kxx, kxy),
                cy: Vec2::new(kxy, kyy),
            };
            let tangent_mass = k.invert();

            let friction_impulse = v3(cols.pool, mpo + M_FRICTION);
            let twist_impulse = cols.pool.get(mpo + M_TWIST);
            let rolling_impulse = v3(cols.pool, mpo + M_ROLLING);

            let twist_k = normal.dot(i_a.add(i_b).mul_v(normal));
            let twist_mass = if twist_k > 0.0 { 1.0 / twist_k } else { 0.0 };

            let mo = m * MC_STRIDE;
            write_v3(cols.mc, mo, normal);
            write_v3(cols.mc, mo + 3, tangent1);
            write_v3(cols.mc, mo + 6, tangent2);
            cols.mc.set(mo + 9, tangent_mass.cx.x);
            cols.mc.set(mo + 10, tangent_mass.cx.y);
            cols.mc.set(mo + 11, tangent_mass.cy.x);
            cols.mc.set(mo + 12, tangent_mass.cy.y);
            cols.mc.set(mo + 13, warm_start_scale * friction_impulse.dot(tangent1));
            cols.mc.set(mo + 14, warm_start_scale * friction_impulse.dot(tangent2));
            cols.mc.set(mo + 15, twist_mass);
            cols.mc.set(mo + 16, warm_start_scale * twist_impulse);
            write_v3(cols.mc, mo + 17, rolling_impulse.scale(warm_start_scale));
            cols.mc.set(mo + 20, tangent_velocity.dot(tangent1));
            cols.mc.set(mo + 21, tangent_velocity.dot(tangent2));
            write_v3(cols.mc, mo + 22, center_a);
            write_v3(cols.mc, mo + 25, center_b);

            let mm = m * MC_META_STRIDE;
            cols.mc_meta.set(mm, point_count as u32);
            cols.mc_meta.set(mm + 1, point_start as u32);
        }
    }
}

// --- warm start -----------------------------------------------------------------------------

/// Seed body velocities from the prior step's impulses (b3WarmStartContacts_Mesh). Reads the
/// transient constraints; mutates the velocity fields of the state column (dynamic bodies only).
pub fn warm_start(cols: &Columns, start: usize, count: usize) {
    for c in start..start + count {
        let cco = c * CC_STRIDE;
        let ccm = c * CC_META_STRIDE;
        let index_a = cols.cc_meta.get(ccm);
        let index_b = cols.cc_meta.get(ccm + 1);
        let manifold_count = cols.cc_meta.get(ccm + 2) as usize;
        let manifold_start = cols.cc_meta.get(ccm + 3) as usize;

        let m_a = cols.cc.get(cco);
        let m_b = cols.cc.get(cco + 1);
        let i_a = read_mat3(cols.cc, cco + 2);
        let i_b = read_mat3(cols.cc, cco + 11);

        let (mut v_a, mut w_a) = read_vel(cols.state, index_a);
        let (mut v_b, mut w_b) = read_vel(cols.state, index_b);

        for m in manifold_start..manifold_start + manifold_count {
            let mo = m * MC_STRIDE;
            let mm = m * MC_META_STRIDE;
            let normal = v3(cols.mc, mo);
            let point_count = cols.mc_meta.get(mm) as usize;
            let point_start = cols.mc_meta.get(mm + 1) as usize;

            for p in point_start..point_start + point_count {
                let po = p * MCP_STRIDE;
                let r_a = v3(cols.mcp, po);
                let r_b = v3(cols.mcp, po + 3);
                let normal_impulse = cols.mcp.get(po + 7);
                let impulse = normal.scale(normal_impulse);
                w_a = w_a.sub(i_a.mul_v(r_a.cross(impulse)));
                v_a = v_a.mul_sub(m_a, impulse);
                w_b = w_b.add(i_b.mul_v(r_b.cross(impulse)));
                v_b = v_b.mul_add(m_b, impulse);
            }

            // Central friction at the manifold origin.
            {
                let r_a = v3(cols.mc, mo + 22);
                let r_b = v3(cols.mc, mo + 25);
                let tangent1 = v3(cols.mc, mo + 3);
                let tangent2 = v3(cols.mc, mo + 6);
                let impulse = tangent1
                    .scale(cols.mc.get(mo + 13))
                    .add(tangent2.scale(cols.mc.get(mo + 14)));
                w_a = w_a.sub(i_a.mul_v(r_a.cross(impulse)));
                v_a = v_a.mul_sub(m_a, impulse);
                w_b = w_b.add(i_b.mul_v(r_b.cross(impulse)));
                v_b = v_b.mul_add(m_b, impulse);
            }

            // Central twist friction.
            {
                let impulse = normal.scale(cols.mc.get(mo + 16));
                w_a = w_a.sub(i_a.mul_v(impulse));
                w_b = w_b.add(i_b.mul_v(impulse));
            }

            // Rolling resistance.
            {
                let impulse = v3(cols.mc, mo + 17);
                w_a = w_a.sub(i_a.mul_v(impulse));
                w_b = w_b.add(i_b.mul_v(impulse));
            }
        }

        write_vel(cols.state, cols.flags, index_a, v_a, w_a);
        write_vel(cols.state, cols.flags, index_b, v_b, w_b);
    }
}

// --- solve / relax --------------------------------------------------------------------------

/// Run one solve (bias) or relax (no bias) pass over the contacts (b3SolveContacts_Mesh). Mutates
/// the accumulated impulses in the transient columns and the body velocities (dynamic bodies only).
pub fn solve(
    cols: &Columns,
    start: usize,
    count: usize,
    use_bias: bool,
    inv_h: f32,
    contact_speed: f32,
) {
    for c in start..start + count {
        let cco = c * CC_STRIDE;
        let ccm = c * CC_META_STRIDE;
        let index_a = cols.cc_meta.get(ccm);
        let index_b = cols.cc_meta.get(ccm + 1);
        let manifold_count = cols.cc_meta.get(ccm + 2) as usize;
        let manifold_start = cols.cc_meta.get(ccm + 3) as usize;

        let m_a = cols.cc.get(cco);
        let m_b = cols.cc.get(cco + 1);
        let i_a = read_mat3(cols.cc, cco + 2);
        let i_b = read_mat3(cols.cc, cco + 11);
        let rolling_mass = read_mat3(cols.cc, cco + 20);
        let soft_bias_rate = cols.cc.get(cco + 29);
        let soft_mass_scale = cols.cc.get(cco + 30);
        let soft_impulse_scale = cols.cc.get(cco + 31);
        let friction = cols.cc.get(cco + 32);
        let rolling_resistance = cols.cc.get(cco + 34);

        let (mut v_a, mut w_a, dp_a, dq_a) = read_solve_state(cols.state, index_a);
        let (mut v_b, mut w_b, dp_b, dq_b) = read_solve_state(cols.state, index_b);
        let dp = dp_b.sub(dp_a);

        for m in manifold_start..manifold_start + manifold_count {
            let mo = m * MC_STRIDE;
            let mm = m * MC_META_STRIDE;
            let normal = v3(cols.mc, mo);
            let point_count = cols.mc_meta.get(mm) as usize;
            let point_start = cols.mc_meta.get(mm + 1) as usize;

            let mut total_normal_impulse = 0.0f32;
            let mut total_twist_limit = 0.0f32;

            for p in point_start..point_start + point_count {
                let po = p * MCP_STRIDE;
                let r_a = v3(cols.mcp, po);
                let r_b = v3(cols.mcp, po + 3);
                let base_separation = cols.mcp.get(po + 6);
                let normal_mass = cols.mcp.get(po + 9);
                let lever_arm = cols.mcp.get(po + 11);
                let normal_impulse = cols.mcp.get(po + 7);

                let ds = dp.add(dq_b.rotate(r_b).sub(dq_a.rotate(r_a)));
                let s = ds.dot(normal) + base_separation;

                let mut velocity_bias = 0.0f32;
                let mut mass_scale = 1.0f32;
                let mut impulse_scale = 0.0f32;
                if s > 0.0 {
                    velocity_bias = s * inv_h;
                } else if use_bias {
                    velocity_bias = maxf(soft_mass_scale * soft_bias_rate * s, -contact_speed);
                    mass_scale = soft_mass_scale;
                    impulse_scale = soft_impulse_scale;
                }

                let vr_a = v_a.add(w_a.cross(r_a));
                let vr_b = v_b.add(w_b.cross(r_b));
                let vn = vr_b.sub(vr_a).dot(normal);

                let mut delta_impulse = -normal_mass * (mass_scale * vn + velocity_bias)
                    - impulse_scale * normal_impulse;

                let new_impulse = maxf(normal_impulse + delta_impulse, 0.0);
                delta_impulse = new_impulse - normal_impulse;
                cols.mcp.set(po + 7, new_impulse);
                cols.mcp.set(po + 8, cols.mcp.get(po + 8) + new_impulse);

                total_normal_impulse += new_impulse;
                total_twist_limit += lever_arm * new_impulse;

                let p_imp = normal.scale(delta_impulse);
                v_a = v_a.mul_sub(m_a, p_imp);
                w_a = w_a.sub(i_a.mul_v(r_a.cross(p_imp)));
                v_b = v_b.mul_add(m_b, p_imp);
                w_b = w_b.add(i_b.mul_v(r_b.cross(p_imp)));
            }

            if use_bias {
                continue;
            }

            // Central twist friction.
            {
                let twist_speed = normal.dot(w_b.sub(w_a));
                let max_impulse = friction * total_twist_limit;
                let delta = -cols.mc.get(mo + 15) * twist_speed;
                let old = cols.mc.get(mo + 16);
                let clamped = clampf(old + delta, -max_impulse, max_impulse);
                cols.mc.set(mo + 16, clamped);
                let applied = clamped - old;
                w_a = w_a.sub(i_a.mul_v(normal.scale(applied)));
                w_b = w_b.add(i_b.mul_v(normal.scale(applied)));
            }

            // Rolling resistance.
            if rolling_resistance > 0.0 {
                let delta = rolling_mass.mul_v(w_b.sub(w_a)).neg();
                let old = v3(cols.mc, mo + 17);
                let mut rolling = old.add(delta);

                let max_impulse = rolling_resistance * total_normal_impulse;
                let mag_sqr = rolling.dot(rolling);
                if mag_sqr > max_impulse * max_impulse + FLT_EPSILON {
                    rolling = rolling.scale(max_impulse / mag_sqr.sqrt());
                }

                let applied = rolling.sub(old);
                write_v3(cols.mc, mo + 17, rolling);
                w_a = w_a.sub(i_a.mul_v(applied));
                w_b = w_b.add(i_b.mul_v(applied));
            }

            // Central friction.
            {
                let tangent1 = v3(cols.mc, mo + 3);
                let tangent2 = v3(cols.mc, mo + 6);
                let r_a = v3(cols.mc, mo + 22);
                let r_b = v3(cols.mc, mo + 25);
                let tangent_mass = Mat2 {
                    cx: Vec2::new(cols.mc.get(mo + 9), cols.mc.get(mo + 10)),
                    cy: Vec2::new(cols.mc.get(mo + 11), cols.mc.get(mo + 12)),
                };
                let tangent_velocity1 = cols.mc.get(mo + 20);
                let tangent_velocity2 = cols.mc.get(mo + 21);
                let friction_impulse = Vec2::new(cols.mc.get(mo + 13), cols.mc.get(mo + 14));

                let vr_a = v_a.add(w_a.cross(r_a));
                let vr_b = v_b.add(w_b.cross(r_b));
                let vr = vr_b.sub(vr_a);
                let vt = Vec2::new(
                    vr.dot(tangent1) - tangent_velocity1,
                    vr.dot(tangent2) - tangent_velocity2,
                );

                let tm = tangent_mass.mul_v(vt);
                let delta = Vec2::new(-tm.x, -tm.y);
                let mut new_impulse =
                    Vec2::new(friction_impulse.x + delta.x, friction_impulse.y + delta.y);

                let max_impulse = friction * total_normal_impulse;
                let length_squared = new_impulse.dot(new_impulse);
                if length_squared > max_impulse * max_impulse {
                    let scale = max_impulse / length_squared.sqrt();
                    new_impulse = Vec2::new(new_impulse.x * scale, new_impulse.y * scale);
                }
                let applied = new_impulse.sub(friction_impulse);
                cols.mc.set(mo + 13, new_impulse.x);
                cols.mc.set(mo + 14, new_impulse.y);

                let p_imp = blend2(applied.x, tangent1, applied.y, tangent2);
                v_a = v_a.mul_sub(m_a, p_imp);
                w_a = w_a.sub(i_a.mul_v(r_a.cross(p_imp)));
                v_b = v_b.mul_add(m_b, p_imp);
                w_b = w_b.add(i_b.mul_v(r_b.cross(p_imp)));
            }
        }

        write_vel(cols.state, cols.flags, index_a, v_a, w_a);
        write_vel(cols.state, cols.flags, index_b, v_b, w_b);
    }
}

// --- restitution ----------------------------------------------------------------------------

/// Apply restitution bounce to approaching contacts (b3ApplyRestitution_Mesh). Mutates the point
/// normal impulses and the body velocities (dynamic bodies only).
pub fn restitution(cols: &Columns, start: usize, count: usize, threshold: f32) {
    for c in start..start + count {
        let cco = c * CC_STRIDE;
        let ccm = c * CC_META_STRIDE;
        let restitution = cols.cc.get(cco + 33);
        if restitution == 0.0 {
            continue;
        }
        let index_a = cols.cc_meta.get(ccm);
        let index_b = cols.cc_meta.get(ccm + 1);
        let manifold_count = cols.cc_meta.get(ccm + 2) as usize;
        let manifold_start = cols.cc_meta.get(ccm + 3) as usize;
        let m_a = cols.cc.get(cco);
        let m_b = cols.cc.get(cco + 1);
        let i_a = read_mat3(cols.cc, cco + 2);
        let i_b = read_mat3(cols.cc, cco + 11);

        let (mut v_a, mut w_a) = read_vel(cols.state, index_a);
        let (mut v_b, mut w_b) = read_vel(cols.state, index_b);

        for m in manifold_start..manifold_start + manifold_count {
            let mo = m * MC_STRIDE;
            let mm = m * MC_META_STRIDE;
            let normal = v3(cols.mc, mo);
            let point_count = cols.mc_meta.get(mm) as usize;
            let point_start = cols.mc_meta.get(mm + 1) as usize;

            for p in point_start..point_start + point_count {
                let po = p * MCP_STRIDE;
                let relative_velocity = cols.mcp.get(po + 10);
                let total_normal_impulse = cols.mcp.get(po + 8);
                // Skip speculative points that never generated a real impulse.
                if relative_velocity > -threshold || total_normal_impulse == 0.0 {
                    continue;
                }
                let r_a = v3(cols.mcp, po);
                let r_b = v3(cols.mcp, po + 3);
                let normal_mass = cols.mcp.get(po + 9);
                let normal_impulse = cols.mcp.get(po + 7);

                let vr_b = v_b.add(w_b.cross(r_b));
                let vr_a = v_a.add(w_a.cross(r_a));
                let vn = vr_b.sub(vr_a).dot(normal);

                let mut impulse = -normal_mass * (vn + restitution * relative_velocity);
                let new_impulse = maxf(normal_impulse + impulse, 0.0);
                impulse = new_impulse - normal_impulse;
                cols.mcp.set(po + 7, new_impulse);
                cols.mcp.set(po + 8, cols.mcp.get(po + 8) + impulse);

                let p_imp = normal.scale(impulse);
                v_a = v_a.mul_sub(m_a, p_imp);
                w_a = w_a.sub(i_a.mul_v(r_a.cross(p_imp)));
                v_b = v_b.mul_add(m_b, p_imp);
                w_b = w_b.add(i_b.mul_v(r_b.cross(p_imp)));
            }

            write_vel(cols.state, cols.flags, index_a, v_a, w_a);
            write_vel(cols.state, cols.flags, index_b, v_b, w_b);
        }
    }
}

// --- store ----------------------------------------------------------------------------------

/// Write the solved impulses back into the persistent manifold pool and flag hit events
/// (b3StoreImpulses_Mesh). The pool records are the persistent warm-start state the next step reads;
/// the hit flag lands in the directory (TS reads it back to build the user-facing events).
pub fn store(cols: &Columns, start: usize, count: usize, hit_event_threshold: f32) {
    const SIM_ENABLE_HIT_EVENT: u32 = 0x0010_0000;
    let neg_hit_threshold = -hit_event_threshold;

    for c in start..start + count {
        let so = c * SLOT_STRIDE;
        let contact_id = cols.slot.get(so + SLOT_CONTACT) as usize;
        let d = read_dir(cols.dir, contact_id);
        let manifold_base = d.manifold_base;
        let ccm = c * CC_META_STRIDE;
        let manifold_count = cols.cc_meta.get(ccm + 2) as usize;
        let manifold_start = cols.cc_meta.get(ccm + 3) as usize;
        let check_hit_events = (d.flags & SIM_ENABLE_HIT_EVENT) != 0;
        let mut flagged = false;

        for mi in 0..manifold_count {
            let m = manifold_start + mi; // transient manifold record
            let mpo = (manifold_base + mi) * MANIFOLD_STRIDE; // persistent pool record
            let mo = m * MC_STRIDE;
            let mm = m * MC_META_STRIDE;

            let tangent1 = v3(cols.mc, mo + 3);
            let tangent2 = v3(cols.mc, mo + 6);
            let friction = blend2(cols.mc.get(mo + 13), tangent1, cols.mc.get(mo + 14), tangent2);
            cols.pool.set(mpo + M_TWIST, cols.mc.get(mo + 16)); // twistImpulse
            write_v3(cols.pool, mpo + M_FRICTION, friction); // frictionImpulse
            write_v3(cols.pool, mpo + M_ROLLING, v3(cols.mc, mo + 17)); // rollingImpulse

            let point_count = cols.mc_meta.get(mm) as usize;
            let point_start = cols.mc_meta.get(mm + 1) as usize;
            for pi in 0..point_count {
                let p = point_start + pi; // transient point record
                let pp = mpo + M_POINTS + pi * POOL_POINT_STRIDE; // pool point record
                let po = p * MCP_STRIDE;
                let normal_impulse = cols.mcp.get(po + 7);
                let total_normal_impulse = cols.mcp.get(po + 8);
                let normal_velocity = cols.mcp.get(po + 10);
                cols.pool.set(pp + P_NORMAL_IMPULSE, normal_impulse);
                cols.pool.set(pp + P_TOTAL_NORMAL_IMPULSE, total_normal_impulse);
                cols.pool.set(pp + P_NORMAL_VELOCITY, normal_velocity);

                // One flag per contact: a confirmed impulse approaching faster than the threshold.
                if check_hit_events
                    && !flagged
                    && normal_velocity < neg_hit_threshold
                    && total_normal_impulse > 0.0
                {
                    set_hit(cols.dir, contact_id);
                    flagged = true;
                }
            }
        }
    }
}

/// (linearVelocity, angularVelocity, deltaPosition, deltaRotation) for a body index; the identity
/// body state (zero velocity/position, identity rotation) for a static (NULL) body.
#[inline]
fn read_solve_state(state_col: Col<f32>, index: u32) -> (Vec3, Vec3, Vec3, Quat) {
    if index == NULL_INDEX {
        (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY)
    } else {
        let o = index as usize * STATE_STRIDE;
        (
            v3(state_col, o),
            v3(state_col, o + 3),
            v3(state_col, o + 6),
            Quat {
                v: v3(state_col, o + 9),
                s: state_col.get(o + 12),
            },
        )
    }
}

/// (linearVelocity, angularVelocity) for a body index, zero for a static (NULL) body.
#[inline]
fn read_vel(state_col: Col<f32>, index: u32) -> (Vec3, Vec3) {
    if index == NULL_INDEX {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        let o = index as usize * STATE_STRIDE;
        (v3(state_col, o), v3(state_col, o + 3))
    }
}

/// Write velocities back, only for a real dynamic body (static/kinematic bodies keep theirs).
#[inline]
fn write_vel(state_col: Col<f32>, flags_col: Col<u32>, index: u32, v: Vec3, w: Vec3) {
    if index != NULL_INDEX && flags_col.get(index as usize) & body_flags::DYNAMIC != 0 {
        let o = index as usize * STATE_STRIDE;
        write_v3(state_col, o, v);
        write_v3(state_col, o + 3, w);
    }
}

/// (invMass, invInertiaWorld, linearVelocity, angularVelocity) for a body index, zeroed for static.
#[inline]
fn body_terms(sim_col: Col<f32>, state_col: Col<f32>, index: u32) -> (f32, Mat3, Vec3, Vec3) {
    if index == NULL_INDEX {
        (0.0, Mat3::ZERO, Vec3::ZERO, Vec3::ZERO)
    } else {
        let i = index as usize;
        let sim = read_sim(sim_col, i);
        let state = read_state(state_col, i);
        (
            sim.inv_mass,
            sim.inv_inertia_world,
            state.linear_velocity,
            state.angular_velocity,
        )
    }
}
