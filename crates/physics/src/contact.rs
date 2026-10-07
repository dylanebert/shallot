//! Box3D scalar mesh and overflow contact constraints.

use crate::body::flags as body_flags;
use crate::body::{read_sim, read_state, STATE_STRIDE};
use crate::col::Col;
use crate::manifold_abi::{
    read_dir, MANIFOLD_STRIDE, M_FRICTION, M_NORMAL, M_POINTS, M_POINT_COUNT, M_ROLLING, M_TWIST,
    POOL_POINT_STRIDE, P_ANCHOR_A, P_ANCHOR_B, P_NORMAL_IMPULSE, P_NORMAL_VELOCITY, P_SEPARATION,
    P_TOTAL_NORMAL_IMPULSE,
};
use crate::math::{blend2, clampf, maxf, Mat2, Mat3, Quat, Vec2, Vec3, FLT_EPSILON};

/// Sentinel body index for a static body (no solver state), mirroring box3d's `B3_NULL_INDEX`.
pub const NULL_INDEX: u32 = u32::MAX;
const SPECULATIVE_DISTANCE: f32 = 0.02;
const MIN_FRICTION_WEIGHT: f32 = 1.0e-10;

/// Soft-constraint coefficients (b3Softness), a per-step scalar the solver reads from the context.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Softness {
    pub bias_rate: f32,
    pub mass_scale: f32,
    pub impulse_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ManifoldConstraintPoint {
    pub r_a: Vec3,
    pub r_b: Vec3,
    pub base_separation: f32,
    pub relative_velocity: f32,
    pub normal_impulse: f32,
    pub total_normal_impulse: f32,
    pub normal_mass: f32,
    pub lever_arm: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ManifoldConstraint {
    pub points: [ManifoldConstraintPoint; 4],
    pub point_count: i32,
    pub normal: Vec3,
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    pub center_a: Vec3,
    pub center_b: Vec3,
    pub twist_mass: f32,
    pub twist_impulse: f32,
    pub tangent_mass: Mat2,
    pub friction_impulse: Vec2,
    pub rolling_impulse: Vec3,
    pub tangent_velocity1: f32,
    pub tangent_velocity2: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContactConstraint {
    pub constraints: *mut ManifoldConstraint,
    pub contact: *mut crate::manifold_abi::ContactRecord,
    pub index_a: u32,
    pub index_b: u32,
    pub inv_mass_a: f32,
    pub inv_mass_b: f32,
    pub inv_ia: Mat3,
    pub inv_ib: Mat3,
    pub softness: Softness,
    pub rolling_mass: Mat3,
    pub friction: f32,
    pub restitution: f32,
    pub rolling_resistance: f32,
    pub manifold_count: i32,
}

// --- column read/write helpers --------------------------------------------------------------

// SAFETY: referenced manifolds belong to this constraint; graph coloring separates body writes,
// and contact flag writes use atomics. All referenced storage stays put until workers join.
unsafe impl Send for ContactConstraint {}

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

#[derive(Clone, Copy)]
pub struct Columns<'a> {
    pub state: Col<'a, f32>,
    pub flags: Col<'a, u32>,
    pub sim: Col<'a, f32>,
    pub spans: Col<'a, crate::contact_spans::ContactPrepareSpan>,
    pub dir: Col<'a, u32>,
    pub pool: Col<'a, f32>,
    pub cc: Col<'a, ContactConstraint>,
    pub mc: Col<'a, ManifoldConstraint>,
}

// --- prepare --------------------------------------------------------------------------------

pub fn prepare(
    cols: &Columns,
    start: usize,
    count: usize,
    contact_softness: Softness,
    static_softness: Softness,
    warm_start_scale: f32,
) {
    unsafe {
        const CONTACT_STATIC_FLAG: u32 = 0x0000_0008;

        for (c, spec) in crate::contact_spans::contact_specs(cols.spans, start, count) {
            let contact_id = spec.contact_id as usize;
            let cc = cols.cc.ptr().add(c);
            (*cc).constraints = cols.mc.ptr().add(spec.manifold_start as usize);
            (*cc).contact = cols
                .dir
                .ptr()
                .add(contact_id * crate::manifold_abi::DIR_STRIDE)
                .cast();

            let d = read_dir(cols.dir, contact_id);
            let index_a = d.index_a;
            let index_b = d.index_b;
            let manifold_count = spec.manifold_count as usize;
            let cols = Columns {
                pool: crate::manifold_abi::block_col(cols.pool, d.manifold_base, manifold_count),
                ..*cols
            };
            let manifold_base = 0;
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

            (*cc).inv_mass_a = m_a;
            (*cc).inv_mass_b = m_b;
            (*cc).inv_ia = i_a;
            (*cc).inv_ib = i_b;
            (*cc).rolling_mass = rolling_mass;
            (*cc).softness.bias_rate = softness.bias_rate;
            (*cc).softness.mass_scale = softness.mass_scale;
            (*cc).softness.impulse_scale = softness.impulse_scale;
            (*cc).friction = friction;
            (*cc).restitution = restitution;
            (*cc).rolling_resistance = rolling_resistance;

            (*cc).index_a = index_a;
            (*cc).index_b = index_b;
            (*cc).manifold_count = manifold_count as i32;

            for mi in 0..manifold_count {
                let mc = (*cc).constraints.add(mi);
                let mpo = (manifold_base + mi) * MANIFOLD_STRIDE; // persistent pool record
                let point_count = cols.pool.get(mpo + M_POINT_COUNT).to_bits() as usize;

                let normal = v3(cols.pool, mpo + M_NORMAL);
                let tangent1 = normal.perp();
                let tangent2 = tangent1.cross(normal);

                let mut center_a = Vec3::ZERO;
                let mut center_b = Vec3::ZERO;
                let mut total_friction_weight = 0.0f32;
                let inv_tau = 1.0f32 / SPECULATIVE_DISTANCE;

                for pi in 0..point_count {
                    let pp = mpo + M_POINTS + pi * POOL_POINT_STRIDE; // pool point record
                    let cp = core::ptr::addr_of_mut!((*mc).points[pi]);
                    let r_a = v3(cols.pool, pp + P_ANCHOR_A);
                    let r_b = v3(cols.pool, pp + P_ANCHOR_B);
                    let separation = cols.pool.get(pp + P_SEPARATION);
                    let mp_normal_impulse = cols.pool.get(pp + P_NORMAL_IMPULSE);

                    let rn_a = r_a.cross(normal);
                    let rn_b = r_b.cross(normal);
                    let k_normal =
                        m_a + m_b + rn_a.dot(i_a.mul_v(rn_a)) + rn_b.dot(i_b.mul_v(rn_b));

                    let vr_a = v_a.add(w_a.cross(r_a));
                    let vr_b = v_b.add(w_b.cross(r_b));

                    let base_separation = separation - r_b.sub(r_a).dot(normal);
                    let normal_mass = if k_normal > 0.0 { 1.0 / k_normal } else { 0.0 };
                    let relative_velocity = normal.dot(vr_b.sub(vr_a));

                    (*cp).r_a = r_a;
                    (*cp).r_b = r_b;
                    (*cp).base_separation = base_separation;
                    (*cp).normal_impulse = warm_start_scale * mp_normal_impulse;
                    (*cp).total_normal_impulse = 0.0; // totalNormalImpulse
                    (*cp).normal_mass = normal_mass;
                    (*cp).relative_velocity = relative_velocity;
                    (*cp).lever_arm = 0.0; // leverArm, filled below

                    let weight = clampf(2.0 - separation * inv_tau, MIN_FRICTION_WEIGHT, 1.0);
                    center_a = center_a.add(r_a.scale(weight));
                    center_b = center_b.add(r_b.scale(weight));
                    total_friction_weight += weight;
                }

                let inv_weight = 1.0 / total_friction_weight;
                center_a = center_a.scale(inv_weight);
                center_b = center_b.scale(inv_weight);

                for pi in 0..point_count {
                    let cp = core::ptr::addr_of_mut!((*mc).points[pi]);

                    let r_a = (*cp).r_a;
                    (*cp).lever_arm = r_a.distance(center_a);
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

                (*mc).normal = normal;
                (*mc).tangent1 = tangent1;
                (*mc).tangent2 = tangent2;
                (*mc).tangent_mass.cx.x = tangent_mass.cx.x;
                (*mc).tangent_mass.cx.y = tangent_mass.cx.y;
                (*mc).tangent_mass.cy.x = tangent_mass.cy.x;
                (*mc).tangent_mass.cy.y = tangent_mass.cy.y;
                (*mc).friction_impulse.x = warm_start_scale * friction_impulse.dot(tangent1);
                (*mc).friction_impulse.y = warm_start_scale * friction_impulse.dot(tangent2);
                (*mc).twist_mass = twist_mass;
                (*mc).twist_impulse = warm_start_scale * twist_impulse;
                (*mc).rolling_impulse = rolling_impulse.scale(warm_start_scale);
                (*mc).tangent_velocity1 = tangent_velocity.dot(tangent1);
                (*mc).tangent_velocity2 = tangent_velocity.dot(tangent2);
                (*mc).center_a = center_a;
                (*mc).center_b = center_b;

                (*mc).point_count = point_count as i32;
            }
        }
    }
}

// --- warm start -----------------------------------------------------------------------------

/// Seed body velocities from the prior step's impulses (b3WarmStartContacts_Mesh). Reads the
/// transient constraints; mutates the velocity fields of the state column (dynamic bodies only).
pub fn warm_start(cols: &Columns, start: usize, count: usize) {
    unsafe {
        for c in start..start + count {
            let cc = cols.cc.ptr().add(c);

            let index_a = (*cc).index_a;
            let index_b = (*cc).index_b;
            let manifold_count = (*cc).manifold_count as usize;

            let m_a = (*cc).inv_mass_a;
            let m_b = (*cc).inv_mass_b;
            let i_a = (*cc).inv_ia;
            let i_b = (*cc).inv_ib;

            let (mut v_a, mut w_a) = read_vel(cols.state, index_a);
            let (mut v_b, mut w_b) = read_vel(cols.state, index_b);

            for mi in 0..manifold_count {
                let mc = (*cc).constraints.add(mi);

                let normal = (*mc).normal;
                let point_count = (*mc).point_count as usize;

                for pi in 0..point_count {
                    let cp = core::ptr::addr_of_mut!((*mc).points[pi]);

                    let r_a = (*cp).r_a;
                    let r_b = (*cp).r_b;
                    let normal_impulse = (*cp).normal_impulse;
                    let impulse = normal.scale(normal_impulse);
                    w_a = w_a.sub(i_a.mul_v(r_a.cross(impulse)));
                    v_a = v_a.mul_sub(m_a, impulse);
                    w_b = w_b.add(i_b.mul_v(r_b.cross(impulse)));
                    v_b = v_b.mul_add(m_b, impulse);
                }

                // Central friction at the manifold origin.
                {
                    let r_a = (*mc).center_a;
                    let r_b = (*mc).center_b;
                    let tangent1 = (*mc).tangent1;
                    let tangent2 = (*mc).tangent2;
                    let impulse = tangent1
                        .scale((*mc).friction_impulse.x)
                        .add(tangent2.scale((*mc).friction_impulse.y));
                    w_a = w_a.sub(i_a.mul_v(r_a.cross(impulse)));
                    v_a = v_a.mul_sub(m_a, impulse);
                    w_b = w_b.add(i_b.mul_v(r_b.cross(impulse)));
                    v_b = v_b.mul_add(m_b, impulse);
                }

                // Central twist friction.
                {
                    let impulse = normal.scale((*mc).twist_impulse);
                    w_a = w_a.sub(i_a.mul_v(impulse));
                    w_b = w_b.add(i_b.mul_v(impulse));
                }

                // Rolling resistance.
                {
                    let impulse = (*mc).rolling_impulse;
                    w_a = w_a.sub(i_a.mul_v(impulse));
                    w_b = w_b.add(i_b.mul_v(impulse));
                }
            }

            write_vel(cols.state, cols.flags, index_a, v_a, w_a);
            write_vel(cols.state, cols.flags, index_b, v_b, w_b);
        }
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
    unsafe {
        for c in start..start + count {
            let cc = cols.cc.ptr().add(c);

            let index_a = (*cc).index_a;
            let index_b = (*cc).index_b;
            let manifold_count = (*cc).manifold_count as usize;

            let m_a = (*cc).inv_mass_a;
            let m_b = (*cc).inv_mass_b;
            let i_a = (*cc).inv_ia;
            let i_b = (*cc).inv_ib;
            let rolling_mass = (*cc).rolling_mass;
            let soft_bias_rate = (*cc).softness.bias_rate;
            let soft_mass_scale = (*cc).softness.mass_scale;
            let soft_impulse_scale = (*cc).softness.impulse_scale;
            let friction = (*cc).friction;
            let rolling_resistance = (*cc).rolling_resistance;

            let (mut v_a, mut w_a, dp_a, dq_a) = read_solve_state(cols.state, index_a);
            let (mut v_b, mut w_b, dp_b, dq_b) = read_solve_state(cols.state, index_b);
            let dp = dp_b.sub(dp_a);

            for mi in 0..manifold_count {
                let mc = (*cc).constraints.add(mi);

                let normal = (*mc).normal;
                let point_count = (*mc).point_count as usize;

                let mut total_normal_impulse = 0.0f32;
                let mut total_twist_limit = 0.0f32;

                for pi in 0..point_count {
                    let cp = core::ptr::addr_of_mut!((*mc).points[pi]);

                    let r_a = (*cp).r_a;
                    let r_b = (*cp).r_b;
                    let base_separation = (*cp).base_separation;
                    let normal_mass = (*cp).normal_mass;
                    let lever_arm = (*cp).lever_arm;
                    let normal_impulse = (*cp).normal_impulse;

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
                    (*cp).normal_impulse = new_impulse;
                    (*cp).total_normal_impulse = (*cp).total_normal_impulse + new_impulse;

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
                    let delta = -(*mc).twist_mass * twist_speed;
                    let old = (*mc).twist_impulse;
                    let clamped = clampf(old + delta, -max_impulse, max_impulse);
                    (*mc).twist_impulse = clamped;
                    let applied = clamped - old;
                    w_a = w_a.sub(i_a.mul_v(normal.scale(applied)));
                    w_b = w_b.add(i_b.mul_v(normal.scale(applied)));
                }

                // Rolling resistance.
                if rolling_resistance > 0.0 {
                    let delta = rolling_mass.mul_v(w_b.sub(w_a)).neg();
                    let old = (*mc).rolling_impulse;
                    let mut rolling = old.add(delta);

                    let max_impulse = rolling_resistance * total_normal_impulse;
                    let mag_sqr = rolling.dot(rolling);
                    if mag_sqr > max_impulse * max_impulse + FLT_EPSILON {
                        rolling = rolling.scale(max_impulse / mag_sqr.sqrt());
                    }

                    let applied = rolling.sub(old);
                    (*mc).rolling_impulse = rolling;
                    w_a = w_a.sub(i_a.mul_v(applied));
                    w_b = w_b.add(i_b.mul_v(applied));
                }

                // Central friction.
                {
                    let tangent1 = (*mc).tangent1;
                    let tangent2 = (*mc).tangent2;
                    let r_a = (*mc).center_a;
                    let r_b = (*mc).center_b;
                    let tangent_mass = Mat2 {
                        cx: Vec2::new((*mc).tangent_mass.cx.x, (*mc).tangent_mass.cx.y),
                        cy: Vec2::new((*mc).tangent_mass.cy.x, (*mc).tangent_mass.cy.y),
                    };
                    let tangent_velocity1 = (*mc).tangent_velocity1;
                    let tangent_velocity2 = (*mc).tangent_velocity2;
                    let friction_impulse =
                        Vec2::new((*mc).friction_impulse.x, (*mc).friction_impulse.y);

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
                    (*mc).friction_impulse.x = new_impulse.x;
                    (*mc).friction_impulse.y = new_impulse.y;

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
}

// --- restitution ----------------------------------------------------------------------------

/// Apply restitution bounce to approaching contacts (b3ApplyRestitution_Mesh). Mutates the point
/// normal impulses and the body velocities (dynamic bodies only).
pub fn restitution(cols: &Columns, start: usize, count: usize, threshold: f32) {
    unsafe {
        for c in start..start + count {
            let cc = cols.cc.ptr().add(c);

            let restitution = (*cc).restitution;
            if restitution == 0.0 {
                continue;
            }
            let index_a = (*cc).index_a;
            let index_b = (*cc).index_b;
            let manifold_count = (*cc).manifold_count as usize;

            let m_a = (*cc).inv_mass_a;
            let m_b = (*cc).inv_mass_b;
            let i_a = (*cc).inv_ia;
            let i_b = (*cc).inv_ib;

            let (mut v_a, mut w_a) = read_vel(cols.state, index_a);
            let (mut v_b, mut w_b) = read_vel(cols.state, index_b);

            for mi in 0..manifold_count {
                let mc = (*cc).constraints.add(mi);

                let normal = (*mc).normal;
                let point_count = (*mc).point_count as usize;

                for pi in 0..point_count {
                    let cp = core::ptr::addr_of_mut!((*mc).points[pi]);

                    let relative_velocity = (*cp).relative_velocity;
                    let total_normal_impulse = (*cp).total_normal_impulse;
                    // Skip speculative points that never generated a real impulse.
                    if relative_velocity > -threshold || total_normal_impulse == 0.0 {
                        continue;
                    }
                    let r_a = (*cp).r_a;
                    let r_b = (*cp).r_b;
                    let normal_mass = (*cp).normal_mass;
                    let normal_impulse = (*cp).normal_impulse;

                    let vr_b = v_b.add(w_b.cross(r_b));
                    let vr_a = v_a.add(w_a.cross(r_a));
                    let vn = vr_b.sub(vr_a).dot(normal);

                    let mut impulse = -normal_mass * (vn + restitution * relative_velocity);
                    let new_impulse = maxf(normal_impulse + impulse, 0.0);
                    impulse = new_impulse - normal_impulse;
                    (*cp).normal_impulse = new_impulse;
                    (*cp).total_normal_impulse = (*cp).total_normal_impulse + impulse;

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
}

// --- store ----------------------------------------------------------------------------------

/// Write the solved impulses back into the persistent manifold pool and mark hit events
/// (b3StoreImpulses_Mesh). The pool records are the persistent warm-start state the next step reads;
/// Hits mark the calling worker's bitset for publication after the solve joins.
pub fn store(
    cols: &Columns,
    start: usize,
    count: usize,
    hit_event_threshold: f32,
    mut mark_hit: impl FnMut(usize),
) {
    unsafe {
        const SIM_ENABLE_HIT_EVENT: u32 = 0x0010_0000;
        let neg_hit_threshold = -hit_event_threshold;

        for c in start..start + count {
            let cc = cols.cc.ptr().add(c);
            let contact_id = (*(*cc).contact).contact_id as usize;
            let d = read_dir(cols.dir, contact_id);
            let cols = Columns {
                pool: crate::manifold_abi::block_col(cols.pool, d.manifold_base, d.manifold_count),
                ..*cols
            };
            let manifold_base = 0;

            let manifold_count = (*cc).manifold_count as usize;

            let check_hit_events = (d.flags & SIM_ENABLE_HIT_EVENT) != 0;
            let mut flagged = false;

            for mi in 0..manifold_count {
                let mc = (*cc).constraints.add(mi);
                let mpo = (manifold_base + mi) * MANIFOLD_STRIDE; // persistent pool record

                let tangent1 = (*mc).tangent1;
                let tangent2 = (*mc).tangent2;
                let friction = blend2(
                    (*mc).friction_impulse.x,
                    tangent1,
                    (*mc).friction_impulse.y,
                    tangent2,
                );
                cols.pool.set(mpo + M_TWIST, (*mc).twist_impulse); // twistImpulse
                write_v3(cols.pool, mpo + M_FRICTION, friction); // frictionImpulse
                write_v3(cols.pool, mpo + M_ROLLING, (*mc).rolling_impulse); // rollingImpulse

                let point_count = (*mc).point_count as usize;

                for pi in 0..point_count {
                    let cp = core::ptr::addr_of_mut!((*mc).points[pi]);
                    let pp = mpo + M_POINTS + pi * POOL_POINT_STRIDE; // pool point record

                    let normal_impulse = (*cp).normal_impulse;
                    let total_normal_impulse = (*cp).total_normal_impulse;
                    let normal_velocity = (*cp).relative_velocity;
                    cols.pool.set(pp + P_NORMAL_IMPULSE, normal_impulse);
                    cols.pool
                        .set(pp + P_TOTAL_NORMAL_IMPULSE, total_normal_impulse);
                    cols.pool.set(pp + P_NORMAL_VELOCITY, normal_velocity);

                    // One flag per contact: a confirmed impulse approaching faster than the threshold.
                    if check_hit_events
                        && !flagged
                        && normal_velocity < neg_hit_threshold
                        && total_normal_impulse > 0.0
                    {
                        mark_hit(contact_id);
                        flagged = true;
                    }
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
    if index != NULL_INDEX
        && flags_col.get(index as usize * STATE_STRIDE) & body_flags::DYNAMIC != 0
    {
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
