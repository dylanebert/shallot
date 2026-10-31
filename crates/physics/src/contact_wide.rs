//! Four-lane convex contact constraints.

use crate::body::flags as body_flags;
use crate::body::{read_sim, read_state, STATE_STRIDE};
use crate::col::Col;
use crate::contact::{Softness, NULL_INDEX};
use crate::manifold_abi as mabi;
use crate::manifold_abi::read_dir;
use crate::math::{Mat2, Mat3, Vec2, Vec3, FLT_EPSILON};

const SPECULATIVE_DISTANCE: f32 = 0.02;
const MIN_FRICTION_WEIGHT: f32 = 1.0e-10;
use crate::simd::FloatW;
use crate::wide::{
    add_v2w, add_vw, cross_w, dot_w, inv_rotate_vector_w, mul_add_mvw, mul_add_svw, mul_mv2w,
    mul_mvw, mul_sub_mvw, mul_sub_svw, mul_svw, rotate_vector_w, sub_vw, sym_clamp, QuatW,
    SymMatrix2W, SymMatrix3W, Vec2W, Vec3W,
};
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// Max manifold points (B3_MAX_MANIFOLD_POINTS). Convex contacts have one manifold with up to this
/// many points; the wide constraint always carries all four slots (unused zeroed).
pub const MAX_POINTS: usize = 4;
/// SIMD lane width (B3_SIMD_WIDTH).
pub const LANES: usize = 4;

#[repr(C, align(16))]
pub struct ContactConstraintWide {
    index_a: [u32; LANES],
    index_b: [u32; LANES],
    point_counts: [i32; LANES],
    inv_mass_a: [f32; 4],
    inv_mass_b: [f32; 4],
    inv_ia: [f32; 24],
    inv_ib: [f32; 24],
    normal: [f32; 12],
    tangent1: [f32; 12],
    tangent2: [f32; 12],
    center_a: [f32; 12],
    center_b: [f32; 12],
    twist_mass: [f32; 4],
    twist_impulse: [f32; 4],
    tangent_mass: [f32; 12],
    friction_impulse: [f32; 8],
    rolling_mass: [f32; 24],
    rolling_impulse: [f32; 12],
    friction: [f32; 4],
    rolling_resistance: [f32; 4],
    tangent_velocity1: [f32; 4],
    tangent_velocity2: [f32; 4],
    bias_rate: [f32; 4],
    mass_scale: [f32; 4],
    impulse_scale: [f32; 4],
    restitution: [f32; 4],
    manifolds: [*mut mabi::ManifoldRecord; LANES],
    points: [[f32; POINT_STRIDE]; MAX_POINTS],
}

use core::mem::{offset_of, size_of};

const INV_MASS_A: usize = offset_of!(ContactConstraintWide, inv_mass_a) / 4;
const INV_MASS_B: usize = offset_of!(ContactConstraintWide, inv_mass_b) / 4;
const INV_IA: usize = offset_of!(ContactConstraintWide, inv_ia) / 4;
const INV_IB: usize = offset_of!(ContactConstraintWide, inv_ib) / 4;
const NORMAL: usize = offset_of!(ContactConstraintWide, normal) / 4;
const TANGENT1: usize = offset_of!(ContactConstraintWide, tangent1) / 4;
const TANGENT2: usize = offset_of!(ContactConstraintWide, tangent2) / 4;
const ORIGIN_A: usize = offset_of!(ContactConstraintWide, center_a) / 4;
const ORIGIN_B: usize = offset_of!(ContactConstraintWide, center_b) / 4;
const TWIST_MASS: usize = offset_of!(ContactConstraintWide, twist_mass) / 4;
const TWIST_IMPULSE: usize = offset_of!(ContactConstraintWide, twist_impulse) / 4;
const TANGENT_MASS: usize = offset_of!(ContactConstraintWide, tangent_mass) / 4;
const FRICTION_IMPULSE: usize = offset_of!(ContactConstraintWide, friction_impulse) / 4;
const ROLLING_MASS: usize = offset_of!(ContactConstraintWide, rolling_mass) / 4;
const ROLLING_IMPULSE: usize = offset_of!(ContactConstraintWide, rolling_impulse) / 4;
const FRICTION: usize = offset_of!(ContactConstraintWide, friction) / 4;
const ROLLING_RESISTANCE: usize = offset_of!(ContactConstraintWide, rolling_resistance) / 4;
const TANGENT_VELOCITY1: usize = offset_of!(ContactConstraintWide, tangent_velocity1) / 4;
const TANGENT_VELOCITY2: usize = offset_of!(ContactConstraintWide, tangent_velocity2) / 4;
const BIAS_RATE: usize = offset_of!(ContactConstraintWide, bias_rate) / 4;
const MASS_SCALE: usize = offset_of!(ContactConstraintWide, mass_scale) / 4;
const IMPULSE_SCALE: usize = offset_of!(ContactConstraintWide, impulse_scale) / 4;
const RESTITUTION: usize = offset_of!(ContactConstraintWide, restitution) / 4;
const POINTS: usize = offset_of!(ContactConstraintWide, points) / 4;
const POINT_COUNTS: usize = offset_of!(ContactConstraintWide, point_counts) / 4;
const MANIFOLDS: usize = offset_of!(ContactConstraintWide, manifolds) / 4;
const POINT_STRIDE: usize = 52;
// point sub-offsets (relative to a point's base)
const P_ANCHOR_A: usize = 0;
const P_ANCHOR_B: usize = 12;
const P_BASE_SEP: usize = 24;
const P_NORMAL_IMP: usize = 28;
const P_TOTAL_NORMAL_IMP: usize = 32;
const P_NORMAL_MASS: usize = 36;
const P_LEVER_ARM: usize = 40;
const P_REL_VEL: usize = 44;
const P_RESTITUTION_IMP: usize = 48;

pub const WIDE_STRIDE: usize = size_of::<ContactConstraintWide>() / 4;

#[inline]
fn ld(col: Col<f32>, o: usize) -> FloatW {
    FloatW::set(col.get(o), col.get(o + 1), col.get(o + 2), col.get(o + 3))
}
#[inline]
fn st(col: Col<f32>, o: usize, v: FloatW) {
    let a = v.to_array();
    col.set(o, a[0]);
    col.set(o + 1, a[1]);
    col.set(o + 2, a[2]);
    col.set(o + 3, a[3]);
}
#[inline]
fn ld_v3(col: Col<f32>, o: usize) -> Vec3W {
    Vec3W {
        x: ld(col, o),
        y: ld(col, o + 4),
        z: ld(col, o + 8),
    }
}
#[inline]
fn st_v3(col: Col<f32>, o: usize, v: Vec3W) {
    st(col, o, v.x);
    st(col, o + 4, v.y);
    st(col, o + 8, v.z);
}
#[inline]
fn ld_v2(col: Col<f32>, o: usize) -> Vec2W {
    Vec2W {
        x: ld(col, o),
        y: ld(col, o + 4),
    }
}
#[inline]
fn st_v2(col: Col<f32>, o: usize, v: Vec2W) {
    st(col, o, v.x);
    st(col, o + 4, v.y);
}
#[inline]
fn ld_sym3(col: Col<f32>, o: usize) -> SymMatrix3W {
    SymMatrix3W {
        cxx: ld(col, o),
        cxy: ld(col, o + 4),
        cxz: ld(col, o + 8),
        cyy: ld(col, o + 12),
        cyz: ld(col, o + 16),
        czz: ld(col, o + 20),
    }
}
#[inline]
fn ld_sym2(col: Col<f32>, o: usize) -> SymMatrix2W {
    SymMatrix2W {
        cxx: ld(col, o),
        cxy: ld(col, o + 4),
        cyy: ld(col, o + 8),
    }
}

#[inline]
fn st_lane_v3(wide: Col<f32>, o: usize, lane: usize, v: Vec3) {
    wide.set(o + lane, v.x);
    wide.set(o + LANES + lane, v.y);
    wide.set(o + 2 * LANES + lane, v.z);
}

#[inline]
fn st_lane_sym3(wide: Col<f32>, o: usize, lane: usize, m: Mat3) {
    for (component, value) in [m.cx.x, m.cx.y, m.cx.z, m.cy.y, m.cy.z, m.cz.z]
        .into_iter()
        .enumerate()
    {
        wide.set(o + component * LANES + lane, value);
    }
}

// Prepare gathers one lane at a time from Shallot's split directory/pool. Keep the wide C
// symmetric matrix arithmetic, including its right-associated sums, in each scalar lane.
#[inline]
fn mul_sym(m: Mat3, a: Vec3) -> Vec3 {
    Vec3::new(
        m.cx.x * a.x + (m.cx.y * a.y + m.cx.z * a.z),
        m.cx.y * a.x + (m.cy.y * a.y + m.cy.z * a.z),
        m.cx.z * a.x + (m.cy.z * a.y + m.cz.z * a.z),
    )
}

fn invert_sym(m: Mat3) -> Mat3 {
    let cxx = m.cy.y * m.cz.z - m.cy.z * m.cy.z;
    let cxy = m.cx.z * m.cy.z - m.cx.y * m.cz.z;
    let cxz = m.cx.y * m.cy.z - m.cx.z * m.cy.y;
    let cyy = m.cx.x * m.cz.z - m.cx.z * m.cx.z;
    let cyz = m.cx.y * m.cx.z - m.cx.x * m.cy.z;
    let czz = m.cx.x * m.cy.y - m.cx.y * m.cx.y;
    let det = m.cx.x * cxx + (m.cx.y * cxy + m.cx.z * cxz);
    let inv_det = if det.abs() > 1000.0 * f32::MIN_POSITIVE {
        1.0 / det
    } else {
        0.0
    };
    Mat3 {
        cx: Vec3::new(inv_det * cxx, inv_det * cxy, inv_det * cxz),
        cy: Vec3::new(inv_det * cxy, inv_det * cyy, inv_det * cyz),
        cz: Vec3::new(inv_det * cxz, inv_det * cyz, inv_det * czz),
    }
}

#[inline]
fn st_lane_sym2(wide: Col<f32>, o: usize, lane: usize, m: Mat2) {
    wide.set(o + lane, m.cx.x);
    wide.set(o + LANES + lane, m.cx.y);
    wide.set(o + 2 * LANES + lane, m.cy.y);
}

#[inline]
fn v3(col: Col<f32>, o: usize) -> Vec3 {
    Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2))
}

/// Body mass/inertia/velocity terms for prepare, zeroed for a static (null-index) body.
fn body_terms(sim: Col<f32>, state: Col<f32>, index: u32) -> (f32, Mat3, Vec3, Vec3) {
    if index == NULL_INDEX {
        (0.0, Mat3::ZERO, Vec3::ZERO, Vec3::ZERO)
    } else {
        let s = read_sim(sim, index as usize);
        let st = read_state(state, index as usize);
        (
            s.inv_mass,
            s.inv_inertia_world,
            st.linear_velocity,
            st.angular_velocity,
        )
    }
}

// Null gathers select a local dummy; scatters skip null lanes so no shared dummy is written.

/// Wide body solver state (b3BodyStateW): the four gathered bodies' velocities + deltas across lanes.
struct BodyStateW {
    v: Vec3W,
    w: Vec3W,
    dp: Vec3W,
    dq: QuatW,
}

// Box3D contact_solver.c's SIMD scatter writes only velocities; explicit lane blocks keep null and
// non-dynamic bodies out of the write set.
fn scatter(state: Col<f32>, flags: Col<u32>, idx: Col<u32>, io: usize, v: &Vec3W, w: &Vec3W) {
    let vx = v.x.to_array();
    let vy = v.y.to_array();
    let vz = v.z.to_array();
    let wx = w.x.to_array();
    let wy = w.y.to_array();
    let wz = w.z.to_array();
    let scatter_lane = |lane: usize| {
        let i = idx.get(io + lane);
        if i == 0 {
            return;
        }
        let b = (i - 1) as usize;
        let f = flags.get(b * STATE_STRIDE);
        if f & body_flags::DYNAMIC == 0 {
            return;
        }
        let v = [vx[lane], vy[lane], vz[lane]];
        let w = [wx[lane], wy[lane], wz[lane]];
        let o = b * STATE_STRIDE;
        state.set(o, v[0]);
        state.set(o + 1, v[1]);
        state.set(o + 2, v[2]);
        state.set(o + 3, w[0]);
        state.set(o + 4, w[1]);
        state.set(o + 5, w[2]);
    };
    scatter_lane(0);
    scatter_lane(1);
    scatter_lane(2);
    scatter_lane(3);
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn fw(a: [f32; 4]) -> FloatW {
    FloatW::set(a[0], a[1], a[2], a[3])
}

/// Gather four bodies' full solver state into lanes (b3GatherBodies). Index 0 is the null/static body,
/// contributing an identity state (zero velocity/delta, identity rotation).
#[cfg(not(target_arch = "wasm32"))]
fn gather(state: Col<f32>, idx: Col<u32>, io: usize) -> BodyStateW {
    let mut vx = [0.0f32; 4];
    let mut vy = [0.0f32; 4];
    let mut vz = [0.0f32; 4];
    let mut wx = [0.0f32; 4];
    let mut wy = [0.0f32; 4];
    let mut wz = [0.0f32; 4];
    let mut dpx = [0.0f32; 4];
    let mut dpy = [0.0f32; 4];
    let mut dpz = [0.0f32; 4];
    let mut qx = [0.0f32; 4];
    let mut qy = [0.0f32; 4];
    let mut qz = [0.0f32; 4];
    let mut qs = [1.0f32; 4]; // identity rotation for null lanes
    for lane in 0..LANES {
        let i = idx.get(io + lane);
        if i == 0 {
            continue;
        }
        let s = read_state(state, (i - 1) as usize);
        vx[lane] = s.linear_velocity.x;
        vy[lane] = s.linear_velocity.y;
        vz[lane] = s.linear_velocity.z;
        wx[lane] = s.angular_velocity.x;
        wy[lane] = s.angular_velocity.y;
        wz[lane] = s.angular_velocity.z;
        dpx[lane] = s.delta_position.x;
        dpy[lane] = s.delta_position.y;
        dpz[lane] = s.delta_position.z;
        qx[lane] = s.delta_rotation.v.x;
        qy[lane] = s.delta_rotation.v.y;
        qz[lane] = s.delta_rotation.v.z;
        qs[lane] = s.delta_rotation.s;
    }
    BodyStateW {
        v: Vec3W {
            x: fw(vx),
            y: fw(vy),
            z: fw(vz),
        },
        w: Vec3W {
            x: fw(wx),
            y: fw(wy),
            z: fw(wz),
        },
        dp: Vec3W {
            x: fw(dpx),
            y: fw(dpy),
            z: fw(dpz),
        },
        dq: QuatW {
            v: Vec3W {
                x: fw(qx),
                y: fw(qy),
                z: fw(qz),
            },
            s: fw(qs),
        },
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn gather_vel(state: Col<f32>, idx: Col<u32>, io: usize) -> (Vec3W, Vec3W) {
    let mut vx = [0.0f32; 4];
    let mut vy = [0.0f32; 4];
    let mut vz = [0.0f32; 4];
    let mut wx = [0.0f32; 4];
    let mut wy = [0.0f32; 4];
    let mut wz = [0.0f32; 4];
    for lane in 0..LANES {
        let i = idx.get(io + lane);
        if i == 0 {
            continue;
        }
        let s = read_state(state, (i - 1) as usize);
        vx[lane] = s.linear_velocity.x;
        vy[lane] = s.linear_velocity.y;
        vz[lane] = s.linear_velocity.z;
        wx[lane] = s.angular_velocity.x;
        wy[lane] = s.angular_velocity.y;
        wz[lane] = s.angular_velocity.z;
    }
    (
        Vec3W {
            x: fw(vx),
            y: fw(vy),
            z: fw(vz),
        },
        Vec3W {
            x: fw(wx),
            y: fw(wy),
            z: fw(wz),
        },
    )
}

/// 4×4 f32 transpose (unpcklps/unpckhps + movlhps/movhlps decomposition).
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn transpose4(a: v128, b: v128, c: v128, d: v128) -> (v128, v128, v128, v128) {
    let t0 = i32x4_shuffle::<0, 4, 1, 5>(a, b); // a0 b0 a1 b1
    let t1 = i32x4_shuffle::<2, 6, 3, 7>(a, b); // a2 b2 a3 b3
    let t2 = i32x4_shuffle::<0, 4, 1, 5>(c, d); // c0 d0 c1 d1
    let t3 = i32x4_shuffle::<2, 6, 3, 7>(c, d); // c2 d2 c3 d3
    let r0 = i32x4_shuffle::<0, 1, 4, 5>(t0, t2); // a0 b0 c0 d0
    let r1 = i32x4_shuffle::<2, 3, 6, 7>(t0, t2); // a1 b1 c1 d1
    let r2 = i32x4_shuffle::<0, 1, 4, 5>(t1, t3); // a2 b2 c2 d2
    let r3 = i32x4_shuffle::<2, 3, 6, 7>(t1, t3); // a3 b3 c3 d3
    (r0, r1, r2, r3)
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn gather_body(state: Col<f32>, index: u32, dummy: *const f32) -> *const f32 {
    if index == 0 {
        dummy
    } else {
        unsafe { state.ptr().add((index - 1) as usize * STATE_STRIDE) }
    }
}

/// Gathers full solver state; null lanes contribute the identity state.
#[cfg(target_arch = "wasm32")]
fn gather(state: Col<f32>, idx: Col<u32>, io: usize) -> BodyStateW {
    let mut dummy = [0.0f32; STATE_STRIDE];
    dummy[12] = 1.0;
    unsafe {
        let p0 = gather_body(state, idx.get(io), dummy.as_ptr());
        let p1 = gather_body(state, idx.get(io + 1), dummy.as_ptr());
        let p2 = gather_body(state, idx.get(io + 2), dummy.as_ptr());
        let p3 = gather_body(state, idx.get(io + 3), dummy.as_ptr());
        let (vx, vy, vz, wx) = transpose4(
            v128_load(p0 as *const v128),
            v128_load(p1 as *const v128),
            v128_load(p2 as *const v128),
            v128_load(p3 as *const v128),
        );
        let (wy, wz, dpx, dpy) = transpose4(
            v128_load(p0.add(4) as *const v128),
            v128_load(p1.add(4) as *const v128),
            v128_load(p2.add(4) as *const v128),
            v128_load(p3.add(4) as *const v128),
        );
        let (dpz, qx, qy, qz) = transpose4(
            v128_load(p0.add(8) as *const v128),
            v128_load(p1.add(8) as *const v128),
            v128_load(p2.add(8) as *const v128),
            v128_load(p3.add(8) as *const v128),
        );
        // Only dq.s is live; a full vector load would run past the 56-byte state or local dummy.
        let (qs, _, _, _) = transpose4(
            v128_load32_zero(p0.add(12).cast()),
            v128_load32_zero(p1.add(12).cast()),
            v128_load32_zero(p2.add(12).cast()),
            v128_load32_zero(p3.add(12).cast()),
        );
        BodyStateW {
            v: Vec3W {
                x: FloatW::from_v128(vx),
                y: FloatW::from_v128(vy),
                z: FloatW::from_v128(vz),
            },
            w: Vec3W {
                x: FloatW::from_v128(wx),
                y: FloatW::from_v128(wy),
                z: FloatW::from_v128(wz),
            },
            dp: Vec3W {
                x: FloatW::from_v128(dpx),
                y: FloatW::from_v128(dpy),
                z: FloatW::from_v128(dpz),
            },
            dq: QuatW {
                v: Vec3W {
                    x: FloatW::from_v128(qx),
                    y: FloatW::from_v128(qy),
                    z: FloatW::from_v128(qz),
                },
                s: FloatW::from_v128(qs),
            },
        }
    }
}

/// Warm start reads velocities only; delta fields are not loaded.
#[cfg(target_arch = "wasm32")]
fn gather_vel(state: Col<f32>, idx: Col<u32>, io: usize) -> (Vec3W, Vec3W) {
    let mut dummy = [0.0f32; STATE_STRIDE];
    dummy[12] = 1.0;
    unsafe {
        let p0 = gather_body(state, idx.get(io), dummy.as_ptr());
        let p1 = gather_body(state, idx.get(io + 1), dummy.as_ptr());
        let p2 = gather_body(state, idx.get(io + 2), dummy.as_ptr());
        let p3 = gather_body(state, idx.get(io + 3), dummy.as_ptr());
        let (vx, vy, vz, wx) = transpose4(
            v128_load(p0 as *const v128),
            v128_load(p1 as *const v128),
            v128_load(p2 as *const v128),
            v128_load(p3 as *const v128),
        );
        let (wy, wz, _, _) = transpose4(
            v128_load(p0.add(4) as *const v128),
            v128_load(p1.add(4) as *const v128),
            v128_load(p2.add(4) as *const v128),
            v128_load(p3.add(4) as *const v128),
        );
        (
            Vec3W {
                x: FloatW::from_v128(vx),
                y: FloatW::from_v128(vy),
                z: FloatW::from_v128(vz),
            },
            Vec3W {
                x: FloatW::from_v128(wx),
                y: FloatW::from_v128(wy),
                z: FloatW::from_v128(wz),
            },
        )
    }
}

/// Build the wide constraints for the records in `[start, start+count)` (b3PrepareContacts_Convex).
/// Gathers each active lane's convex contact (one manifold) through its contactId → the persistent
/// directory + pool, plus body sim/state; writes the wide + index columns. `warm_start_scale` is 1
/// (warm starting on) or 0.
#[allow(clippy::too_many_arguments)]
pub fn prepare(
    wide: Col<f32>,
    idx: Col<u32>,
    spans: Col<crate::contact_spans::WidePrepareSpan>,
    state: Col<f32>,
    sim: Col<f32>,
    dir: Col<u32>,
    pool: Col<f32>,
    start: usize,
    count: usize,
    contact_softness: Softness,
    static_softness: Softness,
    warm_start_scale: f32,
) -> bool {
    let mut any_restitution = false;
    for (r, span) in crate::contact_spans::wide_records(spans, start, count) {
        let wo = r * WIDE_STRIDE;
        let io = r * WIDE_STRIDE;
        let local = (r - span.start as usize) * LANES;
        let lane_count = (span.count as usize - local).min(LANES);
        let mut any_rolling = false;
        let mut sample_velocity = false;
        for lane in 0..lane_count {
            let d = read_dir(dir, unsafe { *span.contacts.add(local + lane) as usize });
            any_rolling |= d.rolling_resistance > 0.0;
            any_restitution |= d.restitution > 0.0;
            sample_velocity |=
                d.restitution > 0.0 || d.flags & mabi::contact_flags::SIM_ENABLE_HIT_EVENT != 0;
        }

        for lane in 0..lane_count {
            let contact_id = unsafe { *span.contacts.add(local + lane) as usize };
            let d = read_dir(dir, contact_id);
            let index_a = d.index_a;
            let index_b = d.index_b;
            let pool = mabi::block_col(pool, d.manifold_base, 1);
            unsafe {
                wide.ptr()
                    .add(wo + MANIFOLDS)
                    .cast::<*mut mabi::ManifoldRecord>()
                    .add(lane)
                    .write(pool.ptr().cast());
            }
            let mpo = 0; // convex: exactly one manifold

            idx.set(io + lane, index_a.wrapping_add(1));
            idx.set(io + LANES + lane, index_b.wrapping_add(1));

            let (m_a, i_a, v_a, w_a) = body_terms(sim, state, index_a);
            let (m_b, i_b, v_b, w_b) = body_terms(sim, state, index_b);
            wide.set(wo + INV_MASS_A + lane, m_a);
            wide.set(wo + INV_MASS_B + lane, m_b);
            st_lane_sym3(wide, wo + INV_IA, lane, i_a);
            st_lane_sym3(wide, wo + INV_IB, lane, i_b);

            wide.set(wo + FRICTION + lane, d.friction);
            wide.set(wo + RESTITUTION + lane, d.restitution);
            wide.set(wo + ROLLING_RESISTANCE + lane, d.rolling_resistance);
            let tangent_velocity = d.tangent_velocity;

            let n = v3(pool, mpo + mabi::M_NORMAL);
            let t1 = n.perp();
            let t2 = t1.cross(n);
            st_lane_v3(wide, wo + NORMAL, lane, n);
            st_lane_v3(wide, wo + TANGENT1, lane, t1);
            st_lane_v3(wide, wo + TANGENT2, lane, t2);
            wide.set(wo + TANGENT_VELOCITY1 + lane, tangent_velocity.dot(t1));
            wide.set(wo + TANGENT_VELOCITY2 + lane, tangent_velocity.dot(t2));

            let soft = if index_a == NULL_INDEX || index_b == NULL_INDEX {
                static_softness
            } else {
                contact_softness
            };
            wide.set(wo + BIAS_RATE + lane, soft.bias_rate);
            wide.set(wo + MASS_SCALE + lane, soft.mass_scale);
            wide.set(wo + IMPULSE_SCALE + lane, soft.impulse_scale);

            let point_count = pool.get(mpo + mabi::M_POINT_COUNT).to_bits() as usize;
            idx.set(io + POINT_COUNTS + lane, point_count as u32);

            let mut center_a = Vec3::ZERO;
            let mut center_b = Vec3::ZERO;
            let mut total_friction_weight = 0.0f32;
            let inv_tau = 1.0f32 / SPECULATIVE_DISTANCE;
            for pi in 0..point_count {
                let pp = mpo + mabi::M_POINTS + pi * mabi::POOL_POINT_STRIDE;
                let r_a = v3(pool, pp + mabi::P_ANCHOR_A);
                let r_b = v3(pool, pp + mabi::P_ANCHOR_B);
                let separation = pool.get(pp + mabi::P_SEPARATION);
                let mp_normal_impulse = pool.get(pp + mabi::P_NORMAL_IMPULSE);

                let weight = (2.0 - separation * inv_tau).clamp(MIN_FRICTION_WEIGHT, 1.0);
                center_a = center_a.add(r_a.scale(weight));
                center_b = center_b.add(r_b.scale(weight));
                total_friction_weight += weight;

                let pb = wo + POINTS + pi * POINT_STRIDE;
                st_lane_v3(wide, pb + P_ANCHOR_A, lane, r_a);
                st_lane_v3(wide, pb + P_ANCHOR_B, lane, r_b);
                wide.set(pb + P_BASE_SEP + lane, separation - r_b.sub(r_a).dot(n));
                wide.set(
                    pb + P_NORMAL_IMP + lane,
                    warm_start_scale * mp_normal_impulse,
                );
                wide.set(pb + P_TOTAL_NORMAL_IMP + lane, 0.0);
                wide.set(pb + P_RESTITUTION_IMP + lane, 0.0);

                let rn_a = r_a.cross(n);
                let rn_b = r_b.cross(n);
                let k_normal =
                    m_a + m_b + rn_a.dot(mul_sym(i_a, rn_a)) + rn_b.dot(mul_sym(i_b, rn_b));
                wide.set(
                    pb + P_NORMAL_MASS + lane,
                    if k_normal > 0.0 { 1.0 / k_normal } else { 0.0 },
                );
                let vr_a = v_a.add(w_a.cross(r_a));
                let vr_b = v_b.add(w_b.cross(r_b));
                let hit_events = d.flags & mabi::contact_flags::SIM_ENABLE_HIT_EVENT != 0;
                let vn = if sample_velocity {
                    n.dot(vr_b.sub(vr_a))
                } else {
                    0.0
                };
                wide.set(pb + P_REL_VEL + lane, vn);
                pool.set(
                    pp + mabi::P_NORMAL_VELOCITY,
                    if hit_events { vn } else { 0.0 },
                );
            }
            let inv_weight = 1.0 / total_friction_weight;
            center_a = center_a.scale(inv_weight);
            center_b = center_b.scale(inv_weight);
            st_lane_v3(wide, wo + ORIGIN_A, lane, center_a);
            st_lane_v3(wide, wo + ORIGIN_B, lane, center_b);
            for pi in 0..point_count {
                let pp = mpo + mabi::M_POINTS + pi * mabi::POOL_POINT_STRIDE;
                let pb = wo + POINTS + pi * POINT_STRIDE;
                wide.set(
                    pb + P_LEVER_ARM + lane,
                    v3(pool, pp + mabi::P_ANCHOR_A).distance(center_a),
                );
            }

            let rt_a1 = center_a.cross(t1);
            let rt_a2 = center_a.cross(t2);
            let rt_b1 = center_b.cross(t1);
            let rt_b2 = center_b.cross(t2);
            let kxx = m_a + m_b + rt_a1.dot(mul_sym(i_a, rt_a1)) + rt_b1.dot(mul_sym(i_b, rt_b1));
            let kyy = m_a + m_b + rt_a2.dot(mul_sym(i_a, rt_a2)) + rt_b2.dot(mul_sym(i_b, rt_b2));
            let kxy = rt_a1.dot(mul_sym(i_a, rt_a2)) + rt_b1.dot(mul_sym(i_b, rt_b2));
            st_lane_sym2(
                wide,
                wo + TANGENT_MASS,
                lane,
                Mat2 {
                    cx: Vec2::new(kxx, kxy),
                    cy: Vec2::new(kxy, kyy),
                }
                .invert(),
            );

            let mf_friction_impulse = v3(pool, mpo + mabi::M_FRICTION);
            wide.set(
                wo + FRICTION_IMPULSE + lane,
                warm_start_scale * mf_friction_impulse.dot(t1),
            );
            wide.set(
                wo + FRICTION_IMPULSE + LANES + lane,
                warm_start_scale * mf_friction_impulse.dot(t2),
            );

            let iab = i_a.add(i_b);
            let twist_k = n.dot(mul_sym(iab, n));
            wide.set(
                wo + TWIST_MASS + lane,
                if twist_k > 0.0 { 1.0 / twist_k } else { 0.0 },
            );
            wide.set(
                wo + TWIST_IMPULSE + lane,
                warm_start_scale * pool.get(mpo + mabi::M_TWIST),
            );
            st_lane_sym3(
                wide,
                wo + ROLLING_MASS,
                lane,
                if any_rolling {
                    invert_sym(iab)
                } else {
                    Mat3::ZERO
                },
            );
            st_lane_v3(
                wide,
                wo + ROLLING_IMPULSE,
                lane,
                if d.rolling_resistance > 0.0 {
                    v3(pool, mpo + mabi::M_ROLLING).scale(warm_start_scale)
                } else {
                    Vec3::ZERO
                },
            );
            for pi in point_count..MAX_POINTS {
                let pb = wo + POINTS + pi * POINT_STRIDE;
                for component in (0..POINT_STRIDE).step_by(LANES) {
                    wide.set(pb + component + lane, 0.0);
                }
            }
        }
    }
    any_restitution
}

/// Seed body velocities from the warm-start impulses (b3WarmStartContacts_Convex) for records
/// `[start, start+count)`.
pub fn warm_start(
    wide: Col<f32>,
    idx: Col<u32>,
    state: Col<f32>,
    flags: Col<u32>,
    start: usize,
    count: usize,
) {
    for r in start..start + count {
        let wo = r * WIDE_STRIDE;
        let io = r * WIDE_STRIDE;
        let point_count = idx
            .get(io + POINT_COUNTS)
            .max(idx.get(io + POINT_COUNTS + 1))
            .max(
                idx.get(io + POINT_COUNTS + 2)
                    .max(idx.get(io + POINT_COUNTS + 3)),
            ) as usize;
        let ia = ld_sym3(wide, wo + INV_IA);
        let ib = ld_sym3(wide, wo + INV_IB);
        let inv_ma = ld(wide, wo + INV_MASS_A);
        let inv_mb = ld(wide, wo + INV_MASS_B);
        let normal = ld_v3(wide, wo + NORMAL);
        let (mut ba_v, mut ba_w) = gather_vel(state, idx, io);
        let (mut bb_v, mut bb_w) = gather_vel(state, idx, io + LANES);

        let mut total_normal_impulse = FloatW::zero();
        let mut moment_a = Vec3W::splat0();
        let mut moment_b = Vec3W::splat0();
        for pi in 0..point_count {
            let pb = wo + POINTS + pi * POINT_STRIDE;
            let impulse = ld(wide, pb + P_NORMAL_IMP);
            total_normal_impulse = total_normal_impulse.add(impulse);
            moment_a = mul_add_svw(moment_a, impulse, ld_v3(wide, pb + P_ANCHOR_A));
            moment_b = mul_add_svw(moment_b, impulse, ld_v3(wide, pb + P_ANCHOR_B));
        }
        let fi = ld_v2(wide, wo + FRICTION_IMPULSE);
        let friction_impulse = mul_add_svw(
            mul_svw(fi.x, ld_v3(wide, wo + TANGENT1)),
            fi.y,
            ld_v3(wide, wo + TANGENT2),
        );
        let linear_impulse = mul_add_svw(friction_impulse, total_normal_impulse, normal);
        let twist = mul_svw(ld(wide, wo + TWIST_IMPULSE), normal);
        let mut angular_a = add_vw(
            add_vw(
                cross_w(moment_a, normal),
                cross_w(ld_v3(wide, wo + ORIGIN_A), friction_impulse),
            ),
            twist,
        );
        let mut angular_b = add_vw(
            add_vw(
                cross_w(moment_b, normal),
                cross_w(ld_v3(wide, wo + ORIGIN_B), friction_impulse),
            ),
            twist,
        );
        if !ld(wide, wo + ROLLING_RESISTANCE).all_zero() {
            let rolling = ld_v3(wide, wo + ROLLING_IMPULSE);
            angular_a = add_vw(angular_a, rolling);
            angular_b = add_vw(angular_b, rolling);
        }
        ba_w = mul_sub_mvw(ba_w, ia, angular_a);
        ba_v = mul_sub_svw(ba_v, inv_ma, linear_impulse);
        bb_w = mul_add_mvw(bb_w, ib, angular_b);
        bb_v = mul_add_svw(bb_v, inv_mb, linear_impulse);

        scatter(state, flags, idx, io, &ba_v, &ba_w);
        scatter(state, flags, idx, io + LANES, &bb_v, &bb_w);
    }
}

/// One TGS solve/relax pass over records `[start, start+count)` (b3SolveContacts_Convex). `use_bias`
/// selects the biased solve (position drift removal, friction skipped) vs the relax pass (no bias,
/// friction applied). `inv_h` is the inverse sub-step, `contact_speed` the max separation speed.
#[allow(clippy::too_many_arguments)]
pub fn solve(
    wide: Col<f32>,
    idx: Col<u32>,
    state: Col<f32>,
    flags: Col<u32>,
    start: usize,
    count: usize,
    use_bias: bool,
    inv_h: f32,
    contact_speed: f32,
) {
    let inv_h_w = FloatW::splat(inv_h);
    let contact_speed_w = FloatW::splat(-contact_speed);
    let one = FloatW::splat(1.0);
    let zero = FloatW::zero();
    let eps = FloatW::splat(FLT_EPSILON);

    for r in start..start + count {
        let wo = r * WIDE_STRIDE;
        let io = r * WIDE_STRIDE;
        let point_count = idx
            .get(io + POINT_COUNTS)
            .max(idx.get(io + POINT_COUNTS + 1))
            .max(
                idx.get(io + POINT_COUNTS + 2)
                    .max(idx.get(io + POINT_COUNTS + 3)),
            ) as usize;
        let mut ba = gather(state, idx, io);
        let mut bb = gather(state, idx, io + LANES);

        let ia = ld_sym3(wide, wo + INV_IA);
        let ib = ld_sym3(wide, wo + INV_IB);
        let inv_ma = ld(wide, wo + INV_MASS_A);
        let inv_mb = ld(wide, wo + INV_MASS_B);
        let normal = ld_v3(wide, wo + NORMAL);

        let (bias_rate, mass_scale, impulse_scale) = if use_bias {
            (
                ld(wide, wo + MASS_SCALE).mul(ld(wide, wo + BIAS_RATE)),
                ld(wide, wo + MASS_SCALE),
                ld(wide, wo + IMPULSE_SCALE),
            )
        } else {
            (zero, one, zero)
        };

        let dp = sub_vw(bb.dp, ba.dp);
        let normal_separation = dot_w(normal, dp);
        let normal_a = inv_rotate_vector_w(ba.dq, normal);
        let normal_b = inv_rotate_vector_w(bb.dq, normal);
        let mut total_normal_impulse = zero;
        let mut total_twist_limit = zero;

        for pi in 0..point_count {
            let pb = wo + POINTS + pi * POINT_STRIDE;
            let ra = ld_v3(wide, pb + P_ANCHOR_A);
            let rb = ld_v3(wide, pb + P_ANCHOR_B);

            let s = normal_separation
                .add(dot_w(normal_b, rb).sub(dot_w(normal_a, ra)))
                .add(ld(wide, pb + P_BASE_SEP));

            let mask = s.greater_than(zero);
            let spec_bias = s.mul(inv_h_w);
            let soft_bias = bias_rate.mul(s).max(contact_speed_w);
            let bias = if use_bias {
                FloatW::blend(soft_bias, spec_bias, mask)
            } else {
                zero.max(spec_bias)
            };
            let point_mass_scale = FloatW::blend(mass_scale, one, mask);
            let point_impulse_scale = FloatW::blend(impulse_scale, zero, mask);

            let vra = add_vw(ba.v, cross_w(ba.w, ra));
            let vrb = add_vw(bb.v, cross_w(bb.w, rb));
            let vn = dot_w(sub_vw(vrb, vra), normal);

            let normal_mass = ld(wide, pb + P_NORMAL_MASS);
            let old_impulse = ld(wide, pb + P_NORMAL_IMP);
            let neg_impulse = if use_bias {
                normal_mass
                    .mul(point_mass_scale.mul(vn).add(bias))
                    .add(point_impulse_scale.mul(old_impulse))
            } else {
                normal_mass.mul(vn.add(bias))
            };
            let new_impulse = old_impulse.sub(neg_impulse).max(zero);
            let delta_impulse = new_impulse.sub(old_impulse);
            st(wide, pb + P_NORMAL_IMP, new_impulse);
            if !use_bias {
                let total = ld(wide, pb + P_TOTAL_NORMAL_IMP).add(new_impulse);
                st(wide, pb + P_TOTAL_NORMAL_IMP, total);
                total_normal_impulse = total_normal_impulse.add(new_impulse);
                total_twist_limit =
                    total_twist_limit.add(ld(wide, pb + P_LEVER_ARM).mul(new_impulse));
            }

            let p = mul_svw(delta_impulse, normal);
            ba.w = mul_sub_mvw(ba.w, ia, cross_w(ra, p));
            ba.v = mul_sub_svw(ba.v, inv_ma, p);
            bb.w = mul_add_mvw(bb.w, ib, cross_w(rb, p));
            bb.v = mul_add_svw(bb.v, inv_mb, p);
        }

        if !use_bias {
            let rolling_resistance = ld(wide, wo + ROLLING_RESISTANCE);
            if !rolling_resistance.all_zero() {
                let rolling_mass = ld_sym3(wide, wo + ROLLING_MASS);
                let old = ld_v3(wide, wo + ROLLING_IMPULSE);
                let delta = mul_mvw(rolling_mass, sub_vw(ba.w, bb.w));
                let mut rolling = add_vw(old, delta);
                let max_impulse = rolling_resistance.mul(total_normal_impulse);
                let length_squared = dot_w(rolling, rolling);
                let mask = length_squared.greater_than(eps.mul_add(max_impulse, max_impulse));
                let normalize = max_impulse.div(length_squared.sqrt().add(eps));
                let scale = FloatW::blend(one, normalize, mask);
                let rolling_mask = rolling_resistance.greater_than(zero);
                let scaled = mul_svw(scale, rolling);
                rolling = Vec3W {
                    x: FloatW::blend(zero, scaled.x, rolling_mask),
                    y: FloatW::blend(zero, scaled.y, rolling_mask),
                    z: FloatW::blend(zero, scaled.z, rolling_mask),
                };
                st_v3(wide, wo + ROLLING_IMPULSE, rolling);
                let d = sub_vw(rolling, old);
                ba.w = mul_sub_mvw(ba.w, ia, d);
                bb.w = mul_add_mvw(bb.w, ib, d);
            }

            {
                let twist_speed = dot_w(normal, sub_vw(bb.w, ba.w));
                let friction = ld(wide, wo + FRICTION);
                let twist_mass = ld(wide, wo + TWIST_MASS);
                let max_lambda = friction.mul(total_twist_limit);
                let delta = twist_mass.mul(twist_speed).neg();
                let old = ld(wide, wo + TWIST_IMPULSE);
                let new = sym_clamp(old.add(delta), max_lambda);
                st(wide, wo + TWIST_IMPULSE, new);
                let d = new.sub(old);
                let l = mul_svw(d, normal);
                ba.w = mul_sub_mvw(ba.w, ia, l);
                bb.w = mul_add_mvw(bb.w, ib, l);
            }

            {
                let t1 = ld_v3(wide, wo + TANGENT1);
                let t2 = ld_v3(wide, wo + TANGENT2);
                let ra = ld_v3(wide, wo + ORIGIN_A);
                let rb = ld_v3(wide, wo + ORIGIN_B);
                let vra = add_vw(ba.v, cross_w(ba.w, ra));
                let vrb = add_vw(bb.v, cross_w(bb.w, rb));
                let vr = sub_vw(vrb, vra);
                let vt = Vec2W {
                    x: dot_w(vr, t1).sub(ld(wide, wo + TANGENT_VELOCITY1)),
                    y: dot_w(vr, t2).sub(ld(wide, wo + TANGENT_VELOCITY2)),
                };
                let tangent_mass = ld_sym2(wide, wo + TANGENT_MASS);
                let d0 = mul_mv2w(tangent_mass, vt);
                let delta = Vec2W {
                    x: d0.x.neg(),
                    y: d0.y.neg(),
                };
                let old = ld_v2(wide, wo + FRICTION_IMPULSE);
                let mut new = add_v2w(old, delta);
                let friction = ld(wide, wo + FRICTION);
                let max_impulse = friction.mul(total_normal_impulse);
                let length_squared = new.x.mul(new.x).add(new.y.mul(new.y));
                let mask = length_squared.greater_than(max_impulse.mul(max_impulse));
                let normalize = max_impulse.div(length_squared.sqrt().add(eps));
                let scale = FloatW::blend(one, normalize, mask);
                new = Vec2W {
                    x: scale.mul(new.x),
                    y: scale.mul(new.y),
                };
                let delta = Vec2W {
                    x: new.x.sub(old.x),
                    y: new.y.sub(old.y),
                };
                st_v2(wide, wo + FRICTION_IMPULSE, new);
                let p = add_vw(mul_svw(delta.x, t1), mul_svw(delta.y, t2));
                ba.w = mul_sub_mvw(ba.w, ia, cross_w(ra, p));
                ba.v = mul_sub_svw(ba.v, inv_ma, p);
                bb.w = mul_add_mvw(bb.w, ib, cross_w(rb, p));
                bb.v = mul_add_svw(bb.v, inv_mb, p);
            }
        }

        scatter(state, flags, idx, io, &ba.v, &ba.w);
        scatter(state, flags, idx, io + LANES, &bb.v, &bb.w);
    }
}

/// Apply restitution bounce over records `[start, start+count)` (b3ApplyRestitution_Convex).
#[allow(clippy::too_many_arguments)]
pub fn restitution(
    wide: Col<f32>,
    idx: Col<u32>,
    state: Col<f32>,
    flags: Col<u32>,
    start: usize,
    count: usize,
    threshold: f32,
    inv_h: f32,
    propagate: bool,
) {
    let threshold_w = FloatW::splat(-threshold);
    let inv_h = FloatW::splat(inv_h);
    let zero = FloatW::zero();

    for r in start..start + count {
        let wo = r * WIDE_STRIDE;
        let io = r * WIDE_STRIDE;
        let point_count = idx
            .get(io + POINT_COUNTS)
            .max(idx.get(io + POINT_COUNTS + 1))
            .max(
                idx.get(io + POINT_COUNTS + 2)
                    .max(idx.get(io + POINT_COUNTS + 3)),
            ) as usize;
        let rest = ld(wide, wo + RESTITUTION);
        if !propagate && rest.all_zero() {
            continue;
        }
        let mut ba = gather(state, idx, io);
        let mut bb = gather(state, idx, io + LANES);
        let ia = ld_sym3(wide, wo + INV_IA);
        let ib = ld_sym3(wide, wo + INV_IB);
        let inv_ma = ld(wide, wo + INV_MASS_A);
        let inv_mb = ld(wide, wo + INV_MASS_B);
        let normal = ld_v3(wide, wo + NORMAL);
        let restitution_mask = rest.greater_than(zero);
        let dp = sub_vw(bb.dp, ba.dp);

        for pi in 0..point_count {
            let pb = wo + POINTS + pi * POINT_STRIDE;
            let rel_vel = ld(wide, pb + P_REL_VEL);
            let total_normal = ld(wide, pb + P_TOTAL_NORMAL_IMP);
            let restitution_impulse = ld(wide, pb + P_RESTITUTION_IMP);
            let compression = total_normal.sub(restitution_impulse);
            let armed = restitution_mask
                .and(rel_vel.less_than(threshold_w))
                .and(compression.greater_than(zero));
            let normal_mass = ld(wide, pb + P_NORMAL_MASS);
            let mass = if propagate {
                normal_mass
            } else {
                FloatW::blend(zero, normal_mass, restitution_mask)
            };

            let ra = ld_v3(wide, pb + P_ANCHOR_A);
            let rb = ld_v3(wide, pb + P_ANCHOR_B);
            let ds = add_vw(
                dp,
                sub_vw(rotate_vector_w(bb.dq, rb), rotate_vector_w(ba.dq, ra)),
            );
            let s = dot_w(normal, ds).add(ld(wide, pb + P_BASE_SEP));
            let spec_bias = zero.max(s.mul(inv_h));
            let bias = FloatW::blend(spec_bias, rest.mul(rel_vel), armed);
            let vra = add_vw(ba.v, cross_w(ba.w, ra));
            let vrb = add_vw(bb.v, cross_w(bb.w, rb));
            let vn = dot_w(sub_vw(vrb, vra), normal);

            let neg_impulse = mass.mul(vn.add(bias));
            let old_impulse = ld(wide, pb + P_NORMAL_IMP);
            let new_impulse = old_impulse.sub(neg_impulse).max(zero);
            let impulse = new_impulse.sub(old_impulse);
            let approach = mass.mul(vn).neg().max(zero).min(impulse.max(zero));
            let allowance = rest.mul(compression.add(approach)).sub(restitution_impulse);
            let max_impulse = approach.add(allowance.max(zero));
            let delta_impulse = FloatW::blend(impulse, impulse.min(max_impulse), armed);
            st(wide, pb + P_NORMAL_IMP, old_impulse.add(delta_impulse));
            st(
                wide,
                pb + P_RESTITUTION_IMP,
                restitution_impulse.add(delta_impulse.sub(approach)),
            );
            st(
                wide,
                pb + P_TOTAL_NORMAL_IMP,
                total_normal.add(delta_impulse),
            );

            let p = mul_svw(delta_impulse, normal);
            ba.w = mul_sub_mvw(ba.w, ia, cross_w(ra, p));
            ba.v = mul_sub_svw(ba.v, inv_ma, p);
            bb.w = mul_add_mvw(bb.w, ib, cross_w(rb, p));
            bb.v = mul_add_svw(bb.v, inv_mb, p);
        }

        scatter(state, flags, idx, io, &ba.v, &ba.w);
        scatter(state, flags, idx, io + LANES, &bb.v, &bb.w);
    }
}

/// Write solved impulses back into the pool manifolds and flag hit events (b3StoreImpulses_Convex)
/// for records `[start, start+count)`. Each lane's contactId (from `spans`) resolves to its manifold
/// through the directory; `hit_event_threshold` is the (positive) hit-speed threshold.
pub fn store(
    wide: Col<f32>,
    spans: Col<crate::contact_spans::WidePrepareSpan>,
    dir: Col<u32>,
    start: usize,
    count: usize,
    hit_event_threshold: f32,
    mut mark_hit: impl FnMut(usize),
) {
    let neg_hit = -hit_event_threshold;

    for (r, span) in crate::contact_spans::wide_records(spans, start, count) {
        let wo = r * WIDE_STRIDE;
        let local = (r - span.start as usize) * LANES;
        let lane_count = (span.count as usize - local).min(LANES);

        let f1 = ld(wide, wo + FRICTION_IMPULSE).to_array();
        let f2 = ld(wide, wo + FRICTION_IMPULSE + 4).to_array();
        let t1x = ld(wide, wo + TANGENT1).to_array();
        let t1y = ld(wide, wo + TANGENT1 + 4).to_array();
        let t1z = ld(wide, wo + TANGENT1 + 8).to_array();
        let t2x = ld(wide, wo + TANGENT2).to_array();
        let t2y = ld(wide, wo + TANGENT2 + 4).to_array();
        let t2z = ld(wide, wo + TANGENT2 + 8).to_array();
        let twist = ld(wide, wo + TWIST_IMPULSE).to_array();
        let rix = ld(wide, wo + ROLLING_IMPULSE).to_array();
        let riy = ld(wide, wo + ROLLING_IMPULSE + 4).to_array();
        let riz = ld(wide, wo + ROLLING_IMPULSE + 8).to_array();

        for lane in 0..lane_count {
            let manifold = unsafe {
                *wide
                    .ptr()
                    .add(wo + MANIFOLDS)
                    .cast::<*mut mabi::ManifoldRecord>()
                    .add(lane)
            };
            if manifold.is_null() {
                continue;
            }
            let pool = unsafe { Col::new(manifold.cast::<f32>(), mabi::MANIFOLD_STRIDE) };
            let mpo = 0; // convex: exactly one manifold
            pool.set(
                mpo + mabi::M_FRICTION,
                f1[lane] * t1x[lane] + f2[lane] * t2x[lane],
            );
            pool.set(
                mpo + mabi::M_FRICTION + 1,
                f1[lane] * t1y[lane] + f2[lane] * t2y[lane],
            );
            pool.set(
                mpo + mabi::M_FRICTION + 2,
                f1[lane] * t1z[lane] + f2[lane] * t2z[lane],
            );
            pool.set(mpo + mabi::M_TWIST, twist[lane]);
            pool.set(mpo + mabi::M_ROLLING, rix[lane]);
            pool.set(mpo + mabi::M_ROLLING + 1, riy[lane]);
            pool.set(mpo + mabi::M_ROLLING + 2, riz[lane]);

            let point_count = pool.get(mpo + mabi::M_POINT_COUNT).to_bits() as usize;
            for pi in 0..point_count {
                let pb = wo + POINTS + pi * POINT_STRIDE; // wide-record point
                let pp = mpo + mabi::M_POINTS + pi * mabi::POOL_POINT_STRIDE; // pool point
                pool.set(
                    pp + mabi::P_NORMAL_IMPULSE,
                    ld(wide, pb + P_NORMAL_IMP).to_array()[lane],
                );
                pool.set(
                    pp + mabi::P_TOTAL_NORMAL_IMPULSE,
                    ld(wide, pb + P_TOTAL_NORMAL_IMP).to_array()[lane],
                );
            }

            let contact_id = unsafe { *span.contacts.add(local + lane) as usize };
            let flags = dir.get(contact_id * mabi::DIR_STRIDE + mabi::DIR_FLAGS);
            if flags & mabi::contact_flags::SIM_ENABLE_HIT_EVENT != 0 {
                for pi in 0..point_count {
                    let pp = mpo + mabi::M_POINTS + pi * mabi::POOL_POINT_STRIDE;
                    let normal_velocity = pool.get(pp + mabi::P_NORMAL_VELOCITY);
                    let total_normal_impulse = pool.get(pp + mabi::P_TOTAL_NORMAL_IMPULSE);
                    if normal_velocity < neg_hit && total_normal_impulse > 0.0 {
                        mark_hit(contact_id);
                        break;
                    }
                }
            }
        }
    }
}
