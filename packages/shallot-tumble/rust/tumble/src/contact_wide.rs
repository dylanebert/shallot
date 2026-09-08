//! The wide (4-lane SIMD) convex contact solver, ported op-for-op from box3d's `contact_solver.c`
//! `_Convex` functions. Four single-manifold convex contacts sharing no body (graph coloring, done
//! TS-side) are gathered into one `b3ContactConstraintWide` and solved together across `FloatW` lanes.
//!
//! This is the lane win over the scalar (`contact`) path: identical soft-step arithmetic, four
//! contacts per pass. The scalar path stays for mesh contacts (multi-manifold) and the overflow color.
//! The friction sub-step order here (rolling → twist → central) differs from `_Mesh` (twist → rolling
//! → central), so this path is gold-verified against its own C reference, not the scalar port.
//!
//! Column groups over shared wasm memory:
//!   - **persistent** — the same column-resident contact-manifold store the scalar path gathers
//!     (`manifold_abi`): a directory record + one pool manifold per convex contact. `meta` carries
//!     each wide record's four lanes' contactIds; the gather reads material + indices + manifold
//!     through `directory[contactId]` + `pool[manifoldBase]`.
//!   - **wide** — the transient `b3ContactConstraintWide` records, SoA lanes. Prepare builds them,
//!     solve/restitution mutate the impulses, store writes the results back into the pool manifolds.
//!
//! All wide arithmetic is `FloatW` (simd128 on wasm, scalar `[f32;4]` fallback natively) — bit-
//! identical per lane, the same property that makes box3d's SIMD and `DISABLE_SIMD` builds agree.
//! Prepare's per-lane scalar staging reuses the `math` b3 ports so its operand order matches the
//! scalar path (and thus the reference) exactly.

use crate::body::flags as body_flags;
use crate::body::{read_sim, read_state, STATE_STRIDE};
use crate::col::Col;
use crate::contact::{Softness, NULL_INDEX};
use crate::manifold_abi as mabi;
use crate::manifold_abi::{read_dir, set_hit};
use crate::math::{Mat2, Mat3, Vec2, Vec3, FLT_EPSILON};
use crate::simd::FloatW;
use crate::wide::{
    add_v2w, add_vw, cross_w, dot_w, mul_add_mvw, mul_add_svw, mul_mv2w, mul_mvw, mul_sub_mvw,
    mul_sub_svw, mul_svw, rotate_vector_w, sub_vw, sym_clamp, QuatW, SymMatrix2W, SymMatrix3W,
    Vec2W, Vec3W,
};
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// Max manifold points (B3_MAX_MANIFOLD_POINTS). Convex contacts have one manifold with up to this
/// many points; the wide constraint always carries all four slots (unused zeroed).
pub const MAX_POINTS: usize = 4;
/// SIMD lane width (B3_SIMD_WIDTH).
pub const LANES: usize = 4;

// --- wide constraint column layout (f32 offsets within one WIDE_STRIDE record) --------------
// A FloatW occupies 4 contiguous f32 (one lane vector). Field order mirrors b3ContactConstraintWide;
// indices/manifolds live in the meta/index columns instead of here.

const INV_MASS_A: usize = 0;
const INV_MASS_B: usize = 4;
const INV_IA: usize = 8; // SymMatrix3W: cxx,cxy,cxz,cyy,cyz,czz (6 FloatW)
const INV_IB: usize = 32;
const NORMAL: usize = 56; // Vec3W
const TANGENT1: usize = 68;
const TANGENT2: usize = 80;
const ORIGIN_A: usize = 92;
const ORIGIN_B: usize = 104;
const TWIST_MASS: usize = 116;
const TWIST_IMPULSE: usize = 120;
const TANGENT_MASS: usize = 124; // SymMatrix2W: cxx,cxy,cyy (3 FloatW)
const FRICTION_IMPULSE: usize = 136; // Vec2W
const ROLLING_MASS: usize = 144; // SymMatrix3W
const ROLLING_IMPULSE: usize = 168; // Vec3W
const FRICTION: usize = 180;
const ROLLING_RESISTANCE: usize = 184;
const TANGENT_VELOCITY1: usize = 188;
const TANGENT_VELOCITY2: usize = 192;
const BIAS_RATE: usize = 196;
const MASS_SCALE: usize = 200;
const IMPULSE_SCALE: usize = 204;
const RESTITUTION: usize = 208;
const POINTS: usize = 212;
const POINT_STRIDE: usize = 48; // 12 FloatW per point

// point sub-offsets (relative to a point's base)
const P_ANCHOR_A: usize = 0;
const P_ANCHOR_B: usize = 12;
const P_BASE_SEP: usize = 24;
const P_NORMAL_IMP: usize = 28;
const P_TOTAL_NORMAL_IMP: usize = 32;
const P_NORMAL_MASS: usize = 36;
const P_LEVER_ARM: usize = 40;
const P_REL_VEL: usize = 44;

/// f32 stride of one wide constraint record (b3ContactConstraintWide, 101 FloatW).
pub const WIDE_STRIDE: usize = POINTS + MAX_POINTS * POINT_STRIDE; // 404

/// u32 stride of the per-record wide meta: `laneContact[4]` (input contact record per lane, NULL for
/// inactive) then `laneCount` (active lanes, 1..=4).
pub const WIDE_META_STRIDE: usize = LANES + 1;
/// u32 stride of the per-record base-1 body sim indices: `indexA[4]` then `indexB[4]` (0 = null).
pub const WIDE_IDX_STRIDE: usize = 2 * LANES;

// --- column load/store helpers --------------------------------------------------------------

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
    Vec3W { x: ld(col, o), y: ld(col, o + 4), z: ld(col, o + 8) }
}
#[inline]
fn st_v3(col: Col<f32>, o: usize, v: Vec3W) {
    st(col, o, v.x);
    st(col, o + 4, v.y);
    st(col, o + 8, v.z);
}
#[inline]
fn ld_v2(col: Col<f32>, o: usize) -> Vec2W {
    Vec2W { x: ld(col, o), y: ld(col, o + 4) }
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
    SymMatrix2W { cxx: ld(col, o), cxy: ld(col, o + 4), cyy: ld(col, o + 8) }
}

// SoA store: pack one field's four lanes (indexed by lane) into the record's lane vectors.
#[inline]
fn st_lanes_v3(wide: Col<f32>, o: usize, lanes: &[Vec3; LANES]) {
    st(wide, o, FloatW::set(lanes[0].x, lanes[1].x, lanes[2].x, lanes[3].x));
    st(wide, o + 4, FloatW::set(lanes[0].y, lanes[1].y, lanes[2].y, lanes[3].y));
    st(wide, o + 8, FloatW::set(lanes[0].z, lanes[1].z, lanes[2].z, lanes[3].z));
}
#[inline]
fn st_lanes_sym3(wide: Col<f32>, o: usize, m: &[Mat3; LANES]) {
    // Symmetric components cxx,cxy,cxz,cyy,cyz,czz (column-major diagonal walk).
    st(wide, o, FloatW::set(m[0].cx.x, m[1].cx.x, m[2].cx.x, m[3].cx.x));
    st(wide, o + 4, FloatW::set(m[0].cx.y, m[1].cx.y, m[2].cx.y, m[3].cx.y));
    st(wide, o + 8, FloatW::set(m[0].cx.z, m[1].cx.z, m[2].cx.z, m[3].cx.z));
    st(wide, o + 12, FloatW::set(m[0].cy.y, m[1].cy.y, m[2].cy.y, m[3].cy.y));
    st(wide, o + 16, FloatW::set(m[0].cy.z, m[1].cy.z, m[2].cy.z, m[3].cy.z));
    st(wide, o + 20, FloatW::set(m[0].cz.z, m[1].cz.z, m[2].cz.z, m[3].cz.z));
}
#[inline]
fn st_lanes_sym2(wide: Col<f32>, o: usize, m: &[Mat2; LANES]) {
    st(wide, o, FloatW::set(m[0].cx.x, m[1].cx.x, m[2].cx.x, m[3].cx.x));
    st(wide, o + 4, FloatW::set(m[0].cx.y, m[1].cx.y, m[2].cx.y, m[3].cx.y));
    st(wide, o + 8, FloatW::set(m[0].cy.y, m[1].cy.y, m[2].cy.y, m[3].cy.y));
}
#[inline]
fn st_lanes_f(wide: Col<f32>, o: usize, l: &[f32; LANES]) {
    st(wide, o, FloatW::set(l[0], l[1], l[2], l[3]));
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
        (s.inv_mass, s.inv_inertia_world, st.linear_velocity, st.angular_velocity)
    }
}

// --- body gather / scatter ------------------------------------------------------------------
//
// 4c: the wasm-simd path gathers each body as 4 aligned `v128` record loads + unpack-only 4×4 shuffle
// transposes, and scatters velocities back with a whole-vector store (b3GatherBodies/b3ScatterBodies's
// actual SSE shape — `research/gather-spike` measured 1.9× JSC / 2.5× V8 over the field-wise scalar
// gather this replaced, bit-identical output). It reads/writes raw pointers into the resident state
// column (bodies.rs), so a null/static lane (idx 0) remaps to the region's trailing identity record
// (`ident_rec`) instead of the per-lane branch the scalar path takes. Pure data movement — no fixture
// regen; the wasm transpose is fixture-gated (52/52), and native `cargo test` keeps the field-wise
// scalar gather below as the bit-identical reference the gold vectors exercise.

/// Wide body solver state (b3BodyStateW): the four gathered bodies' velocities + deltas across lanes.
struct BodyStateW {
    v: Vec3W,
    w: Vec3W,
    dp: Vec3W,
    dq: QuatW,
}

const ALL_LOCKS: u32 = body_flags::LOCK_LINEAR_X
    | body_flags::LOCK_LINEAR_Y
    | body_flags::LOCK_LINEAR_Z
    | body_flags::LOCK_ANGULAR_X
    | body_flags::LOCK_ANGULAR_Y
    | body_flags::LOCK_ANGULAR_Z;

/// Record index of `worker`'s trailing null-lane record, one of the [`IDENT_RECORDS`] the persistent
/// body region reserves past its `bodyCap` real records (bodies.rs), each initialised to zero
/// velocity/delta + identity rotation + DYNAMIC. 4c's wasm gather remaps every null/static lane (idx 0)
/// there. Native never touches it — its field-wise gather staples an identity into null lanes directly —
/// so the fallback returns 0.
///
/// **Per worker, not one.** `scatter_t` writes the record for every null lane it fast-paths, so two
/// blocks of a stage holding static-lane records would write the same bytes from two threads. Always
/// the *same* bytes (a null lane gathers zero velocity and carries `invMass = 0` / `invI = 0`, so the
/// solve hands back what it gathered) — benign by value, but still a same-address concurrent write, and
/// the only element of the state column that is not write-disjoint. Giving each worker its own record
/// makes the whole column disjoint, which is what `Col`'s promise (col.rs) rests on.
#[cfg(target_arch = "wasm32")]
#[inline]
fn ident_rec(worker: usize) -> usize {
    debug_assert!(worker < crate::bodies::IDENT_RECORDS);
    crate::bodies::body_cap() + worker
}
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn ident_rec(_worker: usize) -> usize {
    0
}

/// Per-lane scalar scatter (b3ScatterBodies SSE2 branch): write each dynamic lane's velocity back with
/// motion locks; null lanes (idx 0) and statics are skipped. The native path's only scatter, and the
/// wasm path's fallback when a lane is locked/static and the whole-vector store can't apply the lock.
fn scatter_scalar(
    state: Col<f32>,
    flags: Col<u32>,
    idx: Col<u32>,
    io: usize,
    v: &Vec3W,
    w: &Vec3W,
) {
    let vx = v.x.to_array();
    let vy = v.y.to_array();
    let vz = v.z.to_array();
    let wx = w.x.to_array();
    let wy = w.y.to_array();
    let wz = w.z.to_array();
    for lane in 0..LANES {
        let i = idx.get(io + lane);
        if i == 0 {
            continue;
        }
        let b = (i - 1) as usize;
        let f = flags.get(b);
        if f & body_flags::DYNAMIC == 0 {
            continue;
        }
        let mut v = [vx[lane], vy[lane], vz[lane]];
        let mut w = [wx[lane], wy[lane], wz[lane]];
        if f & ALL_LOCKS != 0 {
            if f & body_flags::LOCK_LINEAR_X != 0 {
                v[0] = 0.0;
            }
            if f & body_flags::LOCK_LINEAR_Y != 0 {
                v[1] = 0.0;
            }
            if f & body_flags::LOCK_LINEAR_Z != 0 {
                v[2] = 0.0;
            }
            if f & body_flags::LOCK_ANGULAR_X != 0 {
                w[0] = 0.0;
            }
            if f & body_flags::LOCK_ANGULAR_Y != 0 {
                w[1] = 0.0;
            }
            if f & body_flags::LOCK_ANGULAR_Z != 0 {
                w[2] = 0.0;
            }
        }
        let o = b * STATE_STRIDE;
        state.set(o, v[0]);
        state.set(o + 1, v[1]);
        state.set(o + 2, v[2]);
        state.set(o + 3, w[0]);
        state.set(o + 4, w[1]);
        state.set(o + 5, w[2]);
    }
}

// --- native (scalar) gather / scatter: the bit-identical reference for `cargo test` ----------

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn fw(a: [f32; 4]) -> FloatW {
    FloatW::set(a[0], a[1], a[2], a[3])
}

/// Gather four bodies' full solver state into lanes (b3GatherBodies). Index 0 is the null/static body,
/// contributing an identity state (zero velocity/delta, identity rotation).
#[cfg(not(target_arch = "wasm32"))]
fn gather(state: Col<f32>, idx: Col<u32>, io: usize, _ident: usize) -> BodyStateW {
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
        v: Vec3W { x: fw(vx), y: fw(vy), z: fw(vz) },
        w: Vec3W { x: fw(wx), y: fw(wy), z: fw(wz) },
        dp: Vec3W { x: fw(dpx), y: fw(dpy), z: fw(dpz) },
        dq: QuatW { v: Vec3W { x: fw(qx), y: fw(qy), z: fw(qz) }, s: fw(qs) },
    }
}

/// Velocities-only gather (warm start reads no deltas): 6 scalar loads per lane.
#[cfg(not(target_arch = "wasm32"))]
fn gather_vel(state: Col<f32>, idx: Col<u32>, io: usize, _ident: usize) -> (Vec3W, Vec3W) {
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
    (Vec3W { x: fw(vx), y: fw(vy), z: fw(vz) }, Vec3W { x: fw(wx), y: fw(wy), z: fw(wz) })
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn scatter(
    state: Col<f32>,
    flags: Col<u32>,
    idx: Col<u32>,
    io: usize,
    v: &Vec3W,
    w: &Vec3W,
    _ident: usize,
) {
    scatter_scalar(state, flags, idx, io, v, w);
}

// --- wasm (simd128) record-transpose gather / scatter ---------------------------------------

/// Branchless null remap: idx 0 (null/static) → the identity record, idx k → record k-1.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn rec_of(i: u32, ident: usize) -> usize {
    let m = ((i == 0) as usize).wrapping_neg(); // 0 or usize::MAX
    ((i.wrapping_sub(1) as usize) & !m) | (ident & m)
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

/// Gather four bodies' full solver state: 4 aligned v128 record loads + shuffle transposes → 13 lanes.
#[cfg(target_arch = "wasm32")]
fn gather(state: Col<f32>, idx: Col<u32>, io: usize, ident: usize) -> BodyStateW {
    let p = state.ptr();
    unsafe {
        let p0 = p.add(rec_of(idx.get(io), ident) * STATE_STRIDE);
        let p1 = p.add(rec_of(idx.get(io + 1), ident) * STATE_STRIDE);
        let p2 = p.add(rec_of(idx.get(io + 2), ident) * STATE_STRIDE);
        let p3 = p.add(rec_of(idx.get(io + 3), ident) * STATE_STRIDE);
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
        // block 3: only lane 0 (dq.s) is live; the dead transpose outputs fold away.
        let (qs, _, _, _) = transpose4(
            v128_load(p0.add(12) as *const v128),
            v128_load(p1.add(12) as *const v128),
            v128_load(p2.add(12) as *const v128),
            v128_load(p3.add(12) as *const v128),
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

/// Velocities-only gather (warm start reads no deltas): 2 v128 loads + 2 transposes per body quad.
#[cfg(target_arch = "wasm32")]
fn gather_vel(state: Col<f32>, idx: Col<u32>, io: usize, ident: usize) -> (Vec3W, Vec3W) {
    let p = state.ptr();
    unsafe {
        let p0 = p.add(rec_of(idx.get(io), ident) * STATE_STRIDE);
        let p1 = p.add(rec_of(idx.get(io + 1), ident) * STATE_STRIDE);
        let p2 = p.add(rec_of(idx.get(io + 2), ident) * STATE_STRIDE);
        let p3 = p.add(rec_of(idx.get(io + 3), ident) * STATE_STRIDE);
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
            Vec3W { x: FloatW::from_v128(vx), y: FloatW::from_v128(vy), z: FloatW::from_v128(vz) },
            Vec3W { x: FloatW::from_v128(wx), y: FloatW::from_v128(wy), z: FloatW::from_v128(wz) },
        )
    }
}

/// Lock-free fast scatter: block 0 `[vx vy vz wx]` transposed back + whole v128 store per body; block 1
/// `[wy wz dpx dpy]` is half-written → load, shuffle-blend the new wy/wz in, store back (dp preserved).
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn scatter_t(state: *mut f32, recs: &[usize; 4], v: &Vec3W, w: &Vec3W) {
    let (b0, b1, b2, b3) = transpose4(v.x.v128(), v.y.v128(), v.z.v128(), w.x.v128());
    let t0 = i32x4_shuffle::<0, 4, 1, 5>(w.y.v128(), w.z.v128()); // wy0 wz0 wy1 wz1
    let t1 = i32x4_shuffle::<2, 6, 3, 7>(w.y.v128(), w.z.v128()); // wy2 wz2 wy3 wz3

    let p0 = state.add(recs[0] * STATE_STRIDE);
    v128_store(p0 as *mut v128, b0);
    let c0 = v128_load(p0.add(4) as *const v128);
    v128_store(p0.add(4) as *mut v128, i32x4_shuffle::<0, 1, 6, 7>(t0, c0));

    let p1 = state.add(recs[1] * STATE_STRIDE);
    v128_store(p1 as *mut v128, b1);
    let c1 = v128_load(p1.add(4) as *const v128);
    v128_store(p1.add(4) as *mut v128, i32x4_shuffle::<2, 3, 6, 7>(t0, c1));

    let p2 = state.add(recs[2] * STATE_STRIDE);
    v128_store(p2 as *mut v128, b2);
    let c2 = v128_load(p2.add(4) as *const v128);
    v128_store(p2.add(4) as *mut v128, i32x4_shuffle::<0, 1, 6, 7>(t1, c2));

    let p3 = state.add(recs[3] * STATE_STRIDE);
    v128_store(p3 as *mut v128, b3);
    let c3 = v128_load(p3.add(4) as *const v128);
    v128_store(p3.add(4) as *mut v128, i32x4_shuffle::<2, 3, 6, 7>(t1, c3));
}

/// Write four gathered bodies' velocities back (b3ScatterBodies). When every lane is dynamic and
/// unlocked (the pyramid-bulk case), one hoisted flags check gates the whole-vector `scatter_t`; the
/// null/static lanes it also writes land on the identity record (zero velocity, DYNAMIC) harmlessly.
/// Any locked or non-dynamic lane falls back to the per-lane scalar scatter.
#[cfg(target_arch = "wasm32")]
fn scatter(
    state: Col<f32>,
    flags: Col<u32>,
    idx: Col<u32>,
    io: usize,
    v: &Vec3W,
    w: &Vec3W,
    ident: usize,
) {
    let recs = [
        rec_of(idx.get(io), ident),
        rec_of(idx.get(io + 1), ident),
        rec_of(idx.get(io + 2), ident),
        rec_of(idx.get(io + 3), ident),
    ];
    let fp = flags.ptr();
    unsafe {
        let f0 = *fp.add(recs[0]);
        let f1 = *fp.add(recs[1]);
        let f2 = *fp.add(recs[2]);
        let f3 = *fp.add(recs[3]);
        if (f0 | f1 | f2 | f3) & ALL_LOCKS == 0 && (f0 & f1 & f2 & f3) & body_flags::DYNAMIC != 0 {
            scatter_t(state.ptr(), &recs, v, w);
        } else {
            scatter_scalar(state, flags, idx, io, v, w);
        }
    }
}

// --- prepare --------------------------------------------------------------------------------

/// Build the wide constraints for the records in `[start, start+count)` (b3PrepareContacts_Convex).
/// Gathers each active lane's convex contact (one manifold) through its contactId → the persistent
/// directory + pool, plus body sim/state; writes the wide + index columns. `warm_start_scale` is 1
/// (warm starting on) or 0.
#[allow(clippy::too_many_arguments)]
pub fn prepare(
    wide: Col<f32>,
    idx: Col<u32>,
    meta: Col<u32>,
    state: Col<f32>,
    sim: Col<f32>,
    dir: Col<u32>,
    pool: Col<f32>,
    start: usize,
    count: usize,
    contact_softness: Softness,
    static_softness: Softness,
    warm_start_scale: f32,
) {
    for r in start..start + count {
        let wo = r * WIDE_STRIDE;
        let mo = r * WIDE_META_STRIDE;
        let io = r * WIDE_IDX_STRIDE;
        let lane_count = meta.get(mo + LANES) as usize;

        // Null every lane's body index up front (base-1, so 0 = null → gather contributes identity).
        // Only `lane_count` lanes are filled below; this zeroes the tail lanes of a partial record,
        // mirroring box3d's memset of the remainder wide slot. Without it a stale nonzero index would
        // gather a bogus body in `warm_start`/`solve`.
        for k in 0..WIDE_IDX_STRIDE {
            idx.set(io + k, 0);
        }

        // Per-lane staging; SoA lane vectors are written into the record after the lane loop.
        let mut inv_mass_a = [0.0f32; 4];
        let mut inv_mass_b = [0.0f32; 4];
        let mut ia = [Mat3::ZERO; 4];
        let mut ib = [Mat3::ZERO; 4];
        let mut normal = [Vec3::ZERO; 4];
        let mut tangent1 = [Vec3::ZERO; 4];
        let mut tangent2 = [Vec3::ZERO; 4];
        let mut origin_a = [Vec3::ZERO; 4];
        let mut origin_b = [Vec3::ZERO; 4];
        let mut friction = [0.0f32; 4];
        let mut restitution = [0.0f32; 4];
        let mut rolling_resistance = [0.0f32; 4];
        let mut tangent_velocity1 = [0.0f32; 4];
        let mut tangent_velocity2 = [0.0f32; 4];
        let mut bias_rate = [0.0f32; 4];
        let mut mass_scale = [0.0f32; 4];
        let mut impulse_scale = [0.0f32; 4];
        let mut twist_mass = [0.0f32; 4];
        let mut twist_impulse = [0.0f32; 4];
        let mut friction_impulse_x = [0.0f32; 4];
        let mut friction_impulse_y = [0.0f32; 4];
        let mut rolling_mass = [Mat3::ZERO; 4];
        let mut rolling_impulse = [Vec3::ZERO; 4];
        let mut tangent_mass = [Mat2 { cx: Vec2::new(0.0, 0.0), cy: Vec2::new(0.0, 0.0) }; 4];
        let mut p_anchor_a = [[Vec3::ZERO; 4]; MAX_POINTS];
        let mut p_anchor_b = [[Vec3::ZERO; 4]; MAX_POINTS];
        let mut p_base_sep = [[0.0f32; 4]; MAX_POINTS];
        let mut p_normal_imp = [[0.0f32; 4]; MAX_POINTS];
        let mut p_normal_mass = [[0.0f32; 4]; MAX_POINTS];
        let mut p_lever_arm = [[0.0f32; 4]; MAX_POINTS];
        let mut p_rel_vel = [[0.0f32; 4]; MAX_POINTS];

        for lane in 0..lane_count {
            let contact_id = meta.get(mo + lane) as usize;
            let d = read_dir(dir, contact_id);
            let index_a = d.index_a;
            let index_b = d.index_b;
            let mpo = d.manifold_base * mabi::MANIFOLD_STRIDE; // convex: exactly one manifold

            idx.set(io + lane, index_a.wrapping_add(1));
            idx.set(io + LANES + lane, index_b.wrapping_add(1));

            let (m_a, i_a, v_a, w_a) = body_terms(sim, state, index_a);
            let (m_b, i_b, v_b, w_b) = body_terms(sim, state, index_b);
            inv_mass_a[lane] = m_a;
            inv_mass_b[lane] = m_b;
            ia[lane] = i_a;
            ib[lane] = i_b;

            friction[lane] = d.friction;
            restitution[lane] = d.restitution;
            rolling_resistance[lane] = d.rolling_resistance;
            let tangent_velocity = d.tangent_velocity;

            let n = v3(pool, mpo + mabi::M_NORMAL);
            let t1 = n.perp();
            let t2 = t1.cross(n);
            normal[lane] = n;
            tangent1[lane] = t1;
            tangent2[lane] = t2;
            tangent_velocity1[lane] = tangent_velocity.dot(t1);
            tangent_velocity2[lane] = tangent_velocity.dot(t2);

            let soft = if index_a == NULL_INDEX || index_b == NULL_INDEX {
                static_softness
            } else {
                contact_softness
            };
            bias_rate[lane] = soft.bias_rate;
            mass_scale[lane] = soft.mass_scale;
            impulse_scale[lane] = soft.impulse_scale;

            let point_count = pool.get(mpo + mabi::M_POINT_COUNT).to_bits() as usize;

            let mut center_a = Vec3::ZERO;
            let mut center_b = Vec3::ZERO;
            for pi in 0..point_count {
                let pp = mpo + mabi::M_POINTS + pi * mabi::POOL_POINT_STRIDE;
                let r_a = v3(pool, pp + mabi::P_ANCHOR_A);
                let r_b = v3(pool, pp + mabi::P_ANCHOR_B);
                let separation = pool.get(pp + mabi::P_SEPARATION);
                let mp_normal_impulse = pool.get(pp + mabi::P_NORMAL_IMPULSE);

                let rn_a = r_a.cross(n);
                let rn_b = r_b.cross(n);
                let k_normal = m_a + m_b + rn_a.dot(i_a.mul_v(rn_a)) + rn_b.dot(i_b.mul_v(rn_b));
                let vr_a = v_a.add(w_a.cross(r_a));
                let vr_b = v_b.add(w_b.cross(r_b));

                p_anchor_a[pi][lane] = r_a;
                p_anchor_b[pi][lane] = r_b;
                p_base_sep[pi][lane] = separation - r_b.sub(r_a).dot(n);
                p_normal_imp[pi][lane] = warm_start_scale * mp_normal_impulse;
                p_normal_mass[pi][lane] = if k_normal > 0.0 { 1.0 / k_normal } else { 0.0 };
                p_rel_vel[pi][lane] = n.dot(vr_b.sub(vr_a));

                center_a = center_a.add(r_a);
                center_b = center_b.add(r_b);
            }
            let inv_count = 1.0 / point_count as f32;
            center_a = center_a.scale(inv_count);
            center_b = center_b.scale(inv_count);
            origin_a[lane] = center_a;
            origin_b[lane] = center_b;
            for pi in 0..point_count {
                p_lever_arm[pi][lane] = p_anchor_a[pi][lane].distance(center_a);
            }

            let rt_a1 = center_a.cross(t1);
            let rt_a2 = center_a.cross(t2);
            let rt_b1 = center_b.cross(t1);
            let rt_b2 = center_b.cross(t2);
            let kxx = m_a + m_b + rt_a1.dot(i_a.mul_v(rt_a1)) + rt_b1.dot(i_b.mul_v(rt_b1));
            let kyy = m_a + m_b + rt_a2.dot(i_a.mul_v(rt_a2)) + rt_b2.dot(i_b.mul_v(rt_b2));
            let kxy = rt_a1.dot(i_a.mul_v(rt_a2)) + rt_b1.dot(i_b.mul_v(rt_b2));
            tangent_mass[lane] = Mat2 { cx: Vec2::new(kxx, kxy), cy: Vec2::new(kxy, kyy) }.invert();

            let mf_friction_impulse = v3(pool, mpo + mabi::M_FRICTION);
            friction_impulse_x[lane] = warm_start_scale * mf_friction_impulse.dot(t1);
            friction_impulse_y[lane] = warm_start_scale * mf_friction_impulse.dot(t2);

            let iab = i_a.add(i_b);
            let twist_k = n.dot(iab.mul_v(n));
            twist_mass[lane] = if twist_k > 0.0 { 1.0 / twist_k } else { 0.0 };
            twist_impulse[lane] = warm_start_scale * pool.get(mpo + mabi::M_TWIST);
            rolling_mass[lane] = iab.invert();
            rolling_impulse[lane] = v3(pool, mpo + mabi::M_ROLLING).scale(warm_start_scale);
        }

        st_lanes_f(wide, wo + INV_MASS_A, &inv_mass_a);
        st_lanes_f(wide, wo + INV_MASS_B, &inv_mass_b);
        st_lanes_sym3(wide, wo + INV_IA, &ia);
        st_lanes_sym3(wide, wo + INV_IB, &ib);
        st_lanes_v3(wide, wo + NORMAL, &normal);
        st_lanes_v3(wide, wo + TANGENT1, &tangent1);
        st_lanes_v3(wide, wo + TANGENT2, &tangent2);
        st_lanes_v3(wide, wo + ORIGIN_A, &origin_a);
        st_lanes_v3(wide, wo + ORIGIN_B, &origin_b);
        st_lanes_f(wide, wo + TWIST_MASS, &twist_mass);
        st_lanes_f(wide, wo + TWIST_IMPULSE, &twist_impulse);
        st_lanes_sym2(wide, wo + TANGENT_MASS, &tangent_mass);
        st_lanes_f(wide, wo + FRICTION_IMPULSE, &friction_impulse_x);
        st_lanes_f(wide, wo + FRICTION_IMPULSE + 4, &friction_impulse_y);
        st_lanes_sym3(wide, wo + ROLLING_MASS, &rolling_mass);
        st_lanes_v3(wide, wo + ROLLING_IMPULSE, &rolling_impulse);
        st_lanes_f(wide, wo + FRICTION, &friction);
        st_lanes_f(wide, wo + ROLLING_RESISTANCE, &rolling_resistance);
        st_lanes_f(wide, wo + TANGENT_VELOCITY1, &tangent_velocity1);
        st_lanes_f(wide, wo + TANGENT_VELOCITY2, &tangent_velocity2);
        st_lanes_f(wide, wo + BIAS_RATE, &bias_rate);
        st_lanes_f(wide, wo + MASS_SCALE, &mass_scale);
        st_lanes_f(wide, wo + IMPULSE_SCALE, &impulse_scale);
        st_lanes_f(wide, wo + RESTITUTION, &restitution);
        for pi in 0..MAX_POINTS {
            let pb = wo + POINTS + pi * POINT_STRIDE;
            st_lanes_v3(wide, pb + P_ANCHOR_A, &p_anchor_a[pi]);
            st_lanes_v3(wide, pb + P_ANCHOR_B, &p_anchor_b[pi]);
            st_lanes_f(wide, pb + P_BASE_SEP, &p_base_sep[pi]);
            st_lanes_f(wide, pb + P_NORMAL_IMP, &p_normal_imp[pi]);
            st(wide, pb + P_TOTAL_NORMAL_IMP, FloatW::zero());
            st_lanes_f(wide, pb + P_NORMAL_MASS, &p_normal_mass[pi]);
            st_lanes_f(wide, pb + P_LEVER_ARM, &p_lever_arm[pi]);
            st_lanes_f(wide, pb + P_REL_VEL, &p_rel_vel[pi]);
        }
    }
}

// --- warm start -----------------------------------------------------------------------------

/// Seed body velocities from the warm-start impulses (b3WarmStartContacts_Convex) for records
/// `[start, start+count)`. `worker` names the thread running this block — it selects the null-lane
/// identity record the gather/scatter writes ([`ident_rec`]); 0 on the serial path.
pub fn warm_start(
    wide: Col<f32>,
    idx: Col<u32>,
    state: Col<f32>,
    flags: Col<u32>,
    start: usize,
    count: usize,
    worker: usize,
) {
    let ident = ident_rec(worker);
    for r in start..start + count {
        let wo = r * WIDE_STRIDE;
        let io = r * WIDE_IDX_STRIDE;
        let ia = ld_sym3(wide, wo + INV_IA);
        let ib = ld_sym3(wide, wo + INV_IB);
        let inv_ma = ld(wide, wo + INV_MASS_A);
        let inv_mb = ld(wide, wo + INV_MASS_B);
        let normal = ld_v3(wide, wo + NORMAL);
        // Warm start touches only velocities; gather v/w alone (no delta_position/rotation).
        let (mut ba_v, mut ba_w) = gather_vel(state, idx, io, ident);
        let (mut bb_v, mut bb_w) = gather_vel(state, idx, io + LANES, ident);

        for pi in 0..MAX_POINTS {
            let pb = wo + POINTS + pi * POINT_STRIDE;
            let ra = ld_v3(wide, pb + P_ANCHOR_A);
            let rb = ld_v3(wide, pb + P_ANCHOR_B);
            let n_imp = ld(wide, pb + P_NORMAL_IMP);
            let impulse =
                Vec3W { x: n_imp.mul(normal.x), y: n_imp.mul(normal.y), z: n_imp.mul(normal.z) };
            ba_w = mul_sub_mvw(ba_w, ia, cross_w(ra, impulse));
            ba_v = mul_sub_svw(ba_v, inv_ma, impulse);
            bb_w = mul_add_mvw(bb_w, ib, cross_w(rb, impulse));
            bb_v = mul_add_svw(bb_v, inv_mb, impulse);
        }

        // Central friction
        {
            let ra = ld_v3(wide, wo + ORIGIN_A);
            let rb = ld_v3(wide, wo + ORIGIN_B);
            let fi = ld_v2(wide, wo + FRICTION_IMPULSE);
            let t1 = ld_v3(wide, wo + TANGENT1);
            let t2 = ld_v3(wide, wo + TANGENT2);
            let mut impulse = mul_svw(fi.x, t1);
            impulse = mul_add_svw(impulse, fi.y, t2);
            ba_w = mul_sub_mvw(ba_w, ia, cross_w(ra, impulse));
            ba_v = mul_sub_svw(ba_v, inv_ma, impulse);
            bb_w = mul_add_mvw(bb_w, ib, cross_w(rb, impulse));
            bb_v = mul_add_svw(bb_v, inv_mb, impulse);
        }
        // Central twist friction
        {
            let twist = ld(wide, wo + TWIST_IMPULSE);
            let impulse = mul_svw(twist, normal);
            ba_w = mul_sub_mvw(ba_w, ia, impulse);
            bb_w = mul_add_mvw(bb_w, ib, impulse);
        }
        // Rolling resistance
        {
            let impulse = ld_v3(wide, wo + ROLLING_IMPULSE);
            ba_w = mul_sub_mvw(ba_w, ia, impulse);
            bb_w = mul_add_mvw(bb_w, ib, impulse);
        }

        scatter(state, flags, idx, io, &ba_v, &ba_w, ident);
        scatter(state, flags, idx, io + LANES, &bb_v, &bb_w, ident);
    }
}

// --- solve / relax --------------------------------------------------------------------------

/// One TGS solve/relax pass over records `[start, start+count)` (b3SolveContacts_Convex). `use_bias`
/// selects the biased solve (position drift removal, friction skipped) vs the relax pass (no bias,
/// friction applied). `inv_h` is the inverse sub-step, `contact_speed` the max separation speed.
/// `worker` names the thread running this block ([`ident_rec`]); 0 on the serial path.
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
    worker: usize,
) {
    let inv_h_w = FloatW::splat(inv_h);
    let contact_speed_w = FloatW::splat(-contact_speed);
    let one = FloatW::splat(1.0);
    let zero = FloatW::zero();
    let eps = FloatW::splat(FLT_EPSILON);
    let ident = ident_rec(worker);

    for r in start..start + count {
        let wo = r * WIDE_STRIDE;
        let io = r * WIDE_IDX_STRIDE;
        let mut ba = gather(state, idx, io, ident);
        let mut bb = gather(state, idx, io + LANES, ident);

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
        let mut total_normal_impulse = zero;
        let mut total_twist_limit = zero;

        for pi in 0..MAX_POINTS {
            let pb = wo + POINTS + pi * POINT_STRIDE;
            let ra = ld_v3(wide, pb + P_ANCHOR_A);
            let rb = ld_v3(wide, pb + P_ANCHOR_B);

            let rs_a = rotate_vector_w(ba.dq, ra);
            let rs_b = rotate_vector_w(bb.dq, rb);
            let ds = add_vw(dp, sub_vw(rs_b, rs_a));
            let s = dot_w(normal, ds).add(ld(wide, pb + P_BASE_SEP));

            let mask = s.greater_than(zero);
            let spec_bias = s.mul(inv_h_w);
            let soft_bias = bias_rate.mul(s).max(contact_speed_w);
            let bias = FloatW::blend(soft_bias, spec_bias, mask);
            let point_mass_scale = FloatW::blend(mass_scale, one, mask);
            let point_impulse_scale = FloatW::blend(impulse_scale, zero, mask);

            let vra = add_vw(ba.v, cross_w(ba.w, ra));
            let vrb = add_vw(bb.v, cross_w(bb.w, rb));
            let vn = dot_w(sub_vw(vrb, vra), normal);

            let normal_mass = ld(wide, pb + P_NORMAL_MASS);
            let old_impulse = ld(wide, pb + P_NORMAL_IMP);
            // negImpulse = normalMass*(pointMassScale*vn + bias) + pointImpulseScale*normalImpulse
            let neg_impulse = normal_mass
                .mul(point_mass_scale.mul(vn).add(bias))
                .add(point_impulse_scale.mul(old_impulse));
            let new_impulse = old_impulse.sub(neg_impulse).max(zero);
            let delta_impulse = new_impulse.sub(old_impulse);
            st(wide, pb + P_NORMAL_IMP, new_impulse);
            let total = ld(wide, pb + P_TOTAL_NORMAL_IMP).add(new_impulse);
            st(wide, pb + P_TOTAL_NORMAL_IMP, total);

            total_normal_impulse = total_normal_impulse.add(new_impulse);
            total_twist_limit = total_twist_limit.add(ld(wide, pb + P_LEVER_ARM).mul(new_impulse));

            let p = mul_svw(delta_impulse, normal);
            ba.w = mul_sub_mvw(ba.w, ia, cross_w(ra, p));
            ba.v = mul_sub_svw(ba.v, inv_ma, p);
            bb.w = mul_add_mvw(bb.w, ib, cross_w(rb, p));
            bb.v = mul_add_svw(bb.v, inv_mb, p);
        }

        if !use_bias {
            // Rolling resistance
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
                let mut scale = FloatW::blend(one, normalize, mask);
                let rolling_mask = rolling_resistance.greater_than(zero);
                scale = FloatW::blend(zero, scale, rolling_mask);
                rolling = mul_svw(scale, rolling);
                st_v3(wide, wo + ROLLING_IMPULSE, rolling);
                let d = sub_vw(rolling, old);
                ba.w = mul_sub_mvw(ba.w, ia, d);
                bb.w = mul_add_mvw(bb.w, ib, d);
            }

            // Central twist friction
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

            // Central friction
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
                let delta = Vec2W { x: d0.x.neg(), y: d0.y.neg() };
                let old = ld_v2(wide, wo + FRICTION_IMPULSE);
                let mut new = add_v2w(old, delta);
                let friction = ld(wide, wo + FRICTION);
                let max_impulse = friction.mul(total_normal_impulse);
                let length_squared = new.x.mul(new.x).add(new.y.mul(new.y));
                let mask = length_squared.greater_than(max_impulse.mul(max_impulse));
                let normalize = max_impulse.div(length_squared.sqrt().add(eps));
                let scale = FloatW::blend(one, normalize, mask);
                new = Vec2W { x: scale.mul(new.x), y: scale.mul(new.y) };
                let delta = Vec2W { x: new.x.sub(old.x), y: new.y.sub(old.y) };
                st_v2(wide, wo + FRICTION_IMPULSE, new);
                let p = add_vw(mul_svw(delta.x, t1), mul_svw(delta.y, t2));
                ba.w = mul_sub_mvw(ba.w, ia, cross_w(ra, p));
                ba.v = mul_sub_svw(ba.v, inv_ma, p);
                bb.w = mul_add_mvw(bb.w, ib, cross_w(rb, p));
                bb.v = mul_add_svw(bb.v, inv_mb, p);
            }
        }

        scatter(state, flags, idx, io, &ba.v, &ba.w, ident);
        scatter(state, flags, idx, io + LANES, &bb.v, &bb.w, ident);
    }
}

// --- restitution ----------------------------------------------------------------------------

/// Apply restitution bounce over records `[start, start+count)` (b3ApplyRestitution_Convex). `worker`
/// names the thread running this block ([`ident_rec`]); 0 on the serial path.
#[allow(clippy::too_many_arguments)]
pub fn restitution(
    wide: Col<f32>,
    idx: Col<u32>,
    state: Col<f32>,
    flags: Col<u32>,
    start: usize,
    count: usize,
    threshold: f32,
    worker: usize,
) {
    let threshold_w = FloatW::splat(threshold);
    let zero = FloatW::zero();
    let ident = ident_rec(worker);

    for r in start..start + count {
        let wo = r * WIDE_STRIDE;
        let io = r * WIDE_IDX_STRIDE;
        let rest = ld(wide, wo + RESTITUTION);
        if rest.all_zero() {
            continue;
        }
        let mut ba = gather(state, idx, io, ident);
        let mut bb = gather(state, idx, io + LANES, ident);
        let ia = ld_sym3(wide, wo + INV_IA);
        let ib = ld_sym3(wide, wo + INV_IB);
        let inv_ma = ld(wide, wo + INV_MASS_A);
        let inv_mb = ld(wide, wo + INV_MASS_B);
        let normal = ld_v3(wide, wo + NORMAL);
        let restitution_mask = rest.equals(zero);

        for pi in 0..MAX_POINTS {
            let pb = wo + POINTS + pi * POINT_STRIDE;
            let rel_vel = ld(wide, pb + P_REL_VEL);
            let total_normal = ld(wide, pb + P_TOTAL_NORMAL_IMP);
            let mask1 = rel_vel.add(threshold_w).greater_than(zero);
            let mask2 = total_normal.equals(zero);
            let mask = mask1.or(mask2).or(restitution_mask);
            let mass = FloatW::blend(ld(wide, pb + P_NORMAL_MASS), zero, mask);

            let ra = ld_v3(wide, pb + P_ANCHOR_A);
            let rb = ld_v3(wide, pb + P_ANCHOR_B);
            let vra = add_vw(ba.v, cross_w(ba.w, ra));
            let vrb = add_vw(bb.v, cross_w(bb.w, rb));
            let vn = dot_w(sub_vw(vrb, vra), normal);

            let neg_impulse = mass.mul(vn.add(rest.mul(rel_vel)));
            let old_impulse = ld(wide, pb + P_NORMAL_IMP);
            let new_impulse = old_impulse.sub(neg_impulse).max(zero);
            let delta_impulse = new_impulse.sub(old_impulse);
            st(wide, pb + P_NORMAL_IMP, new_impulse);
            st(wide, pb + P_TOTAL_NORMAL_IMP, total_normal.add(delta_impulse));

            let p = mul_svw(delta_impulse, normal);
            ba.w = mul_sub_mvw(ba.w, ia, cross_w(ra, p));
            ba.v = mul_sub_svw(ba.v, inv_ma, p);
            bb.w = mul_add_mvw(bb.w, ib, cross_w(rb, p));
            bb.v = mul_add_svw(bb.v, inv_mb, p);
        }

        scatter(state, flags, idx, io, &ba.v, &ba.w, ident);
        scatter(state, flags, idx, io + LANES, &bb.v, &bb.w, ident);
    }
}

// --- store ----------------------------------------------------------------------------------

/// Write solved impulses back into the pool manifolds and flag hit events (b3StoreImpulses_Convex)
/// for records `[start, start+count)`. Each lane's contactId (from `meta`) resolves to its manifold
/// through the directory; `hit_event_threshold` is the (positive) hit-speed threshold.
pub fn store(
    wide: Col<f32>,
    meta: Col<u32>,
    dir: Col<u32>,
    pool: Col<f32>,
    start: usize,
    count: usize,
    hit_event_threshold: f32,
) {
    const ENABLE_HIT_EVENT: u32 = 0x0010_0000; // b3_simEnableHitEvent (contact.h)
    let neg_hit = -hit_event_threshold;

    for r in start..start + count {
        let wo = r * WIDE_STRIDE;
        let mo = r * WIDE_META_STRIDE;
        let lane_count = meta.get(mo + LANES) as usize;

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
            let contact_id = meta.get(mo + lane) as usize;
            let d = read_dir(dir, contact_id);
            let mpo = d.manifold_base * mabi::MANIFOLD_STRIDE; // convex: exactly one manifold
            let contact_flags = d.flags;

            pool.set(mpo + mabi::M_FRICTION, f1[lane] * t1x[lane] + f2[lane] * t2x[lane]);
            pool.set(mpo + mabi::M_FRICTION + 1, f1[lane] * t1y[lane] + f2[lane] * t2y[lane]);
            pool.set(mpo + mabi::M_FRICTION + 2, f1[lane] * t1z[lane] + f2[lane] * t2z[lane]);
            pool.set(mpo + mabi::M_TWIST, twist[lane]);
            pool.set(mpo + mabi::M_ROLLING, rix[lane]);
            pool.set(mpo + mabi::M_ROLLING + 1, riy[lane]);
            pool.set(mpo + mabi::M_ROLLING + 2, riz[lane]);

            let point_count = pool.get(mpo + mabi::M_POINT_COUNT).to_bits() as usize;
            for pi in 0..point_count {
                let pb = wo + POINTS + pi * POINT_STRIDE; // wide-record point
                let pp = mpo + mabi::M_POINTS + pi * mabi::POOL_POINT_STRIDE; // pool point
                pool.set(pp + mabi::P_NORMAL_IMPULSE, ld(wide, pb + P_NORMAL_IMP).to_array()[lane]);
                pool.set(
                    pp + mabi::P_TOTAL_NORMAL_IMPULSE,
                    ld(wide, pb + P_TOTAL_NORMAL_IMP).to_array()[lane],
                );
                pool.set(pp + mabi::P_NORMAL_VELOCITY, ld(wide, pb + P_REL_VEL).to_array()[lane]);
            }

            if contact_flags & ENABLE_HIT_EVENT != 0 {
                for pi in 0..point_count {
                    let pp = mpo + mabi::M_POINTS + pi * mabi::POOL_POINT_STRIDE;
                    let normal_velocity = pool.get(pp + mabi::P_NORMAL_VELOCITY);
                    let total_normal_impulse = pool.get(pp + mabi::P_TOTAL_NORMAL_IMPULSE);
                    if normal_velocity < neg_hit && total_normal_impulse > 0.0 {
                        set_hit(dir, contact_id);
                        break;
                    }
                }
            }
        }
    }
}
