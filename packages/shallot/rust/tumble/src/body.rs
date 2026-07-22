//! The body-state SoA column ABI: the shared memory layout the TS side hands the kernel and every
//! solver phase reads. AoS-per-body (one body's fields contiguous) — matching box3d's own
//! `b3BodyState` / `b3BodySim` array storage, which the wide contact solver gathers from by index.
//! Scalar phases (integrate, finalize) read one body's struct; the wide solver (stage 3b) will gather
//! four. Layout is free (the fixture hash pins arithmetic + solve order, not memory); this is the
//! least-surprising choice and the one the gather pattern wants.
//!
//! Three body columns: `state` (velocity/delta), the `flags` u32 sidecar, and `sim` (the `b3BodySim`
//! fields the solver gathers — mass, inertia, damping, force/torque, rotation). Each carries only the
//! fields its phases consume, not the whole struct, and grows by appending so an offset never shifts.
//! The pose-finalize geometric fields (`center`, `localCenter`, `maxExtent`, `transform.p`) live in a
//! fourth `fin` column, kept out of `sim` because the wide gather never touches them (see below).

use crate::col::Col;
use crate::math::{Mat3, Quat, Vec3};

// --- state column ---------------------------------------------------------------------------

/// f32 stride of the per-body state column: linearVelocity(3) angularVelocity(3) deltaPosition(3)
/// deltaRotation(4) = 13 live fields, padded to 16 (power-of-two) so each record is cache-line
/// aligned for the wide solver's random gathers through contact indices (~14 passes/contact/step).
/// The pad is free — the fixture hash pins arithmetic + solve order, not memory layout. Body `i`
/// occupies `[i*STATE_STRIDE .. i*STATE_STRIDE+13]`; slots 13..16 are unused. Flags live in a
/// parallel `u32` column (bitwise ops want an integer view), one per body.
pub const STATE_STRIDE: usize = 16;

/// Count of live f32 fields in a state record (the padding is slots `STATE_LIVE..STATE_STRIDE`).
/// The gold layout and any serialize path walk the live fields; `STATE_STRIDE` governs indexing.
pub const STATE_LIVE: usize = 13;

/// The solver velocity/delta state of one body (b3BodyState, minus flags — see the `u32` column).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub delta_position: Vec3,
    pub delta_rotation: Quat,
}

#[inline]
pub fn read_state(col: Col<f32>, i: usize) -> State {
    let o = i * STATE_STRIDE;
    State {
        linear_velocity: Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2)),
        angular_velocity: Vec3::new(col.get(o + 3), col.get(o + 4), col.get(o + 5)),
        delta_position: Vec3::new(col.get(o + 6), col.get(o + 7), col.get(o + 8)),
        delta_rotation: Quat {
            v: Vec3::new(col.get(o + 9), col.get(o + 10), col.get(o + 11)),
            s: col.get(o + 12),
        },
    }
}

#[inline]
pub fn write_state(col: Col<f32>, i: usize, s: &State) {
    let o = i * STATE_STRIDE;
    col.set(o, s.linear_velocity.x);
    col.set(o + 1, s.linear_velocity.y);
    col.set(o + 2, s.linear_velocity.z);
    col.set(o + 3, s.angular_velocity.x);
    col.set(o + 4, s.angular_velocity.y);
    col.set(o + 5, s.angular_velocity.z);
    col.set(o + 6, s.delta_position.x);
    col.set(o + 7, s.delta_position.y);
    col.set(o + 8, s.delta_position.z);
    col.set(o + 9, s.delta_rotation.v.x);
    col.set(o + 10, s.delta_rotation.v.y);
    col.set(o + 11, s.delta_rotation.v.z);
    col.set(o + 12, s.delta_rotation.s);
}

// --- sim column -----------------------------------------------------------------------------

/// f32 stride of the per-body sim column, the integrate-relevant `b3BodySim` fields:
/// invMass(1) gravityScale(1) linearDamping(1) angularDamping(1) force(3) torque(3)
/// invInertiaLocal(9) invInertiaWorld(9) transform.q(4). Grows (append-only) as later phases
/// need more sim fields.
pub const SIM_STRIDE: usize = 32;

/// The `b3BodySim` fields the integrate phases read (velocity integration + gyroscopic step).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimIntegrate {
    pub inv_mass: f32,
    pub gravity_scale: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub force: Vec3,
    pub torque: Vec3,
    pub inv_inertia_local: Mat3,
    pub inv_inertia_world: Mat3,
    pub rotation: Quat,
}

#[inline]
fn mat3(col: Col<f32>, o: usize) -> Mat3 {
    Mat3 {
        cx: Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2)),
        cy: Vec3::new(col.get(o + 3), col.get(o + 4), col.get(o + 5)),
        cz: Vec3::new(col.get(o + 6), col.get(o + 7), col.get(o + 8)),
    }
}

#[inline]
pub fn read_sim(col: Col<f32>, i: usize) -> SimIntegrate {
    let o = i * SIM_STRIDE;
    SimIntegrate {
        inv_mass: col.get(o),
        gravity_scale: col.get(o + 1),
        linear_damping: col.get(o + 2),
        angular_damping: col.get(o + 3),
        force: Vec3::new(col.get(o + 4), col.get(o + 5), col.get(o + 6)),
        torque: Vec3::new(col.get(o + 7), col.get(o + 8), col.get(o + 9)),
        inv_inertia_local: mat3(col, o + 10),
        inv_inertia_world: mat3(col, o + 19),
        rotation: Quat {
            v: Vec3::new(col.get(o + 28), col.get(o + 29), col.get(o + 30)),
            s: col.get(o + 31),
        },
    }
}

// --- finalize column ------------------------------------------------------------------------

/// f32 stride of the per-body finalize column: the pure kinematic/geometric `b3BodySim` fields the
/// pose-finalize phase reads and writes — center(3) localCenter(3) maxExtent(3) transformP(3).
/// Separate from `sim` on purpose: these fields are never gathered by the wide contact solver
/// (finalize is scalar), so keeping them out of the `sim` record leaves that record the solver's hot
/// set — and leaves the contact/integrate gold harnesses' hardcoded stride-32 `sim` untouched.
/// `transform.q` stays in `sim` (finalize reads+writes it there, alongside the inertia/force fields).
pub const FIN_STRIDE: usize = 12;

/// f32 stride of the per-body finalize output: the two sleep/continuous decision scalars TS branches
/// on downstream — sleepVelocity, maxMotion. TS owns the branches (sleep, CCD, islands); the kernel
/// only hands it these two derived values so the branch arithmetic isn't reimplemented TS-side.
pub const FIN_OUT_STRIDE: usize = 2;

/// The finalize column's geometric fields for one body (`center` is read then overwritten with the
/// advanced value; `transform_p` is write-only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimFinalize {
    pub center: Vec3,
    pub local_center: Vec3,
    pub max_extent: Vec3,
}

#[inline]
pub fn read_fin(col: Col<f32>, i: usize) -> SimFinalize {
    let o = i * FIN_STRIDE;
    SimFinalize {
        center: Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2)),
        local_center: Vec3::new(col.get(o + 3), col.get(o + 4), col.get(o + 5)),
        max_extent: Vec3::new(col.get(o + 6), col.get(o + 7), col.get(o + 8)),
    }
}

#[inline]
pub fn write_fin_center(col: Col<f32>, i: usize, center: Vec3) {
    let o = i * FIN_STRIDE;
    col.set(o, center.x);
    col.set(o + 1, center.y);
    col.set(o + 2, center.z);
}

#[inline]
pub fn write_fin_transform_p(col: Col<f32>, i: usize, p: Vec3) {
    let o = i * FIN_STRIDE + 9;
    col.set(o, p.x);
    col.set(o + 1, p.y);
    col.set(o + 2, p.z);
}

// --- sim writers (finalize) -----------------------------------------------------------------
// Finalize advances the dynamics fields in place: the rotation it just integrated, the world inertia
// derived from it, and the force/torque accumulators it clears for the next step. Offsets match the
// `read_sim` layout above (force@4 torque@7 invInertiaWorld@19 rotation@28).

#[inline]
pub fn write_sim_rotation(col: Col<f32>, i: usize, q: Quat) {
    let o = i * SIM_STRIDE + 28;
    col.set(o, q.v.x);
    col.set(o + 1, q.v.y);
    col.set(o + 2, q.v.z);
    col.set(o + 3, q.s);
}

#[inline]
pub fn write_sim_inv_inertia_world(col: Col<f32>, i: usize, m: Mat3) {
    let o = i * SIM_STRIDE + 19;
    col.set(o, m.cx.x);
    col.set(o + 1, m.cx.y);
    col.set(o + 2, m.cx.z);
    col.set(o + 3, m.cy.x);
    col.set(o + 4, m.cy.y);
    col.set(o + 5, m.cy.z);
    col.set(o + 6, m.cz.x);
    col.set(o + 7, m.cz.y);
    col.set(o + 8, m.cz.z);
}

/// Zero the force + torque accumulators (sim slots 4..10), consumed by the next step's integrate.
#[inline]
pub fn clear_sim_force_torque(col: Col<f32>, i: usize) {
    let o = i * SIM_STRIDE + 4;
    for k in 0..6 {
        col.set(o + k, 0.0);
    }
}

// --- sim2 column ----------------------------------------------------------------------------

/// 4-byte stride of the second resident sim column: the `BodySim` fields the per-step `sim`/`fin`
/// columns omit — rotation0(4) center0(3) minExtent(1) maxAngularVelocity(1) bodyId(1) flags(1) — plus
/// the headShapeId lane below, 12 slots exactly. The region backs the full `BodySim`/`BodyState` surface
/// (the 4a.2/4a.3 views need every field), laid out once here so a later slice never re-spaces it.
pub const SIM2_STRIDE: usize = 12;

/// Slot of the sweep-base rotation (`BodySim.rotation0`, a quat: v3 + s) in the sim2 column — the
/// pose the next step's continuous sweep rotates from. Finalize writes it for every non-fast body
/// (`columns.ts` `S2_ROTATION0`).
pub const S2_ROTATION0: usize = 0;

/// Slot of the sweep-base center (`BodySim.center0`, a Vec3) in the sim2 column — the pose the next
/// step's continuous sweep translates from (`columns.ts` `S2_CENTER0`).
pub const S2_CENTER0: usize = 4;

/// Slot of `BodySim.minExtent` in the sim2 column — the smallest shape half-extent, against which the
/// finalize fast-body test compares this step's motion (`columns.ts` `S2_MIN_EXTENT`).
pub const S2_MIN_EXTENT: usize = 7;

/// Slot of the head of the body's shape list (its `nextShapeId` chain runs through the shape column,
/// shapes.rs), written TS-side at marshal-in and on any shape-list mutation of an awake body. The lane
/// rides the whole-record migration an awake-set swap-remove does. `NULL_SHAPE` when the body has no
/// shapes. Read through a u32 view of the column.
pub const S2_HEAD_SHAPE: usize = 11;

// --- flags ----------------------------------------------------------------------------------

/// b3BodyFlags bits the integrate phases read/write. The full set lives TS-side (`body.ts`); the
/// kernel mirrors only the bits its phases touch.
pub mod flags {
    pub const LOCK_LINEAR_X: u32 = 0x0000_0001;
    pub const LOCK_LINEAR_Y: u32 = 0x0000_0002;
    pub const LOCK_LINEAR_Z: u32 = 0x0000_0004;
    pub const LOCK_ANGULAR_X: u32 = 0x0000_0008;
    pub const LOCK_ANGULAR_Y: u32 = 0x0000_0010;
    pub const LOCK_ANGULAR_Z: u32 = 0x0000_0020;
    pub const IS_SPEED_CAPPED: u32 = 0x0000_0100;
    pub const ALLOW_FAST_ROTATION: u32 = 0x0000_0400;
    /// Set on dynamic bodies; the contact solver only writes velocity back to bodies that carry it.
    pub const DYNAMIC: u32 = 0x0000_1000;
}
