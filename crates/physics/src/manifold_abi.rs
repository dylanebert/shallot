//! The persistent contact-manifold column ABI — the narrowphase → solver handoff, keyed by contactId.
//! The solver gathers each touching contact's material + manifold data straight out of the persistent
//! store (`src/manifoldstore.ts`, and in the wasm build `kernel/src/manifolds.rs`) instead of a
//! per-step marshaled copy. This module is the single source of truth for the two persistent columns'
//! layout + the per-record slot index; it is always compiled (the wasm `manifolds` region and the
//! native gold harnesses both mirror it), so it carries no wasm intrinsics.
//!
//! Two columns:
//!   - **directory** — one record per contactId: the contact material row, the per-step body sim
//!     indices, the block descriptor (manifoldCount + manifoldBase into the pool), and a per-step
//!     hit-event flag. TS writes the material/indices each step; the block descriptor is written on
//!     `alloc` (narrowphase); the solver reads it and writes the hit flag.
//!   - **pool** — the variable manifold records (b3Manifold, points inline). Each contact owns a run
//!     of `manifoldCount` records at `manifoldBase`. The narrowphase writes the manifolds; the solver
//!     reads them in `prepare` and writes the solved impulses back in `store` (the next step's warm
//!     start reads them there — they *are* the persistent state).
//!
//! A per-color **slot** index maps each scalar solver record to its contactId and its slice of the
//! transient constraint columns (which stay per-step-sequential); the wide path carries the contactId
//! per lane in its own meta column instead.

use crate::col::Col;
use crate::math::Vec3;

/// u32 stride of a directory record, matching `src/manifoldstore.ts` `DIR_STRIDE`. Material slots
/// (0..5) are f32 read through `f32::from_bits`; the rest are u32. Slots 0..11 are the solver's per-step
/// row; slots 12..21 are the convex narrowphase's persistent GJK/SAT cache (`DIR_CACHE`); slots 22..36 are
/// the in-kernel recycle loop's cached relative pose (`DIR_CACHED_*`, 4b.3c). Both tails are folded here
/// because they share the directory's key (contactId) and its grow-in-place lifecycle (the directory sits
/// at the region anchor, so a grow preserves it with no memmove).
pub const DIR_STRIDE: usize = 37;
const DIR_FRICTION: usize = 0;
const DIR_RESTITUTION: usize = 1;
const DIR_ROLLING_RESISTANCE: usize = 2;
const DIR_TANGENT_VELOCITY: usize = 3; // 3..5
const DIR_FLAGS: usize = 6;
const DIR_MANIFOLD_COUNT: usize = 7;
pub const DIR_MANIFOLD_BASE: usize = 8;
const DIR_INDEX_A: usize = 9;
const DIR_INDEX_B: usize = 10;
const DIR_HIT: usize = 11;
/// First slot of the convex cache union (10 slots, 12..21): the wider `SimplexCache` (metric f32 +
/// count + indexA[4] + indexB[4]) overlaps the narrower `SatCache` (separation f32 + type + indexA +
/// indexB + hit), exactly like box3d's union — a contact uses one or the other by shape pair.
pub const DIR_CACHE: usize = 12;

// The recycle record (4b.3c): the pose cached last full narrowphase that the in-kernel recycle test
// reads (has the contact barely moved?) and writes back (on a miss, caching this step's pose). Mirrors
// the TS `Contact.cachedRotation*`/`cachedRelativePose` fields, column-resident for kernel contacts. No
// cold needed: a contact is eligible to recycle only once TS has set `relativeTransformValid`, which
// happens only after this record was written, so a fresh/recycled contactId's stale bytes are never read.
/// Cached rotation of body A last full narrowphase (q4: 22..25).
pub const DIR_CACHED_ROT_A: usize = 22;
/// Cached rotation of body B last full narrowphase (q4: 26..29).
pub const DIR_CACHED_ROT_B: usize = 26;
/// Cached relative pose `inv(xfA)·xfB` last full narrowphase (p3 + q4: 30..36).
pub const DIR_CACHED_REL_POSE: usize = 30;

/// f32 stride of a pool manifold record (b3Manifold, 67 f32): an 11-slot header (normal 3,
/// frictionImpulse 3, twistImpulse 1, rollingImpulse 3, pointCount 1) followed by 4 inline point
/// records of 14 slots each. Matches `src/manifoldstore.ts` `MANIFOLD_STRIDE`.
pub const MANIFOLD_STRIDE: usize = 67;
pub const M_NORMAL: usize = 0; // 0..2
pub const M_FRICTION: usize = 3; // 3..5
pub const M_TWIST: usize = 6;
pub const M_ROLLING: usize = 7; // 7..9
pub const M_POINT_COUNT: usize = 10; // read via `to_bits`
pub const M_POINTS: usize = 11; // first inline point record
pub const POOL_POINT_STRIDE: usize = 14;
// Point sub-offsets, relative to a point record's start.
pub const P_ANCHOR_A: usize = 0; // 0..2
pub const P_ANCHOR_B: usize = 3; // 3..5
pub const P_SEPARATION: usize = 6;
// baseSeparation: the recycle test's per-step reference separation (cached by the finish pass,
// read + rewritten by the recycle-success separation update — see recycle.rs).
pub const P_BASE_SEPARATION: usize = 7;
pub const P_NORMAL_IMPULSE: usize = 8;
pub const P_TOTAL_NORMAL_IMPULSE: usize = 9;
pub const P_NORMAL_VELOCITY: usize = 10;
pub const P_FEATURE_ID: usize = 11; // u32 (bits stored through the f32 pool)
pub const P_TRIANGLE_INDEX: usize = 12; // i32
pub const P_PERSISTED: usize = 13; // u32 (0/1)

/// u32 stride of one scalar solver-record slot: contactId, manifoldStart (transient `mc` base),
/// pointStart (transient `mcp` base). Maps a per-color scalar record to its persistent contact and its
/// slice of the per-step-sequential transient constraint columns.
pub const SLOT_STRIDE: usize = 3;
pub const SLOT_CONTACT: usize = 0;
pub const SLOT_MANIFOLD_START: usize = 1;
pub const SLOT_POINT_START: usize = 2;

/// The directory record the solver gathers for one contact.
pub struct DirEntry {
    pub friction: f32,
    pub restitution: f32,
    pub rolling_resistance: f32,
    pub tangent_velocity: Vec3,
    pub flags: u32,
    pub manifold_count: usize,
    pub manifold_base: usize,
    pub index_a: u32,
    pub index_b: u32,
}

/// Read contact `contact_id`'s directory record (material row + body indices + block descriptor).
#[inline]
pub fn read_dir(dir: Col<u32>, contact_id: usize) -> DirEntry {
    let o = contact_id * DIR_STRIDE;
    DirEntry {
        friction: f32::from_bits(dir.get(o + DIR_FRICTION)),
        restitution: f32::from_bits(dir.get(o + DIR_RESTITUTION)),
        rolling_resistance: f32::from_bits(dir.get(o + DIR_ROLLING_RESISTANCE)),
        tangent_velocity: Vec3::new(
            f32::from_bits(dir.get(o + DIR_TANGENT_VELOCITY)),
            f32::from_bits(dir.get(o + DIR_TANGENT_VELOCITY + 1)),
            f32::from_bits(dir.get(o + DIR_TANGENT_VELOCITY + 2)),
        ),
        flags: dir.get(o + DIR_FLAGS),
        manifold_count: dir.get(o + DIR_MANIFOLD_COUNT) as usize,
        manifold_base: dir.get(o + DIR_MANIFOLD_BASE) as usize,
        index_a: dir.get(o + DIR_INDEX_A),
        index_b: dir.get(o + DIR_INDEX_B),
    }
}

/// Set contact `contact_id`'s per-step hit-event flag (the solver `store` phase; TS reads it back to
/// build the user-facing hit events).
#[inline]
pub fn set_hit(dir: Col<u32>, contact_id: usize) {
    dir.set(contact_id * DIR_STRIDE + DIR_HIT, 1);
}
