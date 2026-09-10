//! The joint column ABI — the constraint-graph joint → kernel solver handoff, one flat record per
//! joint slot. This is the single source of truth for the joint column layout + the per-record slot
//! index; the wasm arena (`arena.rs`), the staged solve (`solve.rs`), the native gold harness
//! (`tests/joint_gold.rs`), and the TS marshal (`src/jointcolumns.ts`) all mirror it. Always compiled
//! (native + wasm), so it carries no wasm intrinsics.
//!
//! **The record is one f32 column, slot-scalar (indexed one joint at a time), keyed by a flat
//! per-color-concatenated joint slot** — box3d's `jointPrepareSpans` order. Unlike contacts there is
//! no separate persistent store: joints are few, so the whole record (config + the warm-start impulses
//! + the per-step prepared scratch) marshals in each step and reads back out (`src/jointcolumns.ts`),
//! and the impulses live authoritatively TS-side (`JointSim.data`). The column is a per-step transient
//! arena column, reserved pre-solve like the wide/mesh columns.
//!
//! Three integer fields (type, the two resident state indices) ride the f32 column through
//! `f32::to_bits`/`from_bits`, matching the manifold pool's `pointCount`/`featureId` pattern.
//!
//! ## The template (joints-in-kernel 3c copies)
//!
//! The record is a **common header** followed by a **type payload**. The header is identical for every
//! joint type; the payload is the per-type union (box3d's `b3JointSim` union), reinterpreted by
//! `J_TYPE`. Adding a joint type in 3c: give it a `prepare`/`warm_start`/`solve` over the header + its
//! own payload reading; grow `JOINT_STRIDE` if its payload is wider than the current max. The header
//! carries everything `prepare` needs from the bodies (marshaled by TS each step: the two sim indices,
//! invMass/invInertiaWorld, and the pose fields `prepare` derives the anchors from — a body's world
//! rotation, local center, and center) plus the base constraint frequency + the base-computed
//! `constraintSoftness`. The payload carries the type config, the persistent impulses, and the prepared
//! scratch (`prepare` writes, `warm_start`/`solve` read).

use crate::col::Col;
use crate::math::{Mat3, Quat, Transform, Vec3};

/// Joint type codes — the TS `JointType` values verbatim (`src/joint.ts`), the dispatch key in
/// `J_TYPE`. Only `Distance` has a kernel path today; the rest are the 3c partition.
pub const TY_PARALLEL: u32 = 0;
pub const TY_DISTANCE: u32 = 1;
pub const TY_FILTER: u32 = 2;
pub const TY_MOTOR: u32 = 3;
pub const TY_PRISMATIC: u32 = 4;
pub const TY_REVOLUTE: u32 = 5;
pub const TY_SPHERICAL: u32 = 6;
pub const TY_WELD: u32 = 7;
pub const TY_WHEEL: u32 = 8;

/// The sim-index sentinel for a non-awake (static / sleeping) body — TS `NULL_INDEX` (-1) through a u32
/// view. `warm_start`/`solve` read the identity body state for a null index (box3d's `dummyState`).
pub const NULL_INDEX: u32 = u32::MAX;

// --- common header (every joint type) ---------------------------------------------------------

/// Joint type (`TY_*`), read via `to_bits`.
pub const J_TYPE: usize = 0;
/// Resident `state`-column index of body A, or `NULL_INDEX` (u32 bits).
pub const J_SIM_INDEX_A: usize = 1;
/// Resident `state`-column index of body B, or `NULL_INDEX` (u32 bits).
pub const J_SIM_INDEX_B: usize = 2;
/// Body A inverse mass.
pub const J_INV_MASS_A: usize = 3;
/// Body B inverse mass.
pub const J_INV_MASS_B: usize = 4;
/// Body A world inverse inertia (mat3, column-major: 5..13).
pub const J_INV_IA: usize = 5;
/// Body B world inverse inertia (mat3: 14..22).
pub const J_INV_IB: usize = 14;
/// Body A world rotation quaternion (23..26) — `prepare` rotates the local anchor by it.
pub const J_QA: usize = 23;
/// Body A local center of mass (27..29).
pub const J_LOCAL_CENTER_A: usize = 27;
/// Body A center (world COM, 30..32) — `prepare`'s `deltaCenter`.
pub const J_CENTER_A: usize = 30;
/// Body B world rotation quaternion (33..36).
pub const J_QB: usize = 33;
/// Body B local center of mass (37..39).
pub const J_LOCAL_CENTER_B: usize = 37;
/// Body B center (world COM, 40..42).
pub const J_CENTER_B: usize = 40;
/// Body-A local joint frame (Transform: p 43..45, q 46..49) — the anchor point + frame in A's body space.
pub const J_LOCAL_FRAME_A: usize = 43;
/// Body-B local joint frame (Transform: p 50..52, q 53..56).
pub const J_LOCAL_FRAME_B: usize = 50;
/// Base constraint hertz (b3JointSim.constraintHertz).
pub const J_CONSTRAINT_HERTZ: usize = 57;
/// Base constraint damping ratio.
pub const J_CONSTRAINT_DAMPING: usize = 58;
/// Base constraint softness (biasRate, massScale, impulseScale: 59..61) — `prepare_joint` writes it.
pub const J_CONSTRAINT_SOFTNESS: usize = 59;

/// First slot of the type payload union. Everything below is reinterpreted by `J_TYPE`.
pub const J_PAYLOAD: usize = 62;

// --- distance-joint payload -------------------------------------------------------------------
// Config (TS writes at marshal), then the persistent impulses (marshaled in/out), then the prepared
// scratch (`prepare` writes, `warm_start`/`solve` read).

/// Rest length.
pub const DJ_LENGTH: usize = J_PAYLOAD; // 62
/// Spring frequency.
pub const DJ_HERTZ: usize = J_PAYLOAD + 1;
/// Spring damping ratio.
pub const DJ_DAMPING_RATIO: usize = J_PAYLOAD + 2;
/// Spring lower force bound.
pub const DJ_LOWER_SPRING_FORCE: usize = J_PAYLOAD + 3;
/// Spring upper force bound.
pub const DJ_UPPER_SPRING_FORCE: usize = J_PAYLOAD + 4;
/// Lower length limit.
pub const DJ_MIN_LENGTH: usize = J_PAYLOAD + 5;
/// Upper length limit.
pub const DJ_MAX_LENGTH: usize = J_PAYLOAD + 6;
/// Max motor force.
pub const DJ_MAX_MOTOR_FORCE: usize = J_PAYLOAD + 7;
/// Motor speed.
pub const DJ_MOTOR_SPEED: usize = J_PAYLOAD + 8;
/// Enable bitfield (u32 bits): bit0 spring, bit1 limit, bit2 motor.
pub const DJ_ENABLE: usize = J_PAYLOAD + 9;
/// Persistent axial impulse (marshaled).
pub const DJ_IMPULSE: usize = J_PAYLOAD + 10;
/// Persistent lower-limit impulse.
pub const DJ_LOWER_IMPULSE: usize = J_PAYLOAD + 11;
/// Persistent upper-limit impulse.
pub const DJ_UPPER_IMPULSE: usize = J_PAYLOAD + 12;
/// Persistent motor impulse.
pub const DJ_MOTOR_IMPULSE: usize = J_PAYLOAD + 13;
/// Prepared world anchor A relative to A's center (14..16 off payload).
pub const DJ_ANCHOR_A: usize = J_PAYLOAD + 14;
/// Prepared world anchor B relative to B's center (17..19).
pub const DJ_ANCHOR_B: usize = J_PAYLOAD + 17;
/// Prepared center delta `centerB - centerA` (20..22).
pub const DJ_DELTA_CENTER: usize = J_PAYLOAD + 20;
/// Prepared axial effective mass.
pub const DJ_AXIAL_MASS: usize = J_PAYLOAD + 23;
/// Prepared spring softness (biasRate, massScale, impulseScale: 24..26).
pub const DJ_DIST_SOFTNESS: usize = J_PAYLOAD + 24;

/// Enable bits.
pub const DJ_ENABLE_SPRING: u32 = 0x1;
pub const DJ_ENABLE_LIMIT: u32 = 0x2;
pub const DJ_ENABLE_MOTOR: u32 = 0x4;

// --- weld-joint payload -----------------------------------------------------------------------
// Rigidly fixes two bodies: an angular constraint + a point-to-point linear constraint, each softened
// by an optional spring (linearHertz / angularHertz — a zero hertz falls back to the base constraint
// softness). Config, then the two persistent impulses, then prepare's scratch (frames, angular mass,
// the resolved softnesses, fixedRotation).

pub const WJ_LINEAR_HERTZ: usize = J_PAYLOAD; // 62
pub const WJ_LINEAR_DAMPING_RATIO: usize = J_PAYLOAD + 1;
pub const WJ_ANGULAR_HERTZ: usize = J_PAYLOAD + 2;
pub const WJ_ANGULAR_DAMPING_RATIO: usize = J_PAYLOAD + 3;
/// Persistent linear impulse (vec3, marshaled).
pub const WJ_LINEAR_IMPULSE: usize = J_PAYLOAD + 4;
/// Persistent angular impulse (vec3, marshaled).
pub const WJ_ANGULAR_IMPULSE: usize = J_PAYLOAD + 7;
/// Prepared world frame A (Transform p 10..12, q 13..16).
pub const WJ_FRAME_A: usize = J_PAYLOAD + 10;
/// Prepared world frame B (Transform p 17..19, q 20..23).
pub const WJ_FRAME_B: usize = J_PAYLOAD + 17;
/// Prepared center delta `centerB - centerA` (24..26).
pub const WJ_DELTA_CENTER: usize = J_PAYLOAD + 24;
/// Prepared angular mass = inv(invIA + invIB) (mat3 27..35).
pub const WJ_ANGULAR_MASS: usize = J_PAYLOAD + 27;
/// Prepared linear spring softness (biasRate, massScale, impulseScale 36..38).
pub const WJ_LINEAR_SPRING: usize = J_PAYLOAD + 36;
/// Prepared angular spring softness (39..41).
pub const WJ_ANGULAR_SPRING: usize = J_PAYLOAD + 39;
/// Prepared `fixedRotation` flag (1.0 = true), read `!= 0.0`.
pub const WJ_FIXED_ROTATION: usize = J_PAYLOAD + 42;

// --- revolute-joint payload -------------------------------------------------------------------
// The hinge: a point-to-point linear constraint + a collinearity (perpendicularity) constraint about
// body A's local z, with an optional angular spring, motor, and twist limit. Config, the persistent
// impulses (linear vec3 + perp vec2 + four scalar), then prepare's scratch (frames, axes, axial mass,
// spring softness, fixedRotation).

pub const RJ_HERTZ: usize = J_PAYLOAD; // 62
pub const RJ_DAMPING_RATIO: usize = J_PAYLOAD + 1;
pub const RJ_MAX_MOTOR_TORQUE: usize = J_PAYLOAD + 2;
pub const RJ_MOTOR_SPEED: usize = J_PAYLOAD + 3;
pub const RJ_TARGET_ANGLE: usize = J_PAYLOAD + 4;
pub const RJ_LOWER_ANGLE: usize = J_PAYLOAD + 5;
pub const RJ_UPPER_ANGLE: usize = J_PAYLOAD + 6;
/// Enable bitfield (u32 bits): bit0 spring, bit1 motor, bit2 limit.
pub const RJ_ENABLE: usize = J_PAYLOAD + 7;
/// Persistent linear impulse (vec3, marshaled).
pub const RJ_LINEAR_IMPULSE: usize = J_PAYLOAD + 8;
/// Persistent perpendicular (collinearity) impulse (vec2, marshaled).
pub const RJ_PERP_IMPULSE: usize = J_PAYLOAD + 11;
pub const RJ_SPRING_IMPULSE: usize = J_PAYLOAD + 13;
pub const RJ_MOTOR_IMPULSE: usize = J_PAYLOAD + 14;
pub const RJ_LOWER_IMPULSE: usize = J_PAYLOAD + 15;
pub const RJ_UPPER_IMPULSE: usize = J_PAYLOAD + 16;
/// Prepared world frame A (Transform p 17..19, q 20..23).
pub const RJ_FRAME_A: usize = J_PAYLOAD + 17;
/// Prepared world frame B (Transform p 24..26, q 27..30).
pub const RJ_FRAME_B: usize = J_PAYLOAD + 24;
/// Prepared hinge axis = frameA.q · z (31..33).
pub const RJ_ROTATION_AXIS_Z: usize = J_PAYLOAD + 31;
/// Prepared warm-start perpendicular axes (34..36, 37..39).
pub const RJ_PERP_AXIS_X: usize = J_PAYLOAD + 34;
pub const RJ_PERP_AXIS_Y: usize = J_PAYLOAD + 37;
/// Prepared center delta `centerB - centerA` (40..42).
pub const RJ_DELTA_CENTER: usize = J_PAYLOAD + 40;
/// Prepared axial effective mass (43).
pub const RJ_AXIAL_MASS: usize = J_PAYLOAD + 43;
/// Prepared spring softness (biasRate, massScale, impulseScale 44..46).
pub const RJ_SPRING_SOFTNESS: usize = J_PAYLOAD + 44;
/// Prepared `fixedRotation` flag (1.0 = true), read `!= 0.0`.
pub const RJ_FIXED_ROTATION: usize = J_PAYLOAD + 47;

/// Enable bits.
pub const RJ_ENABLE_SPRING: u32 = 0x1;
pub const RJ_ENABLE_MOTOR: u32 = 0x2;
pub const RJ_ENABLE_LIMIT: u32 = 0x4;

// --- spherical-joint payload ------------------------------------------------------------------
// Ball-and-socket: a point-to-point linear constraint + optional angular spring (toward a target
// rotation), angular-velocity motor (vec3, magnitude-clamped), a swing (cone) limit about body A's
// local z, and a twist limit about the shared axis. Config, the persistent impulses (three vec3 + three
// scalar), then prepare's scratch (frames, swing/twist axes, rotation mass, spring softness, masses,
// fixedRotation). The swing axis / twist jacobian / their masses default to zero when the limit is off.

pub const SJ_HERTZ: usize = J_PAYLOAD; // 62
pub const SJ_DAMPING_RATIO: usize = J_PAYLOAD + 1;
pub const SJ_MAX_MOTOR_TORQUE: usize = J_PAYLOAD + 2;
pub const SJ_MOTOR_VELOCITY: usize = J_PAYLOAD + 3; // vec3 3..5
pub const SJ_LOWER_TWIST_ANGLE: usize = J_PAYLOAD + 6;
pub const SJ_UPPER_TWIST_ANGLE: usize = J_PAYLOAD + 7;
pub const SJ_CONE_ANGLE: usize = J_PAYLOAD + 8;
pub const SJ_TARGET_ROTATION: usize = J_PAYLOAD + 9; // quat 9..12
/// Enable bitfield (u32 bits): bit0 spring, bit1 motor, bit2 cone limit, bit3 twist limit.
pub const SJ_ENABLE: usize = J_PAYLOAD + 13;
/// Persistent linear impulse (vec3, marshaled).
pub const SJ_LINEAR_IMPULSE: usize = J_PAYLOAD + 14;
/// Persistent angular spring impulse (vec3).
pub const SJ_SPRING_IMPULSE: usize = J_PAYLOAD + 17;
/// Persistent motor impulse (vec3).
pub const SJ_MOTOR_IMPULSE: usize = J_PAYLOAD + 20;
pub const SJ_LOWER_TWIST_IMPULSE: usize = J_PAYLOAD + 23;
pub const SJ_UPPER_TWIST_IMPULSE: usize = J_PAYLOAD + 24;
pub const SJ_SWING_IMPULSE: usize = J_PAYLOAD + 25;
/// Prepared world frame A (Transform p 26..28, q 29..32).
pub const SJ_FRAME_A: usize = J_PAYLOAD + 26;
/// Prepared world frame B (Transform p 33..35, q 36..39).
pub const SJ_FRAME_B: usize = J_PAYLOAD + 33;
/// Prepared center delta `centerB - centerA` (40..42).
pub const SJ_DELTA_CENTER: usize = J_PAYLOAD + 40;
/// Prepared cone-limit swing axis (43..45; zero when the cone limit is off).
pub const SJ_SWING_AXIS: usize = J_PAYLOAD + 43;
/// Prepared twist-limit jacobian (46..48; zero when the twist limit is off).
pub const SJ_TWIST_JACOBIAN: usize = J_PAYLOAD + 46;
/// Prepared rotation mass = inv(invIA + invIB) (mat3 49..57; zero for a fixed-rotation pair).
pub const SJ_ROTATION_MASS: usize = J_PAYLOAD + 49;
/// Prepared cone-limit effective mass (58).
pub const SJ_SWING_MASS: usize = J_PAYLOAD + 58;
/// Prepared twist-limit effective mass (59).
pub const SJ_TWIST_MASS: usize = J_PAYLOAD + 59;
/// Prepared spring softness (biasRate, massScale, impulseScale 60..62).
pub const SJ_SPRING_SOFTNESS: usize = J_PAYLOAD + 60;
/// Prepared `fixedRotation` flag (1.0 = true), read `!= 0.0`.
pub const SJ_FIXED_ROTATION: usize = J_PAYLOAD + 63;

/// Enable bits.
pub const SJ_ENABLE_SPRING: u32 = 0x1;
pub const SJ_ENABLE_MOTOR: u32 = 0x2;
pub const SJ_ENABLE_CONE_LIMIT: u32 = 0x4;
pub const SJ_ENABLE_TWIST_LIMIT: u32 = 0x8;

// --- prismatic-joint payload ------------------------------------------------------------------
// Slider: two bodies slide along body A's local x-axis. A 2-DOF point-to-line linear constraint + a
// full rotation constraint, plus optional axial spring (toward a target translation), motor
// (force-clamped), and translation limits. Config, the persistent impulses (perp vec2 + angular vec3 +
// four scalar), then prepare's scratch (frames, the three world axes, delta center, rotation mass,
// spring softness, fixedRotation). All scratch is written unconditionally — the axial effective mass is
// recomputed fresh each solve (not prepared), so no conditional-scratch zeroing is needed.

pub const PJ_HERTZ: usize = J_PAYLOAD; // 62
pub const PJ_DAMPING_RATIO: usize = J_PAYLOAD + 1;
pub const PJ_MAX_MOTOR_FORCE: usize = J_PAYLOAD + 2;
pub const PJ_MOTOR_SPEED: usize = J_PAYLOAD + 3;
pub const PJ_TARGET_TRANSLATION: usize = J_PAYLOAD + 4;
pub const PJ_LOWER_TRANSLATION: usize = J_PAYLOAD + 5;
pub const PJ_UPPER_TRANSLATION: usize = J_PAYLOAD + 6;
/// Enable bitfield (u32 bits): bit0 spring, bit1 motor, bit2 limit.
pub const PJ_ENABLE: usize = J_PAYLOAD + 7;
/// Persistent perpendicular impulse (vec2, marshaled): 8..9.
pub const PJ_PERP_IMPULSE: usize = J_PAYLOAD + 8;
/// Persistent angular impulse (vec3): 10..12.
pub const PJ_ANGULAR_IMPULSE: usize = J_PAYLOAD + 10;
pub const PJ_SPRING_IMPULSE: usize = J_PAYLOAD + 13;
pub const PJ_MOTOR_IMPULSE: usize = J_PAYLOAD + 14;
pub const PJ_LOWER_IMPULSE: usize = J_PAYLOAD + 15;
pub const PJ_UPPER_IMPULSE: usize = J_PAYLOAD + 16;
/// Prepared world frame A (Transform p 17..19, q 20..23).
pub const PJ_FRAME_A: usize = J_PAYLOAD + 17;
/// Prepared world frame B (Transform p 24..26, q 27..30).
pub const PJ_FRAME_B: usize = J_PAYLOAD + 24;
/// Prepared world joint axis = frameA rotation matrix column x (31..33).
pub const PJ_JOINT_AXIS: usize = J_PAYLOAD + 31;
/// Prepared world perpendicular axis Y = frameA column y (34..36).
pub const PJ_PERP_AXIS_Y: usize = J_PAYLOAD + 34;
/// Prepared world perpendicular axis Z = frameA column z (37..39).
pub const PJ_PERP_AXIS_Z: usize = J_PAYLOAD + 37;
/// Prepared center delta `centerB - centerA` (40..42).
pub const PJ_DELTA_CENTER: usize = J_PAYLOAD + 40;
/// Prepared rotation mass = inv(invIA + invIB) (mat3 43..51; always inverted, unlike spherical).
pub const PJ_ROTATION_MASS: usize = J_PAYLOAD + 43;
/// Prepared spring softness (biasRate, massScale, impulseScale 52..54).
pub const PJ_SPRING_SOFTNESS: usize = J_PAYLOAD + 52;
/// Prepared `fixedRotation` flag (1.0 = true), read `!= 0.0`.
pub const PJ_FIXED_ROTATION: usize = J_PAYLOAD + 55;

/// Enable bits.
pub const PJ_ENABLE_SPRING: u32 = 0x1;
pub const PJ_ENABLE_MOTOR: u32 = 0x2;
pub const PJ_ENABLE_LIMIT: u32 = 0x4;

// --- wheel-joint payload ----------------------------------------------------------------------
// Car suspension: the wheel slides along body A's local x (suspension spring + limits), spins about
// body B's local z (optional spin motor), and either steers about body A's x (soft spring + limits) or
// is held collinear. Closed by a point-to-line linear constraint. Config, the persistent impulses (two
// vec2 + seven scalar), then prepare's scratch (frames, delta center, three effective masses, two
// softnesses, fixedRotation). All scratch is written unconditionally in prepare — no conditional zeroing.

pub const WHJ_MAX_SPIN_TORQUE: usize = J_PAYLOAD; // 62
pub const WHJ_SPIN_SPEED: usize = J_PAYLOAD + 1;
pub const WHJ_LOWER_SUSPENSION_LIMIT: usize = J_PAYLOAD + 2;
pub const WHJ_UPPER_SUSPENSION_LIMIT: usize = J_PAYLOAD + 3;
pub const WHJ_SUSPENSION_HERTZ: usize = J_PAYLOAD + 4;
pub const WHJ_SUSPENSION_DAMPING_RATIO: usize = J_PAYLOAD + 5;
pub const WHJ_LOWER_STEERING_LIMIT: usize = J_PAYLOAD + 6;
pub const WHJ_UPPER_STEERING_LIMIT: usize = J_PAYLOAD + 7;
pub const WHJ_TARGET_STEERING_ANGLE: usize = J_PAYLOAD + 8;
pub const WHJ_MAX_STEERING_TORQUE: usize = J_PAYLOAD + 9;
pub const WHJ_STEERING_HERTZ: usize = J_PAYLOAD + 10;
pub const WHJ_STEERING_DAMPING_RATIO: usize = J_PAYLOAD + 11;
/// Enable bitfield (u32 bits): bit0 spin motor, bit1 suspension spring, bit2 suspension limit,
/// bit3 steering, bit4 steering limit.
pub const WHJ_ENABLE: usize = J_PAYLOAD + 12;
/// Persistent point-to-line linear impulse (vec2, marshaled): 13..14.
pub const WHJ_LINEAR_IMPULSE: usize = J_PAYLOAD + 13;
/// Persistent collinearity angular impulse (vec2): 15..16.
pub const WHJ_ANGULAR_IMPULSE: usize = J_PAYLOAD + 15;
pub const WHJ_SPIN_IMPULSE: usize = J_PAYLOAD + 17;
pub const WHJ_SUSPENSION_SPRING_IMPULSE: usize = J_PAYLOAD + 18;
pub const WHJ_LOWER_SUSPENSION_IMPULSE: usize = J_PAYLOAD + 19;
pub const WHJ_UPPER_SUSPENSION_IMPULSE: usize = J_PAYLOAD + 20;
pub const WHJ_STEERING_SPRING_IMPULSE: usize = J_PAYLOAD + 21;
pub const WHJ_LOWER_STEERING_IMPULSE: usize = J_PAYLOAD + 22;
pub const WHJ_UPPER_STEERING_IMPULSE: usize = J_PAYLOAD + 23;
/// Prepared world frame A (Transform p 24..26, q 27..30).
pub const WHJ_FRAME_A: usize = J_PAYLOAD + 24;
/// Prepared world frame B (Transform p 31..33, q 34..37).
pub const WHJ_FRAME_B: usize = J_PAYLOAD + 31;
/// Prepared center delta `centerB - centerA` (38..40).
pub const WHJ_DELTA_CENTER: usize = J_PAYLOAD + 38;
/// Prepared spin (rotation-axis) effective mass (41).
pub const WHJ_SPIN_MASS: usize = J_PAYLOAD + 41;
/// Prepared suspension (axial) effective mass (42).
pub const WHJ_SUSPENSION_MASS: usize = J_PAYLOAD + 42;
/// Prepared steering (twist-axis) effective mass (43).
pub const WHJ_STEERING_MASS: usize = J_PAYLOAD + 43;
/// Prepared suspension spring softness (biasRate, massScale, impulseScale 44..46).
pub const WHJ_SUSPENSION_SOFTNESS: usize = J_PAYLOAD + 44;
/// Prepared steering spring softness (47..49).
pub const WHJ_STEERING_SOFTNESS: usize = J_PAYLOAD + 47;
/// Prepared `fixedRotation` flag (1.0 = true), read `!= 0.0`.
pub const WHJ_FIXED_ROTATION: usize = J_PAYLOAD + 50;

/// Enable bits.
pub const WHJ_ENABLE_SPIN_MOTOR: u32 = 0x1;
pub const WHJ_ENABLE_SUSPENSION_SPRING: u32 = 0x2;
pub const WHJ_ENABLE_SUSPENSION_LIMIT: u32 = 0x4;
pub const WHJ_ENABLE_STEERING: u32 = 0x8;
pub const WHJ_ENABLE_STEERING_LIMIT: u32 = 0x10;

// --- motor-joint payload ----------------------------------------------------------------------
// Drives two bodies toward target relative linear + angular velocities (each capped by a max
// force/torque) and optionally holds a soft spring at the reference pose. Four independent sub-solves
// gated on their max effort. No enable bitfield (each branch keys on `max* > 0`); no `fixedRotation`
// (motor's solve never reads it). Config, the four persistent vec3 impulses, then prepare's scratch
// (frames, delta center, the two resolved springs, angular mass). All scratch is written
// unconditionally in prepare — no conditional zeroing.

pub const MJ_LINEAR_VELOCITY: usize = J_PAYLOAD; // vec3 0..2
pub const MJ_ANGULAR_VELOCITY: usize = J_PAYLOAD + 3; // vec3 3..5
pub const MJ_MAX_VELOCITY_FORCE: usize = J_PAYLOAD + 6;
pub const MJ_MAX_VELOCITY_TORQUE: usize = J_PAYLOAD + 7;
pub const MJ_LINEAR_HERTZ: usize = J_PAYLOAD + 8;
pub const MJ_LINEAR_DAMPING_RATIO: usize = J_PAYLOAD + 9;
pub const MJ_ANGULAR_HERTZ: usize = J_PAYLOAD + 10;
pub const MJ_ANGULAR_DAMPING_RATIO: usize = J_PAYLOAD + 11;
pub const MJ_MAX_SPRING_FORCE: usize = J_PAYLOAD + 12;
pub const MJ_MAX_SPRING_TORQUE: usize = J_PAYLOAD + 13;
/// Persistent linear velocity-motor impulse (vec3, marshaled): 14..16.
pub const MJ_LINEAR_VELOCITY_IMPULSE: usize = J_PAYLOAD + 14;
/// Persistent angular velocity-motor impulse (vec3): 17..19.
pub const MJ_ANGULAR_VELOCITY_IMPULSE: usize = J_PAYLOAD + 17;
/// Persistent linear spring impulse (vec3): 20..22.
pub const MJ_LINEAR_SPRING_IMPULSE: usize = J_PAYLOAD + 20;
/// Persistent angular spring impulse (vec3): 23..25.
pub const MJ_ANGULAR_SPRING_IMPULSE: usize = J_PAYLOAD + 23;
/// Prepared world frame A (Transform p 26..28, q 29..32).
pub const MJ_FRAME_A: usize = J_PAYLOAD + 26;
/// Prepared world frame B (Transform p 33..35, q 36..39).
pub const MJ_FRAME_B: usize = J_PAYLOAD + 33;
/// Prepared center delta `centerB - centerA` (40..42).
pub const MJ_DELTA_CENTER: usize = J_PAYLOAD + 40;
/// Prepared linear spring softness (biasRate, massScale, impulseScale 43..45).
pub const MJ_LINEAR_SPRING: usize = J_PAYLOAD + 43;
/// Prepared angular spring softness (46..48).
pub const MJ_ANGULAR_SPRING: usize = J_PAYLOAD + 46;
/// Prepared angular mass = inv(invIA + invIB) (mat3 49..57).
pub const MJ_ANGULAR_MASS: usize = J_PAYLOAD + 49;

// --- parallel-joint payload -------------------------------------------------------------------
// Holds two bodies' rotation frames collinear about their local z axes (a 2-DOF angular constraint),
// driven by a soft spring capped at maxTorque; no linear constraint. Solve ignores useBias (a pure
// soft constraint) — the kernel arm mirrors that. Config, the one persistent vec2 impulse, then
// prepare's scratch (the two world joint quats, the two warm-start perp axes, the softness, and
// fixedRotation — which solve reads, unlike motor). All scratch is written unconditionally in prepare.

pub const PLJ_HERTZ: usize = J_PAYLOAD; // 62
pub const PLJ_DAMPING_RATIO: usize = J_PAYLOAD + 1;
pub const PLJ_MAX_TORQUE: usize = J_PAYLOAD + 2;
/// Persistent perpendicular (collinearity) impulse (vec2, marshaled): 3..4.
pub const PLJ_PERP_IMPULSE: usize = J_PAYLOAD + 3;
/// Prepared world joint quaternion A (quat 5..8).
pub const PLJ_QUAT_A: usize = J_PAYLOAD + 5;
/// Prepared world joint quaternion B (quat 9..12).
pub const PLJ_QUAT_B: usize = J_PAYLOAD + 9;
/// Prepared warm-start perpendicular axes (13..15, 16..18).
pub const PLJ_PERP_AXIS_X: usize = J_PAYLOAD + 13;
pub const PLJ_PERP_AXIS_Y: usize = J_PAYLOAD + 16;
/// Prepared spring softness (biasRate, massScale, impulseScale 19..21).
pub const PLJ_SOFTNESS: usize = J_PAYLOAD + 19;
/// Prepared `fixedRotation` flag (1.0 = true), read `!= 0.0`.
pub const PLJ_FIXED_ROTATION: usize = J_PAYLOAD + 22;

/// f32 stride of one joint record — the header plus the widest ported type payload. 3c grows this as it
/// ports wider joints; the column is transient (re-reserved each step), so a larger stride costs only
/// scratch bytes. Widest ported so far: spherical (64 slots); motor (58) / prismatic (56) / parallel
/// (23) are narrower.
pub const JOINT_STRIDE: usize = J_PAYLOAD + 64; // 126

// --- accessors --------------------------------------------------------------------------------

#[inline]
pub fn joint_type(col: Col<f32>, slot: usize) -> u32 {
    col.get(slot * JOINT_STRIDE + J_TYPE).to_bits()
}

#[inline]
fn read_vec3(col: Col<f32>, o: usize) -> Vec3 {
    Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2))
}

#[inline]
fn write_vec3(col: Col<f32>, o: usize, v: Vec3) {
    col.set(o, v.x);
    col.set(o + 1, v.y);
    col.set(o + 2, v.z);
}

#[inline]
fn read_quat(col: Col<f32>, o: usize) -> Quat {
    Quat {
        v: Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2)),
        s: col.get(o + 3),
    }
}

#[inline]
fn read_mat3(col: Col<f32>, o: usize) -> Mat3 {
    Mat3 {
        cx: Vec3::new(col.get(o), col.get(o + 1), col.get(o + 2)),
        cy: Vec3::new(col.get(o + 3), col.get(o + 4), col.get(o + 5)),
        cz: Vec3::new(col.get(o + 6), col.get(o + 7), col.get(o + 8)),
    }
}

/// The base joint fields `prepare`/`warm_start`/`solve` share, gathered for one slot.
pub struct JointBase {
    pub sim_index_a: u32,
    pub sim_index_b: u32,
    pub inv_mass_a: f32,
    pub inv_mass_b: f32,
    pub inv_ia: Mat3,
    pub inv_ib: Mat3,
    pub local_frame_a: Transform,
    pub local_frame_b: Transform,
}

/// The body-pose inputs `prepare` derives the anchors from (marshaled by TS; awake or static alike).
pub struct JointPose {
    pub qa: Quat,
    pub local_center_a: Vec3,
    pub center_a: Vec3,
    pub qb: Quat,
    pub local_center_b: Vec3,
    pub center_b: Vec3,
}

#[inline]
pub fn read_base(col: Col<f32>, slot: usize) -> JointBase {
    let o = slot * JOINT_STRIDE;
    JointBase {
        sim_index_a: col.get(o + J_SIM_INDEX_A).to_bits(),
        sim_index_b: col.get(o + J_SIM_INDEX_B).to_bits(),
        inv_mass_a: col.get(o + J_INV_MASS_A),
        inv_mass_b: col.get(o + J_INV_MASS_B),
        inv_ia: read_mat3(col, o + J_INV_IA),
        inv_ib: read_mat3(col, o + J_INV_IB),
        local_frame_a: Transform {
            p: read_vec3(col, o + J_LOCAL_FRAME_A),
            q: read_quat(col, o + J_LOCAL_FRAME_A + 3),
        },
        local_frame_b: Transform {
            p: read_vec3(col, o + J_LOCAL_FRAME_B),
            q: read_quat(col, o + J_LOCAL_FRAME_B + 3),
        },
    }
}

#[inline]
pub fn read_pose(col: Col<f32>, slot: usize) -> JointPose {
    let o = slot * JOINT_STRIDE;
    JointPose {
        qa: read_quat(col, o + J_QA),
        local_center_a: read_vec3(col, o + J_LOCAL_CENTER_A),
        center_a: read_vec3(col, o + J_CENTER_A),
        qb: read_quat(col, o + J_QB),
        local_center_b: read_vec3(col, o + J_LOCAL_CENTER_B),
        center_b: read_vec3(col, o + J_CENTER_B),
    }
}

#[inline]
pub fn get(col: Col<f32>, slot: usize, field: usize) -> f32 {
    col.get(slot * JOINT_STRIDE + field)
}

#[inline]
pub fn set(col: Col<f32>, slot: usize, field: usize, v: f32) {
    col.set(slot * JOINT_STRIDE + field, v);
}

#[inline]
pub fn get_vec3(col: Col<f32>, slot: usize, field: usize) -> Vec3 {
    read_vec3(col, slot * JOINT_STRIDE + field)
}

#[inline]
pub fn set_vec3(col: Col<f32>, slot: usize, field: usize, v: Vec3) {
    write_vec3(col, slot * JOINT_STRIDE + field, v);
}

#[inline]
pub fn get_quat(col: Col<f32>, slot: usize, field: usize) -> Quat {
    read_quat(col, slot * JOINT_STRIDE + field)
}

#[inline]
pub fn set_quat(col: Col<f32>, slot: usize, field: usize, q: Quat) {
    let o = slot * JOINT_STRIDE + field;
    col.set(o, q.v.x);
    col.set(o + 1, q.v.y);
    col.set(o + 2, q.v.z);
    col.set(o + 3, q.s);
}

#[inline]
pub fn get_mat3(col: Col<f32>, slot: usize, field: usize) -> Mat3 {
    read_mat3(col, slot * JOINT_STRIDE + field)
}

#[inline]
pub fn set_mat3(col: Col<f32>, slot: usize, field: usize, m: Mat3) {
    let o = slot * JOINT_STRIDE + field;
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

/// Read a prepared world joint frame (Transform: `p` at `field`, `q` at `field + 3`).
#[inline]
pub fn get_transform(col: Col<f32>, slot: usize, field: usize) -> Transform {
    Transform {
        p: get_vec3(col, slot, field),
        q: get_quat(col, slot, field + 3),
    }
}

/// Write a prepared world joint frame (Transform: `p` at `field`, `q` at `field + 3`).
#[inline]
pub fn set_transform(col: Col<f32>, slot: usize, field: usize, t: Transform) {
    set_vec3(col, slot, field, t.p);
    set_quat(col, slot, field + 3, t.q);
}
