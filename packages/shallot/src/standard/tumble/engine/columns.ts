// Typed-array views over the kernel's shared solver columns (kernel/src/arena.rs). `reserveColumns`
// lays out the columns in wasm linear memory for one step's counts and returns fresh views. Because
// `reserve` may `memory.grow` (which detaches every existing view), the views are re-derived on every
// call — the cost is a handful of typed-array constructions, no copy, and it sidesteps stale-buffer
// bugs entirely.
//
// The strides and column order MIRROR the Rust ABI (kernel/src/body.rs, kernel/src/contact.rs). The
// wasm layout is the contract; a mismatch here silently corrupts the solve, so keep them in lockstep.

import { kernel } from "./kernel";
import type { Mat3 } from "./math";

/** Write a Mat3 into `col` at `o` in the kernel's row order (cx, cy, cz) — the sim column's inertia
 * layout (read_sim, kernel/src/body.rs). Shared by the body-store marshal and finalize's raw write. */
export function writeMat3(col: Float32Array, o: number, m: Mat3): void {
    col[o] = m.cx.x;
    col[o + 1] = m.cx.y;
    col[o + 2] = m.cx.z;
    col[o + 3] = m.cy.x;
    col[o + 4] = m.cy.y;
    col[o + 5] = m.cy.z;
    col[o + 6] = m.cz.x;
    col[o + 7] = m.cz.y;
    col[o + 8] = m.cz.z;
}

// Body columns (body.rs + bodies.rs). All resident in the persistent body region (4a) — the awake
// `BodySim`/`BodyState` are offset-backed views over them (bodycolumns.ts), so no per-step marshal.
export const STATE_STRIDE = 16;
/** Live f32 fields in a state record; slots STATE_LIVE..STATE_STRIDE are alignment padding. */
export const STATE_LIVE = 13;
export const SIM_STRIDE = 32;
export const FIN_STRIDE = 12;
export const FIN_OUT_STRIDE = 2;
/** The `BodySim` fields the per-step `sim`/`fin` columns omit (kernel never gathers them), held in a
 * second resident column: rotation0(4) center0(3) minExtent(1) maxAngularVelocity(1) bodyId(1)
 * flags(1) headShapeId(1) = 12 (body.rs `SIM2_STRIDE`). Mirrors the Rust ABI. */
export const SIM2_STRIDE = 12;
export const SIM2_LIVE = 12;
// sim2 field offsets.
export const S2_ROTATION0 = 0; // v3 + s
export const S2_CENTER0 = 4;
export const S2_MIN_EXTENT = 7;
export const S2_MAX_ANGULAR_VELOCITY = 8;
export const S2_BODY_ID = 9;
export const S2_FLAGS = 10;
/** Head of the body's shape list — the lane the kernel finalize refit walks the shape column from
 * (shapes.rs). u32: `NULL_INDEX` (-1) wraps to 0xFFFFFFFF, the kernel's `NULL_SHAPE`. */
export const S2_HEAD_SHAPE = 11;

// Per scalar solver-record slot (manifold_abi.rs): contactId, transient mc base, transient mcp base.
// The narrowphase → solver map for the scalar (mesh/overflow) path; the persistent directory + pool
// it points into live in the manifold region (manifoldstore.ts).
export const SLOT_STRIDE = 3;

// Wide (convex) transient constraint columns (contact_wide.rs). Only the meta column is written
// TS-side (the lane → contactId map); the record + index columns are kernel-internal.
export const WIDE_STRIDE = 404;
export const WIDE_META_STRIDE = 5;
export const WIDE_IDX_STRIDE = 8;

// Per-active-color span (contact_wide.rs / arena.rs): wideStart, wideCount, meshStart, meshCount,
// jointStart, jointCount. The batched (jointless) color loop reads only the first four; the staged
// solve reads the joint pair too (joints-in-kernel).
export const COLOR_SPAN_STRIDE = 6;
export const CS_JOINT_START = 4;
export const CS_JOINT_COUNT = 5;

// Joint record (kernel/src/joint_abi.rs). One flat f32 record per joint slot: a common header (the
// two resident state indices via u32 bits, invMass/invInertia, the pose fields prepare derives anchors
// from, the local frames, the base constraint frequency + softness) then a per-type payload (distance's
// config, its persistent impulses, and prepare's scratch). Mirror of `joint_abi.rs` — keep in sync.
export const JOINT_STRIDE = 126;
export const J_TYPE = 0;
export const J_SIM_INDEX_A = 1;
export const J_SIM_INDEX_B = 2;
export const J_INV_MASS_A = 3;
export const J_INV_MASS_B = 4;
export const J_INV_IA = 5; // mat3 5..13
export const J_INV_IB = 14; // mat3 14..22
export const J_QA = 23; // quat 23..26
export const J_LOCAL_CENTER_A = 27; // 27..29
export const J_CENTER_A = 30; // 30..32
export const J_QB = 33; // quat 33..36
export const J_LOCAL_CENTER_B = 37; // 37..39
export const J_CENTER_B = 40; // 40..42
export const J_LOCAL_FRAME_A = 43; // Transform p 43..45 q 46..49
export const J_LOCAL_FRAME_B = 50; // Transform p 50..52 q 53..56
export const J_CONSTRAINT_HERTZ = 57;
export const J_CONSTRAINT_DAMPING = 58;
export const J_CONSTRAINT_SOFTNESS = 59; // 59..61
const J_PAYLOAD = 62;
export const DJ_LENGTH = J_PAYLOAD;
export const DJ_HERTZ = J_PAYLOAD + 1;
export const DJ_DAMPING_RATIO = J_PAYLOAD + 2;
export const DJ_LOWER_SPRING_FORCE = J_PAYLOAD + 3;
export const DJ_UPPER_SPRING_FORCE = J_PAYLOAD + 4;
export const DJ_MIN_LENGTH = J_PAYLOAD + 5;
export const DJ_MAX_LENGTH = J_PAYLOAD + 6;
export const DJ_MAX_MOTOR_FORCE = J_PAYLOAD + 7;
export const DJ_MOTOR_SPEED = J_PAYLOAD + 8;
export const DJ_ENABLE = J_PAYLOAD + 9;
export const DJ_IMPULSE = J_PAYLOAD + 10;
export const DJ_LOWER_IMPULSE = J_PAYLOAD + 11;
export const DJ_UPPER_IMPULSE = J_PAYLOAD + 12;
export const DJ_MOTOR_IMPULSE = J_PAYLOAD + 13;
export const DJ_ENABLE_SPRING = 0x1;
export const DJ_ENABLE_LIMIT = 0x2;
export const DJ_ENABLE_MOTOR = 0x4;

// Weld-joint payload (joint_abi.rs weld section). Config, the two persistent impulses, then prepare's
// scratch (frames, angular mass, resolved softnesses, fixedRotation). Marshal writes config + impulses.
export const WJ_LINEAR_HERTZ = J_PAYLOAD;
export const WJ_LINEAR_DAMPING_RATIO = J_PAYLOAD + 1;
export const WJ_ANGULAR_HERTZ = J_PAYLOAD + 2;
export const WJ_ANGULAR_DAMPING_RATIO = J_PAYLOAD + 3;
export const WJ_LINEAR_IMPULSE = J_PAYLOAD + 4; // vec3
export const WJ_ANGULAR_IMPULSE = J_PAYLOAD + 7; // vec3

// Revolute-joint payload (joint_abi.rs revolute section). Config, the persistent impulses (linear vec3 +
// perp vec2 + four scalar), then prepare's scratch. Marshal writes config + impulses.
export const RJ_HERTZ = J_PAYLOAD;
export const RJ_DAMPING_RATIO = J_PAYLOAD + 1;
export const RJ_MAX_MOTOR_TORQUE = J_PAYLOAD + 2;
export const RJ_MOTOR_SPEED = J_PAYLOAD + 3;
export const RJ_TARGET_ANGLE = J_PAYLOAD + 4;
export const RJ_LOWER_ANGLE = J_PAYLOAD + 5;
export const RJ_UPPER_ANGLE = J_PAYLOAD + 6;
export const RJ_ENABLE = J_PAYLOAD + 7;
export const RJ_LINEAR_IMPULSE = J_PAYLOAD + 8; // vec3
export const RJ_PERP_IMPULSE = J_PAYLOAD + 11; // vec2
export const RJ_SPRING_IMPULSE = J_PAYLOAD + 13;
export const RJ_MOTOR_IMPULSE = J_PAYLOAD + 14;
export const RJ_LOWER_IMPULSE = J_PAYLOAD + 15;
export const RJ_UPPER_IMPULSE = J_PAYLOAD + 16;
export const RJ_ENABLE_SPRING = 0x1;
export const RJ_ENABLE_MOTOR = 0x2;
export const RJ_ENABLE_LIMIT = 0x4;

// Spherical-joint payload (joint_abi.rs spherical section). Config, the persistent impulses (three vec3 +
// three scalar), then prepare's scratch. Marshal writes config + impulses.
export const SJ_HERTZ = J_PAYLOAD;
export const SJ_DAMPING_RATIO = J_PAYLOAD + 1;
export const SJ_MAX_MOTOR_TORQUE = J_PAYLOAD + 2;
export const SJ_MOTOR_VELOCITY = J_PAYLOAD + 3; // vec3
export const SJ_LOWER_TWIST_ANGLE = J_PAYLOAD + 6;
export const SJ_UPPER_TWIST_ANGLE = J_PAYLOAD + 7;
export const SJ_CONE_ANGLE = J_PAYLOAD + 8;
export const SJ_TARGET_ROTATION = J_PAYLOAD + 9; // quat
export const SJ_ENABLE = J_PAYLOAD + 13;
export const SJ_LINEAR_IMPULSE = J_PAYLOAD + 14; // vec3
export const SJ_SPRING_IMPULSE = J_PAYLOAD + 17; // vec3
export const SJ_MOTOR_IMPULSE = J_PAYLOAD + 20; // vec3
export const SJ_LOWER_TWIST_IMPULSE = J_PAYLOAD + 23;
export const SJ_UPPER_TWIST_IMPULSE = J_PAYLOAD + 24;
export const SJ_SWING_IMPULSE = J_PAYLOAD + 25;
export const SJ_ENABLE_SPRING = 0x1;
export const SJ_ENABLE_MOTOR = 0x2;
export const SJ_ENABLE_CONE_LIMIT = 0x4;
export const SJ_ENABLE_TWIST_LIMIT = 0x8;

// Prismatic-joint payload (joint_abi.rs prismatic section). Config, the persistent impulses (perp vec2 +
// angular vec3 + four scalar), then prepare's scratch. Marshal writes config + impulses.
export const PJ_HERTZ = J_PAYLOAD;
export const PJ_DAMPING_RATIO = J_PAYLOAD + 1;
export const PJ_MAX_MOTOR_FORCE = J_PAYLOAD + 2;
export const PJ_MOTOR_SPEED = J_PAYLOAD + 3;
export const PJ_TARGET_TRANSLATION = J_PAYLOAD + 4;
export const PJ_LOWER_TRANSLATION = J_PAYLOAD + 5;
export const PJ_UPPER_TRANSLATION = J_PAYLOAD + 6;
export const PJ_ENABLE = J_PAYLOAD + 7;
export const PJ_PERP_IMPULSE = J_PAYLOAD + 8; // vec2
export const PJ_ANGULAR_IMPULSE = J_PAYLOAD + 10; // vec3
export const PJ_SPRING_IMPULSE = J_PAYLOAD + 13;
export const PJ_MOTOR_IMPULSE = J_PAYLOAD + 14;
export const PJ_LOWER_IMPULSE = J_PAYLOAD + 15;
export const PJ_UPPER_IMPULSE = J_PAYLOAD + 16;
export const PJ_ENABLE_SPRING = 0x1;
export const PJ_ENABLE_MOTOR = 0x2;
export const PJ_ENABLE_LIMIT = 0x4;

// Wheel-joint payload (joint_abi.rs wheel section). Config, the persistent impulses (two vec2 + seven
// scalar), then prepare's scratch. Marshal writes config + impulses.
export const WHJ_MAX_SPIN_TORQUE = J_PAYLOAD;
export const WHJ_SPIN_SPEED = J_PAYLOAD + 1;
export const WHJ_LOWER_SUSPENSION_LIMIT = J_PAYLOAD + 2;
export const WHJ_UPPER_SUSPENSION_LIMIT = J_PAYLOAD + 3;
export const WHJ_SUSPENSION_HERTZ = J_PAYLOAD + 4;
export const WHJ_SUSPENSION_DAMPING_RATIO = J_PAYLOAD + 5;
export const WHJ_LOWER_STEERING_LIMIT = J_PAYLOAD + 6;
export const WHJ_UPPER_STEERING_LIMIT = J_PAYLOAD + 7;
export const WHJ_TARGET_STEERING_ANGLE = J_PAYLOAD + 8;
export const WHJ_MAX_STEERING_TORQUE = J_PAYLOAD + 9;
export const WHJ_STEERING_HERTZ = J_PAYLOAD + 10;
export const WHJ_STEERING_DAMPING_RATIO = J_PAYLOAD + 11;
export const WHJ_ENABLE = J_PAYLOAD + 12;
export const WHJ_LINEAR_IMPULSE = J_PAYLOAD + 13; // vec2
export const WHJ_ANGULAR_IMPULSE = J_PAYLOAD + 15; // vec2
export const WHJ_SPIN_IMPULSE = J_PAYLOAD + 17;
export const WHJ_SUSPENSION_SPRING_IMPULSE = J_PAYLOAD + 18;
export const WHJ_LOWER_SUSPENSION_IMPULSE = J_PAYLOAD + 19;
export const WHJ_UPPER_SUSPENSION_IMPULSE = J_PAYLOAD + 20;
export const WHJ_STEERING_SPRING_IMPULSE = J_PAYLOAD + 21;
export const WHJ_LOWER_STEERING_IMPULSE = J_PAYLOAD + 22;
export const WHJ_UPPER_STEERING_IMPULSE = J_PAYLOAD + 23;
export const WHJ_ENABLE_SPIN_MOTOR = 0x1;
export const WHJ_ENABLE_SUSPENSION_SPRING = 0x2;
export const WHJ_ENABLE_SUSPENSION_LIMIT = 0x4;
export const WHJ_ENABLE_STEERING = 0x8;
export const WHJ_ENABLE_STEERING_LIMIT = 0x10;

// Motor-joint payload (joint_abi.rs motor section). Config, the four persistent vec3 impulses, then
// prepare's scratch. Marshal writes config + impulses (no enable bitfield — each branch keys on max*>0).
export const MJ_LINEAR_VELOCITY = J_PAYLOAD; // vec3
export const MJ_ANGULAR_VELOCITY = J_PAYLOAD + 3; // vec3
export const MJ_MAX_VELOCITY_FORCE = J_PAYLOAD + 6;
export const MJ_MAX_VELOCITY_TORQUE = J_PAYLOAD + 7;
export const MJ_LINEAR_HERTZ = J_PAYLOAD + 8;
export const MJ_LINEAR_DAMPING_RATIO = J_PAYLOAD + 9;
export const MJ_ANGULAR_HERTZ = J_PAYLOAD + 10;
export const MJ_ANGULAR_DAMPING_RATIO = J_PAYLOAD + 11;
export const MJ_MAX_SPRING_FORCE = J_PAYLOAD + 12;
export const MJ_MAX_SPRING_TORQUE = J_PAYLOAD + 13;
export const MJ_LINEAR_VELOCITY_IMPULSE = J_PAYLOAD + 14; // vec3
export const MJ_ANGULAR_VELOCITY_IMPULSE = J_PAYLOAD + 17; // vec3
export const MJ_LINEAR_SPRING_IMPULSE = J_PAYLOAD + 20; // vec3
export const MJ_ANGULAR_SPRING_IMPULSE = J_PAYLOAD + 23; // vec3

// Parallel-joint payload (joint_abi.rs parallel section). Config, the one persistent vec2 impulse, then
// prepare's scratch. Marshal writes config + impulse.
export const PLJ_HERTZ = J_PAYLOAD;
export const PLJ_DAMPING_RATIO = J_PAYLOAD + 1;
export const PLJ_MAX_TORQUE = J_PAYLOAD + 2;
export const PLJ_PERP_IMPULSE = J_PAYLOAD + 3; // vec2

// Convex narrowphase dispatch record (arena.rs `DISPATCH_STRIDE`). Float slots are written through an
// f32 view, contactId/types/hull geoIndex through a u32 view over the same column. Geom slots hold a
// sphere (center3 + radius), a capsule (center1_3 + center2_3 + radius), or a hull (geoIndex at slot 0).
export const DISPATCH_STRIDE = 31;
export const D_CONTACT = 0;
export const D_TYPE_A = 1;
export const D_TYPE_B = 2;
export const D_XF_A = 3; // p3 + q4
export const D_XF_B = 10; // p3 + q4
export const D_GEOM_A = 17; // ≤7 slots
export const D_GEOM_B = 24; // ≤7 slots

// Contact-recycle input record (arena.rs `RECYCLE_STRIDE`, 4b.3c). All u32: contactId, the two bodies'
// awake localIndices (resident-column records), the two shapeIds (fat-AABB column records), and a bits
// word (bit0 eligible, bit1 wasTouching). The kernel reads the bodies' transforms/centers/extents and
// the cached pose from the resident columns; only these indices + bits cross per contact.
export const RECYCLE_STRIDE = 6;
export const R_CONTACT = 0;
export const R_LOCAL_A = 1;
export const R_LOCAL_B = 2;
export const R_SHAPE_A = 3;
export const R_SHAPE_B = 4;
export const R_BITS = 5;
/** bit0: the contact may recycle this step (recycleDistance>0 && relativeTransformValid && recycleFlag). */
export const R_ELIGIBLE = 1;
/** bit1: the contact was touching at step entry (selects the recycle tolerance). */
export const R_WAS_TOUCHING = 2;

// LAYOUT header indices (arena.rs), in memory order. STATE/FLAGS/SIM/FIN are resident (their LAYOUT
// entries point into the body region — bodycolumns.ts), consumed through the `BodySim`/`BodyState`
// views, so the per-step reservation never derives a scratch view for them. FIN_OUT is resident too
// but transient (finalize's two decision scalars), read TS-side per step through `finOut` below.
const FIN_OUT = 4;
const SLOT_SCALAR = 5;
const WIDE_META = 12;
const COLOR_SPAN = 14;
const JOINT = 15;
const N_COLS = 16;

/** The column views the TS side reads or writes. The body columns (`state`/`flags`/`sim`/`fin`) are
 * resident (4a) — held across steps in the body region and viewed through the `BodySim`/`BodyState`
 * views (bodycolumns.ts), not here. The transient constraint columns (cc/mc/mcp) and the persistent
 * directory/pool are kernel-internal. */
export type Columns = {
    /** Finalize's two per-body decision scalars (sleepVelocity, maxMotion). Resident (in the body
     * region) but transient — recomputed each step, read once TS-side in finalize. */
    finOut: Float32Array;
    /** Per scalar solver-record slot (contactId, transient mc base, transient mcp base). TS writes it
     * in graph-color order; the kernel scalar `prepare`/`store` gather each contact through it. */
    slotScalar: Uint32Array;
    /** Per-wide-record lane map: laneContact[4] (contactId per lane) + laneCount. TS writes convex
     * contacts here in color order; the kernel wide `prepare`/`store` gather through it. */
    wideMeta: Uint32Array;
    /** Per-active-color spans (wide/mesh/joint start+count) the batched color shims + staged solve
     * loop over. TS writes them from the layout. */
    colorSpan: Uint32Array;
    /** Flat joint records (`JOINT_STRIDE` f32 per slot, jointcolumns.ts). Colored joints first
     * (per-color concatenated), then the overflow joints. Empty on the jointless path. */
    joint: Float32Array;
};

/**
 * Reserve the solver columns for one step's counts and return fresh typed-array views over them.
 * Call once per step, before driving the kernel phases; the returned views are valid until the next
 * `reserveColumns` (or any other call that can grow memory).
 */
export function reserveColumns(
    body: number,
    contact: number,
    manifold: number,
    point: number,
    wide: number,
    color: number,
    joint = 0,
): Columns {
    const k = kernel();
    k.reserve(body, contact, manifold, point, wide, color, joint);
    const buf = k.memory.buffer;
    const layout = new Uint32Array(buf, k.layoutPtr(), N_COLS);
    return {
        finOut: new Float32Array(buf, layout[FIN_OUT], body * FIN_OUT_STRIDE),
        slotScalar: new Uint32Array(buf, layout[SLOT_SCALAR], contact * SLOT_STRIDE),
        wideMeta: new Uint32Array(buf, layout[WIDE_META], wide * WIDE_META_STRIDE),
        colorSpan: new Uint32Array(buf, layout[COLOR_SPAN], color * COLOR_SPAN_STRIDE),
        joint: new Float32Array(buf, layout[JOINT], joint * JOINT_STRIDE),
    };
}
