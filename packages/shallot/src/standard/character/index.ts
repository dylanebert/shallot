import { f32, type Plugin, type State, type System, sparse } from "../../engine";
import { type Mirror, MirrorPlugin, mirror } from "../mirror";
import { Body, PhysicsPlugin, ShapeKind } from "../physics";
import { type Hull, Hulls, Physics, StepSystem } from "../physics/core";
import { jumped, moves, resetDrive, states } from "./drive";
import { type CharState, type SweepBody, sweepCharacter } from "./sweep";

// Character — the kinematic capsule controller (Phase 6.4), the base a higher-level controller (the
// first-person Player) composes. The Character entity IS a capsule Body (mass <= 0) whose pose the CPU
// SWEEP owns: each fixed tick `CharacterSweepSystem` runs the collide-and-slide (`sweep.ts`, the f32-tier
// twin of the f64 oracle `tests/avbd/character.ts`) on the CPU, BEFORE the GPU solve, then uploads the
// swept pose into the GPU `bodies` buffer as a kinematic body (`setKinematic`). So the player's input →
// pose → camera is a same-frame CPU path with no GPU readback, and the GPU dynamics collide against the
// CURRENT-tick player. The coupling is one-way: the CPU writes the player's fresh pose (the GPU reads it to
// push dynamics + carry riders), and the CPU sweep reads dynamic poses one tick stale off a `Mirror` of the
// `bodies` firehose (a frame-old dynamic pose is fine for the sweep; the static collision world doesn't move).
//
// This module is the authoring + driving surface: the tuning component, the per-tick sweep system, the
// eid-keyed drive (move/jump) + the swept-pose / grounded readback a follower (a camera) reads from the
// CPU controller state. The CPU sweep is the SOLE runtime controller — there is no GPU character pass (it
// was deleted with the camera-follow rewire); the f64 oracle `tests/avbd/character.ts` is the spec, and the
// sweep is validated against it by `character-sweep.oracle.ts`. `Player` composes this controller (look +
// a camera) on top, snapshotting `pose` off this CPU state `after: [CharacterSweepSystem]`.

const DEG = Math.PI / 180;

/**
 * a kinematic character: a capsule {@link Body} (`mass <= 0`) swept against the scene's bodies each fixed
 * step (collide-and-slide on the CPU). Authors the walkable slope, the jump launch speed, and a per-character
 * gravity; the controller sweeps every `[Character, Body]`, drives it via {@link move} / {@link jump}, and
 * reads {@link grounded} / {@link pose} back. The first-person {@link Player} composes this.
 *
 * @example
 * ```
 * const body = state.create();
 * state.add(body, Body); state.add(body, Character);
 * Body.shape.set(body, ShapeKind.Capsule);
 * Body.halfExtents.set(body, 0, 0.5, 0, 0.3); Body.mass.set(body, 0);
 * Character.jumpSpeed.set(body, 5);   // 0 = no jump
 * ```
 */
export const Character = {
    /** steepest walkable slope in degrees; a contact flatter than this grounds the character, steeper it slides */
    maxSlope: sparse(f32),
    /** the launch velocity a buffered + grounded {@link jump} sets. 0 disables jumping */
    jumpSpeed: sparse(f32),
    /** per-character gravity (negative, snappier than the world for a player). 0 = the configured world gravity */
    gravity: sparse(f32),
};

// a Mirror of the GPU `bodies` firehose — the sweep reads dynamic (and moving-platform) candidate poses +
// velocities from its snapshot (one fixed tick stale). The character's OWN pose comes from its CharState
// (current, never the Mirror). Lazily allocated once Physics.step exists (after every warm + Mirror.reset).
let bodyMirror: Mirror | null = null;

// last-registered signature — re-sync `states` ONLY on a change to the authored set / tuning (the GPU
// register's FNV discipline). FNV_BASIS = the empty set, so a character-free scene never syncs.
const FNV_BASIS = 2166136261;
const fold = (h: number, v: number): number => Math.imul(h ^ (v >>> 0), 16777619);
const _sigF32 = new Float32Array(1);
const _sigU32 = new Uint32Array(_sigF32.buffer);
const sigBits = (x: number): number => {
    _sigF32[0] = x;
    return _sigU32[0];
};
let charSig = FNV_BASIS;

function signature(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query([Character, Body])) {
        h = fold(h, eid);
        h = fold(h, sigBits(Character.maxSlope.get(eid)));
        h = fold(h, sigBits(Character.jumpSpeed.get(eid)));
        h = fold(h, sigBits(Character.gravity.get(eid)));
    }
    return h;
}

// build a fresh controller state from a character's authored Body — the spawn pose + capsule geometry + the
// walkable-slope cutoff. Velocity / grounded / jump timers start cleared (a dropped capsule falls to rest).
function buildState(eid: number): CharState {
    return {
        pos: [Body.pos.x.get(eid), Body.pos.y.get(eid), Body.pos.z.get(eid)],
        quat: [
            Body.quat.x.get(eid),
            Body.quat.y.get(eid),
            Body.quat.z.get(eid),
            Body.quat.w.get(eid),
        ],
        half: Body.halfExtents.y.get(eid),
        radius: Body.halfExtents.w.get(eid),
        maxSlopeCos: Math.cos(Character.maxSlope.get(eid) * DEG),
        jumpSpeed: Character.jumpSpeed.get(eid),
        vel: [0, 0, 0],
        realizedVel: [0, 0, 0],
        grounded: false,
        groundNormal: [0, 0, 0],
        coyote: 0,
        buffer: 0,
    };
}

// re-sync `states` to the authored `[Character, Body]` set on a signature change. A new character builds a
// fresh state; an existing one KEEPS its live pose + motion (the controller owns the pose, like the GPU char
// pass — a sibling spawning must not teleport a walking character back to its spawn) and only picks up a
// tuning edit; a removed one is dropped. Edit-time pose edits land at play start (a fresh play State syncs
// from the edited Body slab — `states` is empty then).
function syncStates(state: State): void {
    const sig = signature(state);
    if (sig === charSig) return;
    charSig = sig;
    const seen = new Set<number>();
    for (const eid of state.query([Character, Body])) {
        seen.add(eid);
        const st = states.get(eid);
        if (st) {
            st.maxSlopeCos = Math.cos(Character.maxSlope.get(eid) * DEG);
            st.jumpSpeed = Character.jumpSpeed.get(eid);
            st.half = Body.halfExtents.y.get(eid);
            st.radius = Body.halfExtents.w.get(eid);
        } else {
            states.set(eid, buildState(eid));
        }
    }
    for (const eid of [...states.keys()]) {
        if (!seen.has(eid)) {
            states.delete(eid);
            moves.delete(eid);
            jumped.delete(eid);
        }
    }
}

// reused candidate scratch — the bodies are split into static (mass <= 0: walls / ground / platforms / other
// characters — the carry reads their velocity) and dynamic (mass > 0 — shoved by the push) sets each tick.
// A growing pool of SweepBody objects avoids per-tick allocation as the scan walks every Body.
const _pool: SweepBody[] = [];
const _statics: SweepBody[] = [];
const _push: SweepBody[] = [];
const _pushEids: number[] = [];
const _pushVel0: number[] = []; // pre-sweep dynamic velocities, to detect which the push actually shoved

function poolBody(i: number): SweepBody {
    let b = _pool[i];
    if (!b) {
        b = {
            shape: 0,
            pos: [0, 0, 0],
            quat: [0, 0, 0, 1],
            half: [0, 0, 0],
            radius: 0,
            vel: [0, 0, 0],
        };
        _pool[i] = b;
    }
    return b;
}

const hullById = (id: number): Hull | undefined => Hulls.get(Hulls.name(id) ?? "");

// the bodies SoA columns the sweep reads off the Mirror snapshot (must match step.ts B_POS / B_QUAT / B_VELL)
const B_POS = 0;
const B_QUAT = 1;
const B_VELL = 6;

// one character's sweep: gather candidates (geometry from the authored Body slab, live pose + velocity from
// the bodies Mirror snapshot — the static world is unchanged by the one-tick lag, a dynamic / platform pose
// is fine one tick old), run the collide-and-slide, upload the swept pose as a kinematic body, and apply the
// full-speed push to shoved dynamics (variant A — the full-CPU apply: the swept body's stale-velocity +
// shove is written straight to the GPU `B_VELL`, no GPU character work; see the push-apply A/B in the gym).
function sweepEid(
    eid: number,
    st: CharState,
    state: State,
    view: Float32Array | null,
    cap: number,
): void {
    const step = Physics.step;
    if (!step) return;
    _statics.length = 0;
    _push.length = 0;
    _pushEids.length = 0;
    let pi = 0;
    for (const b of state.query([Body])) {
        if (b === eid) continue; // the character never collides against itself (it IS `start`)
        const shape = Body.shape.get(b);
        const sb = poolBody(pi++);
        sb.shape = shape;
        sb.half[0] = Body.halfExtents.x.get(b);
        sb.half[1] = Body.halfExtents.y.get(b);
        sb.half[2] = Body.halfExtents.z.get(b);
        const hw = Body.halfExtents.w.get(b); // a rounding radius (sphere/capsule) OR a hull id (shape 3)
        if (shape === ShapeKind.Hull) {
            sb.radius = 0;
            sb.hull = hullById(hw);
        } else {
            sb.radius = hw;
            sb.hull = undefined;
        }
        if (view) {
            const p = (B_POS * cap + b) * 4;
            const q = (B_QUAT * cap + b) * 4;
            const v = (B_VELL * cap + b) * 4;
            sb.pos[0] = view[p];
            sb.pos[1] = view[p + 1];
            sb.pos[2] = view[p + 2];
            sb.quat[0] = view[q];
            sb.quat[1] = view[q + 1];
            sb.quat[2] = view[q + 2];
            sb.quat[3] = view[q + 3];
            sb.vel[0] = view[v];
            sb.vel[1] = view[v + 1];
            sb.vel[2] = view[v + 2];
        } else {
            // cold start (no snapshot yet): the authored spawn pose, velocity 0 — correct for the static
            // collision world the character needs from frame 1, and a freshly-spawned dynamic hasn't moved.
            sb.pos[0] = Body.pos.x.get(b);
            sb.pos[1] = Body.pos.y.get(b);
            sb.pos[2] = Body.pos.z.get(b);
            sb.quat[0] = Body.quat.x.get(b);
            sb.quat[1] = Body.quat.y.get(b);
            sb.quat[2] = Body.quat.z.get(b);
            sb.quat[3] = Body.quat.w.get(b);
            sb.vel[0] = 0;
            sb.vel[1] = 0;
            sb.vel[2] = 0;
        }
        if (Body.mass.get(b) > 0) {
            _push.push(sb);
            _pushEids.push(b);
        } else {
            _statics.push(sb);
        }
    }

    const m = moves.get(eid);
    const input: [number, number, number] = [m ? m[0] : 0, 0, m ? m[1] : 0];
    const g = Character.gravity.get(eid);
    const gravity = g !== 0 ? g : step.gravity;

    // snapshot the dynamics' velocities so we can tell which the sweep actually shoved (the push loop only
    // mutates a touched dynamic's `vel`) — a no-op velocity rewrite would wake every nearby resting body.
    for (let i = 0; i < _push.length; i++) {
        const v = _push[i].vel;
        _pushVel0[3 * i] = v[0];
        _pushVel0[3 * i + 1] = v[1];
        _pushVel0[3 * i + 2] = v[2];
    }

    sweepCharacter(st, input, _statics, gravity, step.dt, jumped.has(eid), _push);

    // kinematic upload — the swept pose, with the realized velocity (snap excluded) as the explicit B_VELL so
    // the carry-of-riders + broadphase pad read the swept motion, not the cosmetic ground snap.
    step.setKinematic(eid, st.pos, st.quat, false, st.realizedVel);

    // full-speed push (variant A): write each shoved dynamic's new velocity straight to the GPU. setVelocity
    // wakes the body, so apply it only to the ones the sweep changed.
    for (let i = 0; i < _push.length; i++) {
        const v = _push[i].vel;
        if (
            v[0] !== _pushVel0[3 * i] ||
            v[1] !== _pushVel0[3 * i + 1] ||
            v[2] !== _pushVel0[3 * i + 2]
        ) {
            step.setVelocity(_pushEids[i], v[0], v[1], v[2]);
        }
    }
}

// Fixed group — the deterministic dt the sweep integrates gravity over. Plays only (no `mode: "always"`):
// edit mode doesn't simulate, so a play start re-syncs `states` from the current Body slab.
/**
 * the kinematic-character sweep: runs collide-and-slide for every `[Character, Body]` each fixed step,
 * before the physics solve, uploading the swept pose as a kinematic body. Exported as an ordering anchor: a
 * follower that reads the swept pose (a camera, an attached prop) declares `after: [CharacterSweepSystem]` so
 * it sees this tick's pose, not last tick's.
 */
export const CharacterSweepSystem: System = {
    name: "character",
    group: "fixed",
    before: [StepSystem],
    update(state: State) {
        const step = Physics.step;
        if (!step) return;
        syncStates(state);
        if (states.size === 0) return;
        if (!bodyMirror) bodyMirror = mirror(step.bodies);
        const snap = bodyMirror.snapshot;
        const view = snap ? new Float32Array(snap.bytes) : null;
        const cap = step.eidCap;
        for (const [eid, st] of states) sweepEid(eid, st, state, view, cap);
        jumped.clear();
    },
};

/** kinematic-character plugin: registers every `[Character, Body]` and sweeps it (collide-and-slide) each
 *  fixed step, before the physics solve. Depends on {@link PhysicsPlugin} (the collision world it sweeps
 *  against) and mirror (the frame-stale dynamic poses it reads). Add it, then drive characters with the
 *  {@link move} / {@link jump} surface, or add {@link Player} for a ready first-person controller. */
export const CharacterPlugin: Plugin = {
    name: "Character",
    components: { Character },
    systems: [CharacterSweepSystem],
    dependencies: [PhysicsPlugin, MirrorPlugin],
    traits: {
        Character: {
            requires: [Body],
            defaults: () => ({
                maxSlope: 45,
                jumpSpeed: 0, // no jump
                gravity: 0, // = the configured world gravity
            }),
        },
    },
    dispose() {
        bodyMirror?.dispose();
        bodyMirror = null;
        resetDrive();
        charSig = FNV_BASIS;
    },
};
