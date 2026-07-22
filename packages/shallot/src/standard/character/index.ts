import { f32, type Plugin, type State, type System, sparse } from "../../engine";
import { Body, Physics, ShapeKind } from "../physics";
import { type Hull, Hulls, type PhysicsBackend, StepSystem } from "../physics/core";
import { jumped, moves, resetDrive, states } from "./drive";
import { type CharState, type SweepBody, sweepCharacter } from "./sweep";

// Character — the kinematic capsule controller (Phase 6.4), the base a higher-level controller (the
// first-person Player) composes. The Character entity IS a capsule Body (mass <= 0) whose pose the CPU
// SWEEP owns: each fixed tick `CharacterSweepSystem` runs the collide-and-slide (`sweep.ts`, the f32-tier
// twin of the f64 oracle `tests/avbd/character.ts`) on the CPU, BEFORE the physics solve, then uploads the
// swept pose as a kinematic body (`Physics.backend.setKinematic`). So the player's input → pose → camera is
// a same-frame CPU path with no GPU readback, and the backend's dynamics collide against the CURRENT-tick
// player. The coupling is one-way: the CPU writes the player's fresh pose (the backend reads it to push
// dynamics + carry riders), and the CPU sweep reads every other body's live pose through the installed
// backend's pose-read seam (`Physics.backend.readBody`, up to one fixed tick stale for a GPU backend — a
// frame-old dynamic pose is fine for the sweep; the static collision world doesn't move).
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
// the create-stamp each `states` entry was built at (ecs.md "An eid is a borrow"). A same-update
// destroy+create recycling a character's eid with identical tuning hashes to the SAME signature, so folding
// the stamp into the signature is what makes the realias visible; the per-eid compare in `syncStates` then
// rebuilds the controller state (a stale pose/velocity kept across the recycle is the bug this closes).
const stamps = new Map<number, number>();

function signature(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query([Character, Body])) {
        h = fold(h, eid);
        h = fold(h, state.stamp(eid));
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
        const stamp = state.stamp(eid);
        const st = states.get(eid);
        if (st && stamps.get(eid) === stamp) {
            st.maxSlopeCos = Math.cos(Character.maxSlope.get(eid) * DEG);
            st.jumpSpeed = Character.jumpSpeed.get(eid);
            st.half = Body.halfExtents.y.get(eid);
            st.radius = Body.halfExtents.w.get(eid);
        } else {
            if (st) {
                // realias: drive input keyed to the destroyed owner is stale. A fresh spawn keeps
                // input queued before its first sync.
                moves.delete(eid);
                jumped.delete(eid);
            }
            states.set(eid, buildState(eid));
            stamps.set(eid, stamp);
        }
    }
    for (const eid of [...states.keys()]) {
        if (!seen.has(eid)) {
            states.delete(eid);
            moves.delete(eid);
            jumped.delete(eid);
            stamps.delete(eid);
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

// one character's sweep: gather candidates (geometry from the authored Body slab, live pose + velocity
// through the installed backend's pose-read seam — the static world is unchanged by the possible one-tick
// lag, a dynamic / platform pose is fine one tick old), run the collide-and-slide, upload the swept pose as
// a kinematic body, and apply the full-speed push to shoved dynamics (variant A — the full-CPU apply: the
// swept body's stale-velocity + shove is written straight through `setVelocity`, no GPU character work; see
// the push-apply A/B in the gym).
function sweepEid(eid: number, st: CharState, state: State, backend: PhysicsBackend): void {
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
        const live = backend.readBody(b);
        if (live) {
            sb.pos[0] = live.pos[0];
            sb.pos[1] = live.pos[1];
            sb.pos[2] = live.pos[2];
            sb.quat[0] = live.quat[0];
            sb.quat[1] = live.quat[1];
            sb.quat[2] = live.quat[2];
            sb.quat[3] = live.quat[3];
            sb.vel[0] = live.vel[0];
            sb.vel[1] = live.vel[1];
            sb.vel[2] = live.vel[2];
        } else {
            // cold start (no live pose yet): the authored spawn pose, velocity 0 — correct for the static
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
    const gravity = g !== 0 ? g : backend.gravity;

    // snapshot the dynamics' velocities so we can tell which the sweep actually shoved (the push loop only
    // mutates a touched dynamic's `vel`) — a no-op velocity rewrite would wake every nearby resting body.
    for (let i = 0; i < _push.length; i++) {
        const v = _push[i].vel;
        _pushVel0[3 * i] = v[0];
        _pushVel0[3 * i + 1] = v[1];
        _pushVel0[3 * i + 2] = v[2];
    }

    sweepCharacter(st, input, _statics, gravity, backend.dt, jumped.has(eid), _push);

    // kinematic upload — the swept pose, with the realized velocity (snap excluded) as the explicit
    // velocity so the carry-of-riders + broadphase pad read the swept motion, not the cosmetic ground snap.
    backend.setKinematic(eid, st.pos, st.quat, false, st.realizedVel);

    // full-speed push (variant A): write each shoved dynamic's new velocity straight through the backend.
    // setVelocity wakes the body, so apply it only to the ones the sweep changed.
    for (let i = 0; i < _push.length; i++) {
        const v = _push[i].vel;
        if (
            v[0] !== _pushVel0[3 * i] ||
            v[1] !== _pushVel0[3 * i + 1] ||
            v[2] !== _pushVel0[3 * i + 2]
        ) {
            backend.setVelocity(_pushEids[i], v[0], v[1], v[2]);
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
        const backend = Physics.backend;
        if (!backend) return;
        syncStates(state);
        if (states.size === 0) return;
        for (const [eid, st] of states) sweepEid(eid, st, state, backend);
        jumped.clear();
    },
};

/** kinematic-character plugin: registers every `[Character, Body]` and sweeps it (collide-and-slide) each
 *  fixed step, before the physics solve. Backend-neutral: add a physics backend plugin (`TumblePlugin` or
 *  `AvbdPlugin`) to the scene alongside it, and the sweep runs against whichever backend is installed
 *  (it no-ops without one). Drive characters with the {@link move} / {@link jump} surface, or add
 *  {@link Player} for a ready first-person controller. */
export const CharacterPlugin: Plugin = {
    name: "Character",
    components: { Character },
    systems: [CharacterSweepSystem],
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
        resetDrive();
        stamps.clear();
        charSig = FNV_BASIS;
    },
};
