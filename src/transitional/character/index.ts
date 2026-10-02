// Destination: standard/physics; owner: physics-boundary.md.
import { f32, type Plugin, registration, type System, Time, type World } from "../../engine";
import {
    Body,
    type BodyStateOut,
    type Hull,
    Hulls,
    Physics,
    physicsWorld,
    readBody,
    ShapeKind,
    StepPhysicsSystem,
    setKinematic,
    setVelocity,
} from "../physics";
import { driveFor, resetDrive } from "./drive";
import { type CharState, type SweepBody, sweepCharacter } from "./sweep";

// Character is the kinematic capsule controller that Player composes. Its Body is produced by the CPU
// sweep: each fixed tick SweepCharactersSystem runs collide-and-slide before the physics solve, writes the
// swept GlobalTransform through setKinematic, and the solver collides dynamics against that same tick's
// body. The CPU sweep reads other bodies through readBody; there is no readback in this path.
//
// This module owns the tuning component, the per-tick sweep, and the eid-keyed move/jump controls plus
// GlobalTransform-position and grounded reads used by followers. The CPU sweep is the sole runtime
// controller; the f64 controller oracle specifies the behavior. Player composes look and a camera above it.

const fixedDeltaTime = Time.FIXED_DT;
const DEG = Math.PI / 180;
const _worldGravity = { x: 0, y: 0, z: 0 };

/**
 * a kinematic character: a capsule {@link Body} (`mass <= 0`) swept against the scene's bodies each fixed
 * step (collide-and-slide on the CPU). Authors the walkable slope, the jump launch speed, and a per-character
 * gravity; the controller sweeps every `[Character, Body]`, drives it via {@link move} / {@link jump}, and
 * reads {@link grounded} / {@link globalTransform} back. The first-person {@link Player} composes this.
 *
 * @example
 * ```
 * const body = world.create();
 * world.add(body, Body); world.add(body, Character);
 * const bodies = world.storage(Body);
 * bodies.shape.set(body, ShapeKind.Capsule);
 * bodies.halfExtents.set(body, 0, 0.5, 0, 0.3); bodies.mass.set(body, 0);
 * world.storage(Character).jumpSpeed.set(body, 5);   // 0 = no jump
 * ```
 */
export const Character = {
    /** steepest walkable slope in degrees; a contact flatter than this grounds the character, steeper it slides */
    maxSlope: f32,
    /** the launch velocity a buffered + grounded {@link jump} sets. 0 disables jumping */
    jumpSpeed: f32,
    /** per-character gravity (negative, snappier than the world for a player). 0 = the configured world gravity */
    gravity: f32,
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

// the create-stamp each `states` entry was built at. A same-update
// destroy+create recycling a character's eid with identical tuning hashes to the SAME signature, so folding
// the stamp into the signature is what makes the realias visible; the per-eid compare in `syncStates` then
// rebuilds the controller state (stale position/velocity kept across the recycle is the bug this closes).

// query terms held once, so a steady signature mints no array.
const CHARACTER_TERMS = [Character, Body];

function signature(world: World): number {
    let h = FNV_BASIS;
    for (const eid of world.query(CHARACTER_TERMS)) {
        h = fold(h, eid);
        h = fold(h, world.generation(eid));
        h = fold(h, sigBits(world.storage(Character).maxSlope.get(eid)));
        h = fold(h, sigBits(world.storage(Character).jumpSpeed.get(eid)));
        h = fold(h, sigBits(world.storage(Character).gravity.get(eid)));
    }
    return h;
}

// build a fresh controller state from a character's authored Body placement, capsule geometry and walkable-slope cutoff. Velocity / grounded / jump timers start cleared (a dropped capsule falls to rest).
function buildState(world: World, eid: number): CharState {
    return {
        pos: [
            world.storage(Body).position.x.get(eid),
            world.storage(Body).position.y.get(eid),
            world.storage(Body).position.z.get(eid),
        ],
        quat: [
            world.storage(Body).rotation.x.get(eid),
            world.storage(Body).rotation.y.get(eid),
            world.storage(Body).rotation.z.get(eid),
            world.storage(Body).rotation.w.get(eid),
        ],
        half: world.storage(Body).halfExtents.y.get(eid),
        radius: world.storage(Body).halfExtents.w.get(eid),
        maxSlopeCos: Math.cos(world.storage(Character).maxSlope.get(eid) * DEG),
        jumpSpeed: world.storage(Character).jumpSpeed.get(eid),
        vel: [0, 0, 0],
        realizedVel: [0, 0, 0],
        grounded: false,
        groundNormal: [0, 0, 0],
        coyote: 0,
        buffer: 0,
    };
}

// re-sync `states` to the authored `[Character, Body]` set on a signature change. A new character builds a
// fresh state; an existing one KEEPS its live Body position and motion (the controller owns the fixed-tick
// GlobalTransform — a sibling spawn must not reset a walking character) and only picks up a tuning edit; a
// removed one is dropped. A fresh World starts with `states` empty, so its first sync reads authored Body fields.
function syncStates(world: World): void {
    const drive = driveFor(world);
    const sig = signature(world);
    if (sig === drive.signature) return;
    drive.signature = sig;
    rebuildStates(world, drive);
}

function rebuildStates(world: World, drive: ReturnType<typeof driveFor>): void {
    const seen = new Set<number>();
    for (const eid of world.query([Character, Body])) {
        seen.add(eid);
        const stamp = world.generation(eid);
        const st = drive.states.get(eid);
        if (st && drive.stamps.get(eid) === stamp) {
            st.maxSlopeCos = Math.cos(world.storage(Character).maxSlope.get(eid) * DEG);
            st.jumpSpeed = world.storage(Character).jumpSpeed.get(eid);
            st.half = world.storage(Body).halfExtents.y.get(eid);
            st.radius = world.storage(Body).halfExtents.w.get(eid);
        } else {
            if (st) {
                // realias: drive input keyed to the destroyed owner is stale. A fresh spawn keeps
                // input queued before its first sync.
                drive.moves.delete(eid);
                drive.jumped.delete(eid);
            }
            drive.states.set(eid, buildState(world, eid));
            drive.stamps.set(eid, stamp);
        }
    }
    for (const eid of [...drive.states.keys()]) {
        if (!seen.has(eid)) {
            drive.states.delete(eid);
            drive.moves.delete(eid);
            drive.jumped.delete(eid);
            drive.stamps.delete(eid);
        }
    }
}

// reused candidate scratch — the bodies are split into static (mass <= 0: walls / ground / platforms / other
// characters — the carry reads their velocity) and dynamic (mass > 0 — shoved by the push) sets each tick.
// A growing pool of SweepBody objects avoids per-tick allocation as the scan walks every Body; the lists are
// refilled in place and trimmed only when their length changes, since a length of 0 releases the store.
const _pool: SweepBody[] = [];
const _statics: SweepBody[] = [];
const _push: SweepBody[] = [];
const _pushEids: number[] = [];
const _pushVel0: number[] = []; // pre-sweep dynamic velocities, to detect which the push actually shoved
const _live: BodyStateOut = {
    position: [0, 0, 0],
    rotation: [0, 0, 0, 1],
    linearVelocity: [0, 0, 0],
};
const _input: [number, number, number] = [0, 0, 0];
const BODY_TERMS = [Body];

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

const hullById = (world: World, id: number): Hull | undefined => {
    const hulls = world.resource(Hulls);
    return hulls.get(hulls.name(id) ?? "");
};

// one character's sweep: gather candidates (geometry from authored Body fields, live Body placement + velocity
// through the backend read seam — the static world is unchanged by the possible one-tick lag, and one-tick-old
// dynamic or platform data is fine), run collide-and-slide, upload the result as a kinematic body, and apply
// full-speed pushes to shoved dynamics (variant A — the CPU writes swept velocity directly through `setVelocity`).
function sweepEid(eid: number, st: CharState, world: World): void {
    let pi = 0;
    let ns = 0;
    let np = 0;
    for (const b of world.query(BODY_TERMS)) {
        if (b === eid) continue; // the character never collides against itself (it IS `start`)
        const shape = world.storage(Body).shape.get(b);
        const sb = poolBody(pi++);
        sb.shape = shape;
        sb.half[0] = world.storage(Body).halfExtents.x.get(b);
        sb.half[1] = world.storage(Body).halfExtents.y.get(b);
        sb.half[2] = world.storage(Body).halfExtents.z.get(b);
        const hw = world.storage(Body).halfExtents.w.get(b); // a rounding radius (sphere/capsule) OR a hull id (shape 3)
        if (shape === ShapeKind.Hull) {
            sb.radius = 0;
            sb.hull = hullById(world, hw);
        } else {
            sb.radius = hw;
            sb.hull = undefined;
        }
        const live = readBody(world, b, _live);
        if (live) {
            sb.pos[0] = live.position[0];
            sb.pos[1] = live.position[1];
            sb.pos[2] = live.position[2];
            sb.quat[0] = live.rotation[0];
            sb.quat[1] = live.rotation[1];
            sb.quat[2] = live.rotation[2];
            sb.quat[3] = live.rotation[3];
            sb.vel[0] = live.linearVelocity[0];
            sb.vel[1] = live.linearVelocity[1];
            sb.vel[2] = live.linearVelocity[2];
        } else {
            // cold start (no live Body state yet): authored Body placement, velocity 0 — correct for the static
            // collision world the character needs from frame 1, and a freshly spawned dynamic hasn't moved.
            sb.pos[0] = world.storage(Body).position.x.get(b);
            sb.pos[1] = world.storage(Body).position.y.get(b);
            sb.pos[2] = world.storage(Body).position.z.get(b);
            sb.quat[0] = world.storage(Body).rotation.x.get(b);
            sb.quat[1] = world.storage(Body).rotation.y.get(b);
            sb.quat[2] = world.storage(Body).rotation.z.get(b);
            sb.quat[3] = world.storage(Body).rotation.w.get(b);
            sb.vel[0] = 0;
            sb.vel[1] = 0;
            sb.vel[2] = 0;
        }
        if (world.storage(Body).mass.get(b) > 0) {
            _pushEids[np] = b;
            _push[np++] = sb;
        } else {
            _statics[ns++] = sb;
        }
    }
    if (_statics.length !== ns) _statics.length = ns;
    if (_push.length !== np) _push.length = np;

    const drive = driveFor(world);
    const m = drive.moves.get(eid);
    const input = _input;
    input[0] = m ? m[0] : 0;
    input[2] = m ? m[1] : 0;
    const g = world.storage(Character).gravity.get(eid);
    const gravity =
        g !== 0 ? g : (physicsWorld(world)?.getGravity(_worldGravity).y ?? Physics.gravity);

    // snapshot the dynamics' velocities so we can tell which the sweep actually shoved (the push loop only
    // mutates a touched dynamic's `vel`) — a no-op velocity rewrite would wake every nearby resting body.
    for (let i = 0; i < _push.length; i++) {
        const v = _push[i].vel;
        _pushVel0[3 * i] = v[0];
        _pushVel0[3 * i + 1] = v[1];
        _pushVel0[3 * i + 2] = v[2];
    }

    sweepCharacter(st, input, _statics, gravity, fixedDeltaTime, drive.jumped.has(eid), _push);

    // kinematic upload — the swept position and rotation, with realized velocity (snap excluded) so the carry
    // of riders and broadphase pad follow actual motion, not the cosmetic ground snap.
    setKinematic(world, eid, st.pos, st.quat, false, st.realizedVel);

    // full-speed push (variant A): write each shoved dynamic's new velocity straight through the backend.
    // setVelocity wakes the body, so apply it only to the ones the sweep changed.
    for (let i = 0; i < _push.length; i++) {
        const v = _push[i].vel;
        if (
            v[0] !== _pushVel0[3 * i] ||
            v[1] !== _pushVel0[3 * i + 1] ||
            v[2] !== _pushVel0[3 * i + 2]
        ) {
            setVelocity(world, _pushEids[i], v[0], v[1], v[2]);
        }
    }
}

// the map walk's callback, given the World as its `this`, so a steady update mints no entries iterator.
function sweepEach(this: World, st: CharState, eid: number): void {
    sweepEid(eid, st, this);
}

// Fixed group — the deterministic dt the sweep integrates gravity over.
/**
 * The kinematic-character sweep runs collide-and-slide for every `[Character, Body]` each fixed step before
 * the physics solve and updates the body's GlobalTransform through the physics backend. A follower that reads
 * GlobalTransform (a camera or attached prop) declares `after: [SweepCharactersSystem]` to read this tick's value.
 */
export const SweepCharactersSystem: System = {
    name: "character",
    group: "fixed",
    before: [StepPhysicsSystem],
    update(world: World) {
        if (!physicsWorld(world)) return;
        syncStates(world);
        const drive = driveFor(world);
        if (drive.states.size === 0) return;
        drive.states.forEach(sweepEach, world);
        if (drive.jumped.size !== 0) drive.jumped.clear();
    },
};

/** kinematic-character plugin: registers every `[Character, Body]` and sweeps it (collide-and-slide) each
 *  fixed step, before the physics solve. Add `PhysicsPlugin` to the scene alongside it; the sweep no-ops without
 *  a physics world. Drive characters with the {@link move} / {@link jump} surface, or add
 *  {@link Player} for a ready first-person controller. */
export const CharacterPlugin: Plugin = {
    name: "Character",
    components: [
        registration("Character", Character, {
            defaults: () => ({
                maxSlope: 45,
                jumpSpeed: 0, // no jump
                gravity: 0, // = the configured world gravity
            }),
        }),
    ],
    systems: [SweepCharactersSystem],

    dispose(world: World) {
        resetDrive(world);
    },
};

// Character extension surface — the eid-keyed drive (`move` / `jump`) + readback (`globalTransform` / `grounded`), for
// custom controllers. The happy path (the `Character` component +
// `CharacterPlugin`, which registers every `[Character, Body]` with the solver) ships on the barrel.

export { globalTransform, grounded, jump, move, teleport } from "./drive";
export {
    type CharState,
    MAX_CHAR_CANDIDATES,
    type SweepBody,
    type SweepDiag,
    sweepCharacter,
} from "./sweep";
