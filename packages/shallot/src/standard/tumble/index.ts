import { Compute, capacity, type Plugin, type State, type System, Time } from "../../engine";
import {
    Body,
    bodyTraits,
    ComposeSystem,
    ConstraintSystem,
    installBackend,
    Joint,
    jointTraits,
    type PhysicsBackend,
    Spring,
    StepSystem,
    springTraits,
    uninstallBackend,
} from "../physics";
import { Hulls } from "../physics/core";
import { SlabPlugin } from "../slab";
import { nlerpShortest, renderScale } from "./compose";
import { init, type Body as TumbleBody, World as TumbleWorld } from "./engine";
import { resetConstraints, syncJoints, syncSprings } from "./joints";
import { marshalBody } from "./marshal";

// tumble physics — the default backend for the physics substrate (`standard/physics`), a CPU rigid-body
// solver behind the same `PhysicsBackend` handle `standard/avbd`'s `AvbdPlugin` implements. Runs entirely on
// the CPU (no GPU device needed to step, so it's testable headless): world lifecycle (one live World, exclusive
// resident wasm region — see tumble.md), membership-driven body create/destroy in deterministic (ascending
// eid) order, shape marshaling (`marshal.ts`), and CPU writeback (move events → an interpolated pose written
// into the `transforms` firehose, movers only — a sleeping/static body skips the write for free).
//
// Storage: an eid↔tumble-Body map plus a capacity-sized prev/curr pose double-buffer for render interpolation
// (`compose`, below) — CPU arrays, no slab/mirror needed (tumble serves this tick's fresh state directly,
// the structural reason CPU physics pairs with live skinning, tumble.md). Constraint (Spring/
// Joint) marshaling is `joints.ts` — the ConstraintSystem's authored defs reconciled onto tumble joints.

const GRAVITY = -10; // matches AVBD's world gravity (physics.md), so the backend-swap gym atom feels the same
const SUBSTEPS = 4; // the engine's own recommended sub-step count (World.step's default)

/** the running tumble state: custom tooling reads/drives the live `World` directly (9 joint configs,
 *  sensors, contact/hit events, mesh/heightfield/compound, native queries — everything past the substrate's
 *  atomic `PhysicsBackend` core). `null` until {@link TumblePlugin.warm} creates a world. `body(eid)` returns
 *  the live tumble handle for a substrate `Body` entity — the escape hatch's eid↔handle bridge, so tooling
 *  (a ragdoll wiring cone/twist/filter joints between named bodies) can reach a specific body. `null` before
 *  the entity is marshaled (its first `fixed`-group tick) or for a non-`Body` eid. */
export const Tumble: { world: TumbleWorld | null; body(eid: number): TumbleBody | null } = {
    world: null,
    body: (eid) => bodies.get(eid) ?? null,
};

let world: TumbleWorld | null = null;
const bodies = new Map<number, TumbleBody>();
// the create-stamp each body was marshaled at (ecs.md "An eid is a borrow"). Presence in `bodies` catches a
// plain spawn/despawn; a same-update destroy+create recycling an eid keeps Body membership AND the map entry,
// so the stamp is the only signal that the slot now holds a new body, and a mismatch re-marshals it.
const stamps = new Map<number, number>();
// last pose passed to setKinematic per eid — setKinematic derives a platform's velocity from its per-step
// pose delta against this (mirrors AVBD step.ts's `_kinPrev`); teleport resets it so the derived delta is 0.
const kinPrev = new Map<number, [number, number, number]>();
// bodies whose marshal failed (an unregistered/unbuildable hull) — keyed to the stamp + hull-registry size
// they failed at, so SyncSystem retries the marshal (and re-warns) only when the eid recycles or a new hull
// is registered, never every frame. Normally empty; the error path never thrashes the frame loop.
const failed = new Map<number, { stamp: number; hulls: number }>();

// render-interpolation double buffer, capacity-sized flat arrays indexed by eid (3 lanes pos, 4 lanes quat).
// Rewritten only for a body that actually moved this fixed tick (from `getBodyEvents`), so a sleeping/static
// body's prev==curr holds from whenever it last moved (or its spawn pose) — compose then blends a no-op.
let prevPos = new Float32Array(0);
let prevQuat = new Float32Array(0);
let currPos = new Float32Array(0);
let currQuat = new Float32Array(0);
// the eids `getBodyEvents` reported this fixed tick — compose (draw group, every render frame) rewrites
// exactly these into the transforms firehose until the next fixed tick's move events replace the set.
const movedThisTick = new Set<number>();

function seedPose(
    eid: number,
    px: number,
    py: number,
    pz: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
): void {
    const p = eid * 3;
    const q = eid * 4;
    prevPos[p] = px;
    prevPos[p + 1] = py;
    prevPos[p + 2] = pz;
    currPos[p] = px;
    currPos[p + 1] = py;
    currPos[p + 2] = pz;
    prevQuat[q] = qx;
    prevQuat[q + 1] = qy;
    prevQuat[q + 2] = qz;
    prevQuat[q + 3] = qw;
    currQuat[q] = qx;
    currQuat[q + 1] = qy;
    currQuat[q + 2] = qz;
    currQuat[q + 3] = qw;
}

// membership-driven create/destroy, ascending eid order (state.query's natural order — creation order is
// load-bearing for tumble determinism, tumble.md "Creation order"). Runs every fixed tick
// before the solve so a body spawned this frame joins THIS tick's step, not next tick's (no pack latency —
// unlike AVBD's GPU firehose, a CPU per-entity diff is the substrate-established shape for CPU-side sync,
// mirroring ConstraintSystem's own state.query([Spring]) walk).
const SyncSystem: System = {
    name: "tumble-sync",
    group: "fixed",
    before: [ConstraintSystem, StepSystem],
    update(state: State) {
        if (!world) return;
        for (const eid of state.query([Body])) {
            const stamp = state.stamp(eid);
            if (bodies.has(eid)) {
                if (stamps.get(eid) === stamp) continue;
                // the eid was recycled to a new Body in one update — the old handle + its derived
                // kinematic-prev / move state belong to the destroyed body.
                bodies.get(eid)?.destroy();
                bodies.delete(eid);
                kinPrev.delete(eid);
                movedThisTick.delete(eid);
            }
            // skip a body whose marshal already failed at this stamp + hull-registry state (a bad hull id):
            // retry only when the eid recycles (stamp changes) or a hull is registered (registry grows).
            const f = failed.get(eid);
            if (f && f.stamp === stamp && f.hulls === Hulls.size) continue;
            const tb = marshalBody(world, eid);
            if (!tb) {
                failed.set(eid, { stamp, hulls: Hulls.size });
                continue;
            }
            failed.delete(eid);
            bodies.set(eid, tb);
            stamps.set(eid, stamp);
            seedPose(
                eid,
                Body.pos.x.get(eid),
                Body.pos.y.get(eid),
                Body.pos.z.get(eid),
                Body.quat.x.get(eid),
                Body.quat.y.get(eid),
                Body.quat.z.get(eid),
                Body.quat.w.get(eid),
            );
        }
        if (failed.size > 0) {
            for (const eid of failed.keys()) {
                if (!state.has(eid, Body)) failed.delete(eid);
            }
        }
        if (bodies.size === 0) return;
        const stale: number[] = [];
        for (const eid of bodies.keys()) {
            if (!state.has(eid, Body)) stale.push(eid);
        }
        for (const eid of stale) {
            bodies.get(eid)?.destroy();
            bodies.delete(eid);
            stamps.delete(eid);
            kinPrev.delete(eid);
            movedThisTick.delete(eid);
        }
    },
};

// this backend's `PhysicsBackend` implementation — installed at `Physics.backend` in `warm()`, uninstalled in
// `dispose()`. Every method reads the module-level `world`/`bodies` fresh (never captured), so it stays
// correct across a State rebuild that tears down + recreates the world.
const backendHandle: PhysicsBackend = {
    step() {
        if (!world) return;
        world.step(Time.FIXED_DT, SUBSTEPS);
        movedThisTick.clear();
        const events = world.getBodyEvents();
        for (let i = 0; i < events.count; i++) {
            const ev = events.moveEvents[i];
            const eid = ev.userData as number;
            const p = eid * 3;
            const q = eid * 4;
            prevPos[p] = currPos[p];
            prevPos[p + 1] = currPos[p + 1];
            prevPos[p + 2] = currPos[p + 2];
            prevQuat[q] = currQuat[q];
            prevQuat[q + 1] = currQuat[q + 1];
            prevQuat[q + 2] = currQuat[q + 2];
            prevQuat[q + 3] = currQuat[q + 3];
            currPos[p] = ev.transform.p.x;
            currPos[p + 1] = ev.transform.p.y;
            currPos[p + 2] = ev.transform.p.z;
            currQuat[q] = ev.transform.q.v.x;
            currQuat[q + 1] = ev.transform.q.v.y;
            currQuat[q + 2] = ev.transform.q.v.z;
            currQuat[q + 3] = ev.transform.q.s;
            movedThisTick.add(eid);
        }
    },
    readBody(eid) {
        const tb = bodies.get(eid);
        if (!tb) return null;
        const pos = tb.getPosition();
        const quat = tb.getRotation();
        const vel = tb.getLinearVelocity();
        return {
            pos: [pos.x, pos.y, pos.z],
            quat: [quat.v.x, quat.v.y, quat.v.z, quat.s],
            vel: [vel.x, vel.y, vel.z],
        };
    },
    setKinematic(eid, pos, quat, teleport = false, vel) {
        const tb = bodies.get(eid);
        if (!tb) return;
        let prev = kinPrev.get(eid);
        // did the pose actually move? a slept kinematic body that only setTransform moves emits no move
        // event, so its render firehose slot (movedThisTick) keeps the stale pose while readBody sees the
        // new one — a swap-parity divergence vs AVBD (which composes every live eid). Waking it on a real
        // move makes the solver report the move; a same-pose call leaves it free to sleep (a parked platform
        // whose script keeps re-asserting its pose).
        const moved =
            !prev || teleport || pos[0] !== prev[0] || pos[1] !== prev[1] || pos[2] !== prev[2];
        tb.setTransform(
            { x: pos[0], y: pos[1], z: pos[2] },
            { v: { x: quat[0], y: quat[1], z: quat[2] }, s: quat[3] },
        );
        if (!prev || teleport) {
            prev = [pos[0], pos[1], pos[2]];
            kinPrev.set(eid, prev);
        }
        const dt = Time.FIXED_DT;
        const v = vel ?? [
            (pos[0] - prev[0]) / dt,
            (pos[1] - prev[1]) / dt,
            (pos[2] - prev[2]) / dt,
        ];
        tb.setLinearVelocity({ x: v[0], y: v[1], z: v[2] });
        // setTransform never wakes and setLinearVelocity wakes only on a nonzero velocity, so a zero-velocity
        // teleport (or a move whose derived velocity rounds to zero) needs an explicit wake.
        if (moved && !tb.isAwake()) tb.setAwake(true);
        prev[0] = pos[0];
        prev[1] = pos[1];
        prev[2] = pos[2];
    },
    setVelocity(eid, vx, vy, vz) {
        bodies.get(eid)?.setLinearVelocity({ x: vx, y: vy, z: vz });
    },
    setSprings(springs) {
        if (world) syncSprings(world, bodies, springs);
    },
    setJoints(joints) {
        if (world) syncJoints(world, bodies, joints);
    },
    get gravity() {
        return world ? world.getGravity().y : 0;
    },
    get dt() {
        return Time.FIXED_DT;
    },
    compose(_encoder, transforms, alpha) {
        if (!Compute.device) return;
        for (const eid of movedThisTick) {
            const p = eid * 3;
            const q = eid * 4;
            const pos: [number, number, number] = [
                prevPos[p] * (1 - alpha) + currPos[p] * alpha,
                prevPos[p + 1] * (1 - alpha) + currPos[p + 1] * alpha,
                prevPos[p + 2] * (1 - alpha) + currPos[p + 2] * alpha,
            ];
            const quat = nlerpShortest(
                [prevQuat[q], prevQuat[q + 1], prevQuat[q + 2], prevQuat[q + 3]],
                [currQuat[q], currQuat[q + 1], currQuat[q + 2], currQuat[q + 3]],
                alpha,
            );
            const scale = renderScale(
                Body.shape.get(eid),
                [
                    Body.halfExtents.x.get(eid),
                    Body.halfExtents.y.get(eid),
                    Body.halfExtents.z.get(eid),
                ],
                Body.halfExtents.w.get(eid),
            );
            _record[0] = pos[0];
            _record[1] = pos[1];
            _record[2] = pos[2];
            _record[3] = 0;
            _record[4] = quat[0];
            _record[5] = quat[1];
            _record[6] = quat[2];
            _record[7] = quat[3];
            _record[8] = scale[0];
            _record[9] = scale[1];
            _record[10] = scale[2];
            _record[11] = 0;
            Compute.device.queue.writeBuffer(transforms, eid * 48, _record);
        }
    },
};

// one reused Xform-shaped record (48 B / 12 f32: pos.xyz+pad, quat.xyzw, scale.xyz+pad — XFORM_WGSL) —
// zero per-mover allocation in `compose`'s steady state.
const _record = new Float32Array(12);

/**
 * tumble physics: the default rigid-body backend, installing `Body`/`Spring`/`Joint` and implementing the
 * physics substrate's `PhysicsBackend` handle at `Physics.backend`. Opt-in — add it to a scene to run physics
 * (it's not in the default plugins). Gameplay-complete (9 joint types, mesh/heightfield/compound colliders,
 * sensors, CCD, sleeping) and GPU-feature-free, right for most game-scale scenes; `standard/avbd`'s
 * `AvbdPlugin` is the specialized swap-in for massive GPU-resident scale.
 *
 * @example
 * ```
 * export const config: Config = { plugins: [TumblePlugin], scene: "scenes/scene.scene" };
 * ```
 */
export const TumblePlugin: Plugin = {
    name: "Tumble",
    components: { Body, Spring, Joint },
    systems: [SyncSystem, ConstraintSystem, StepSystem, ComposeSystem],
    dependencies: [SlabPlugin],
    traits: {
        Body: bodyTraits,
        Spring: springTraits,
        Joint: jointTraits,
    },

    initialize() {
        Tumble.world = null;
    },

    async warm() {
        await init(); // async wasm compile — the browser main thread can't compile it synchronously
        world?.destroy();
        world = new TumbleWorld({ gravity: { x: 0, y: GRAVITY, z: 0 } });
        Tumble.world = world;
        bodies.clear();
        stamps.clear();
        kinPrev.clear();
        movedThisTick.clear();
        failed.clear();
        prevPos = new Float32Array(capacity * 3);
        prevQuat = new Float32Array(capacity * 4);
        currPos = new Float32Array(capacity * 3);
        currQuat = new Float32Array(capacity * 4);
        resetConstraints(); // stale handles died with the old world
        installBackend(backendHandle);
    },

    dispose() {
        bodies.clear();
        stamps.clear();
        kinPrev.clear();
        movedThisTick.clear();
        failed.clear();
        resetConstraints();
        world?.destroy();
        world = null;
        Tumble.world = null;
        uninstallBackend();
    },
};
