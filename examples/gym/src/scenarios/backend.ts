import {
    AmbientLight,
    Body,
    Camera,
    CameraMode,
    Character,
    CharacterPlugin,
    Color,
    Compute,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    Joint,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    Physics,
    type Plugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    ShapeKind,
    SlabPlugin,
    Spring,
    type State,
    type System,
    Time,
    Transform,
    TransformsPlugin,
    TumblePlugin,
} from "@dylanebert/shallot";
import { AvbdPlugin } from "@dylanebert/shallot/avbd";
import { grounded, move, pose } from "@dylanebert/shallot/character/core";
import { Profile, ProfilePlugin } from "@dylanebert/shallot/extras";
import { bodyCandidates, raycast, StepSystem } from "@dylanebert/shallot/physics/core";
// the tumble kernel's resolved thread count (read-only diagnostic on the extension subpath) — the
// isolation gate reads it to confirm the multithreaded boot engaged
import { threads } from "@dylanebert/shallot/tumble/core";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";

// backend — the substrate swap gate (specs/tumble-shallot.md stage 4): ONE scene, authored purely against
// the `standard/physics` substrate (Body components, `Physics.backend`'s kinematic drive, the CPU raycast,
// the `transforms` firehose), that runs unmodified under EITHER `TumblePlugin` (default) or `AvbdPlugin`
// (`--param backend=tumble|avbd`) — the one-line manifest swap the substrate's typed `PhysicsBackend`
// handle exists to make possible (physics.md substrate rule, `standard/physics/index.ts`). Where the
// sibling `pile`/`constraints`/`character` scenarios gate the AVBD SOLVER's math against the f64 oracle,
// this scenario gates the SUBSTRATE's contract: the same behavioral assertions must hold under both
// backends, since two solvers can never bit-match a trajectory (Lyapunov) — cross-backend parity is
// behavioral, never bit-exact (the spec's locked decision).
//
// The gate set:
//   • settle + no-fall-through — a dropped grid of boxes comes to rest on the floor, under either backend.
//   • raycast — the backend-neutral CPU raycast (`physics/core`) hits the settled target box (bodyCandidates
//     reads live poses through `Physics.backend`, so the cast never touches a backend internal).
//   • drive + writeback — a kinematic platform driven by `Physics.backend.setKinematic` each fixed tick
//     tracks its commanded trajectory (the drive), and the `transforms` firehose reflects that pose (the
//     writeback `ComposeSystem` delegates to every backend) — the two atomic-core primitives past pose/step.
//   • constraints — the authored `Spring`/`Joint` component path (`ConstraintSystem` → the backend's
//     `setSprings`/`setJoints`): a hanging spring block settles at the mg/k equilibrium (the stiffness law is
//     backend-neutral — tumble derives its hertz from it), a spherical pendulum holds its pin length, a fixed
//     joint holds its authored pose. Cross-backend behavioral bands, never trajectories.
//   • character — the SHARED CPU sweep (`standard/character`, backend-neutral since stage 5 decoupled it from
//     AVBD) drives a capsule to a waypoint and grounds it, under either backend, through the same
//     `readBody`/`setKinematic` seams the drive gate exercises directly.
//   • measured — the per-tick CPU spans (`Profile.cpu`, the scheduler's automatic per-system timing) for the
//     shared substrate systems (`step` / `constraints` / `compose` / `character`) plus each backend's own
//     sync system (`tumble-sync` / `pack`), so a `count` sweep gives a comparable backend-vs-backend perf
//     snapshot.

const FLOOR_HALF_Y = 0.5;
const FLOOR_MARGIN = 2; // clearance past the grid's outer edge / the platform's swept lane
const BOX_HALF = 0.4;
const SPACING = 1.2;
const DROP_Y = 5;
const REST_Y = FLOOR_HALF_Y + BOX_HALF; // 0.9 — the box settles on the floor top
const PLATFORM_HALF: [number, number, number] = [1.5, 0.25, 1.5];
const PLATFORM_Y = 1.5;
const PLATFORM_X0 = 6;
const PLATFORM_AMPLITUDE = 2;
const PLATFORM_PERIOD = 6; // seconds — slow, so a stale (AVBD Mirror) readback stays within tolerance

const platformX = (t: number): number =>
    PLATFORM_X0 + PLATFORM_AMPLITUDE * Math.sin((2 * Math.PI * t) / PLATFORM_PERIOD);

// the constraint rig hangs off a static anchor on the −x lane (clear of the grid, which offsets in −z,
// and the platform's +x lane): a spring block (equilibrium mg/k below rest), a spherical pendulum bob
// (pin length invariant), a welded arm (authored pose held).
const ANCHOR_POS: [number, number, number] = [-6, 10, 0];
const SPRING_STIFFNESS = 100;
const SPRING_REST = 4;
const SPRING_MASS = 2;
const PENDULUM_ARM = 2;
const WELD_OFFSET: [number, number, number] = [0, -1, 0];

// the character station: a capsule on the +z lane driven to a waypoint by the shared CPU sweep
const CHAR_SPAWN: [number, number, number] = [0, 1.3, 4]; // floor top 0.5 + half 0.5 + radius 0.3
const CHAR_TARGET_X = 4;
const CHAR_SPEED = 2;

// the same-update recycle station: a probe body seeds at one z, then in ONE update it's
// destroyed and a NEW body created at the recycled eid at a DISTINCT z. Both spawn high + isolated so they
// only free-fall (horizontal pose preserved) — the recycled body must read at its OWN spawn z, never the
// destroyed probe's. Without the substrate's realias fix the new body inherits the probe's seeded pose
// (AVBD's `seeded` flag / tumble's stale handle), reading at the OLD z. The one gate covering both backends.
const RECYCLE_X = 0;
const RECYCLE_Y = 20; // far above everything — no contact in the short sim window, pure free-fall
const RECYCLE_Z_OLD = -1.5;
const RECYCLE_Z_NEW = 1.5;

let boxEids: number[] = [];
let targetEid = -1;
let targetX = 0;
let targetZ = 0;
let platformEid = -1;
let springBlockEid = -1;
let bobEid = -1;
let armEid = -1;
let charEid = -1;
let recycleEid = -1;
// which backend the current build installed — the isolation gate runs only under tumble (avbd never boots
// the tumble kernel, so threads() would stay 1)
let backendName: "tumble" | "avbd" = "tumble";
let xformMirror: Mirror | null = null;

// drives the platform each fixed tick via the substrate's OWN kinematic primitive (`Physics.backend`,
// `physics/core`), and the character via the shared drive surface (`character/core` `move`, consumed by
// the backend-neutral CPU sweep). `before: [StepSystem]` mirrors `ConstraintSystem`/
// `CharacterSweepSystem`: the kinematic pose must land before THIS tick's solve, or the backend
// collides against last tick's platform pose.
const DriverPlugin: Plugin = {
    name: "BackendDriver",
    systems: [
        {
            name: "backend-driver",
            group: "fixed",
            before: [StepSystem],
            update(state: State) {
                const backend = Physics.backend;
                if (!backend || platformEid < 0) return;
                const t = state.time.fixedTick * Time.FIXED_DT;
                backend.setKinematic(platformEid, [platformX(t), PLATFORM_Y, 0], [0, 0, 0, 1]);
                // walk the character to its waypoint, then stop (move() persists until changed)
                if (charEid >= 0 && pose(charEid, _charPose)) {
                    move(charEid, _charPose[0] < CHAR_TARGET_X ? CHAR_SPEED : 0, 0);
                }
            },
        } satisfies System,
    ],
};
const _charPose: [number, number, number] = [0, 0, 0];

function floor(state: State, halfX: number, halfZ: number): void {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, 0, 0, 0, 0);
    Body.halfExtents.set(eid, halfX, FLOOR_HALF_Y, halfZ, 0);
    Body.mass.set(eid, 0);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, 0.28, 0.3, 0.34, 1);
}

function box(state: State, x: number, z: number, color: [number, number, number]): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, x, DROP_Y, z, 0);
    Body.halfExtents.set(eid, BOX_HALF, BOX_HALF, BOX_HALF, 0);
    Body.mass.set(eid, 1);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

function platform(state: State): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, PLATFORM_X0, PLATFORM_Y, 0, 0);
    Body.halfExtents.set(eid, PLATFORM_HALF[0], PLATFORM_HALF[1], PLATFORM_HALF[2], 0);
    Body.mass.set(eid, 0); // kinematic — moved by DriverPlugin, never falls
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, 0.5, 0.55, 0.85, 1);
    return eid;
}

function rigBody(state: State, pos: [number, number, number], half: number, mass: number): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.halfExtents.set(eid, half, half, half, 0);
    Body.mass.set(eid, mass);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, 0.85, 0.6, 0.4, 1);
    return eid;
}

// the constraint rig, authored as Spring/Joint COMPONENT entities — the path ConstraintSystem uploads
// through the installed backend's setSprings/setJoints (the def mapping under test).
function constraintRig(state: State): void {
    const [ax, ay, az] = ANCHOR_POS;
    const anchor = rigBody(state, ANCHOR_POS, 0.15, 0);
    springBlockEid = rigBody(state, [ax, ay - SPRING_REST, az], 0.4, SPRING_MASS);
    bobEid = rigBody(state, [ax + PENDULUM_ARM, ay, az], 0.25, 1);
    armEid = rigBody(
        state,
        [ax + WELD_OFFSET[0], ay + WELD_OFFSET[1], az + WELD_OFFSET[2]],
        0.25,
        1,
    );

    const spring = state.create();
    state.add(spring, Spring);
    Spring.a.set(spring, anchor);
    Spring.b.set(spring, springBlockEid);
    Spring.stiffness.set(spring, SPRING_STIFFNESS);
    Spring.rest.set(spring, SPRING_REST);

    const pin = state.create();
    state.add(pin, Joint);
    Joint.a.set(pin, anchor);
    Joint.b.set(pin, bobEid);
    Joint.rB.set(pin, -PENDULUM_ARM, 0, 0, 0);

    const weld = state.create();
    state.add(weld, Joint);
    Joint.a.set(weld, anchor);
    Joint.b.set(weld, armEid);
    Joint.rA.set(weld, WELD_OFFSET[0], WELD_OFFSET[1], WELD_OFFSET[2], 0);
    Joint.stiffnessAng.set(weld, Number.POSITIVE_INFINITY);
}

function character(state: State): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Capsule);
    Body.pos.set(eid, CHAR_SPAWN[0], CHAR_SPAWN[1], CHAR_SPAWN[2], 0);
    Body.halfExtents.set(eid, 0, 0.5, 0, 0.3);
    Body.mass.set(eid, 0); // kinematic — the CPU sweep owns the pose
    state.add(eid, Character);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, 0.4, 0.85, 0.5, 1);
    return eid;
}

// a dynamic box high above the scene at a given z — the recycle station's probe + its replacement
function recycleBox(state: State, z: number): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, RECYCLE_X, RECYCLE_Y, z, 0);
    Body.halfExtents.set(eid, BOX_HALF, BOX_HALF, BOX_HALF, 0);
    Body.mass.set(eid, 1);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, 0.9, 0.5, 0.4, 1);
    return eid;
}

const scenario: Scenario = {
    name: "backend",
    params: [
        {
            key: "backend",
            type: "select",
            options: ["tumble", "avbd"],
            default: "tumble",
            rebuild: true,
        },
        // the body-count sweep the perf snapshot reads (`scripts/physics-bench.ts`-style, per-backend):
        // a grid of `count` dropped boxes, out of the drive/raycast stations' way.
        { key: "count", type: "number", default: 9, min: 1, max: 400, step: 1, rebuild: true },
    ],

    async build(_canvas, p: Params) {
        const backend = (p.backend as string) === "avbd" ? "avbd" : "tumble";
        backendName = backend;
        const { state, dispose } = await run({
            defaults: false,
            capacity: 64 + (p.count as number),
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                MirrorPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                backend === "avbd" ? AvbdPlugin : TumblePlugin,
                CharacterPlugin, // backend-neutral: the SAME plugin under either backend
                DriverPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ],
        });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);

        // size the floor + the grid's z-offset off the ACTUAL grid extent (the `count` sweep spans 1..400,
        // a fixed floor footprint would run boxes off its edge at high count — a fall-through that's a
        // scene bug, not a backend difference). The grid sits entirely behind (−z of) the platform's swept
        // lane, cleared by FLOOR_MARGIN regardless of how large the grid grows.
        const count = Math.max(1, (p.count as number) | 0);
        const side = Math.ceil(Math.sqrt(count));
        const gridHalf = ((side - 1) * SPACING) / 2;
        const zOffset = -(gridHalf + PLATFORM_HALF[2] + BOX_HALF + FLOOR_MARGIN);
        const floorHalfX = Math.max(
            10,
            gridHalf + BOX_HALF + FLOOR_MARGIN,
            PLATFORM_X0 + PLATFORM_AMPLITUDE + PLATFORM_HALF[0] + 1,
        );
        const floorHalfZ = Math.max(10, gridHalf + BOX_HALF + Math.abs(zOffset) + FLOOR_MARGIN);
        floor(state, floorHalfX, floorHalfZ);

        boxEids = [];
        for (let i = 0; i < count; i++) {
            const gx = i % side;
            const gz = Math.floor(i / side);
            const x = gx * SPACING - gridHalf;
            const z = gz * SPACING - gridHalf + zOffset;
            const eid = box(state, x, z, [0.5, 0.55 + 0.02 * (i % 5), 0.85 - 0.02 * (i % 5)]);
            boxEids.push(eid);
            if (i === 0) {
                targetEid = eid;
                targetX = x;
                targetZ = z;
            }
        }

        platformEid = platform(state);
        constraintRig(state);
        charEid = character(state);

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 6);
        Orbit.pitch.set(cam, Math.PI / 10);
        Orbit.distance.set(cam, 24 + Math.sqrt(count));

        await frames(3);
        const transforms = Compute.buffers.get("transforms");
        if (transforms) xformMirror = mirror(transforms);
        await frames(240); // settle the grid; the platform drives the whole time

        // the same-update recycle station: seed a probe, then destroy + same-update recreate a DISTINCT body
        // at the recycled eid. No frame runs between destroy and create, so the eid never leaves Body
        // membership — the realias the substrate's stamp diff must catch (ecs.md "An eid is a borrow").
        const probe = recycleBox(state, RECYCLE_Z_OLD);
        await frames(6); // let the backend seed/marshal the probe (its `seeded` flag / handle)
        state.destroy(probe);
        recycleEid = recycleBox(state, RECYCLE_Z_NEW);
        if (recycleEid !== probe) {
            throw new Error(
                `recycle station: create did not reuse the destroyed eid (${recycleEid} vs ${probe})`,
            );
        }
        await frames(12); // the recycled body free-falls from its OWN spawn z

        return {
            state,
            dispose() {
                xformMirror?.dispose();
                xformMirror = null;
                boxEids = [];
                targetEid = -1;
                platformEid = -1;
                springBlockEid = -1;
                bobEid = -1;
                armEid = -1;
                charEid = -1;
                recycleEid = -1;
                dispose();
            },
        };
    },

    async assert(state: State): Promise<Check[]> {
        const checks: Check[] = [];
        checks.push(...settleGates());
        checks.push(raycastGate(state));
        checks.push(...(await driveWritebackGates(state)));
        checks.push(...constraintGates());
        checks.push(...characterGates());
        checks.push(recycleGate());
        if (backendName === "tumble") checks.push(isolationGate());
        checks.push(await measured());
        return checks;
    },

    live(): string {
        const backend = Physics.backend;
        if (!backend) return "backend — warming";
        const p = backend.readBody(platformEid);
        const b0 = boxEids.length > 0 ? backend.readBody(boxEids[0]) : null;
        const c: [number, number, number] = [0, 0, 0];
        const hasChar = charEid >= 0 && pose(charEid, c);
        return [
            `backend — ${boxEids.length} boxes, drive + raycast + constraints + character`,
            `platform x ${p?.pos[0]?.toFixed(2) ?? "—"}  box0 y ${b0?.pos[1]?.toFixed(2) ?? "—"}  char x ${hasChar ? c[0].toFixed(2) : "—"} ${charEid >= 0 && grounded(charEid) ? "grounded" : "air"}`,
        ].join("\n");
    },
};

// ── settle + no-fall-through: every box rests on the floor, under either backend ──

function settleGates(): Check[] {
    const backend = Physics.backend;
    if (!backend) return [{ name: "backend", pass: false, detail: "no physics backend" }];
    const checks: Check[] = [];
    let maxErr = 0;
    let minY = Number.POSITIVE_INFINITY;
    let allFinite = true;
    for (const eid of boxEids) {
        const b = backend.readBody(eid);
        if (!b) {
            allFinite = false;
            continue;
        }
        if (!b.pos.every(Number.isFinite)) allFinite = false;
        minY = Math.min(minY, b.pos[1]);
        maxErr = Math.max(maxErr, Math.abs(b.pos[1] - REST_Y));
    }
    checks.push({
        name: "settle (every dropped box rests on the floor)",
        pass: allFinite && maxErr < 0.1,
        detail: `max |y − rest| ${maxErr.toFixed(4)} (rest ${REST_Y}), all finite ${allFinite}`,
    });
    checks.push({
        name: "no fall-through (nothing sank below the floor)",
        pass: allFinite && minY > REST_Y - 0.5,
        detail: `min y ${minY.toFixed(3)} (rest ${REST_Y})`,
    });
    return checks;
}

// ── raycast: the backend-neutral CPU cast (physics/core) hits the settled target box ──

function raycastGate(state: State): Check {
    const backend = Physics.backend;
    if (!backend) return { name: "raycast", pass: false, detail: "no physics backend" };
    const candidates = bodyCandidates(state, backend);
    const hit = raycast({ origin: [targetX, 10, targetZ], dir: [0, -1, 0] }, candidates);
    const pass = hit !== null && hit.eid === targetEid && hit.distance < 10;
    return {
        name: "raycast hits the settled target box (bodyCandidates reads live pose through Physics.backend)",
        pass,
        detail: hit
            ? `hit eid ${hit.eid} (target ${targetEid}) at distance ${hit.distance.toFixed(3)}`
            : "no hit",
    };
}

// ── drive + writeback: the kinematic platform tracks its commanded trajectory, and the transforms
// firehose (ComposeSystem, shared by both backends) reflects that pose ──

async function driveWritebackGates(state: State): Promise<Check[]> {
    const backend = Physics.backend;
    if (!backend) return [{ name: "backend drive", pass: false, detail: "no physics backend" }];
    const live = backend.readBody(platformEid);
    const checks: Check[] = [];
    if (!live) {
        checks.push({ name: "backend drive", pass: false, detail: "no live platform pose" });
        return checks;
    }
    // the exact commanded x for the LAST fixed tick the driver ran (state.time.fixedTick, the same value
    // the driver read this tick) — a tight parity check, not a static band, so a broken/never-called drive
    // (the platform stuck at spawn) fails it even when spawn happens to sit inside the amplitude band. A
    // stale AVBD Mirror read (up to ~2 ticks behind) drifts by at most 2 · dt · the trajectory's peak
    // velocity (2π·AMPLITUDE/PERIOD ≈ 2.1 m/s) ≈ 0.07 m — DRIVE_TOL covers it with margin.
    const DriveTol = 0.15;
    const expectedX = platformX(state.time.fixedTick * Time.FIXED_DT);
    checks.push({
        name: "drive (Physics.backend.setKinematic moves the platform along its commanded trajectory)",
        pass:
            Math.abs(live.pos[0] - expectedX) < DriveTol &&
            Number.isFinite(live.pos[1]) &&
            Math.abs(live.pos[1] - PLATFORM_Y) < 0.05,
        detail: `platform x ${live.pos[0].toFixed(3)} vs commanded ${expectedX.toFixed(3)} (tol ${DriveTol}), y ${live.pos[1].toFixed(3)} (held ${PLATFORM_Y})`,
    });

    if (xformMirror) {
        await settle(xformMirror);
        const snap = xformMirror.snapshot;
        if (snap) {
            const f = new Float32Array(snap.bytes);
            const o = platformEid * 12; // Xform = 12 f32 (pos.xyz+pad, quat.xyzw, scale.xyz+pad)
            const fx = f[o];
            const fy = f[o + 1];
            checks.push({
                name: "writeback (the transforms firehose reflects the backend's composed platform pose)",
                pass: Math.abs(fx - live.pos[0]) < 0.3 && Math.abs(fy - live.pos[1]) < 0.1,
                detail: `firehose (${fx.toFixed(3)}, ${fy.toFixed(3)}) vs live (${live.pos[0].toFixed(3)}, ${live.pos[1].toFixed(3)})`,
            });
        }
    }
    return checks;
}

// ── constraints: the authored Spring/Joint path holds its behavioral bands under either backend ──

function constraintGates(): Check[] {
    const backend = Physics.backend;
    if (!backend) return [{ name: "constraints", pass: false, detail: "no physics backend" }];
    const checks: Check[] = [];
    const [ax, ay, az] = ANCHOR_POS;

    // the spring block hangs at extension mg/k past rest — the stiffness law both backends share
    // (AVBD's elastic f = k·C; tumble's derived hertz reproduces the same k). ±0.1 is a behavioral
    // band over two different solvers, not a solver tolerance.
    const restY = ay - SPRING_REST - (SPRING_MASS * Math.abs(backend.gravity)) / SPRING_STIFFNESS;
    const block = backend.readBody(springBlockEid);
    checks.push({
        name: "spring settles at the mg/k equilibrium (Spring → backend setSprings)",
        pass: block !== null && Math.abs(block.pos[1] - restY) < 0.1,
        detail: block
            ? `block y ${block.pos[1].toFixed(3)} vs equilibrium ${restY.toFixed(3)}`
            : "no live pose",
    });

    // the pendulum bob's pin length is invariant through the swing (the two backends damp differently,
    // so the PHASE differs — the length is the cross-backend contract)
    const bob = backend.readBody(bobEid);
    let pinLen = Number.NaN;
    if (bob) {
        const dx = bob.pos[0] - ax;
        const dy = bob.pos[1] - ay;
        const dz = bob.pos[2] - az;
        pinLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    checks.push({
        name: "spherical joint holds its pin length (Joint → backend setJoints)",
        pass: bob !== null && Math.abs(pinLen - PENDULUM_ARM) < 0.1,
        detail: bob ? `pin length ${pinLen.toFixed(3)} (rod ${PENDULUM_ARM})` : "no live pose",
    });

    // the welded arm holds its authored pose (the fixed-joint mapping: AVBD ∞ stiffnessAng, tumble weld)
    const arm = backend.readBody(armEid);
    const armHeld =
        arm !== null &&
        Math.abs(arm.pos[0] - (ax + WELD_OFFSET[0])) < 0.1 &&
        Math.abs(arm.pos[1] - (ay + WELD_OFFSET[1])) < 0.1 &&
        Math.abs(arm.pos[2] - (az + WELD_OFFSET[2])) < 0.1;
    checks.push({
        name: "fixed joint holds the authored pose (stiffness-ang: fixed)",
        pass: armHeld,
        detail: arm
            ? `arm (${arm.pos[0].toFixed(2)}, ${arm.pos[1].toFixed(2)}, ${arm.pos[2].toFixed(2)}) vs authored (${ax + WELD_OFFSET[0]}, ${ay + WELD_OFFSET[1]}, ${az + WELD_OFFSET[2]})`
            : "no live pose",
    });
    return checks;
}

// ── character: the shared CPU sweep (backend-neutral since stage 5) walks to its waypoint and grounds ──

function characterGates(): Check[] {
    if (charEid < 0) return [{ name: "character", pass: false, detail: "no character" }];
    const checks: Check[] = [];
    const p: [number, number, number] = [0, 0, 0];
    const has = pose(charEid, p);
    checks.push({
        name: "character walks to its waypoint (shared sweep over Physics.backend)",
        pass: has && Math.abs(p[0] - CHAR_TARGET_X) < 0.3,
        detail: has
            ? `char x ${p[0].toFixed(3)} (waypoint ${CHAR_TARGET_X})`
            : "no controller state",
    });
    checks.push({
        name: "character grounded at rest height",
        pass: has && grounded(charEid) && Math.abs(p[1] - CHAR_SPAWN[1]) < 0.05,
        detail: has
            ? `grounded ${grounded(charEid)}, y ${p[1].toFixed(3)} (rest ${CHAR_SPAWN[1]})`
            : "no controller state",
    });
    return checks;
}

// ── same-update recycle: the body created at a recycled eid simulates from ITS spawn, not the destroyed
// body's inherited pose — the one gate covering the substrate realias fix under BOTH backends ──

function recycleGate(): Check {
    const backend = Physics.backend;
    if (!backend) return { name: "recycle", pass: false, detail: "no physics backend" };
    if (recycleEid < 0) return { name: "recycle", pass: false, detail: "no recycle body" };
    const b = backend.readBody(recycleEid);
    if (!b) return { name: "recycle", pass: false, detail: "no live recycled pose" };
    // the recycled body free-falls from its own spawn, so its z is preserved: it must read at the NEW
    // spawn z, far from the destroyed probe's OLD z. Inheriting the probe's seeded pose reads the OLD z.
    const dNew = Math.abs(b.pos[2] - RECYCLE_Z_NEW);
    const dOld = Math.abs(b.pos[2] - RECYCLE_Z_OLD);
    return {
        name: "same-update recycle simulates as the NEW body (not the destroyed body's inherited pose)",
        pass: b.pos.every(Number.isFinite) && dNew < 0.2 && dOld > 1,
        detail: `recycled z ${b.pos[2].toFixed(3)} — new ${RECYCLE_Z_NEW} (Δ ${dNew.toFixed(3)}), old ${RECYCLE_Z_OLD} (Δ ${dOld.toFixed(3)}); x ${b.pos[0].toFixed(2)} y ${b.pos[1].toFixed(2)}`,
    };
}

// ── MT isolation (tumble only) — the served page IS cross-origin isolated AND the tumble kernel booted
// multithreaded. Guards the dev/preview COOP/COEP headers (bin devConfig / serveEjected / serveDist /
// run preview) end to end in a real browser: a header regression silently degrades tumble to single
// thread and every other gate still passes, so assert the positive. Runs only under TumblePlugin — avbd
// never boots the tumble kernel, so threads() would stay 1 there ──
function isolationGate(): Check {
    const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    const t = threads();
    return {
        name: "MT isolation (page cross-origin isolated, tumble multithreaded)",
        pass: isolated && t > 1,
        detail: `crossOriginIsolated ${isolated}, threads ${t}`,
    };
}

// ── perf reporter (always-pass) — the CPU spans the scheduler already times per system, shared substrate
// names (`step`/`constraints`/`compose`/`character`) plus each backend's own sync system, so a `count`
// sweep gives a direct backend-vs-backend comparison at the `bun bench` tier ──

async function measured(): Promise<Check> {
    const names = ["step", "constraints", "compose", "character", "tumble-sync", "pack"];
    // scheduler spans are plugin-namespaced (`Tumble/step`, `Avbd/pack` — scheduler.ts `_names`), so
    // match by the name AFTER the slash; and `Profile.cpu` holds one frame's spans (cleared at frame
    // begin) while the fixed group runs ~0.5 ticks/frame, so a single-frame sample coin-flips to zero —
    // accumulate over a window and report ms/frame.
    const Window = 60;
    const totals: Record<string, number> = {};
    for (const n of names) totals[n] = 0;
    for (let i = 0; i < Window; i++) {
        await frames(1);
        for (const [span, ms] of Profile.cpu) {
            const base = span.slice(span.indexOf("/") + 1);
            if (base in totals) totals[base] += ms;
        }
    }
    const data: Record<string, number> = {};
    for (const n of names) data[n] = totals[n] / Window;
    data.bodies = boxEids.length + 2; // + the floor + the platform
    const parts = names.map((n) => `${n} ${data[n].toFixed(4)}`).join(" · ");
    return {
        name: "measured (per-system CPU ms/frame)",
        pass: true,
        detail: `${data.bodies} bodies — ${parts} ms`,
        data,
    };
}

register(scenario);
