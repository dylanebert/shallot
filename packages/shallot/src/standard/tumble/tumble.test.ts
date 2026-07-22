import { afterAll, afterEach, describe, expect, test } from "bun:test";
import { attach, stepFor } from "../../../tests/helpers";
import { Compute, State, Time } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { Character, CharacterPlugin } from "../character";
import { grounded, move, pose } from "../character/core";
import {
    Body,
    bodyTraits,
    Joint,
    jointTraits,
    Physics,
    ShapeKind,
    Spring,
    springTraits,
} from "../physics";
import { Hulls } from "../physics/core";
import { Slab } from "../slab";
import {
    BodyType,
    createHull,
    defaultShapeDef,
    defaultSurfaceMaterial,
    type HullData,
    hashWorldState,
    makeBoxHull,
    shutdown,
    type Body as TumbleBody,
    World,
} from "./engine";
import { Tumble, TumblePlugin } from "./index";

// a small tetrahedron, registered under `ShapeKind.Hull`'s id lookup (`Body.halfExtents.w`) — exercises
// `marshal.ts`'s `hullFromRegistry` path (the Hull branch `attachShape` doesn't otherwise reach). Faces/
// edges are placeholder-shaped (never read: `hullFromRegistry` feeds only `.verts` into the engine's own
// `createHull`, which rebuilds the hull structure itself); only the vertex positions need to be real.
const TETRA_ID = Hulls.register({
    name: "tumble-test-tetra",
    verts: [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    faces: [
        { normal: [0, 0, -1], offset: 0, verts: [0, 2, 1] },
        { normal: [0, -1, 0], offset: 0, verts: [0, 1, 3] },
        { normal: [-1, 0, 0], offset: 0, verts: [0, 3, 2] },
        { normal: [1, 1, 1], offset: 1, verts: [1, 2, 3] },
    ],
    edges: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
});

// The marshaling gate (tumble.md "The marshaling gate — dual-run hash equality"): the same scene
// built two ways — once through `TumblePlugin` on a headless `State` (reading a scene's authored `Body`
// entities), once by hand through the raw engine `World`/`Body` API with the identical literal values —
// stepped in lockstep, asserting per-step `hashWorldState` equality. Proves the ECS → tumble marshaling
// (shape dispatch, pose, mass/density, deterministic creation order) end to end with no fixture files. The
// two paths share NO code (the reference build below never imports `marshal.ts`), so a real ordering/shape/
// pose/mass bug in the plugin's marshaling diverges the hash.

const GRAVITY = -10;
const SUBSTEPS = 4; // must match TumblePlugin's own SUBSTEPS
const STEPS = 20;

type SceneBody = {
    shape: number;
    halfExtents: [number, number, number, number];
    pos: [number, number, number];
    quat: [number, number, number, number];
    mass: number;
    friction: number;
};

// three shapes + a static floor, deliberately in a fixed creation order (the order both runs replay).
const SCENE: SceneBody[] = [
    {
        shape: ShapeKind.Box,
        halfExtents: [0.5, 0.5, 0.5, 0],
        pos: [0, 5, 0],
        quat: [0, 0, 0, 1],
        mass: 1,
        friction: 0.5,
    },
    {
        shape: ShapeKind.Sphere,
        halfExtents: [0, 0, 0, 0.4],
        pos: [1.2, 6, 0.3],
        quat: [0, 0, 0, 1],
        mass: 2,
        friction: 0.3,
    },
    {
        shape: ShapeKind.Capsule,
        halfExtents: [0, 0.5, 0, 0.25],
        pos: [-1, 7, 0],
        quat: [0.13, 0.02, 0, 0.99],
        mass: 0.7,
        friction: 0.6,
    },
    {
        shape: ShapeKind.Box,
        halfExtents: [10, 0.5, 10, 0],
        pos: [0, 0, 0],
        quat: [0, 0, 0, 1],
        mass: 0, // static floor
        friction: 0.8,
    },
    {
        shape: ShapeKind.Hull,
        halfExtents: [0, 0, 0, TETRA_ID],
        pos: [2, 6, -1],
        quat: [0, 0, 0, 1],
        mass: 1.5,
        friction: 0.4,
    },
    // the constraint rig (indices 5-8): a static anchor carrying a hanging spring block, a spherical
    // pendulum bob, and a welded arm — one of each Spring/Joint mapping in the hash gate.
    {
        shape: ShapeKind.Box,
        halfExtents: [0.1, 0.1, 0.1, 0],
        pos: [6, 10, 0],
        quat: [0, 0, 0, 1],
        mass: 0, // static anchor
        friction: 0.5,
    },
    {
        shape: ShapeKind.Box,
        halfExtents: [0.5, 0.5, 0.5, 0],
        pos: [6, 6, 0],
        quat: [0, 0, 0, 1],
        mass: 2, // spring block
        friction: 0.5,
    },
    {
        shape: ShapeKind.Box,
        halfExtents: [0.25, 0.25, 0.25, 0],
        pos: [8, 10, 0],
        quat: [0, 0, 0, 1],
        mass: 1, // pendulum bob
        friction: 0.5,
    },
    {
        shape: ShapeKind.Box,
        halfExtents: [0.25, 0.25, 0.25, 0],
        pos: [6, 9, 0],
        quat: [0, 0, 0, 1],
        mass: 1, // welded arm
        friction: 0.5,
    },
];

// SCENE indices of the constraint rig above — the plugin authors these as Spring/Joint entities, the
// reference creates the mapped tumble joints directly, in the same order (springs, then joints).
const ANCHOR = 5;
const BLOCK = 6;
const BOB = 7;
const ARM = 8;
const SPRING_STIFFNESS = 100;
const SPRING_REST = 4;

// the reference build: raw engine API, literal values, no shared code with `marshal.ts` or `joints.ts`.
function buildReference(): World {
    const world = new World({ gravity: { x: 0, y: GRAVITY, z: 0 } });
    const handles: TumbleBody[] = [];
    for (const b of SCENE) {
        const tb = world.createBody({
            type: b.mass > 0 ? BodyType.Dynamic : BodyType.Kinematic,
            position: { x: b.pos[0], y: b.pos[1], z: b.pos[2] },
            rotation: { v: { x: b.quat[0], y: b.quat[1], z: b.quat[2] }, s: b.quat[3] },
        });
        const baseMaterial = { ...defaultSurfaceMaterial(), friction: b.friction };
        if (b.shape === ShapeKind.Sphere) {
            const radius = b.halfExtents[3];
            const volume = (4 / 3) * Math.PI * radius ** 3;
            const density = b.mass > 0 ? b.mass / volume : defaultShapeDef().density;
            tb.createSphere({ baseMaterial, density }, { center: { x: 0, y: 0, z: 0 }, radius });
        } else if (b.shape === ShapeKind.Capsule) {
            const hy = b.halfExtents[1];
            const radius = b.halfExtents[3];
            const volume = Math.PI * radius * radius * (2 * hy) + (4 / 3) * Math.PI * radius ** 3;
            const density = b.mass > 0 ? b.mass / volume : defaultShapeDef().density;
            tb.createCapsule(
                { baseMaterial, density },
                { center1: { x: 0, y: -hy, z: 0 }, center2: { x: 0, y: hy, z: 0 }, radius },
            );
        } else {
            // Box (a hull of `halfExtents`) and Hull (a registered hull, looked up by `halfExtents.w` —
            // the same `Hulls` registry `marshal.ts` reads) share the engine's `createHull` call.
            const hull =
                b.shape === ShapeKind.Hull
                    ? (createHull(
                          Hulls.get(Hulls.name(b.halfExtents[3]) ?? "")!.verts.map(([x, y, z]) => ({
                              x,
                              y,
                              z,
                          })),
                          4,
                      ) as HullData)
                    : makeBoxHull(b.halfExtents[0], b.halfExtents[1], b.halfExtents[2]);
            const density = b.mass > 0 ? b.mass / hull.volume : defaultShapeDef().density;
            tb.createHull({ baseMaterial, density }, hull);
        }
        handles.push(tb);
    }
    // the constraint rig, mirroring joints.ts's mapping with independent literals: Spring →
    // DistanceJoint-with-spring (hertz = √(k/m_eff)/2π off the derived mass, critically damped),
    // spherical-mapped Joint, weld-mapped fixed Joint (identity relative rotation — both spawn
    // unrotated). Springs upload before joints (ConstraintSystem's order), so create in that order.
    const zero = { x: 0, y: 0, z: 0 };
    const identity = { v: { x: 0, y: 0, z: 0 }, s: 1 };
    world.createDistanceJoint(handles[ANCHOR], handles[BLOCK], {
        localFrameA: { p: zero, q: identity },
        localFrameB: { p: zero, q: identity },
        length: SPRING_REST,
        enableSpring: true,
        hertz: Math.sqrt(SPRING_STIFFNESS / handles[BLOCK].getMassData().mass) / (2 * Math.PI),
        dampingRatio: 1,
    });
    world.createSphericalJoint(handles[ANCHOR], handles[BOB], {
        localFrameA: { p: zero, q: identity },
        localFrameB: { p: { x: -2, y: 0, z: 0 }, q: identity },
    });
    world.createWeldJoint(handles[ANCHOR], handles[ARM], {
        localFrameA: { p: { x: 0, y: -1, z: 0 }, q: identity },
        localFrameB: { p: zero, q: identity },
        linearHertz: 0,
        angularHertz: 0,
        angularDampingRatio: 0,
    });
    return world;
}

// the live State of the current test — afterEach disposes it so a FAILED assert can't leak the
// installed backend into the next test/file (the single-backend guard would then throw in warm;
// the class stage 3 fixed in physics/backend.test.ts). CharacterPlugin.dispose is a no-op for the
// marshaling test (no drive state) and required for the character test.
let liveState: State | null = null;
afterEach(() => {
    if (liveState) {
        CharacterPlugin.dispose?.(liveState);
        TumblePlugin.dispose?.(liveState);
    }
    liveState = null;
});

// Release the multithreaded worker pool warm()/init() boots (the wasm kernel is a process singleton) at
// file teardown, so its solve path doesn't leak into sibling engine test files that assume single-thread.
afterAll(shutdown);

async function buildScene(): Promise<State> {
    clear();
    const state = new State();
    liveState = state;
    register("body", Body, bodyTraits);
    register("spring", Spring, springTraits);
    register("joint", Joint, jointTraits);
    Slab.collect();
    TumblePlugin.initialize?.(state);
    await TumblePlugin.warm?.(state);
    attach(state, TumblePlugin);
    const eids: number[] = [];
    for (const b of SCENE) {
        const eid = state.create();
        state.add(eid, Body);
        Body.shape.set(eid, b.shape);
        Body.halfExtents.set(eid, ...b.halfExtents);
        Body.pos.set(eid, b.pos[0], b.pos[1], b.pos[2], 0);
        Body.quat.set(eid, ...b.quat);
        Body.mass.set(eid, b.mass);
        Body.friction.set(eid, b.friction);
        eids.push(eid);
    }
    // the constraint rig as authored components — ConstraintSystem derives the defs and uploads them
    // on the first fixed tick (after SyncSystem created the bodies, before the solve).
    const spring = state.create();
    state.add(spring, Spring);
    Spring.a.set(spring, eids[ANCHOR]);
    Spring.b.set(spring, eids[BLOCK]);
    Spring.stiffness.set(spring, SPRING_STIFFNESS);
    Spring.rest.set(spring, SPRING_REST);
    const pin = state.create();
    state.add(pin, Joint);
    Joint.a.set(pin, eids[ANCHOR]);
    Joint.b.set(pin, eids[BOB]);
    Joint.rB.set(pin, -2, 0, 0, 0);
    const weld = state.create();
    state.add(weld, Joint);
    Joint.a.set(weld, eids[ANCHOR]);
    Joint.b.set(weld, eids[ARM]);
    Joint.rA.set(weld, 0, -1, 0, 0);
    Joint.stiffnessAng.set(weld, Number.POSITIVE_INFINITY);
    return state;
}

describe("tumble marshaling gate", () => {
    test("TumblePlugin reproduces the raw engine API bit-exactly, per step", async () => {
        // the wasm kernel is a singleton (ONE live resident region, tumble.md "Singleton,
        // single-live-world") — the two runs can't be live at once, so "lockstep" means: run the
        // reference to completion first (recording every step's hash), destroy it, THEN run the plugin
        // to completion (recording the same), and compare the two per-step hash sequences afterward.
        const reference = buildReference();
        const referenceHashes: bigint[] = [];
        for (let i = 0; i < STEPS; i++) {
            reference.step(Time.FIXED_DT, SUBSTEPS);
            referenceHashes.push(hashWorldState(reference.state));
        }
        reference.destroy();

        const state = await buildScene();
        const pluginHashes: bigint[] = [];
        for (let i = 0; i < STEPS; i++) {
            state.step(Time.FIXED_DT);
            // the plugin's sync system creates its bodies on this same first fixed tick (before the
            // solve, `SyncSystem`'s `before: [StepSystem]`), so the first recorded hash is already
            // post-creation-and-step, matching the reference's first recorded hash.
            expect(Tumble.world).not.toBeNull();
            pluginHashes.push(hashWorldState(Tumble.world!.state));
        }

        expect(pluginHashes).toEqual(referenceHashes);
    });
});

describe("character under the tumble backend", () => {
    test("the shared CPU sweep grounds, drives, and uploads through the tumble handle", async () => {
        // the backend-neutral collide-and-slide (standard/character) run against TumblePlugin: the
        // sweep reads candidates through `Physics.backend.readBody` and uploads the swept pose through
        // `setKinematic` — the exact seams the stage-2 substrate established, now under the second
        // backend. Sweep correctness itself is oracle-gated (character-sweep.oracle.ts); this pins the
        // tumble INTEGRATION: ground on a tumble-marshaled floor, drive, and the kinematic round-trip.
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("character", Character, CharacterPlugin.traits?.Character);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);
        attach(state, CharacterPlugin);

        const floor = state.create();
        state.add(floor, Body);
        Body.halfExtents.set(floor, 10, 0.5, 10, 0);
        Body.mass.set(floor, 0);

        const char = state.create();
        state.add(char, Body);
        Body.shape.set(char, ShapeKind.Capsule);
        Body.halfExtents.set(char, 0, 0.5, 0, 0.3);
        Body.pos.set(char, 0, 2, 0, 0);
        Body.mass.set(char, 0);
        state.add(char, Character);

        // drop to rest: floor top 0.5 + capsule half 0.5 + radius 0.3 ⇒ grounded at center y = 1.3
        stepFor(state, 1.5);
        expect(grounded(char)).toBe(true);
        const p: [number, number, number] = [0, 0, 0];
        expect(pose(char, p)).toBe(true);
        expect(Math.abs(p[1] - 1.3)).toBeLessThan(0.02);

        // drive at 2 m/s for 1 s — the sweep's horizontal velocity is direct, so x advances ~2 m
        for (let i = 0; i < 60; i++) {
            move(char, 2, 0);
            state.step(Time.FIXED_DT);
        }
        pose(char, p);
        expect(p[0]).toBeGreaterThan(1.5);

        // the kinematic upload reached tumble: the backend's pose tracks the controller's. x may lead
        // by one tick's advance (tumble integrates a kinematic body's velocity after the upload; the
        // next sweep overwrites it) — y is tight since the grounded realized velocity is 0.
        const live = Physics.backend?.readBody(char);
        expect(live).not.toBeNull();
        expect(Math.abs((live?.pos[0] ?? 0) - p[0])).toBeLessThan(2.5 * Time.FIXED_DT);
        expect(Math.abs((live?.pos[1] ?? 0) - p[1])).toBeLessThan(1e-3);
    });

    test("a recycled character eid rebuilds its controller state and drops stale drive input", async () => {
        // the drive-map realias: syncStates keys `states` on the authored [Character, Body] set by an FNV
        // signature, and a same-update destroy+create with identical tuning hashes to the SAME signature —
        // so the destroyed character's swept pose + drive input would survive onto the new one. The fix
        // folds state.stamp into the signature (making the realias visible) and rebuilds on a stamp mismatch.
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("character", Character, CharacterPlugin.traits?.Character);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);
        attach(state, CharacterPlugin);

        const floor = state.create();
        state.add(floor, Body);
        Body.halfExtents.set(floor, 40, 0.5, 40, 0);
        Body.mass.set(floor, 0);

        // character A spawns at the origin and drives +x for a second, so its swept pose drifts well away
        // from any spawn pose; then a large stale move input is left keyed to its eid.
        const charA = state.create();
        state.add(charA, Body);
        Body.shape.set(charA, ShapeKind.Capsule);
        Body.halfExtents.set(charA, 0, 0.5, 0, 0.3);
        Body.pos.set(charA, 0, 2, 0, 0);
        Body.mass.set(charA, 0);
        state.add(charA, Character);
        for (let i = 0; i < 60; i++) {
            move(charA, 2, 0);
            state.step(Time.FIXED_DT);
        }
        const a: [number, number, number] = [0, 0, 0];
        pose(charA, a);
        expect(a[0]).toBeGreaterThan(1.5); // drifted from spawn
        move(charA, 50, 0); // a stale drive input that must not carry to the recycled character

        // recycle the eid with a fresh character at x=20, identical tuning (same signature but for the stamp)
        state.destroy(charA);
        const charB = state.create();
        expect(charB).toBe(charA);
        state.add(charB, Body);
        Body.shape.set(charB, ShapeKind.Capsule);
        Body.halfExtents.set(charB, 0, 0.5, 0, 0.3);
        Body.pos.set(charB, 20, 2, 0, 0);
        Body.mass.set(charB, 0);
        state.add(charB, Character);

        // ten ticks with NO drive: a rebuilt state sits at the new spawn x≈20; the destroyed char's kept
        // state (x≈2) + stale 50 m/s input would instead march it to x≈10.
        for (let i = 0; i < 10; i++) state.step(Time.FIXED_DT);
        const b: [number, number, number] = [0, 0, 0];
        expect(pose(charB, b)).toBe(true);
        expect(Math.abs(b[0] - 20)).toBeLessThan(1);
    });

    test("drive input queued before a fresh character's first sync is kept, not evicted", async () => {
        // the realias eviction must not eat a legitimate spawn-tick input: `moves` persists as held
        // input, so an entry set between create() and the first fixed tick belongs to THIS character.
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("character", Character, CharacterPlugin.traits?.Character);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);
        attach(state, CharacterPlugin);

        const floor = state.create();
        state.add(floor, Body);
        Body.halfExtents.set(floor, 10, 0.5, 10, 0);
        Body.mass.set(floor, 0);

        const char = state.create();
        state.add(char, Body);
        Body.shape.set(char, ShapeKind.Capsule);
        Body.halfExtents.set(char, 0, 0.5, 0, 0.3);
        Body.pos.set(char, 0, 2, 0, 0);
        Body.mass.set(char, 0);
        state.add(char, Character);
        move(char, 2, 0); // queued before the first fixed tick ever runs

        for (let i = 0; i < 10; i++) state.step(Time.FIXED_DT);
        const p: [number, number, number] = [0, 0, 0];
        expect(pose(char, p)).toBe(true);
        expect(p[0]).toBeGreaterThan(0.2); // the held input drove the char; eviction would leave x at 0
    });
});

describe("Tumble.body handle accessor", () => {
    test("returns the marshaled handle after a sync tick, null before / for a non-body", async () => {
        // the escape-hatch eid↔handle bridge (a ragdoll wiring cone/twist/filter joints between named
        // bodies reaches each `TumbleBody` this way, then hands it to `Tumble.world.create*Joint`). The
        // handle only exists once SyncSystem has marshaled the entity — its first `fixed` tick.
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);

        const eid = state.create();
        state.add(eid, Body);
        Body.pos.set(eid, 0, 5, 0, 0);

        // before the first fixed tick the body isn't marshaled yet
        expect(Tumble.body(eid)).toBeNull();
        // a non-Body eid never resolves
        const bare = state.create();
        expect(Tumble.body(bare)).toBeNull();

        state.step(Time.FIXED_DT);

        const handle = Tumble.body(eid);
        expect(handle).not.toBeNull();
        // it's the live handle: its position reflects the marshaled spawn pose (before gravity has pulled
        // it far), and it round-trips the same object on a second lookup
        expect(handle?.getPosition().y).toBeGreaterThan(4);
        expect(Tumble.body(eid)).toBe(handle);

        // freed with the entity: destroy drops the mapping on the next tick
        state.destroy(eid);
        state.step(Time.FIXED_DT);
        expect(Tumble.body(eid)).toBeNull();
    });
});

describe("same-update destroy+create realias", () => {
    test("a box destroyed and a sphere created at the recycled eid marshals the new body", async () => {
        // the same-update destroy+create identity bug (tumble.md "Eid presence is not identity"): a box
        // is marshaled, then destroyed and its eid recycled by a NEW sphere Body in one update. SyncSystem
        // keys create/destroy on presence alone, so it neither sweeps the eid (it still has Body) nor
        // re-marshals it (it's still in `bodies`) — the old box handle survives entirely. The fix (1b)
        // compares `state.stamp(eid)` against the stored stamp and rebinds on mismatch.
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);

        const box = state.create();
        state.add(box, Body);
        Body.shape.set(box, ShapeKind.Box);
        Body.halfExtents.set(box, 0.5, 0.5, 0.5, 0);
        Body.pos.set(box, 0, 5, 0, 0);
        Body.mass.set(box, 1);
        state.step(Time.FIXED_DT); // marshals the box into `bodies`, falls a hair

        // same update: destroy the box, recycle its eid with a sphere at a distinct pos + mass
        state.destroy(box);
        const sphere = state.create();
        expect(sphere).toBe(box); // LIFO recycle hands back the same slot
        state.add(sphere, Body);
        Body.shape.set(sphere, ShapeKind.Sphere);
        Body.halfExtents.set(sphere, 0, 0, 0, 0.5);
        Body.pos.set(sphere, 10, 8, 0, 0);
        Body.mass.set(sphere, 5);
        state.step(Time.FIXED_DT);

        // the live tumble handle must be the NEW sphere: gravity is -y only, so x stays at the sphere's
        // authored 10 (the old box fell straight down from x=0), and its mass is the sphere's 5, not 1.
        const handle = Tumble.body(sphere);
        expect(handle).not.toBeNull();
        expect(handle?.getPosition().x).toBeCloseTo(10, 1);
        expect(handle?.getMass()).toBeCloseTo(5, 1);
    });

    test("the recycled eid drops the destroyed body's kinematic-prev pose", async () => {
        // kinPrev is setKinematic's per-eid previous pose — it derives a platform's velocity from the
        // per-step delta. It must be evicted with the old body, or the first setKinematic on the recycled
        // body derives a bogus velocity from the destroyed body's last pose (same realias class).
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);

        // a kinematic (mass 0) platform at x=10; the first setKinematic seeds its kinPrev to that pose.
        const platA = state.create();
        state.add(platA, Body);
        Body.shape.set(platA, ShapeKind.Box);
        Body.halfExtents.set(platA, 0.5, 0.5, 0.5, 0);
        Body.pos.set(platA, 10, 0, 0, 0);
        Body.mass.set(platA, 0);
        state.step(Time.FIXED_DT); // marshals the platform
        Physics.backend?.setKinematic(platA, [10, 0, 0], [0, 0, 0, 1]); // kinPrev := [10,0,0]

        // recycle the eid with a fresh kinematic body at the origin, one update
        state.destroy(platA);
        const platB = state.create();
        expect(platB).toBe(platA);
        state.add(platB, Body);
        Body.shape.set(platB, ShapeKind.Box);
        Body.halfExtents.set(platB, 0.5, 0.5, 0.5, 0);
        Body.pos.set(platB, 0, 0, 0, 0);
        Body.mass.set(platB, 0);
        state.step(Time.FIXED_DT); // SyncSystem re-marshals + evicts the stale kinPrev

        // the first setKinematic on the recycled body re-seeds its prev to the current pose (derived
        // velocity 0), not derive (0 − 10)/dt ≈ −600 from the destroyed platform's kinPrev.
        Physics.backend?.setKinematic(platB, [0, 0, 0], [0, 0, 0, 1]);
        const live = Physics.backend?.readBody(platB);
        expect(live).not.toBeNull();
        expect(Math.abs(live?.vel[0] ?? 0)).toBeLessThan(1e-6);
    });

    test("an authored Joint rebinds to a body recycled at its endpoint eid in one update", async () => {
        // The full jointed-realias chain on the real backend, end to end: SyncSystem (before
        // ConstraintSystem) destroys the stale handle on a stamp mismatch, tumble cascades the joint off
        // the destroyed body, and ConstraintSystem's signature — folding each endpoint's `state.stamp` —
        // re-uploads so `syncJoints`' isValid() check drops the cascaded joint and rebuilds it against the
        // NEW handle. Neutralize the stamp fold in springSignature/jointSignature (physics/index.ts) and the
        // signature never changes: the joint never re-uploads, the new bob's cascaded joint is never rebuilt,
        // and it free-falls. The link-by-link mechanism is proven with stubs; this composes it on tumble.
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);

        const Pivot: [number, number, number] = [0, 10, 0];
        const Length = 2; // |rB|: the spherical joint holds the bob center this far from the pivot
        const distToPivot = (eid: number): number => {
            const p = Tumble.body(eid)!.getPosition();
            return Math.hypot(p.x - Pivot[0], p.y - Pivot[1], p.z - Pivot[2]);
        };

        // a static anchor at the pivot point
        const anchor = state.create();
        state.add(anchor, Body);
        Body.shape.set(anchor, ShapeKind.Box);
        Body.halfExtents.set(anchor, 0.25, 0.25, 0.25, 0);
        Body.pos.set(anchor, Pivot[0], Pivot[1], Pivot[2], 0);
        Body.mass.set(anchor, 0);

        // a dynamic bob offset +x from the pivot; rB = [-LENGTH,0,0] pins its local point onto the pivot,
        // coincident at spawn — a horizontal pendulum that swings down and hangs at LENGTH below the pivot
        const bob = state.create();
        state.add(bob, Body);
        Body.shape.set(bob, ShapeKind.Box);
        Body.halfExtents.set(bob, 0.25, 0.25, 0.25, 0);
        Body.pos.set(bob, Length, 10, 0, 0);
        Body.mass.set(bob, 1);

        const joint = state.create();
        state.add(joint, Joint);
        Joint.a.set(joint, anchor);
        Joint.b.set(joint, bob);
        Joint.rA.set(joint, 0, 0, 0, 0);
        Joint.rB.set(joint, -Length, 0, 0, 0);
        Joint.stiffnessAng.set(joint, 0); // spherical (free rotation)

        // the joint constrains the bob: it swings down but its center stays at LENGTH from the pivot (it
        // rides the sphere of radius LENGTH around the pivot, so y never drops below PIVOT.y - LENGTH = 8)
        stepFor(state, 1.5);
        expect(distToPivot(bob)).toBeGreaterThan(1.7);
        expect(distToPivot(bob)).toBeLessThan(2.3);
        expect(Tumble.body(bob)!.getPosition().y).toBeGreaterThan(7.5);

        // realias: destroy the bob and recreate a DISTINCT bob at the recycled eid in ONE update — the eid
        // never leaves Body membership, so only the create-stamp fold in the joint signature can see it
        state.destroy(bob);
        const bob2 = state.create();
        expect(bob2).toBe(bob); // LIFO recycle hands back the same slot
        state.add(bob2, Body);
        Body.shape.set(bob2, ShapeKind.Box);
        Body.halfExtents.set(bob2, 0.25, 0.25, 0.25, 0);
        Body.pos.set(bob2, Length, 10, 2, 0); // distinct spawn (z=2), a mild yank onto the joint
        Body.mass.set(bob2, 1);

        stepFor(state, 1.5);

        // the joint constrains the NEW occupant at its new position: its center holds at LENGTH from the
        // pivot. A dangling (un-rebuilt) joint leaves the new bob in free fall — after 1.5s dist ≈ 11 and
        // y ≈ -1, so both thresholds separate a rebound joint from a suppressed re-upload by a wide margin.
        expect(distToPivot(bob2)).toBeLessThan(2.5);
        expect(Tumble.body(bob2)!.getPosition().y).toBeGreaterThan(7.5);
    });
});

describe("kinematic sleep + teleport", () => {
    test("a teleport wakes a slept kinematic body so its render slot tracks the new pose", async () => {
        // a kinematic body that sleeps (parked, v=0) then teleports via setKinematic: setTransform never
        // wakes and setLinearVelocity wakes only on a nonzero velocity, so a zero-velocity teleport left the
        // body asleep — it emits no move event, so the compose firehose slot (movedThisTick) held the stale
        // pose while readBody saw the new one (an AVBD swap-parity divergence: AVBD composes every live eid).
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);

        const plat = state.create();
        state.add(plat, Body);
        Body.shape.set(plat, ShapeKind.Box);
        Body.halfExtents.set(plat, 0.5, 0.5, 0.5, 0);
        Body.pos.set(plat, 0, 0, 0, 0);
        Body.mass.set(plat, 0);
        state.step(Time.FIXED_DT); // marshal

        const handle = Tumble.body(plat);
        expect(handle).not.toBeNull();
        // park it: unmoved with zero velocity, the kinematic body falls asleep
        stepFor(state, 2.0);
        expect(handle?.isAwake()).toBe(false);

        // teleport it to x=10 (teleport ⇒ derived velocity 0, so setLinearVelocity won't wake it)
        Physics.backend?.setKinematic(plat, [10, 0, 0], [0, 0, 0, 1], true);
        // readBody sees the new pose immediately (setTransform applied)
        expect(Physics.backend?.readBody(plat)?.pos[0]).toBeCloseTo(10, 3);

        const writes = new Map<number, Float32Array>();
        const compute = Compute as unknown as { device: GPUDevice | undefined };
        const priorDevice = compute.device;
        compute.device = {
            queue: {
                writeBuffer: (_b: GPUBuffer, offset: number, data: Float32Array) => {
                    writes.set(offset, data.slice());
                },
            },
        } as unknown as GPUDevice;
        try {
            state.step(Time.FIXED_DT); // the wake makes the solver report the teleport as a move event
            Physics.backend?.compose(undefined as unknown as GPUCommandEncoder, {} as GPUBuffer, 1);
            const rec = writes.get(plat * 48);
            expect(rec).toBeDefined();
            expect(rec?.[0]).toBeCloseTo(10, 1); // the render firehose slot tracks the teleported pose
        } finally {
            compute.device = priorDevice;
        }
    });
});

describe("unregistered hull marshaling", () => {
    test("a body with a bogus hull id warns + skips, the rest of the scene still simulates", async () => {
        // an unregistered hull id must not throw out of `hullFromRegistry` through SyncSystem — a throw
        // there pauses the whole sync loop (the scheduler pauses a throwing system) so nothing else
        // marshals. Marshaling warns + skips just that body (joints.ts's skip-a-bad-constraint convention),
        // the rest of the scene still simulates, and the skip is sticky per stamp so it never re-warns.
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);

        const floor = state.create();
        state.add(floor, Body);
        Body.halfExtents.set(floor, 10, 0.5, 10, 0);
        Body.mass.set(floor, 0);

        const box = state.create();
        state.add(box, Body);
        Body.pos.set(box, 0, 5, 0, 0);
        Body.mass.set(box, 1);

        // a body referencing a hull id that was never registered
        const bad = state.create();
        state.add(bad, Body);
        Body.shape.set(bad, ShapeKind.Hull);
        Body.halfExtents.set(bad, 0, 0, 0, 9999);
        Body.pos.set(bad, 3, 5, 0, 0);
        Body.mass.set(bad, 1);

        const warns: string[] = [];
        const orig = console.warn;
        console.warn = (...a: unknown[]) => warns.push(a.join(" "));
        try {
            state.step(Time.FIXED_DT); // marshal tick: the bad hull warns + is skipped, floor + box marshal
            const warnCount = warns.length;
            stepFor(state, 0.5); // keep simulating — the skip is sticky, so it does NOT re-warn every tick
            expect(warns.length).toBe(warnCount);
        } finally {
            console.warn = orig;
        }

        // warned once about the missing hull id (the diagnostic names the id, not the brand)
        expect(warns.some((w) => w.includes("no hull registered") && w.includes("9999"))).toBe(
            true,
        );
        // the bad body never marshaled — no handle exists for it
        expect(Tumble.body(bad)).toBeNull();
        // the rest of the scene still simulates: the normal box marshaled and fell under gravity
        const live = Physics.backend?.readBody(box);
        expect(live).not.toBeNull();
        expect(live?.pos[1]).toBeLessThan(5);
    });
});

describe("compose covers static bodies", () => {
    test("a static body's firehose record is written on its marshal tick, a dynamic one every move", async () => {
        // tumble reports every NEW body in its first tick's move events and never again for a static —
        // so the static's spawn record lands in the transforms firehose exactly once, and it only STAYS
        // there because the Transform compose is membership-gated (an ungated scatter would stomp the
        // Body slot with the unset Transform slab next frame — the invisible-floor bug). This pins the
        // CPU half: the marshal-tick write happens, statics then go quiet, dynamics keep refreshing.
        // Compose's GPU write is one writeBuffer per record — capture it through a stub device.
        clear();
        const state = new State();
        liveState = state;
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        TumblePlugin.initialize?.(state);
        await TumblePlugin.warm?.(state);
        attach(state, TumblePlugin);

        const floor = state.create();
        state.add(floor, Body);
        Body.pos.set(floor, 0, -0.1, 0, 0);
        Body.halfExtents.set(floor, 7, 0.1, 7, 0);
        Body.mass.set(floor, 0);
        const box = state.create();
        state.add(box, Body);
        Body.pos.set(box, 0, 5, 0, 0);

        const writes = new Map<number, Float32Array>();
        const compute = Compute as unknown as { device: GPUDevice | undefined };
        const priorDevice = compute.device;
        compute.device = {
            queue: {
                writeBuffer: (_b: GPUBuffer, offset: number, data: Float32Array) => {
                    writes.set(offset, data.slice());
                },
            },
        } as unknown as GPUDevice;
        try {
            state.step(Time.FIXED_DT);
            const buffer = {} as GPUBuffer;
            Physics.backend?.compose(undefined as unknown as GPUCommandEncoder, buffer, 1);

            // the static floor composed its spawn record (pos + Xform render scale = 2·halfExtents)
            const rec = writes.get(floor * 48);
            expect(rec).toBeDefined();
            expect(rec?.[1]).toBeCloseTo(-0.1);
            expect(rec?.[8]).toBeCloseTo(14);
            expect(rec?.[9]).toBeCloseTo(0.2);
            // the falling box composed too
            expect(writes.get(box * 48)).toBeDefined();

            // later ticks keep the falling box's record refreshing, and the never-moving floor's
            // record — whenever tumble stops reporting it — persists only because the membership-gated
            // Transform compose leaves non-Transform slots alone (the gym ragdoll floor check pins that
            // half on the real GPU)
            stepFor(state, 1.0);
            writes.clear();
            Physics.backend?.compose(undefined as unknown as GPUCommandEncoder, buffer, 1);
            expect(writes.get(box * 48)).toBeDefined();
            const still = writes.get(floor * 48);
            if (still) expect(still[1]).toBeCloseTo(-0.1); // if reported, it's still the spawn pose
        } finally {
            compute.device = priorDevice;
        }
    });
});

describe("tumble/core mirror", () => {
    test("core re-exports the engine barrel minus shutdown, plus nlerpShortest", async () => {
        const engine = await import("./engine");
        const core = await import("./core");
        const engineKeys = new Set(Object.keys(engine));
        const coreKeys = new Set(Object.keys(core));
        expect([...engineKeys].filter((k) => !coreKeys.has(k))).toEqual(["shutdown"]);
        expect([...coreKeys].filter((k) => !engineKeys.has(k))).toEqual(["nlerpShortest"]);
    });
});
