import {
    AmbientLight,
    Body,
    Camera,
    CameraMode,
    Color,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    type Plugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    ShapeKind,
    SlabPlugin,
    type State,
    type System,
    Transform,
    TransformsPlugin,
    Tumble,
    TumblePlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// queries — the tumble spatial-query surface (`castRayClosest`, `castShape`, `overlapAABB` on `Tumble.world`,
// the escape hatch past the substrate). One deterministic scene of fixed kinematic obstacles gates all three
// in one run: a ray finds the closest hit on a sphere, a swept sphere resolves to its first contact fraction,
// and a broad-phase box overlap counts a known grid. The obstacles are substrate `Body` entities so they
// render; `spin` (off headless, so the assert stays deterministic) rotates them for the live view.

const RAY_TARGET: [number, number, number] = [-4, 4, 0]; // sphere, radius 1 → top at y 5
const SHAPE_TARGET: [number, number, number] = [4, 4, 0]; // box, half 1 → front face at z −1
const GRID = 3; // 3×3 overlap grid at y 1
const SHAPE_RADIUS = 0.3;
const SHAPE_START: [number, number, number] = [4, 4, -6];
const SHAPE_TRAVEL = 10; // +z sweep distance

let rayTargetEid = -1;
let gridEids: number[] = [];

function kinematic(
    state: State,
    x: number,
    y: number,
    z: number,
    shape: number,
    half: [number, number, number],
    radius: number,
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, shape);
    Body.pos.set(eid, x, y, z, 0);
    Body.halfExtents.set(eid, half[0], half[1], half[2], radius);
    Body.mass.set(eid, 0); // kinematic — never driven, so it holds its spawn pose (deterministic queries)
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// spin the obstacles in place for the live view; skipped headless so the query targets stay put.
const spinner: System = {
    name: "queries-spin",
    group: "simulation",
    update(state: State) {
        for (const eid of state.query([Body])) {
            Tumble.body(eid)?.setAngularVelocity({ x: 0.4, y: 0.8, z: 0.3 });
        }
    },
};

const scenario: Scenario = {
    name: "queries",
    params: [{ key: "spin", type: "bool", default: false }],

    async build(_canvas, p: Params) {
        const { state, dispose } = await run({
            defaults: false,
            capacity: 64,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                TumblePlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ] as Plugin[],
        });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);

        rayTargetEid = kinematic(
            state,
            RAY_TARGET[0],
            RAY_TARGET[1],
            RAY_TARGET[2],
            ShapeKind.Sphere,
            [0, 0, 0],
            1,
            [0.85, 0.6, 0.4],
        );
        kinematic(
            state,
            SHAPE_TARGET[0],
            SHAPE_TARGET[1],
            SHAPE_TARGET[2],
            ShapeKind.Box,
            [1, 1, 1],
            0,
            [0.5, 0.55, 0.85],
        );
        gridEids = [];
        for (let gx = 0; gx < GRID; gx++) {
            for (let gz = 0; gz < GRID; gz++) {
                gridEids.push(
                    kinematic(
                        state,
                        gx * 2 - 2,
                        1,
                        gz * 2 - 2,
                        ShapeKind.Box,
                        [0.4, 0.4, 0.4],
                        0,
                        [0.5, 0.7, 0.6],
                    ),
                );
            }
        }

        if (p.spin) state.addSystem(spinner);

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, 0.5);
        Orbit.pitch.set(cam, 0.35);
        Orbit.distance.set(cam, 22);

        await frames(3);

        return {
            state,
            dispose() {
                rayTargetEid = -1;
                gridEids = [];
                dispose();
            },
        };
    },

    assert(): Promise<Check[]> {
        const world = Tumble.world;
        if (!world)
            return Promise.resolve([{ name: "queries", pass: false, detail: "no tumble world" }]);
        const checks: Check[] = [];

        // ray: closest hit on the sphere, at its top (y 5), and it's the ray target body.
        // Identity is by-value: the marshaled body carries its eid as userData (a query mints a
        // fresh Body wrapper each call, so `===` against a cached handle never holds).
        const r = world.castRayClosest(
            { x: RAY_TARGET[0], y: 10, z: RAY_TARGET[2] },
            { x: 0, y: -8, z: 0 },
        );
        const hitEid = r.hit && r.shape ? r.shape.getBody().getUserData() : undefined;
        checks.push({
            name: "castRayClosest hits the sphere top (closest hit)",
            pass: r.hit && !!r.shape && hitEid === rayTargetEid && Math.abs(r.point.y - 5) < 0.3,
            detail: r.hit
                ? `hit y ${r.point.y.toFixed(3)} (expect ~5), eid ${hitEid} (expect ${rayTargetEid})`
                : "no hit",
        });

        // shape cast: swept sphere resolves to the box front face, fraction ~0.47
        let fraction = 1;
        let scHit = false;
        world.castShape(
            { x: SHAPE_START[0], y: SHAPE_START[1], z: SHAPE_START[2] },
            { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: SHAPE_RADIUS },
            { x: 0, y: 0, z: SHAPE_TRAVEL },
            (h) => {
                fraction = h.fraction;
                scHit = true;
                return h.fraction;
            },
        );
        checks.push({
            name: "castShape resolves to the first contact fraction",
            pass: scHit && fraction > 0.35 && fraction < 0.6,
            detail: scHit ? `fraction ${fraction.toFixed(3)} (expect ~0.47)` : "no hit",
        });

        // overlap: the broad-phase box counts exactly the 3×3 grid, nothing else
        let count = 0;
        world.overlapAABB(
            { lowerBound: { x: -3, y: 0, z: -3 }, upperBound: { x: 3, y: 2, z: 3 } },
            () => {
                count++;
                return true;
            },
        );
        checks.push({
            name: "overlapAABB counts the known grid",
            pass: count === gridEids.length,
            detail: `overlapping ${count} (expect ${gridEids.length})`,
        });
        return Promise.resolve(checks);
    },

    live(): string {
        const world = Tumble.world;
        if (!world) return "queries — warming";
        const r = world.castRayClosest(
            { x: RAY_TARGET[0], y: 10, z: RAY_TARGET[2] },
            { x: 0, y: -8, z: 0 },
        );
        return `queries — ray ${r.hit ? `hit y ${r.point.y.toFixed(2)}` : "miss"}, grid ${gridEids.length}`;
    },
};

register(scenario);
