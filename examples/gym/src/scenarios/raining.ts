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
    Physics,
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
    TumblePlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// raining — the streaming-spawn stress: bodies rain onto a pile continuously and the oldest are recycled once
// the live count hits the cap, so the world churns at a steady body budget (the debris / particle / projectile
// pattern — bounded, not unbounded growth). It gates the substrate's live create + destroy path (`state.create`
// / `state.destroy` → the tumble backend's marshal / unmarshal) under constant load, and benchmarks stepping a
// full pile while it turns over. Deterministic: a seeded PRNG places every drop, so headless runs the same pile.

const SPAWN_EVERY = 2; // fixed ticks between drops

let queue: number[] = []; // FIFO of live body eids; the head is recycled when the cap is reached
let spawned = 0;
let cap = 60;
let seed = 0x9e3779b9;

function rand(): number {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function drop(state: State): void {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Box);
    Body.pos.set(eid, (rand() - 0.5) * 6, 12, (rand() - 0.5) * 6, 0);
    Body.halfExtents.set(eid, 0.3, 0.3, 0.3, 0);
    Body.mass.set(eid, 1);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, 0.6 + 0.3 * rand(), 0.5, 0.4, 1);
    queue.push(eid);
    spawned++;
    if (queue.length > cap) state.destroy(queue.shift()!);
}

const spawner: System = {
    name: "raining-spawner",
    group: "fixed",
    update(state: State) {
        if (state.time.fixedTick % SPAWN_EVERY === 0) drop(state);
    },
};

const scenario: Scenario = {
    name: "raining",
    params: [
        { key: "count", type: "number", default: 60, min: 10, max: 300, step: 10, rebuild: true },
    ],

    async build(_canvas, p: Params) {
        cap = Math.max(10, (p.count as number) | 0);
        queue = [];
        spawned = 0;
        seed = 0x9e3779b9;

        const { state, dispose } = await run({
            defaults: false,
            capacity: cap + 16,
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

        const ground = state.create();
        state.add(ground, Body);
        Body.pos.set(ground, 0, 0, 0, 0);
        Body.halfExtents.set(ground, 6, 0.5, 6, 0);
        Body.mass.set(ground, 0);
        state.add(ground, Part);
        state.add(ground, Color);
        Color.rgba.set(ground, 0.4, 0.42, 0.46, 1);

        state.addSystem(spawner);

        const camEid = state.create();
        state.add(camEid, Transform);
        state.add(camEid, Camera);
        state.add(camEid, Sear);
        state.add(camEid, Orbit);
        Camera.mode.set(camEid, CameraMode.Perspective);
        Orbit.yaw.set(camEid, 0.5);
        Orbit.pitch.set(camEid, 0.35);
        Orbit.distance.set(camEid, 22);

        // run past the cap so recycling is exercised before the assert samples
        for (let i = 0; i < 2000 && spawned < cap * 3; i++) await frames(1);

        return {
            state,
            dispose() {
                queue = [];
                spawned = 0;
                dispose();
            },
        };
    },

    assert(): Promise<Check[]> {
        const backend = Physics.backend;
        if (!backend)
            return Promise.resolve([{ name: "raining", pass: false, detail: "no backend" }]);
        let finite = true;
        for (const eid of queue) {
            const b = backend.readBody(eid);
            if (!b?.pos.every(Number.isFinite)) finite = false;
        }
        return Promise.resolve([
            {
                name: "streaming stays at its cap (oldest bodies recycled)",
                pass: queue.length <= cap,
                detail: `live ${queue.length} (cap ${cap})`,
            },
            {
                name: "the stream turned over (spawned well past the cap)",
                pass: spawned > cap * 2,
                detail: `spawned ${spawned} total (cap ${cap})`,
            },
            {
                name: "every live body is finite",
                pass: finite,
                detail: `${queue.length} live bodies checked`,
            },
        ]);
    },

    live(): string {
        return `raining — ${queue.length}/${cap} live, ${spawned} spawned`;
    },
};

register(scenario);
