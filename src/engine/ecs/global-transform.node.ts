import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { attachCanvas, Camera, RenderPlugin } from "../../core/rendering";
import {
    Body,
    forwardRay,
    GlobalTransform,
    PhysicsPlugin,
    StepSystem,
    setKinematic,
} from "../../transitional/physics";
import { build } from "../app";
import { CanvasContext } from "../app/canvas.fixture";
import * as engine from "../index";
import { globalTransformTable, probeBuffer, Transform, u32 } from "../index";
import type { System } from "./scheduler";
import { Time } from "./scheduler";

setDefaultTimeout(CEILING.node);
if (typeof ResizeObserver === "undefined") {
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });
}
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

const SpawnedPlacement = { marker: u32 };

function bounded<T>(promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error("GlobalTransform readback exceeded 750 ms")),
            750,
        );
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}
test("GlobalTransform is an engine-owned public schema, independent of Physics", () => {
    expect(Reflect.get(engine, "GlobalTransform")).toBe(GlobalTransform);
    expect(Reflect.get(engine, "markGlobalTransformDiscontinuity")).toBeUndefined();
});

test("Transform placement lands in the fixed-tick GlobalTransform column and the renderer table", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const state = app.state;
        const eid = state.create();
        state.add(eid, Transform);
        const placement = state.of(Transform);
        placement.pos.set(eid, 12, 7, -3, 0);
        placement.scale.set(eid, 2, 3, 4, 0);
        attachTestCamera(state);
        state.step(Time.FIXED_DT);
        expect(state.has(eid, GlobalTransform)).toBe(true);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(12);
        expect(state.of(GlobalTransform).pos.y.get(eid)).toBe(7);
        expect(state.of(GlobalTransform).pos.z.get(eid)).toBe(-3);
        const table = globalTransformTable(state);
        const row = table.rowIndex(eid);
        expect(row).toBeGreaterThanOrEqual(0);
        const words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([2, 3, 4]);
    } finally {
        app.dispose();
    }
});

test("a Body writes scale as part of fixed-tick GlobalTransform instead of deriving it only in renderer rows", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const state = app.state;
        const eid = state.create();
        state.add(eid, Body);
        state.of(Body).pos.set(eid, 12, 7, -3, 0);
        state.of(Body).halfExtents.set(eid, 1, 2, 3, 0);
        state.of(Body).mass.set(eid, 0);
        attachTestCamera(state);
        state.step(Time.FIXED_DT);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(12);
        expect(Reflect.get(GlobalTransform, "scale")).toBeDefined();
        const scale = Reflect.get(state.of(GlobalTransform), "scale");
        expect([scale.x.get(eid), scale.y.get(eid), scale.z.get(eid)]).toEqual([2, 4, 6]);
        const table = globalTransformTable(state);
        const row = table.rowIndex(eid);
        const words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([2, 4, 6]);
    } finally {
        app.dispose();
    }
});

test("a physics camera query reads fixed-tick GlobalTransform without requiring Transform", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const state = app.state;
        const camera = state.registry.getComponent("camera");
        if (!camera) throw new Error("RenderPlugin must register Camera");
        const eid = state.create();
        state.add(eid, Body);
        state.add(eid, camera);
        state.of(Body).pos.set(eid, 12, 7, -3, 0);
        state.of(Body).mass.set(eid, 0);
        state.step(Time.FIXED_DT);
        expect(state.has(eid, Transform)).toBe(false);
        expect(forwardRay(state, eid)).toEqual({ origin: [12, 7, -3], dir: [0, 0, -1] });
    } finally {
        app.dispose();
    }
});

function addStaticBody(state: engine.State, eid: number, x: number): void {
    state.add(eid, Body);
    state.of(Body).pos.set(eid, x, 0, 0, 0);
    state.of(Body).mass.set(eid, 0);
}

function addTransform(state: engine.State, eid: number, x: number): void {
    state.add(eid, Transform);
    state.of(Transform).pos.set(eid, x, 0, 0, 0);
}

function attachTestCamera(state: engine.State): void {
    let context: CanvasContext;
    const canvas = {
        width: 32,
        height: 24,
        style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width: 32, height: 24 }),
    } as unknown as HTMLCanvasElement;
    context = new CanvasContext(canvas, 32, 24);
    const camera = state.create();
    state.add(camera, Transform);
    state.add(camera, Camera);
    state.of(Transform).pos.set(camera, 0, 0, 5, 0);
    attachCanvas(camera, canvas, state);
}

async function renderedX(
    state: engine.State,
    table: ReturnType<typeof globalTransformTable>,
    eid: number,
): Promise<number> {
    const row = table.rowIndex(eid);
    expect(row).toBeGreaterThanOrEqual(0);
    const result = await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }));
    return new Float32Array(result.bytes)[row * 12];
}

async function handoverApp(initial: "Body" | "Transform"): Promise<{
    app: Awaited<ReturnType<typeof build>>;
    handover(action: (state: engine.State, eid: number) => void): void;
}> {
    let action: ((state: engine.State, eid: number) => void) | undefined;
    let eid = -1;
    const handoverSystem: System = {
        group: "simulation",
        update(state) {
            const current = action;
            if (!current) return;
            action = undefined;
            current(state, eid);
        },
    };
    const app = await build({
        defaults: false,
        plugins: [PhysicsPlugin, { name: "Handover", systems: [handoverSystem] }],
        setup(state) {
            eid = state.create();
            if (initial === "Body") addStaticBody(state, eid, 10);
            else addTransform(state, eid, 10);
        },
    });
    app.state.step(Time.FIXED_DT);
    return { app, handover: (next) => (action = next) };
}

test("Body to Transform keeps its GlobalTransform row for a same-frame producer handover", async () => {
    const { app } = await handoverApp("Body");
    const { state } = app;
    const eid = [...state.query([Body])][0];
    const table = globalTransformTable(state);
    const row = table.rowIndex(eid);
    try {
        state.remove(eid, Body);
        addTransform(state, eid, 42);
        const bystander = state.create();
        addTransform(state, bystander, -9);
        state.step(Time.FIXED_DT);
        expect(table.rowIndex(eid)).toBe(row);
        expect(state.has(eid, GlobalTransform)).toBe(true);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(42);
    } finally {
        app.dispose();
    }
});

test("Transform to Body keeps its GlobalTransform row for a same-frame producer handover", async () => {
    const { app } = await handoverApp("Transform");
    const { state } = app;
    const eid = [...state.query([Transform])][0];
    const table = globalTransformTable(state);
    const row = table.rowIndex(eid);
    try {
        state.remove(eid, Transform);
        const bystander = state.create();
        addTransform(state, bystander, -9);
        addStaticBody(state, eid, 42);
        state.step(Time.FIXED_DT);
        expect(table.rowIndex(eid)).toBe(row);
        expect(state.has(eid, GlobalTransform)).toBe(true);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(42);
    } finally {
        app.dispose();
    }
});

test("Body to Transform keeps its GlobalTransform row when handover crosses a fixed tick", async () => {
    const { app, handover } = await handoverApp("Body");
    const { state } = app;
    const eid = [...state.query([Body])][0];
    const table = globalTransformTable(state);
    const row = table.rowIndex(eid);
    try {
        handover((world, target) => {
            world.remove(target, Body);
            addTransform(world, target, 42);
            const bystander = world.create();
            addTransform(world, bystander, -9);
        });
        state.step(Time.FIXED_DT);
        state.step(Time.FIXED_DT);
        expect(table.rowIndex(eid)).toBe(row);
        expect(state.has(eid, GlobalTransform)).toBe(true);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(42);
    } finally {
        app.dispose();
    }
});

test("Transform to Body keeps its GlobalTransform row when handover crosses a fixed tick", async () => {
    const { app, handover } = await handoverApp("Transform");
    const { state } = app;
    const eid = [...state.query([Transform])][0];
    const table = globalTransformTable(state);
    const row = table.rowIndex(eid);
    try {
        handover((world, target) => {
            world.remove(target, Transform);
            const bystander = world.create();
            addTransform(world, bystander, -9);
            addStaticBody(world, target, 42);
        });
        state.step(Time.FIXED_DT);
        state.step(Time.FIXED_DT);
        expect(table.rowIndex(eid)).toBe(row);
        expect(state.has(eid, GlobalTransform)).toBe(true);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(42);
    } finally {
        app.dispose();
    }
});

test("the first Body spawn renders at its placement at half a fixed step", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const { state } = app;
        attachTestCamera(state);
        const table = globalTransformTable(state);
        state.step(Time.FIXED_DT);
        const eid = state.create();
        addStaticBody(state, eid, 100);
        state.step(Time.FIXED_DT * 1.5);
        expect(state.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(state, table, eid)).toBeCloseTo(100, 5);
    } finally {
        app.dispose();
    }
});

test("a newly spawned GlobalTransform producer renders at its placement at half a fixed step", async () => {
    let spawned = -1;
    const app = await build({
        defaults: false,
        plugins: [
            RenderPlugin,
            {
                name: "SpawnedPlacement",
                components: { SpawnedPlacement },
                traits: { SpawnedPlacement: { provides: [GlobalTransform] } },
                systems: [
                    {
                        group: "simulation",
                        update(state) {
                            if (spawned >= 0) return;
                            spawned = state.create();
                            state.add(spawned, SpawnedPlacement);
                            state.of(GlobalTransform).pos.set(spawned, 100, 0, 0, 0);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const { state } = app;
        attachTestCamera(state);
        const table = globalTransformTable(state);
        state.step(Time.FIXED_DT * 1.5);
        expect(state.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(state, table, spawned)).toBeCloseTo(100, 5);
    } finally {
        app.dispose();
    }
});

test("an author-marked Transform jump of any size snaps instead of interpolating", async () => {
    const app = await build({ defaults: false, plugins: [RenderPlugin] });
    try {
        const { state } = app;
        const eid = state.create();
        addTransform(state, eid, 0);
        attachTestCamera(state);
        const table = globalTransformTable(state);
        state.step(Time.FIXED_DT);
        state.of(Transform).pos.set(eid, 0.25, 0, 0, 0);
        state.teleport(eid);
        state.step(Time.FIXED_DT / 2);
        expect(state.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(state, table, eid)).toBeCloseTo(0.25, 5);
    } finally {
        app.dispose();
    }
});

test("setKinematic publishes moved body placement to the fixed GlobalTransform table after one step", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const { state } = app;
        const eid = state.create();
        addStaticBody(state, eid, 0);
        attachTestCamera(state);
        globalTransformTable(state);
        state.step(Time.FIXED_DT);
        const table = state.globalTransformRuntime!.current!;
        const row = table.rowIndex(eid);
        expect(row).toBeGreaterThanOrEqual(0);
        setKinematic(state, eid, [17, 3, -2], [0, 0, 0, 1], false);
        // No solver tick can republish the position on this draw-only step.
        state.step(0);
        const words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([17, 3, -2]);
    } finally {
        app.dispose();
    }
});

test("a kinematic teleport renders at its new placement at half a fixed step", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const { state } = app;
        const eid = state.create();
        addStaticBody(state, eid, 0);
        attachTestCamera(state);
        const table = globalTransformTable(state);
        state.step(Time.FIXED_DT);
        const teleport: System = {
            group: "fixed",
            after: [StepSystem],
            update(world) {
                setKinematic(world, eid, [100, 0, 0], [0, 0, 0, 1], true);
            },
        };
        state.addSystem(teleport);
        state.step(Time.FIXED_DT * 1.5);
        expect(state.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(state, table, eid)).toBeCloseTo(100, 5);
    } finally {
        app.dispose();
    }
});
