import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { Body, BodyType } from "../../core/physics";
import {
    attachCanvas,
    Camera,
    RenderingPlugin,
    Views,
    viewportToWorld,
} from "../../core/rendering";
import * as engine from "../../engine";
import { createApp, probeBuffer, u32 } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { component } from "../../engine/ecs/component";
import type { System } from "../../engine/ecs/scheduler";
import { Time } from "../../engine/ecs/scheduler";
import { StandardPhysicsPlugin, StepPhysicsSystem, setKinematic } from "../../standard/physics";
import * as transform from "./index";
import {
    GlobalTransform,
    GlobalTransformTickEndSystem,
    GlobalTransformTickStartSystem,
    globalTransformTable,
    PrepareGlobalTransformSystem,
    Transform,
    TransformPlugin,
    TransformRuntime,
    teleport as teleportPlacement,
} from "./index";

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

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

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
test("GlobalTransform is a transform-module public schema, independent of Physics", () => {
    expect(transform.GlobalTransform).toBe(GlobalTransform);
    expect(Reflect.get(engine, "GlobalTransform")).toBeUndefined();
});

test("TransformPlugin boundaries bracket exact ticks and gather after every simulation system", async () => {
    let eid = -1;
    let start = 0;
    const app = await createApp({
        defaults: false,
        plugins: [
            TransformPlugin,
            {
                name: "FirstPlacementReader",
                systems: [
                    {
                        group: "fixed",
                        first: true,
                        update: (world) => {
                            start = world.storage(GlobalTransform).translation.x.get(eid);
                        },
                    },
                ],
            },
        ],
    });
    const world = app.world;
    try {
        for (const system of [
            GlobalTransformTickStartSystem,
            GlobalTransformTickEndSystem,
            PrepareGlobalTransformSystem,
        ])
            expect(world.hasSystem(system)).toBe(true);
        eid = world.create();
        world.add(eid, Transform);
        const source = world.storage(Transform).translation;
        const global = world.storage(GlobalTransform).translation;
        world.addSystem({
            group: "fixed",
            last: true,
            update: () => source.x.set(eid, 7),
        });
        world.addSystem({
            group: "fixed",
            terminal: true,
            update: () => source.x.set(eid, source.x.get(eid) + 2),
        });
        source.x.set(eid, 3);
        world.tick();
        expect(start).toBe(3);
        expect(global.x.get(eid)).toBe(9);
        world.addSystem({ group: "simulation", last: true, update: () => source.x.set(eid, 11) });
        let drawn = 0;
        world.addSystem({
            group: "draw",
            update: () => {
                drawn = global.x.get(eid);
            },
        });
        world.step(0);
        expect(drawn).toBe(11);
    } finally {
        app.dispose();
    }
});

test("Transform placement lands in the fixed-tick GlobalTransform column and the renderer table", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, RenderingPlugin],
    });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Transform);
        const placement = world.storage(Transform);
        placement.translation.set(eid, 12, 7, -3, 0);
        placement.scale.set(eid, 2, 3, 4, 0);
        attachTestCamera(world);
        world.step(Time.FIXED_DT);
        expect(world.has(eid, GlobalTransform)).toBe(true);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(12);
        expect(world.storage(GlobalTransform).translation.y.get(eid)).toBe(7);
        expect(world.storage(GlobalTransform).translation.z.get(eid)).toBe(-3);
        const table = globalTransformTable(world);
        const row = table.rowIndex(eid);
        expect(row).toBeGreaterThanOrEqual(0);
        const words = new Float32Array(
            (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([2, 3, 4]);
    } finally {
        app.dispose();
    }
});

test("a Body publishes unit scale to fixed-tick GlobalTransform and renderer rows", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, RenderingPlugin],
    });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body);
        world.storage(Body).position.set(eid, 12, 7, -3, 0);
        world.storage(Body).halfExtents.set(eid, 1, 2, 3, 0);
        attachTestCamera(world);
        world.step(Time.FIXED_DT);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(12);
        expect(Reflect.get(GlobalTransform, "scale")).toBeDefined();
        const scale = Reflect.get(world.storage(GlobalTransform), "scale");
        expect([scale.x.get(eid), scale.y.get(eid), scale.z.get(eid)]).toEqual([1, 1, 1]);
        const table = globalTransformTable(world);
        const row = table.rowIndex(eid);
        const words = new Float32Array(
            (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([1, 1, 1]);
    } finally {
        app.dispose();
    }
});

test("viewportToWorld reads a body camera's fixed-tick GlobalTransform without requiring Transform", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, RenderingPlugin],
    });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Camera);
        world.storage(Body).position.set(eid, 12, 7, -3, 0);
        world.step(Time.FIXED_DT);
        expect(world.has(eid, Transform)).toBe(false);
        const bound = attachTestCamera(world, eid);
        const viewport = world
            .resource(engine.Viewports)
            .get(world.resource(Views).get(bound)!.viewportIndex)!;
        const near = world.storage(Camera).near.get(eid);
        expect(viewportToWorld(world, eid, viewport.cssWidth / 2, viewport.cssHeight / 2)).toEqual({
            origin: [12, 7, -3 - near],
            dir: [0, 0, -1],
        });
    } finally {
        app.dispose();
    }
});

function addStaticBody(world: engine.World, eid: number, x: number): void {
    world.add(eid, Body);
    world.storage(Body).position.set(eid, x, 0, 0, 0);
}

function addTransform(world: engine.World, eid: number, x: number): void {
    world.add(eid, Transform);
    world.storage(Transform).translation.set(eid, x, 0, 0, 0);
}

function attachTestCamera(world: engine.World, existing?: number): number {
    let context: CanvasContext;
    const canvas = {
        width: 32,
        height: 24,
        style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width: 32, height: 24 }),
    } as unknown as HTMLCanvasElement;
    context = new CanvasContext(canvas, 32, 24);
    const camera = existing ?? world.create();
    if (existing === undefined) {
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
    }
    attachCanvas(camera, canvas, world);
    return camera;
}

async function renderedX(
    world: engine.World,
    table: ReturnType<typeof globalTransformTable>,
    eid: number,
): Promise<number> {
    const row = table.rowIndex(eid);
    expect(row).toBeGreaterThanOrEqual(0);
    const result = await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size }));
    return new Float32Array(result.bytes)[row * 12];
}

async function handoverApp(initial: "Body" | "Transform"): Promise<{
    app: Awaited<ReturnType<typeof createApp>>;
    handover(action: (world: engine.World, eid: number) => void): void;
}> {
    let action: ((world: engine.World, eid: number) => void) | undefined;
    let eid = -1;
    const handoverSystem: System = {
        group: "simulation",
        update(world) {
            const current = action;
            if (!current) return;
            action = undefined;
            current(world, eid);
        },
    };
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, { name: "Handover", systems: [handoverSystem] }],
        setup(world) {
            eid = world.create();
            if (initial === "Body") addStaticBody(world, eid, 10);
            else addTransform(world, eid, 10);
        },
    });
    app.world.step(Time.FIXED_DT);
    return { app, handover: (next) => (action = next) };
}

test("Body to Transform keeps its GlobalTransform row for a same-frame producer handover", async () => {
    const { app } = await handoverApp("Body");
    const { world } = app;
    const eid = [...world.query([Body])][0];
    const table = globalTransformTable(world);
    const row = table.rowIndex(eid);
    try {
        world.remove(eid, Body);
        addTransform(world, eid, 42);
        const bystander = world.create();
        addTransform(world, bystander, -9);
        world.step(Time.FIXED_DT);
        expect(table.rowIndex(eid)).toBe(row);
        expect(world.has(eid, GlobalTransform)).toBe(true);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(42);
    } finally {
        app.dispose();
    }
});

test("Transform to Body keeps its GlobalTransform row for a same-frame producer handover", async () => {
    const { app } = await handoverApp("Transform");
    const { world } = app;
    const eid = [...world.query([Transform])][0];
    const table = globalTransformTable(world);
    const row = table.rowIndex(eid);
    try {
        world.remove(eid, Transform);
        const bystander = world.create();
        addTransform(world, bystander, -9);
        addStaticBody(world, eid, 42);
        world.step(Time.FIXED_DT);
        expect(table.rowIndex(eid)).toBe(row);
        expect(world.has(eid, GlobalTransform)).toBe(true);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(42);
    } finally {
        app.dispose();
    }
});

test("Body to Transform keeps its GlobalTransform row when handover crosses a fixed tick", async () => {
    const { app, handover } = await handoverApp("Body");
    const { world } = app;
    const eid = [...world.query([Body])][0];
    const table = globalTransformTable(world);
    const row = table.rowIndex(eid);
    try {
        handover((world, target) => {
            world.remove(target, Body);
            addTransform(world, target, 42);
            const bystander = world.create();
            addTransform(world, bystander, -9);
        });
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(table.rowIndex(eid)).toBe(row);
        expect(world.has(eid, GlobalTransform)).toBe(true);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(42);
    } finally {
        app.dispose();
    }
});

test("Transform to Body keeps its GlobalTransform row when handover crosses a fixed tick", async () => {
    const { app, handover } = await handoverApp("Transform");
    const { world } = app;
    const eid = [...world.query([Transform])][0];
    const table = globalTransformTable(world);
    const row = table.rowIndex(eid);
    try {
        handover((world, target) => {
            world.remove(target, Transform);
            const bystander = world.create();
            addTransform(world, bystander, -9);
            addStaticBody(world, target, 42);
        });
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(table.rowIndex(eid)).toBe(row);
        expect(world.has(eid, GlobalTransform)).toBe(true);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(42);
    } finally {
        app.dispose();
    }
});

test("the first Body spawn renders at its placement at half a fixed step", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, RenderingPlugin],
    });
    try {
        const { world } = app;
        attachTestCamera(world);
        const table = globalTransformTable(world);
        world.step(Time.FIXED_DT);
        const eid = world.create();
        addStaticBody(world, eid, 100);
        world.step(Time.FIXED_DT * 1.5);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(world, table, eid)).toBeCloseTo(100, 5);
    } finally {
        app.dispose();
    }
});

test("a newly spawned GlobalTransform producer renders at its placement at half a fixed step", async () => {
    let spawned = -1;
    const app = await createApp({
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "SpawnedPlacement",
                components: [
                    component("SpawnedPlacement", SpawnedPlacement, {
                        requires: [GlobalTransform],
                    }),
                ],

                systems: [
                    {
                        group: "simulation",
                        update(world) {
                            if (spawned >= 0) return;
                            spawned = world.create();
                            world.add(spawned, SpawnedPlacement);
                            world.storage(GlobalTransform).translation.set(spawned, 100, 0, 0, 0);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const { world } = app;
        attachTestCamera(world);
        const table = globalTransformTable(world);
        world.step(Time.FIXED_DT * 1.5);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(world, table, spawned)).toBeCloseTo(100, 5);
    } finally {
        app.dispose();
    }
});

test("an author-marked Transform jump of any size snaps instead of interpolating", async () => {
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin] });
    try {
        const { world } = app;
        const eid = world.create();
        addTransform(world, eid, 0);
        attachTestCamera(world);
        const table = globalTransformTable(world);
        world.step(Time.FIXED_DT);
        world.storage(Transform).translation.set(eid, 0.25, 0, 0, 0);
        teleportPlacement(world, eid);
        world.step(Time.FIXED_DT / 2);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(world, table, eid)).toBeCloseTo(0.25, 5);
    } finally {
        app.dispose();
    }
});

test("setKinematic publishes moved body placement to the fixed GlobalTransform table after one step", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, RenderingPlugin],
    });
    try {
        const { world } = app;
        const eid = world.create();
        addStaticBody(world, eid, 0);
        world.storage(Body).type.set(eid, BodyType.Kinematic);
        attachTestCamera(world);
        globalTransformTable(world);
        world.step(Time.FIXED_DT);
        const table = world.resource(TransformRuntime).current!;
        const row = table.rowIndex(eid);
        expect(row).toBeGreaterThanOrEqual(0);
        setKinematic(world, eid, [17, 3, -2], [0, 0, 0, 1], false);
        // No solver tick can republish the position on this draw-only step.
        world.step(0);
        const words = new Float32Array(
            (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([17, 3, -2]);
    } finally {
        app.dispose();
    }
});

test("a kinematic teleport renders at its new placement at half a fixed step", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, RenderingPlugin],
    });
    try {
        const { world } = app;
        const eid = world.create();
        addStaticBody(world, eid, 0);
        world.storage(Body).type.set(eid, BodyType.Kinematic);
        attachTestCamera(world);
        const table = globalTransformTable(world);
        world.step(Time.FIXED_DT);
        const teleport: System = {
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                setKinematic(world, eid, [100, 0, 0], [0, 0, 0, 1], true);
            },
        };
        world.addSystem(teleport);
        world.step(Time.FIXED_DT * 1.5);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(world, table, eid)).toBeCloseTo(100, 5);
    } finally {
        app.dispose();
    }
});
