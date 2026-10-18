import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createTestGpuApp as createApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance, MeshPlugin } from "../../core/mesh";
import { Body, BodyType } from "../../core/physics";
import {
    attachCanvas,
    Camera,
    PointLight,
    RenderingPlugin,
    SpotLight,
    Views,
    viewportToWorld,
} from "../../core/rendering";
import * as engine from "../../engine";
import { probeBuffer, u32 } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { component } from "../../engine/ecs/component";
import type { System } from "../../engine/ecs/scheduler";
import { Time } from "../../engine/ecs/scheduler";
import {
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    setTargetTransform,
    setTransform,
} from "../../standard/physics";
import * as transform from "../transform";
import {
    GlobalTransform,
    GlobalTransformTickEndSystem,
    GlobalTransformTickStartSystem,
    PrepareGlobalTransformSystem,
    resetInterpolation,
    Transform,
    TransformPlugin,
} from "../transform";
import {
    globalTransformTable,
    prepareGlobalTransformFrame,
    readRenderGlobalTransform,
    GlobalTransformHistory as TransformRuntime,
} from "./global-transform";
import { BeginFrameSystem } from "./substrate";

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

const SpawnedPlacement = { marker: u32 };

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
        await disposeGpuApps([app]);
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
            (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([2, 3, 4]);
    } finally {
        await disposeGpuApps([app]);
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
        world.storage(Transform).translation.set(eid, 12, 7, -3);
        attachTestCamera(world);
        world.step(Time.FIXED_DT);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(12);
        expect(Reflect.get(GlobalTransform, "scale")).toBeDefined();
        const scale = Reflect.get(world.storage(GlobalTransform), "scale");
        expect([scale.x.get(eid), scale.y.get(eid), scale.z.get(eid)]).toEqual([1, 1, 1]);
        const table = globalTransformTable(world);
        const row = table.rowIndex(eid);
        const words = new Float32Array(
            (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([1, 1, 1]);
    } finally {
        await disposeGpuApps([app]);
    }
});

test("viewportToWorld reads a body camera's fixed-tick GlobalTransform", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, RenderingPlugin],
    });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Camera);
        world.storage(Transform).translation.set(eid, 12, 7, -3);
        world.step(Time.FIXED_DT);
        expect(world.has(eid, Transform)).toBe(true);
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
        await disposeGpuApps([app]);
    }
});

function addStaticBody(world: engine.World, eid: number, x: number): void {
    world.add(eid, Body);
    world.storage(Transform).translation.set(eid, x, 0, 0);
}

function addTransform(world: engine.World, eid: number, x: number): void {
    if (!world.has(eid, Transform)) world.add(eid, Transform);
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
    attachCanvas(world, camera, canvas);
    return camera;
}

async function renderedX(
    world: engine.World,
    table: ReturnType<typeof globalTransformTable>,
    eid: number,
): Promise<number> {
    const row = table.rowIndex(eid);
    expect(row).toBeGreaterThanOrEqual(0);
    const result = await probeBuffer(world, table.buffer, { size: table.buffer.size });
    const gpu = new Float32Array(result.bytes);
    if (world.has(eid, PointLight) || world.has(eid, SpotLight)) {
        const cpu = new Float32Array(12);
        readRenderGlobalTransform(world, eid, cpu);
        for (const lane of [0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
            expect(cpu[lane]).toBeCloseTo(gpu[row * 12 + lane]!, 5);
    }
    return gpu[row * 12];
}

async function interpolatedRows(
    world: engine.World,
    table: ReturnType<typeof globalTransformTable>,
): Promise<Float32Array> {
    const runtime = world.resource(TransformRuntime);
    world.gpu.device.queue.writeBuffer(
        runtime.params!,
        0,
        new Float32Array([world.time.fixedAlpha, runtime.current!.count, 0, 0]),
    );
    const encoder = world.gpu.device.createCommandEncoder();
    prepareGlobalTransformFrame(world, encoder);
    const pass = encoder.beginComputePass();
    pass.setPipeline(runtime.pipeline!);
    pass.setBindGroup(0, runtime.group!);
    pass.dispatchWorkgroups(Math.ceil(runtime.current!.count / 64));
    pass.end();
    world.gpu.device.queue.submit([encoder.finish()]);
    return new Float32Array(
        (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
    );
}

async function placementHistoryApp(): Promise<{
    app: Awaited<ReturnType<typeof createApp>>;
    world: engine.World;
    eid: number;
    writeOnTick(x: number): void;
}> {
    let eid = -1;
    let tickWrite: number | undefined;
    const app = await createApp({
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "PlacementHistoryWriter",
                recovery: "stateless",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            if (tickWrite === undefined) return;
                            world.storage(Transform).translation.x.set(eid, tickWrite);
                            tickWrite = undefined;
                        },
                    },
                ],
            },
        ],
    });
    const world = app.world;
    eid = world.create();
    world.add(eid, PointLight);
    globalTransformTable(world);
    world.step(0);
    await world.gpu.device.queue.onSubmittedWorkDone();
    return { app, world, eid, writeOnTick: (x) => (tickWrite = x) };
}

test("aborted GlobalTransform history matches across fixed, restore, and interpolation-reset sequences", async () => {
    type Action =
        | { kind: "tick"; x?: number; resetInterpolation?: boolean; abort?: boolean }
        | { kind: "draw"; resetInterpolation?: boolean; abort?: boolean }
        | { kind: "restore" };
    const sequences: { id: number; actions: Action[] }[] = [
        {
            id: 1,
            actions: [
                { kind: "tick", x: 20, resetInterpolation: true, abort: true },
                { kind: "tick", x: 30 },
            ],
        },
        { id: 2, actions: [{ kind: "tick", x: 20, abort: true }, { kind: "tick" }] },
        {
            id: 3,
            actions: [
                { kind: "tick", x: 20, abort: true },
                { kind: "tick", x: 30, abort: true },
                { kind: "tick", x: 40 },
            ],
        },
        {
            id: 4,
            actions: [
                { kind: "tick", x: 20, resetInterpolation: true, abort: true },
                { kind: "draw", resetInterpolation: true },
            ],
        },
        {
            id: 5,
            actions: [
                { kind: "tick", x: 20, resetInterpolation: true, abort: true },
                { kind: "restore" },
                { kind: "draw" },
                { kind: "tick", x: 30 },
            ],
        },
        {
            id: 6,
            actions: [{ kind: "restore" }, { kind: "draw", abort: true }, { kind: "tick", x: 30 }],
        },
        {
            id: 7,
            actions: [
                { kind: "tick", x: 20, abort: true },
                { kind: "restore" },
                { kind: "draw", abort: true },
                { kind: "draw", resetInterpolation: true },
            ],
        },
        {
            id: 8,
            actions: [
                { kind: "tick", x: 20, abort: true },
                { kind: "tick", x: 30, resetInterpolation: true },
            ],
        },
    ];

    async function runSequence(
        actions: Action[],
        abortVariant: boolean,
    ): Promise<{
        current: number[];
        previous: number[];
        render: number[];
        flags: {
            tickCount: number;
            ranges: number[];
            historyNeedsPromotion: boolean;
            historyPromotionOnSubmit: boolean;
            historyDiscarded: boolean;
            historyDiscontinuitiesOnSubmit: number[];
            discontinuities: number[];
        };
    }> {
        const { app, world, eid, writeOnTick } = await placementHistoryApp();
        let resetInterpolationNextTick = false;
        let abortNextDraw = false;
        try {
            attachTestCamera(world);
            world.addSystem(
                {
                    name: "DifferentialPlacementReset",
                    group: "fixed",
                    terminal: true,
                    update(world) {
                        if (!resetInterpolationNextTick) return;
                        resetInterpolationNextTick = false;
                        resetInterpolation(world, eid);
                    },
                },
                "PlacementHistoryWriter",
            );
            if (abortVariant)
                world.addSystem({
                    name: "DifferentialDrawAbort",
                    group: "draw",
                    terminal: true,
                    update() {
                        if (!abortNextDraw) return;
                        abortNextDraw = false;
                        throw new Error("abort differential draw");
                    },
                });

            writeOnTick(10);
            world.step(Time.FIXED_DT);
            await world.gpu.device.queue.onSubmittedWorkDone();
            const snapshot = world.snapshot();

            function draw(delta: number, abort: boolean): void {
                abortNextDraw = abort && abortVariant;
                try {
                    if (abortNextDraw)
                        expect(() => world.step(delta)).toThrow("abort differential draw");
                    else world.step(delta);
                } finally {
                    abortNextDraw = false;
                }
            }

            for (const action of actions) {
                if (action.kind === "restore") {
                    world.restore(snapshot);
                } else if (action.kind === "tick") {
                    if (action.x !== undefined) writeOnTick(action.x);
                    resetInterpolationNextTick = action.resetInterpolation ?? false;
                    draw(Time.FIXED_DT, action.abort ?? false);
                } else {
                    if (action.resetInterpolation) resetInterpolation(world, eid);
                    draw(0, action.abort ?? false);
                }
            }

            world.step(0);
            await world.gpu.device.queue.onSubmittedWorkDone();

            const runtime = world.resource(TransformRuntime);
            async function row(table: typeof runtime.current): Promise<number[]> {
                if (!table) throw new Error("GlobalTransform differential table is unavailable");
                const index = table.rowIndex(eid);
                expect(index).toBeGreaterThanOrEqual(0);
                const result = await probeBuffer(world, table.buffer, { size: table.buffer.size });
                const values = new Float32Array(result.bytes);
                return Array.from(values.subarray(index * 12, index * 12 + 12));
            }
            const render = await row(runtime.render);
            const cpu = new Float32Array(12);
            readRenderGlobalTransform(world, eid, cpu);
            for (const lane of [0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
                expect(cpu[lane]).toBeCloseTo(render[lane]!, 5);
            return {
                current: await row(runtime.current),
                previous: await row(runtime.previous),
                render,
                flags: {
                    tickCount: runtime.tickCount,
                    ranges: Array.from(runtime.ranges),
                    historyNeedsPromotion: runtime.historyNeedsPromotion,
                    historyPromotionOnSubmit: runtime.historyPromotionOnSubmit,
                    historyDiscarded: runtime.historyDiscarded,
                    historyDiscontinuitiesOnSubmit: Array.from(
                        runtime.historyDiscontinuitiesOnSubmit,
                    ),
                    discontinuities: Array.from(
                        world.resource(transform.TransformRuntime).discontinuities,
                    ),
                },
            };
        } finally {
            await disposeGpuApps([app]);
        }
    }

    for (const sequence of sequences) {
        const reference = await runSequence(sequence.actions, false);
        const aborted = await runSequence(sequence.actions, true);
        try {
            expect(aborted).toEqual(reference);
        } catch (error) {
            throw new Error(`GlobalTransform differential sequence ${sequence.id} diverged`, {
                cause: error,
            });
        }
    }
});

test("a GlobalTransform created after history preparation remains pending for the next draw", async () => {
    const { app, world } = await placementHistoryApp();
    let eid = -1;
    try {
        world.addSystem({
            group: "draw",
            last: true,
            update(world) {
                if (eid >= 0) return;
                eid = world.create();
                world.add(eid, Transform);
            },
        });
        world.step(0);
        const marks = world.resource(transform.TransformRuntime).discontinuities;
        const mask = 1 << (eid & 31);
        expect(marks[eid >>> 5]! & mask).not.toBe(0);

        world.step(0);
        expect(marks[eid >>> 5]! & mask).toBe(0);
    } finally {
        await disposeGpuApps([app]);
    }
});

test("an unrelated successful draw cannot commit aborted GlobalTransform history", async () => {
    const { app, world, eid, writeOnTick } = await placementHistoryApp();
    try {
        attachTestCamera(world);
        world.step(0);
        writeOnTick(10);
        world.step(Time.FIXED_DT);
        await world.gpu.device.queue.onSubmittedWorkDone();

        writeOnTick(20);
        let abort = true;
        world.addSystem({
            group: "draw",
            last: true,
            update() {
                if (!abort) return;
                abort = false;
                throw new Error("abort GlobalTransform history");
            },
        });
        expect(() => world.step(Time.FIXED_DT)).toThrow("abort GlobalTransform history");

        world.removeSystem(BeginFrameSystem);
        world.step(0);
        world.addSystem(BeginFrameSystem);
        world.step(0);
        await world.gpu.device.queue.onSubmittedWorkDone();

        const current = world.resource(TransformRuntime).current!;
        const previous = world.resource(TransformRuntime).previous!;
        const row = current.rowIndex(eid);
        expect(row).toBeGreaterThanOrEqual(0);
        const result = await probeBuffer(world, previous.buffer, { size: previous.buffer.size });
        expect(new Float32Array(result.bytes)[row * 12]).toBe(10);
    } finally {
        await disposeGpuApps([app]);
    }
});

async function expectCurrentPlacement(world: engine.World, eid: number): Promise<void> {
    const current = world.resource(TransformRuntime).current!;
    expect(await renderedX(world, current, eid)).toBe(
        world.storage(GlobalTransform).translation.x.get(eid),
    );
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
        plugins: [
            StandardPhysicsPlugin,
            RenderingPlugin,
            { name: "Handover", systems: [handoverSystem] },
        ],
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
        await disposeGpuApps([app]);
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
        await disposeGpuApps([app]);
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
        await disposeGpuApps([app]);
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
        await disposeGpuApps([app]);
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
        await disposeGpuApps([app]);
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
        await disposeGpuApps([app]);
    }
});

test("a frame-written mesh row snaps at any alpha and step(0), then the next tick interpolates from the write", async () => {
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin, MeshPlugin] });
    try {
        const { world } = app;
        const eid = world.create();
        addTransform(world, eid, 0);
        attachTestCamera(world);
        const table = globalTransformTable(world);
        world.add(eid, MeshInstance);
        let written = 0;
        let frameWrites = true;
        world.addSystem({
            group: "simulation",
            update() {
                if (frameWrites) world.storage(Transform).translation.x.set(eid, ++written);
            },
        });
        for (const delta of [1, 0.5, 0.5, 0.75, 0.75, 0, 0]) {
            world.step(delta * Time.FIXED_DT);
            expect(await renderedX(world, table, eid)).toBeCloseTo(written, 5);
        }
        frameWrites = false;
        world.addSystem({
            group: "fixed",
            update() {
                world.storage(Transform).translation.x.set(eid, written + 4);
            },
        });
        world.step(Time.FIXED_DT);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(world, table, eid)).toBeCloseTo(written + 2, 5);
    } finally {
        await disposeGpuApps([app]);
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
        world.addSystem({
            group: "fixed",
            update() {
                world.storage(Transform).translation.set(eid, 0.25, 0, 0, 0);
                resetInterpolation(world, eid);
            },
        });
        world.step(Time.FIXED_DT * 1.5);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(world, table, eid)).toBeCloseTo(0.25, 5);
    } finally {
        await disposeGpuApps([app]);
    }
});

test("setTargetTransform publishes moved body placement to the fixed GlobalTransform table after one step", async () => {
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
        setTargetTransform(world, eid, [17, 3, -2], [0, 0, 0, 1]);
        // No solver tick can republish the position on this draw-only step.
        world.step(0);
        const words = new Float32Array(
            (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([17, 3, -2]);
    } finally {
        await disposeGpuApps([app]);
    }
});

test("restore re-stages ticked GlobalTransform rows for the next GPU table upload", async () => {
    const { app, world, eid, writeOnTick } = await placementHistoryApp();
    try {
        writeOnTick(10);
        world.tick();
        const snapshot = world.snapshot();
        world.restore(snapshot);
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(10);
        await expectCurrentPlacement(world, eid);
    } finally {
        await disposeGpuApps([app]);
    }
});

test("a later no-write tick does not replay the earlier phase-1 GlobalTransform range", async () => {
    const { app, world, eid, writeOnTick } = await placementHistoryApp();
    try {
        writeOnTick(10);
        world.tick();
        world.storage(Transform).translation.x.set(eid, 20);
        world.tick();
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(20);
        world.step(0);
        await expectCurrentPlacement(world, eid);
    } finally {
        await disposeGpuApps([app]);
    }
});

test("a fixed step does not replay the earlier phase-1 GlobalTransform range", async () => {
    const { app, world, eid, writeOnTick } = await placementHistoryApp();
    try {
        writeOnTick(10);
        world.tick();
        world.storage(Transform).translation.x.set(eid, 20);
        world.step(Time.FIXED_DT);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(20);
        await expectCurrentPlacement(world, eid);
    } finally {
        await disposeGpuApps([app]);
    }
});

test("restoring more than 32 GlobalTransforms leaves every interpolated row at its restored pose", async () => {
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin] });
    try {
        const { world } = app;
        const eids: number[] = [];
        const poses: number[] = [];
        for (let i = 0; i < 40; i++) {
            const eid = world.create();
            eids.push(eid);
            poses.push(i + 1);
            addTransform(world, eid, i + 1);
        }
        const table = globalTransformTable(world);
        world.step(Time.FIXED_DT);
        await world.gpu.device.queue.onSubmittedWorkDone();
        const snapshot = world.snapshot();

        for (let i = 0; i < eids.length; i++)
            world.storage(Transform).translation.x.set(eids[i], poses[i] + 100);
        world.step(Time.FIXED_DT);
        await world.gpu.device.queue.onSubmittedWorkDone();
        world.restore(snapshot);
        world.step(0);

        const rows = await interpolatedRows(world, table);
        expect(eids.length).toBeGreaterThan(32);
        for (let i = 0; i < eids.length; i++) {
            const row = table.rowIndex(eids[i]);
            expect(row).toBeGreaterThanOrEqual(0);
            expect(rows[row * 12]).toBeCloseTo(poses[i], 5);
        }
    } finally {
        await disposeGpuApps([app]);
    }
});

test("restore followed by a tick still interpolates between restored and ticked placement", async () => {
    let eid = -1;
    let advance = false;
    const app = await createApp({
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "RestoreInterpolation",
                recovery: "stateless",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            if (advance) world.storage(Transform).translation.x.set(eid, 10);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const { world } = app;
        eid = world.create();
        addTransform(world, eid, 0);
        const table = globalTransformTable(world);
        world.step(Time.FIXED_DT);
        await world.gpu.device.queue.onSubmittedWorkDone();
        const snapshot = world.snapshot();

        world.storage(Transform).translation.x.set(eid, 100);
        world.step(0);
        world.restore(snapshot);
        advance = true;
        world.tick();
        advance = false;
        world.step(Time.FIXED_DT / 2);

        expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        const rows = await interpolatedRows(world, table);
        expect(rows[table.rowIndex(eid) * 12]).toBeCloseTo(5, 5);
    } finally {
        await disposeGpuApps([app]);
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
                setTransform(world, eid, [100, 0, 0], [0, 0, 0, 1]);
            },
        };
        world.addSystem(teleport);
        world.step(Time.FIXED_DT * 1.5);
        expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(await renderedX(world, table, eid)).toBeCloseTo(100, 5);
    } finally {
        await disposeGpuApps([app]);
    }
});
