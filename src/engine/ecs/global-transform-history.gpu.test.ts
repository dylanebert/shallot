import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { attachCanvas, Camera, PointLight, RenderingPlugin } from "../../core/rendering";
import { CanvasContext } from "../app/canvas.fixture";
import {
    type createApp,
    GlobalTransform,
    globalTransformTable,
    probeBuffer,
    Time,
    Transform,
    type World,
} from "../index";

setDefaultTimeout(CEILING.gpu);
if (typeof ResizeObserver === "undefined") {
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });
}

import { gpuApps } from "../../../scripts/gpu.fixture";
import { beginGlobalTransformTick, endGlobalTransformTick } from "./global-transform";

const configs: Parameters<typeof createApp>[0][] = [];
const subjects = gpuApps(import.meta.path, configs);
function attachTestCamera(world: World): void {
    let context: CanvasContext;
    const canvas = {
        width: 32,
        height: 24,
        style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width: 32, height: 24 }),
    } as unknown as HTMLCanvasElement;
    context = new CanvasContext(canvas, 32, 24);
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
    attachCanvas(camera, canvas, world);
}

function bounded<T>(promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error("GlobalTransform observation exceeded 750 ms")),
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

{
    let eid = -1;
    configs.push({
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "FixedPlacement",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            const input = world.storage(Transform);
                            const x = world.time.fixedTick * 10;
                            input.translation.set(eid, x, 2, 3, 0);
                            input.scale.set(eid, x + 1, 1, 1, 0);
                            input.rotation.set(eid, 0, 0, 0, world.time.fixedTick % 2 ? -1 : 1);
                        },
                    },
                ],
            },
        ],
        setup(world) {
            eid = world.create();
            world.add(eid, Transform);
        },
    });
    test("catch-up ticks retain the penultimate GlobalTransform and no-tick draws advance only interpolation", async () => {
        const app = subjects()[0];
        const world = app.world;
        attachTestCamera(world);
        const table = globalTransformTable(world);
        const row = table.rowIndex(eid);
        try {
            world.step(0);
            world.gpu.device.pushErrorScope("validation");
            world.step(Time.FIXED_DT * 2.5);
            let words = new Float32Array(
                (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size })))
                    .bytes,
            );
            expect(words[row * 12]).toBeCloseTo(15, 5);
            expect(words[row * 12 + 8]).toBeCloseTo(16, 5);
            expect(words[row * 12 + 7]).toBeCloseTo(1, 5);
            expect(world.storage(GlobalTransform).translation.x.get(eid)).toBeCloseTo(20, 7);
            expect(world.storage(GlobalTransform).scale.x.get(eid)).toBeCloseTo(21, 7);
            world.step(Time.FIXED_DT * 0.25);
            expect(world.time.fixedSteps).toBe(0);
            expect(world.storage(GlobalTransform).translation.x.get(eid)).toBeCloseTo(20, 7);
            words = new Float32Array(
                (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size })))
                    .bytes,
            );
            expect(words[row * 12]).toBeCloseTo(17.5, 5);
            expect(words[row * 12 + 8]).toBeCloseTo(18.5, 5);
            world.step(Time.FIXED_DT * 0.5);
            expect(world.time.fixedSteps).toBe(1);
            expect(world.storage(GlobalTransform).translation.x.get(eid)).toBeCloseTo(30, 7);
            words = new Float32Array(
                (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size })))
                    .bytes,
            );
            expect(words[row * 12]).toBeCloseTo(22.5, 5);
            expect(() => world.globalTransformRuntime!.previous!.bytes).toThrow("GPU-only");
            expect(await bounded(world.gpu.device.popErrorScope())).toBeNull();
        } finally {
            app.dispose();
        }
    });
}

{
    let eid = -1;
    configs.push({
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "CatchupSpawn",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            if (eid < 0) {
                                eid = world.create();
                                world.add(eid, Transform);
                            }
                            world
                                .storage(Transform)
                                .translation.x.set(eid, world.time.fixedTick * 10);
                        },
                    },
                ],
            },
        ],
    });
    test("a producer spawned during catch-up keeps motion after its spawn tick", async () => {
        const app = subjects()[1];
        try {
            const { world } = app;
            attachTestCamera(world);
            const table = globalTransformTable(world);
            world.step(Time.FIXED_DT * 2.5);
            expect(world.time.fixedTick).toBe(2);
            expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
            expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(20);
            const words = new Float32Array(
                (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size })))
                    .bytes,
            );
            expect(words[table.rowIndex(eid) * 12]).toBeCloseTo(15, 5);
        } finally {
            app.dispose();
        }
    });
}

{
    let eid = -1;
    configs.push({
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "CatchupTeleport",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            if (world.time.fixedTick === 1) {
                                world.storage(Transform).translation.x.set(eid, 100);
                                world.teleport(eid);
                            } else {
                                world
                                    .storage(Transform)
                                    .translation.x.set(eid, 100 + (world.time.fixedTick - 1) * 10);
                            }
                        },
                    },
                ],
            },
        ],
        setup(world) {
            eid = world.create();
            world.add(eid, Transform);
            world.storage(Transform).translation.x.set(eid, 0);
        },
    });
    test("a teleport on the first catch-up tick keeps later tick motion", async () => {
        const app = subjects()[2];
        try {
            const { world } = app;
            attachTestCamera(world);
            const table = globalTransformTable(world);
            world.step(Time.FIXED_DT * 2.5);
            expect(world.time.fixedTick).toBe(2);
            expect(world.time.fixedAlpha).toBeCloseTo(0.5, 5);
            expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(110);
            const words = new Float32Array(
                (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size })))
                    .bytes,
            );
            expect(words[table.rowIndex(eid) * 12]).toBeCloseTo(105, 5);
        } finally {
            app.dispose();
        }
    });
}

{
    let eid = -1;
    configs.push({
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "FastPlacement",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            if (eid >= 0)
                                world
                                    .storage(Transform)
                                    .translation.x.set(eid, world.time.fixedTick * 4);
                        },
                    },
                ],
            },
        ],
        setup(world) {
            eid = world.create();
            world.add(eid, Transform);
        },
    });
    test("a renderer interpolates GlobalTransform when the scene has no lights", async () => {
        const app = subjects()[3];
        try {
            const { world } = app;
            attachTestCamera(world);
            expect([...world.query([PointLight])]).toHaveLength(0);
            const table = globalTransformTable(world);
            world.step(Time.FIXED_DT);
            world.step(Time.FIXED_DT * 1.5);
            const row = table.rowIndex(eid);
            expect(row).toBeGreaterThanOrEqual(0);
            const words = new Float32Array(
                (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size })))
                    .bytes,
            );
            expect(words[row * 12]).toBeCloseTo(6, 5);
        } finally {
            app.dispose();
        }
    });
}

{
    let eid = -1;
    configs.push({
        defaults: false,
        plugins: [RenderingPlugin],
        setup(world) {
            eid = world.create();
            world.add(eid, Transform);
        },
    });
    test("0, 1, 2 and 8 ticks before a frame retain only the last tick pair", async () => {
        const app = subjects()[4];
        const { world } = app;
        try {
            attachTestCamera(world);
            const table = globalTransformTable(world);
            world.step(0);
            let x = 0;
            let previous = 0;
            for (const ticks of [0, 1, 2, 8, 0]) {
                for (let tick = 0; tick < ticks; tick++) {
                    beginGlobalTransformTick(world);
                    previous = x;
                    x += 10;
                    world.storage(Transform).translation.x.set(eid, x);
                    endGlobalTransformTick(world);
                }
                world.step(Time.FIXED_DT * 0.1);
                const words = new Float32Array(
                    (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size })))
                        .bytes,
                );
                expect(words[table.rowIndex(eid) * 12]).toBeCloseTo(
                    previous + (x - previous) * world.time.fixedAlpha,
                    5,
                );
            }
        } finally {
            app.dispose();
        }
    });
}
