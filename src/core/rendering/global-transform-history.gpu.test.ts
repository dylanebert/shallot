import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { attachCanvas, Camera, PointLight, RenderingPlugin } from "../../core/rendering";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import {
    type createApp,
    GlobalTransform,
    globalTransformTable,
    probeBuffer,
    Time,
    Transform,
    teleport,
    type World,
} from "../../index";

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
import { deriveTransforms } from "../transform";
import {
    GlobalTransformHistoryEndSystem,
    GlobalTransformHistoryStartSystem,
    GlobalTransformHistory as TransformRuntime,
} from "./global-transform";

const startTick = (world: World) => {
    deriveTransforms(world);
    GlobalTransformHistoryStartSystem.update!(world);
};
const endTick = (world: World) => {
    deriveTransforms(world);
    GlobalTransformHistoryEndSystem.update!(world);
};

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
            expect(() => world.resource(TransformRuntime).previous!.bytes).toThrow("GPU-only");
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
                                teleport(world, eid);
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
                    startTick(world);
                    previous = x;
                    x += 10;
                    world.storage(Transform).translation.x.set(eid, x);
                    endTick(world);
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

{
    const eids: number[] = [];
    const placements = [20, -30];
    configs.push({
        defaults: false,
        plugins: [RenderingPlugin],
        setup(world) {
            for (let i = 0; i < 30; i++) world.create();
            for (const x of placements) {
                const eid = world.create();
                world.add(eid, Transform, { translation: [x, 0, 0, 0] });
                eids.push(eid);
            }
        },
    });
    test("GPU history startup seeds every placement present before its table is enabled", async () => {
        const app = subjects()[5];
        const { world } = app;
        try {
            attachTestCamera(world);
            const table = globalTransformTable(world);
            world.step(0);
            const words = new Float32Array(
                (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size })))
                    .bytes,
            );
            for (let i = 0; i < eids.length; i++)
                expect(words[table.rowIndex(eids[i]) * 12]).toBeCloseTo(placements[i], 5);
        } finally {
            app.dispose();
        }
    });
}

configs.push({ defaults: false, plugins: [RenderingPlugin] });
test("public exact ticks submit growth copies and retain interpolation, teleport and spawn history beyond the catch-up cap", async () => {
    const app = subjects()[6];
    const { world } = app;
    try {
        attachTestCamera(world);
        const stationary = world.create();
        const moving = world.create();
        const jumping = world.create();
        for (const eid of [stationary, moving, jumping]) world.add(eid, Transform);
        const source = world.storage(Transform).translation;
        source.x.set(stationary, 33);
        const table = globalTransformTable(world);
        world.step(0);
        const capacity = table.buffer.size;
        let spawned = -1;
        world.addSystem({
            group: "fixed",
            update() {
                const tick = world.time.fixedTick;
                source.x.set(moving, tick * 10);
                source.x.set(jumping, tick * 10);
                if (tick === 2) {
                    for (let i = 0; i < 100; i++) {
                        const eid = world.create();
                        world.add(eid, Transform, { translation: [i + 1000, 0, 0, 0] });
                    }
                }
                if (tick === 8) {
                    source.x.set(jumping, 800);
                    teleport(world, jumping);
                    spawned = world.create();
                    world.add(spawned, Transform, { translation: [900, 0, 0, 0] });
                }
            },
        });
        const queue = world.gpu.device.queue;
        const descriptor = Object.getOwnPropertyDescriptor(queue, "submit");
        const submit = queue.submit.bind(queue);
        let submissions = 0;
        Object.defineProperty(queue, "submit", {
            configurable: true,
            value: (...args: Parameters<GPUQueue["submit"]>) => {
                submissions++;
                return submit(...args);
            },
        });
        const frame = world.gpu.frame;
        try {
            for (let i = 0; i < 8; i++) world.tick();
            expect(submissions).toBeGreaterThan(0);
            expect(world.gpu.frame).toBe(frame);
            expect(table.buffer.size).toBeGreaterThan(capacity);
        } finally {
            if (descriptor) Object.defineProperty(queue, "submit", descriptor);
            else Reflect.deleteProperty(queue, "submit");
        }
        world.step(Time.FIXED_DT / 2);
        const words = new Float32Array(
            (await bounded(probeBuffer(world, table.buffer, { size: table.buffer.size }))).bytes,
        );
        for (const [eid, x] of [
            [stationary, 33],
            [moving, 75],
            [jumping, 800],
            [spawned, 900],
        ])
            expect(words[table.rowIndex(eid) * 12]).toBeCloseTo(x, 5);
    } finally {
        app.dispose();
    }
});
