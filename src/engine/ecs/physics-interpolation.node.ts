import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { Body } from "../../core/physics";
import { attachCanvas, Camera, RenderingPlugin } from "../../core/rendering";
import { readBody, StandardPhysicsPlugin } from "../../standard/physics";
import { CanvasContext } from "../app/canvas.fixture";
import { createApp, globalTransformTable, probeBuffer, Time, Transform } from "../index";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();
if (typeof ResizeObserver === "undefined") {
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });
}

function attachTestCamera(world: import("../index").World): void {
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

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after 750 ms`)), 750);
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

test("engine interpolation uploads one GlobalTransform range and preserves unmoved renderer rows", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, RenderingPlugin],
    });
    const world = app.world;
    attachTestCamera(world);
    const body = world.storage(Body);
    function falling(y: number): number {
        const eid = world.create();
        world.add(eid, Body);
        body.position.set(eid, 0, y, 0, 0);
        body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
        body.mass.set(eid, 1);
        return eid;
    }
    const first = falling(10);
    const middle = world.create();
    world.add(middle, Transform);
    world.storage(Transform).translation.set(middle, 23, 7, 9, 0);
    const second = falling(30);
    const table = globalTransformTable(world);
    const queue = world.gpu.device.queue;
    const descriptor = Object.getOwnPropertyDescriptor(queue, "writeBuffer");
    const write = queue.writeBuffer.bind(queue);
    let globalTransformWrites = 0;
    let globalTransformBytes = 0;
    Object.defineProperty(queue, "writeBuffer", {
        configurable: true,
        value: (...args: Parameters<GPUQueue["writeBuffer"]>) => {
            if (world.globalTransformRuntime!.stages.includes(args[0])) {
                globalTransformWrites++;
                globalTransformBytes += args[4] ?? 0;
            }
            return write(...args);
        },
    });
    try {
        world.step(Time.FIXED_DT);
        const previous = readBody(world, first);
        if (!previous) throw new Error("first falling Body has no solver state");
        globalTransformWrites = 0;
        globalTransformBytes = 0;
        world.gpu.device.pushErrorScope("validation");
        world.step(Time.FIXED_DT);
        expect(globalTransformWrites).toBe(1);
        expect(globalTransformBytes).toBe(
            (table.rowIndex(second) - table.rowIndex(first) + 1) * table.rowBytes,
        );
        const error = await bounded(
            "bulk interpolated GlobalTransform validation",
            world.gpu.device.popErrorScope(),
        );
        if (error) throw new Error(error.message);
        const result = await bounded(
            "bulk interpolated GlobalTransform readback",
            probeBuffer(world, table.buffer, {
                size: table.buffer.size,
                label: "physics-global-transform-range",
            }),
        );
        const words = new Float32Array(result.bytes);
        expect(words[table.rowIndex(first) * 12 + 1]).toBeCloseTo(previous.position[1], 5);
        expect(
            Array.from(
                words.subarray(table.rowIndex(middle) * 12, table.rowIndex(middle) * 12 + 3),
            ),
        ).toEqual([23, 7, 9]);
        expect(
            Array.from(
                words.subarray(table.rowIndex(first) * 12 + 4, table.rowIndex(first) * 12 + 8),
            ),
        ).toEqual([0, 0, 0, 1]);
    } finally {
        if (descriptor) Object.defineProperty(queue, "writeBuffer", descriptor);
        else Reflect.deleteProperty(queue, "writeBuffer");
        app.dispose();
    }
});
