import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { attachCanvas, Camera, RenderPlugin } from "../../core/rendering";
import { build, globalTransformTable, probeBuffer, Time, Transform } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { Body, PhysicsPlugin, readBody } from "./index";

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

function attachTestCamera(state: import("../../engine").World): void {
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
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    const state = app.state;
    attachTestCamera(state);
    const body = state.of(Body);
    function falling(y: number): number {
        const eid = state.create();
        state.add(eid, Body);
        body.pos.set(eid, 0, y, 0, 0);
        body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
        body.mass.set(eid, 1);
        return eid;
    }
    const first = falling(10);
    const middle = state.create();
    state.add(middle, Transform);
    state.of(Transform).pos.set(middle, 23, 7, 9, 0);
    const second = falling(30);
    const table = globalTransformTable(state);
    const queue = state.gpu.device.queue;
    const descriptor = Object.getOwnPropertyDescriptor(queue, "writeBuffer");
    const write = queue.writeBuffer.bind(queue);
    let globalTransformWrites = 0;
    let globalTransformBytes = 0;
    Object.defineProperty(queue, "writeBuffer", {
        configurable: true,
        value: (...args: Parameters<GPUQueue["writeBuffer"]>) => {
            if (state.globalTransformRuntime!.stages.includes(args[0])) {
                globalTransformWrites++;
                globalTransformBytes += args[4] ?? 0;
            }
            return write(...args);
        },
    });
    try {
        state.step(Time.FIXED_DT);
        const previous = readBody(state, first);
        if (!previous) throw new Error("first falling Body has no solver state");
        globalTransformWrites = 0;
        globalTransformBytes = 0;
        state.gpu.device.pushErrorScope("validation");
        state.step(Time.FIXED_DT);
        expect(globalTransformWrites).toBe(1);
        expect(globalTransformBytes).toBe(
            (table.rowIndex(second) - table.rowIndex(first) + 1) * table.rowBytes,
        );
        const error = await bounded(
            "bulk interpolated GlobalTransform validation",
            state.gpu.device.popErrorScope(),
        );
        if (error) throw new Error(error.message);
        const result = await bounded(
            "bulk interpolated GlobalTransform readback",
            probeBuffer(state, table.buffer, {
                size: table.buffer.size,
                label: "physics-global-transform-range",
            }),
        );
        const words = new Float32Array(result.bytes);
        expect(words[table.rowIndex(first) * 12 + 1]).toBeCloseTo(previous.pos[1], 5);
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
