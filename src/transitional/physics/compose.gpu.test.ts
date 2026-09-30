import { expect, setDefaultTimeout, test } from "bun:test";

setDefaultTimeout(1000);

import { RenderPlugin } from "../../core/rendering";
import { build, probeBuffer, Time } from "../../engine";
import { Transform, transformTable } from "../transforms";
import { Body, PhysicsPlugin, readBody } from "./index";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after 1000 ms`)), 1000);
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

test("physics interpolation uploads one dense pose range and preserves unmoved rows inside it", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    const state = app.state;
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
    const table = transformTable(state);
    const queue = state.gpu.device.queue;
    const descriptor = Object.getOwnPropertyDescriptor(queue, "writeBuffer");
    const write = queue.writeBuffer.bind(queue);
    let poseWrites = 0;
    let poseBytes = 0;
    Object.defineProperty(queue, "writeBuffer", {
        configurable: true,
        value: (...args: Parameters<GPUQueue["writeBuffer"]>) => {
            if (args[0] === table.buffer) {
                poseWrites++;
                poseBytes += args[4] ?? 0;
            }
            return write(...args);
        },
    });
    try {
        state.step(Time.FIXED_DT);
        const previous = readBody(state, first);
        if (!previous) throw new Error("first falling Body has no solver pose");
        poseWrites = 0;
        poseBytes = 0;
        state.gpu.device.pushErrorScope("validation");
        state.step(Time.FIXED_DT);
        expect(poseWrites).toBe(1);
        expect(poseBytes).toBe(
            (table.rowIndex(second) - table.rowIndex(first) + 1) * table.rowBytes,
        );
        const error = await bounded(
            "bulk interpolated pose validation",
            state.gpu.device.popErrorScope(),
        );
        if (error) throw new Error(error.message);
        const result = await bounded(
            "bulk interpolated pose readback",
            probeBuffer(state, table.buffer, {
                size: table.buffer.size,
                label: "physics-pose-range",
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
}, 250);
