import { expect, setDefaultTimeout, test } from "bun:test";

setDefaultTimeout(1000);

import * as d from "typegpu/data";
import { build } from "../../engine";
import { Mirror, MirrorPlugin, mirror } from "./index";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

async function snapshot(subject: Mirror, label: string) {
    const deadline = performance.now() + 1000;
    while (!subject.snapshot) {
        if (performance.now() >= deadline) throw new Error(`${label} timed out after 1000 ms`);
        await new Promise((resolve) => setTimeout(resolve, 1));
    }
    return subject.snapshot;
}

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

test("Mirror follows a growing table's record generation and invalidates its old-size snapshot", async () => {
    const app = await build({ defaults: false, plugins: [MirrorPlugin] });
    const state = app.state;
    const device = state.gpu.device;
    const table = state.table("mirror-generation", d.struct({ value: d.u32 }));
    const first = table.acquire(10);
    new DataView(table.bytes.buffer).setUint32(first * table.rowBytes, 42, true);
    table.markRange(first, 1);
    table.upload();
    const subject = mirror(state, table.buffer, { ring: 1 });
    try {
        Mirror.flush(state);
        const original = await snapshot(subject, "initial table Mirror snapshot");
        expect(new Uint32Array(original.bytes)[first]).toBe(42);
        const generation = table.generation;
        device.pushErrorScope("validation");
        const second = table.acquire(20);
        expect(table.generation).toBeGreaterThan(generation);
        expect(subject.size).toBe(table.buffer.size);
        expect(subject.snapshot).toBeNull();
        new DataView(table.bytes.buffer).setUint32(second * table.rowBytes, 73, true);
        table.markRange(second, 1);
        table.upload();
        state.gpu.frame = 9;
        Mirror.flush(state);
        const error = await bounded("growing table Mirror validation", device.popErrorScope());
        if (error) throw new Error(error.message);
        const current = await snapshot(subject, "grown table Mirror snapshot");
        expect(current.frame).toBe(9);
        expect(current.bytes.byteLength).toBe(table.buffer.size);
        expect(Array.from(new Uint32Array(current.bytes))).toEqual([42, 73]);
    } finally {
        subject.dispose();
        app.dispose();
    }
}, 100);

for (const kind of ["active", "map"] as const) {
    test(`Mirror follows a growing table's ${kind} generation and invalidates its old-size snapshot`, async () => {
        const app = await build({ defaults: false, plugins: [MirrorPlugin] });
        const state = app.state;
        const table = state.table(`mirror-${kind}-generation`, d.struct({ value: d.u32 }));
        table.acquire(4);
        if (kind === "map") table.enableEidLookup();
        table.upload();
        const buffer = () => (kind === "active" ? table.activeRowsBuffer! : table.eidToRowBuffer!);
        const subject = mirror(state, buffer(), { ring: 1 });
        try {
            Mirror.flush(state);
            const original = await snapshot(subject, `initial ${kind} Mirror snapshot`);
            const size = original.bytes.byteLength;
            const device = state.gpu.device;
            device.pushErrorScope("validation");
            table.acquire(33);
            expect(buffer().size).toBeGreaterThan(size);
            expect(state.tableForBuffer(buffer())).toBe(table);
            expect(subject.size).toBe(buffer().size);
            expect(subject.snapshot).toBeNull();
            table.upload();
            state.gpu.frame = 9;
            Mirror.flush(state);
            const error = await bounded(`grown ${kind} Mirror validation`, device.popErrorScope());
            if (error) throw new Error(error.message);
            const current = await snapshot(subject, `grown ${kind} Mirror snapshot`);
            expect(current.frame).toBe(9);
            expect(current.bytes.byteLength).toBe(buffer().size);
            const values = new Uint32Array(current.bytes);
            if (kind === "active") expect(Array.from(values)).toEqual([4, 0, 33, 1]);
            else {
                expect(values[4]).toBe(1);
                expect(values[33]).toBe(2);
            }
        } finally {
            subject.dispose();
            app.dispose();
        }
    });
}
