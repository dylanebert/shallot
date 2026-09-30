import { expect, setDefaultTimeout, test } from "bun:test";
import { build } from "../app";
import { rawDevice } from "./gpu";
import { probeBuffer } from "./probe";
import { controlledReadback } from "./readback.fixture";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

async function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
        return await Promise.race([
            promise,
            new Promise<never>((_, reject) => {
                timer = setTimeout(() => reject(new Error(`${label} exceeded 750 ms`)), 750);
            }),
        ]);
    } finally {
        clearTimeout(timer);
    }
}

async function freshDevice() {
    const adapter = await bounded("readback lifecycle adapter", navigator.gpu.requestAdapter());
    if (!adapter) throw new Error("readback lifecycle adapter unavailable");
    return bounded("readback lifecycle device", adapter.requestDevice());
}

test("world pools on a shared device own separate staging and release it without affecting a sibling", async () => {
    const device = await freshDevice();
    const raw = rawDevice(device);
    const original = raw.createBuffer.bind(raw);
    let live = 0;
    raw.createBuffer = (descriptor) => {
        const buffer = original(descriptor);
        if (descriptor.label === "shallot-readback-staging") {
            live++;
            const destroy = buffer.destroy.bind(buffer);
            let destroyed = false;
            buffer.destroy = () => {
                if (!destroyed) {
                    live--;
                    destroyed = true;
                }
                destroy();
            };
        }
        return buffer;
    };
    const first = await build({ defaults: false, plugins: [], device });
    const second = await build({ defaults: false, plugins: [], device });
    const source = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    try {
        device.queue.writeBuffer(source, 0, new Uint32Array([17]));
        const [a, b] = await Promise.all([
            probeBuffer(first.state, source),
            probeBuffer(second.state, source),
        ]);
        expect(first.state.readback).not.toBe(second.state.readback);
        expect(live).toBe(2);
        first.dispose();
        expect(live).toBe(1);
        expect(new Uint32Array(b.bytes)[0]).toBe(17);
        expect(new Uint32Array(a.bytes)[0]).toBe(17);
        await probeBuffer(second.state, source);
        expect(live).toBe(1);
        second.dispose();
        expect(live).toBe(0);
    } finally {
        source.destroy();
        first.dispose();
        second.dispose();
        raw.createBuffer = original;
        device.destroy();
    }
});

test("a request after device loss creates no staging", async () => {
    const device = await freshDevice();
    const app = await build({ defaults: false, plugins: [], device });
    const source = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC });
    const pool = app.state.readback;
    try {
        device.destroy();
        await bounded("lost readback device notification", device.lost);
        await expect(probeBuffer(app.state, source)).rejects.toThrow("disposed");
        expect(pool.allocated).toBe(0);
    } finally {
        app.dispose();
    }
});

test("device loss rejects a pending request and releases staging", async () => {
    const device = await freshDevice();
    const app = await build({ defaults: false, plugins: [], device });
    const source = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC });
    const pool = app.state.readback;
    try {
        const pending = probeBuffer(app.state, source);
        // Install rejection observation before destroying the device.
        const outcome = pending.then(
            () => undefined,
            (error: unknown) => error,
        );
        device.destroy();
        await bounded("pending readback device loss notification", device.lost);
        expect(await outcome).toBeDefined();
        expect(pool.allocated).toBe(0);
    } finally {
        app.dispose();
    }
});

test("a stalled readback rejects with its copy label, frame and tick within the GPU ceiling", async () => {
    await controlledReadback(async (state, slots) => {
        state.step(1 / 60);
        const pending = state.readback.request(4, "stalled counter", () => {});
        await expect(pending).rejects.toThrow(
            "stalled counter: frame 1 tick 1 readback map exceeded 750 ms",
        );
        expect(slots[0].destroyed).toBe(true);
        expect(state.readback.allocated).toBe(0);
    });
});
