import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { compileGpuFile, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../app";
import { probeBuffer } from "./probe";
import { countStaging } from "./readback.fixture";

setDefaultTimeout(CEILING.gpu);
const worlds: Awaited<ReturnType<typeof createApp>>[] = [];
const devices: GPUDevice[] = [];
const subjects = compileGpuFile(import.meta.path, async () => {
    let counts!: ReturnType<typeof countStaging>["counts"];
    for (let i = 0; i < 3; i++) {
        const device = await freshDevice();
        devices.push(device);
        const count = i === 0 ? 2 : 1;
        const tracker = i === 0 ? countStaging(device) : undefined;
        if (tracker) counts = tracker.counts;
        try {
            for (let j = 0; j < count; j++)
                worlds.push(await createApp({ defaults: false, plugins: [], device }));
        } finally {
            tracker?.restore();
        }
    }
    return { worlds, counts };
});
afterAll(async () => {
    await disposeGpuApps(worlds);
    for (const device of devices) device.destroy();
});

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
    const {
        worlds: [first, second],
        counts,
    } = subjects();
    const device = devices[0];
    const source = first.world.gpu.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const secondSource = second.world.gpu.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    try {
        device.queue.writeBuffer(source, 0, new Uint32Array([17]));
        device.queue.writeBuffer(secondSource, 0, new Uint32Array([17]));
        const [a, b] = await Promise.all([
            probeBuffer(first.world, source),
            probeBuffer(second.world, secondSource),
        ]);
        expect(first.world.readback).not.toBe(second.world.readback);
        expect(counts.live).toBe(2);
        first.dispose();
        expect(counts.live).toBe(1);
        expect(new Uint32Array(b.bytes)[0]).toBe(17);
        expect(new Uint32Array(a.bytes)[0]).toBe(17);
        await probeBuffer(second.world, secondSource);
        expect(counts.live).toBe(1);
        second.dispose();
        expect(counts.live).toBe(0);
    } finally {
        source.destroy();
        secondSource.destroy();
        first.dispose();
        second.dispose();
        device.destroy();
    }
});

test("a request after device loss creates no staging", async () => {
    const device = devices[1];
    const app = subjects().worlds[2];
    const source = app.world.gpu.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC });
    const pool = app.world.readback;
    try {
        device.destroy();
        await bounded("lost readback device notification", device.lost);
        await expect(probeBuffer(app.world, source)).rejects.toThrow("disposed");
        expect(pool.allocated).toBe(0);
    } finally {
        app.dispose();
    }
});

test("device loss rejects a pending request and releases staging", async () => {
    const device = devices[2];
    const app = subjects().worlds[3];
    const source = app.world.gpu.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC });
    const pool = app.world.readback;
    try {
        const pending = probeBuffer(app.world, source, { label: "lost counter" });
        // Install rejection observation before destroying the device.
        const outcome = pending.then(
            () => new Error("readback unexpectedly resolved"),
            (error: unknown) => error,
        );
        device.destroy();
        await bounded("pending readback device loss notification", device.lost);
        const failure = await outcome;
        expect(failure).toBeInstanceOf(Error);
        expect((failure as Error).message).toMatch(
            /^lost counter: frame \d+ tick \d+ readback failed: /,
        );
        expect(pool.allocated).toBe(0);
    } finally {
        app.dispose();
    }
});
