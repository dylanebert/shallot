import { expect, setDefaultTimeout, test } from "bun:test";
import { sharedGpuBuild } from "../app/gpu.fixture";
import { probeBuffer, probeTexture } from "./probe";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();
const build = await sharedGpuBuild();

test("one-shot readback stamps its copy and reuses then releases world staging", async () => {
    const app = await build({ defaults: false, plugins: [] });
    const state = app.state;
    const device = state.gpu.device;
    const source = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    try {
        state.step(1 / 60);
        device.queue.writeBuffer(source, 0, new Uint32Array([10, 20, 30, 40]));
        const frame = state.gpu.frame;
        const fixedTick = state.time.fixedTick;
        const pending = probeBuffer(state, source, { offset: 4, size: 4 });
        state.step(1 / 60);
        const first = await pending;
        expect(first.frame).toBe(frame);
        expect(first.fixedTick).toBe(fixedTick);
        expect(new Uint32Array(first.bytes)[0]).toBe(20);
        expect(state.readback.allocated).toBe(1);
        await probeBuffer(state, source, { size: 4 });
        expect(state.readback.allocated).toBe(1);
        state.readback.maxUnusedFrames = 2;
        state.step(1 / 60);
        state.step(1 / 60);
        expect(state.readback.allocated).toBe(0);
    } finally {
        source.destroy();
        app.dispose();
    }
});

test("a texture request shares buffer staging and returns tightly packed owned bytes", async () => {
    const app = await build({ defaults: false, plugins: [] });
    const state = app.state;
    const device = state.gpu.device;
    const texture = device.createTexture({
        size: [1, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
    const buffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    try {
        device.queue.writeTexture(
            { texture },
            new Uint8Array([1, 2, 3, 4]),
            { bytesPerRow: 4 },
            [1, 1],
        );
        const result = await probeTexture(state, texture);
        expect([...new Uint8Array(result.bytes)]).toEqual([1, 2, 3, 4]);
        await probeBuffer(state, buffer);
        expect(state.readback.allocated).toBe(1);
    } finally {
        texture.destroy();
        buffer.destroy();
        app.dispose();
    }
});

test("readback bytes and stamps are plain owned data that survive world disposal", async () => {
    const app = await build({ defaults: false, plugins: [] });
    try {
        const source = app.state.gpu.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        app.state.gpu.device.queue.writeBuffer(source, 0, new Uint32Array([17]));
        const result = await probeBuffer(app.state, source);
        const retained = new Uint32Array(result.bytes);
        for (const field of ["bytes", "frame", "fixedTick"]) {
            expect(Object.getOwnPropertyDescriptor(result, field)?.get).toBeUndefined();
        }
        app.dispose();
        expect(retained[0]).toBe(17);
        expect(new Uint32Array(result.bytes)[0]).toBe(17);
        expect(result.frame).toBe(0);
        expect(result.fixedTick).toBe(0);
    } finally {
        app.dispose();
    }
});
