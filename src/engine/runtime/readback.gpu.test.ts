import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { probeBuffer, probeTexture } from "./probe";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(
    import.meta.path,
    Array.from({ length: 3 }, () => ({ defaults: false, plugins: [] })),
);

test("one-shot readback stamps its copy and reuses then releases world staging", async () => {
    const app = subjects()[0];
    const world = app.world;
    const device = world.gpu.device;
    const source = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    try {
        world.step(1 / 60);
        device.queue.writeBuffer(source, 0, new Uint32Array([10, 20, 30, 40]));
        const frame = world.gpu.frame;
        const fixedTick = world.time.fixedTick;
        const pending = probeBuffer(world, source, { offset: 4, size: 4 });
        world.step(1 / 60);
        const first = await pending;
        expect(first.frame).toBe(frame);
        expect(first.fixedTick).toBe(fixedTick);
        expect(new Uint32Array(first.bytes)[0]).toBe(20);
        expect(world.readback.allocated).toBe(1);
        await probeBuffer(world, source, { size: 4 });
        expect(world.readback.allocated).toBe(1);
        world.readback.maxUnusedFrames = 2;
        world.step(1 / 60);
        world.step(1 / 60);
        expect(world.readback.allocated).toBe(0);
    } finally {
        source.destroy();
        app.dispose();
    }
});

test("a texture request shares buffer staging and returns tightly packed owned bytes", async () => {
    const app = subjects()[1];
    const world = app.world;
    const device = world.gpu.device;
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
        const result = await probeTexture(world, texture);
        expect([...new Uint8Array(result.bytes)]).toEqual([1, 2, 3, 4]);
        await probeBuffer(world, buffer);
        expect(world.readback.allocated).toBe(1);
    } finally {
        texture.destroy();
        buffer.destroy();
        app.dispose();
    }
});

test("readback bytes and stamps are plain owned data that survive world disposal", async () => {
    const app = subjects()[2];
    try {
        const source = app.world.gpu.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        app.world.gpu.device.queue.writeBuffer(source, 0, new Uint32Array([17]));
        const result = await probeBuffer(app.world, source);
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
