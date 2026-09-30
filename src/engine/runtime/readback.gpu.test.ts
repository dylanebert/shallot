import { expect, setDefaultTimeout, test } from "bun:test";
import createRenderedSubject from "../../../diagnostics/readback-allocation/render.entry";
import { build } from "../app";
import { rawDevice } from "./gpu";
import { probeBuffer, probeTexture } from "./probe";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

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

test("rendered frames without a request map nothing", async () => {
    const owner = await build({ defaults: false, plugins: [] });
    const device = rawDevice(owner.state.gpu.device);
    const original = device.createBuffer.bind(device);
    let maps = 0;
    device.createBuffer = (descriptor) => {
        const buffer = original(descriptor);
        const map = buffer.mapAsync.bind(buffer);
        buffer.mapAsync = (...args) => {
            maps++;
            return map(...args);
        };
        return buffer;
    };
    let subject: Awaited<ReturnType<typeof createRenderedSubject>> | undefined;
    try {
        subject = await createRenderedSubject("", device);
        for (let i = 0; i < 482; i++) subject.step();
        await subject.wait();
        expect(maps).toBe(0);
    } finally {
        device.createBuffer = original;
        subject?.dispose();
        owner.dispose();
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

test("a deterministic consumer cannot read bytes or stamps requested by another world", async () => {
    const producer = await build({ defaults: false, plugins: [] });
    const source = producer.state.gpu.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_SRC,
    });
    const result = await probeBuffer(producer.state, source);
    let refusals = 0;
    const consumer = await build({
        defaults: false,
        plugins: [
            {
                name: "CrossWorldConsumer",
                systems: [
                    {
                        group: "fixed",
                        update() {
                            for (const read of [
                                () => result.bytes,
                                () => result.frame,
                                () => result.fixedTick,
                            ]) {
                                try {
                                    read();
                                } catch (error) {
                                    if (String(error).includes("CrossWorldConsumer")) refusals++;
                                    else throw error;
                                }
                            }
                        },
                    },
                ],
            },
        ],
    });
    try {
        producer.dispose();
        consumer.state.step(1 / 60);
        expect(refusals).toBe(3);
    } finally {
        source.destroy();
        producer.dispose();
        consumer.dispose();
    }
});

for (const deterministic of [true, false]) {
    test(`fixed readback ${deterministic ? "refuses deterministic" : "allows explicitly non-deterministic"} plugins`, async () => {
        let result: Awaited<ReturnType<typeof probeBuffer>> | undefined;
        let error: unknown;
        let reads = 0;
        const app = await build({
            defaults: false,
            plugins: [
                {
                    name: "ReadbackConsumer",
                    deterministic,
                    systems: [
                        {
                            group: "fixed",
                            update() {
                                if (!result) return;
                                try {
                                    new Uint32Array(result.bytes);
                                    reads++;
                                } catch (cause) {
                                    error = cause;
                                }
                            },
                        },
                    ],
                },
            ],
        });
        const state = app.state;
        const source = state.gpu.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC });
        try {
            result = await probeBuffer(state, source);
            state.step(1 / 60);
            if (deterministic) {
                expect(reads).toBe(0);
                expect(String(error)).toContain("ReadbackConsumer");
                expect(String(error)).toContain("deterministic: false");
            } else {
                expect(reads).toBe(1);
                expect(error).toBeUndefined();
            }
        } finally {
            source.destroy();
            app.dispose();
        }
    });
}
