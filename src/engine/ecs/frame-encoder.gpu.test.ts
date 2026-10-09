import { expect, setDefaultTimeout, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp, type Plugin } from "../app";
import { probeBuffer } from "../runtime";
import type { World } from "./world";

setDefaultTimeout(CEILING.gpu);
await setupGlobals();

const outputLayout = tgpu.bindGroupLayout({
    output: { storage: d.arrayOf(d.u32), access: "mutable" },
});
const increment = tgpu.computeFn({
    workgroupSize: [1],
    in: { gid: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    if (input.gid.x === 0) outputLayout.$.output[0] = outputLayout.$.output[0] + 1;
});

const runtime: {
    enabled: boolean;
    output?: GPUBuffer;
    pipeline?: GPUComputePipeline;
    group?: GPUBindGroup;
} = { enabled: true };

const ComputePlugin: Plugin = {
    name: "FrameEncoderCompute",
    gpu: {},
    initialize(world) {
        const device = world.gpu.device;
        runtime.output = device.createBuffer({
            label: "frame-encoder-compute-output",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        world.own(runtime.output);
        device.queue.writeBuffer(runtime.output, 0, new Uint32Array([0]));
        runtime.pipeline = world.gpu.root.unwrap(
            world.gpu.root.createComputePipeline({ compute: increment }),
        );
        runtime.group = device.createBindGroup({
            layout: runtime.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: runtime.output } }],
        });
    },
    systems: [
        {
            group: "draw",
            update(world) {
                if (!runtime.enabled) return;
                const encoder = world.frameEncoder();
                if (!encoder) throw new Error("GPU device did not provide a frame encoder");
                const pass = encoder.beginComputePass();
                pass.setPipeline(runtime.pipeline!);
                pass.setBindGroup(0, runtime.group!);
                pass.dispatchWorkgroups(1);
                pass.end();
            },
        },
    ],
};

test("compute-only steps submit their lazy frame encoder once and idle device steps submit nothing", async () => {
    const app = await createApp({ defaults: false, plugins: [ComputePlugin] });
    const { world } = app;
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
    try {
        console.info("frame-encoder adapter:", world.gpu.adapter);
        expect(() => world.frameEncoder()).toThrow("only during the draw group");
        for (const expected of [1, 2]) {
            const issued = world.gpu.fences.issued;
            const before = submissions;
            world.step(0);
            expect(submissions - before).toBe(1);
            expect(world.frameFence).toBeDefined();
            expect(world.gpu.fences.issued).toBe(issued + 1);
            await world.frameFence;
            const result = await probeBuffer(world, runtime.output!);
            expect(new Uint32Array(result.bytes)[0]).toBe(expected);
        }

        runtime.enabled = false;
        const before = submissions;
        const issued = world.gpu.fences.issued;
        world.step(0);
        expect(submissions).toBe(before);
        expect(world.frameFence).toBeUndefined();
        expect(world.gpu.fences.issued).toBe(issued);
    } finally {
        runtime.enabled = true;
        if (descriptor) Object.defineProperty(queue, "submit", descriptor);
        else Reflect.deleteProperty(queue, "submit");
        app.dispose();
    }
});

test("a propagated draw failure discards its commands and fence", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [{ name: "DrawFailure", gpu: {} }],
    });
    const { world } = app;
    const device = world.gpu.device;
    const queue = device.queue;
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
    const buffer = device.createBuffer({
        label: "discarded-draw-buffer",
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    world.own(buffer);
    device.queue.writeBuffer(buffer, 0, new Uint32Array([37]));
    const clearAndThrow = {
        group: "draw" as const,
        update(world: World) {
            world.frameEncoder()!.clearBuffer(buffer);
            throw new Error("discard this draw");
        },
    };
    world.addSystem(clearAndThrow);
    const frame = world.gpu.frame;
    const issued = world.gpu.fences.issued;
    try {
        device.pushErrorScope("validation");
        expect(() => world.step(0)).toThrow("discard this draw");
        expect(submissions).toBe(0);
        expect(world.frameFence).toBeUndefined();
        expect(world.gpu.fences.issued).toBe(issued);
        expect(world.gpu.frame).toBe(frame);
        expect(await device.popErrorScope()).toBeNull();
        const result = await probeBuffer(world, buffer);
        expect(new Uint32Array(result.bytes)[0]).toBe(37);
        submissions = 0;

        world.removeSystem(clearAndThrow);
        world.addSystem({
            group: "draw",
            update(world) {
                world.frameEncoder()!.beginComputePass();
                throw new Error("discard open pass");
            },
        });
        const openPassFrame = world.gpu.frame;
        const openPassFence = world.gpu.fences.issued;
        device.pushErrorScope("validation");
        expect(() => world.step(0)).toThrow("discard open pass");
        expect(submissions).toBe(0);
        expect(world.frameFence).toBeUndefined();
        expect(world.gpu.fences.issued).toBe(openPassFence);
        expect(world.gpu.frame).toBe(openPassFrame);
        expect(await device.popErrorScope()).toBeNull();
    } finally {
        if (descriptor) Object.defineProperty(queue, "submit", descriptor);
        else Reflect.deleteProperty(queue, "submit");
        app.dispose();
    }
});
