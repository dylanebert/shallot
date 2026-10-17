import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compileGpuFile, createTestGpuApp } from "../../../scripts/gpu.fixture";
import { createRegather, prepareRegather, SHADOW_ARG_STRIDE } from "./regather";

setDefaultTimeout(TIMEOUT.node);
const subject = compileGpuFile(import.meta.path, async () => {
    const app = await createTestGpuApp({
        defaults: false,
        plugins: [{ name: "RegatherTest", gpu: { features: ["indirect-first-instance"] } }],
    });
    await prepareRegather(app.world, app.world.gpu.device, 16384);
    return app;
});
afterAll(() => subject().dispose());

test("parallel regather preserves every payload, mesh run and cascade depth against a CPU survivor reference", async () => {
    const { world } = subject();
    const device = world.gpu.device;
    const owned: { destroy(): void }[] = [];
    const buffer = (data: Uint32Array, usage: number): GPUBuffer => {
        const b = device.createBuffer({
            size: data.byteLength,
            usage: usage | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(b, 0, data);
        owned.push(b);
        return b;
    };
    const gather = createRegather(world, "reference-test");
    gather.reset(() => {});
    try {
        // Cross several scan levels, with empty runs around both the large mesh and small meshes.
        const D = 519,
            C = 4,
            capacity = 16384;
        const pairs = Array.from({ length: D }, (_, i) => D - 1 - i);
        const slots = [5, 1, 7, 3];
        const args = new Uint32Array(8 * D * 5);
        const packed = new Uint32Array(8 * capacity * 4);
        const referenceArgs = new Uint32Array(D * 5);
        const survivors: number[] = [];
        let id = 0;
        for (let i = 0; i < D; i++) {
            const base = survivors.length / 4;
            for (let c = 0; c < C; c++) {
                const count =
                    i === 1 ? 4096 - c * 137 : i % 17 === 0 ? 0 : (i + c) % 3 === 0 ? 0 : 1;
                // Distinct disjoint source ranges, unrelated to output mesh order.
                const src = slots[c] * capacity + (i === 1 ? 0 : 5000 + i);
                args.set([3, count, 0, 0, src], (slots[c] * D + pairs[i]) * 5);
                for (let k = 0; k < count; k++) {
                    const payload = [id++, i + 11, k + 23];
                    packed.set([...payload, 99], (src + k) * 4);
                    survivors.push(...payload, c);
                }
            }
            referenceArgs.set([3, survivors.length / 4 - base, 0, 0, base], i * 5);
        }
        const expected = new Uint32Array(survivors);
        const sourceArgs = buffer(args, GPUBufferUsage.STORAGE);
        const sourcePacked = buffer(packed, GPUBufferUsage.STORAGE);
        const cpuArgs = buffer(referenceArgs, GPUBufferUsage.INDIRECT);
        const cpuPacked = buffer(expected, GPUBufferUsage.STORAGE);
        gather.ensure(C, capacity);
        gather.reserve(D);
        const encoder = device.createCommandEncoder();
        const compute = encoder.beginComputePass();
        gather.run(compute, sourceArgs, sourcePacked, slots, C, pairs, D, D);
        compute.end();
        const read = device.createBuffer({
            size: expected.byteLength + referenceArgs.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        owned.push(read);
        encoder.copyBufferToBuffer(gather.eids()!, 0, read, 0, expected.byteLength);
        encoder.copyBufferToBuffer(
            gather.args()!,
            0,
            read,
            expected.byteLength,
            referenceArgs.byteLength,
        );
        const shader = device.createShaderModule({
            code: `
@group(0) @binding(0) var<storage, read> payload: array<vec4u>;
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> @builtin(position) vec4f {
    let p = payload[i];
    let corners = array<vec2f, 3>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1));
    let xy = vec2f(f32(p.x % 128u + (p.w % 2u) * 128u), f32(p.x / 128u + (p.w / 2u) * 128u));
    return vec4f((xy + corners[v] * 1.9) / 128.0 - 1.0, f32(p.y % 100u + 1u) / 128.0, 1);
}`,
        });
        const pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: { module: shader, entryPoint: "vs" },
            primitive: { topology: "triangle-list" },
            depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
        });
        const indices = buffer(new Uint32Array([0, 1, 2]), GPUBufferUsage.INDEX);
        const depthReads: GPUBuffer[] = [];
        for (const [instances, indirect] of [
            [gather.eids()!, gather.args()!],
            [cpuPacked, cpuArgs],
        ]) {
            const texture = device.createTexture({
                size: [256, 256],
                format: "depth32float",
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            });
            owned.push(texture);
            const pass = encoder.beginRenderPass({
                colorAttachments: [],
                depthStencilAttachment: {
                    view: texture.createView(),
                    depthClearValue: 1,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(
                0,
                device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: { buffer: instances } }],
                }),
            );
            pass.setIndexBuffer(indices, "uint32");
            for (let i = 0; i < D; i++) pass.drawIndexedIndirect(indirect, i * SHADOW_ARG_STRIDE);
            pass.end();
            const depth = device.createBuffer({
                size: 256 * 256 * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            owned.push(depth);
            depthReads.push(depth);
            encoder.copyTextureToBuffer(
                { texture, aspect: "depth-only" },
                { buffer: depth, bytesPerRow: 1024 },
                [256, 256],
            );
        }
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const mapped = read.getMappedRange();
        expect(new Uint32Array(mapped, 0, expected.length)).toEqual(expected);
        expect(new Uint32Array(mapped, expected.byteLength)).toEqual(referenceArgs);
        read.unmap();
        await Promise.all(depthReads.map((b) => b.mapAsync(GPUMapMode.READ)));
        const actualDepth = new Float32Array(depthReads[0].getMappedRange());
        expect(actualDepth.some((v) => v < 1)).toBe(true);
        expect(actualDepth).toEqual(new Float32Array(depthReads[1].getMappedRange()));
        for (const b of depthReads) b.unmap();
    } finally {
        gather.dispose();
        for (const resource of owned) resource.destroy();
    }
});

test("one view-independent mesh duplicates its direct payload across every combo", async () => {
    const { world } = subject();
    const device = world.gpu.device;
    const gather = createRegather(world, "direct-reference-test");
    const sourceArgs = device.createBuffer({
        size: 20,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const packed = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const read = device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        device.queue.writeBuffer(sourceArgs, 0, new Uint32Array([3, 2, 0, 0, 0]));
        device.queue.writeBuffer(packed, 0, new Uint32Array([1, 2, 3, 99, 4, 5, 6, 99]));
        gather.ensure(4, 2);
        gather.reserve(1);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        gather.run(pass, sourceArgs, packed, [7, 3, 5, 1], 4, [0], 1, 0);
        pass.end();
        encoder.copyBufferToBuffer(gather.eids()!, 0, read, 0, 128);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        expect(new Uint32Array(read.getMappedRange())).toEqual(
            new Uint32Array(Array.from({ length: 4 }, (_, c) => [1, 2, 3, c, 4, 5, 6, c]).flat()),
        );
        read.unmap();
    } finally {
        gather.dispose();
        sourceArgs.destroy();
        packed.destroy();
        read.destroy();
    }
});
