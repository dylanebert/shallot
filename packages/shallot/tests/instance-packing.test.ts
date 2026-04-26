import { initGPU, shouldSkipGPU } from "./helpers/gpu";
import { readBuffer } from "../src/standard/compute/readback";
import { describe, test, expect, beforeAll, beforeEach } from "bun:test";
import {
    registerProperties,
    clearProperties,
    instancePackingShader,
    instanceStride,
    propertyCount,
} from "../src/standard/render/surface";
import type { Property } from "../src/standard/render/surface";

const skipReason = shouldSkipGPU();
let device: GPUDevice;

beforeAll(async () => {
    if (skipReason) return;
    const ctx = await initGPU();
    device = ctx.device;
});

beforeEach(() => {
    clearProperties();
});

function createPackingPipeline(device: GPUDevice, shader: string) {
    const module = device.createShaderModule({ code: shader });
    const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
    });
    return { module, pipeline };
}

async function runPacking(
    fields: Property[],
    entityCount: number,
    sourceValues: number[][],
): Promise<DataView> {
    registerProperties(fields);
    const shader = instancePackingShader();
    const stride = instanceStride();
    const { pipeline } = createPackingPipeline(device, shader);

    const soaData = new Float32Array(fields.length * entityCount);
    for (let f = 0; f < fields.length; f++) {
        for (let e = 0; e < entityCount; e++) {
            soaData[f * entityCount + e] = sourceValues[f][e];
        }
    }

    const sourceBuffer = device.createBuffer({
        size: soaData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(sourceBuffer, 0, soaData);

    const outputBuffer = device.createBuffer({
        size: entityCount * stride,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const countBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(countBuffer, 0, new Uint32Array([entityCount]));

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sourceBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
            { binding: 2, resource: { buffer: countBuffer } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(entityCount / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);

    const result = await readBuffer(device, outputBuffer, entityCount * stride);

    sourceBuffer.destroy();
    outputBuffer.destroy();
    countBuffer.destroy();

    return new DataView(result);
}

describe("instance packing GPU round-trip", () => {
    test.skipIf(!!skipReason)("single f32 field", async () => {
        const entityCount = 4;
        const values = [3.14, -1.0, 0.0, 999.5];

        const view = await runPacking([{ name: "height", type: "f32" }], entityCount, [values]);

        const stride = instanceStride();
        expect(stride).toBe(16);

        for (let i = 0; i < entityCount; i++) {
            expect(view.getFloat32(i * stride, true)).toBeCloseTo(values[i], 4);
        }
    });

    test.skipIf(!!skipReason)("mixed types: f32 + u32 + i32", async () => {
        const entityCount = 3;
        const floatValues = [1.5, 2.5, 3.5];
        const uintValues = [10, 20, 30];
        const intValues = [-5, 0, 42];

        const fields: Property[] = [
            { name: "speed", type: "f32" },
            { name: "flags", type: "u32" },
            { name: "health", type: "i32" },
        ];

        const soaBuf = new ArrayBuffer(fields.length * entityCount * 4);
        const soaU32 = new Uint32Array(soaBuf);
        const soaF32 = new Float32Array(soaBuf);
        for (let e = 0; e < entityCount; e++) {
            soaF32[0 * entityCount + e] = floatValues[e];
            soaU32[1 * entityCount + e] = uintValues[e];
            new DataView(soaBuf).setInt32((2 * entityCount + e) * 4, intValues[e], true);
        }

        registerProperties(fields);
        const shader = instancePackingShader();
        const stride = instanceStride();
        expect(stride).toBe(16);

        const { pipeline } = createPackingPipeline(device, shader);

        const sourceBuffer = device.createBuffer({
            size: soaBuf.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(sourceBuffer, 0, soaU32);

        const outputBuffer = device.createBuffer({
            size: entityCount * stride,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const countBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(countBuffer, 0, new Uint32Array([entityCount]));

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: sourceBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: countBuffer } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(entityCount / 64));
        pass.end();
        device.queue.submit([encoder.finish()]);

        const result = await readBuffer(device, outputBuffer, entityCount * stride);
        const view = new DataView(result);

        for (let i = 0; i < entityCount; i++) {
            const base = i * stride;
            expect(view.getFloat32(base + 0, true)).toBeCloseTo(floatValues[i], 4);
            expect(view.getUint32(base + 4, true)).toBe(uintValues[i]);
            expect(view.getInt32(base + 8, true)).toBe(intValues[i]);
        }

        sourceBuffer.destroy();
        outputBuffer.destroy();
        countBuffer.destroy();
    });

    test.skipIf(!!skipReason)("max fields (128-byte cap with padding)", async () => {
        const maxBytes = 128;
        const maxFields = maxBytes / 4;
        const fields: Property[] = [];
        for (let i = 0; i < maxFields; i++) {
            fields.push({ name: `f${i}`, type: "f32" });
        }

        const entityCount = 2;
        const sourceValues: number[][] = [];
        for (let f = 0; f < maxFields; f++) {
            sourceValues.push([f * 10 + 1, f * 10 + 2]);
        }

        const view = await runPacking(fields, entityCount, sourceValues);
        const stride = instanceStride();
        expect(stride).toBe(128);
        expect(propertyCount()).toBe(maxFields);

        for (let e = 0; e < entityCount; e++) {
            for (let f = 0; f < maxFields; f++) {
                expect(view.getFloat32(e * stride + f * 4, true)).toBeCloseTo(
                    sourceValues[f][e],
                    4,
                );
            }
        }
    });

    test.skipIf(!!skipReason)("fields beyond 128-byte cap are skipped", () => {
        const fields: Property[] = [];
        for (let i = 0; i < 33; i++) {
            fields.push({ name: `f${i}`, type: "f32" });
        }

        registerProperties(fields);
        expect(propertyCount()).toBe(32);
    });

    test.skipIf(!!skipReason)(
        "entity count boundary — threads beyond count produce no output",
        async () => {
            const entityCount = 3;
            const values = [1.0, 2.0, 3.0];

            registerProperties([{ name: "val", type: "f32" }]);
            const shader = instancePackingShader();
            const stride = instanceStride();
            const { pipeline } = createPackingPipeline(device, shader);

            const soaData = new Float32Array(entityCount);
            for (let i = 0; i < entityCount; i++) soaData[i] = values[i];

            const totalSlots = 64;
            const sourceBuffer = device.createBuffer({
                size: totalSlots * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(sourceBuffer, 0, soaData);

            const outputBuffer = device.createBuffer({
                size: totalSlots * stride,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });

            const countBuffer = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(countBuffer, 0, new Uint32Array([entityCount]));

            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: sourceBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                    { binding: 2, resource: { buffer: countBuffer } },
                ],
            });

            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1);
            pass.end();
            device.queue.submit([encoder.finish()]);

            const result = await readBuffer(device, outputBuffer, totalSlots * stride);
            const view = new DataView(result);

            for (let i = 0; i < entityCount; i++) {
                expect(view.getFloat32(i * stride, true)).toBeCloseTo(values[i], 4);
            }

            for (let i = entityCount; i < totalSlots; i++) {
                expect(view.getFloat32(i * stride, true)).toBe(0);
            }

            sourceBuffer.destroy();
            outputBuffer.destroy();
            countBuffer.destroy();
        },
    );
});
