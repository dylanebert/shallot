import { initGPU, shouldSkipGPU } from "./helpers/gpu";
import { describe, test, expect, beforeAll, beforeEach } from "bun:test";
import { readBuffer, readFloat32, readUint32 } from "../src/standard/compute/readback";
import { capacity, write, clearBuf } from "../src/engine/ecs/capacity";
import { Shape } from "../src/engine/utils";
import { Part, PartGeometry, MeshShape, clearMeshes, getMeshId } from "../src/standard/render/mesh";

const ENTITY_ID_SIZE = 4;
const SHAPE_STRIDE = 4;

function createEntityIdBuffer(device: GPUDevice, maxInstances: number): GPUBuffer {
    return device.createBuffer({
        label: "entityIds",
        size: maxInstances * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
}

function createPropertyBuffer(
    device: GPUDevice,
    config: { maxEntities: number; stride: number },
): GPUBuffer {
    return device.createBuffer({
        label: "property",
        size: config.maxEntities * config.stride,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
}

describe("GPU", () => {
    let device: GPUDevice | null = null;

    beforeAll(async () => {
        const skipReason = shouldSkipGPU();
        if (skipReason) return;
        const ctx = await initGPU();
        device = ctx.device;
    });

    describe("readback", () => {
        test("readBuffer reads contents", async () => {
            if (!device) return;

            const buffer = device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });

            device.queue.writeBuffer(buffer, 0, new Float32Array([1, 2, 3, 4]));

            const data = await readBuffer(device, buffer, 16);
            const floats = new Float32Array(data);

            expect(Array.from(floats)).toEqual([1, 2, 3, 4]);

            buffer.destroy();
        });

        test("readFloat32 reads as Float32Array", async () => {
            if (!device) return;

            const buffer = device.createBuffer({
                size: 12,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });

            device.queue.writeBuffer(buffer, 0, new Float32Array([1.5, 2.5, 3.5]));

            const data = await readFloat32(device, buffer, 3);

            expect(Array.from(data)).toEqual([1.5, 2.5, 3.5]);

            buffer.destroy();
        });

        test("readUint32 reads as Uint32Array", async () => {
            if (!device) return;

            const buffer = device.createBuffer({
                size: 12,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });

            device.queue.writeBuffer(buffer, 0, new Uint32Array([100, 200, 300]));

            const data = await readUint32(device, buffer, 3);

            expect(Array.from(data)).toEqual([100, 200, 300]);

            buffer.destroy();
        });
    });

    describe("entity ID buffer", () => {
        test("creates buffer with correct size", () => {
            if (!device) return;

            const buffer = createEntityIdBuffer(device, 100);
            expect(buffer.size).toBe(100 * ENTITY_ID_SIZE);
        });

        test("writes and reads entity IDs", async () => {
            if (!device) return;

            const buffer = createEntityIdBuffer(device, 10);
            const entityIds = new Uint32Array([5, 2, 8, 0, 42]);

            device.queue.writeBuffer(buffer, 0, entityIds);

            const data = await readBuffer(device, buffer, 5 * ENTITY_ID_SIZE);
            const result = new Uint32Array(data);

            expect(result[0]).toBe(5);
            expect(result[1]).toBe(2);
            expect(result[2]).toBe(8);
            expect(result[3]).toBe(0);
            expect(result[4]).toBe(42);
        });
    });

    describe("property buffer", () => {
        test("creates buffer with correct size for stride", () => {
            if (!device) return;

            const buffer = createPropertyBuffer(device, { maxEntities: 100, stride: 16 });
            expect(buffer.size).toBe(100 * 16);
        });

        test("batch writes entire property array", async () => {
            if (!device) return;

            const maxEntities = 10;
            const stride = 12;
            const buffer = createPropertyBuffer(device, { maxEntities, stride });

            const velocities = new Float32Array(maxEntities * 3);
            velocities[2 * 3 + 0] = 1.0;
            velocities[2 * 3 + 1] = 2.0;
            velocities[2 * 3 + 2] = 3.0;
            velocities[5 * 3 + 0] = 5.0;
            velocities[5 * 3 + 1] = 6.0;
            velocities[5 * 3 + 2] = 7.0;

            device.queue.writeBuffer(buffer, 0, velocities);

            const data = await readBuffer(device, buffer, buffer.size);
            const view = new DataView(data);

            expect(view.getFloat32(2 * stride, true)).toBeCloseTo(1.0);
            expect(view.getFloat32(2 * stride + 4, true)).toBeCloseTo(2.0);
            expect(view.getFloat32(2 * stride + 8, true)).toBeCloseTo(3.0);

            expect(view.getFloat32(5 * stride, true)).toBeCloseTo(5.0);
            expect(view.getFloat32(5 * stride + 4, true)).toBeCloseTo(6.0);
            expect(view.getFloat32(5 * stride + 8, true)).toBeCloseTo(7.0);
        });
    });

    describe("entity ID → property indirection", () => {
        test("simulates shader lookup pattern", async () => {
            if (!device) return;

            const maxEntities = 100;

            const entityIdBuffer = createEntityIdBuffer(device, 10);
            device.queue.writeBuffer(entityIdBuffer, 0, new Uint32Array([5, 2, 8]));

            const stride = 16;
            const colorBuffer = createPropertyBuffer(device, { maxEntities, stride });

            const colors = new Float32Array(maxEntities * 4);
            colors[2 * 4 + 0] = 0.0;
            colors[2 * 4 + 1] = 1.0;
            colors[2 * 4 + 2] = 0.0;
            colors[2 * 4 + 3] = 1.0;
            colors[5 * 4 + 0] = 1.0;
            colors[5 * 4 + 1] = 0.0;
            colors[5 * 4 + 2] = 0.0;
            colors[5 * 4 + 3] = 1.0;
            colors[8 * 4 + 0] = 0.0;
            colors[8 * 4 + 1] = 0.0;
            colors[8 * 4 + 2] = 1.0;
            colors[8 * 4 + 3] = 1.0;

            device.queue.writeBuffer(colorBuffer, 0, colors);

            const entityIds = new Uint32Array(
                await readBuffer(device, entityIdBuffer, 3 * ENTITY_ID_SIZE),
            );
            const colorData = await readBuffer(device, colorBuffer, maxEntities * stride);
            const colorView = new DataView(colorData);

            const eid0 = entityIds[0];
            expect(eid0).toBe(5);
            expect(colorView.getFloat32(eid0 * stride, true)).toBeCloseTo(1.0);

            const eid1 = entityIds[1];
            expect(eid1).toBe(2);
            expect(colorView.getFloat32(eid1 * stride + 4, true)).toBeCloseTo(1.0);

            const eid2 = entityIds[2];
            expect(eid2).toBe(8);
            expect(colorView.getFloat32(eid2 * stride + 8, true)).toBeCloseTo(1.0);
        });
    });

    describe("shapes buffer", () => {
        beforeEach(() => {
            clearMeshes();
            clearBuf(PartGeometry);
        });

        test("PartGeometry is chunked Uint32Array storage", () => {
            expect(PartGeometry.chunks[0]).toBeInstanceOf(Uint32Array);
            expect(PartGeometry.chunks[0].length).toBe(capacity());
        });

        test("getMeshId maps Shape enum to mesh registry IDs", () => {
            const eid = 5;
            Part.shape[eid] = Shape.Box;
            expect(getMeshId(eid)).toBe(MeshShape.Box);

            Part.shape[eid] = Shape.Sphere;
            expect(getMeshId(eid)).toBe(MeshShape.Sphere);

            Part.shape[eid] = Shape.Plane;
            expect(getMeshId(eid)).toBe(MeshShape.Plane);
        });

        test("geometry buffer uploads correctly", async () => {
            if (!device) return;

            const buffer = createPropertyBuffer(device, {
                maxEntities: capacity(),
                stride: SHAPE_STRIDE,
            });

            Part.shape[0] = Shape.Box;
            Part.shape[1] = Shape.Sphere;
            Part.shape[2] = Shape.Plane;
            Part.shape[42] = Shape.Sphere;

            for (const eid of [0, 1, 2, 42]) {
                PartGeometry.chunks[0][eid] = getMeshId(eid);
            }

            write(device.queue, buffer, 0, PartGeometry, capacity());

            const data = await readBuffer(device, buffer, capacity() * SHAPE_STRIDE);
            const result = new Uint32Array(data);

            expect(result[0]).toBe(MeshShape.Box);
            expect(result[1]).toBe(MeshShape.Sphere);
            expect(result[2]).toBe(MeshShape.Plane);
            expect(result[42]).toBe(MeshShape.Sphere);

            buffer.destroy();
        });

        test("geometry buffer supports sparse entity IDs", async () => {
            if (!device) return;

            const buffer = createPropertyBuffer(device, {
                maxEntities: capacity(),
                stride: SHAPE_STRIDE,
            });

            const entities = [10, 100, 500, 1000];
            const shapes = [Shape.Box, Shape.Sphere, Shape.Plane, Shape.Box];
            const expectedGeometry = [
                MeshShape.Box,
                MeshShape.Sphere,
                MeshShape.Plane,
                MeshShape.Box,
            ];

            for (let i = 0; i < entities.length; i++) {
                Part.shape[entities[i]] = shapes[i];
                PartGeometry.chunks[0][entities[i]] = getMeshId(entities[i]);
            }

            write(device.queue, buffer, 0, PartGeometry, capacity());

            const data = await readBuffer(device, buffer, capacity() * SHAPE_STRIDE);
            const result = new Uint32Array(data);

            for (let i = 0; i < entities.length; i++) {
                expect(result[entities[i]]).toBe(expectedGeometry[i]);
            }

            buffer.destroy();
        });
    });
});
