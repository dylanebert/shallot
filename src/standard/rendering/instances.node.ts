import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { MeshInstance } from "../../core/mesh";
import { globalTransformTable } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { Surfaces } from "./contract";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { meshInstanceTable } from "./preprocess";
import { Draws } from "./registry";
import "../../standard";

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after 5000 ms`)), 5000);
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

test("MeshInstance compaction carries independent dense GlobalTransform and MeshInstance slots with each logical eid", async () => {
    const app = await createApp({ plugins: [] });
    const world = app.world;
    const device = world.gpu.device;
    const readback = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        for (let i = 0; i < 1000; i++) world.create();
        const extra = world.create();
        world.add(extra, Transform);
        const a = world.create();
        const b = world.create();
        world.add(a, Transform);
        world.add(b, Transform);
        // Different membership order forces unrelated row slots.
        world.add(b, MeshInstance);
        world.add(a, MeshInstance);
        const globalTransforms = globalTransformTable(world);
        const meshInstances = meshInstanceTable(world);
        expect(globalTransforms.rowIndex(b)).not.toBe(meshInstances.rowIndex(b));
        device.pushErrorScope("validation");
        world.step();
        const instances = world.gpu.buffers.get("eids");
        if (!instances) throw new Error("MeshInstance did not publish its instance list");
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(instances, 0, readback, 0, 32);
        device.queue.submit([encoder.finish()]);
        const error = await bounded("MeshInstance payload validation", device.popErrorScope());
        if (error) throw new Error(error.message);
        await bounded("MeshInstance payload readback", readback.mapAsync(GPUMapMode.READ));
        const words = new Uint32Array(readback.getMappedRange());
        const records = [Array.from(words.subarray(0, 4)), Array.from(words.subarray(4, 8))].sort(
            (x, y) => x[0]! - y[0]!,
        );
        expect(records).toEqual(
            [a, b].map((eid) => [
                eid,
                globalTransforms.rowIndex(eid),
                meshInstances.rowIndex(eid) + 1,
                0,
            ]),
        );
        readback.unmap();
    } finally {
        readback.destroy();
        app.dispose();
    }
});

test("mesh instances return to default draws when MeshMaterial is removed", async () => {
    const app = await createApp({ plugins: [] });
    const world = app.world;
    const device = world.gpu.device;
    const readback = device.createBuffer({
        size: 60,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, MeshInstance);
        async function counts() {
            device.pushErrorScope("validation");
            world.step();
            const encoder = device.createCommandEncoder();
            for (const [index, surface] of ["default", "unlit", "vertex"].entries()) {
                const draw = Array.from(world.resource(Draws)).find(
                    (draw) => draw.surface === surface && draw.mesh === "cube",
                )!;
                encoder.copyBufferToBuffer(
                    world.gpu.root.unwrap(draw.args.indirect),
                    draw.args.offset ?? 0,
                    readback,
                    index * 20,
                    20,
                );
            }
            device.queue.submit([encoder.finish()]);
            expect(
                await bounded("surface selection validation", device.popErrorScope()),
            ).toBeNull();
            await bounded("surface selection readback", readback.mapAsync(GPUMapMode.READ));
            const words = new Uint32Array(readback.getMappedRange());
            const result = [words[1], words[6], words[11]];
            readback.unmap();
            return result;
        }
        expect(await counts()).toEqual([1, 0, 0]);
        const materialIds: number[] = [];
        for (const [index, surface] of ["unlit", "vertex"].entries()) {
            const material = world
                .resource(Materials)
                .add(StandardMaterial({ surface: world.resource(Surfaces).id(surface)! }));
            materialIds.push(material);
            world.add(eid, MeshMaterial, { material });
            expect(await counts()).toEqual(index === 0 ? [0, 1, 0] : [0, 0, 1]);
            world.remove(eid, MeshMaterial);
            expect(await counts()).toEqual([1, 0, 0]);
        }
        const materials = world.resource(Materials);
        const material = materialIds[0]!;
        world.add(eid, MeshMaterial, { material });
        expect(await counts()).toEqual([0, 1, 0]);
        world.storage(MeshMaterial).material.set(eid, materialIds[1]!);
        expect(await counts()).toEqual([0, 0, 1]);
        world.storage(MeshMaterial).material.set(eid, material);
        materials.update(material, { surface: world.resource(Surfaces).id("default")! });
        expect(await counts()).toEqual([1, 0, 0]);
        world.remove(eid, MeshMaterial);
        expect(await counts()).toEqual([1, 0, 0]);
    } finally {
        readback.destroy();
        app.dispose();
    }
});
