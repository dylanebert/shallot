import { expect, setDefaultTimeout, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { MeshInstance } from "../../core/mesh";
import { globalTransformTable } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { StandardRenderingPlugin } from "./forward";
import { MaterialPlugin, Materials, MeshMaterial, StandardMaterial } from "./material";
import {
    materialFragmentContext,
    materialLayout,
    materialType,
    materialTypeId,
} from "./material-type";
import { MeshRenderPlugin } from "./mesh-render";
import { meshInstanceTable } from "./preprocess";
import { Draws } from "./registry";
import "../../standard";

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

const TintParams = d.struct({ tint: d.vec4f });
const tintLayout = materialLayout(TintParams, {});
const TintContext = materialFragmentContext();
const tintFragment = tgpu.fn(
    [TintContext],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return TintParams(tintLayout.$.materialParameters[ctx.material]).tint;
});
const TintMaterial = materialType({
    name: "InstanceTypeChangeTint",
    parameters: TintParams,
    layout: tintLayout,
    fragment: tintFragment,
    defaults: { tint: d.vec4f(1) },
});
const tintPlugin = MaterialPlugin(TintMaterial);

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

test("changing MeshMaterial type moves the instance to that type's mesh draw on the next frame", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, tintPlugin],
    });
    const world = app.world;
    const device = world.gpu.device;
    const readback = device.createBuffer({
        size: 40,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, MeshInstance);
        const standardType = 0;
        const tintType = materialTypeId(world, TintMaterial);
        const mesh = world.storage(MeshInstance).mesh.get(eid);
        async function counts() {
            device.pushErrorScope("validation");
            world.step();
            const encoder = device.createCommandEncoder();
            for (const [index, materialType] of [standardType, tintType].entries()) {
                const draw = Array.from(world.resource(Draws)).find(
                    (candidate) =>
                        candidate.materialType === materialType && candidate.mesh === mesh,
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
                await bounded("material-type selection validation", device.popErrorScope()),
            ).toBeNull();
            await bounded("material-type selection readback", readback.mapAsync(GPUMapMode.READ));
            const words = new Uint32Array(readback.getMappedRange());
            const result = [words[1], words[6]];
            readback.unmap();
            return result;
        }
        expect(await counts()).toEqual([1, 0]);
        const tint = world.resource(TintMaterial).add({ tint: d.vec4f(1, 0, 0, 1) });
        world.add(eid, MeshMaterial, tint);
        expect(await counts()).toEqual([0, 1]);
        const standard = world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [0, 1, 0, 1] }));
        world.storage(MeshMaterial).type.set(eid, standard.type);
        world.storage(MeshMaterial).material.set(eid, standard.material);
        expect(await counts()).toEqual([1, 0]);
    } finally {
        readback.destroy();
        app.dispose();
    }
});
