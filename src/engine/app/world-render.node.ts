import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import { RenderingPlugin } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    MeshInstanceInput,
    MeshRenderPlugin,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { probeBuffer } from "../runtime";
import { createApp } from "./index";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} exceeded 5000 ms`)), 5000);
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

test("MeshInstance and StandardRenderer warm and compact a component-bound dense instance", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [RenderingPlugin, MeshRenderPlugin, StandardRenderingPlugin],
    });
    const { world } = app;
    try {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, MeshInstance);
        world.gpu.device.pushErrorScope("validation");
        world.step(0);
        await bounded("MeshInstance submissions", world.gpu.device.queue.onSubmittedWorkDone());
        expect(
            await bounded("MeshInstance validation", world.gpu.device.popErrorScope()),
        ).toBeNull();
        const packed = await probeBuffer(world, world.gpu.buffers.get("eids")!, { size: 4 });
        expect(new Uint32Array(packed.bytes)[0]).toBe(eid);
        const active = await probeBuffer(
            world,
            world.gpu.buffers.get("meshInstances:active-rows")!,
            {
                size: 8,
            },
        );
        const [activeEid, row] = new Uint32Array(active.bytes);
        expect(activeEid).toBe(eid);
        const recordSize = d.sizeOf(MeshInstanceInput);
        const materialOffset = d.memoryLayoutOf(
            MeshInstanceInput,
            (value) => value.material,
        ).offset;
        const record = await probeBuffer(world, world.gpu.buffers.get("meshInstances")!, {
            offset: row * recordSize,
            size: recordSize,
        });
        const data = new DataView(record.bytes);
        expect(data.getUint32(materialOffset, true)).toBe(0);
    } finally {
        app.dispose();
    }
});
