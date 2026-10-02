import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { CEILING } from "../../../scripts/test-tiers";
import { Mesh3d } from "../../core/mesh";
import { RenderingPlugin } from "../../core/rendering";
import { Mesh3dInput, PartPlugin, StandardRenderingPlugin } from "../../standard/rendering";
import { Transform } from "../index";
import { probeBuffer } from "../runtime";
import { createApp } from "./index";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
await (await import(peerModule)).setupGlobals();

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

test("Mesh3d and StandardRenderer warm and compact a component-bound dense instance", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [RenderingPlugin, PartPlugin, StandardRenderingPlugin],
    });
    const { world } = app;
    try {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Mesh3d);
        world.gpu.device.pushErrorScope("validation");
        world.step(0);
        await bounded("Mesh3d submissions", world.gpu.device.queue.onSubmittedWorkDone());
        expect(await bounded("Mesh3d validation", world.gpu.device.popErrorScope())).toBeNull();
        const packed = await probeBuffer(world, world.gpu.buffers.get("eids")!, { size: 4 });
        expect(new Uint32Array(packed.bytes)[0]).toBe(eid);
        const active = await probeBuffer(world, world.gpu.buffers.get("partInputs:active-rows")!, {
            size: 8,
        });
        const [activeEid, row] = new Uint32Array(active.bytes);
        expect(activeEid).toBe(eid);
        const recordSize = d.sizeOf(Mesh3dInput);
        const materialOffset = d.memoryLayoutOf(Mesh3dInput, (value) => value.material).offset;
        const record = await probeBuffer(world, world.gpu.buffers.get("partInputs")!, {
            offset: row * recordSize,
            size: recordSize,
        });
        const data = new DataView(record.bytes);
        expect(data.getUint32(materialOffset, true)).toBe(0);
    } finally {
        app.dispose();
    }
});
