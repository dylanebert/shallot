import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { createTestGpuApp as createApp } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { RenderingPlugin } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    MeshInstanceInput,
    MeshRenderPlugin,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { MeshDrawBuffers, meshInstanceTable } from "../../standard/rendering/preprocess";
import { probeBuffer } from "../runtime";

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

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
        await world.gpu.device.queue.onSubmittedWorkDone();
        expect(await world.gpu.device.popErrorScope()).toBeNull();
        const packed = await probeBuffer(
            world,
            world.gpu.root.unwrap(world.resource(MeshDrawBuffers).packedEids!),
            { size: 4 },
        );
        expect(new Uint32Array(packed.bytes)[0]).toBe(eid);
        const active = await probeBuffer(world, meshInstanceTable(world).activeRowsBuffer!, {
            size: 8,
        });
        const [activeEid, row] = new Uint32Array(active.bytes);
        expect(activeEid).toBe(eid);
        const recordSize = d.sizeOf(MeshInstanceInput);
        const materialOffset = d.memoryLayoutOf(
            MeshInstanceInput,
            (value) => value.material,
        ).offset;
        const record = await probeBuffer(world, meshInstanceTable(world).buffer, {
            offset: row * recordSize,
            size: recordSize,
        });
        const data = new DataView(record.bytes);
        expect(data.getUint32(materialOffset, true)).toBe(0);
    } finally {
        app.dispose();
    }
});
