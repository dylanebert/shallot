import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { Transform } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { PointLight, RenderPlugin } from "./index";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [RenderPlugin] }]);

test("render light inputs upload as active dense table rows", async () => {
    const app = subjects()[0];
    try {
        const { world } = app;
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, PointLight);
        world.storage(Transform).translation.set(eid, 2, 3, 4, 0);
        world.storage(PointLight).color.set(eid, 0xffd9a0);
        world.storage(PointLight).intensity.set(eid, 2.5);
        world.storage(PointLight).range.set(eid, 7);
        world.storage(PointLight).radius.set(eid, 0.25);

        world.gpu.device.pushErrorScope("validation");
        world.step(0);
        await world.gpu.device.queue.onSubmittedWorkDone();
        expect(await world.gpu.device.popErrorScope()).toBeNull();
        const active = await probeBuffer(world, world.gpu.buffers.get("lightInputs:active-rows")!, {
            size: 8,
        });
        const [activeEid, row] = new Uint32Array(active.bytes);
        expect(activeEid).toBe(eid);
        const record = await probeBuffer(world, world.gpu.buffers.get("lightInputs")!, {
            offset: row * 32,
            size: 32,
        });
        const data = new DataView(record.bytes);
        expect(data.getFloat32(0, true)).toBe(Math.fround(0xffd9a0));
        expect(data.getFloat32(4, true)).toBe(2.5);
        expect(data.getFloat32(8, true)).toBe(7);
        expect(data.getFloat32(12, true)).toBe(0.25);
    } finally {
        app.dispose();
    }
});
