import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { PointLight, RenderPlugin } from "../../core/rendering";
import { Transform } from "../index";
import { probeBuffer } from "../runtime";
import { build } from "./index";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
await (await import(peerModule)).setupGlobals();

test("render light inputs upload as active dense table rows", async () => {
    const app = await build({ defaults: false, plugins: [RenderPlugin] });
    try {
        const { state } = app;
        const eid = state.create();
        state.add(eid, Transform);
        state.add(eid, PointLight);
        Transform.pos.set(eid, 2, 3, 4, 0);
        PointLight.color.set(eid, 0xffd9a0);
        PointLight.intensity.set(eid, 2.5);
        PointLight.range.set(eid, 7);
        PointLight.radius.set(eid, 0.25);

        state.gpu.device.pushErrorScope("validation");
        state.step(0);
        await state.gpu.device.queue.onSubmittedWorkDone();
        expect(await state.gpu.device.popErrorScope()).toBeNull();
        const active = await probeBuffer(state, state.gpu.buffers.get("lightInputs:active-rows")!, {
            size: 8,
        });
        const [activeEid, row] = new Uint32Array(active.bytes);
        expect(activeEid).toBe(eid);
        const record = await probeBuffer(state, state.gpu.buffers.get("lightInputs")!, {
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
