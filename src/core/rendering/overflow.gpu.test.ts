import { expect, setDefaultTimeout, test } from "bun:test";
import createRenderedSubject from "../../../diagnostics/readback-allocation/render.entry";
import { probeBuffer } from "../../engine/runtime";
import { Transform } from "../../transitional/transforms";
import { CLUSTER_COUNT, LIGHT_POOL, requestLightOverflow } from "./cluster";
import { PointLight } from "./lighting";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("light culling clamps an overflowing index pool without readback and exposes drops only on request", async () => {
    const subject = await createRenderedSubject();
    const state = subject.state;
    try {
        for (let i = 0; i < 35; i++) {
            const eid = state.create();
            state.add(eid, Transform);
            state.add(eid, PointLight);
            state.of(PointLight).range.set(eid, 100000);
        }
        subject.step();
        await subject.wait();
        const frame = state.gpu.frame;
        const tick = state.time.fixedTick;
        const result = await requestLightOverflow(state);
        expect(result.frame).toBe(frame);
        expect(result.fixedTick).toBe(tick);
        expect(result.dropped).toBeGreaterThan(0);
        const grid = await probeBuffer(state, state.gpu.buffers.get("lightGrid")!, {
            size: CLUSTER_COUNT * 8,
            label: "clamped light grid",
        });
        const words = new Uint32Array(grid.bytes);
        const poolWords = state.gpu.buffers.get("lightIndices")!.size / 4;
        let total = 0;
        for (let i = 0; i < CLUSTER_COUNT; i++) {
            const start = words[i * 2],
                count = words[i * 2 + 1];
            if (count > 0) expect(start + count).toBeLessThanOrEqual(poolWords);
            total += count;
        }
        expect(total).toBe(LIGHT_POOL);
    } finally {
        subject.dispose();
    }
});
