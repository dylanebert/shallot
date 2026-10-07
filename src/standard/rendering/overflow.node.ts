import { expect, setDefaultTimeout, test } from "bun:test";
import createRenderedSubject from "../../../diagnostics/readback-allocation/render.entry";
import { CEILING } from "../../../scripts/test-tiers";
import { PointLight } from "../../core/rendering";
import { Transform } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { CLUSTER_COUNT, LIGHT_GRID_OFFSET, LIGHT_POOL, requestLightOverflow } from "./cluster";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("light culling clamps an overflowing index pool without readback and exposes drops only on request", async () => {
    const subject = await createRenderedSubject();
    const world = subject.world;
    try {
        for (let i = 0; i < 35; i++) {
            const eid = world.create();
            world.add(eid, Transform);
            world.add(eid, PointLight);
            world.storage(PointLight).range.set(eid, 100000);
        }
        subject.step();
        await subject.wait();
        const frame = world.gpu.frame;
        const tick = world.time.fixedTick;
        const result = await requestLightOverflow(world);
        expect(result.frame).toBe(frame);
        expect(result.fixedTick).toBe(tick);
        expect(result.dropped).toBeGreaterThan(0);
        const grid = await probeBuffer(world, world.gpu.buffers.get("lightClusters")!, {
            offset: LIGHT_GRID_OFFSET,
            size: CLUSTER_COUNT * 8,
            label: "clamped light grid",
        });
        const words = new Uint32Array(grid.bytes);
        const poolWords = LIGHT_POOL + 2;
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
