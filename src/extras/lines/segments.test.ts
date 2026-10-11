import { expect, test } from "bun:test";
import { World } from "../../engine";
import { initializeSegmentState, lineStableIds, lineStableParts, push } from "./segments";

test("line stable identities survive segment staging growth", () => {
    const world = new World();
    try {
        initializeSegmentState(world);
        const count = 17_000;
        for (let part = 0; part < count; part++)
            push(world, 0, 0, 0, 1, 0, 0, 1, 0xffffffff, 123, part);
        const stableIds = lineStableIds(world);
        const stableParts = lineStableParts(world);
        expect(stableIds.length).toBeGreaterThan(count);
        expect(stableIds[0]).toBe(123);
        expect(stableIds[16_383]).toBe(123);
        expect(stableIds[count - 1]).toBe(123);
        expect(stableParts[0]).toBe(0);
        expect(stableParts[16_383]).toBe(16_383);
        expect(stableParts[count - 1]).toBe(count - 1);
    } finally {
        world.dispose();
    }
});
