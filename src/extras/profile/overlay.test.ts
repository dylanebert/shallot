import { expect, test } from "bun:test";
import { DiagnosticsStore } from "../../engine";
import { collectStats } from "./index";

test("gap is frame time minus CPU total regardless of overlapping GPU latency", () => {
    const store = new DiagnosticsStore();
    store.record("frame_time", 16, 1, "ms");
    store.record("render/draw/elapsed_cpu", 5, 1, "ms");
    store.record("render/draw%2Fpart/elapsed_cpu", 2, 1, "ms");
    store.record("render/update/elapsed_cpu", 3, 1, "ms");
    for (const latency of [0, 4, 20]) {
        store.remove("frame/fence_wait");
        store.record("frame/fence_wait", latency, 1, "ms");
        expect(collectStats(store).fenceWaitMs).toBe(latency);
        expect(collectStats(store).gapMs).toBe(8);
    }
});
