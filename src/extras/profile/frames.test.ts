import { expect, test } from "bun:test";
import { DiagnosticsStore } from "../../engine";
import { cpuTotal } from "./cpu";
import { FrameTimings } from "./frames";

test("60 Hz spans on a 120 Hz display include absent frames and repeated runs sum once", () => {
    const store = new DiagnosticsStore();
    const frames = new FrameTimings(store);
    const path = "render/fixed/elapsed_cpu";
    const clock = new Float64Array(1);
    for (let frame = 0; frame < 240; frame++) {
        if (frame % 2 === 0) {
            frames.add(path, 2);
            frames.add(path, 2);
        }
        clock[0] = frame / 120;
        frames.finish(clock);
    }
    const samples = [...store.get(path)!.measurements()];
    expect(samples.reduce((sum, sample) => sum + sample.value, 0) / samples.length).toBe(2);
    expect(store.get(path)!.smoothed()).toBeGreaterThan(1.8);
    expect(store.get(path)!.smoothed()).toBeLessThan(2.2);
});

test("scene changes decay CPU and GPU rows, and steady totals fit their frame without clamping", () => {
    for (const field of ["elapsed_cpu", "elapsed_gpu"]) {
        const store = new DiagnosticsStore();
        const frames = new FrameTimings(store);
        frames.add(`render/old/${field}`, 60);
        const clock = new Float64Array(1);
        frames.finish(clock);
        for (let frame = 1; frame < 240; frame++) {
            frames.add(`render/current/${field}`, 3);
            frames.add(`render/other/${field}`, 2);
            clock[0] = frame / 120;
            frames.finish(clock);
        }
        const rows = new Map([...store.iter()].map((row) => [row.path.path, row.smoothed()!]));
        expect(store.get(`render/old/${field}`)!.value()).toBe(0);
        expect(store.get(`render/old/${field}`)!.smoothed()!).toBeLessThan(0.005);
        expect(cpuTotal(rows)).toBeLessThan(1000 / 120);
        expect(cpuTotal(rows)).toBeCloseTo(5);
    }
});
