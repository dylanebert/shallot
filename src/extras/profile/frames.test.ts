import { expect, test } from "bun:test";
import { Diagnostic, DiagnosticsStore } from "../../engine";
import { cpuTotal } from "./cpu";
import { EmptyFrames } from "./empty";
import { FrameTimings } from "./frames";

test("a new label seeds only presented frames, not a padded history window", () => {
    const store = new DiagnosticsStore();
    const frames = new FrameTimings(store);
    const clock = new Float64Array(1);
    frames.finish(clock);
    frames.add("render/new/elapsed_cpu", 6);
    clock[0] = 1 / 120;
    frames.finish(clock);
    const diagnostic = store.get("render/new/elapsed_cpu")!;
    expect([...diagnostic.measurements()]).toEqual([
        { time: 0, value: 0 },
        { time: 1 / 120, value: 6 },
    ]);
    expect(diagnostic.historyLength).toBe(2);
    expect(diagnostic.average()).toBe(3);
});

test.each([120, 121, 240])(
    "work every %i frames returns with zero-filled smoothing and history",
    (period) => {
        const store = new DiagnosticsStore();
        const frames = new FrameTimings(store);
        const reference = new Diagnostic("reference");
        const clock = new Float64Array(2);
        const path = "render/periodic/elapsed_cpu";
        for (let frame = 0; frame < 240 + period * 4; frame++) {
            clock[0] = frame / 120;
            clock[1] = frame >= 240 && (frame - 240) % period === 0 ? 6 : 0;
            if (clock[1]) frames.add(path, clock[1]);
            frames.finish(clock);
            reference.addMeasurementBuffer(clock);
            if (clock[1]) {
                const diagnostic = store.get(path)!;
                expect(diagnostic.smoothed()!).toBeGreaterThan(0.52);
                expect(diagnostic.smoothed()!).toBeLessThan(0.53);
                expect(Math.abs(diagnostic.smoothed()! - reference.smoothed()!)).toBeLessThan(
                    0.005,
                );
                expect(diagnostic.average()).toBeCloseTo(reference.average()!, 10);
            }
        }
    },
);

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

test("one thousand one-shot labels retire after their history and decay, and can return", () => {
    const store = new DiagnosticsStore();
    const frames = new FrameTimings(store);
    const clock = new Float64Array(1);
    for (let i = 0; i < 1000; i++) frames.add(`render/once-${i}/elapsed_cpu`, 6);
    frames.finish(clock);
    for (let frame = 1; frame <= 240; frame++) {
        clock[0] = frame / 120;
        frames.finish(clock);
        if (frame === 119) expect([...store.iter()]).toHaveLength(1000);
    }
    expect([...store.iter()]).toHaveLength(0);
    frames.add("render/once-0/elapsed_cpu", 2);
    frames.finish(clock);
    expect(store.get("render/once-0/elapsed_cpu")!.historyLength).toBe(120);
    expect(store.get("render/once-0/elapsed_cpu")!.average()).toBeCloseTo(2 / 120);
    expect(store.get("render/once-0/elapsed_cpu")!.value()).toBe(2);
});

test("bounded empty-frame history folds old zeros without changing irregular-frame smoothing", () => {
    const store = new DiagnosticsStore();
    const reference = new DiagnosticsStore();
    const frames = new FrameTimings(store);
    const direct = new FrameTimings(reference);
    const empty = new EmptyFrames();
    const clock = new Float64Array(1);
    const path = "render/long-lived/elapsed_gpu";
    frames.add(path, 1e30);
    direct.add(path, 1e30);
    frames.finish(clock);
    direct.finish(clock);
    empty.reset(0);
    for (let i = 0; i < 500; i++) {
        clock[0] += i % 2 === 0 ? 1 / 120 : 1 / 144;
        empty.push(clock);
        direct.finish(clock);
    }
    empty.drain(frames);
    expect([...store.get(path)!.measurements()]).toEqual([...reference.get(path)!.measurements()]);
    expect(store.get(path)!.smoothed()! / reference.get(path)!.smoothed()!).toBeCloseTo(1, 10);
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
        expect(store.get(`render/old/${field}`)).toBeUndefined();
        expect(cpuTotal(rows)).toBeLessThan(1000 / 120);
        expect(cpuTotal(rows)).toBeCloseTo(5);
    }
});
