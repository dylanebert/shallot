import { expect, test } from "bun:test";
import { Diagnostic, DiagnosticPath, DiagnosticsStore } from "./index";

test("three frame samples retain history and use time-weighted smoothing", () => {
    const store = new DiagnosticsStore();
    const path = new DiagnosticPath("render/example/elapsed_cpu");
    store.add(new Diagnostic(path, { smoothingFactor: 0.1, maxHistoryLength: 3, suffix: "ms" }));
    store.record(path, 2, 0);
    store.record(path, 6, 0.05);
    store.record(path, 8, 0.1);
    const diagnostic = store.get("render/example/elapsed_cpu")!;
    expect([...diagnostic.measurements()]).toEqual([
        { time: 0, value: 2 },
        { time: 0.05, value: 6 },
        { time: 0.1, value: 8 },
    ]);
    expect(diagnostic.value()).toBe(8);
    expect(diagnostic.average()).toBeCloseTo(16 / 3);
    expect(diagnostic.smoothed()).toBe(6);
    // No observation does not fabricate zeros or evict an intermittent span.
    store.record("another", 1, 5);
    expect(diagnostic.smoothed()).toBe(6);
    store.record(path, 10, 5);
    expect([...diagnostic.measurements()].map((sample) => sample.value)).toEqual([6, 8, 10]);
    expect(diagnostic.average()).toBe(8);
    expect(diagnostic.smoothed()).toBe(10);
    diagnostic.clearHistory();
    expect(diagnostic.value()).toBeUndefined();
    expect(diagnostic.smoothed()).toBeUndefined();
});

test("packed telemetry copies borrowed measurements into diagnostic history", () => {
    const diagnostic = new Diagnostic("render/reused/elapsed_cpu", { smoothingFactor: 0.1 });
    const measurement = new Float64Array(2);
    for (const [time, value] of [
        [0, 2],
        [0.05, 6],
        [0.1, 8],
    ]) {
        measurement[0] = time;
        measurement[1] = value;
        diagnostic.addMeasurementBuffer(measurement);
    }
    measurement.fill(99);
    expect([...diagnostic.measurements()].map((sample) => sample.value)).toEqual([2, 6, 8]);
    expect(diagnostic.smoothed()).toBe(6);
});

test("disabled diagnostics preserve their history; paths validate components and compare by text", () => {
    const store = new DiagnosticsStore();
    store.record(new DiagnosticPath("test/value"), 3, 0);
    const diagnostic = store.get(new DiagnosticPath("test/value"))!;
    diagnostic.isEnabled = false;
    store.record("test/value", 7, 1);
    expect(diagnostic.value()).toBe(3);
    for (const path of ["", "/test", "test/", "test//value"])
        expect(() => new DiagnosticPath(path)).toThrow();
    expect(() => new Diagnostic("test", { maxHistoryLength: 0 })).toThrow();
});
