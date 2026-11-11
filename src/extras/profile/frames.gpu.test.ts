import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { DiagnosticsStore } from "../../engine";
import { diagnosticEntries, ProfilePlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const apps = gpuApps(import.meta.path, [
    { defaults: false, plugins: [ProfilePlugin] },
    { defaults: false, plugins: [ProfilePlugin] },
    { defaults: false, plugins: [ProfilePlugin] },
    { defaults: false, plugins: [ProfilePlugin] },
]);

test("nonempty frames beyond timestamp capacity are unsampled, counted, and never relabeled or zero-filled", async () => {
    const world = apps()[1].world;
    if (!world.gpu.device.features.has("timestamp-query")) {
        world.step(1 / 120);
        for (let frame = 1; frame <= 6; frame++) {
            expect(world.gpu.span!(frame <= 4 ? "capacity" : "unsampled")).toBeUndefined();
            world.step(1 / 120);
        }
        // Unavailable timing is not readback saturation, even beyond the ring's capacity.
        expect(world.resource(DiagnosticsStore).get("render/gpu_unsampled_frames")?.value()).toBe(
            0,
        );
        return;
    }
    const pool = world.readback;
    const request = pool.request;
    const complete: ((result: { bytes: ArrayBuffer; frame: number; fixedTick: number }) => void)[] =
        [];
    pool.request = () =>
        new Promise((resolve) => {
            complete.push(resolve);
        });
    try {
        world.step(1 / 120);
        for (let frame = 1; frame <= 6; frame++) {
            world.gpu.span!(frame <= 4 ? "capacity" : "unsampled");
            world.step(1 / 120);
        }
        const store = world.resource(DiagnosticsStore);
        expect(complete).toHaveLength(4);
        expect(store.get("render/gpu_unsampled_frames")?.value()).toBe(2);
        for (const index of [3, 2, 1]) {
            complete[index]({
                bytes: new BigUint64Array([0n, BigInt(index + 1) * 1_000_000n]).buffer,
                frame: index + 1,
                fixedTick: 0,
            });
        }
        await Promise.resolve();
        expect(store.get("render/capacity/elapsed_gpu")).toBeUndefined();
        complete[0]({ bytes: new BigUint64Array([0n, 1_000_000n]).buffer, frame: 1, fixedTick: 0 });
        await Promise.resolve();
        world.step(1 / 120);
        const samples = [...store.get("render/capacity/elapsed_gpu")!.measurements()].filter(
            (sample) => sample.time > 0,
        );
        expect(samples.map((sample) => sample.value)).toEqual([1, 2, 3, 4, 0]);
        expect([...store.get("frame/elapsed_gpu")!.measurements()]).toEqual([
            ...store.get("render/capacity/elapsed_gpu")!.measurements(),
        ]);
        expect(store.get("frame/elapsed_gpu")!.smoothed()).toBe(
            store.get("render/capacity/elapsed_gpu")!.smoothed(),
        );
        expect(samples.map((sample) => Math.round(sample.time * 120))).toEqual([1, 2, 3, 4, 7]);
        expect(store.get("render/unsampled/elapsed_gpu")).toBeUndefined();
        expect(store.get("render/gpu_unsampled_frames")?.value()).toBe(2);
    } finally {
        pool.request = request;
    }
});

test("GPU frames sum repeated passes and idle frames cannot overtake a delayed timestamp result", async () => {
    const world = apps()[0].world;
    if (!world.gpu.device.features.has("timestamp-query")) {
        world.step(1 / 120);
        expect(world.gpu.span!("retired")).toBeUndefined();
        expect(world.gpu.span!("retired")).toBeUndefined();
        world.step(1 / 120);
        const store = world.resource(DiagnosticsStore);
        // Idle frames cannot turn unavailable repeated-pass timings into zero samples.
        for (let i = 0; i < 131; i++) {
            world.step(1 / 120);
            expect(store.get("render/retired/elapsed_gpu")).toBeUndefined();
        }
        return;
    }
    const pool = world.readback;
    const request = pool.request;
    let complete!: (result: { bytes: ArrayBuffer; frame: number; fixedTick: number }) => void;
    pool.request = () =>
        new Promise((resolve) => {
            complete = resolve;
        });
    try {
        world.step(1 / 120);
        expect(world.gpu.span!("retired")).toBeDefined();
        expect(world.gpu.span!("retired")).toBeDefined();
        world.step(1 / 120);
        // Empty frames queue behind the unresolved frame, not ahead of it.
        for (let i = 0; i < 10; i++) world.step(1 / 120);
        const store = world.resource(DiagnosticsStore);
        expect(store.get("render/retired/elapsed_gpu")).toBeUndefined();
        complete({
            bytes: new BigUint64Array([0n, 2_000_000n, 3_000_000n, 6_000_000n]).buffer,
            frame: 1,
            fixedTick: 0,
        });
        await Promise.resolve();
        world.step(1 / 120);
        const diagnostic = store.get("render/retired/elapsed_gpu")!;
        expect([...store.get("frame/elapsed_gpu")!.measurements()]).toEqual([
            ...diagnostic.measurements(),
        ]);
        expect(
            [...diagnostic.measurements()]
                .filter((sample) => sample.time > 0)
                .map((sample) => sample.value),
        ).toEqual([5, ...new Array(11).fill(0)]);
        const times = [...diagnostic.measurements()]
            .filter((sample) => sample.time > 0)
            .map((sample) => sample.time);
        expect(times[1] - times[0]).toBeCloseTo(1 / 120);
        expect(times[2] - times[1]).toBeCloseTo(1 / 120);
        for (let i = 0; i < 120; i++) world.step(1 / 120);
        expect(diagnostic.value()).toBe(0);
        expect(diagnostic.smoothed()!).toBeLessThan(0.005);
    } finally {
        pool.request = request;
    }
});

test("GPU frame total unions unordered overlapping intervals and sums disjoint intervals without becoming a pass row", async () => {
    const world = apps()[2].world;
    if (!world.gpu.device.features.has("timestamp-query")) {
        const store = world.resource(DiagnosticsStore);
        for (let frame = 0; frame < 6; frame++) {
            expect(world.gpu.span!("union")).toBeUndefined();
            world.step(1 / 120);
            expect(store.get("frame/elapsed_gpu")).toBeUndefined();
            // The overlay uses this availability flag instead of displaying a numeric GPU total.
            expect(store.get("render/timestamp_query")?.value()).toBe(0);
            expect([...diagnosticEntries(store, "render/", "/elapsed_gpu")]).toEqual([]);
        }
        expect(world.readback.allocated).toBe(0);
        return;
    }
    const pool = world.readback;
    const request = pool.request;
    let complete!: (result: { bytes: ArrayBuffer; frame: number; fixedTick: number }) => void;
    pool.request = () =>
        new Promise((resolve) => {
            complete = resolve;
        });
    try {
        world.step(1 / 120);
        for (const [intervals, union, passSum] of [
            [[4n, 8n, 1n, 6n, 2n, 3n, 10n, 12n], 9, 6],
            [[8n, 11n, 1n, 3n, 15n, 16n, 20n, 22n], 8, 6],
        ] as const) {
            for (let i = 0; i < 4; i++) world.gpu.span!("union");
            world.step(1 / 120);
            complete({
                bytes: new BigUint64Array(intervals.map((n) => n * 1_000_000n)).buffer,
                frame: 1,
                fixedTick: 0,
            });
            await Promise.resolve();
            world.step(1 / 120);
            const store = world.resource(DiagnosticsStore);
            // The current empty frame follows the injected sample in the same drain.
            const samples = [...store.get("frame/elapsed_gpu")!.measurements()];
            expect(samples.at(-2)!.value).toBe(union);
            expect(samples.at(-1)!.value).toBe(0);
            const pass = [...store.get("render/union/elapsed_gpu")!.measurements()];
            expect(pass.at(-2)!.value).toBe(passSum);
            expect(samples.map((s) => s.time)).toEqual(pass.map((s) => s.time));
            expect(
                [...diagnosticEntries(store, "render/", "/elapsed_gpu")].map(([name]) => name),
            ).toEqual(["union"]);
        }
    } finally {
        pool.request = request;
    }
});

test("GPU pass rows measure serial spans and overlapping progress in submission order", async () => {
    const world = apps()[3].world;
    const names = ["serial", "serial", "chain", "chain", "contained", "after-contained"];
    if (!world.gpu.device.features.has("timestamp-query")) {
        world.step(1 / 120);
        for (const name of names) expect(world.gpu.span!(name)).toBeUndefined();
        world.step(1 / 120);
        const store = world.resource(DiagnosticsStore);
        for (const name of names) expect(store.get(`render/${name}/elapsed_gpu`)).toBeUndefined();
        return;
    }
    const pool = world.readback;
    const request = pool.request;
    let complete!: (result: { bytes: ArrayBuffer; frame: number; fixedTick: number }) => void;
    pool.request = () =>
        new Promise((resolve) => {
            complete = resolve;
        });
    try {
        world.step(1 / 120);
        for (const name of names) expect(world.gpu.span!(name)).toBeDefined();
        world.step(1 / 120);
        complete({
            bytes: new BigUint64Array(
                [1n, 3n, 5n, 8n, 7n, 12n, 10n, 16n, 11n, 13n, 14n, 18n].map((n) => n * 1_000_000n),
            ).buffer,
            frame: 1,
            fixedTick: 0,
        });
        await Promise.resolve();
        world.step(1 / 120);
        const store = world.resource(DiagnosticsStore);
        // Serial spans sum to 2 + 3; the chain adds (12 - 8) + (16 - 12).
        // A contained pass adds zero and cannot lower the latest end for the next row.
        for (const [name, value] of [
            ["serial", 5],
            ["chain", 8],
            ["contained", 0],
            ["after-contained", 2],
        ] as const) {
            const samples = [...store.get(`render/${name}/elapsed_gpu`)!.measurements()];
            expect(samples.at(-2)!.value).toBe(value);
            expect(samples.at(-1)!.value).toBe(0);
        }
    } finally {
        pool.request = request;
    }
});
