import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { DiagnosticsStore } from "../../engine";
import { ProfilePlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const apps = gpuApps(import.meta.path, [
    { defaults: false, plugins: [ProfilePlugin] },
    { defaults: false, plugins: [ProfilePlugin] },
]);

test("nonempty frames beyond timestamp capacity are unsampled, counted, and never relabeled or zero-filled", async () => {
    const world = apps()[1].world;
    if (!world.gpu.device.features.has("timestamp-query"))
        throw new Error("requires timestamp-query");
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
        expect(samples.map((sample) => Math.round(sample.time * 120))).toEqual([1, 2, 3, 4, 7]);
        expect(store.get("render/unsampled/elapsed_gpu")).toBeUndefined();
        expect(store.get("render/gpu_unsampled_frames")?.value()).toBe(2);
    } finally {
        pool.request = request;
    }
});

test("GPU frames sum repeated passes and idle frames cannot overtake a delayed timestamp result", async () => {
    const world = apps()[0].world;
    if (!world.gpu.device.features.has("timestamp-query"))
        throw new Error("requires timestamp-query");
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
