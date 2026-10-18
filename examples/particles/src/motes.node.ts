import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import {
    type createApp,
    DiagnosticsStore,
    Exposure,
    probeBuffer,
    setTransform,
} from "@dylanebert/shallot";
import { Fog } from "@dylanebert/shallot/extras";
import {
    AmbientLight,
    attachTexture,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { MOTE_COUNT, Motes, MotesPlugin } from "./motes";
import { Room } from "./room";

setDefaultTimeout(TIMEOUT.node);
let app: Awaited<ReturnType<typeof createApp>>;
let untimed: Awaited<ReturnType<typeof createApp>>;
let untimedDevice: GPUDevice;
beforeAll(async () => {
    app = await openGpuApp({ plugins: [MotesPlugin] });
    console.log("particles adapter", app.world.gpu.adapter);
    attachTexture(app.world, app.world.resource(Room).camera, { width: 64, height: 64 });
    app.world.step(1 / 60);
    app.world.step(1 / 60);
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("particles requires an adapter");
    untimedDevice = await adapter.requestDevice({
        requiredFeatures: ["indirect-first-instance"],
    });
    untimed = await openGpuApp({ plugins: [MotesPlugin], device: untimedDevice, adapter });
    attachTexture(untimed.world, untimed.world.resource(Room).camera, {
        width: 2048,
        height: 2048,
    });
});
afterAll(async () => {
    app?.dispose();
    untimed?.dispose();
    if (untimedDevice) {
        const error = spyOn(console, "error").mockImplementation(() => {});
        try {
            untimedDevice.destroy();
            await untimedDevice.lost;
            await Promise.resolve();
            expect(error).toHaveBeenCalledWith("GPU device lost destroyed: device was destroyed");
        } finally {
            error.mockRestore();
        }
    }
});
test("the example compiles its compute and draw pipelines and counts its storage and uniform bytes", () => {
    app.world.step(0);
    const profile = app.world.resource(DiagnosticsStore);
    expect(profile.get("memory/bufferBytes")?.value()).toBeGreaterThanOrEqual(MOTE_COUNT * 16 + 32);
    expect(!!profile.get("compile/particles:compute/elapsed")).toBe(true);
    expect(!!profile.get("compile/particles:draw/elapsed")).toBe(true);
});
test("motes near a moving player's fixed path end farther from the path than with the player still", async () => {
    const world = app.world;
    const motes = world.resource(Motes);
    const initial = await probeBuffer(world, motes.positions!);
    const snapshot = world.snapshot();
    const run = async (moving: boolean) => {
        world.restore(snapshot);
        world.gpu.device.queue.writeBuffer(motes.positions!, 0, initial.bytes);
        for (let i = 0; i < 90; i++) {
            setTransform(
                world,
                world.resource(Room).player,
                [moving ? -1 + (i * 2) / 89 : 3, 1.2, 0],
                [0, 0, 0, 1],
            );
            world.step(1 / 60);
        }
        return new Float32Array((await probeBuffer(world, motes.positions!)).bytes);
    };
    const still = await run(false);
    const moved = await run(true);
    const start = new Float32Array(initial.bytes);
    let count = 0,
        stillDistance = 0,
        movedDistance = 0;
    for (let i = 0; i < MOTE_COUNT; i++) {
        const j = i * 4;
        if (Math.abs(start[j]) > 1 || Math.hypot(start[j + 1] - 1.2, start[j + 2]) > 0.7) continue;
        count++;
        stillDistance += Math.hypot(
            Math.max(0, Math.abs(still[j]) - 1),
            still[j + 1] - 1.2,
            still[j + 2],
        );
        movedDistance += Math.hypot(
            Math.max(0, Math.abs(moved[j]) - 1),
            moved[j + 1] - 1.2,
            moved[j + 2],
        );
    }
    expect(count).toBeGreaterThan(50);
    console.log("path distance", {
        count,
        still: stillDistance / count,
        moved: movedDistance / count,
    });
    expect(movedDistance / count).toBeGreaterThan(stillDistance / count + 0.1);
});
test("the profiler lists the motes compute pass when the adapter grants timestamp-query", async () => {
    const { world } = app;
    const stats = world.resource(DiagnosticsStore);
    if (!world.gpu.device.features.has("timestamp-query")) {
        expect(stats.get("render/particles:advance/elapsed_gpu")).toBeUndefined();
        return;
    }
    while (!stats.get("render/particles:advance/elapsed_gpu")) {
        world.step(0);
        await world.gpu.sync();
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(stats.get("render/particles:advance/elapsed_gpu")!.historyLength).toBeGreaterThan(0);
    expect(Number.isFinite(stats.get("render/particles:advance/elapsed_gpu")!.value())).toBe(true);
});
test("without timestamp-query the example draws and reports requires timestamp-query rather than zero GPU timings", async () => {
    const { world } = untimed;
    world.step(1 / 60);
    await world.gpu.sync();
    world.step(0);
    const stats = world.resource(DiagnosticsStore);
    expect(world.gpu.device.features.has("timestamp-query")).toBe(false);
    expect(stats.get("render/particles:advance/elapsed_gpu")).toBeUndefined();
    expect(stats.get("compile/particles:compute/elapsed")).toBeDefined();
    expect(stats.get("compile/particles:draw/elapsed")).toBeDefined();
});
/** One mote alone in a scene each geometry test sets for itself, since hosts run tests one at a time. */
function moteImages(world: typeof untimed.world) {
    const camera = world.resource(Room).camera;
    if (!world.has(camera, AmbientLight)) world.add(camera, AmbientLight);
    world.storage(AmbientLight).brightness.set(camera, 10000);
    world.storage(Exposure).ev100.set(camera, 8);
    if (!world.has(camera, Tonemapping)) world.add(camera, Tonemapping);
    world.storage(Tonemapping).method.set(camera, TonemappingMethod.None);
    // Past viewDistance the mote's light is analytic and the grid holds 5 cm of 1e-5 density,
    // uniform, so jitter changes nothing.
    const fog = world.only([Fog]);
    world.storage(Fog).density.set(fog, 0.00001);
    world.storage(Fog).viewDistance.set(fog, 0.15);
    const motes = world.resource(Motes);
    const positions = new Float32Array(MOTE_COUNT * 4);
    // All other instances stay outside the room and the view.
    for (let i = 0; i < MOTE_COUNT; i++) positions[i * 4] = 100;
    // A changed Fog drops the grid's history, and the next frame blends it back in at rgba16float
    // precision, so the first frame can differ from the second by a byte. Compare settled frames.
    motes.visible = false;
    world.step(0);
    return async (depth: number, visible: boolean) => {
        positions.set([0, 1.7, 2 - depth, 0], 0);
        world.gpu.device.queue.writeBuffer(motes.positions!, 0, positions);
        motes.visible = visible;
        world.step(0);
        return (await captureTexture(world, camera)).rgba;
    };
}
test("a mote just in front of the camera adds no light while a mote two metres away still does", async () => {
    const image = moteImages(untimed.world);
    const background = await image(0.2, false);
    const near = await image(0.2, true);
    const far = await image(2, true);
    expect(near).toEqual(background);
    let increase = 0;
    for (let i = 0; i < far.length; i += 4) increase += Math.max(0, far[i]! - background[i]!);
    expect(increase).toBeGreaterThan(0);
});
test("a one-metre mote covers no more pixels than at the two-metre size cap and a four-metre mote keeps its original footprint", async () => {
    const image = moteImages(untimed.world);
    const background = await image(1, false);
    const coverage = async (depth: number) => {
        const pixels = await image(depth, true);
        let count = 0;
        for (let i = 0; i < pixels.length; i += 4) {
            if (pixels[i]! > background[i]!) count++;
        }
        return count;
    };
    const near = await coverage(1);
    const cap = await coverage(2);
    const far = await coverage(4);
    console.log("mote pixel coverage", { near, cap, far });
    expect(cap).toBeGreaterThan(0);
    expect(near).toBeLessThanOrEqual(cap);
    // At four metres the original 0.006 clip-space half-width is 1.536 pixels.
    expect(far).toBe(4);
});
test("the profiler buffer total includes exactly the motes storage and frame uniform allocations", () => {
    const { world } = app;
    const stats = world.resource(DiagnosticsStore);
    const before = stats.get("memory/bufferBytes")!.value()!;
    const motes = world.resource(Motes);
    motes.positions!.destroy();
    motes.frame!.destroy();
    expect(before - stats.get("memory/bufferBytes")!.value()!).toBe(MOTE_COUNT * 16 + 32);
});
