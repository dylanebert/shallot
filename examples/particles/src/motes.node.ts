import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { createApp, probeBuffer, setTransform } from "@dylanebert/shallot";
import { Profile } from "@dylanebert/shallot/extras";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "../tests/gpu-app";
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
    attachTexture(untimed.world, untimed.world.resource(Room).camera, { width: 16, height: 16 });
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
    const profile = app.world.resource(Profile);
    expect(profile.bufferBytes).toBeGreaterThanOrEqual(MOTE_COUNT * 16 + 32);
    expect(profile.compile.has("particles:compute")).toBe(true);
    expect(profile.compile.has("particles:draw")).toBe(true);
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
    const stats = world.resource(Profile);
    if (!world.gpu.device.features.has("timestamp-query")) {
        expect(stats.gpuTiming).toBe("requires timestamp-query");
        return;
    }
    expect(stats.gpuTiming).toBe("available");
    while (!stats.gpuTime.has("particles:advance")) {
        world.step(0);
        await world.gpu.sync();
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(stats.gpuFires.get("particles:advance")).toBeGreaterThan(0);
    expect(Number.isFinite(stats.gpuTime.get("particles:advance"))).toBe(true);
});
test("without timestamp-query the example draws and reports requires timestamp-query rather than zero GPU timings", async () => {
    const { world } = untimed;
    world.step(1 / 60);
    await world.gpu.sync();
    world.step(0);
    const stats = world.resource(Profile);
    expect(world.gpu.device.features.has("timestamp-query")).toBe(false);
    expect(stats.gpuTiming).toBe("requires timestamp-query");
    expect(stats.gpu.has("particles:advance")).toBe(false);
    expect(stats.compile.has("particles:compute")).toBe(true);
    expect(stats.compile.has("particles:draw")).toBe(true);
});
test("the profiler buffer total includes exactly the motes storage and frame uniform allocations", () => {
    const { world } = app;
    const stats = world.resource(Profile);
    const before = stats.bufferBytes;
    const motes = world.resource(Motes);
    motes.positions!.destroy();
    motes.frame!.destroy();
    expect(before - stats.bufferBytes).toBe(MOTE_COUNT * 16 + 32);
});
