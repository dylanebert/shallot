import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { compileGpuFile } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../app";
import type { World } from "../ecs";
import { rawDevice } from "./gpu";
import { probeBuffer } from "./probe";
import { countStaging } from "./readback.fixture";

setDefaultTimeout(CEILING.gpu);
const apps: Awaited<ReturnType<typeof createApp>>[] = [];
const subject = compileGpuFile(import.meta.path, async () => {
    const owner = await createApp({ defaults: false, plugins: [] });
    apps.push(owner);
    const device = rawDevice(owner.world.gpu.device);
    const tracker = countStaging(device);
    try {
        for (let i = 0; i < 2; i++)
            apps.push(await createApp({ defaults: false, plugins: [], device }));
        return { worlds: apps.slice(1), counts: tracker.counts };
    } finally {
        tracker.restore();
    }
});
afterAll(() => {
    for (const app of apps.reverse()) app.dispose();
});
let nextSubject = 0;

async function trackedPool(
    body: (
        world: World,
        source: GPUBuffer,
        counts: { created: number; live: number },
    ) => Promise<void>,
) {
    const { worlds, counts } = subject();
    const app = worlds[nextSubject++];
    const source = app.world.gpu.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC });
    try {
        await body(app.world, source, counts);
    } finally {
        source.destroy();
        app.dispose();
    }
}

// Keep the old device-only call at the baseline boundary so the native resource claims can run red
// against pre-stage main, rather than failing only because the new World argument is absent.
function probeOwner(world: World): World {
    return ("readback" in (world as object) ? world : world.gpu.device) as World;
}

test("successive one-shot ranges reuse one native staging allocation", async () => {
    await trackedPool(async (world, source, counts) => {
        await probeBuffer(probeOwner(world), source, { size: 4 });
        await probeBuffer(probeOwner(world), source, { offset: 4, size: 4 });
        expect(counts.created).toBe(1);
        expect(counts.live).toBe(1);
    });
});

test("unused staging remains pooled until its declared idle frame count then is destroyed", async () => {
    await trackedPool(async (world, source, counts) => {
        await probeBuffer(probeOwner(world), source, { size: 4 });
        expect(counts.live).toBe(1);
        for (let i = 0; i < 9; i++) world.step(0);
        expect(counts.live).toBe(1);
        world.step(0);
        expect(counts.live).toBe(0);
    });
});
