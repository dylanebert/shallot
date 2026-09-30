import { expect, setDefaultTimeout, test } from "bun:test";
import { build } from "../app";
import type { State } from "../ecs";
import { rawDevice } from "./gpu";
import { probeBuffer } from "./probe";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

async function trackedPool(
    body: (
        state: State,
        source: GPUBuffer,
        counts: { created: number; live: number },
    ) => Promise<void>,
) {
    const owner = await build({ defaults: false, plugins: [] });
    const device = rawDevice(owner.state.gpu.device);
    const original = device.createBuffer.bind(device);
    const counts = { created: 0, live: 0 };
    device.createBuffer = (descriptor) => {
        const buffer = original(descriptor);
        if ((descriptor.usage & GPUBufferUsage.MAP_READ) !== 0) {
            counts.created++;
            counts.live++;
            const destroy = buffer.destroy.bind(buffer);
            let destroyed = false;
            buffer.destroy = () => {
                if (!destroyed) {
                    destroyed = true;
                    counts.live--;
                }
                destroy();
            };
        }
        return buffer;
    };
    const app = await build({ defaults: false, plugins: [], device });
    const source = app.state.gpu.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC });
    try {
        await body(app.state, source, counts);
    } finally {
        source.destroy();
        app.dispose();
        device.createBuffer = original;
        owner.dispose();
    }
}

// Keep the old device-only call at the baseline boundary so the native resource claims can run red
// against pre-stage main, rather than failing only because the new State argument is absent.
function probeOwner(state: State): State {
    return ("readback" in (state as object) ? state : state.gpu.device) as State;
}

test("successive one-shot ranges reuse one native staging allocation", async () => {
    await trackedPool(async (state, source, counts) => {
        await probeBuffer(probeOwner(state), source, { size: 4 });
        await probeBuffer(probeOwner(state), source, { offset: 4, size: 4 });
        expect(counts.created).toBe(1);
        expect(counts.live).toBe(1);
    });
});

test("unused staging remains pooled until its declared idle frame count then is destroyed", async () => {
    await trackedPool(async (state, source, counts) => {
        await probeBuffer(probeOwner(state), source, { size: 4 });
        expect(counts.live).toBe(1);
        for (let i = 0; i < 9; i++) state.step(0);
        expect(counts.live).toBe(1);
        state.step(0);
        expect(counts.live).toBe(0);
    });
});
