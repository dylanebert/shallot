import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compileGpuFile, createTestGpuApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { expectError } from "../../../scripts/test-console";
import type { createApp } from "../app";
import { rawDevice } from "./gpu";

setDefaultTimeout(TIMEOUT.gpu);
const apps: Awaited<ReturnType<typeof createApp>>[] = [];
const subject = compileGpuFile(import.meta.path, async () => {
    const owner = await createTestGpuApp({
        defaults: false,
        plugins: [{ name: "CapabilitiesOwner", gpu: {} }],
    });
    apps.push(owner);
    const native = rawDevice(owner.world.gpu.device);
    const methods = new Map<PropertyKey, unknown>();
    let reads = 0;
    const supplied = new Proxy(native, {
        get(target, key) {
            if (key === "limits" || key === "features") reads++;
            const value = Reflect.get(target, key, target);
            if (typeof value !== "function") return value;
            if (!methods.has(key)) methods.set(key, value.bind(target));
            return methods.get(key);
        },
    });
    const app = await createTestGpuApp({ defaults: false, plugins: [], device: supplied });
    apps.push(app);
    return { app, reads: () => reads };
});
afterAll(() => {
    // The supplied proxy has its own device-loss observer.
    expectError(/^GPU device lost destroyed: device was destroyed$/);
    return disposeGpuApps(apps);
});

test("world devices resolve immutable capabilities once rather than invoking native getters during play", () => {
    const { app, reads } = subject();
    const before = reads();
    const limits = app.world.gpu.device.limits;
    const features = app.world.gpu.device.features;
    for (let i = 0; i < 100; i++) {
        expect(app.world.gpu.device.limits).toBe(limits);
        expect(app.world.gpu.device.features).toBe(features);
    }
    expect(reads()).toBe(before);
});
