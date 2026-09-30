import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { compileGpuFile } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { build } from "../app";
import { rawDevice } from "./gpu";

setDefaultTimeout(CEILING.gpu);
const apps: Awaited<ReturnType<typeof build>>[] = [];
const subject = compileGpuFile(import.meta.path, async () => {
    const owner = await build({ defaults: false, plugins: [] });
    apps.push(owner);
    const native = rawDevice(owner.state.gpu.device);
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
    const app = await build({ defaults: false, plugins: [], device: supplied });
    apps.push(app);
    return { app, reads: () => reads };
});
afterAll(() => {
    for (const app of apps.reverse()) app.dispose();
});

test("world devices resolve immutable capabilities once rather than invoking native getters during play", () => {
    const { app, reads } = subject();
    const before = reads();
    const limits = app.state.gpu.device.limits;
    const features = app.state.gpu.device.features;
    for (let i = 0; i < 100; i++) {
        expect(app.state.gpu.device.limits).toBe(limits);
        expect(app.state.gpu.device.features).toBe(features);
    }
    expect(reads()).toBe(before);
});
