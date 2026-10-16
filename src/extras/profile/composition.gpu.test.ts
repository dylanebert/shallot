import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import type { Plugin } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import { Profile, ProfilePlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const allocate: Plugin = {
    name: "BeforeProfile",
    gpu: {},
    initialize(world) {
        const device = world.gpu.device;
        device.createBuffer({ size: 1024 * 1024, usage: GPUBufferUsage.STORAGE });
        device.createTexture({
            size: [16, 16],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        device.createComputePipeline({
            label: "before-profile",
            layout: "auto",
            compute: {
                module: device.createShaderModule({
                    code: "@compute @workgroup_size(1) fn main() {}",
                }),
                entryPoint: "main",
            },
        });
    },
};
const apps = gpuApps(import.meta.path, [
    { defaults: false, plugins: [allocate, ProfilePlugin] },
    { plugins: [...DEFAULT_PLUGINS, ProfilePlugin] },
    { defaults: false, plugins: [{ name: "NoProfile", gpu: {} }] },
]);

test("a profiler listed last counts earlier buffers", () => {
    expect(apps()[0].world.resource(Profile).bufferBytes).toBeGreaterThanOrEqual(1024 * 1024);
});
test("a profiler listed last counts earlier textures", () => {
    expect(apps()[0].world.resource(Profile).textureBytes).toBeGreaterThanOrEqual(16 * 16 * 4);
});
test("a profiler listed last records earlier pipeline labels", () => {
    expect(apps()[0].world.resource(Profile).compile.has("before-profile")).toBe(true);
});
test("default allocations form a live baseline after game buffer destruction", () => {
    const world = apps()[1].world;
    const stats = world.resource(Profile);
    const baseline = stats.bufferBytes;
    expect(baseline).toBeGreaterThan(0);
    const buffer = world.gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    expect(stats.bufferBytes).toBe(baseline + 1024);
    buffer.destroy();
    expect(stats.bufferBytes).toBe(baseline);
});
test("without ProfilePlugin createBuffer has no profiler override", () => {
    const device = apps()[2].world.gpu.device;
    const original = device.createBuffer;
    expect(original.name).toBe("ownedBuffer");
    expect(apps()[2].world.resource(Profile).bufferBytes).toBe(0);
});
