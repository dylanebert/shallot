import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Diagnostic, DiagnosticsStore, type Plugin, precompile } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import { DiagnosticsOverlayPlugin, ProfilePlugin } from "./index";

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
        const pipeline = device.createComputePipeline({
            label: "before-profile",
            layout: "auto",
            compute: {
                module: device.createShaderModule({
                    code: "@compute @workgroup_size(1) fn main() {}",
                }),
                entryPoint: "main",
            },
        });
        precompile(world, "before-profile", () => [pipeline]);
    },
};
const labels = ["example//compute", "", "%2F", "\ud800"];
function labeledPipelines(labels: readonly string[]): Plugin {
    return {
        name: "DiagnosticLabels",
        gpu: {},
        initialize(world) {
            const device = world.gpu.device;
            const module = device.createShaderModule({
                code: "@compute @workgroup_size(1) fn main() {}",
            });
            for (const label of labels) {
                const pipeline = device.createComputePipeline({
                    label,
                    layout: "auto",
                    compute: { module, entryPoint: "main" },
                });
                precompile(world, label, () => [pipeline]);
            }
        },
    };
}
const apps = gpuApps(import.meta.path, [
    { defaults: false, plugins: [allocate, labeledPipelines(labels), ProfilePlugin] },
    { plugins: [...DEFAULT_PLUGINS, ProfilePlugin] },
    { defaults: false, plugins: [{ name: "NoProfile", gpu: {} }] },
    {
        defaults: false,
        plugins: [allocate, labeledPipelines(["", "(unlabeled)"]), DiagnosticsOverlayPlugin],
    },
]);

test("a CPU label absent from a presented frame contributes zero instead of its last run", () => {
    const world = apps()[0].world;
    world.step(1 / 120);
    world.recordSink!("once", new Float64Array([0, 6]));
    world.step(1 / 120);
    const diagnostic = world.resource(DiagnosticsStore).get("render/once/elapsed_cpu")!;
    expect(diagnostic.value()).toBe(6);
    world.step(1 / 120);
    expect(diagnostic.value()).toBe(0);
});

test("three frames record the same named CPU diagnostic with and without the optional HUD", () => {
    const worlds = [apps()[0].world, apps()[3].world];
    for (const world of worlds) {
        world.resource(DiagnosticsStore).add(
            new Diagnostic("render/test-span/elapsed_cpu", {
                smoothingFactor: 0.1,
                suffix: "ms",
            }),
        );
        for (const [time, ms] of [
            [0, 2],
            [50, 6],
            [100, 8],
        ]) {
            world.recordSink!("test-span", new Float64Array([time / 1000, ms]));
            world.step(0.05);
        }
    }
    const stores = worlds.map((world) => world.resource(DiagnosticsStore));
    expect(stores[0].get("memory/bufferBytes")!.value()).toBe(
        stores[1].get("memory/bufferBytes")!.value(),
    );
    expect(stores[0].get("memory/textureBytes")!.value()).toBe(
        stores[1].get("memory/textureBytes")!.value(),
    );
    expect(stores[0].get("compile/before-profile/elapsed")).toBeDefined();
    expect(stores[1].get("compile/before-profile/elapsed")).toBeDefined();
    const diagnostics = worlds.map(
        (world) => world.resource(DiagnosticsStore).get("render/test-span/elapsed_cpu")!,
    );
    for (const diagnostic of diagnostics) {
        expect([...diagnostic.measurements()].map((sample) => sample.value)).toEqual([2, 6, 8]);
        expect(diagnostic.smoothed()).toBeCloseTo(6);
    }
});

test("arbitrary WebGPU labels and empty labels remain valid diagnostic components", () => {
    const world = apps()[0].world;
    const store = world.resource(DiagnosticsStore);
    for (const label of labels) {
        world.recordSink!(label, new Float64Array([0, 2]));
        world.gpu.indirect!(label, 1);
        const buffer = world.gpu.device.createBuffer({
            label,
            size: 16,
            usage: GPUBufferUsage.STORAGE,
        });
        buffer.destroy();
    }
    world.step(1 / 120);
    expect(store.get("compile/example%2F%2Fcompute/elapsed")).toBeDefined();
    expect(store.get("compile/%/elapsed")).toBeDefined();
    expect(store.get("render/example%2F%2Fcompute/elapsed_cpu")?.value()).toBe(2);
    expect(store.get("render/%/elapsed_cpu")?.value()).toBe(2);
    expect(store.get("render/example%2F%2Fcompute/indirect")?.value()).toBe(1);
    expect(store.get("render/%/indirect")?.value()).toBe(1);
    expect(store.get("render/%252F/elapsed_cpu")?.value()).toBe(2);
    expect(store.get("render/\ud800/elapsed_cpu")?.value()).toBe(2);
    expect(store.get("compile/%252F/elapsed")).toBeDefined();
});

test("empty and literal unlabeled pipeline labels keep separate diagnostic histories", () => {
    const world = apps()[3].world;
    const store = world.resource(DiagnosticsStore);
    expect(store.get("compile/%/elapsed")?.historyLength).toBe(1);
    expect(store.get("compile/(unlabeled)/elapsed")?.historyLength).toBe(1);
});

test("diagnostics report each frame's own fixed-step count and throttle state", () => {
    const world = apps()[0].world;
    const store = world.resource(DiagnosticsStore);
    for (const [delta, steps, throttled] of [
        [1 / 60, 1, 0],
        [0, 0, 0],
        [1, 4, 1],
    ]) {
        world.step(delta);
        expect(world.time.fixedSteps).toBe(steps);
        expect(Number(world.time.throttled)).toBe(throttled);
        expect(store.get("fixed_steps")?.value()).toBe(steps);
        expect(store.get("throttled")?.value()).toBe(throttled);
    }
});

test("a profiler listed last counts earlier buffers", () => {
    expect(
        apps()[0].world.resource(DiagnosticsStore).get("memory/bufferBytes")?.value(),
    ).toBeGreaterThanOrEqual(1024 * 1024);
});
test("a profiler listed last counts earlier textures", () => {
    expect(
        apps()[0].world.resource(DiagnosticsStore).get("memory/textureBytes")?.value(),
    ).toBeGreaterThanOrEqual(16 * 16 * 4);
});
test("a profiler listed last records earlier pipeline labels", () => {
    expect(!!apps()[0].world.resource(DiagnosticsStore).get("compile/before-profile/elapsed")).toBe(
        true,
    );
});
test("default allocations form a live baseline after game buffer destruction", () => {
    const world = apps()[1].world;
    const stats = world.resource(DiagnosticsStore);
    const baseline = stats.get("memory/bufferBytes")!.value()!;
    expect(baseline).toBeGreaterThan(0);
    const buffer = world.gpu.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    expect(stats.get("memory/bufferBytes")?.value()).toBe(baseline + 1024);
    buffer.destroy();
    expect(stats.get("memory/bufferBytes")?.value()).toBe(baseline);
});
test("without ProfilePlugin createBuffer has no profiler override", () => {
    const device = apps()[2].world.gpu.device;
    const original = device.createBuffer;
    expect(original.name).toBe("ownedBuffer");
    expect(apps()[2].world.resource(DiagnosticsStore).get("memory/bufferBytes")).toBeUndefined();
});
