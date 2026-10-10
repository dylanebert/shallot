import { expect, setDefaultTimeout, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../scripts/test-tiers";
import { Meshes, MeshInstance, MeshPlugin, registerMesh } from "../../src/core/mesh";
import { AmbientLight, attachTexture, Camera } from "../../src/core/rendering";
import { Transform } from "../../src/core/transform";
import type { World } from "../../src/engine";
import { createApp } from "../../src/engine";
import { rawDevice } from "../../src/engine/runtime";
import { Profile, ProfilePlugin } from "../../src/extras/profile";
import {
    MeshRenderPlugin,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../src/standard/rendering";

setDefaultTimeout(CEILING.node);

const vertices = new Float32Array([
    -2, -2, 0, 0, 0, 0, 1, 0, 2, -2, 0, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 1, -2, 2, 0, 0, 0, 0, 1,
    1,
]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
const initialize = (world: World) => registerMesh(world, { name: "slab", vertices, indices });
const plugins = [
    ProfilePlugin,
    StandardRenderingPlugin,
    MeshRenderPlugin,
    { name: "InitializeSlab", dependencies: [MeshPlugin], initialize },
] as const;
const passNames = [
    "mesh:preprocess",
    "cluster:aabbs",
    "light:cull",
    "standard:color",
    "tonemapping",
] as const;
type PassName = (typeof passNames)[number];

async function scene(device?: GPUDevice) {
    const app = await createApp({
        defaults: false,
        plugins: [...plugins],
        ...(device ? { device: rawDevice(device) } : {}),
    });
    const { world } = app;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });
    world.add(world.create(), AmbientLight, { intensity: 1 });
    const slab = world.create();
    world.add(slab, Transform);
    world.add(slab, MeshInstance, { mesh: world.resource(Meshes).id("slab") });
    return app;
}

function withoutPass(device: GPUDevice, name: PassName): () => void {
    const original = device.createCommandEncoder;
    device.createCommandEncoder = (descriptor) => {
        const encoder = Reflect.apply(original, device, [descriptor]) as GPUCommandEncoder;
        const beginComputePass = encoder.beginComputePass;
        encoder.beginComputePass = (passDescriptor) => {
            const label = passDescriptor?.label;
            const suppress =
                (name === "mesh:preprocess" && label === "shallot-mesh-preprocess") ||
                (name === "cluster:aabbs" && label === "shallot-cluster-aabbs") ||
                (name === "light:cull" && label === "shallot-light-cull");
            if (!suppress) return Reflect.apply(beginComputePass, encoder, [passDescriptor]);
            return new Proxy({} as GPUComputePassEncoder, { get: () => () => undefined });
        };
        const beginRenderPass = encoder.beginRenderPass;
        encoder.beginRenderPass = (passDescriptor) => {
            const label = passDescriptor.label ?? "";
            const suppress =
                (name === "standard:color" && label.startsWith("standard-color/")) ||
                (name === "tonemapping" && label === "tonemapping");
            if (!suppress) return Reflect.apply(beginRenderPass, encoder, [passDescriptor]);
            return new Proxy({} as GPURenderPassEncoder, { get: () => () => undefined });
        };
        return encoder;
    };
    return () => {
        device.createCommandEncoder = original;
    };
}

async function frameTime(app: Awaited<ReturnType<typeof scene>>, removed?: PassName) {
    const device = app.world.gpu.device;
    const restore = removed ? withoutPass(device, removed) : () => {};
    const start = performance.now();
    try {
        app.world.step(0);
        await device.queue.onSubmittedWorkDone();
        return performance.now() - start;
    } finally {
        restore();
    }
}

function median(samples: number[]): number {
    return samples.sort((a, b) => a - b)[Math.floor(samples.length / 2)];
}

test("measure register.node first-frame GPU passes", async () => {
    await setupGlobals();
    const owner = await scene();
    const device = rawDevice(owner.world.gpu.device);
    const adapter = owner.world.gpu.adapter;
    const timestampQuery = owner.world.gpu.device.features.has("timestamp-query");
    const useTimestamps = timestampQuery && process.env.PORTABLE_MEASURE_REMOVE_PASSES !== "1";
    console.info(
        `[portable-tiers-pass-measure] adapter=${JSON.stringify(adapter)} timestamp-query=${timestampQuery} method=${useTimestamps ? "timestamps" : "pass removal"}`,
    );

    try {
        if (useTimestamps) {
            const { world } = owner;
            world.step(0);
            await world.gpu.device.queue.onSubmittedWorkDone();
            world.step(0);
            await world.gpu.device.queue.onSubmittedWorkDone();
            await new Promise((resolve) => setTimeout(resolve, 25));
            world.step(0);
            await world.gpu.device.queue.onSubmittedWorkDone();

            const profile = world.resource(Profile);
            console.info(
                `[portable-tiers-pass-measure] first-frame=${JSON.stringify(
                    [...profile.gpuTime].map(([name, ms]) => ({
                        name,
                        ms,
                        fires: profile.gpuFires.get(name),
                    })),
                )}`,
            );
            expect(profile.gpuTime.size).toBeGreaterThan(0);
            expect([...profile.gpuFires.values()].every((fires) => fires === 1)).toBe(true);
            return;
        }

        const baseline = [await frameTime(owner)];
        owner.dispose();
        for (let i = 1; i < 3; i++) {
            const app = await scene(device);
            try {
                baseline.push(await frameTime(app));
            } finally {
                app.dispose();
            }
        }
        const baseMs = median(baseline);
        const removals = [];
        for (const name of passNames) {
            const samples = [];
            for (let i = 0; i < 3; i++) {
                const app = await scene(device);
                try {
                    samples.push(await frameTime(app, name));
                } finally {
                    app.dispose();
                }
            }
            const withoutMs = median(samples);
            removals.push({ name, withoutPassMs: withoutMs, removedDeltaMs: baseMs - withoutMs });
        }
        console.info(
            `[portable-tiers-pass-measure] three first-frame wall-to-fence samples; baseline=${baseMs.toFixed(3)}ms; removal=${JSON.stringify(removals)}`,
        );
        expect(removals).toHaveLength(passNames.length);
        expect(baseline.every(Number.isFinite)).toBe(true);
    } finally {
        owner.dispose();
    }
});
