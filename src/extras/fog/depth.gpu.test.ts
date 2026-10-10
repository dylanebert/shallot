import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, captureTexture, DepthPrepass, Views } from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { maskLayoutOcclude, maskLayoutOccludeMultisampled } from "../outline/passes";
import { Fog, FogPlugin, NoFog } from ".";
import { fogLayout0, fogLayout0Multisampled, fogLayout1 } from "./pipeline";

setDefaultTimeout(CEILING.gpu);

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);

function addCamera(world: World, noFog = false) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { clearColor: 0x204060 });
    world.add(camera, StandardRenderer);
    if (noFog) world.add(camera, NoFog);
    attachTexture(world, camera, { width: 24, height: 24 });
    return camera;
}

function storageCount(layout: unknown, stage: "compute" | "vertex" | "fragment"): number {
    const entries = (
        layout as {
            entries: Record<string, { storage?: unknown; visibility?: readonly string[] }>;
        }
    ).entries;
    return Object.values(entries).filter(
        (entry) => entry.storage && entry.visibility?.includes(stage),
    ).length;
}

function maximumDifference(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
    let max = 0;
    for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
    return max;
}

test("depth-reader variants keep their per-stage storage-buffer counts", () => {
    const counts = {
        fogCompute: storageCount(fogLayout0, "compute") + storageCount(fogLayout1, "compute"),
        fogMultisampledCompute:
            storageCount(fogLayout0Multisampled, "compute") + storageCount(fogLayout1, "compute"),
        outlineVertex: storageCount(maskLayoutOcclude, "vertex"),
        outlineFragment: storageCount(maskLayoutOcclude, "fragment"),
        outlineMultisampledVertex: storageCount(maskLayoutOccludeMultisampled, "vertex"),
        outlineMultisampledFragment: storageCount(maskLayoutOccludeMultisampled, "fragment"),
    };
    console.log("shared depth layout storage buffers", counts);
    expect(counts).toEqual({
        fogCompute: 1,
        fogMultisampledCompute: 1,
        outlineVertex: 6,
        outlineFragment: 1,
        outlineMultisampledVertex: 6,
        outlineMultisampledFragment: 1,
    });
    expect(Math.max(...Object.values(counts))).toBeLessThanOrEqual(8);
});

test("fog requests a multisampled depth lane without DepthPrepass and skips NoFog cameras", async () => {
    const { world } = subjects()[0]!;
    const camera = addCamera(world);
    const noFogCamera = addCamera(world, true);
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    const fog = world.create();
    world.add(fog, Fog, { density: 0.5, color: 0xff0000, steps: 16, jitter: 0 });

    const created: { label: string | undefined; sampleCount: number }[] = [];
    const device = world.gpu.device;
    const createTexture = device.createTexture.bind(device);
    device.createTexture = (descriptor) => {
        created.push({ label: descriptor.label, sampleCount: descriptor.sampleCount ?? 1 });
        return createTexture(descriptor);
    };
    world.step(0);
    world.step(0);

    const view = world.resource(Views).get(camera)!;
    const noFogView = world.resource(Views).get(noFogCamera)!;
    expect(view.depth).not.toBeNull();
    expect(noFogView.depth).toBeNull();
    expect(created.find((target) => target.label === `standard-depth-${camera}`)?.sampleCount).toBe(
        4,
    );

    const fogged = await captureTexture(world, camera);
    const clear = await captureTexture(world, noFogCamera);
    expect(maximumDifference(fogged.rgba, clear.rgba)).toBeGreaterThan(10);
    expect(maximumDifference(fogged.rgba.slice(0, 4), clear.rgba.slice(0, 4))).toBe(0);
});

test("NoFog opts out even when the user explicitly requests DepthPrepass", async () => {
    const { world } = subjects()[1]!;
    const camera = addCamera(world);
    world.add(camera, DepthPrepass);
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    const fog = world.create();
    world.add(fog, Fog, { density: 0.5, color: 0xff0000, steps: 16, jitter: 0 });
    world.step(0);
    world.step(0);
    const fogged = await captureTexture(world, camera);

    world.remove(fog, Fog);
    world.step(0);
    const clear = await captureTexture(world, camera);

    world.add(fog, Fog, { density: 0.5, color: 0xff0000, steps: 16, jitter: 0 });
    world.add(camera, NoFog);
    world.step(0);
    const optedOut = await captureTexture(world, camera);
    expect(world.resource(Views).get(camera)!.depth).not.toBeNull();
    expect(maximumDifference(fogged.rgba, clear.rgba)).toBeGreaterThan(10);
    expect(maximumDifference(optedOut.rgba, clear.rgba)).toBe(0);
});
