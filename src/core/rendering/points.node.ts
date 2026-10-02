import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { Transform } from "../../engine";
import { DepthPrepass, SearPlugin, StandardRenderer } from "../../standard/rendering";
import { Glaze, GlazePlugin } from "../../transitional/glaze";
import { MeshInstance, PartPlugin } from "../../transitional/part";
import { Clusters, CullLightsSystem, LightCull, UpdateLightClustersSystem } from "./cluster";
import { Backgrounds, Surfaces } from "./contract";
import { attachTexture, Camera, CameraMode, captureTexture, RenderPlugin } from "./index";
import { Meshes } from "./mesh";
import { PointsPlugin } from "./points.fixture";
import { Draws } from "./registry";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [RenderPlugin, PointsPlugin] },
    { defaults: false, plugins: [RenderPlugin, SearPlugin, PartPlugin, GlazePlugin, PointsPlugin] },
]);

function registeredResources() {
    // Inspect registration without lazy resource() access creating the very state under test.
    return (subjects()[0].world as unknown as { _resources: Map<unknown, unknown> })._resources;
}

test.todo("stage 3: core-only rendering registers no mesh, surface, background or draw resource", () => {
    const resources = registeredResources();
    expect([Meshes, Surfaces, Backgrounds, Draws].filter((key) => resources.has(key))).toEqual([]);
});

test.todo("stage 4: core-only rendering registers no cluster or light-cull system or resource", () => {
    const world = subjects()[0].world;
    expect({
        clusters: registeredResources().has(Clusters),
        lightCull: registeredResources().has(LightCull),
        update: world.hasSystem(UpdateLightClustersSystem),
        cull: world.hasSystem(CullLightsSystem),
    }).toEqual({ clusters: false, lightCull: false, update: false, cull: false });
});

test("points beside a mesh share the view depth: side and front points show, the rear point is occluded", async () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, DepthPrepass);
    world.add(camera, Glaze);
    world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
    world.storage(Camera).mode.set(camera, CameraMode.Orthographic);
    world.storage(Camera).size.set(camera, 4);
    attachTexture(world, camera, { width: 64, height: 64 });
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    world.storage(Transform).translation.set(mesh, 0, 0, 1, 0);
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    const { rgba } = await captureTexture(world, camera);
    expect(await world.gpu.device.popErrorScope()).toBeNull();
    const green: [number, number][] = [];
    for (let y = 0; y < 64; y++)
        for (let x = 0; x < 64; x++) {
            const i = (y * 64 + x) * 4;
            if (rgba[i + 1] > 100 && rgba[i] < 50 && rgba[i + 2] < 50) green.push([x, y]);
        }
    expect(green).toEqual([
        [31, 29],
        [23, 31],
        [39, 31],
    ]);
});
