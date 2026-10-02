import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import type { Resource, System } from "../../engine";
import { Transform } from "../../engine";
import { ClearChangeMarksSystem } from "../../engine/app";
import { precompileState, typegpuRoot } from "../../engine/runtime/gpu";
import { PartPlugin, StandardRenderer, StandardRenderingPlugin } from "../../standard/rendering";
import {
    Clusters,
    CullLightsSystem,
    clusterGpuKey,
    LightCull,
    lightInputKey,
    UpdateLightClustersSystem,
} from "../../standard/rendering/cluster";
import { Backgrounds, Surfaces } from "../../standard/rendering/contract";
import { backgroundsKey, surfacesKey } from "../../standard/rendering/contract-state";
import { Lighting, lightingKey } from "../../standard/rendering/lighting";
import { Draws, drawsKey } from "../../standard/rendering/registry";
import { Glaze, GlazePlugin } from "../../transitional/glaze";
import { Meshes, meshResourcesKey } from "../mesh/mesh";
import { Frame, frameKey } from "./frame";
import { EndFrameSystem, renderFrameKey } from "./frame-state";
import { blitPipelinesKey } from "./image";
import {
    attachTexture,
    BeginFrameSystem,
    Camera,
    CameraMode,
    captureTexture,
    DepthPrepass,
    OverlaySystem,
    PresentationSystem,
    RenderingPlugin,
} from "./index";
import { PointsPlugin, PointsSystem } from "./points.fixture";
import { Render, renderKey } from "./render";
import { viewTargetsKey } from "./targets";
import { Views, viewResourcesKey } from "./view";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [RenderingPlugin, PointsPlugin] },
    {
        defaults: false,
        plugins: [RenderingPlugin, StandardRenderingPlugin, PartPlugin, GlazePlugin, PointsPlugin],
    },
]);

const coreResources = {
    typegpuRoot,
    precompileState,
    renderKey,
    viewTargetsKey,
    Render,
    renderFrameKey,
    viewResourcesKey,
    Views,
    frameKey,
    Frame,
    blitPipelinesKey,
};
const meshResources = {
    meshResourcesKey,
    Meshes,
};
const contractResources = {
    surfacesKey,
    Surfaces,
    backgroundsKey,
    Backgrounds,
    drawsKey,
    Draws,
};
const lightResources = { clusterGpuKey, lightInputKey, Clusters, LightCull, lightingKey, Lighting };
const coreSystems = {
    BeginFrameSystem,
    OverlaySystem,
    PresentationSystem,
    EndFrameSystem,
    PointsSystem,
    ClearChangeMarksSystem,
};
const lightSystems = { UpdateLightClustersSystem, CullLightsSystem };

function registration() {
    // Inspect identities without lazy resource() access creating the state under test.
    return subjects()[0].world as unknown as {
        _resources: Map<Resource<unknown>, unknown>;
        _scheduler: { _systems: Set<System> };
    };
}

function exactNames<T>(
    actual: Iterable<T>,
    allowed: Record<string, T>,
    deferred: Record<string, T>,
): string[] {
    const names = new Map(Object.entries(allowed).map(([name, key]) => [key, name]));
    const otherStage = new Set(Object.values(deferred));
    return [...actual]
        .filter((key) => !otherStage.has(key))
        .map((key) => names.get(key) ?? "UNLISTED declaration")
        .sort();
}

function assertRegistration(
    deferredResources: Record<string, Resource<unknown>>,
    deferredSystems: Record<string, System>,
) {
    const { _resources, _scheduler } = registration();
    expect({
        resources: exactNames(_resources.keys(), coreResources, deferredResources),
        systems: exactNames(_scheduler._systems, coreSystems, deferredSystems),
    }).toEqual({
        resources: Object.keys(coreResources).sort(),
        systems: Object.keys(coreSystems).sort(),
    });
}

test("stage 3: core-only rendering registers no mesh resources", () => {
    for (const key of Object.values(meshResources)) {
        expect(registration()._resources.has(key)).toBe(false);
    }
    assertRegistration({ ...contractResources, ...lightResources }, lightSystems);
});

test("stage 4: core-only rendering registers no surface, background or draw resources", () => {
    assertRegistration(lightResources, lightSystems);
});

test("stage 6: core-only rendering registers no cluster or light-cull declarations", () => {
    assertRegistration(contractResources, {});
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
