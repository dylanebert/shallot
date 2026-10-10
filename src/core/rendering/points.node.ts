import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import type { Resource, System } from "../../engine";
import { ClearChangeMarksSystem } from "../../engine/app";
import { precompileState, typegpuRoot } from "../../engine/runtime/gpu";
import {
    MaterialTypes,
    MeshRenderPlugin,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import {
    Clusters,
    CullLightsSystem,
    clusterGpuKey,
    LightCull,
    lightInputKey,
    UpdateLightClustersSystem,
} from "../../standard/rendering/cluster";
import { Backgrounds } from "../../standard/rendering/contract";
import { backgroundsKey } from "../../standard/rendering/contract-state";
import { Lighting, lightingKey } from "../../standard/rendering/lighting";
import { Draws, drawsKey } from "../../standard/rendering/registry";
import { Meshes, meshResourcesKey } from "../mesh/mesh";
import {
    GlobalTransformTickEndSystem,
    GlobalTransformTickStartSystem,
    PrepareGlobalTransformSystem,
    Transform,
} from "../transform";
import { TransformRuntime } from "../transform/global-transform";
import { Frame, frameKey } from "./frame";
import { EndFrameSystem, renderFrameKey } from "./frame-state";
import {
    GlobalTransformHistory,
    GlobalTransformHistoryEndSystem,
    GlobalTransformHistoryStartSystem,
    PrepareGlobalTransformHistorySystem,
} from "./global-transform";
import { blitPipelinesKey } from "./image";
import {
    attachTexture,
    BeginFrameSystem,
    Camera,
    CameraMode,
    CorePipelinePlugin,
    captureTexture,
    EffectPasses,
    MainPassSystem,
    OverlaySystem,
    PrepassSystem,
    PresentationSystem,
    RenderingPlugin,
    RenderPhases,
    Tonemapping,
    TonemappingMethod,
} from "./index";
import { PointsPlugin, pointsState } from "./points.fixture";
import { RenderContext, renderKey } from "./render";
import { viewTargetsKey } from "./targets";
import { compositeCacheKey } from "./tonemapping";
import { TonemappingSystem, tonemappingStateKey } from "./tonemapping-state";
import { Views, viewResourcesKey } from "./view";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [RenderingPlugin, PointsPlugin] },
    {
        defaults: false,
        plugins: [
            RenderingPlugin,
            PointsPlugin,
            StandardRenderingPlugin,
            MeshRenderPlugin,
            CorePipelinePlugin,
        ],
    },
    { defaults: false, plugins: [RenderingPlugin] },
]);

const coreResources = {
    pointsState,
    typegpuRoot,
    precompileState,
    renderKey,
    viewTargetsKey,
    RenderPhases,
    RenderContext,
    renderFrameKey,
    viewResourcesKey,
    Views,
    TransformRuntime: TransformRuntime.key!,
    GlobalTransformHistory: GlobalTransformHistory.key!,
    frameKey,
    Frame,
    blitPipelinesKey,
    compositeCacheKey,
    tonemappingStateKey,
    EffectPasses,
};
const meshResources = {
    meshResourcesKey,
    Meshes,
};
const contractResources = {
    backgroundsKey,
    Backgrounds,
    MaterialTypes,
    drawsKey,
    Draws,
};
const lightResources = { clusterGpuKey, lightInputKey, Clusters, LightCull, lightingKey, Lighting };
const coreSystems = {
    BeginFrameSystem,
    OverlaySystem,
    PresentationSystem,
    EndFrameSystem,
    MainPassSystem,
    PrepassSystem,
    TonemappingSystem,
    ClearChangeMarksSystem,
    GlobalTransformTickStartSystem,
    GlobalTransformTickEndSystem,
    PrepareGlobalTransformSystem,
    GlobalTransformHistoryStartSystem,
    GlobalTransformHistoryEndSystem,
    PrepareGlobalTransformHistorySystem,
};
const lightSystems = { UpdateLightClustersSystem, CullLightsSystem };

function composition() {
    // Inspect identities without lazy resource() access creating the state under test.
    return subjects()[0].world as unknown as {
        _resources: Map<unknown, unknown>;
        _scheduler: { _systems: Set<System> };
    };
}

function exactNames(
    actual: Iterable<unknown>,
    allowed: Record<string, unknown>,
    deferred: Record<string, unknown>,
): string[] {
    const names = new Map(Object.entries(allowed).map(([name, key]) => [key, name]));
    const otherStage = new Set(Object.values(deferred));
    return [...actual]
        .filter((key) => !otherStage.has(key))
        .map((key) => names.get(key) ?? "UNLISTED declaration")
        .sort();
}

function assertRegistration(
    deferredResources: Record<string, unknown>,
    deferredSystems: Record<string, System>,
) {
    const { _resources, _scheduler } = composition();
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
        expect(composition()._resources.has(key)).toBe(false);
    }
    assertRegistration({ ...contractResources, ...lightResources }, lightSystems);
});

test("stage 4: core-only rendering registers no material, background or draw resources", () => {
    assertRegistration(lightResources, lightSystems);
});

test("stage 6: core-only rendering registers no cluster or light-cull declarations", () => {
    assertRegistration(contractResources, {});
});

test("RenderingPlugin without CorePipelinePlugin builds and steps an attached view without validation errors", async () => {
    const { world } = subjects()[2];
    const resources = world as unknown as { _resources: Map<Resource<unknown>, unknown> };
    expect(resources._resources.has(viewTargetsKey)).toBe(false);
    expect(resources._resources.has(RenderPhases)).toBe(false);
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    attachTexture(world, camera, { width: 8, height: 8 });
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    expect(await world.gpu.device.popErrorScope()).toBeNull();
});

test("warmed unchanged points frames create no bind groups", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    attachTexture(world, camera, { width: 8, height: 8 });
    world.step(0);
    world.step(0);
    const device = world.gpu.device;
    const createBindGroup = device.createBindGroup;
    let creations = 0;
    device.createBindGroup = function (descriptor) {
        creations++;
        return createBindGroup.call(this, descriptor);
    };
    device.pushErrorScope("validation");
    try {
        for (let frame = 0; frame < 10; frame++) world.step(0);
        expect(creations).toBe(0);
        world.storage(Camera).antialias.set(camera, 0);
        world.step(0);
        expect(creations).toBe(0);
        const render = world.resource(RenderContext);
        const view = world.resource(Views).get(camera)!;
        const original = render.viewBuffers[view.slot];
        const replacement = device.createBuffer({
            size: original.size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        try {
            render.viewBuffers[view.slot] = replacement;
            world.step(0);
            expect(creations).toBe(1);
            for (let frame = 0; frame < 10; frame++) world.step(0);
            expect(creations).toBe(1);
        } finally {
            render.viewBuffers[view.slot] = original;
            replacement.destroy();
        }
    } finally {
        device.createBindGroup = createBindGroup;
        expect(await device.popErrorScope()).toBeNull();
    }
});

test("points beside a mesh share the view depth: side and front points show, the rear point is occluded", async () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.KhronosPbrNeutral });
    world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
    world.storage(Camera).mode.set(camera, CameraMode.Orthographic);
    world.storage(Camera).size.set(camera, 4);
    world.storage(Camera).antialias.set(camera, 0);
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
