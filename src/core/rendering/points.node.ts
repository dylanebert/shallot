import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import type { Resource, System } from "../../engine";
import { Transform } from "../../engine";
import { ClearChangeMarksSystem } from "../../engine/app";
import { Scheduler } from "../../engine/ecs/scheduler";
import { precompileState, typegpuRoot } from "../../engine/runtime/gpu";
import { FogPlugin } from "../../extras/fog";
import { LinesPlugin } from "../../extras/lines";
import { OutlinePlugin } from "../../extras/outline";
import { SkyPlugin } from "../../extras/sky";
import { SpritePlugin } from "../../extras/sprite";
import { TextPlugin } from "../../extras/text";
import { DepthPrepass, SearPlugin, StandardRenderer } from "../../standard/rendering";
import { Glaze, GlazePlugin, GlazeSystem } from "../../transitional/glaze";
import { MeshInstance, PartPlugin } from "../../transitional/part";
import {
    Clusters,
    CullLightsSystem,
    clusterGpuKey,
    LightCull,
    lightInputKey,
    UpdateLightClustersSystem,
} from "./cluster";
import { Backgrounds, backgroundsKey, Surfaces, surfacesKey } from "./contract";
import { Frame, frameKey } from "./frame";
import { blitPipelinesKey } from "./image";
import {
    attachTexture,
    BeginFrameSystem,
    Camera,
    CameraMode,
    captureTexture,
    EndFrameSystem,
    OverlaySystem,
    PresentationSystem,
    RenderPlugin,
    renderFrameKey,
} from "./index";
import { Lighting, lightingKey } from "./lighting";
import { Meshes, meshResourcesKey } from "./mesh";
import { PointsPlugin, PointsSystem } from "./points.fixture";
import { Draws, drawsKey } from "./registry";
import { Render, renderKey } from "./render";
import { Views, viewResourcesKey } from "./view";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [RenderPlugin, PointsPlugin] },
    { defaults: false, plugins: [RenderPlugin, SearPlugin, PartPlugin, GlazePlugin, PointsPlugin] },
]);

const coreResources = {
    typegpuRoot,
    precompileState,
    renderKey,
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

test.todo("stage 3: core-only rendering registers exactly the core allowlist, deferring only stage 4 light declarations", () => {
    assertRegistration(lightResources, lightSystems);
});

test.todo("stage 4: core-only rendering registers exactly the core allowlist, deferring only stage 3 mesh declarations", () => {
    assertRegistration(meshResources, {});
});

test("registration allowlists reject a retained replacement declaration under a new key", () => {
    const { _resources, _scheduler } = registration();
    const resources = new Map(_resources);
    const systems = new Set(_scheduler._systems);
    try {
        // Model the completed moves, then retain aliases with identical factories/updates under new identities.
        for (const key of [...Object.values(meshResources), ...Object.values(lightResources)])
            _resources.delete(key);
        for (const key of Object.values(lightSystems)) _scheduler._systems.delete(key);
        assertRegistration(lightResources, lightSystems);
        assertRegistration(meshResources, {});
        const replacementResource = { create: Meshes.create };
        _resources.set(replacementResource, {});
        expect(() => assertRegistration(lightResources, lightSystems)).toThrow();
        expect(() => assertRegistration(meshResources, {})).toThrow();
        _resources.delete(replacementResource);
        _scheduler._systems.add({ ...CullLightsSystem });
        expect(() => assertRegistration(lightResources, lightSystems)).toThrow();
        expect(() => assertRegistration(meshResources, {})).toThrow();
    } finally {
        _resources.clear();
        for (const [key, value] of resources) _resources.set(key, value);
        _scheduler._systems.clear();
        for (const system of systems) _scheduler._systems.add(system);
    }
});

test("presentation anchor preserves the updating-system order for fog, outline, sky, lines, sprite and text", () => {
    const plugins = [
        RenderPlugin,
        SearPlugin,
        PartPlugin,
        GlazePlugin,
        FogPlugin,
        OutlinePlugin,
        SkyPlugin,
        LinesPlugin,
        SpritePlugin,
        TextPlugin,
    ];
    const entries = plugins.flatMap((plugin) =>
        (plugin.systems ?? []).map((system) => ({ system, name: plugin.name })),
    );
    function schedule(previous: boolean): string[] {
        const scheduler = new Scheduler();
        const copies = new Map(entries.map(({ system }) => [system, { ...system }]));
        for (const { system, name } of entries) {
            if (previous && system === PresentationSystem) continue;
            const copy = copies.get(system)!;
            copy.before = system.before
                ?.filter((ref) => !previous || ref !== PresentationSystem)
                .map((ref) => copies.get(ref) ?? ref);
            copy.after = (
                previous && system === GlazeSystem
                    ? [BeginFrameSystem, OverlaySystem]
                    : system.after
            )
                ?.filter((ref) => !previous || ref !== PresentationSystem)
                .map((ref) => copies.get(ref) ?? ref);
            scheduler.register(copy, name);
        }
        const sorted = (scheduler as unknown as { getSorted(group: string): System[] }).getSorted(
            "draw",
        );
        return sorted
            .filter((system) => system.update)
            .map((system) => {
                const entry = entries.find(
                    ({ system: original }) => copies.get(original) === system,
                )!;
                return `${entry.name}/${entry.system.name ?? entries.filter(({ name }) => name === entry.name).findIndex(({ system }) => system === entry.system)}`;
            });
    }
    const current = schedule(false);
    expect(current).toEqual(schedule(true));
    expect(
        current.filter((name) => /^(Fog|Outline|Sky|Lines|Sprite|Text)\//.test(name)),
    ).toHaveLength(6);
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
