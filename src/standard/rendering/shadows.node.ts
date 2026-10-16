import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DirectionalLight,
    PointLight,
    SpotLight,
} from "../../core/rendering";
import { GlobalTransform, Transform, TransformPlugin } from "../../core/transform";
import { lookAtRotation, type World } from "../../engine";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import {
    cascadeComboEids,
    cascadeCount,
    cascadeCovers,
    cascadeFars,
    destroyCascades,
    resetCascades,
    updateCascades,
} from "./shadows";

const rendering = { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin] };
// three headless worlds for the cascade-pool rows, three rendering worlds for the bound row, one for
// the re-gather growth row, then changed and fresh worlds for the light-change rows
const subjects = gpuApps(import.meta.path, [
    ...Array.from({ length: 3 }, () => ({ defaults: false, plugins: [TransformPlugin] })),
    ...Array.from({ length: 14 }, () => rendering),
]);
let nextSubject = 0;

// `updateCascades` rebuilds the sun's boxes only when its inputs change, so the pooled cascade cameras keep
// the camera GlobalTransform the last build wrote. These rows pin what that skip must still restore: a rebuilt pool, and a
// camera whose size or far was written from outside the pass. Both are silent otherwise — the cull frustum
// would simply stop matching the box the atlas renders.

let live: ReturnType<typeof subjects>[number] | null = null;

afterEach(async () => {
    // destroy, not just forget: the pooled cameras hold views keyed by eid, and the next row's World
    // hands out the same eids
    if (live) destroyCascades(live.world);
    if (live) await disposeGpuApps([live]);
    live = null;
});

// a headless World with one posed perspective camera and one shadow-casting sun
async function sunScene() {
    live = subjects()[nextSubject++];
    const world = live.world;
    const main = world.create();
    world.add(main, Transform);
    world.add(main, Camera);
    world.storage(Transform).translation.set(main, 0, 2, 10, 0);
    world.storage(Transform).rotation.set(main, 0, 0, 0, 1);
    world.storage(Transform).scale.set(main, 1, 1, 1, 1);
    world.storage(Camera).mode.set(main, CameraMode.Perspective);
    world.storage(Camera).fov.set(main, 60);
    world.storage(Camera).near.set(main, 0.1);
    world.storage(Camera).far.set(main, 500);
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.3, -0.8, -0.55) });
    world.add(sun, DirectionalLight);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    world.storage(DirectionalLight).maximumDistance.set(sun, 80);
    world.storage(DirectionalLight).numCascades.set(sun, 4);
    world.storage(DirectionalLight).firstCascadeFarBound.set(sun, 10);
    world.storage(DirectionalLight).overlapProportion.set(sun, 0.2);
    world.storage(DirectionalLight).depthBias.set(sun, 0);
    world.storage(DirectionalLight).shadowNormalBias.set(sun, 0);
    world.step(0);
    return { world, main, sun };
}

test("cascade cameras created after a pool rebuild keep a zeroed size and far because the pass reads its inputs as unchanged, so every caster would cull against a degenerate frustum", async () => {
    const { world, main } = await sunScene();
    updateCascades(world, main);
    const n = cascadeCount(world);
    expect(n).toBeGreaterThan(0);
    const covers = Array.from(cascadeCovers(world).slice(0, n));
    for (const eid of cascadeComboEids(world).slice(0, n)) {
        expect(world.storage(Camera).size.get(eid)).toBeGreaterThan(0);
        expect(world.storage(Camera).far.get(eid)).toBeGreaterThan(0);
    }

    // the pool is dropped and rebuilt on fresh eids, with every input otherwise identical
    destroyCascades(world);
    resetCascades(world);
    updateCascades(world, main);
    expect(cascadeCount(world)).toBe(n);
    const rebuilt = cascadeComboEids(world).slice(0, n);
    for (let i = 0; i < n; i++) {
        expect(world.storage(Camera).size.get(rebuilt[i])).toBeCloseTo(covers[i], 4);
        expect(world.storage(Camera).far.get(rebuilt[i])).toBeGreaterThan(0);
    }
});

test("a cascade camera whose size or far is overwritten between frames keeps the foreign value, so its cull frustum would no longer match the box the atlas renders into its tile", async () => {
    const { world, main } = await sunScene();
    updateCascades(world, main);
    const n = cascadeCount(world);
    const cams = cascadeComboEids(world).slice(0, n);
    const size = world.storage(Camera).size.get(cams[0]);
    const far = world.storage(Camera).far.get(cams[0]);

    // an unchanged frame reposes nothing: a field the pass does not compare keeps a foreign value
    world.storage(Camera).near.set(cams[0], 7);
    updateCascades(world, main);
    expect(world.storage(Camera).near.get(cams[0])).toBe(7);

    // size and far are compared, so writing either restores the complete camera state
    world.storage(Camera).size.set(cams[0], size + 3);
    updateCascades(world, main);
    expect(world.storage(Camera).size.get(cams[0])).toBeCloseTo(size, 4);
    expect(world.storage(Camera).near.get(cams[0])).toBe(0);

    world.storage(Camera).far.set(cams[0], far + 3);
    updateCascades(world, main);
    expect(world.storage(Camera).far.get(cams[0])).toBeCloseTo(far, 4);
});

test("the cascade pass rebuilds its boxes after the main camera GlobalTransform changes", async () => {
    const { world, main } = await sunScene();
    updateCascades(world, main);
    const n = cascadeCount(world);
    const cams = cascadeComboEids(world).slice(0, n);
    const before = cams.map((eid) => [
        world.storage(GlobalTransform).translation.x.get(eid),
        world.storage(GlobalTransform).translation.y.get(eid),
        world.storage(GlobalTransform).translation.z.get(eid),
    ]);

    world.storage(Transform).translation.set(main, 200, 2, -150, 0);
    world.step(0);
    updateCascades(world, main);
    let moved = false;
    for (let i = 0; i < n; i++) {
        const eid = cams[i];
        if (
            world.storage(GlobalTransform).translation.x.get(eid) !== before[i][0] ||
            world.storage(GlobalTransform).translation.y.get(eid) !== before[i][1] ||
            world.storage(GlobalTransform).translation.z.get(eid) !== before[i][2]
        )
            moved = true;
    }
    expect(moved).toBe(true);
});

// a cube on a floor under a shadowed sun with default cascade fields, drawn by a 32×32 camera
function renderedScene(world: World): { camera: number; sun: number } {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 3, 8, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });
    const material = world.resource(Materials).add(StandardMaterial());
    const floor = world.create();
    world.add(floor, Transform, { translation: [0, -0.5, 0, 0], scale: [40, 0.2, 40, 0] });
    world.add(floor, MeshInstance);
    world.add(floor, MeshMaterial, material);
    const cube = world.create();
    world.add(cube, Transform, { translation: [0, 0.5, 0, 0] });
    world.add(cube, MeshInstance);
    world.add(cube, MeshMaterial, material);
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -1, -0.55) });
    world.add(sun, DirectionalLight);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    return { camera, sun };
}

// one frame; returns the validation error, if any
async function frame(world: World): Promise<string | null> {
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    return (await world.gpu.device.popErrorScope())?.message ?? null;
}

// asserts the first `n` cascade far bounds follow Bevy's `calculate_cascade_bounds` (bevy_light cascade.rs)
// within f32 rounding: Bevy evaluates it in f32, Shallot in f64 stored as f32, so the bits may differ in
// the last place, while a different split misses by far more than the 1e-6 relative tolerance
function expectBevyBounds(fars: Float32Array, n: number, first: number, max: number): void {
    const base = n === 1 ? 1 : (max / first) ** (1 / (n - 1));
    for (let i = 0; i < n; i++) {
        const bound = n === 1 ? max : first * base ** i;
        expect(Math.abs(fars[i] / bound - 1)).toBeLessThanOrEqual(1e-6);
    }
}

test("a sun whose firstCascadeFarBound reaches or passes its maximumDistance splits by Bevy's cascade bounds and keeps rendering, and one cascade ignores the bound", async () => {
    const [a, b, reference] = subjects().slice(3);
    const light = (world: World) => world.storage(DirectionalLight);

    // Bevy's deferred_rendering example: 3 cascades out to 10, the first bound left at its default 10
    const sceneA = renderedScene(a.world);
    light(a.world).numCascades.set(sceneA.sun, 3);
    light(a.world).maximumDistance.set(sceneA.sun, 10);
    expect(await frame(a.world)).toBeNull();
    expect(cascadeCount(a.world)).toBe(3);
    expectBevyBounds(cascadeFars(a.world), 3, 10, 10);
    light(a.world).numCascades.set(sceneA.sun, 1);
    expect(await frame(a.world)).toBeNull();
    expectBevyBounds(cascadeFars(a.world), 1, 10, 10);

    // 4 cascades out to 8, past the default bound: Bevy's bounds shrink from 10 to 8
    const sceneB = renderedScene(b.world);
    light(b.world).maximumDistance.set(sceneB.sun, 8);
    expect(await frame(b.world)).toBeNull();
    expect(cascadeCount(b.world)).toBe(4);
    expectBevyBounds(cascadeFars(b.world), 4, 10, 8);

    // the corrected distance renders the frame a fresh default app does
    light(b.world).maximumDistance.set(sceneB.sun, 50);
    expect(await frame(b.world)).toBeNull();
    const sceneR = renderedScene(reference.world);
    expect(await frame(reference.world)).toBeNull();
    const { rgba } = await captureTexture(b.world, sceneB.camera);
    expect(rgba).toEqual((await captureTexture(reference.world, sceneR.camera)).rgba);
});

test("spawning an entity a frame under a casting sun and point light reallocates each re-gather list and its bind groups only as the high-water mark doubles, not every frame", async () => {
    const { world } = subjects()[6];
    renderedScene(world);
    const point = world.create();
    world.add(point, Transform, { translation: [2, 2, 3, 0] });
    world.add(point, PointLight, { intensity: 315225.92, range: 10 });
    world.storage(PointLight).shadowMapsEnabled.set(point, 1);
    expect(await frame(world)).toBeNull();

    const device = world.gpu.device;
    const createBuffer = device.createBuffer.bind(device);
    const createBindGroup = device.createBindGroup.bind(device);
    const lists = { "standard-cascade-regather-eids": 0, "standard-point-regather-eids": 0 };
    let groups = 0;
    let firstFrameGroups = 0;
    device.createBuffer = (descriptor) => {
        const label = descriptor.label as keyof typeof lists;
        if (label in lists) lists[label]++;
        return createBuffer(descriptor);
    };
    device.createBindGroup = (descriptor) => {
        groups++;
        return createBindGroup(descriptor);
    };
    const start = world.entityHighWater;
    device.pushErrorScope("validation");
    try {
        for (let frame = 0; frame < 100; frame++) {
            world.create();
            world.step(0);
            if (frame === 0) firstFrameGroups = groups;
        }
    } finally {
        device.createBuffer = createBuffer;
        device.createBindGroup = createBindGroup;
    }
    expect((await device.popErrorScope())?.message ?? null).toBeNull();

    // one growth per doubling of the high-water mark across the run, plus the first spawn's
    const growths = Math.ceil(Math.log2(world.entityHighWater / start)) + 1;
    expect(lists["standard-cascade-regather-eids"]).toBeGreaterThan(0);
    expect(lists["standard-point-regather-eids"]).toBeGreaterThan(0);
    expect(lists["standard-cascade-regather-eids"]).toBeLessThanOrEqual(growths);
    expect(lists["standard-point-regather-eids"]).toBeLessThanOrEqual(growths);
    expect(groups).toBeLessThanOrEqual(growths * firstFrameGroups);
});

// a point and a spot light casting beside the cube, both shadowing
function localLights(
    world: World,
    point: [number, number, number],
    spot: [number, number, number],
): { point: number; spot: number } {
    const pointEid = world.create();
    world.add(pointEid, Transform, { translation: point });
    world.add(pointEid, PointLight, { intensity: 315225.92, range: 10 });
    world.storage(PointLight).shadowMapsEnabled.set(pointEid, 1);
    const spotEid = world.create();
    world.add(spotEid, Transform, {
        translation: [-2, 3, 2, 0],
        rotation: lookAtRotation(-2, 3, 2, spot[0], spot[1], spot[2]),
    });
    world.add(spotEid, SpotLight, { intensity: 315225.92, range: 12 });
    world.storage(SpotLight).shadowMapsEnabled.set(spotEid, 1);
    return { point: pointEid, spot: spotEid };
}

test("the first frame after a casting sun turns matches a fresh frame of the turned sun", async () => {
    const [changed, fresh] = subjects().slice(7, 9);
    const scene = renderedScene(changed.world);
    expect(await frame(changed.world)).toBeNull();
    changed.world
        .storage(Transform)
        .rotation.set(scene.sun, ...lookAtRotation(0, 0, 0, 0.6, -1, 0.3));
    changed.world.step(0);

    const reference = renderedScene(fresh.world);
    fresh.world
        .storage(Transform)
        .rotation.set(reference.sun, ...lookAtRotation(0, 0, 0, 0.6, -1, 0.3));
    expect(await frame(fresh.world)).toBeNull();
    expect((await captureTexture(changed.world, scene.camera)).rgba).toEqual(
        (await captureTexture(fresh.world, reference.camera)).rgba,
    );
});

test("the first frame after a casting point light moves and a casting spot light turns matches a fresh frame of both", async () => {
    const [changed, fresh] = subjects().slice(9, 11);
    const scene = renderedScene(changed.world);
    changed.world.storage(DirectionalLight).shadowMapsEnabled.set(scene.sun, 0);
    const lights = localLights(changed.world, [2, 2, 3], [0, 0, 0]);
    expect(await frame(changed.world)).toBeNull();
    changed.world.storage(Transform).translation.set(lights.point, -2, 2, 3, 0);
    changed.world
        .storage(Transform)
        .rotation.set(lights.spot, ...lookAtRotation(-2, 3, 2, 2, 0, -1));
    changed.world.step(0);

    const reference = renderedScene(fresh.world);
    fresh.world.storage(DirectionalLight).shadowMapsEnabled.set(reference.sun, 0);
    localLights(fresh.world, [-2, 2, 3], [2, 0, -1]);
    expect(await frame(fresh.world)).toBeNull();
    expect((await captureTexture(changed.world, scene.camera)).rgba).toEqual(
        (await captureTexture(fresh.world, reference.camera)).rgba,
    );
});

test("casting point and spot shadows follow fractional fixed placement, including step(0)", async () => {
    const [changed, fresh] = subjects().slice(11, 13);
    const scene = renderedScene(changed.world);
    changed.world.storage(DirectionalLight).shadowMapsEnabled.set(scene.sun, 0);
    const lights = localLights(changed.world, [2, 2, 3], [0, 0, 0]);
    expect(await frame(changed.world)).toBeNull();
    const start = lookAtRotation(-2, 3, 2, 0, 0, 0);
    const end = lookAtRotation(-2, 3, 2, 2, 0, -1);
    const sign = start.reduce((sum, v, i) => sum + v * end[i]!, 0) < 0 ? -1 : 1;
    const mid = start.map((v, i) => v * sign + end[i]!);
    const length = Math.hypot(...mid);
    const rotation = mid.map((v) => v / length) as [number, number, number, number];
    changed.world.addSystem({
        group: "fixed",
        update(world) {
            world.storage(Transform).translation.set(lights.point, -2, 2, 3);
            world.storage(Transform).rotation.set(lights.spot, ...end);
        },
    });
    const reference = renderedScene(fresh.world);
    fresh.world.storage(DirectionalLight).shadowMapsEnabled.set(reference.sun, 0);
    const referenceLights = localLights(fresh.world, [0, 2, 3], [0, 0, 0]);
    fresh.world.storage(Transform).rotation.set(referenceLights.spot, ...rotation);
    expect(await frame(fresh.world)).toBeNull();
    changed.world.step(1.5 / 60);
    for (let i = 0; i < 2; i++) {
        expect((await captureTexture(changed.world, scene.camera)).rgba).toEqual(
            (await captureTexture(fresh.world, reference.camera)).rgba,
        );
        changed.world.step(0);
    }
});

test("a simulation-written flashlight shadows from the camera pose at fractional alpha and step(0)", async () => {
    const [changed, ...references] = subjects().slice(13, 17);
    const scene = renderedScene(changed.world);
    changed.world.storage(DirectionalLight).shadowMapsEnabled.set(scene.sun, 0);
    const light = changed.world.create();
    changed.world.add(light, Transform, { translation: [-2, 3, 2] });
    changed.world.add(light, SpotLight, { intensity: 315225.92, range: 12 });
    changed.world.storage(SpotLight).shadowMapsEnabled.set(light, 1);
    expect(await frame(changed.world)).toBeNull();
    let frameIndex = 0;
    const positions: [number, number, number][] = [
        [0.5, 3, 7],
        [-0.5, 2.8, 6.5],
        [1, 3.2, 7.5],
    ];
    const rotations = positions.map((p, i) => lookAtRotation(...p, (i - 1) * 0.7, 0, 0));
    const moveCamera = {
        name: "test-move-camera",
        group: "simulation" as const,
        update(world: World) {
            world.storage(Transform).translation.set(scene.camera, ...positions[frameIndex]!);
            world.storage(Transform).rotation.set(scene.camera, ...rotations[frameIndex]!);
        },
    };
    changed.world.addSystem(moveCamera);
    changed.world.addSystem({
        group: "simulation",
        after: [moveCamera],
        update(world) {
            const pose = world.storage(Transform);
            world
                .storage(Transform)
                .translation.set(
                    light,
                    pose.translation.x.get(scene.camera),
                    pose.translation.y.get(scene.camera),
                    pose.translation.z.get(scene.camera),
                );
            world
                .storage(Transform)
                .rotation.set(
                    light,
                    pose.rotation.x.get(scene.camera),
                    pose.rotation.y.get(scene.camera),
                    pose.rotation.z.get(scene.camera),
                    pose.rotation.w.get(scene.camera),
                );
        },
    });
    for (frameIndex = 0; frameIndex < positions.length; frameIndex++) {
        const fresh = references[frameIndex]!;
        const reference = renderedScene(fresh.world);
        fresh.world.storage(DirectionalLight).shadowMapsEnabled.set(reference.sun, 0);
        const translation = positions[frameIndex]!;
        const rotation = rotations[frameIndex]!;
        fresh.world.storage(Transform).translation.set(reference.camera, ...translation);
        fresh.world.storage(Transform).rotation.set(reference.camera, ...rotation);
        const referenceLight = fresh.world.create();
        fresh.world.add(referenceLight, Transform, { translation, rotation });
        fresh.world.add(referenceLight, SpotLight, { intensity: 315225.92, range: 12 });
        fresh.world.storage(SpotLight).shadowMapsEnabled.set(referenceLight, 1);
        expect(await frame(fresh.world)).toBeNull();
        changed.world.step(frameIndex === 0 ? 1.5 / 60 : 0);
        expect(changed.world.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect((await captureTexture(changed.world, scene.camera)).rgba).toEqual(
            (await captureTexture(fresh.world, reference.camera)).rgba,
        );
    }
});
