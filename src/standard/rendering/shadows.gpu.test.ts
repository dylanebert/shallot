import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.gpu);

import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DirectionalLight,
} from "../../core/rendering";
import { Transform, type World } from "../../engine";
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
// three headless worlds for the cascade-pool rows, then three rendering worlds for the bound row
const subjects = gpuApps(import.meta.path, [
    ...Array.from({ length: 3 }, () => ({ defaults: false, plugins: [] })),
    rendering,
    rendering,
    rendering,
]);
let nextSubject = 0;

// `updateCascades` rebuilds the sun's boxes only when its inputs change, so the pooled cascade cameras keep
// the camera GlobalTransform the last build wrote. These rows pin what that skip must still restore: a rebuilt pool, and a
// camera whose size or far was written from outside the pass. Both are silent otherwise — the cull frustum
// would simply stop matching the box the atlas renders.

let live: ReturnType<typeof subjects>[number] | null = null;

afterEach(() => {
    // destroy, not just forget: the pooled cameras hold views keyed by eid, and the next row's World
    // hands out the same eids
    if (live) destroyCascades(live.world);
    live?.dispose();
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
    world.add(sun, DirectionalLight);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    world.storage(DirectionalLight).direction.set(sun, -0.3, -0.8, -0.55, 0);
    world.storage(DirectionalLight).maximumDistance.set(sun, 80);
    world.storage(DirectionalLight).numCascades.set(sun, 4);
    world.storage(DirectionalLight).firstCascadeFarBound.set(sun, 10);
    world.storage(DirectionalLight).overlapProportion.set(sun, 0.2);
    world.storage(DirectionalLight).shadowDepthBias.set(sun, 0);
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
        world.storage(Transform).translation.x.get(eid),
        world.storage(Transform).translation.y.get(eid),
        world.storage(Transform).translation.z.get(eid),
    ]);

    world.storage(Transform).translation.set(main, 200, 2, -150, 0);
    world.step(0);
    updateCascades(world, main);
    let moved = false;
    for (let i = 0; i < n; i++) {
        const eid = cams[i];
        if (
            world.storage(Transform).translation.x.get(eid) !== before[i][0] ||
            world.storage(Transform).translation.y.get(eid) !== before[i][1] ||
            world.storage(Transform).translation.z.get(eid) !== before[i][2]
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
    world.add(floor, MeshMaterial, { material });
    const cube = world.create();
    world.add(cube, Transform, { translation: [0, 0.5, 0, 0] });
    world.add(cube, MeshInstance);
    world.add(cube, MeshMaterial, { material });
    const sun = world.create();
    world.add(sun, DirectionalLight, { direction: [-0.4, -1, -0.55, 0] });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    return { camera, sun };
}

// two frames, since the shadows first appear on the second; returns the validation error, if any
async function frames(world: World): Promise<string | null> {
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
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
    expect(await frames(a.world)).toBeNull();
    expect(cascadeCount(a.world)).toBe(3);
    expectBevyBounds(cascadeFars(a.world), 3, 10, 10);
    light(a.world).numCascades.set(sceneA.sun, 1);
    expect(await frames(a.world)).toBeNull();
    expectBevyBounds(cascadeFars(a.world), 1, 10, 10);

    // 4 cascades out to 8, past the default bound: Bevy's bounds shrink from 10 to 8
    const sceneB = renderedScene(b.world);
    light(b.world).maximumDistance.set(sceneB.sun, 8);
    expect(await frames(b.world)).toBeNull();
    expect(cascadeCount(b.world)).toBe(4);
    expectBevyBounds(cascadeFars(b.world), 4, 10, 8);

    // the corrected distance renders the frame a fresh default app does
    light(b.world).maximumDistance.set(sceneB.sun, 50);
    expect(await frames(b.world)).toBeNull();
    const sceneR = renderedScene(reference.world);
    expect(await frames(reference.world)).toBeNull();
    const { rgba } = await captureTexture(b.world, sceneB.camera);
    expect(rgba).toEqual((await captureTexture(reference.world, sceneR.camera)).rgba);
});
