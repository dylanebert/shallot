import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.gpu);

import { gpuApps } from "../../../scripts/gpu.fixture";
import { Camera, CameraMode, DirectionalLight } from "../../core/rendering";
import { Transform } from "../../engine";
import {
    cascadeComboEids,
    cascadeCount,
    cascadeCovers,
    destroyCascades,
    resetCascades,
    updateCascades,
} from "./shadows";

const subjects = gpuApps(
    import.meta.path,
    Array.from({ length: 4 }, () => ({ defaults: false, plugins: [] })),
);
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

test("a sun whose firstCascadeFarBound reaches its maximumDistance is refused by name, and one cascade ignores the bound", async () => {
    const { world, main, sun } = await sunScene();
    const light = world.storage(DirectionalLight);
    light.firstCascadeFarBound.set(sun, 80);
    expect(() => updateCascades(world, main)).toThrow(
        `standard: DirectionalLight ${sun} firstCascadeFarBound (80) must be less than its maximumDistance (80)`,
    );
    light.firstCascadeFarBound.set(sun, 120);
    expect(() => updateCascades(world, main)).toThrow("firstCascadeFarBound (120)");
    light.numCascades.set(sun, 1);
    updateCascades(world, main);
    expect(cascadeCount(world)).toBe(1);
});
