import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

setDefaultTimeout(1000);

import { Camera, CameraMode, DirectionalLight } from "../../core/rendering";
import { Transform } from "../../engine";
import { sharedGpuBuild } from "../../engine/app/gpu.fixture";
import {
    cascadeComboEids,
    cascadeCount,
    cascadeCovers,
    destroyCascades,
    resetCascades,
    Shadow,
    updateCascades,
} from "./shadows";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();
const build = await sharedGpuBuild();

// `updateCascades` rebuilds the sun's boxes only when its inputs change, so the pooled cascade cameras keep
// the camera GlobalTransform the last build wrote. These rows pin what that skip must still restore: a rebuilt pool, and a
// camera whose size or far was written from outside the pass. Both are silent otherwise — the cull frustum
// would simply stop matching the box the atlas renders.

let live: Awaited<ReturnType<typeof build>> | null = null;

afterEach(() => {
    // destroy, not just forget: the pooled cameras hold views keyed by eid, and the next row's State
    // hands out the same eids
    if (live) destroyCascades(live.state);
    live?.dispose();
    live = null;
});

// a headless State with one posed perspective camera and one shadow-casting sun
async function sunScene() {
    live = await build({ defaults: false, plugins: [] });
    const state = live.state;
    const main = state.create();
    state.add(main, Transform);
    state.add(main, Camera);
    Transform.pos.set(main, 0, 2, 10, 0);
    Transform.rot.set(main, 0, 0, 0, 1);
    Transform.scale.set(main, 1, 1, 1, 1);
    Camera.mode.set(main, CameraMode.Perspective);
    Camera.fov.set(main, 60);
    Camera.near.set(main, 0.1);
    Camera.far.set(main, 500);
    const sun = state.create();
    state.add(sun, DirectionalLight);
    state.add(sun, Shadow);
    DirectionalLight.direction.set(sun, -0.3, -0.8, -0.55, 0);
    Shadow.distance.set(sun, 80);
    Shadow.depthBias.set(sun, 0);
    Shadow.normalBias.set(sun, 0);
    state.step(0);
    return { state, main, sun };
}

test("cascade cameras created after a pool rebuild keep a zeroed size and far because the pass reads its inputs as unchanged, so every caster would cull against a degenerate frustum", async () => {
    const { state, main } = await sunScene();
    updateCascades(state, main);
    const n = cascadeCount(state);
    expect(n).toBeGreaterThan(0);
    const covers = Array.from(cascadeCovers(state).slice(0, n));
    for (const eid of cascadeComboEids(state).slice(0, n)) {
        expect(Camera.size.get(eid)).toBeGreaterThan(0);
        expect(Camera.far.get(eid)).toBeGreaterThan(0);
    }

    // the pool is dropped and rebuilt on fresh eids, with every input otherwise identical
    destroyCascades(state);
    resetCascades(state);
    updateCascades(state, main);
    expect(cascadeCount(state)).toBe(n);
    const rebuilt = cascadeComboEids(state).slice(0, n);
    for (let i = 0; i < n; i++) {
        expect(Camera.size.get(rebuilt[i])).toBeCloseTo(covers[i], 4);
        expect(Camera.far.get(rebuilt[i])).toBeGreaterThan(0);
    }
});

test("a cascade camera whose size or far is overwritten between frames keeps the foreign value, so its cull frustum would no longer match the box the atlas renders into its tile", async () => {
    const { state, main } = await sunScene();
    updateCascades(state, main);
    const n = cascadeCount(state);
    const cams = cascadeComboEids(state).slice(0, n);
    const size = Camera.size.get(cams[0]);
    const far = Camera.far.get(cams[0]);

    // an unchanged frame reposes nothing: a field the pass does not compare keeps a foreign value
    Camera.near.set(cams[0], 7);
    updateCascades(state, main);
    expect(Camera.near.get(cams[0])).toBe(7);

    // size and far are compared, so writing either restores the complete camera state
    Camera.size.set(cams[0], size + 3);
    updateCascades(state, main);
    expect(Camera.size.get(cams[0])).toBeCloseTo(size, 4);
    expect(Camera.near.get(cams[0])).toBe(0);

    Camera.far.set(cams[0], far + 3);
    updateCascades(state, main);
    expect(Camera.far.get(cams[0])).toBeCloseTo(far, 4);
});

test("the cascade pass rebuilds its boxes after the main camera GlobalTransform changes", async () => {
    const { state, main } = await sunScene();
    updateCascades(state, main);
    const n = cascadeCount(state);
    const cams = cascadeComboEids(state).slice(0, n);
    const before = cams.map((eid) => [
        Transform.pos.x.get(eid),
        Transform.pos.y.get(eid),
        Transform.pos.z.get(eid),
    ]);

    Transform.pos.set(main, 200, 2, -150, 0);
    state.step(0);
    updateCascades(state, main);
    let moved = false;
    for (let i = 0; i < n; i++) {
        const eid = cams[i];
        if (
            Transform.pos.x.get(eid) !== before[i][0] ||
            Transform.pos.y.get(eid) !== before[i][1] ||
            Transform.pos.z.get(eid) !== before[i][2]
        )
            moved = true;
    }
    expect(moved).toBe(true);
});
