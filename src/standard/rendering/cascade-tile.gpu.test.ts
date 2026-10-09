import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { compileGpuFile, gpuRequirements } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { createApp, type World } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import {
    cascadeCount,
    cascadeMeta,
    cascadeRecvVP,
    cascadeTileRects,
    DirectionalLightShadowMap,
} from "./shadows";

setDefaultTimeout(CEILING.gpu);

// a cube on a floor under a shadowed sun split into `cascades`, seen by a camera with or without a depth prepass
function scene(world: World, cascades: number, prepass = false): { camera: number; sun: number } {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 3, 8, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    if (prepass) world.add(camera, DepthPrepass);
    attachTexture(world, camera, { width: 16, height: 16 });
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
    world.storage(DirectionalLight).numCascades.set(sun, cascades);
    return { camera, sun };
}

// each active cascade's tile side in atlas pixels (the VS reads the atlas side from meta.z)
function tileSides(world: World): number[] {
    const rects = cascadeTileRects(world);
    const side = cascadeMeta(world)[2];
    return Array.from({ length: cascadeCount(world) }, (_, i) => rects[i * 4 + 2] * side);
}

// a fixed world point's sub-texel x position in each cascade's tile, in that tile's own pixels
function subTexel(world: World): number[] {
    const vp = cascadeRecvVP(world);
    const sides = tileSides(world);
    return sides.map((side, i) => {
        const m = vp.subarray(i * 16, i * 16 + 16);
        const x = m[0] * 0.3 + m[8] * -0.7 + m[12];
        const w = m[3] * 0.3 + m[11] * -0.7 + m[15];
        const t = ((x / w) * 0.5 + 0.5) * side;
        return t - Math.floor(t);
    });
}

// steps the camera sideways in sub-texel moves and returns the largest wrapped drift of each cascade's position
function drift(world: World, camera: number): number[] {
    const first = subTexel(world);
    const worst = first.map(() => 0);
    for (let k = 1; k < 12; k++) {
        world.storage(Transform).translation.set(camera, k * 0.0137, 3, 8, 0);
        world.step(0);
        subTexel(world).forEach((f, i) => {
            const d = Math.abs(f - first[i]);
            worst[i] = Math.max(worst[i], Math.min(d, 1 - d));
        });
    }
    return worst;
}

const subjects = compileGpuFile(import.meta.path, async () => {
    const owner = await createApp({
        defaults: false,
        plugins: [
            {
                name: "CascadeTileTestDevice",
                gpu: gpuRequirements([StandardRenderingPlugin, MeshRenderPlugin]),
            },
        ],
    });
    const device = rawDevice(owner.world.gpu.device);
    return { owner, device };
});

for (const cascades of [4, 2]) {
    test(`a 4096 DirectionalLightShadowMap gives each of ${cascades} cascades a 4096-pixel tile that its texel snap holds still`, async () => {
        const { device } = subjects();
        const app = await createApp({
            defaults: false,
            plugins: [StandardRenderingPlugin, MeshRenderPlugin],
            device,
            setup: (world) => {
                world.resource(DirectionalLightShadowMap).size = 4096;
            },
        });
        try {
            const { camera } = scene(app.world, cascades);
            app.world.gpu.device.pushErrorScope("validation");
            app.world.step(0);
            app.world.step(0);
            expect((await app.world.gpu.device.popErrorScope())?.message ?? null).toBeNull();
            expect(tileSides(app.world)).toEqual(Array(cascades).fill(4096));
            for (const d of drift(app.world, camera)) expect(d).toBeLessThan(1e-3);
        } finally {
            app.dispose();
        }
    });
}

// an app whose world sets DirectionalLightShadowMap.size in setup, on the shared device
function build(size: number) {
    return createApp({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin],
        device: subjects().device,
        setup: (world) => {
            world.resource(DirectionalLightShadowMap).size = size;
        },
    });
}

for (const [size0, cascades0, size1, cascades1] of [
    [2048, 1, 2048, 4],
    [2048, 4, 2048, 1],
    [1024, 4, 2048, 4],
]) {
    for (const prepass of [false, true]) {
        test(`changing ${size0}x${cascades0} to ${size1}x${cascades1} after the sun first casts renders as an app built with ${size1}x${cascades1}${prepass ? ", under a depth prepass" : ""}`, async () => {
            const app = await build(size0);
            const fresh = await build(size1);
            try {
                const live = scene(app.world, cascades0, prepass);
                app.world.step(0);
                app.world.step(0);
                app.world.resource(DirectionalLightShadowMap).size = size1;
                app.world.storage(DirectionalLight).numCascades.set(live.sun, cascades1);
                app.world.gpu.device.pushErrorScope("validation");
                app.world.step(0);
                app.world.step(0);
                expect((await app.world.gpu.device.popErrorScope())?.message ?? null).toBeNull();
                const reference = scene(fresh.world, cascades1, prepass);
                fresh.world.step(0);
                fresh.world.step(0);
                expect(tileSides(app.world)).toEqual(tileSides(fresh.world));
                expect((await captureTexture(app.world, live.camera)).rgba).toEqual(
                    (await captureTexture(fresh.world, reference.camera)).rgba,
                );
                for (const d of drift(app.world, live.camera)) expect(d).toBeLessThan(1e-3);
            } finally {
                app.dispose();
                fresh.dispose();
            }
        });
    }
}

afterAll(() => {
    subjects().owner.dispose();
});
