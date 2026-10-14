import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    compileGpuFile,
    createTestGpuApp,
    disposeGpuApps,
    gpuRequirements,
} from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    PointLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { PointShadows, updatePointShadows } from "./shadows";

setDefaultTimeout(TIMEOUT.node);

const subjects = compileGpuFile(import.meta.path, async () => {
    const owner = await createTestGpuApp({
        defaults: false,
        plugins: [
            {
                name: "PointAtlasTestDevice",
                gpu: gpuRequirements([StandardRenderingPlugin, MeshRenderPlugin]),
            },
        ],
    });
    const device = rawDevice(owner.world.gpu.device);
    return { owner, device };
});

// an app whose world sets PointShadows.atlas in setup, on the shared device
function build(atlas: number) {
    return createTestGpuApp({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin],
        device: subjects().device,
        setup: (world) => {
            world.resource(PointShadows).atlas = atlas;
        },
    });
}

// a cube between a shadowed point light and a wall, seen by a camera with or without a depth prepass
function scene(world: World, prepass: boolean): number {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 6, 0] });
    world.add(camera, Camera);
    world.add(camera, AmbientLight, { brightness: 0 });
    world.add(camera, StandardRenderer);
    if (prepass) world.add(camera, DepthPrepass);
    attachTexture(world, camera, { width: 32, height: 32 });
    const material = world.resource(Materials).add(StandardMaterial());
    const wall = world.create();
    world.add(wall, Transform, { translation: [0, 0, -1, 0], scale: [20, 20, 0.2, 0] });
    world.add(wall, MeshInstance);
    world.add(wall, MeshMaterial, material);
    const cube = world.create();
    world.add(cube, Transform, { translation: [0, 0, 1, 0] });
    world.add(cube, MeshInstance);
    world.add(cube, MeshMaterial, material);

    const light = world.create();
    world.add(light, Transform, { translation: [0.6, 0.6, 3, 0] });
    world.add(light, PointLight, { intensity: 788064.8, range: 20 });
    world.storage(PointLight).shadowMapsEnabled.set(light, 1);
    return camera;
}

for (const [atlas0, atlas1] of [
    [2048, 1024],
    [1024, 2048],
    [2048, 256],
]) {
    for (const prepass of [false, true]) {
        test(`changing PointShadows.atlas ${atlas0} to ${atlas1} after a point light first casts renders as an app built with ${atlas1}${prepass ? ", under a depth prepass" : ""}`, async () => {
            const app = await build(atlas0);
            const fresh = await build(atlas1);
            try {
                const camera = scene(app.world, prepass);
                app.world.step(0);
                app.world.step(0);
                app.world.resource(PointShadows).atlas = atlas1;
                app.world.gpu.device.pushErrorScope("validation");
                app.world.step(0);
                app.world.step(0);
                expect((await app.world.gpu.device.popErrorScope())?.message ?? null).toBeNull();
                const reference = scene(fresh.world, prepass);
                fresh.world.step(0);
                fresh.world.step(0);
                expect((await captureTexture(app.world, camera)).rgba).toEqual(
                    (await captureTexture(fresh.world, reference)).rgba,
                );
            } finally {
                await disposeGpuApps([app, fresh]);
            }
        });
    }
}

test("an aborted point-shadow draw does not promote its unsubmitted caster to an incumbent", async () => {
    expectWarning(/2 shadowed point lights exceed the 1 caster cap/);
    expectWarning(/2 shadowed point lights exceed the 1 caster cap/);
    const reference = await build(256);
    const aborted = await build(256);
    try {
        const makeIncumbents = (world: World) => {
            const camera = world.create();
            world.add(camera, Transform);
            world.add(camera, Camera);
            world.add(camera, StandardRenderer);
            attachTexture(world, camera, { width: 16, height: 16 });
            const lights = [10, 10.54].map((x) => {
                const eid = world.create();
                world.add(eid, Transform, { translation: [x, 0, 0, 0] });
                world.add(eid, PointLight, { intensity: 100, range: 10 });
                world.storage(PointLight).shadowMapsEnabled.set(eid, 1);
                return eid;
            });
            Object.assign(world.resource(PointShadows), {
                atlas: 256,
                casters: 1,
                hysteresis: 0.25,
            });
            return { camera, lights };
        };
        const referenceScene = makeIncumbents(reference.world);
        const abortedScene = makeIncumbents(aborted.world);
        const selected = (world: World, camera: number) => {
            const frames: Parameters<typeof updatePointShadows>[2] = [];
            const count = updatePointShadows(world, camera, frames);
            return frames.slice(0, count).map((frame) => frame.light);
        };
        for (const app of [reference, aborted]) {
            app.world.step(0);
            app.world.step(0);
        }
        const incumbent = referenceScene.lights[0]!;
        const challenger = abortedScene.lights[1]!;
        expect(selected(reference.world, referenceScene.camera)).toEqual([incumbent]);
        expect(selected(aborted.world, abortedScene.camera)).toEqual([incumbent]);

        reference.world.step(0);
        const expected = selected(reference.world, referenceScene.camera);
        aborted.world.storage(PointLight).intensity.set(challenger, 500);
        let fail = true;
        aborted.world.addSystem({
            group: "draw",
            last: true,
            update() {
                if (!fail) return;
                fail = false;
                throw new Error("abort point-shadow draw");
            },
        });
        expect(() => aborted.world.step(0)).toThrow("abort point-shadow draw");
        aborted.world.storage(PointLight).intensity.set(challenger, 100);
        aborted.world.step(0);
        expect(selected(aborted.world, abortedScene.camera)).toEqual(expected);
    } finally {
        await disposeGpuApps([reference, aborted]);
    }
});

afterAll(() => disposeGpuApps([subjects().owner]));
