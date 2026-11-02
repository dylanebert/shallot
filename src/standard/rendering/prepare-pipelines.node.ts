import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    NormalPrepass,
    NotShadowCaster,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type World } from "../../engine";
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    preparePipelines,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);

function assets(world: World) {
    const mesh = world.resource(Meshes).add({
        name: "late-streams",
        vertices: new Float32Array([
            -1, -1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0.5, 1, 0, 1, 0,
        ]),
        indices: new Uint32Array([0, 1, 2]),
        color: new Float32Array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
    });
    const material = world.resource(Materials).add(StandardMaterial({ emissive: [1, 1, 1] }), {
        alphaMode: AlphaMode.Blend,
        cullMode: "none",
    });
    return { mesh, material };
}
const warmAssets: ReturnType<typeof assets>[] = [];
let pendingBeforeWarm = false;
const subjects = gpuApps(
    import.meta.path,
    [true, false].map((atWarm) => ({
        defaults: false,
        plugins: [
            MeshRenderPlugin,
            StandardRenderingPlugin,
            {
                name: "PreparationScene",
                dependencies: [MeshRenderPlugin, StandardRenderingPlugin],
                async initialize(world: World) {
                    const camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    attachTexture(world, camera, { width: 64, height: 64 });
                    if (atWarm) {
                        const registered = assets(world);
                        warmAssets.push(registered);
                        await preparePipelines(world);
                        pendingBeforeWarm =
                            world.resource(Meshes).get(registered.mesh)!.pending === true;
                    }
                },
            },
        ],
    })),
);

test("preparation before warm resolves without packing staged meshes; the build owns preparation", () => {
    expect(pendingBeforeWarm).toBe(true);
});

test("awaiting preparation after warm draws new alpha, cull and stream keys exactly as preparation at warm without pipeline warnings", async () => {
    const captures = [];
    for (const [index, { world }] of subjects().entries()) {
        const { mesh, material } = index === 0 ? warmAssets[0]! : assets(world);
        await preparePipelines(world);
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, MeshMaterial, material);
        world.step(0);
        world.step(0);
        const camera = [...world.query([Camera])][0]!;
        captures.push((await captureTexture(world, camera)).rgba);
    }
    expect(captures[0]!.some((value, i) => i % 4 !== 3 && value > 64)).toBe(true);
    expect(captures[1]).toEqual(captures[0]);
});

test("a re-warm failing before material preparation leaves public preparation inactive", async () => {
    const world = subjects()[1]!.world;
    await expect(StandardRenderingPlugin.warm!(world)).rejects.toThrow(
        'duplicate precompile label "standard:prepass-resolve"',
    );
    const registered = assets(world);
    await preparePipelines(world);
    expect(world.resource(Meshes).get(registered.mesh)!.pending).toBe(true);
});

function shadowAssets(world: World) {
    const mesh = world.resource(Meshes).add({
        name: "late-shadow-streams",
        vertices: new Float32Array([
            -0.4, -0.4, 0, 0, 0, 0, 0, 1, 0.4, -0.4, 0, 1, 0, 0, 0, 1, 0, 0.4, 0, 0.5, 1, 0, 0, 1,
        ]),
        indices: new Uint32Array([0, 1, 2]),
        color: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    });
    const material = world.resource(Materials).add(StandardMaterial({ baseColor: [1, 1, 1, 1] }), {
        alphaMode: AlphaMode.Mask(0.5),
        cullMode: "none",
    });
    return { mesh, material };
}
const warmShadowAssets: ReturnType<typeof shadowAssets>[] = [];
const shadowSubjects = gpuApps(
    import.meta.path,
    [true, false].map((atWarm) => ({
        defaults: false,
        plugins: [
            {
                name: "ShadowPreparationScene",
                dependencies: [MeshRenderPlugin, StandardRenderingPlugin],
                initialize(world: World) {
                    const camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    world.add(camera, DepthPrepass);
                    world.add(camera, NormalPrepass);
                    attachTexture(world, camera, { width: 64, height: 64 });
                    const sun = world.create();
                    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.5, 0, -1) });
                    world.add(sun, DirectionalLight, { illuminance: 10000 });
                    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
                    const mesh = world.resource(Meshes).add({
                        name: "receiver",
                        vertices: new Float32Array([
                            -2, -2, 0, 0, 0, 0, 0, 1, 2, -2, 0, 1, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 0,
                            1, -2, 2, 0, 0, 1, 0, 0, 1,
                        ]),
                        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
                    });
                    const receiver = world.create();
                    world.add(receiver, Transform);
                    world.add(receiver, MeshInstance, { mesh });
                    world.add(
                        receiver,
                        MeshMaterial,
                        world
                            .resource(Materials)
                            .add(StandardMaterial({ baseColor: [1, 1, 1, 1] })),
                    );
                    if (atWarm) warmShadowAssets.push(shadowAssets(world));
                },
            },
        ],
    })),
);

test("late masked stream preparation includes shadow and depth/normal prepass variants with the same received shadow as warm", async () => {
    const captures = [];
    for (const [index, { world }] of shadowSubjects().entries()) {
        const { mesh, material } = index === 0 ? warmShadowAssets[0]! : shadowAssets(world);
        await preparePipelines(world);
        const caster = world.create();
        world.add(caster, Transform, { translation: [0, 0, 1, 0] });
        world.add(caster, MeshInstance, { mesh });
        world.add(caster, MeshMaterial, material);
        world.step(0);
        world.step(0);
        const camera = [...world.query([Camera])][0]!;
        const shadowed = (await captureTexture(world, camera)).rgba;
        captures.push(shadowed);
        world.add(caster, NotShadowCaster);
        world.step(0);
        world.step(0);
        const unshadowed = (await captureTexture(world, camera)).rgba;
        let shaded = 0;
        for (let i = 0; i < shadowed.length; i += 4)
            if (unshadowed[i]! > shadowed[i]! + 8) shaded++;
        expect(shaded).toBeGreaterThan(5);
    }
    expect(captures[1]).toEqual(captures[0]);
});
