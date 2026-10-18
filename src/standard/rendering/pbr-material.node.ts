import { expect, setDefaultTimeout, test } from "bun:test";
import { lookAtRotation } from "@dylanebert/shallot";
import { MeshInstance } from "@dylanebert/shallot/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
} from "@dylanebert/shallot/rendering";
import {
    applyPbrLighting,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    pbrInputFromMaterial,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Transform } from "@dylanebert/shallot/transform";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";

setDefaultTimeout(TIMEOUT.node);
const Parameters = d.struct({
    tint: d.vec4f,
    metal: d.f32,
    rough: d.f32,
    glow: d.vec3f,
    ao: d.f32,
});
const Custom = materialType({
    name: "IndependentPbr",
    parameters: Parameters,
    depthPass: { prepass: true, shadows: true },
    fragment: (ctx) => {
        "use gpu";
        const input = pbrInputFromMaterial(ctx);
        input.baseColor = d.vec4f(ctx.parameters.tint);
        input.metallic = ctx.parameters.metal;
        input.perceptualRoughness = ctx.parameters.rough;
        input.emissive = d.vec3f(ctx.parameters.glow);
        input.occlusion = ctx.parameters.ao;
        return applyPbrLighting(input);
    },
    defaults: { tint: d.vec4f(1), metal: 0, rough: 0.5, glow: d.vec3f(0), ao: 1 },
});
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(Custom)],
    },
]);

test("a material with independent parameters opts into standard PBR lighting", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 64, height: 64 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
    const surface = world.create();
    world.add(surface, Transform);
    world.add(surface, MeshInstance);
    const frame = async (handle: {
        type: number;
        material: number;
        alphaMode: number;
        alphaCutoff: number;
    }) => {
        world.remove(surface, MeshMaterial);
        world.add(surface, MeshMaterial, handle);
        world.step(0);
        world.step(0);
        return (await captureTexture(world, camera)).rgba;
    };
    for (const rough of [0.2, 0.8]) {
        const tint = d.vec4f(0.8, 0.3, 0.1, 1);
        const glow = d.vec3f(0.02, 0.01, 0);
        const base = world.resource(Materials).add(
            StandardMaterial({
                baseColor: tint,
                metallic: 0.7,
                perceptualRoughness: rough,
                emissive: glow,
                occlusion: 0.6,
            }),
        );
        const custom = world.resource(Custom).add({ tint, metal: 0.7, rough, glow, ao: 0.6 });
        const reference = await frame(base);
        expect(reference.some((v, i) => i % 4 !== 3 && v > 0)).toBe(true);
        expect(await frame(custom)).toEqual(reference);
    }
});
