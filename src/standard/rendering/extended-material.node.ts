import { expect, setDefaultTimeout, test } from "bun:test";
import { lookAtRotation } from "@dylanebert/shallot";
import { Meshes, MeshInstance, sphere } from "@dylanebert/shallot/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    ImageBindings,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import {
    AlphaMode,
    applyPbrLighting,
    extendedMaterial,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    pbrInputFromStandardMaterial,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Transform } from "@dylanebert/shallot/transform";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";

setDefaultTimeout(TIMEOUT.node);
const Composed = extendedMaterial(Materials, {
    name: "ComposedStandard",
    fragment: (ctx) => {
        "use gpu";
        return applyPbrLighting(pbrInputFromStandardMaterial(ctx), ctx);
    },
});
const Rough = extendedMaterial(Materials, {
    name: "RoughStandard",
    fragment: (ctx) => {
        "use gpu";
        const input = pbrInputFromStandardMaterial(ctx);
        input.perceptualRoughness = 0.8;
        return applyPbrLighting(input, ctx);
    },
});
const Banded = extendedMaterial(Materials, {
    name: "BandedStandard",
    fragment: (ctx) => {
        "use gpu";
        const color = applyPbrLighting(pbrInputFromStandardMaterial(ctx), ctx);
        const value = std.min(std.floor(color.r * 4), 3) / 3;
        return d.vec4f(d.vec3f(value), color.a);
    },
});
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            MaterialPlugin(Composed),
            MaterialPlugin(Rough),
            MaterialPlugin(Banded),
        ],
    },
]);

test("extended standard preserves alpha and shading with prepass and sun shadows, permits roughness edits and bands lit gradients", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, DepthPrepass);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 64 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance);
    const receiver = world.create();
    world.add(receiver, Transform, { translation: [0, 0, -1.5, 0], scale: [3, 3, 0.1, 0] });
    world.add(receiver, MeshInstance);
    world.add(receiver, MeshMaterial, world.resource(Materials).add(StandardMaterial()));
    const image = world.gpu.device.createTexture({
        size: [2, 1, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(image);
    world.gpu.device.queue.writeTexture(
        { texture: image },
        new Uint8Array([255, 128, 64, 64, 64, 128, 255, 192]),
        { bytesPerRow: 8 },
        [2, 1, 1],
    );
    world.resource(ImageBindings).images = image;
    const frame = async (handle: {
        type: number;
        material: number;
        alphaMode: number;
        alphaCutoff: number;
    }) => {
        world.remove(eid, MeshMaterial);
        world.add(eid, MeshMaterial, handle);
        world.step(0);
        world.step(0);
        return (await captureTexture(world, camera)).rgba;
    };
    for (const values of [
        StandardMaterial({ baseColorTexture: 0 }),
        StandardMaterial(),
        StandardMaterial({ metallic: 0.9, perceptualRoughness: 0.2 }),
        StandardMaterial({ emissive: d.vec3f(0.2, 0.1, 0) }),
        StandardMaterial({ unlit: true }),
    ]) {
        for (const alphaMode of [
            AlphaMode.Opaque,
            AlphaMode.Mask(),
            AlphaMode.AlphaToCoverage,
            AlphaMode.Blend,
        ]) {
            const base = world.resource(Materials).add(values, { alphaMode });
            const extension = world.resource(Composed).add(values, { alphaMode });
            expect(await frame(extension)).toEqual(await frame(base));
        }
    }
    const shadowed = await frame(world.resource(Composed).add(StandardMaterial()));
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 0);
    const unshadowed = await frame(world.resource(Composed).add(StandardMaterial()));
    expect(shadowed).not.toEqual(unshadowed);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const reference = world
        .resource(Materials)
        .add(StandardMaterial({ metallic: 0.8, perceptualRoughness: 0.8 }));
    const modified = world
        .resource(Rough)
        .add(StandardMaterial({ metallic: 0.8, perceptualRoughness: 0.1 }));
    expect(await frame(modified)).toEqual(await frame(reference));
    const glossy = world
        .resource(Materials)
        .add(StandardMaterial({ metallic: 0.8, perceptualRoughness: 0.1 }));
    expect(await frame(glossy)).not.toEqual(await frame(reference));

    world.remove(eid, MeshInstance);
    world.add(eid, MeshInstance, {
        mesh: world.resource(Meshes).add({ name: "gradient sphere", ...sphere(1, 64, 32) }),
    });
    world.storage(Camera).antialias.set(camera, 0);
    const white = StandardMaterial({ baseColor: d.vec4f(1), perceptualRoughness: 1 });
    const baseline = await frame(world.resource(Materials).add(white));
    const quantized = await frame(world.resource(Banded).add(white));
    const colors = new Set<number>();
    const gradient = new Set<number>();
    for (let y = 24; y < 40; y++)
        for (let x = 24; x < 40; x++) {
            const offset = (y * 64 + x) * 4;
            colors.add(quantized[offset]!);
            gradient.add(baseline[offset]!);
        }
    expect(gradient.size).toBeGreaterThan(4);
    expect(colors.size).toBeGreaterThan(1);
    expect(colors.size).toBeLessThanOrEqual(4);
});
