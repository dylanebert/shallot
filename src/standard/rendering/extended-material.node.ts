import { expect, setDefaultTimeout, test } from "bun:test";
import { lookAtRotation, type World } from "@dylanebert/shallot";
import { Fog, FogPlugin } from "@dylanebert/shallot/extras";
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
    type MaterialHandle,
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
import { gpuScene } from "../../../scripts/gpu.fixture";

setDefaultTimeout(TIMEOUT.node);
const Composed = extendedMaterial(Materials, {
    name: "ComposedStandard",
    fragment: (ctx) => {
        "use gpu";
        return applyPbrLighting(pbrInputFromStandardMaterial(ctx));
    },
});
const Rough = extendedMaterial(Materials, {
    name: "RoughStandard",
    fragment: (ctx) => {
        "use gpu";
        const input = pbrInputFromStandardMaterial(ctx);
        input.perceptualRoughness = 0.8;
        return applyPbrLighting(input);
    },
});
const Banded = extendedMaterial(Materials, {
    name: "BandedStandard",
    fragment: (ctx) => {
        "use gpu";
        const color = applyPbrLighting(pbrInputFromStandardMaterial(ctx));
        const value = std.min(std.floor(color.r * 4), 3) / 3;
        return d.vec4f(d.vec3f(value), color.a);
    },
});
const rendering = [StandardRenderingPlugin, MeshRenderPlugin];
function litScene(world: World, antialias: number) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { antialias });
    world.add(camera, StandardRenderer);
    world.add(camera, DepthPrepass);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 64 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
    const receiver = world.create();
    world.add(receiver, Transform, { translation: [0, 0, -1.5, 0], scale: [3, 3, 0.1, 0] });
    world.add(receiver, MeshInstance);
    world.add(receiver, MeshMaterial, world.resource(Materials).add(StandardMaterial()));
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance);
    return { camera, sun, eid };
}
async function frame(world: World, camera: number, eid: number, handle: MaterialHandle) {
    world.remove(eid, MeshMaterial);
    world.add(eid, MeshMaterial, handle);
    world.step(0);
    world.step(0);
    return (await captureTexture(world, camera)).rgba;
}
const parity = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [...rendering, MaterialPlugin(Composed), MaterialPlugin(Rough)],
    },
    (world) => {
        const scene = litScene(world, 1);
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
        const cases = [
            StandardMaterial({ baseColorTexture: 0 }),
            StandardMaterial(),
            StandardMaterial({ metallic: 0.9, perceptualRoughness: 0.2 }),
            StandardMaterial({ emissive: d.vec3f(0.2, 0.1, 0) }),
            StandardMaterial({ unlit: true }),
        ].flatMap((values) =>
            [AlphaMode.Opaque, AlphaMode.Mask(), AlphaMode.AlphaToCoverage, AlphaMode.Blend].map(
                (alphaMode) =>
                    [
                        world.resource(Materials).add(values, { alphaMode }),
                        world.resource(Composed).add(values, { alphaMode }),
                    ] as const,
            ),
        );
        const composed = world.resource(Composed).add(StandardMaterial());
        const reference = world
            .resource(Materials)
            .add(StandardMaterial({ metallic: 0.8, perceptualRoughness: 0.8 }));
        const modified = world
            .resource(Rough)
            .add(StandardMaterial({ metallic: 0.8, perceptualRoughness: 0.1 }));
        const glossy = world
            .resource(Materials)
            .add(StandardMaterial({ metallic: 0.8, perceptualRoughness: 0.1 }));
        return { ...scene, image, cases, composed, reference, modified, glossy };
    },
);
const gradient = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [...rendering, MaterialPlugin(Banded)],
    },
    (world) => {
        const scene = litScene(world, 0);
        world
            .storage(MeshInstance)
            .mesh.set(
                scene.eid,
                world.resource(Meshes).add({ name: "gradient sphere", ...sphere(1, 64, 32) }),
            );
        const white = StandardMaterial({ baseColor: d.vec4f(1), perceptualRoughness: 1 });
        return {
            ...scene,
            baseline: world.resource(Materials).add(white),
            quantized: world.resource(Banded).add(white),
        };
    },
);

test("extended standard preserves alpha and shading with prepass and sun shadows, permits roughness edits and bands lit gradients", async () => {
    const {
        world,
        scene: { camera, sun, eid, image, cases, composed, reference, modified, glossy },
    } = parity();
    world.resource(ImageBindings).images = image;
    for (const [base, extension] of cases)
        expect(await frame(world, camera, eid, extension)).toEqual(
            await frame(world, camera, eid, base),
        );
    const shadowed = await frame(world, camera, eid, composed);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 0);
    const unshadowed = await frame(world, camera, eid, composed);
    expect(shadowed).not.toEqual(unshadowed);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    expect(await frame(world, camera, eid, modified)).toEqual(
        await frame(world, camera, eid, reference),
    );
    expect(await frame(world, camera, eid, glossy)).not.toEqual(
        await frame(world, camera, eid, reference),
    );

    const g = gradient();
    const baseline = await frame(g.world, g.scene.camera, g.scene.eid, g.scene.baseline);
    const quantized = await frame(g.world, g.scene.camera, g.scene.eid, g.scene.quantized);
    const colors = new Set<number>();
    const values = new Set<number>();
    for (let y = 24; y < 40; y++)
        for (let x = 24; x < 40; x++) {
            const offset = (y * 64 + x) * 4;
            colors.add(quantized[offset]!);
            values.add(baseline[offset]!);
        }
    expect(values.size).toBeGreaterThan(4);
    expect(colors.size).toBeGreaterThan(1);
    expect(colors.size).toBeLessThanOrEqual(4);
});

const fogScene = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [...rendering, MaterialPlugin(Banded), FogPlugin],
    },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        attachTexture(world, camera, { width: 32, height: 32 });
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 0, 0, -1) });
        world.add(sun, DirectionalLight);
        const surface = world.create();
        world.add(surface, Transform);
        world.add(surface, MeshInstance);
        world.add(
            surface,
            MeshMaterial,
            world.resource(Banded).add(StandardMaterial({ perceptualRoughness: 1 })),
        );
        const fog = world.create();
        world.add(fog, Fog, {
            color: 0,
            density: 0,
            absorption: 1,
            scattering: 0,
            viewDistance: 20,
        });
        return { camera, surface, fog };
    },
);
test("fog varies smoothly with surface depth inside one extended-material lighting band", async () => {
    const {
        world,
        scene: { camera, surface, fog },
    } = fogScene();
    const sampleDepths = async () => {
        const samples: number[] = [];
        for (let depth = 0; depth < 5; depth++) {
            world.storage(Transform).translation.set(surface, 0, 0, -depth, 0);
            world.step(1 / 60);
            world.step(1 / 60);
            const shot = await captureTexture(world, camera);
            samples.push(shot.rgba[(16 * 32 + 16) * 4]!);
        }
        return samples;
    };
    const unfogged = await sampleDepths();
    expect(new Set(unfogged).size).toBe(1);
    expect(unfogged[0]).toBeGreaterThan(0);
    world.storage(Fog).density.set(fog, 0.02);
    const fogged = await sampleDepths();
    expect(new Set(fogged).size).toBe(5);
    for (let i = 1; i < fogged.length; i++) {
        expect(fogged[i]).toBeLessThan(fogged[i - 1]!);
        expect(fogged[i - 1]! - fogged[i]!).toBeLessThan(8);
    }
});
