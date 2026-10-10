import { expect, setDefaultTimeout, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    Exposure,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { Plugin, World } from "../../engine";
import {
    IndirectLightInput,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    registerIndirectLightSource,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
    VertexMaterialType,
} from "./index";

setDefaultTimeout(CEILING.gpu);

const AMBIENT = 249.52394;
const indirectSource = tgpu.fn(
    [IndirectLightInput],
    d.vec3f,
)((_input) => {
    "use gpu";
    return d.vec3f(AMBIENT, 0, 0);
});

const IndirectTestPlugin: Plugin = {
    name: "IndirectLightTestSource",
    dependencies: [StandardRenderingPlugin],
    initialize(world) {
        registerIndirectLightSource(world, "test-red", indirectSource);
    },
};

const config = { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin] };
const sourceConfig = {
    defaults: false,
    plugins: [StandardRenderingPlugin, MeshRenderPlugin, IndirectTestPlugin],
};
const subjects = gpuApps(import.meta.path, [config, sourceConfig]);

async function ambientFrame(world: World): Promise<Uint8ClampedArray> {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, AmbientLight, { brightness: AMBIENT });
    world.add(camera, Exposure, { ev100: 9.7 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 32 });
    const material = world.resource(Materials).add(
        StandardMaterial({
            baseColor: [0.8, 0.4, 0.2, 1],
            metallic: 0,
            perceptualRoughness: 1,
            occlusion: 1,
            diffuseWrap: 1,
        }),
    );
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    world.add(mesh, MeshMaterial, material);
    world.step(0);
    world.step(0);
    return (await captureTexture(world, camera)).rgba;
}

function pixel(frame: Uint8ClampedArray, x: number): number[] {
    return Array.from(frame.subarray((16 * 64 + x) * 4, (16 * 64 + x) * 4 + 4));
}

function expectedPixel(r: number, g: number, b: number): number[] {
    const encode = (linear: number) =>
        Math.round(
            (linear <= 0.0031308 ? linear * 12.92 : 1.055 * linear ** (1 / 2.4) - 0.055) * 255,
        );
    return [encode(r), encode(g), encode(b), 255];
}

test("a registered indirect source adds to material light and receives occlusion", async () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, AmbientLight, { brightness: AMBIENT });
    world.add(camera, Exposure, { ev100: 9.7 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 32 });
    const parameters = StandardMaterial({
        baseColor: [1, 1, 1, 1],
        metallic: 0,
        perceptualRoughness: 1,
        occlusion: 1,
        diffuseWrap: 1,
    });
    const standardMaterial = world.resource(Materials).add(parameters);
    const vertexMaterial = world.resource(VertexMaterialType).add(parameters);
    const add = (x: number, material: typeof standardMaterial) => {
        const mesh = world.create();
        world.add(mesh, Transform, { translation: [x, 0, 0, 0] });
        world.add(mesh, MeshInstance);
        world.add(mesh, MeshMaterial, material);
    };
    add(-0.7, standardMaterial);
    add(0.7, vertexMaterial);

    world.step(0);
    world.step(0);
    const fullFrame = (await captureTexture(world, camera)).rgba;
    world.resource(Materials).update(standardMaterial, { occlusion: 0.5 });
    world.resource(VertexMaterialType).update(vertexMaterial, { occlusion: 0.5 });
    world.step(0);
    const halfFrame = (await captureTexture(world, camera)).rgba;

    expect(pixel(fullFrame, 28)).toEqual(expectedPixel(0.5, 0.25, 0.25));
    expect(pixel(fullFrame, 36)).toEqual(expectedPixel(0.5, 0.25, 0.25));
    expect(pixel(halfFrame, 28)).toEqual(expectedPixel(0.25, 0.125, 0.125));
    expect(pixel(halfFrame, 36)).toEqual(expectedPixel(0.25, 0.125, 0.125));
});

test("AmbientLight-only material frame matches main", async () => {
    const { world } = subjects()[0];
    const frame = await ambientFrame(world);
    expect(pixel(frame, 32)).toEqual([123, 89, 63, 255]);
    expect(frame).toHaveLength(64 * 32 * 4);
});
