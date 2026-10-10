import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import {
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
    VertexMaterialType,
} from "./index";
import type { MaterialHandle } from "./material-type";

setDefaultTimeout(CEILING.gpu);

// With no sun or point lights, the lit diffuse pixel is baseColor * this white ambient intensity.
// Tonemapping.None still applies linear-to-sRGB presentation encoding in expectedPixel.
const AMBIENT = 0.25;
const LIGHTED = AMBIENT;
const config = { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin] };
const subjects = gpuApps(import.meta.path, [config, config]);

function srgbByte(linear: number): number {
    const encoded = linear <= 0.0031308 ? linear * 12.92 : 1.055 * linear ** (1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
}

function expectedPixel(r: number, g: number, b: number): number[] {
    return [srgbByte(r), srgbByte(g), srgbByte(b), 255];
}

function makeScene(world: World) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 32 });

    world.add(world.create(), AmbientLight, { intensity: AMBIENT });
    const add = (x: number, material: MaterialHandle) => {
        const eid = world.create();
        world.add(eid, Transform, { translation: [x, 0, 0, 0], scale: [0.5, 0.5, 0.5, 0] });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, material);
    };
    const capture = async () => {
        world.step(0);
        world.step(0);
        const rgba = (await captureTexture(world, camera)).rgba;
        return (x: number) => Array.from(rgba.subarray((16 * 64 + x) * 4, (16 * 64 + x) * 4 + 4));
    };
    return { add, capture };
}

test("StandardMaterial unlit preserves base colour under lighting that dims a lit material", async () => {
    const { world } = subjects()[0];
    const { add, capture } = makeScene(world);
    const baseBlue = StandardMaterial({ baseColor: [0, 0, 1, 1], unlit: true });
    const litBlue = StandardMaterial({ baseColor: [0, 0, 1, 1] });
    add(-0.7, world.resource(Materials).add(baseBlue));
    add(0.7, world.resource(Materials).add(litBlue));

    const pixel = await capture();
    expect(pixel(28)).toEqual(expectedPixel(0, 0, 1));
    expect(pixel(36)).toEqual(expectedPixel(0, 0, LIGHTED));
    expect(pixel(36)).not.toEqual(pixel(28));
});

test("VertexMaterialType applies the independently computed ambient light to its base colour", async () => {
    const { world } = subjects()[1];
    const { add, capture } = makeScene(world);
    const material = world.resource(VertexMaterialType).add(
        StandardMaterial({
            baseColor: [0, 1, 0, 1],
            metallic: 0,
            perceptualRoughness: 1,
            occlusion: 1,
            diffuseWrap: 1,
        }),
    );
    add(-0.7, material);

    const litPixel = (await capture())(28);
    const expectedLit = expectedPixel(0, LIGHTED, 0);
    const baseColour = expectedPixel(0, 1, 0);
    expect(litPixel).toEqual(expectedLit);
    expect(litPixel).not.toEqual(baseColour);
});
