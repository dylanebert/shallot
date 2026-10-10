import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    AlphaMode,
    AlphaPipelineKey,
    Draws,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";
import { AlphaModeCode, alphaModeFields, alphaPipelineKey } from "./material-type";

setDefaultTimeout(CEILING.gpu);
const alphaWorld = {
    defaults: false,
    plugins: [StandardRenderingPlugin, MeshRenderPlugin],
};
const subjects = gpuApps(import.meta.path, [alphaWorld, alphaWorld, alphaWorld]);

function srgbByte(linear: number): number {
    const encoded = linear <= 0.0031308 ? linear * 12.92 : 1.055 * linear ** (1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
}

function expectedPixel(r: number, g: number, b: number): number[] {
    return [srgbByte(r), srgbByte(g), srgbByte(b), 255];
}

function linearSrgb(channel: number): number {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

const CLEAR = [0x2e, 0x2b, 0x28].map(linearSrgb);

test("Premultiplied and Add share the material pipeline key", () => {
    expect(alphaPipelineKey(alphaModeFields(AlphaMode.Premultiplied).alphaMode)).toBe(
        alphaPipelineKey(alphaModeFields(AlphaMode.Add).alphaMode),
    );
});

test("one material type routes opaque, masked, and blended instances by their alpha modes", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.storage(Camera).antialias.set(camera, 0);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 96, height: 96 });

    const materials = world.resource(Materials);
    const rows = [
        materials.add(StandardMaterial({ baseColor: [1, 0, 0, 0.2], unlit: true })),
        materials.add(StandardMaterial({ baseColor: [0, 1, 0, 0.49], unlit: true }), {
            alphaMode: AlphaMode.Mask(0.5),
        }),
        materials.add(StandardMaterial({ baseColor: [0, 0, 1, 0.25], unlit: true }), {
            alphaMode: AlphaMode.Blend,
        }),
    ];
    const entities: number[] = [];
    for (let i = 0; i < rows.length; i++) {
        const eid = world.create();
        world.add(eid, Transform, {
            translation: [(i - 1) * 1.2, 0, 0, 0],
            scale: [0.45, 0.45, 0.45, 0],
        });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, rows[i]);
        entities.push(eid);
    }

    world.step(0);
    world.step(0);
    const rgba = (await captureTexture(world, camera)).rgba;
    const pixel = (x: number) =>
        Array.from(rgba.subarray((48 * 96 + x) * 4, (48 * 96 + x) * 4 + 4));
    expect(pixel(27)).toEqual(expectedPixel(1, 0, 0));
    expect(pixel(48)).toEqual([0x2e, 0x2b, 0x28, 255]);
    const blended = expectedPixel(CLEAR[0] * 0.75, CLEAR[1] * 0.75, CLEAR[2] * 0.75 + 0.25);
    expect(
        Math.max(...pixel(69).map((channel, i) => Math.abs(channel - blended[i]))),
    ).toBeLessThanOrEqual(1);

    materials.update(rows[0], {}, { alphaMode: AlphaMode.Mask(0.3) });
    expect(rows[0].alphaMode).toBe(AlphaModeCode.Mask);
    expect(world.storage(MeshMaterial).alphaMode.get(entities[0])).toBe(AlphaModeCode.Mask);
    world.step(0);
    world.step(0);
    const updated = (await captureTexture(world, camera)).rgba;
    expect(Array.from(updated.subarray((48 * 96 + 27) * 4, (48 * 96 + 27) * 4 + 4))).toEqual([
        0x2e, 0x2b, 0x28, 255,
    ]);
});

test("Premultiplied, Add and Multiply use their independently computed composites", async () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.storage(Camera).antialias.set(camera, 0);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 96, height: 96 });

    const materials = world.resource(Materials);
    const rows = [
        materials.add(StandardMaterial({ baseColor: [0.5, 0, 0, 0.5], unlit: true }), {
            alphaMode: AlphaMode.Premultiplied,
        }),
        materials.add(StandardMaterial({ baseColor: [0, 0.4, 0, 0.5], unlit: true }), {
            alphaMode: AlphaMode.Add,
        }),
        materials.add(StandardMaterial({ baseColor: [0.8, 0.8, 0.8, 0.5], unlit: true }), {
            alphaMode: AlphaMode.Multiply,
        }),
    ];
    for (let i = 0; i < rows.length; i++) {
        const eid = world.create();
        world.add(eid, Transform, {
            translation: [(i - 1) * 1.2, 0, 0, 0],
            scale: [0.45, 0.45, 0.45, 0],
        });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, rows[i]);
    }

    world.step(0);
    world.step(0);
    const rgba = (await captureTexture(world, camera)).rgba;
    const pixel = (x: number) =>
        Array.from(rgba.subarray((48 * 96 + x) * 4, (48 * 96 + x) * 4 + 4));
    const premultiplied = expectedPixel(CLEAR[0] * 0.5 + 0.5, CLEAR[1] * 0.5, CLEAR[2] * 0.5);
    const additive = expectedPixel(CLEAR[0], CLEAR[1] + 0.2, CLEAR[2]);
    const multiply = expectedPixel(CLEAR[0] * 0.9, CLEAR[1] * 0.9, CLEAR[2] * 0.9);
    expect(
        Math.max(...pixel(27).map((channel, i) => Math.abs(channel - premultiplied[i]))),
    ).toBeLessThanOrEqual(1);
    expect(
        Math.max(...pixel(48).map((channel, i) => Math.abs(channel - additive[i]))),
    ).toBeLessThanOrEqual(1);
    expect(
        Math.max(...pixel(69).map((channel, i) => Math.abs(channel - multiply[i]))),
    ).toBeLessThanOrEqual(1);
    expect(
        [...world.resource(Draws)].filter(
            (draw) =>
                draw.materialType === 0 && draw.alphaPipelineKey === AlphaPipelineKey.Premultiplied,
        ),
    ).toHaveLength(1);
});

test("AlphaToCoverage selects fractional sample coverage on a multisampled target", async () => {
    const { world } = subjects()[2];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.storage(Camera).antialias.set(camera, 1);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 96, height: 96 });

    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [1, 0, 0, 0.25], unlit: true }), {
            alphaMode: AlphaMode.AlphaToCoverage,
        });
    const eid = world.create();
    world.add(eid, Transform, { scale: [0.45, 0.45, 0.45, 0] });
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial, material);

    world.step(0);
    world.step(0);
    const rgba = (await captureTexture(world, camera)).rgba;
    const pixel = Array.from(rgba.subarray((48 * 96 + 48) * 4, (48 * 96 + 48) * 4 + 4));
    const expected = expectedPixel(CLEAR[0] * 0.75 + 0.25, CLEAR[1] * 0.75, CLEAR[2] * 0.75);
    expect(
        Math.max(...pixel.map((channel, i) => Math.abs(channel - expected[i]))),
    ).toBeLessThanOrEqual(1);

    world.storage(Camera).antialias.set(camera, 0);
    world.step(0);
    world.step(0);
    const singleSample = (await captureTexture(world, camera)).rgba;
    expect(Array.from(singleSample.subarray((48 * 96 + 48) * 4, (48 * 96 + 48) * 4 + 4))).toEqual([
        0x2e, 0x2b, 0x28, 255,
    ]);
});
