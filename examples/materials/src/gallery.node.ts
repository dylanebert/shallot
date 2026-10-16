import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp as createApp } from "../tests/gpu-app";

setDefaultTimeout(TIMEOUT.node);

import { Materials, MeshMaterial } from "@dylanebert/shallot";
import { StandardMaterialInput } from "@dylanebert/shallot/standard/rendering";
import { readFromArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Gallery, GalleryPlugin, presets } from "./gallery";
import { AnimationParameters, CheckerMaterial, CheckerParameters, WaveMaterial } from "./materials";

test("the gallery's dark and orange colours reach the checkerboard's material row", async () => {
    const app = await createApp({ defaults: false, plugins: [GalleryPlugin] });
    try {
        const world = app.world;
        const row = world.resource(Gallery).checker!.material;
        const bytes = world.resource(CheckerMaterial).table.bytes;
        const offset = row * d.sizeOf(CheckerParameters);
        const value = readFromArrayBuffer(
            bytes.slice(offset, offset + d.sizeOf(CheckerParameters)).buffer,
            CheckerParameters,
        );
        expect(Array.from(value.darkColor)).toEqual([0.06, 0.045, 0.03].map(Math.fround));
        expect(Array.from(value.lightColor)).toEqual([0.95, 0.36, 0.08].map(Math.fround));
    } finally {
        app.dispose();
    }
});

test("the wave's own time parameter advances with the clock", async () => {
    const app = await createApp({ defaults: false, plugins: [GalleryPlugin] });
    try {
        const world = app.world;
        world.step(1 / 60);
        world.step(1 / 60);
        const row = world.resource(Gallery).wave!.material;
        const bytes = world.resource(WaveMaterial).table.bytes;
        const offset = row * d.sizeOf(AnimationParameters);
        expect(
            readFromArrayBuffer(
                bytes.slice(offset, offset + d.sizeOf(AnimationParameters)).buffer,
                AnimationParameters,
            ).time,
        ).toBeCloseTo(2 / 60, 6);
    } finally {
        app.dispose();
    }
});

for (const [index, preset] of presets.entries()) {
    test(`${preset.name}'s values reach its sphere's material row`, async () => {
        const app = await createApp({ defaults: false, plugins: [GalleryPlugin] });
        try {
            const world = app.world;
            const eid = world.resource(Gallery).spheres[index]!;
            const row = world.storage(MeshMaterial).material.get(eid);
            const bytes = world.resource(Materials).table.bytes;
            const offset = row * d.sizeOf(StandardMaterialInput);
            const value = readFromArrayBuffer(
                bytes.slice(offset, offset + d.sizeOf(StandardMaterialInput)).buffer,
                StandardMaterialInput,
            );
            expect(Array.from(value.baseColor)).toEqual(preset.baseColor.map(Math.fround));
            expect(value.metallic).toBe(preset.metallic);
            expect(value.perceptualRoughness).toBeCloseTo(preset.perceptualRoughness, 6);
            expect(world.storage(MeshMaterial).alphaMode.get(eid)).toBe(index === 3 ? 2 : 0);
        } finally {
            app.dispose();
        }
    });
}
