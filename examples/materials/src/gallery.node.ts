import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp as createApp } from "@dylanebert/shallot/testing/gpu";

setDefaultTimeout(TIMEOUT.node);

import { Materials, MeshMaterial, StandardMaterial } from "@dylanebert/shallot";
import { Gallery, GalleryPlugin, presets } from "./gallery";
import { CheckerMaterial, QuantizedMaterial, WaveMaterial } from "./materials";

test("the gallery's carbon and red colours reach the checkerboard's material row", async () => {
    const app = await createApp({ defaults: false, plugins: [GalleryPlugin] });
    try {
        const world = app.world;
        const value = world.resource(CheckerMaterial).get(world.resource(Gallery).checker!)!;
        expect(Array.from(value.darkColor)).toEqual([0.042, 0.039, 0.035].map(Math.fround));
        expect(Array.from(value.lightColor)).toEqual([0.658, 0.301, 0.117].map(Math.fround));
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
        expect(world.resource(WaveMaterial).get(world.resource(Gallery).wave!)!.time).toBeCloseTo(
            2 / 60,
            6,
        );
    } finally {
        app.dispose();
    }
});

test("the Banded lighting exhibit uses the quantized standard extension", async () => {
    const app = await createApp({ defaults: false, plugins: [GalleryPlugin] });
    try {
        const { world } = app;
        const eid = world.resource(Gallery).spheres.at(-1)!;
        const handle = world.get(eid, MeshMaterial)!;
        expect(handle.type).toBe(world.resource(QuantizedMaterial).add(StandardMaterial()).type);
        expect(
            world.resource(QuantizedMaterial).get(handle.material)!.perceptualRoughness,
        ).toBeCloseTo(0.8);
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
            const value = world.resource(Materials).get(row)!;
            expect(Array.from(value.baseColor)).toEqual(preset.baseColor.map(Math.fround));
            expect(value.metallic).toBe(preset.metallic);
            expect(value.perceptualRoughness).toBeCloseTo(preset.perceptualRoughness, 6);
            expect(world.storage(MeshMaterial).alphaMode.get(eid)).toBe(index === 3 ? 2 : 0);
        } finally {
            app.dispose();
        }
    });
}
