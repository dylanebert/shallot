import { expect, test } from "bun:test";
import { createApp } from "../../engine/app";
import { AlphaMode, Materials } from "./material";
import { StandardMaterial } from "./material-data";

test("standard material rows are authored without a renderer or device", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const handle = materials.add(
            StandardMaterial({ baseColor: [0.2, 0.4, 0.6, 1], perceptualRoughness: 0.3 }),
        );
        materials.update(handle, { metallic: 0.7 });
        expect(handle).toEqual({ type: 0, material: 1, alphaMode: 0, alphaCutoff: 0.5 });
        const glass = materials.add(StandardMaterial({ baseColor: [1, 1, 1, 0.4] }), {
            alphaMode: AlphaMode.Blend,
        });
        materials.update(glass, {}, { alphaMode: AlphaMode.Mask(0.3) });
        expect(glass).toEqual({ type: 0, material: 2, alphaMode: 1, alphaCutoff: 0.3 });
    } finally {
        app.dispose();
    }
});

test("a material row is removed once, and reuse leaves its removed handle unchanged", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const red = materials.add(StandardMaterial({ baseColor: [1, 0, 0, 1] }));
        materials.remove(red);
        expect(() => materials.remove(red)).toThrow(RangeError);
        const green = materials.add(StandardMaterial({ baseColor: [0, 1, 0, 1] }));
        const blue = materials.add(StandardMaterial({ baseColor: [0, 0, 1, 1] }));
        expect(green.material).not.toBe(blue.material);

        materials.remove(blue);
        const glass = materials.add(StandardMaterial({ baseColor: [1, 1, 1, 0.4] }), {
            alphaMode: AlphaMode.Blend,
        });
        expect(glass.material).toBe(blue.material);
        expect(blue.alphaMode).toBe(0);
        expect(() => materials.remove(blue)).toThrow(RangeError);
    } finally {
        app.dispose();
    }
});

test("update refuses a removed handle, and a reused row's next owner keeps its values and alpha", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const red = materials.add(StandardMaterial({ baseColor: [1, 0, 0, 1] }));
        materials.remove(red);
        expect(() => materials.update(red, { baseColor: [0, 0, 1, 1] })).toThrow(
            "is not held by this handle",
        );
        const green = materials.add(StandardMaterial({ baseColor: [0, 1, 0, 1] }));
        expect(green.material).toBe(red.material);
        expect(() => materials.update(red, { baseColor: [0, 0, 1, 1] })).toThrow(
            "is not held by this handle",
        );
        expect(() => materials.update(red, {}, { alphaMode: AlphaMode.Blend })).toThrow(
            "is not held by this handle",
        );
        expect(Array.from(materials.get(green)!.baseColor)).toEqual([0, 1, 0, 1]);
        expect(materials.alpha(green.material).alphaMode).toBe(0);
        expect(green.alphaMode).toBe(0);
    } finally {
        app.dispose();
    }
});

test("get returns a row's authored values as a record add accepts, before any GPU table exists", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const handle = materials.add(
            StandardMaterial({
                baseColor: [0.25, 0.5, 0.75, 1],
                emissive: [2, 3, 4],
                metallic: 0.5,
            }),
        );
        const values = materials.get(handle)!;
        expect(Array.from(values.baseColor)).toEqual([0.25, 0.5, 0.75, 1]);
        expect(Array.from(values.emissive)).toEqual([2, 3, 4]);
        expect(values.metallic).toBe(0.5);
        expect(values.baseColorTexture).toBe(-1);
        const copy = materials.add(values);
        expect(Array.from(materials.get(copy)!.emissive)).toEqual([2, 3, 4]);
        expect(Array.from(materials.get(0)!.baseColor)).toEqual([1, 1, 1, 1]);
    } finally {
        app.dispose();
    }
});

test("get returns a fresh record that writing does not reach", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const handle = materials.add(StandardMaterial({ metallic: 0.25 }));
        const first = materials.get(handle)!;
        first.metallic = 1;
        first.baseColor[0] = 0;
        expect(materials.get(handle)!.metallic).toBe(0.25);
        expect(materials.get(handle)!.baseColor[0]).toBe(1);
        expect(materials.get(handle)).not.toBe(materials.get(handle));
    } finally {
        app.dispose();
    }
});

test("get reads update's merged row", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const handle = materials.add(StandardMaterial({ baseColor: [1, 0, 0, 1], metallic: 0.25 }));
        materials.update(handle, { metallic: 0.75 });
        const values = materials.get(handle)!;
        expect(values.metallic).toBe(0.75);
        expect(Array.from(values.baseColor)).toEqual([1, 0, 0, 1]);
    } finally {
        app.dispose();
    }
});

test("get is undefined for a removed or reused handle and a row outside the type's rows", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const red = materials.add(StandardMaterial({ baseColor: [1, 0, 0, 1] }));
        materials.remove(red);
        expect(materials.get(red)).toBeUndefined();
        const green = materials.add(StandardMaterial({ baseColor: [0, 1, 0, 1] }));
        expect(green.material).toBe(red.material);
        expect(materials.get(red)).toBeUndefined();
        expect(Array.from(materials.get(green)!.baseColor)).toEqual([0, 1, 0, 1]);
        expect(materials.get(green.material + 1)).toBeUndefined();
        expect(materials.get(-1)).toBeUndefined();
        expect(materials.get(0.5)).toBeUndefined();
    } finally {
        app.dispose();
    }
});

test("get refuses a handle of another material type", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const materials = app.world.resource(Materials);
        const handle = materials.add(StandardMaterial());
        expect(() => materials.get({ ...handle, type: handle.type + 1 })).toThrow(
            "handle belongs to another material type",
        );
    } finally {
        app.dispose();
    }
});
