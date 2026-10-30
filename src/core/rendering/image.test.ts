import { expect, test } from "bun:test";
import { World } from "../../engine";
import { DataImages, ImageAddressMode, ImageFilterMode, Images, registerImage } from "./image";

test("image registration accepts a source without a name", () => {
    const world = new World();
    // @ts-expect-error Images are identified by source, not an alias.
    registerImage(world, "image.png", "alias");
    world.dispose();
});

test("one source registered in both roles has independent color and linear array ids", () => {
    const world = new World();
    registerImage(world, "color-only.png");
    const color = registerImage(world, "shared.png");
    const data = registerImage(world, "shared.png", { isSrgb: false });
    expect(color).toBe(1);
    expect(data).toBe(0);
    expect(registerImage(world, "shared.png", { isSrgb: true })).toBe(color);
    expect(registerImage(world, "shared.png", { isSrgb: false })).toBe(data);
    expect(world.resource(Images).size).toBe(2);
    expect(world.resource(DataImages).size).toBe(1);
    world.dispose();
});

test("one image URL returns one id and two URLs return two", () => {
    const world = new World();
    const first = registerImage(world, "first.png");
    expect(registerImage(world, "first.png")).toBe(first);
    expect(registerImage(world, "second.png")).not.toBe(first);
    world.dispose();
});

test("URL identity includes the effective sampler and omitted fields equal defaults", () => {
    const world = new World();
    const first = registerImage(world, "image.png");
    expect(
        registerImage(world, "image.png", {
            sampler: {
                addressModeU: ImageAddressMode.ClampToEdge,
                addressModeV: ImageAddressMode.ClampToEdge,
                magFilter: ImageFilterMode.Linear,
                minFilter: ImageFilterMode.Linear,
                mipmapFilter: ImageFilterMode.Linear,
            },
        }),
    ).toBe(first);
    for (const sampler of [
        { addressModeU: ImageAddressMode.Repeat },
        { addressModeV: ImageAddressMode.MirrorRepeat },
        { magFilter: ImageFilterMode.Nearest },
        { minFilter: ImageFilterMode.Nearest },
        { mipmapFilter: ImageFilterMode.Nearest },
        { minFilter: ImageFilterMode.Nearest, mipmapFilter: ImageFilterMode.Nearest },
    ]) {
        const id = registerImage(world, "image.png", { sampler });
        expect(id).not.toBe(first);
        expect(registerImage(world, "image.png", { sampler })).toBe(id);
    }
    world.dispose();
});

test("each Blob gets a new id distinct from the URL image0 in either order", () => {
    for (const urlFirst of [false, true]) {
        const world = new World();
        const blob = new Blob();
        const first = registerImage(world, urlFirst ? "image0" : blob);
        const second = registerImage(world, urlFirst ? blob : "image0");
        expect(second).not.toBe(first);
        expect(registerImage(world, blob)).not.toBe(second);
        expect(registerImage(world, "image0")).toBe(urlFirst ? first : second);
        world.dispose();
    }
});
