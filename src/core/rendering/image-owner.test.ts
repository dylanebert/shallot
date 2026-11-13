import { expect, test } from "bun:test";
import { World } from "../../engine";
import { DataImages, Images, registerImage, releaseImage, replaceImage } from "./image";

test("generated image replacements keep one registry slot and release its source", () => {
    const world = new World();
    const id = registerImage(world, new Blob(["first"]));
    for (let n = 0; n < 100; n++) replaceImage(world, id, new Blob([`${n}`]));
    const images = world.resource(Images);
    expect(images.size).toBe(1);
    expect(images.snapshot().names.length).toBe(1);
    releaseImage(world, id);
    expect(images.size).toBe(0);
    releaseImage(world, id);
    expect(() => replaceImage(world, id, new Blob())).toThrow("live");
    expect(registerImage(world, new Blob())).not.toBe(id);
    world.dispose();
});

test("replacement and release select independent roles and preserve reserved slots", () => {
    const world = new World();
    const color = registerImage(world, new Blob(["color"]));
    const data = registerImage(world, new Blob(["data"]), { isSrgb: false });
    releaseImage(world, color);
    replaceImage(world, data, new Blob(["next"]), { isSrgb: false });
    expect(world.resource(Images).size).toBe(0);
    expect(world.resource(DataImages).size).toBe(1);
    const later = registerImage(world, new Blob(["later"]));
    expect(later).toBe(1);
    expect(world.resource(Images).name(color)).toBe("blob:0");
    world.dispose();
});

test("shared URL identities cannot be replaced or released", () => {
    const world = new World();
    const id = registerImage(world, "image.png");
    expect(() => replaceImage(world, id, new Blob())).toThrow("non-URL");
    expect(() => releaseImage(world, id)).toThrow("shared URL");
    expect(registerImage(world, "image.png")).toBe(id);
    world.dispose();
});
