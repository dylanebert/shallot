import { expect, test } from "bun:test";
import { World } from "../../engine";
import { DataImages, Images, registerImage, replaceImage } from "./image";

test("generated image replacements keep one registry slot", () => {
    const world = new World();
    const id = registerImage(world, new Blob(["first"]));
    for (let n = 0; n < 100; n++) replaceImage(world, id, new Blob([`${n}`]));
    const images = world.resource(Images);
    expect(images.size).toBe(1);
    expect(images.snapshot().names.length).toBe(1);
    world.dispose();
});

test("replacement selects independent roles", () => {
    const world = new World();
    const source = new Blob(["color"]);
    const color = registerImage(world, source);
    const data = registerImage(world, new Blob(["data"]), { isSrgb: false });
    replaceImage(world, data, new Blob(["next"]), { isSrgb: false });
    expect(world.resource(Images).get(world.resource(Images).name(color)!)?.source).toBe(source);
    expect(world.resource(Images).size).toBe(1);
    expect(world.resource(DataImages).size).toBe(1);
    world.dispose();
});

test("shared URL identities cannot be replaced", () => {
    const world = new World();
    const id = registerImage(world, "image.png");
    expect(() => replaceImage(world, id, new Blob())).toThrow("non-URL");
    expect(registerImage(world, "image.png")).toBe(id);
    world.dispose();
});
