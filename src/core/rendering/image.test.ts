import { expect, test } from "bun:test";
import { World } from "../../engine";
import { registerImage } from "./image";

test("image registration accepts a source without a name", () => {
    const world = new World();
    // @ts-expect-error Images are identified by source, not an alias.
    registerImage(world, "image.png", "alias");
    world.dispose();
});

test("one image URL returns one id and two URLs return two", () => {
    const world = new World();
    const first = registerImage(world, "first.png");
    expect(registerImage(world, "first.png")).toBe(first);
    expect(registerImage(world, "second.png")).not.toBe(first);
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
