import { expect, test } from "bun:test";
import { World } from "../../engine";
import { registerFont } from "./index";

test("one font URL returns one id and two URLs return two", () => {
    const world = new World();
    const first = registerFont(world, "first.ttf");
    expect(registerFont(world, "first.ttf")).toBe(first);
    expect(registerFont(world, "second.ttf")).not.toBe(first);
    world.dispose();
});

function fontIdentityTypeControl(world: World): void {
    // @ts-expect-error Fonts are identified by URL, not an alias.
    registerFont(world, "font.ttf", "alias");
}
void fontIdentityTypeControl;
