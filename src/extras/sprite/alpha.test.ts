import { expect, test } from "bun:test";
import { World } from "../../engine/ecs/world";
import { Sprite, SpriteAlphaMode, SpriteFill } from "./pack";

test("sprites default to Mask(0.5) and a full image; Mask preserves its threshold", () => {
    const world = new World();
    world.registry.register(Sprite);
    try {
        const eid = world.create();
        world.add(eid, Sprite);
        const sprite = world.storage(Sprite);
        expect([sprite.alphaMode.x.get(eid), sprite.alphaMode.y.get(eid)]).toEqual([
            ...SpriteAlphaMode.Mask(0.5),
        ]);
        expect(sprite.fillAmount.get(eid)).toBe(1);
        expect(sprite.fillMethod.get(eid)).toBe(SpriteFill.None);
        sprite.alphaMode.set(eid, ...SpriteAlphaMode.Mask(0.25));
        expect(sprite.alphaMode.y.get(eid)).toBe(0.25);
        expect(Object.keys(SpriteAlphaMode)).toEqual(["Mask", "Blend"]);
        for (const invalid of [-1, 1.1, Number.NaN, Number.POSITIVE_INFINITY])
            expect(() => SpriteAlphaMode.Mask(invalid)).toThrow(RangeError);
    } finally {
        world.dispose();
    }
});
