import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Transform } from "../../core/transform";
import { Draws, StandardRenderingPlugin } from "../../standard/rendering";
import { isMeshInstanceDraw } from "../../standard/rendering/preprocess";
import { Sprite, SpriteAlphaMode, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [SpritePlugin, StandardRenderingPlugin] },
]);

test("sprite draws follow live row keys rather than seeded or previously used keys", () => {
    const { world } = subjects()[0];
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Sprite, { alphaMode: SpriteAlphaMode.Blend });
    const keys = () =>
        [...world.resource(Draws)]
            .filter((draw) => isMeshInstanceDraw(world, draw))
            .map((draw) => draw.alphaPipelineKey)
            .sort();
    world.step(0);
    expect(keys()).toEqual([2]);
    const reads = spyOn(world.storage(Sprite).alphaMode.x, "get");
    try {
        world.step(0);
        expect(reads).toHaveBeenCalledTimes(0);
    } finally {
        reads.mockRestore();
    }
    world.storage(Sprite).alphaMode.x.set(eid, SpriteAlphaMode.Mask()[0]);
    world.step(0);
    expect(keys()).toEqual([1]);
    const shared = world.create();
    world.add(shared, Transform);
    world.add(shared, Sprite, { alphaMode: SpriteAlphaMode.Blend });
    world.step(0);
    expect(keys()).toEqual([1, 2]);
    world.destroy(eid);
    world.step(0);
    expect(keys()).toEqual([2]);
    world.remove(shared, Sprite);
    world.step(0);
    expect(keys()).toEqual([]);
    world.add(shared, Sprite);
    world.step(0);
    expect(keys()).toEqual([1]);
    world.destroy(shared);
});
