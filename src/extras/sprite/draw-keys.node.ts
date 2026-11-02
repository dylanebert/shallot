import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import { Transform } from "../../core/transform";
import { Draws, StandardRenderingPlugin } from "../../standard/rendering";
import { isMeshInstanceDraw } from "../../standard/rendering/preprocess";
import { Sprite, SpriteAlphaMode, SpriteMaterialType, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [SpritePlugin, StandardRenderingPlugin] },
    { defaults: false, plugins: [SpritePlugin, StandardRenderingPlugin] },
]);

test("an unplaced Mask sprite publishes no key beside a placed Blend sprite", () => {
    expectWarning(/color image 0 has no native texture/);
    const { world } = subjects()[0];
    const placed = world.create();
    const unplaced = world.create();
    world.add(placed, Transform);
    world.add(placed, Sprite, { alphaMode: SpriteAlphaMode.Blend });
    world.add(unplaced, Sprite);
    try {
        world.step(0);
        expect(
            [...world.resource(Draws)]
                .filter((draw) => isMeshInstanceDraw(world, draw))
                .map((draw) => draw.alphaPipelineKey),
        ).toEqual([2]);
    } finally {
        world.destroy(placed);
        world.destroy(unplaced);
        world.step(0);
    }
});

test("each sprite mesh publishes only the keys its own instances resolve to", () => {
    const { world } = subjects()[0];
    const blend = world.create();
    const mask = world.create();
    world.add(blend, Transform);
    world.add(blend, Sprite, { alphaMode: SpriteAlphaMode.Blend });
    world.add(mask, Transform);
    world.add(mask, Sprite);
    try {
        world.step(0);
        const meshes = world.resource(Meshes);
        const original = world.storage(MeshInstance).mesh.get(blend);
        const other = meshes.add({ ...meshes.get(original as MeshHandle)!, name: "other-sprite" });
        world.storage(MeshInstance).mesh.set(mask, other);
        world.step(0);
        const pairs = [...world.resource(Draws)]
            .filter((draw) => isMeshInstanceDraw(world, draw))
            .map((draw) => [draw.mesh, draw.alphaPipelineKey]);
        expect(pairs).toHaveLength(2);
        expect(pairs).toContainEqual([original, 2]);
        expect(pairs).toContainEqual([other, 1]);
    } finally {
        world.destroy(blend);
        world.destroy(mask);
        world.step(0);
    }
});

test("ten thousand color-animated sprites keep their bins and draw publication unchanged", () => {
    expectWarning(/color image 0 has no native texture/);
    const { world } = subjects()[1];
    const count = 10_000;
    const frames = 3;
    const eids = Array.from({ length: count }, () => {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Sprite);
        return eid;
    });
    world.step(0);
    const source = world.resource(SpriteMaterialType.source!);
    const reads = spyOn(source, "alphaMode");
    const publish = spyOn(world.resource(Draws), "register");
    try {
        for (let frame = 0; frame < frames; frame++) {
            for (const eid of eids) world.storage(Sprite).color.set(eid, frame + 1);
            world.step(0);
        }
        expect(reads).toHaveBeenCalledTimes(count * frames);
        expect(publish).toHaveBeenCalledTimes(0);
    } finally {
        reads.mockRestore();
        publish.mockRestore();
        for (const eid of eids) world.destroy(eid);
    }
});

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
