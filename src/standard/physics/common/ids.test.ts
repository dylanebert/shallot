import { expect, test } from "bun:test";
import { kernel } from "../kernel/kernel";
import { createTestBody, createTestWorld, destroyTestWorld } from "../testing";
import { bodyIsValid, destroyBody, makeBodyId } from "../world/body";
import { worldCounters } from "../world/world";
import { BodyType } from "./types";

test("body generations wrap at sixteen bits as Box3D handles do", () => {
    const world = createTestWorld();
    try {
        const k = kernel(world.ecsState);
        for (let generation = 1; generation <= 0x10001; ++generation) {
            const id = k.bodyCreate(world.worldId);
            expect(id).toBe(0);
            expect(k.bodyGeneration(world.worldId, id)).toBe(generation & 0xffff);
            k.bodyDestroy(world.worldId, id);
        }
    } finally {
        destroyTestWorld(world);
    }
});

test("body lifecycle records lose a sibling world's validity, generation, LIFO reuse, or count when another world grows the kernel capacity", () => {
    const growing = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const survivor = createTestBody(sibling, { type: BodyType.Dynamic });
    const freedA = createTestBody(sibling, { type: BodyType.Dynamic });
    const freedB = createTestBody(sibling, { type: BodyType.Dynamic });
    const freedAGeneration = makeBodyId(sibling, freedA).generation;
    const freedBGeneration = makeBodyId(sibling, freedB).generation;
    destroyBody(sibling, freedA);
    destroyBody(sibling, freedB);
    expect(worldCounters(sibling).bodyCount).toBe(1);

    const initialCapacity = kernel(undefined).bodyCap();
    const growingBodies: number[] = [];
    while (kernel(undefined).bodyCap() === initialCapacity) {
        growingBodies.push(createTestBody(growing, { type: BodyType.Dynamic }));
    }

    expect(survivor).toBe(0);
    expect(bodyIsValid(sibling, makeBodyId(sibling, survivor))).toBe(true);
    expect(kernel(undefined).bodyGeneration(sibling.worldId, 0)).toBe(
        makeBodyId(sibling, survivor).generation,
    );
    expect(worldCounters(sibling).bodyCount).toBe(1);

    const reusedB = createTestBody(sibling, { type: BodyType.Dynamic });
    const reusedA = createTestBody(sibling, { type: BodyType.Dynamic });
    expect(reusedB).toBe(freedB);
    expect(reusedA).toBe(freedA);
    expect(makeBodyId(sibling, reusedB).generation).not.toBe(freedBGeneration);
    expect(makeBodyId(sibling, reusedA).generation).not.toBe(freedAGeneration);
    expect(worldCounters(sibling).bodyCount).toBe(3);

    destroyBody(sibling, survivor);
    expect(bodyIsValid(sibling, makeBodyId(sibling, survivor))).toBe(false);
    expect(worldCounters(sibling).bodyCount).toBe(2);
    for (const body of growingBodies) destroyBody(growing, body);
    destroyTestWorld(growing);
    destroyTestWorld(sibling);
});

test("the body id pool leaves holes in its dense range, miscounts live ids against capacity, or recycles freed ids in the wrong order", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const a = createTestBody(world, { type: BodyType.Dynamic });
    const b = createTestBody(world, { type: BodyType.Dynamic });
    const c = createTestBody(world, { type: BodyType.Dynamic });
    expect([a, b, c]).toEqual([0, 1, 2]);
    expect(worldCounters(world).bodyCount).toBe(3);

    destroyBody(world, a);
    destroyBody(world, b);
    expect(worldCounters(world).bodyCount).toBe(1);
    const reusedB = createTestBody(world, { type: BodyType.Dynamic });
    const reusedA = createTestBody(world, { type: BodyType.Dynamic });
    expect(reusedB).toBe(1);
    expect(reusedA).toBe(0);
    expect(createTestBody(world, { type: BodyType.Dynamic })).toBe(3);
    destroyTestWorld(world);
});
