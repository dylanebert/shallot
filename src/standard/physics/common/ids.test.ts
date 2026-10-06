import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { kernel } from "../kernel/kernel";
import { BodyType } from "./types";

test("body generations wrap at sixteen bits as Box3D handles do", () => {
    const world = new PhysicsWorld();
    try {
        const k = kernel(undefined);
        for (let generation = 1; generation <= 0x10001; ++generation) {
            const id = k.bodyCreate(world.state.worldId);
            expect(id).toBe(0);
            expect(k.bodyGeneration(world.state.worldId, id)).toBe(generation & 0xffff);
            k.bodyDestroy(world.state.worldId, id);
        }
    } finally {
        world.destroy();
    }
});

test("body lifecycle records lose a sibling world's validity, generation, LIFO reuse, or count when another world grows the kernel capacity", () => {
    const growing = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const survivor = sibling.createBody({ type: BodyType.Dynamic });
    const freedA = sibling.createBody({ type: BodyType.Dynamic });
    const freedB = sibling.createBody({ type: BodyType.Dynamic });
    const freedAGeneration = freedA.id.generation;
    const freedBGeneration = freedB.id.generation;
    freedA.destroy();
    freedB.destroy();
    expect(sibling.getCounters().bodyCount).toBe(1);

    const initialCapacity = kernel(undefined).bodyCap();
    const growingBodies = [];
    while (kernel(undefined).bodyCap() === initialCapacity) {
        growingBodies.push(growing.createBody({ type: BodyType.Dynamic }));
    }

    expect(survivor.id.index1 - 1).toBe(0);
    expect(survivor.isValid()).toBe(true);
    expect(kernel(undefined).bodyGeneration(sibling.state.worldId, 0)).toBe(survivor.id.generation);
    expect(sibling.getCounters().bodyCount).toBe(1);

    const reusedB = sibling.createBody({ type: BodyType.Dynamic });
    const reusedA = sibling.createBody({ type: BodyType.Dynamic });
    expect(reusedB.id.index1 - 1).toBe(freedB.id.index1 - 1);
    expect(reusedA.id.index1 - 1).toBe(freedA.id.index1 - 1);
    expect(reusedB.id.generation).not.toBe(freedBGeneration);
    expect(reusedA.id.generation).not.toBe(freedAGeneration);
    expect(sibling.getCounters().bodyCount).toBe(3);

    survivor.destroy();
    expect(survivor.isValid()).toBe(false);
    expect(sibling.getCounters().bodyCount).toBe(2);
    for (const body of growingBodies) body.destroy();
    growing.destroy();
    sibling.destroy();
});

test("the public body id pool leaves holes in its dense range, miscounts live ids against capacity, or recycles freed ids in the wrong order", () => {
    const physicsWorld = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const a = physicsWorld.createBody({ type: BodyType.Dynamic });
    const b = physicsWorld.createBody({ type: BodyType.Dynamic });
    const c = physicsWorld.createBody({ type: BodyType.Dynamic });
    expect([a.id.index1 - 1, b.id.index1 - 1, c.id.index1 - 1]).toEqual([0, 1, 2]);
    expect(physicsWorld.getCounters().bodyCount).toBe(3);

    a.destroy();
    b.destroy();
    expect(physicsWorld.getCounters().bodyCount).toBe(1);
    const reusedB = physicsWorld.createBody({ type: BodyType.Dynamic });
    const reusedA = physicsWorld.createBody({ type: BodyType.Dynamic });
    expect(reusedB.id.index1 - 1).toBe(1);
    expect(reusedA.id.index1 - 1).toBe(0);
    expect(physicsWorld.createBody({ type: BodyType.Dynamic }).id.index1 - 1).toBe(3);
});
