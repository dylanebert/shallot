import { expect } from "bun:test";
import { check } from "../../../harness/check";
import { World } from "../api/world";
import { kernel } from "../kernel/kernel";
import { BodyType } from "./types";

check(
    "public body handles preserve their packed index and generation record",
    {
        claim: "body lifecycle records lose a sibling world's validity, generation, LIFO reuse, or count when another world grows the kernel capacity",
    },
    () => {
        const growing = new World({ gravity: { x: 0, y: 0, z: 0 } });
        const sibling = new World({ gravity: { x: 0, y: 0, z: 0 } });
        const survivor = sibling.createBody({ type: BodyType.Dynamic });
        const freedA = sibling.createBody({ type: BodyType.Dynamic });
        const freedB = sibling.createBody({ type: BodyType.Dynamic });
        const freedAGeneration = freedA.id.generation;
        const freedBGeneration = freedB.id.generation;
        freedA.destroy();
        freedB.destroy();
        expect(sibling.getCounters().bodyCount).toBe(1);

        const initialCapacity = kernel().bodyCap();
        const growingBodies = [];
        while (kernel().bodyCap() === initialCapacity) {
            growingBodies.push(growing.createBody({ type: BodyType.Dynamic }));
        }

        expect(survivor.id.index1 - 1).toBe(0);
        expect(survivor.isValid()).toBe(true);
        expect(kernel().bodyGeneration(sibling.state.worldId, 0)).toBe(survivor.id.generation);
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
    },
);

check(
    "the public body pool hands out dense ids and reuses freed ones last in first out",
    {
        claim: "the public body id pool leaves holes in its dense range, miscounts live ids against capacity, or recycles freed ids in the wrong order",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: 0, z: 0 } });
        const a = world.createBody({ type: BodyType.Dynamic });
        const b = world.createBody({ type: BodyType.Dynamic });
        const c = world.createBody({ type: BodyType.Dynamic });
        expect([a.id.index1 - 1, b.id.index1 - 1, c.id.index1 - 1]).toEqual([0, 1, 2]);
        expect(world.getCounters().bodyCount).toBe(3);

        a.destroy();
        b.destroy();
        expect(world.getCounters().bodyCount).toBe(1);
        const reusedB = world.createBody({ type: BodyType.Dynamic });
        const reusedA = world.createBody({ type: BodyType.Dynamic });
        expect(reusedB.id.index1 - 1).toBe(1);
        expect(reusedA.id.index1 - 1).toBe(0);
        expect(world.createBody({ type: BodyType.Dynamic }).id.index1 - 1).toBe(3);
    },
);
