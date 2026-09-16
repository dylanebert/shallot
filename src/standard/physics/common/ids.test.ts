import { expect } from "bun:test";
import { check } from "../../../harness/check";
import { World } from "../api/world";
import { BodyType } from "../common/types";
import { kernel } from "../kernel/kernel";

check(
    "public body handles preserve their packed index and generation record",
    {
        claim: "the public body handle loses or reorders its index or generation across kernel create and destroy",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: 0, z: 0 } });
        const first = world.createBody({ type: BodyType.Dynamic });
        const index = first.id.index1 - 1;
        const firstGeneration = first.id.generation;

        expect(kernel().bodyAlive(world.state.worldId, index)).toBe(1);
        expect(kernel().bodyGeneration(world.state.worldId, index)).toBe(firstGeneration);
        first.destroy();
        expect(kernel().bodyAlive(world.state.worldId, index)).toBe(0);
        expect(first.isValid()).toBe(false);

        const replacement = world.createBody({ type: BodyType.Dynamic });
        expect(replacement.id.index1 - 1).toBe(index);
        expect(replacement.id.generation).not.toBe(firstGeneration);
        expect(kernel().bodyGeneration(world.state.worldId, index)).toBe(replacement.id.generation);
        expect(first.isValid()).toBe(false);
        expect(replacement.isValid()).toBe(true);
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
