import { expect, test } from "bun:test";
import { BodyType } from "../common/types";
import { queryColumns } from "../kernel/querycolumns";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";

const origin = { x: 0, y: 0, z: 0 };
const translation = { x: 10, y: 0, z: 0 };
const masked = { categoryBits: 1n, maskBits: 0n };
function subject() {
    const world = createTestWorld({ gravity: origin });
    const body = createTestBody(world, { type: BodyType.Static, position: { x: 2, y: 0, z: 0 } });
    createSphereShape(world, body, testShapeDef(), { center: origin, radius: 0.5 });
    return world;
}
function ray(world: ReturnType<typeof createTestWorld>, filter?: typeof masked) {
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.translation(translation);
    k.worldQuery(world.worldId, 3, 0);
    return { hit: q.resultU[0] !== 0xffffffff, fraction: q.resultF[5]! };
}

test("a world's default ray filter does not inherit another world's masked query header", () => {
    const a = subject();
    const b = subject();
    try {
        expect(ray(a).hit).toBe(true);
        expect(ray(b, masked).hit).toBe(false);
        expect(ray(a).fraction).toBeCloseTo(0.15, 6);
    } finally {
        destroyTestWorld(a);
        destroyTestWorld(b);
    }
});

test("preparing without a filter restores the default after a masked query on the same world", () => {
    const world = subject();
    try {
        expect(ray(world, masked).hit).toBe(false);
        const q = queryColumns(world);
        const k = q.prepare(origin);
        q.translation(translation);
        k.worldQuery(world.worldId, 3, 0);
        expect(q.resultU[0]).not.toBe(0xffffffff);
        expect(q.resultF[5]).toBeCloseTo(0.15, 6);
    } finally {
        destroyTestWorld(world);
    }
});
