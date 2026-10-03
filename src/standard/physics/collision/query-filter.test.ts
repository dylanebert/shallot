import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { init } from "../kernel/kernel";
import { queryColumns } from "../kernel/querycolumns";

await init(undefined, { threads: 0 });
const origin = { x: 0, y: 0, z: 0 };
const translation = { x: 10, y: 0, z: 0 };
const masked = { categoryBits: 1n, maskBits: 0n };
function subject() {
    const world = new PhysicsWorld({ gravity: origin });
    world
        .createBody({ type: BodyType.Static, position: { x: 2, y: 0, z: 0 } })
        .createSphere({}, { center: origin, radius: 0.5 });
    return world;
}

test("a world's default ray filter does not inherit another world's masked query header", () => {
    const a = subject();
    const b = subject();
    try {
        expect(a.castRayClosest(origin, translation).hit).toBe(true);
        expect(b.castRayClosest(origin, translation, masked).hit).toBe(false);
        expect(a.castRayClosest(origin, translation).fraction).toBeCloseTo(0.15, 6);
    } finally {
        a.destroy();
        b.destroy();
    }
});

test("preparing without a filter restores the default after a masked query on the same world", () => {
    const world = subject();
    try {
        expect(world.castRayClosest(origin, translation, masked).hit).toBe(false);
        const q = queryColumns(world.state);
        const k = q.prepare(origin);
        q.translation(translation);
        k.worldQuery(world.state.worldId, 3, 0);
        expect(q.resultU[0]).not.toBe(0xffffffff);
        expect(q.resultF[5]).toBeCloseTo(0.15, 6);
    } finally {
        world.destroy();
    }
});
