import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { init } from "../kernel/kernel";

await init(undefined, { threads: 0 });
const origin = { x: 0, y: 0, z: 0 };
const translation = { x: 10, y: 0, z: 0 };
function spheres(...positions: number[]): PhysicsWorld {
    const world = new PhysicsWorld({ gravity: origin });
    for (const x of positions)
        world
            .createBody({ type: BodyType.Static, position: { x, y: 0, z: 0 } })
            .createSphere({}, { center: origin, radius: 0.5 });
    return world;
}

test("thousands of throwing ray callbacks rethrow the user's error without poisoning later kernel queries", () => {
    const world = spheres(2);
    try {
        for (let i = 0; i < 3000; ++i) {
            const error = new Error(`user ${i}`);
            let caught: unknown;
            try {
                world.castRay(origin, translation, () => {
                    throw error;
                });
            } catch (value) {
                caught = value;
            }
            expect(caught).toBe(error);
        }
        expect(world.castRayClosest(origin, translation).fraction).toBeCloseTo(0.15, 6);
    } finally {
        world.destroy();
    }
});

test("a callback cannot interleave queries from two worlds on one kernel, and both worlds remain queryable after refusal", () => {
    const b = spheres(100);
    const a = spheres(2, 4, 6);
    try {
        expect(() =>
            a.castRay(origin, translation, () => {
                b.castRayClosest(origin, { x: 110, y: 0, z: 0 });
                return 1;
            }),
        ).toThrow("one kernel cannot interleave two worlds' queries");
        expect(a.castRayClosest(origin, translation).fraction).toBeCloseTo(0.15, 6);
        expect(b.castRayClosest(origin, { x: 110, y: 0, z: 0 }).fraction).toBeCloseTo(
            99.5 / 110,
            6,
        );
    } finally {
        a.destroy();
        b.destroy();
    }
});

test("same-world nested queries preserve all three outer ray hits", () => {
    const world = spheres(2, 4, 6);
    try {
        const fractions: number[] = [];
        world.castRay(origin, translation, (hit) => {
            fractions.push(hit.fraction);
            expect(world.castRayClosest(origin, translation).fraction).toBeCloseTo(0.15, 6);
            return 1;
        });
        expect(fractions).toHaveLength(3);
        for (const [i, fraction] of fractions.entries())
            expect(fraction).toBeCloseTo([0.15, 0.35, 0.55][i], 6);
    } finally {
        world.destroy();
    }
});
