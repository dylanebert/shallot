import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { init } from "../kernel/kernel";

await init(undefined, { threads: 0 });
const origin = { x: 0, y: 0, z: 0 };
const translation = { x: 10, y: 0, z: 0 };
const proxy = { points: [origin], count: 1, radius: 0.1 };
function rays(): PhysicsWorld {
    const world = new PhysicsWorld({ gravity: origin });
    for (const [type, x] of [
        [BodyType.Static, 2],
        [BodyType.Static, 4],
        [BodyType.Kinematic, 6],
        [BodyType.Dynamic, 8],
    ] as const) {
        world
            .createBody({ type, position: { x, y: 0, z: 0 } })
            .createSphere({}, { center: origin, radius: 0.5 });
    }
    return world;
}
for (const kind of ["ray", "shape"] as const) {
    test(`${kind} callback fraction clips farther hits in this tree and later body-type trees`, () => {
        const world = rays();
        try {
            const fractions: number[] = [];
            const callback = (hit: { fraction: number }) => {
                fractions.push(hit.fraction);
                return hit.fraction;
            };
            if (kind === "ray") world.castRay(origin, translation, callback);
            else world.castShape(origin, proxy, translation, callback);
            expect(fractions).toHaveLength(1);
            expect(fractions[0]).toBeGreaterThan(0);
            expect(fractions[0]).toBeLessThan(0.2);
        } finally {
            world.destroy();
        }
    });
    test(`${kind} callback zero terminates this tree and all later body-type trees`, () => {
        const world = rays();
        try {
            let calls = 0;
            const callback = () => {
                ++calls;
                return 0;
            };
            const stats =
                kind === "ray"
                    ? world.castRay(origin, translation, callback)
                    : world.castShape(origin, proxy, translation, callback);
            expect(calls).toBe(1);
            expect(stats.nodeVisits).toBe(2);
        } finally {
            world.destroy();
        }
    });
    test(`${kind} callback minus one filters a hit without clipping later hits`, () => {
        const world = rays();
        try {
            const fractions: number[] = [];
            const callback = (hit: { fraction: number }) => {
                fractions.push(hit.fraction);
                return -1;
            };
            if (kind === "ray") world.castRay(origin, translation, callback);
            else world.castShape(origin, proxy, translation, callback);
            expect(fractions).toHaveLength(4);
            expect(fractions[3]).toBeGreaterThan(0.7);
        } finally {
            world.destroy();
        }
    });
}
for (const kind of ["aabb", "overlap", "mover"] as const) {
    test(`${kind} false stops only the current body-type tree and still queries the next tree`, () => {
        const world = new PhysicsWorld({ gravity: origin });
        try {
            for (const type of [BodyType.Static, BodyType.Kinematic, BodyType.Dynamic]) {
                for (let i = 0; i < 2; ++i)
                    world.createBody({ type }).createSphere({}, { center: origin, radius: 1 });
            }
            let calls = 0;
            const callback = () => {
                ++calls;
                return false;
            };
            if (kind === "aabb")
                world.overlapAABB(
                    { lowerBound: { x: -2, y: -2, z: -2 }, upperBound: { x: 2, y: 2, z: 2 } },
                    callback,
                );
            else if (kind === "overlap") world.overlapShape(origin, proxy, callback);
            else
                world.collideMover(
                    { x: 0, y: 1.1, z: 0 },
                    { center1: origin, center2: { x: 0, y: 0.5, z: 0 }, radius: 0.5 },
                    callback,
                );
            expect(calls).toBe(3);
        } finally {
            world.destroy();
        }
    });
}
