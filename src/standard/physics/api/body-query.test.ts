import { expect, test } from "bun:test";
import { init } from "../kernel/kernel";
import { PhysicsWorld } from "./world";

await init(undefined, { threads: 0 });
const zero = { x: 0, y: 0, z: 0 };
const placement = { p: { x: 4, y: 0, z: 0 }, q: { v: zero, s: 1 } };

test("body casts and overlap use the caller's placement rather than its stored pose", () => {
    const world = new PhysicsWorld({ gravity: zero });
    try {
        const body = world.createBody({ position: { x: 100, y: 0, z: 0 } });
        const shape = body.createSphere({}, { center: zero, radius: 1 });
        const ray = body.castRay(zero, { x: 10, y: 0, z: 0 }, placement);
        expect(ray.hit).toBe(true);
        expect(ray.shape?.id).toEqual(shape.id);
        expect(ray.point).toEqual({ x: 3, y: 0, z: 0 });
        expect(ray.normal).toEqual({ x: -1, y: 0, z: 0 });
        expect(ray.fraction).toBe(Math.fround(0.3));
        expect(body.castRay(zero, { x: 10, y: 0, z: 0 }, placement, undefined, 0.2).hit).toBe(
            false,
        );
        const proxy = { points: [zero], count: 1, radius: 0.5 };
        const cast = body.castShape(zero, proxy, { x: 10, y: 0, z: 0 }, placement);
        expect(cast.hit).toBe(true);
        // Box3D's cast target subtracts the 0.005 linear slop from the combined radius.
        expect(cast.fraction).toBe(Math.fround(Math.fround(4 - Math.fround(1.5 - 0.005)) / 10));
        expect(body.overlapShape({ x: 4, y: 0, z: 0 }, proxy, placement)).toBe(true);
        expect(body.overlapShape(zero, proxy, placement)).toBe(false);
        expect(
            body.castRay(zero, { x: 10, y: 0, z: 0 }, placement, { categoryBits: 0n, maskBits: 0n })
                .hit,
        ).toBe(false);
    } finally {
        world.destroy();
    }
});

test("body closest point reads the stored pose and mover planes obey capacity", () => {
    const world = new PhysicsWorld({ gravity: zero });
    try {
        const body = world.createBody({ position: { x: 100, y: 0, z: 0 } });
        body.createSphere({}, { center: zero, radius: 1 });
        body.createSphere({}, { center: zero, radius: 1 });
        expect(body.getClosestPoint({ x: 105, y: 0, z: 0 })).toEqual({
            point: { x: 100, y: 0, z: 0 },
            distance: 5,
        });
        const mover = {
            center1: { x: 0, y: -0.2, z: 0 },
            center2: { x: 0, y: 0.2, z: 0 },
            radius: 0.4,
        };
        expect(body.collideMover({ x: 5.2, y: 0, z: 0 }, mover, placement, 0)).toHaveLength(0);
        expect(body.collideMover({ x: 5.2, y: 0, z: 0 }, mover, placement, 1)).toHaveLength(1);
        expect(body.collideMover({ x: 5.2, y: 0, z: 0 }, mover, placement, 2)).toHaveLength(2);
    } finally {
        world.destroy();
    }
});
