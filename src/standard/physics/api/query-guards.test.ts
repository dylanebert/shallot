import { expect, test } from "bun:test";
import { init } from "../kernel/kernel";
import { PhysicsWorld } from "./world";

await init(undefined, { threads: 0 });
const zero = { x: 0, y: 0, z: 0 };
const pose = { p: zero, q: { v: zero, s: 1 } };
const proxy = { points: [zero], count: 1, radius: 0 };
const mover = { center1: zero, center2: zero, radius: 1 };

test("locked world and body queries return native empty values before preparing inputs", () => {
    const world = new PhysicsWorld({ gravity: zero });
    const body = world.createBody({});
    const origin = {
        get x(): number {
            throw new Error("query prepared while locked");
        },
        y: 0,
        z: 0,
    };
    const callback = () => {
        throw new Error("callback ran while locked");
    };
    try {
        world.state.locked = true;
        expect(world.castRayClosest(origin, zero).hit).toBe(false);
        expect(world.castRay(origin, zero, callback)).toEqual({ nodeVisits: 0, leafVisits: 0 });
        expect(world.overlapAABB({ lowerBound: origin, upperBound: origin }, callback)).toEqual({
            nodeVisits: 0,
            leafVisits: 0,
        });
        expect(world.overlapShape(origin, proxy, callback)).toEqual({
            nodeVisits: 0,
            leafVisits: 0,
        });
        expect(world.castShape(origin, proxy, zero, callback)).toEqual({
            nodeVisits: 0,
            leafVisits: 0,
        });
        expect(world.collideMover(origin, mover, callback)).toBeUndefined();
        expect(world.castMover(origin, mover, zero)).toBe(1);
        expect(body.castRay(origin, zero, pose).hit).toBe(false);
        expect(body.castShape(origin, proxy, zero, pose).hit).toBe(false);
        expect(body.overlapShape(origin, proxy, pose)).toBe(false);
        expect(body.getClosestPoint(origin)).toEqual({ point: zero, distance: 0 });
        expect(body.collideMover(origin, mover, pose)).toEqual([]);
    } finally {
        world.state.locked = false;
        world.destroy();
    }
});

test("queries on destroyed worlds and bodies return native empty values", () => {
    const world = new PhysicsWorld({ gravity: zero });
    const body = world.createBody({});
    world.destroy();
    expect(world.castRayClosest(zero, zero).hit).toBe(false);
    expect(world.castMover(zero, mover, zero)).toBe(1);
    expect(body.castRay(zero, zero, pose).hit).toBe(false);
    expect(body.getClosestPoint(zero)).toEqual({ point: zero, distance: 0 });
});
