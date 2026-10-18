import { expect, test } from "bun:test";
import { Time, Transform, TransformPlugin, World } from "@dylanebert/shallot";
import { Body, BodyType, PhysicsPlugin, Shape, ShapeKind } from "@dylanebert/shallot/physics";
import type { Pos, Vec3 } from "../common/records";
import {
    castRay,
    castRayClosest,
    castShapeClosest,
    emptyQueryHit,
    StandardPhysicsPlugin,
} from "../runtime";

const origin = { x: 0, y: 0, z: 0 };
const translation = { x: 10, y: 0, z: 0 };

async function spheres(...positions: number[]): Promise<World> {
    const world = new World();
    for (const component of [...TransformPlugin.components!, ...PhysicsPlugin.components!])
        world.registry.register(component, PhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const system of StandardPhysicsPlugin.systems!)
        world.addSystem(system, StandardPhysicsPlugin.name);
    for (const x of positions) {
        const body = world.create();
        world.add(body, Transform, { translation: [x, 0, 0] });
        world.add(body, Body, { type: BodyType.Static });
        world.add(body, Shape, { kind: ShapeKind.Sphere, sphereCenter: [0, 0, 0], radius: 0.5 });
    }
    world.step(Time.FIXED_DT);
    return world;
}

async function dispose(world: World) {
    await StandardPhysicsPlugin.dispose!(world);
    world.dispose();
}

function closest(world: World, origin: Pos, translation: Vec3) {
    const hit = emptyQueryHit();
    expect(castRayClosest(world, origin, translation, hit)).toBe(true);
    return hit;
}

test("emptyQueryHit makes a zeroed miss with its own vectors that both closest casts write into", async () => {
    const zero = {
        shape: 0,
        body: 0,
        point: { x: 0, y: 0, z: 0 },
        normal: { x: 0, y: 0, z: 0 },
        fraction: 0,
        userMaterialId: 0n,
        triangleIndex: 0,
        childIndex: 0,
    };
    const a = emptyQueryHit();
    const b = emptyQueryHit();
    expect(a).toStrictEqual(zero);
    expect(a.point).not.toBe(b.point);
    expect(a.normal).not.toBe(b.normal);
    expect(a.point).not.toBe(a.normal);

    const world = await spheres(2);
    try {
        const away = { x: -10, y: 0, z: 0 };
        const proxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.25 };
        expect(castRayClosest(world, origin, away, a)).toBe(false);
        expect(a).toStrictEqual(zero);
        expect(castShapeClosest(world, origin, proxy, away, b)).toBe(false);
        expect(b).toStrictEqual(zero);
        expect(castRayClosest(world, origin, translation, a)).toBe(true);
        expect(castShapeClosest(world, origin, proxy, translation, b)).toBe(true);
        expect(a.fraction).toBeCloseTo(0.15, 6);
        expect(b.fraction).toBeGreaterThan(0);
        expect(b.fraction).toBeLessThan(a.fraction);
        expect(b.shape).toBe(a.shape);
    } finally {
        await dispose(world);
    }
});

test("thousands of throwing ray callbacks rethrow the user's error without poisoning later kernel queries", async () => {
    const world = await spheres(2);
    try {
        for (let i = 0; i < 3000; ++i) {
            const error = new Error(`user ${i}`);
            let caught: unknown;
            try {
                castRay(world, origin, translation, () => {
                    throw error;
                });
            } catch (value) {
                caught = value;
            }
            expect(caught).toBe(error);
        }
        expect(closest(world, origin, translation).fraction).toBeCloseTo(0.15, 6);
    } finally {
        await dispose(world);
    }
});

test("a ray callback can query a sibling World without interrupting its outer hits", async () => {
    const sibling = await spheres(100);
    const world = await spheres(2, 4, 6);
    try {
        const fractions: number[] = [];
        castRay(world, origin, translation, (hit) => {
            fractions.push(hit.fraction);
            expect(closest(sibling, origin, { x: 110, y: 0, z: 0 }).fraction).toBeCloseTo(
                99.5 / 110,
                6,
            );
            return 1;
        });
        expect(fractions).toHaveLength(3);
        for (const [i, fraction] of fractions.entries())
            expect(fraction).toBeCloseTo([0.15, 0.35, 0.55][i]!, 6);
        expect(closest(world, origin, translation).fraction).toBeCloseTo(0.15, 6);
        expect(closest(sibling, origin, { x: 110, y: 0, z: 0 }).fraction).toBeCloseTo(
            99.5 / 110,
            6,
        );
    } finally {
        await dispose(world);
        await dispose(sibling);
    }
});

test("same-world nested queries preserve all three outer ray hits", async () => {
    const world = await spheres(2, 4, 6);
    try {
        const fractions: number[] = [];
        castRay(world, origin, translation, (hit) => {
            fractions.push(hit.fraction);
            expect(closest(world, origin, translation).fraction).toBeCloseTo(0.15, 6);
            return 1;
        });
        expect(fractions).toHaveLength(3);
        for (const [i, fraction] of fractions.entries())
            expect(fraction).toBeCloseTo([0.15, 0.35, 0.55][i]!, 6);
    } finally {
        await dispose(world);
    }
});
