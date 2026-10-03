import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { World } from "../../engine";
import { BodyType, makeBoxHull, PhysicsWorld } from "./api";

setDefaultTimeout(CEILING.node);

const ray = (world: PhysicsWorld, x = 0) =>
    world.castRayClosest({ x, y: 10, z: 0 }, { x: 0, y: -20, z: 0 });

test("snapshot after a stacked hull scene and ray never clones detached query views", () => {
    const world = new PhysicsWorld({}, new World());
    try {
        world.createBody({ type: BodyType.Static }).createHull({}, makeBoxHull(10, 1, 10));
        for (let i = 0; i < 6; i++) {
            world
                .createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2 + i * 2, z: 0 } })
                .createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
        }
        for (let tick = 0; tick < 90; tick++) world.step(1 / 60);
        ray(world);
        expect(() => world.snapshot()).not.toThrow();
    } finally {
        world.destroy();
    }
});

test("a restored empty queried World uploads geometry added after restore", () => {
    const world = new PhysicsWorld({}, new World());
    try {
        expect(ray(world).hit).toBe(false);
        world.restore(world.snapshot());
        world
            .createBody({ type: BodyType.Static, position: { x: 10, y: 0, z: 0 } })
            .createHull({}, makeBoxHull(1, 1, 1));
        expect(ray(world, 10).hit).toBe(true);
    } finally {
        world.destroy();
    }
});

test("every reachable WorldState after restore is the live root", () => {
    const world = new PhysicsWorld({}, new World());
    try {
        ray(world);
        const saved = world.snapshot();
        world.restore(saved);
        const seen = new Set<object>();
        function visit(value: unknown): void {
            if (value === null || typeof value !== "object" || seen.has(value)) return;
            seen.add(value);
            if ("bodyStore" in value && "broadPhase" in value && "worldId" in value)
                expect(value).toBe(world.state);
            if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return;
            if (value instanceof Map)
                for (const [key, item] of value) {
                    visit(key);
                    visit(item);
                }
            else if (value instanceof Set) for (const item of value) visit(item);
            else
                for (const key of Reflect.ownKeys(value)) {
                    const descriptor = Object.getOwnPropertyDescriptor(value, key);
                    if (descriptor && "value" in descriptor) visit(descriptor.value);
                }
        }
        visit(world.state);
        expect(seen.has(world.state.queryColumns!)).toBe(true);
    } finally {
        world.destroy();
    }
});
