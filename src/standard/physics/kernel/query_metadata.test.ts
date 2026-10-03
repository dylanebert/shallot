import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { quat } from "../common/math";
import { BodyType } from "../common/types";
import { init } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";

await init(undefined, { threads: 0 });
const origin = { x: 0, y: 0, z: 0 };
const translation = { x: 10, y: 0, z: 0 };
for (const count of [1, 32, 256]) {
    test(`warm queries over ${count} static shapes write no per-shape query columns`, () => {
        const world = new PhysicsWorld({ gravity: origin });
        try {
            for (let i = 0; i < count; ++i)
                world
                    .createBody({ type: BodyType.Static, position: { x: 2 + 20 * i, y: 0, z: 0 } })
                    .createSphere({}, { center: origin, radius: 0.5 });
            world.castRayClosest(origin, translation);
            const store = world.state.shapeStore;
            const u = store.shapeU;
            const f = store.shapeF;
            let writes = 0;
            const traps = {
                get(target: Uint32Array | Float32Array, key: string | symbol) {
                    return Reflect.get(target, key, target);
                },
                set(target: Uint32Array | Float32Array, key: string | symbol, value: unknown) {
                    if (typeof key === "string" && /^\d+$/.test(key)) {
                        const lane = Number(key) % SHAPE_STRIDE;
                        if (lane >= 18 && lane <= 32) ++writes;
                    }
                    return Reflect.set(target, key, value, target);
                },
            };
            store.shapeU = new Proxy(u, traps) as typeof u;
            store.shapeF = new Proxy(f, traps) as typeof f;
            try {
                for (let i = 0; i < 10; ++i)
                    expect(world.castRayClosest(origin, translation).fraction).toBeCloseTo(0.15, 6);
                expect(writes).toBe(0);
            } finally {
                store.shapeU = u;
                store.shapeF = f;
            }
        } finally {
            world.destroy();
        }
    });
}

test("a filter change is published by its shape owner before the next query", () => {
    const world = new PhysicsWorld({ gravity: origin });
    try {
        const shape = world
            .createBody({ type: BodyType.Static, position: { x: 2, y: 0, z: 0 } })
            .createSphere({}, { center: origin, radius: 0.5 });
        expect(world.castRayClosest(origin, translation).hit).toBe(true);
        shape.setFilter({ categoryBits: 1n, maskBits: 0n, groupIndex: 0 });
        expect(world.castRayClosest(origin, translation).hit).toBe(false);
        shape.setFilter({ categoryBits: 1n, maskBits: 0xffffffffffffffffn, groupIndex: 0 });
        expect(world.castRayClosest(origin, translation).hit).toBe(true);
    } finally {
        world.destroy();
    }
});

test("sleeping poses, wake indices and compacted awake indices are visible to the next query", () => {
    const world = new PhysicsWorld({ gravity: origin });
    try {
        const a = world.createBody({ type: BodyType.Dynamic, position: { x: 2, y: 0, z: 0 } });
        const b = world.createBody({ type: BodyType.Dynamic, position: { x: 6, y: 0, z: 0 } });
        for (const body of [a, b]) body.createSphere({}, { center: origin, radius: 0.5 });
        a.setTransform({ x: 4, y: 0, z: 0 }, quat.identity());
        a.setAwake(false);
        expect(world.castRayClosest(origin, translation).fraction).toBeCloseTo(0.35, 6);
        a.setTransform({ x: 20, y: 0, z: 0 }, quat.identity());
        expect(world.castRayClosest(origin, translation).fraction).toBeCloseTo(0.55, 6);
        a.setAwake(true);
        a.setTransform({ x: 2, y: 0, z: 0 }, quat.identity());
        expect(world.castRayClosest(origin, translation).fraction).toBeCloseTo(0.15, 6);
        a.destroy();
        expect(world.castRayClosest(origin, translation).fraction).toBeCloseTo(0.55, 6);
    } finally {
        world.destroy();
    }
});

for (const transition of ["compaction", "wake"] as const) {
    test(`${transition} publishes the awake index before kernel stepping changes the body's pose`, () => {
        const world = new PhysicsWorld({ gravity: origin });
        try {
            const a = world.createBody({ type: BodyType.Dynamic, position: { x: 4, y: 0, z: 0 } });
            const b = world.createBody({ type: BodyType.Dynamic, position: { x: 6, y: 0, z: 0 } });
            for (const body of [a, b]) body.createSphere({}, { center: origin, radius: 0.5 });
            a.setAwake(false);
            if (transition === "compaction") {
                b.setLinearVelocity({ x: 60, y: 0, z: 0 });
                world.step(1 / 60);
                expect(
                    world.castRayClosest({ x: 5, y: 0, z: 0 }, translation).fraction,
                ).toBeCloseTo(0.15, 6);
            } else {
                a.setAwake(true);
                a.setLinearVelocity({ x: -60, y: 0, z: 0 });
                world.step(1 / 60);
                expect(world.castRayClosest(origin, translation).fraction).toBeCloseTo(0.25, 6);
            }
        } finally {
            world.destroy();
        }
    });
}

test("a visitor's sensor flag change is published before the next sensor overlap", () => {
    const world = new PhysicsWorld({ gravity: origin });
    try {
        const sensor = world
            .createBody({ type: BodyType.Static })
            .createSphere(
                { isSensor: true, enableSensorEvents: true },
                { center: origin, radius: 1 },
            );
        const visitor = world
            .createBody({ type: BodyType.Dynamic })
            .createSphere({ enableSensorEvents: false }, { center: origin, radius: 0.5 });
        world.step(1 / 60);
        expect(sensor.getSensorOverlaps()).toHaveLength(0);
        visitor.enableSensorEvents(true);
        world.step(1 / 60);
        expect(sensor.getSensorOverlaps()).toHaveLength(1);
        visitor.enableSensorEvents(false);
        world.step(1 / 60);
        expect(sensor.getSensorOverlaps()).toHaveLength(0);
    } finally {
        world.destroy();
    }
});
