import { expect, test } from "bun:test";
import { BodyType, defaultQueryFilter, type QueryFilter } from "../common/types";
import { createSphereShape, getSensorData, setShapeFilter } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodySetAwake, bodySetLinearVelocity, bodySetTransform, destroyBody } from "../world/body";
import { init } from "./kernel";
import { quaternion } from "./math";
import { queryColumns } from "./querycolumns";
import { SHAPE_STRIDE } from "./shapecolumns";
import { ShapeFlags, setShapeFlag } from "./shaperecords";

await init(undefined, { threads: 0 });
const origin = { x: 0, y: 0, z: 0 };
const translation = { x: 10, y: 0, z: 0 };
function rayFraction(
    state: ReturnType<typeof createTestWorld>,
    from = origin,
    delta = translation,
    filter?: QueryFilter,
) {
    const q = queryColumns(state);
    const k = q.prepare(from, filter ?? defaultQueryFilter());
    q.translation(delta);
    k.worldQuery(state.worldId, 3, 0);
    const shape = q.resultU[0]!;
    return shape === 0xffffffff ? 0 : q.resultF[5]!;
}

for (const count of [1, 32, 256]) {
    test(`warm queries over ${count} static shapes write no per-shape query columns`, () => {
        const world = createTestWorld({ gravity: origin });
        try {
            for (let i = 0; i < count; ++i) {
                const body = createTestBody(world, {
                    type: BodyType.Static,
                    position: { x: 2 + 20 * i, y: 0, z: 0 },
                });
                createSphereShape(world, body, testShapeDef(), { center: origin, radius: 0.5 });
            }
            expect(rayFraction(world)).toBeCloseTo(0.15, 6);
            const store = world.shapeStore;
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
                for (let i = 0; i < 10; ++i) expect(rayFraction(world)).toBeCloseTo(0.15, 6);
                expect(writes).toBe(0);
            } finally {
                store.shapeU = u;
                store.shapeF = f;
            }
        } finally {
            destroyTestWorld(world);
        }
    });
}

test("a filter change is published by its shape owner before the next query", () => {
    const world = createTestWorld({ gravity: origin });
    try {
        const body = createTestBody(world, {
            type: BodyType.Static,
            position: { x: 2, y: 0, z: 0 },
        });
        const shape = createSphereShape(world, body, testShapeDef(), {
            center: origin,
            radius: 0.5,
        })!;
        expect(rayFraction(world) > 0).toBe(true);
        setShapeFilter(world, shape, { categoryBits: 1n, maskBits: 0n, groupIndex: 0 });
        expect(rayFraction(world)).toBe(0);
        setShapeFilter(world, shape, {
            categoryBits: 1n,
            maskBits: 0xffffffffffffffffn,
            groupIndex: 0,
        });
        expect(rayFraction(world) > 0).toBe(true);
    } finally {
        destroyTestWorld(world);
    }
});

test("sleeping poses, wake indices and compacted awake indices are visible to the next query", () => {
    const world = createTestWorld({ gravity: origin });
    try {
        const a = createTestBody(world, { type: BodyType.Dynamic, position: { x: 2, y: 0, z: 0 } });
        const b = createTestBody(world, { type: BodyType.Dynamic, position: { x: 6, y: 0, z: 0 } });
        const sphere = { center: origin, radius: 0.5 };
        createSphereShape(world, a, testShapeDef(), sphere);
        createSphereShape(world, b, testShapeDef(), sphere);
        bodySetTransform(world, a, { x: 4, y: 0, z: 0 }, quaternion.identity());
        bodySetAwake(world, a, false);
        expect(rayFraction(world)).toBeCloseTo(0.35, 6);
        bodySetTransform(world, a, { x: 20, y: 0, z: 0 }, quaternion.identity());
        expect(rayFraction(world)).toBeCloseTo(0.55, 6);
        bodySetAwake(world, a, true);
        bodySetTransform(world, a, { x: 2, y: 0, z: 0 }, quaternion.identity());
        expect(rayFraction(world)).toBeCloseTo(0.15, 6);
        destroyBody(world, a);
        expect(rayFraction(world)).toBeCloseTo(0.55, 6);
    } finally {
        destroyTestWorld(world);
    }
});

for (const transition of ["compaction", "wake"] as const) {
    test(`${transition} publishes the awake index before kernel stepping changes the body's pose`, () => {
        const world = createTestWorld({ gravity: origin });
        try {
            const a = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 4, y: 0, z: 0 },
            });
            const b = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 6, y: 0, z: 0 },
            });
            const sphere = { center: origin, radius: 0.5 };
            createSphereShape(world, a, testShapeDef(), sphere);
            createSphereShape(world, b, testShapeDef(), sphere);
            bodySetAwake(world, a, false);
            if (transition === "compaction") {
                bodySetLinearVelocity(world, b, { x: 60, y: 0, z: 0 });
                stepTestWorld(world, 1 / 60);
                expect(rayFraction(world, { x: 5, y: 0, z: 0 })).toBeCloseTo(0.15, 6);
            } else {
                bodySetAwake(world, a, true);
                bodySetLinearVelocity(world, a, { x: -60, y: 0, z: 0 });
                stepTestWorld(world, 1 / 60);
                expect(rayFraction(world)).toBeCloseTo(0.25, 6);
            }
        } finally {
            destroyTestWorld(world);
        }
    });
}

test("a visitor's sensor flag change is published before the next sensor overlap", () => {
    const world = createTestWorld({ gravity: origin });
    try {
        const sensor = createSphereShape(
            world,
            createTestBody(world, { type: BodyType.Static }),
            testShapeDef({ isSensor: true, enableSensorEvents: true }),
            { center: origin, radius: 1 },
        )!;
        const visitor = createSphereShape(
            world,
            createTestBody(world, { type: BodyType.Dynamic }),
            testShapeDef({ enableSensorEvents: false }),
            { center: origin, radius: 0.5 },
        )!;
        stepTestWorld(world, 1 / 60);
        expect(getSensorData(world, sensor)).toHaveLength(0);
        setShapeFlag(world, visitor, ShapeFlags.enableSensorEvents, true);
        stepTestWorld(world, 1 / 60);
        expect(getSensorData(world, sensor)).toHaveLength(1);
        setShapeFlag(world, visitor, ShapeFlags.enableSensorEvents, false);
        stepTestWorld(world, 1 / 60);
        expect(getSensorData(world, sensor)).toHaveLength(0);
    } finally {
        destroyTestWorld(world);
    }
});
