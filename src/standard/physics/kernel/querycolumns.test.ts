import { expect, test } from "bun:test";
import { World } from "../../../engine";
import { StandardPhysicsPlugin } from "../index";
import { physicsState } from "../runtime";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import { shapeBodyId } from "./filtercolumns";
import { kernel } from "./kernel";
import { queryColumns } from "./querycolumns";

test("callback-free mover planes publish the first eight records and prepare clears body exclusion", async () => {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    try {
        const state = physicsState(world)!;
        const origin = { x: 0, y: 0, z: 0 };
        const capsule = {
            center1: { x: 0, y: -0.5, z: 0 },
            center2: { x: 0, y: 0.5, z: 0 },
            radius: 0.3,
        };
        for (let i = 0; i < 12; ++i) {
            const body = createTestBody(state, { position: { x: 0.2, y: 0, z: 0 } });
            createSphereShape(state, body, testShapeDef(), { center: origin, radius: 0.4 });
        }
        const q = queryColumns(state);
        const k = kernel(state.ecsState);
        q.prepare(origin);
        q.mover(capsule.center1, capsule.center2, capsule.radius);
        k.worldQuery(state.worldId, 5, 0);
        expect(q.resultU[0]).toBe(8);
        const expected = Array.from({ length: 8 }, (_, i) => {
            const n = 16 + 8 * i;
            return {
                shape: q.resultU[n]!,
                values: Array.from(q.resultF.subarray(n + 1, n + 8)),
            };
        });
        const excluded = shapeBodyId(state, expected[0]!.shape);
        q.headerU[19] = excluded + 1;
        k.worldQuery(state.worldId, 5, 0);
        expect(q.resultU[0]).toBe(8);
        for (let i = 0; i < 8; ++i)
            expect(shapeBodyId(state, q.resultU[16 + 8 * i]!)).not.toBe(excluded);
        q.prepare(origin);
        expect(q.headerU[19]).toBe(0);
        k.worldQuery(state.worldId, 5, 0);
        expect(q.resultU[0]).toBe(8);
        for (let i = 0; i < 8; ++i) {
            const n = 16 + 8 * i;
            expect(q.resultU[n]).toBe(expected[i]!.shape);
            expect(Array.from(q.resultF.subarray(n + 1, n + 8))).toEqual(expected[i]!.values);
        }
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("query input from a caller object whose getter grows kernel memory lands in current memory", () => {
    const state = createTestWorld();
    try {
        const memory = kernel(state.ecsState).memory;
        // Each caller object's first-read getter grows memory, detaching any view taken before it.
        const growing = <T>(value: T): T => {
            memory.grow(1);
            return value;
        };
        const q = queryColumns(state);
        q.prepare(
            {
                get x() {
                    return growing(1);
                },
                y: 2,
                z: 3,
            },
            {
                get categoryBits() {
                    return growing(0x0000000100000002n);
                },
                maskBits: 0x0000000300000004n,
            },
        );
        const trees = state.broadPhase.trees.flatMap((tree) => [
            tree.root >>> 0,
            tree.nodeCount >>> 0,
        ]);
        expect(Array.from(q.headerU.subarray(0, 10))).toEqual([...trees, 1, 2, 3, 4]);
        expect(Array.from(q.headerF.subarray(10, 13))).toEqual([1, 2, 3]);
        q.translation({
            get x() {
                return growing(11);
            },
            y: 22,
            z: 33,
        });
        expect(Array.from(q.input.subarray(9, 12))).toEqual([11, 22, 33]);
        q.placement(
            {
                p: {
                    get x() {
                        return growing(1);
                    },
                    y: 2,
                    z: 3,
                },
                q: { v: { x: 0, y: 0, z: 0.5 }, s: 0.75 },
            },
            { x: 0, y: 0, z: 0 },
        );
        expect(Array.from(q.input.subarray(0, 7))).toEqual([1, 2, 3, 0, 0, 0.5, 0.75]);
        q.proxy({
            get count() {
                return growing(2);
            },
            radius: 0.25,
            points: [
                { x: 4, y: 5, z: 6 },
                { x: 7, y: 8, z: 9 },
            ],
        });
        expect(Array.from(q.input.subarray(7, 9))).toEqual([2, 0.25]);
        expect(Array.from(q.input.subarray(14, 20))).toEqual([4, 5, 6, 7, 8, 9]);
        q.mover(
            {
                get x() {
                    return growing(1);
                },
                y: 2,
                z: 3,
            },
            { x: 4, y: 5, z: 6 },
            0.5,
        );
        expect(Array.from(q.input.subarray(7, 9))).toEqual([2, 0.5]);
        expect(Array.from(q.input.subarray(14, 20))).toEqual([1, 2, 3, 4, 5, 6]);
        q.bounds({
            lowerBound: {
                get x() {
                    return growing(-1);
                },
                y: -2,
                z: -3,
            },
            upperBound: { x: 1, y: 2, z: 3 },
        });
        expect(Array.from(q.headerF.subarray(13, 19))).toEqual([-1, -2, -3, 1, 2, 3]);
    } finally {
        destroyTestWorld(state);
    }
});
