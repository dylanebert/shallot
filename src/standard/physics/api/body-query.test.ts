import { expect, test } from "bun:test";
import type { ShapeProxy } from "../collision/distance";
import type { Pos, Transform, Vec3 } from "../common/math";
import { defaultQueryFilter } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { setQueryCallback } from "../kernel/kernel";
import { queryColumns } from "../kernel/querycolumns";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";

const zero = { x: 0, y: 0, z: 0 };
const placement = { p: { x: 4, y: 0, z: 0 }, q: { v: zero, s: 1 } };
const proxy: ShapeProxy = { points: [zero], count: 1, radius: 0.5 };

function bodyQuery(
    world: ReturnType<typeof createTestWorld>,
    body: number,
    kind: number,
    origin: Pos,
    pose: Transform,
    translation?: Vec3,
    shape?: ShapeProxy,
    maxFraction = 1,
    canEncroach = false,
) {
    const q = queryColumns(world);
    const k = q.prepare(origin, defaultQueryFilter());
    q.placement(pose, origin);
    if (shape) q.proxy(shape);
    if (translation) q.translation(translation);
    q.input[12] = maxFraction;
    q.input[13] = Number(canEncroach);
    k.bodyQuery(world.worldId, kind, bodyField(world, body, BodyField.headShapeId), 0);
    return {
        shape: q.resultU[0]!,
        fraction: q.resultF[5]!,
        point: {
            x: origin.x + q.resultF[6]!,
            y: origin.y + q.resultF[7]!,
            z: origin.z + q.resultF[8]!,
        },
        normal: { x: q.resultF[9]!, y: q.resultF[10]!, z: q.resultF[11]! },
        distance: q.resultF[3]!,
    };
}

test("body casts and overlap use the caller's placement rather than its stored pose", () => {
    const world = createTestWorld({ gravity: zero });
    try {
        const body = createTestBody(world, { position: { x: 100, y: 0, z: 0 } });
        const shape = createSphereShape(world, body, testShapeDef(), { center: zero, radius: 1 })!;
        const ray = bodyQuery(world, body, 0, zero, placement, { x: 10, y: 0, z: 0 });
        expect(ray.shape).toBe(shape);
        expect(ray.point).toEqual({ x: 3, y: 0, z: 0 });
        expect(ray.normal).toEqual({ x: -1, y: 0, z: 0 });
        expect(ray.fraction).toBe(Math.fround(0.3));
        expect(
            bodyQuery(world, body, 0, zero, placement, { x: 10, y: 0, z: 0 }, undefined, 0.2).shape,
        ).toBe(0xffffffff);
        const cast = bodyQuery(world, body, 1, zero, placement, { x: 10, y: 0, z: 0 }, proxy);
        expect(cast.shape).toBe(shape);
        expect(cast.fraction).toBe(Math.fround(Math.fround(4 - Math.fround(1.5 - 0.005)) / 10));
        expect(
            bodyQuery(world, body, 2, { x: 4, y: 0, z: 0 }, placement, undefined, proxy).shape,
        ).toBe(shape);
        expect(bodyQuery(world, body, 2, zero, placement, undefined, proxy).shape).toBe(0xffffffff);
    } finally {
        destroyTestWorld(world);
    }
});

test("body closest point reads the stored pose and mover planes obey capacity", () => {
    const world = createTestWorld({ gravity: zero });
    try {
        const body = createTestBody(world, { position: { x: 100, y: 0, z: 0 } });
        createSphereShape(world, body, testShapeDef(), { center: zero, radius: 1 });
        createSphereShape(world, body, testShapeDef(), { center: zero, radius: 1 });
        const closest = bodyQuery(
            world,
            body,
            3,
            zero,
            {
                p: { x: 100, y: 0, z: 0 },
                q: { v: zero, s: 1 },
            },
            undefined,
            { points: [{ x: 105, y: 0, z: 0 }], count: 1, radius: 0 },
        );
        expect({ point: closest.point, distance: closest.distance }).toEqual({
            point: { x: 100, y: 0, z: 0 },
            distance: 5,
        });
        const mover = {
            center1: { x: 0, y: -0.2, z: 0 },
            center2: { x: 0, y: 0.2, z: 0 },
            radius: 0.4,
        };
        const q = queryColumns(world);
        const k = q.prepare({ x: 5.2, y: 0, z: 0 }, defaultQueryFilter());
        q.placement(placement, { x: 5.2, y: 0, z: 0 });
        q.mover(mover.center1, mover.center2, mover.radius);
        const results: number[] = [];
        const previous = setQueryCallback(world.ecsState, (_kind, id) => {
            results.push(id);
            return 1;
        });
        try {
            for (const capacity of [0, 1, 2]) {
                results.length = 0;
                k.bodyQuery(
                    world.worldId,
                    4,
                    bodyField(world, body, BodyField.headShapeId),
                    capacity,
                );
                expect(results).toHaveLength(capacity);
            }
        } finally {
            setQueryCallback(world.ecsState, previous);
        }
    } finally {
        destroyTestWorld(world);
    }
});
