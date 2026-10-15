import { expect, test } from "bun:test";
import gold from "../../../../crates/physics/tests/reader_regressions.gold.json";
import { restore, snapshot } from "../api/snapshot";
import { BodyType, ShapeType } from "../common/types";
import {
    computeShapeMass,
    createCapsuleShape,
    createSphereShape,
    destroyShape,
    setShapeFilter,
    shapeIsValid,
} from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import { getMassData } from "../world/body";
import { BodyField, bodyField } from "./bodyrecords";
import { kernel } from "./kernel";
import { readShapeAabb, SHAPE_STRIDE } from "./shapecolumns";
import {
    ShapeField,
    ShapeFlags,
    setShapeField,
    setShapeFlag,
    shapeField,
    shapeFlag,
    shapeScalar,
} from "./shaperecords";

function shapeId(world: ReturnType<typeof createTestWorld>, shape: number) {
    return {
        index1: shape + 1,
        world0: world.worldId,
        generation: shapeField(world, shape, ShapeField.generation),
    };
}

test("shape records, links, proxies, filters, inline materials, u16 generations and LIFO pool restore in the existing kernel column without touching a sibling", () => {
    const world = createTestWorld();
    const sibling = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        const sphere = { center: { x: 0.25, y: 0.5, z: -0.25 }, radius: 0.5 };
        const a = createSphereShape(
            world,
            body,
            testShapeDef({ density: 2.5, name: "first" }),
            sphere,
        )!;
        const b = createSphereShape(
            world,
            body,
            testShapeDef({ density: 3, enableContactEvents: true }),
            sphere,
        )!;
        const c = createCapsuleShape(world, body, testShapeDef({ density: 1 }), {
            center1: { x: 0, y: -1, z: 0 },
            center2: { x: 0, y: 1, z: 0 },
            radius: 0.2,
        })!;
        const ai = a,
            bi = b,
            ci = c;
        const field = (id: number, lane: number) => shapeField(world, id, lane);
        expect(bodyField(world, body, BodyField.headShapeId)).toBe(ci);
        expect(bodyField(world, body, BodyField.shapeCount)).toBe(3);
        expect([
            field(ci, ShapeField.prevShapeId),
            field(ci, ShapeField.nextShapeId),
            field(bi, ShapeField.prevShapeId),
            field(bi, ShapeField.nextShapeId),
            field(ai, ShapeField.prevShapeId),
            field(ai, ShapeField.nextShapeId),
        ]).toEqual([-1, bi, ci, ai, bi, -1]);
        expect(field(ai, ShapeField.id)).toBe(ai);
        expect(field(ai, ShapeField.bodyId)).toBe(body);
        expect(shapeScalar(world, ai, ShapeField.density)).toBe(2.5);
        expect(field(ci, ShapeField.type)).toBe(ShapeType.Capsule);
        expect(field(ai, ShapeField.proxyKey)).not.toBe(-1);
        expect(shapeFlag(world, bi, ShapeFlags.enableContactEvents)).toBe(true);
        const aMass = computeShapeMass(world, a).mass;
        const bMass = computeShapeMass(world, b).mass;
        const cMass = computeShapeMass(world, c).mass;
        const nativeMass = new Float32Array(new Uint32Array([gold.shapeMass]).buffer)[0];
        expect(getMassData(world, body).mass).toBe(nativeMass);
        for (const mass of [aMass, bMass, cMass]) expect(mass).toBeGreaterThan(0);
        const bId = shapeId(world, b);
        destroyShape(world, b, true);
        expect(field(ci, ShapeField.nextShapeId)).toBe(ai);
        expect(field(ai, ShapeField.prevShapeId)).toBe(ci);
        const reused = createSphereShape(world, body, testShapeDef({ density: 0.75 }), sphere)!;
        expect(reused).toBe(bi);
        expect(shapeId(world, reused).generation).toBe(bId.generation + 1);
        expect(shapeIsValid(world, bId)).toBe(false);
        const filter = {
            categoryBits: 0x8000000100000002n,
            maskBits: 0x4000000200000001n,
            groupIndex: -3,
        };
        setShapeFilter(world, a, filter);
        const data = { mutable: 1 };
        world.shapeUserData[ai] = data;
        world.shapeNames[ai] = "captured name";
        const saved = snapshot(world);
        world.shapeStore.refreshViews();
        const shapeWords = world.shapeStore.shapeU.slice(
            ai * SHAPE_STRIDE,
            (ai + 1) * SHAPE_STRIDE,
        );
        const otherBody = createTestBody(sibling, { type: BodyType.Static });
        const other = createSphereShape(sibling, otherBody, testShapeDef(), sphere)!;
        const otherBounds = readShapeAabb(sibling, other, {
            lowerBound: { x: 0, y: 0, z: 0 },
            upperBound: { x: 0, y: 0, z: 0 },
        });
        destroyShape(world, a, true);
        destroyShape(world, c, true);
        destroyShape(world, reused, true);
        restore(world, saved);
        expect(shapeIsValid(world, shapeId(world, ai))).toBe(true);
        expect(world.shapeUserData[ai]).toBe(data);
        expect(world.shapeNames[ai]).toBe("captured name");
        expect(world.shapeStore.shapeU.slice(ai * SHAPE_STRIDE, (ai + 1) * SHAPE_STRIDE)).toEqual(
            shapeWords,
        );
        expect(
            readShapeAabb(sibling, other, {
                lowerBound: { x: 0, y: 0, z: 0 },
                upperBound: { x: 0, y: 0, z: 0 },
            }),
        ).toEqual(otherBounds);
        setShapeField(world, ai, ShapeField.generation, 0xffff);
        destroyShape(world, ai, false);
        const wrapped = createSphereShape(
            world,
            body,
            testShapeDef({ updateBodyMass: false }),
            sphere,
        )!;
        expect(wrapped).toBe(ai);
        expect(shapeField(world, wrapped, ShapeField.generation)).toBe(0);
        expect(shapeIsValid(world, shapeId(world, wrapped))).toBe(true);
        expect(kernel(world.ecsState).shapeCount(world.worldId)).toBe(3);
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(sibling);
    }
});

test("shape names, event flags and fresh mass/bounds outputs retain captured associations through restore", () => {
    const world = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        const shape = createSphereShape(
            world,
            body,
            testShapeDef({ name: "authored", density: 2.5, enableSensorEvents: true }),
            {
                center: { x: 0, y: 0, z: 0 },
                radius: 1,
            },
        )!;
        setShapeFlag(world, shape, ShapeFlags.enableContactEvents, true);
        setShapeFlag(world, shape, ShapeFlags.enableHitEvents, true);
        const saved = snapshot(world);
        const output = {
            name: world.shapeNames[shape] ?? "",
            type: shapeField(world, shape, ShapeField.type),
            density: shapeScalar(world, shape, ShapeField.density),
            body: bodyField(world, shapeField(world, shape, ShapeField.bodyId), BodyField.id),
            sensor: shapeFlag(world, shape, ShapeFlags.enableSensorEvents),
            contact: shapeFlag(world, shape, ShapeFlags.enableContactEvents),
            hit: shapeFlag(world, shape, ShapeFlags.enableHitEvents),
            mass: computeShapeMass(world, shape),
            bounds: readShapeAabb(world, shape, {
                lowerBound: { x: 0, y: 0, z: 0 },
                upperBound: { x: 0, y: 0, z: 0 },
            }),
        };
        expect(output).toMatchObject({
            name: "authored",
            type: ShapeType.Sphere,
            density: 2.5,
            body,
            sensor: true,
            contact: true,
            hit: true,
        });
        output.mass.center.x = 100;
        output.bounds.lowerBound.x = 100;
        expect(computeShapeMass(world, shape).center.x).toBe(0);
        expect(
            readShapeAabb(world, shape, {
                lowerBound: { x: 0, y: 0, z: 0 },
                upperBound: { x: 0, y: 0, z: 0 },
            }).lowerBound.x,
        ).not.toBe(100);
        world.shapeNames[shape] = "later";
        setShapeFlag(world, shape, ShapeFlags.enableSensorEvents, false);
        setShapeFlag(world, shape, ShapeFlags.enableContactEvents, false);
        setShapeFlag(world, shape, ShapeFlags.enableHitEvents, false);
        restore(world, saved);
        expect([
            world.shapeNames[shape],
            shapeFlag(world, shape, ShapeFlags.enableSensorEvents),
            shapeFlag(world, shape, ShapeFlags.enableContactEvents),
            shapeFlag(world, shape, ShapeFlags.enableHitEvents),
        ]).toEqual(["authored", true, true, true]);
        destroyShape(world, shape, true);
        const fresh = createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 1,
        })!;
        expect(world.shapeNames[fresh] ?? "").toBe("");
    } finally {
        destroyTestWorld(world);
    }
});
