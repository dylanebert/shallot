import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType, ShapeType } from "../common/types";
import { BodyField, bodyField } from "./bodyrecords";
import { kernel } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";
import {
    ShapeField,
    ShapeFlags,
    setShapeField,
    shapeField,
    shapeFlag,
    shapeScalar,
} from "./shaperecords";

test("shape records, links, proxies, filters, inline materials, u16 generations and LIFO pool restore in the existing kernel column without touching a sibling", () => {
    const world = new PhysicsWorld();
    const sibling = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        const bodyId = body.id.index1 - 1;
        const sphere = { center: { x: 0.25, y: 0.5, z: -0.25 }, radius: 0.5 };
        const a = body.createSphere({ density: 2.5, name: "first" }, sphere);
        const b = body.createSphere({ density: 3, enableContactEvents: true }, sphere);
        const c = body.createCapsule(
            { density: 1 },
            { center1: { x: 0, y: -1, z: 0 }, center2: { x: 0, y: 1, z: 0 }, radius: 0.2 },
        );
        const ai = a.id.index1 - 1,
            bi = b.id.index1 - 1,
            ci = c.id.index1 - 1;
        const state = world.state;
        const field = (id: number, lane: number) => shapeField(state, id, lane);
        expect(bodyField(state, bodyId, BodyField.headShapeId)).toBe(ci);
        expect(bodyField(state, bodyId, BodyField.shapeCount)).toBe(3);
        expect([
            field(ci, ShapeField.prevShapeId),
            field(ci, ShapeField.nextShapeId),
            field(bi, ShapeField.prevShapeId),
            field(bi, ShapeField.nextShapeId),
            field(ai, ShapeField.prevShapeId),
            field(ai, ShapeField.nextShapeId),
        ]).toEqual([-1, bi, ci, ai, bi, -1]);
        expect(field(ai, ShapeField.id)).toBe(ai);
        expect(field(ai, ShapeField.bodyId)).toBe(bodyId);
        expect(shapeScalar(state, ai, ShapeField.density)).toBe(2.5);
        expect(field(ci, ShapeField.type)).toBe(ShapeType.Capsule);
        expect(field(ai, ShapeField.proxyKey)).not.toBe(-1);
        expect(shapeFlag(state, bi, ShapeFlags.enableContactEvents)).toBe(true);
        const totalMass =
            a.computeMassData().mass + b.computeMassData().mass + c.computeMassData().mass;
        expect(body.getMass()).toBe(
            Math.fround(
                Math.fround(c.computeMassData().mass + b.computeMassData().mass) +
                    a.computeMassData().mass,
            ),
        );
        expect(totalMass).toBeGreaterThan(0);
        b.destroy();
        expect(field(ci, ShapeField.nextShapeId)).toBe(ai);
        expect(field(ai, ShapeField.prevShapeId)).toBe(ci);
        const reused = body.createSphere({ density: 0.75 }, sphere);
        expect(reused.id.index1 - 1).toBe(bi);
        expect(reused.id.generation).toBe(b.id.generation + 1);
        expect(b.isValid()).toBe(false);
        const filter = {
            categoryBits: 0x8000000100000002n,
            maskBits: 0x4000000200000001n,
            groupIndex: -3,
        };
        a.setFilter(filter);
        const data = { mutable: 1 };
        a.setUserData(data);
        a.setName("captured name");
        const snapshot = world.snapshot();
        state.shapeStore.refreshViews();
        const saved = state.shapeStore.shapeU.slice(ai * SHAPE_STRIDE, (ai + 1) * SHAPE_STRIDE);
        const other = sibling.createBody({ type: BodyType.Static }).createSphere({}, sphere);
        const otherBounds = other.getAABB();
        a.destroy();
        c.destroy();
        reused.destroy();
        world.restore(snapshot);
        expect(a.isValid()).toBe(true);
        expect(a.getUserData()).toBe(data);
        expect(a.getName()).toBe("captured name");
        expect(state.shapeStore.shapeU.slice(ai * SHAPE_STRIDE, (ai + 1) * SHAPE_STRIDE)).toEqual(
            saved,
        );
        expect(other.getAABB()).toEqual(otherBounds);
        setShapeField(state, ai, ShapeField.generation, 0xffff);
        a.destroy(false);
        const wrapped = body.createSphere({ updateBodyMass: false }, sphere);
        expect(wrapped.id.index1 - 1).toBe(ai);
        expect(wrapped.id.generation).toBe(0);
        expect(wrapped.isValid()).toBe(true);
        expect(kernel(state.ecsState).shapeCount(state.worldId)).toBe(3);
    } finally {
        world.destroy();
        sibling.destroy();
    }
});

test("public shape names, event flags and fresh mass/bounds outputs retain captured associations through restore", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        const shape = body.createSphere(
            { name: "authored", density: 2.5, enableSensorEvents: true },
            { center: { x: 0, y: 0, z: 0 }, radius: 1 },
        );
        shape.enableContactEvents(true);
        shape.enableHitEvents(true);
        const saved = world.snapshot();
        const output = {
            name: shape.getName(),
            type: shape.getType(),
            density: shape.getDensity(),
            body: shape.getBody().id,
            sensor: shape.areSensorEventsEnabled(),
            contact: shape.areContactEventsEnabled(),
            hit: shape.areHitEventsEnabled(),
            mass: shape.computeMassData(),
            bounds: shape.getAABB(),
        };
        expect(output).toMatchObject({
            name: "authored",
            type: ShapeType.Sphere,
            density: 2.5,
            body: body.id,
            sensor: true,
            contact: true,
            hit: true,
        });
        output.mass.center.x = 100;
        output.bounds.lowerBound.x = 100;
        expect(shape.computeMassData().center.x).toBe(0);
        expect(shape.getAABB().lowerBound.x).not.toBe(100);
        shape.setName("later");
        shape.enableSensorEvents(false);
        shape.enableContactEvents(false);
        shape.enableHitEvents(false);
        world.restore(saved);
        expect([
            shape.getName(),
            shape.areSensorEventsEnabled(),
            shape.areContactEventsEnabled(),
            shape.areHitEventsEnabled(),
        ]).toEqual(["authored", true, true, true]);
        shape.destroy();
        expect(body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 }).getName()).toBe(
            "",
        );
    } finally {
        world.destroy();
    }
});
