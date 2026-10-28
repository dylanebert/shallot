import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, Shape, ShapeKind } from "@dylanebert/shallot";
import {
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    timeOfImpactBodyMover,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

test("body mover TOI borrows per World, clears unbound and removed hits, and preserves large-world offsets", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const other = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const origin = { x: 0, y: 0, z: 0 };
    const mover = {
        center1: { x: 0, y: -0.2, z: 0 },
        center2: { x: 0, y: 0.2, z: 0 },
        radius: 0.25,
    };
    const translation = { x: 8, y: 0, z: 0 };
    const start = { p: { x: 4, y: 0, z: 0 }, q: { v: origin, s: 1 } };
    const miss = { shape: 0, body: 0, fraction: 1, point: origin, normal: origin };
    try {
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        const body = world.create();
        world.add(body, Body, { type: BodyType.Kinematic });
        const shape = world.create();
        world.add(shape, Shape, { body, kind: ShapeKind.Sphere, radius: 0.5 });
        const borrowed = timeOfImpactBodyMover(
            world,
            body,
            origin,
            mover,
            translation,
            start,
            start,
        );
        expect(borrowed).toEqual(miss);
        world.tick();
        expect(timeOfImpactBodyMover(world, body, origin, mover, translation, start, start)).toBe(
            borrowed,
        );
        expect(borrowed.shape).toBe(shape);
        const saved = structuredClone(borrowed);
        const distantOrigin = { x: 1e9 + 10, y: 2, z: -3 };
        const distantStart = { p: { x: distantOrigin.x + 4, y: 2, z: -3 }, q: start.q };
        const distant = timeOfImpactBodyMover(
            world,
            body,
            distantOrigin,
            mover,
            translation,
            distantStart,
            distantStart,
        );
        expect(distant).toBe(borrowed);
        expect(distant.fraction).toBe(saved.fraction);
        expect(distant.normal).toEqual(saved.normal);
        expect(distant.point).toEqual({
            x: distantOrigin.x + saved.point.x,
            y: distantOrigin.y + saved.point.y,
            z: distantOrigin.z + saved.point.z,
        });
        const otherResult = timeOfImpactBodyMover(
            other.world,
            0,
            origin,
            mover,
            translation,
            start,
            start,
        );
        expect(otherResult).not.toBe(borrowed);
        expect(otherResult).toEqual(miss);
        expect(borrowed.shape).toBe(shape);
        world.destroy(shape);
        expect(
            timeOfImpactBodyMover(world, body, origin, mover, translation, start, start),
        ).toEqual(miss);
        world.destroy(body);
        expect(timeOfImpactBodyMover(world, body, origin, mover, translation, start, start)).toBe(
            borrowed,
        );
        expect(borrowed).toEqual(miss);
    } finally {
        app.dispose();
        other.dispose();
    }
});
