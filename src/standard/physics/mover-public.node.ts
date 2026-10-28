import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    Shape,
    ShapeKind,
    StandardPhysicsPlugin,
    Transform,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { castMover, getInverseMass, PhysicsWorldDefinition } from "./index";

setDefaultTimeout(TIMEOUT.node);
test("mover cast and inverse mass records are borrowed per World", async () => {
    const apps = await Promise.all(
        [0, 1].map(() => createApp({ defaults: false, plugins: [StandardPhysicsPlugin] })),
    );
    const mover = {
        center1: { x: 0, y: -0.2, z: 0 },
        center2: { x: 0, y: 0.2, z: 0 },
        radius: 0.25,
    };
    const origin = { x: 0, y: 0, z: 0 };
    const translation = { x: 8, y: 0, z: 0 };
    try {
        const bodies = apps.map(({ world }) => {
            world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
            const bodies = [1, 2].map((radius) => {
                const body = world.create();
                world.add(body, Transform, { translation: [radius === 1 ? 4 : 10, 0, 0] });
                world.add(body, Body, { type: BodyType.Dynamic });
                const shape = world.create();
                world.add(shape, Shape, { body, kind: ShapeKind.Sphere, radius });
                return body;
            });
            world.tick();
            return bodies;
        });
        const a = apps[0]!.world,
            b = apps[1]!.world;
        const mass = getInverseMass(a, bodies[0]![0]!)!;
        const originalMass = mass.inverseMass;
        expect(originalMass).toBeGreaterThan(0);
        expect(getInverseMass(b, bodies[1]![1]!)).not.toBe(mass);
        expect(mass.inverseMass).toBe(originalMass);
        expect(getInverseMass(a, bodies[0]![1]!)).toBe(mass);
        expect(mass.inverseMass).toBeLessThan(originalMass);
        const cast = castMover(a, origin, mover, translation);
        const fraction = cast.fraction;
        expect(fraction).toBeLessThan(1);
        expect(castMover(b, origin, mover, { x: -8, y: 0, z: 0 })).not.toBe(cast);
        expect(cast.fraction).toBe(fraction);
        expect(castMover(a, origin, mover, { x: -8, y: 0, z: 0 })).toBe(cast);
        expect(cast.fraction).toBe(1);
    } finally {
        for (const app of apps) app.dispose();
    }
});
