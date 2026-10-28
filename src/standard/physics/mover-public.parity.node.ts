import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, Hulls, Shape, ShapeKind, Transform } from "@dylanebert/shallot";
import {
    castMover,
    getInverseMass,
    getWorldInverseRotationalInertia,
    hashPhysics,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);
const bits = (value: number) =>
    new Uint32Array(new Float32Array([value]).buffer)[0]!.toString(16).padStart(8, "0");
test("mover public queries equal native inverse mass and world inertia across body states and capsule casts", async () => {
    const rows: string[] = [];
    for (let c = 0; c < 6; c++) {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        const world = app.world;
        try {
            world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
            const body = world.create();
            world.add(body, Transform, { translation: [1, 3, 2], rotation: [0, 0.6, 0, 0.8] });
            world.add(body, Body, {
                type: c === 0 ? BodyType.Static : c === 1 ? BodyType.Kinematic : BodyType.Dynamic,
                angularVelocity: [1, -1, 2],
                motionLocks: c === 3 ? 56 : 0,
                isAwake: Number(c !== 4),
                isEnabled: Number(c !== 5),
            });
            world.add(body, Shape, {
                geometry: world.resource(Hulls).add({ name: "box", halfExtents: [0.5, 0.5, 1] }),
            });
            expect(getInverseMass(world, body)).toBeNull();
            expect(getWorldInverseRotationalInertia(world, body)).toBeNull();
            world.tick();
            for (let step = 0; step < 2; step++) {
                const hash = hashPhysics(world);
                const matrix = {
                    cx: { x: 0, y: 0, z: 0 },
                    cy: { x: 0, y: 0, z: 0 },
                    cz: { x: 0, y: 0, z: 0 },
                };
                expect(getWorldInverseRotationalInertia(world, body, matrix)).toBe(matrix);
                rows.push(
                    `inverse ${c} ${step} ${[getInverseMass(world, body)!.inverseMass, ...[matrix.cx, matrix.cy, matrix.cz].flatMap((v) => [v.x, v.y, v.z])].map(bits).join(" ")}`,
                );
                expect(hashPhysics(world)).toBe(hash);
                if (step === 0) {
                    world.resource(PhysicsWorldDefinition).enableAutomaticStep = true;
                    world.tick();
                }
            }
            world.destroy(body);
            expect(getInverseMass(world, body)).toBeNull();
            expect(getWorldInverseRotationalInertia(world, body)).toBeNull();
        } finally {
            app.dispose();
        }
    }
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        const body = world.create();
        world.add(body, Transform, { translation: [4, 0, 0] });
        world.add(body, Body);
        const shape = world.create();
        world.add(shape, Shape, { body, kind: ShapeKind.Sphere, radius: 1 });
        const mover = {
            center1: { x: 0, y: -0.2, z: 0 },
            center2: { x: 0, y: 0.2, z: 0 },
            radius: 0.25,
        };
        expect(castMover(world, { x: 0, y: 0, z: 0 }, mover, { x: 8, y: 0, z: 0 }).fraction).toBe(
            1,
        );
        world.tick();
        for (let c = 0; c < 7; c++) {
            const origin = { x: c === 2 ? 4 : c === 6 ? 2.75 : 0, y: 0, z: 0 };
            const filter = {
                categoryBits: 0xffffffffffffffffn,
                maskBits: c === 3 ? 0n : 0xffffffffffffffffn,
            };
            const accept = (hitShape: number, hitBody: number) => {
                expect([hitShape, hitBody]).toEqual([shape, body]);
                return c !== 4;
            };
            rows.push(
                `cast ${c} ${bits(castMover(world, origin, mover, { x: c === 1 ? -8 : 8, y: 0, z: 0 }, filter, c === 4 || c === 5 ? accept : undefined).fraction)}`,
            );
        }
        expect(() =>
            castMover(world, { x: 0, y: 0, z: 0 }, mover, { x: 8, y: 0, z: 0 }, undefined, () => {
                throw new Error("filter failed");
            }),
        ).toThrow("filter failed");
        expect(
            castMover(world, { x: 0, y: 0, z: 0 }, mover, { x: 8, y: 0, z: 0 }).fraction,
        ).toBeLessThan(1);
        world.destroy(shape);
        expect(castMover(world, { x: 0, y: 0, z: 0 }, mover, { x: 8, y: 0, z: 0 }).fraction).toBe(
            1,
        );
        expect(
            castMover(world, { x: 0, y: 0, z: 0 }, mover, { x: 8, y: 0, z: 0 }, undefined, () => {
                throw new Error("removed shape reached filter");
            }).fraction,
        ).toBe(1);
    } finally {
        app.dispose();
    }
    expect(rows.join("\n") + "\n").toBe(
        await Bun.file(new URL("./mover-public.gold.txt", import.meta.url)).text(),
    );
});
