import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Hulls, PhysicsMeshes, Shape, ShapeKind } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine";
import type { PhysicsContactData } from "./contact";
import gold from "./contact-data.gold.json";
import { getContactData, PhysicsEvents, StandardPhysicsPlugin, setTransform } from "./index";

setDefaultTimeout(TIMEOUT.node);
const vector = (v: { x: number; y: number; z: number }) => [v.x, v.y, v.z];
function values(data: PhysicsContactData) {
    return data.manifolds.map((m) => [
        vector(m.normal),
        m.twistImpulse,
        vector(m.frictionImpulse),
        vector(m.rollingImpulse),
        m.points.map((p) => [
            vector(p.anchorA),
            vector(p.anchorB),
            p.separation,
            p.baseSeparation,
            p.normalImpulse,
            p.totalNormalImpulse,
            p.normalVelocity,
            p.featureId,
            p.triangleIndex,
            p.persisted,
        ]),
    ]);
}
function rounded<T>(value: T): T {
    return (
        Array.isArray(value)
            ? value.map(rounded)
            : typeof value === "number" && !Number.isInteger(value)
              ? Math.fround(value)
              : value
    ) as T;
}
for (const scene of [0, 1]) {
    test(`contact data equals every native field for ${scene ? "multiple manifolds" : "a stacked pair"}, in either order and after restore`, async () => {
        let ground = 0,
            body = 0;
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                ground = world.create();
                world.add(ground, Body);
                if (scene) {
                    const geometry = world.resource(PhysicsMeshes).add({
                        name: "valley",
                        vertices: new Float32Array([
                            -2, 0.2, -2, -2, 0.2, 2, 0, 0, -2, 0, 0, 2, 2, 0.2, -2, 2, 0.2, 2,
                        ]),
                        indices: new Uint32Array([0, 1, 3, 3, 2, 0, 2, 3, 5, 5, 4, 2]),
                        identifyEdges: true,
                    });
                    world.add(ground, Shape, { kind: ShapeKind.Mesh, geometry });
                } else {
                    world.add(ground, Shape, {
                        geometry: world
                            .resource(Hulls)
                            .add({ name: "ground", halfExtents: [4, 0.5, 4] }),
                    });
                }
                body = world.create();
                world.add(body, Transform, { translation: [0, scene ? 0.5 : 1, 0] });
                world.add(body, Body, { type: BodyType.Dynamic });
                world.add(body, Shape, {
                    enableContactEvents: 1,
                    geometry: world
                        .resource(Hulls)
                        .add({ name: "box", halfExtents: [0.75, 0.5, 0.75] }),
                });
            },
        });
        const world = app.world;
        try {
            expect(getContactData(world, ground, body)).toBeNull();
            for (let tick = 0; tick < 10; tick++) {
                world.tick();
                if (tick === 0) expect(world.resource(PhysicsEvents).contactBegin.count).toBe(1);
                const data = getContactData(world, ground, body)!;
                expect(values(data)).toEqual(rounded(gold[scene]![tick]));
                expect(new Set([data.shapeA, data.shapeB])).toEqual(
                    new Set([world.ref(ground), world.ref(body)]),
                );
                expect(getContactData(world, body, ground)).toEqual(data);
                const saved = world.snapshot();
                world.tick();
                world.restore(saved);
                expect(getContactData(world, ground, body)).toEqual(data);
            }
            setTransform(world, body, [0, 20, 0], [0, 0, 0, 1]);
            world.tick();
            expect(world.resource(PhysicsEvents).contactEnd.count).toBe(1);
            expect(getContactData(world, ground, body)).toBeNull();
            expect(getContactData(world, ground, 0)).toBeNull();
        } finally {
            app.dispose();
        }
    });
}
