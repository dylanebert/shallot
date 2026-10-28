import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Transform } from "../../core/transform";
import { createApp, type EntityRef } from "../../engine";
import type { PhysicsContactData } from "./contact";
import { contactPair } from "./contact-data.fixture";
import gold from "./contact-data.gold.json";
import type { PhysicsContactEvent, PhysicsEventRecords } from "./events";
import { getContactData, PhysicsEvents, StandardPhysicsPlugin, setTransform } from "./index";

setDefaultTimeout(TIMEOUT.node);
const vector = (v: { x: number; y: number; z: number }) => [v.x, v.y, v.z];
function values(data: PhysicsContactData) {
    return data.manifolds
        .slice(0, data.manifoldCount)
        .map((m) => [
            vector(m.normal),
            m.twistImpulse,
            vector(m.frictionImpulse),
            vector(m.rollingImpulse),
            m.points
                .slice(0, m.pointCount)
                .map((p) => [
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
for (const scene of [0, 1, 2]) {
    test(`contact data and begin/end identities equal native for ${["a stacked pair", "multiple manifolds", "two compound children"][scene]}, in either order and after restore`, async () => {
        let ground = 0,
            body = 0;
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                [ground, body] = contactPair(world, scene);
                world.storage(Transform).translation.y.set(body, 3);
            },
        });
        const world = app.world;
        const shapeRefs = [0 as EntityRef, world.ref(ground), world.ref(body)];
        const identities = (events: PhysicsEventRecords<PhysicsContactEvent>) =>
            events.records
                .slice(0, events.count)
                .map((e) => [
                    shapeRefs.indexOf(e.shapeA),
                    shapeRefs.indexOf(e.shapeB),
                    e.childIndex,
                ]);
        try {
            expect(getContactData(world, ground, body)).toBeNull();
            for (let tick = 0; tick < 12; tick++) {
                if (tick === 1)
                    setTransform(world, body, [0, scene === 1 ? 0.5 : 1, 0], [0, 0, 0, 1]);
                if (tick === 11) setTransform(world, body, [0, 20, 0], [0, 0, 0, 1]);
                world.tick();
                const expected = gold[scene]![tick]!;
                const events = world.resource(PhysicsEvents);
                expect(identities(events.contactBegin)).toEqual(expected.begin);
                expect(identities(events.contactEnd)).toEqual(expected.end);
                for (const native of expected.contacts) {
                    const data = getContactData(world, ground, body, native.childIndex)!;
                    expect([data.shapeA, data.shapeB]).toEqual([
                        shapeRefs[native.shapeA],
                        shapeRefs[native.shapeB],
                    ]);
                    expect(data.childIndex).toBe(native.childIndex);
                    expect(data.manifoldCount).toBe(native.manifolds.length);
                    for (let m = 0; m < data.manifoldCount; m++)
                        expect(data.manifolds[m]!.pointCount).toBe(
                            (native.manifolds[m]![4] as unknown[]).length,
                        );
                    expect(values(data)).toEqual(rounded(native.manifolds));
                    const retained = structuredClone(data);
                    expect(getContactData(world, body, ground, native.childIndex)).toEqual(
                        retained,
                    );
                    const saved = world.snapshot();
                    world.tick();
                    world.restore(saved);
                    expect(getContactData(world, ground, body, native.childIndex)).toEqual(
                        retained,
                    );
                }
                if (!expected.contacts.length) {
                    expect(getContactData(world, ground, body)).toBeNull();
                    expect(getContactData(world, ground, body, 1)).toBeNull();
                }
                expect(getContactData(world, ground, body, 99)).toBeNull();
                expect(getContactData(world, ground, 0)).toBeNull();
            }
            expect(gold[2]![1]!.contacts.map((c) => c.childIndex).sort()).toEqual([0, 1]);
        } finally {
            app.dispose();
        }
    });
}
