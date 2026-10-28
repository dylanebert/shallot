import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, Hulls, Shape } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { TIMEOUT } from "../../project/testing.js";
import { getMassData, hashPhysics, StandardPhysicsPlugin, setMassData } from "./index";
import gold from "./mass-data.gold.json";

setDefaultTimeout(TIMEOUT.node);
const override = {
    mass: 3,
    center: { x: 0.5, y: 0.5, z: 1 },
    inertia: { cx: { x: 2, y: 0.25, z: 0 }, cy: { x: 0.25, y: 3, z: 0 }, cz: { x: 0, y: 0, z: 4 } },
};
const zero = {
    mass: 0,
    center: { x: 0, y: 0, z: 0 },
    inertia: { cx: { x: 0, y: 0, z: 0 }, cy: { x: 0, y: 0, z: 0 }, cz: { x: 0, y: 0, z: 0 } },
};
test("mass overrides equal native Box3D for static, kinematic, dynamic, rotation-locked, sleeping and disabled bodies", async () => {
    let row = 0;
    for (let c = 0; c < 6; c++) {
        let eid = 0;
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                eid = world.create();
                world.add(eid, Transform, { translation: [1, 3, 2] });
                world.add(eid, Body, {
                    type:
                        c === 0 ? BodyType.Static : c === 1 ? BodyType.Kinematic : BodyType.Dynamic,
                    angularVelocity: [1, -1, 2],
                    motionLocks: c === 3 ? 56 : 0,
                    isAwake: Number(c !== 4),
                    isEnabled: Number(c !== 5),
                });
                world.add(eid, Shape, {
                    geometry: world
                        .resource(Hulls)
                        .add({ name: "box", halfExtents: [0.5, 0.5, 1] }),
                });
            },
        });
        const world = app.world;
        const snapshot = () => {
            const expected = gold[row++]!;
            const data = getMassData(world, eid)!;
            const values = (v: { x: number; y: number; z: number }) => [v.x, v.y, v.z];
            expect(hashPhysics(world).toString(16).padStart(16, "0"), `case ${c} row ${row}`).toBe(
                expected.hash,
            );
            expect(
                Array.from(
                    world
                        .storage(GlobalTransform)
                        .linearVelocity.column.subarray(eid * 4, eid * 4 + 3),
                ),
            ).toEqual(expected.velocity.map(Math.fround));
            expect(data.mass).toBe(Math.fround(expected.mass));
            expect(values(data.center)).toEqual(expected.center.map(Math.fround));
            expect([data.inertia.cx, data.inertia.cy, data.inertia.cz].flatMap(values)).toEqual(
                expected.inertia.map(Math.fround),
            );
        };
        try {
            setMassData(world, eid, override);
            expect(getMassData(world, eid)).toBeNull();
            world.tick();
            snapshot();
            setMassData(world, eid, { ...override, mass: c === 1 ? 0 : 3 });
            snapshot();
            for (let i = 0; i < 3; i++) {
                world.tick();
                snapshot();
            }
            setMassData(world, eid, zero);
            snapshot();
            world.tick();
            snapshot();
            expect(world.has(eid, GlobalTransform)).toBe(true);
        } finally {
            app.dispose();
        }
    }
    expect(row).toBe(gold.length);
});
