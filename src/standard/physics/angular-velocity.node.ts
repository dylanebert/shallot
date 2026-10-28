import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine";
import gold from "./angular-velocity.gold.json";
import { getAngularVelocity, StandardPhysicsPlugin, setAngularVelocity } from "./index";

setDefaultTimeout(TIMEOUT.node);
test("current angular velocity equals native after stepping, setting and snapshot restore", async () => {
    for (let c = 0; c < 6; c++) {
        let eid = 0;
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                eid = world.create();
                world.add(eid, Transform);
                world.add(eid, Body, {
                    type:
                        c === 0 ? BodyType.Static : c === 1 ? BodyType.Kinematic : BodyType.Dynamic,
                    angularVelocity: [1, -1, 2],
                    motionLocks: c === 3 ? 56 : 0,
                    isAwake: Number(c !== 4),
                    isEnabled: Number(c !== 5),
                });
                world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 1 });
            },
        });
        const world = app.world;
        function check(row: number) {
            const [x, y, z] = gold[c * 3 + row]!.map(Math.fround);
            expect(getAngularVelocity(world, eid), `case ${c}, row ${row}`).toEqual({ x, y, z });
        }
        try {
            expect(getAngularVelocity(world, eid)).toBeNull();
            world.tick();
            check(0);
            const saved = world.snapshot();
            setAngularVelocity(world, eid, { x: 3, y: -2, z: 4 });
            check(1);
            world.tick();
            check(2);
            world.restore(saved);
            check(0);
            setAngularVelocity(world, eid, { x: 3, y: -2, z: 4 });
            check(1);
            world.tick();
            check(2);
            world.destroy(eid);
            expect(getAngularVelocity(world, eid)).toBeNull();
        } finally {
            app.dispose();
        }
    }
});
