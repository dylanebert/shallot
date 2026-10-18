import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Hulls, Shape } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { getLocalPoint, getMassData, StandardPhysicsPlugin, setTargetTransform } from ".";

setDefaultTimeout(TIMEOUT.node);

test("mass data reports solver mass, local center and inertia without lending mutable storage", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const geometry = world.resource(Hulls).add({ name: "cube", halfExtents: [0.5, 0.5, 0.5] });
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, { geometry, density: 12 });
        expect(getMassData(world, eid)).toBeNull();
        world.tick();
        const mass = getMassData(world, eid)!;
        expect(mass.mass).toBe(12);
        expect(mass.center).toEqual({ x: 0, y: 0, z: 0 });
        expect(mass.inertia.cx.x).toBe(2);
        expect(mass.inertia.cy.y).toBe(2);
        expect(mass.inertia.cz.z).toBe(2);
        mass.inertia.cx.x = 100;
        expect(getMassData(world, eid)!.inertia.cx.x).toBe(2);
        world.destroy(eid);
        world.tick();
        expect(getMassData(world, eid)).toBeNull();
    } finally {
        app.dispose();
    }
});

test("local point uses the synchronized body's rotated unscaled frame", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Transform, {
            translation: [2, 3, 4],
            rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
            scale: [2, 2, 2],
        });
        world.add(eid, Body, { type: BodyType.Kinematic });
        const point = { x: 2, y: 5, z: 4 };
        expect(getLocalPoint(world, eid, point)).toBeNull();
        world.tick();
        const local = getLocalPoint(world, eid, point)!;
        expect(local.x).toBeCloseTo(2, 6);
        expect(local.y).toBeCloseTo(0, 6);
        expect(local.z).toBe(0);
        local.x = 100;
        expect(getLocalPoint(world, eid, point)!.x).toBeCloseTo(2, 6);
    } finally {
        app.dispose();
    }
});

test("kinematic target velocity uses Box3D's float timestep reciprocal", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Body, { type: BodyType.Kinematic });
        world.tick();
        setTargetTransform(world, eid, [Math.fround(0.05), 0, 0], [0, 0, 0, 1]);
        expect(world.storage(GlobalTransform).linearVelocity.x.get(eid)).toBe(2.999999761581421);
        world.tick();
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(0.04999999701976776);
    } finally {
        app.dispose();
    }
});
