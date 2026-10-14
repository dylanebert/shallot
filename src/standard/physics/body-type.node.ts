import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createTestGpuApp as createApp } from "../../../scripts/gpu.fixture";
import { Body, BodyType, Shape } from "../../core/physics";
import { GlobalTransform } from "../../core/transform";
import { Time } from "../../engine";
import { StandardPhysicsPlugin, setLinearVelocity, setTargetTransform } from ".";

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("a default Body is static, takes no velocity, and refuses target transforms", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const eid = app.world.create();
        app.world.add(eid, Body, { position: [0, 5, 0, 0] });
        app.world.step(Time.FIXED_DT);
        setLinearVelocity(app.world, eid, { x: 1, y: 2, z: 3 });
        app.world.step(Time.FIXED_DT);
        const pose = app.world.storage(GlobalTransform);
        expect([
            pose.linearVelocity.x.get(eid),
            pose.linearVelocity.y.get(eid),
            pose.linearVelocity.z.get(eid),
        ]).toEqual([0, 0, 0]);
        setTargetTransform(app.world, eid, [4, 5, 6], [0, 0, 0, 1]);
        setTargetTransform(app.world, eid, [7, 8, 9], [0, 0, 0, 1]);
        app.world.step(Time.FIXED_DT);
        expect([
            pose.translation.x.get(eid),
            pose.translation.y.get(eid),
            pose.translation.z.get(eid),
        ]).toEqual([0, 5, 0]);
    } finally {
        app.dispose();
    }
});

test("a dynamic Body with zero mass does not fall", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const eid = app.world.create();
        app.world.add(eid, Body, { type: BodyType.Dynamic, position: [0, 5, 0, 0] });
        app.world.add(eid, Shape, { density: 0 });
        app.world.step(Time.FIXED_DT);
        for (let i = 0; i < 10; i++) app.world.step(Time.FIXED_DT);
        expect(app.world.storage(GlobalTransform).translation.y.get(eid)).toBe(5);
    } finally {
        app.dispose();
    }
});

test("body sync publishes rigid pose without collider scale", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const eid = app.world.create();
        app.world.add(eid, Body);
        app.world.step(Time.FIXED_DT);
        const scale = app.world.storage(GlobalTransform).scale;
        expect([scale.x.get(eid), scale.y.get(eid), scale.z.get(eid)]).toEqual([1, 1, 1]);
    } finally {
        app.dispose();
    }
});
