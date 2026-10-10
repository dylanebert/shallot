import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { createApp, Time, Transform } from "@dylanebert/shallot";
import {
    Body,
    BodyType,
    DistanceJoint,
    Shape,
    ShapeKind,
    SphericalJoint,
} from "@dylanebert/shallot/physics";
import { physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("recycled Shape geometry retries and placement warnings preserve restored bindings", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const warning = spyOn(console, "warn").mockImplementation(() => {});
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Hull, geometry: 999, scale: [1, 1, 1, 0] });
        world.add(eid, Transform);
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(warning).toHaveBeenCalledTimes(2);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(0);
        const failed = world.snapshot();
        world.destroy(eid);
        expect(world.create()).toBe(eid);
        world.add(eid, Body);
        world.add(eid, Transform);
        world.restore(failed);
        world.step(Time.FIXED_DT);
        expect(warning).toHaveBeenCalledTimes(2);
        world.destroy(eid);
        expect(world.create()).toBe(eid);
        world.add(eid, Body);
        world.add(eid, Transform);
        world.step(Time.FIXED_DT);
        expect(warning).toHaveBeenCalledTimes(3);
        const solver = physicsWorld(world)!;
        expect(solver.getBody(eid)).not.toBeNull();
        const live = world.snapshot();
        world.destroy(eid);
        expect(world.create()).toBe(eid);
        world.add(eid, Body, { position: [0, 7, 0, 0] });
        world.restore(live);
        expect(solver.getBody(eid)).not.toBeNull();
        world.step(Time.FIXED_DT);
        const position = { x: 0, y: 0, z: 0 };
        solver.getBody(eid)!.getPosition(position);
        expect(position.y).toBe(0);
        expect(solver.getCounters().bodyCount).toBe(1);
    } finally {
        warning.mockRestore();
        app.dispose();
    }
});

for (const constraint of [SphericalJoint, DistanceJoint]) {
    test(`${constraint === SphericalJoint ? "a spherical joint" : "a distance joint"} constrains nothing when its endpoint is destroyed and recycled in one update`, async () => {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        try {
            const world = app.world;
            const anchor = world.create();
            world.add(anchor, Body, { position: [0, 3, 0, 0] });
            const bob = world.create();
            world.add(bob, Body, { type: BodyType.Dynamic, position: [0, 1, 0, 0] });
            const link = world.create();
            world.add(link, constraint, { a: anchor, b: bob });
            world.step(Time.FIXED_DT);
            const solver = physicsWorld(world)!;
            expect(solver.getCounters().jointCount).toBe(1);
            world.destroy(bob);
            const replacement = world.create();
            expect(replacement).toBe(bob);
            world.add(replacement, Body, { type: BodyType.Dynamic, position: [0, 1, 0, 0] });
            world.step(Time.FIXED_DT);
            expect(solver.getBody(replacement)).not.toBeNull();
            expect(solver.getCounters().jointCount).toBe(0);
        } finally {
            app.dispose();
        }
    });
}
