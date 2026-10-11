import { expect, setDefaultTimeout, test } from "bun:test";
import { GlobalTransform, Time, Transform } from "@dylanebert/shallot";
import {
    Body,
    BodyType,
    DistanceJoint,
    Shape,
    ShapeKind,
    SphericalJoint,
} from "@dylanebert/shallot/physics";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { createTestGpuApp } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { CEILING } from "../../../scripts/test-tiers";
import { physicsBodyIndex, physicsState } from "./runtime";
import { worldCounters } from "./world/world";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("recycled Shape geometry retries and placement warnings preserve restored bindings", async () => {
    const app = await createTestGpuApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Hull, geometry: 999, scale: [1, 1, 1, 0] });
        world.add(eid, Transform);
        expectWarning(new RegExp(`physics-sync: entity ${eid} carries both Body and Transform`));
        expectWarning(
            new RegExp(`^\\[physics\\] Shape ${eid} was not attached:.*missing hull 999`),
        );
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(worldCounters(physicsState(world)!).bodyCount).toBe(1);
        expect(worldCounters(physicsState(world)!).shapeCount).toBe(0);
        const failed = world.snapshot();
        world.destroy(eid);
        expect(world.create()).toBe(eid);
        world.add(eid, Body);
        world.add(eid, Transform);
        world.restore(failed);
        world.step(Time.FIXED_DT);
        world.destroy(eid);
        expect(world.create()).toBe(eid);
        world.add(eid, Body);
        world.add(eid, Transform);
        expectWarning(new RegExp(`physics-sync: entity ${eid} carries both Body and Transform`));
        world.step(Time.FIXED_DT);
        const solver = physicsState(world)!;
        expect(physicsBodyIndex(world, eid)).toBeGreaterThanOrEqual(0);
        const live = world.snapshot();
        world.destroy(eid);
        expect(world.create()).toBe(eid);
        world.add(eid, Body, { position: [0, 7, 0, 0] });
        world.restore(live);
        expect(physicsBodyIndex(world, eid)).toBeGreaterThanOrEqual(0);
        world.step(Time.FIXED_DT);
        expect(world.storage(GlobalTransform).translation.y.get(eid)).toBe(0);
        expect(worldCounters(solver).bodyCount).toBe(1);
    } finally {
        app.dispose();
    }
});

for (const constraint of [SphericalJoint, DistanceJoint]) {
    test(`${constraint === SphericalJoint ? "a spherical joint" : "a distance joint"} constrains nothing when its endpoint is destroyed and recycled in one update`, async () => {
        const app = await createTestGpuApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        try {
            const world = app.world;
            const anchor = world.create();
            world.add(anchor, Body, { position: [0, 3, 0, 0] });
            const bob = world.create();
            world.add(bob, Body, { type: BodyType.Dynamic, position: [0, 1, 0, 0] });
            const link = world.create();
            world.add(link, constraint, { a: anchor, b: bob });
            world.step(Time.FIXED_DT);
            const solver = physicsState(world)!;
            expect(worldCounters(solver).jointCount).toBe(1);
            world.destroy(bob);
            const replacement = world.create();
            expect(replacement).toBe(bob);
            world.add(replacement, Body, { type: BodyType.Dynamic, position: [0, 1, 0, 0] });
            expect(physicsBodyIndex(world, replacement)).toBe(-1);
            expectWarning(new RegExp(`^\\[physics\\] \\w+Joint ${link} endpoint unavailable`));
            world.step(Time.FIXED_DT);
            expect(physicsBodyIndex(world, replacement)).toBeGreaterThanOrEqual(0);
            expect(worldCounters(solver).jointCount).toBe(0);
        } finally {
            app.dispose();
        }
    });
}
