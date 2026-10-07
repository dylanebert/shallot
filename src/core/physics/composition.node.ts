import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp, GlobalTransform } from "../../engine";
import {
    Body,
    DistanceJoint,
    Hulls,
    PhysicsPlugin,
    ShapeKind,
    SphericalJoint,
    UNIT_CUBE_ID,
} from "./index";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("core PhysicsPlugin builds without a solver and accepts shared components with defaults", async () => {
    const app = await createApp({ defaults: false, plugins: [PhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body);
        expect(world.has(body, GlobalTransform)).toBe(true);
        const b = world.storage(Body);
        expect(b.shape.get(body)).toBe(ShapeKind.Box);
        expect([
            b.position.x.get(body),
            b.position.y.get(body),
            b.position.z.get(body),
            b.position.w.get(body),
        ]).toEqual([0, 0, 0, 0]);
        expect([
            b.rotation.x.get(body),
            b.rotation.y.get(body),
            b.rotation.z.get(body),
            b.rotation.w.get(body),
        ]).toEqual([0, 0, 0, 1]);
        expect([
            b.halfExtents.x.get(body),
            b.halfExtents.y.get(body),
            b.halfExtents.z.get(body),
            b.halfExtents.w.get(body),
        ]).toEqual([0.5, 0.5, 0.5, 0]);
        expect(b.mass.get(body)).toBe(1);
        expect(b.friction.get(body)).toBe(0.5);
        const spring = world.create();
        world.add(spring, DistanceJoint);
        const s = world.storage(DistanceJoint);
        expect(s.a.get(spring)).toBe(0);
        expect(s.b.get(spring)).toBe(0);
        expect([
            s.localAnchorA.x.get(spring),
            s.localAnchorA.y.get(spring),
            s.localAnchorA.z.get(spring),
            s.localAnchorA.w.get(spring),
        ]).toEqual([0, 0, 0, 0]);
        expect([
            s.localAnchorB.x.get(spring),
            s.localAnchorB.y.get(spring),
            s.localAnchorB.z.get(spring),
            s.localAnchorB.w.get(spring),
        ]).toEqual([0, 0, 0, 0]);
        expect(s.hertz.get(spring)).toBe(0);
        expect(s.length.get(spring)).toBe(1);
        const joint = world.create();
        world.add(joint, SphericalJoint);
        const j = world.storage(SphericalJoint);
        expect(j.a.get(joint)).toBe(0);
        expect(j.b.get(joint)).toBe(0);
        expect([
            j.localAnchorA.x.get(joint),
            j.localAnchorA.y.get(joint),
            j.localAnchorA.z.get(joint),
            j.localAnchorA.w.get(joint),
        ]).toEqual([0, 0, 0, 0]);
        expect([
            j.localAnchorB.x.get(joint),
            j.localAnchorB.y.get(joint),
            j.localAnchorB.z.get(joint),
            j.localAnchorB.w.get(joint),
        ]).toEqual([0, 0, 0, 0]);
        expect(j.enableSpring.get(joint)).toBe(0);
        expect(j.localRotationB.w.get(joint)).toBe(1);
        expect(world.resource(Hulls).id("__unit_cube__")).toBe(UNIT_CUBE_ID);
    } finally {
        app.dispose();
    }
});
