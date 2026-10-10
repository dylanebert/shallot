import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../../engine";
import { GlobalTransform } from "../transform";
import {
    Body,
    DistanceJoint,
    Hulls,
    PhysicsPlugin,
    Shape,
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
        world.add(body, Shape);
        expect(world.has(body, GlobalTransform)).toBe(true);
        const b = world.storage(Body);
        const shape = world.storage(Shape);
        expect(shape.kind.get(body)).toBe(ShapeKind.Hull);
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
            shape.scale.x.get(body),
            shape.scale.y.get(body),
            shape.scale.z.get(body),
            shape.scale.w.get(body),
        ]).toEqual([0.5, 0.5, 0.5, 0]);
        expect(shape.body.get(body)).toBe(0);
        expect(shape.geometry.get(body)).toBe(0);
        expect([
            shape.sphere.x.get(body),
            shape.sphere.y.get(body),
            shape.sphere.z.get(body),
            shape.sphere.w.get(body),
        ]).toEqual([0, 0, 0, 0.5]);
        expect([
            shape.capsuleA.x.get(body),
            shape.capsuleA.y.get(body),
            shape.capsuleA.z.get(body),
            shape.capsuleA.w.get(body),
        ]).toEqual([0, -0.5, 0, 0]);
        expect([
            shape.capsuleB.x.get(body),
            shape.capsuleB.y.get(body),
            shape.capsuleB.z.get(body),
            shape.capsuleB.w.get(body),
        ]).toEqual([0, 0.5, 0, 0.5]);
        expect(shape.density.get(body)).toBe(1000);
        expect(shape.explosionScale.get(body)).toBe(1);
        expect(shape.friction.get(body)).toBe(Math.fround(0.6));
        expect(shape.restitution.get(body)).toBe(0);
        expect(shape.rollingResistance.get(body)).toBe(0);
        expect([
            shape.tangentVelocity.x.get(body),
            shape.tangentVelocity.y.get(body),
            shape.tangentVelocity.z.get(body),
            shape.tangentVelocity.w.get(body),
        ]).toEqual([0, 0, 0, 0]);
        expect(shape.materialUserIdLow.get(body)).toBe(0);
        expect(shape.materialUserIdHigh.get(body)).toBe(0);
        expect(shape.customColor.get(body)).toBe(0);
        expect(shape.filterCategoryLow.get(body)).toBe(0xffffffff);
        expect(shape.filterCategoryHigh.get(body)).toBe(0xffffffff);
        expect(shape.filterMaskLow.get(body)).toBe(0xffffffff);
        expect(shape.filterMaskHigh.get(body)).toBe(0xffffffff);
        expect(shape.filterGroupIndex.get(body)).toBe(0);
        expect(shape.enableCustomFiltering.get(body)).toBe(0);
        expect(shape.isSensor.get(body)).toBe(0);
        expect(shape.enableSensorEvents.get(body)).toBe(0);
        expect(shape.enableContactEvents.get(body)).toBe(0);
        expect(shape.enableHitEvents.get(body)).toBe(0);
        expect(shape.enablePreSolveEvents.get(body)).toBe(0);
        expect(shape.enableSpeculativeContact.get(body)).toBe(1);
        expect(shape.invokeContactCreation.get(body)).toBe(1);
        expect(shape.updateBodyMass.get(body)).toBe(1);
        expect(shape.materialSet.get(body)).toBe(0);
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
