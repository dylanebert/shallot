import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp, GlobalTransform } from "../../engine";
import { Body, Hulls, Joint, PhysicsPlugin, ShapeKind, Spring, UNIT_CUBE_ID } from "./index";

setDefaultTimeout(CEILING.node);

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
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
        world.add(spring, Spring);
        const s = world.storage(Spring);
        expect(s.a.get(spring)).toBe(0);
        expect(s.b.get(spring)).toBe(0);
        expect([
            s.rA.x.get(spring),
            s.rA.y.get(spring),
            s.rA.z.get(spring),
            s.rA.w.get(spring),
        ]).toEqual([0, 0, 0, 0]);
        expect([
            s.rB.x.get(spring),
            s.rB.y.get(spring),
            s.rB.z.get(spring),
            s.rB.w.get(spring),
        ]).toEqual([0, 0, 0, 0]);
        expect(s.stiffness.get(spring)).toBe(100);
        expect(s.rest.get(spring)).toBe(1);
        const joint = world.create();
        world.add(joint, Joint);
        const j = world.storage(Joint);
        expect(j.a.get(joint)).toBe(0);
        expect(j.b.get(joint)).toBe(0);
        expect([
            j.rA.x.get(joint),
            j.rA.y.get(joint),
            j.rA.z.get(joint),
            j.rA.w.get(joint),
        ]).toEqual([0, 0, 0, 0]);
        expect([
            j.rB.x.get(joint),
            j.rB.y.get(joint),
            j.rB.z.get(joint),
            j.rB.w.get(joint),
        ]).toEqual([0, 0, 0, 0]);
        expect(j.stiffnessAng.get(joint)).toBe(0);
        expect(world.resource(Hulls).id("__unit_cube__")).toBe(UNIT_CUBE_ID);
    } finally {
        app.dispose();
    }
});
