import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind, WeldJoint } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import {
    applyAngularImpulse,
    applyLinearImpulse,
    applyLinearImpulseToCenter,
    defaultExplosionDef,
    explode,
    getAngularVelocity,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    setAngularVelocity,
    setAwake,
    setLinearVelocity,
} from "./index";
import { physicsBodyIndex, physicsState } from "./runtime";
import { bodyGetLinearVelocity } from "./world/body";

setDefaultTimeout(TIMEOUT.node);
function published(world: import("../../engine").World, eid: number) {
    const v = world.storage(GlobalTransform).linearVelocity;
    return { x: v.x.get(eid), y: v.y.get(eid), z: v.z.get(eid) };
}
function solver(world: import("../../engine").World, eid: number) {
    return bodyGetLinearVelocity(physicsState(world)!, physicsBodyIndex(world, eid), {
        x: 0,
        y: 0,
        z: 0,
    });
}
function ball(world: import("../../engine").World, x: number, type: number = BodyType.Dynamic) {
    const eid = world.create();
    world.add(eid, Transform, { translation: [x, 0, 0] });
    world.add(eid, Body, { type });
    world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
    return eid;
}
async function app() {
    return createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        },
    });
}
for (const [name, impulse] of [
    [
        "center impulse",
        (world, eid) => applyLinearImpulseToCenter(world, eid, { x: 1000, y: 500, z: -250 }),
    ],
    [
        "off-center impulse",
        (world, eid) =>
            applyLinearImpulse(world, eid, { x: 1000, y: 500, z: -250 }, { x: 0, y: 1, z: 0 }),
    ],
] as const satisfies readonly [
    string,
    (world: import("../../engine").World, eid: number) => void,
][]) {
    test(`${name} immediately publishes the solver's linear velocity without stepping`, async () => {
        const a = await app();
        try {
            const dynamic = ball(a.world, 0),
                fixed = ball(a.world, 4, BodyType.Static);
            a.world.tick();
            const unbound = ball(a.world, 8);
            impulse(a.world, unbound);
            impulse(a.world, fixed);
            expect(published(a.world, fixed)).toEqual({ x: 0, y: 0, z: 0 });
            expect(a.world.storage(Body).linearVelocity.x.get(unbound)).toBe(0);
            impulse(a.world, dynamic);
            const current = solver(a.world, dynamic);
            expect(current.x).toBeGreaterThan(0);
            expect(published(a.world, dynamic)).toEqual(current);
        } finally {
            a.dispose();
        }
    });
}
test("explode immediately publishes every affected bound body's solver velocity", async () => {
    const a = await app();
    try {
        const bodies = [-2, 2].map((x) => ball(a.world, x));
        const fixed = ball(a.world, 5, BodyType.Static);
        a.world.tick();
        const unbound = ball(a.world, 0);
        explode(a.world, { ...defaultExplosionDef(), radius: 8, impulsePerArea: 1000 });
        for (const eid of bodies) {
            const current = solver(a.world, eid);
            expect(Math.abs(current.x)).toBeGreaterThan(0);
            expect(published(a.world, eid)).toEqual(current);
        }
        expect(published(a.world, fixed)).toEqual({ x: 0, y: 0, z: 0 });
        expect(a.world.storage(Body).linearVelocity.x.get(unbound)).toBe(0);
    } finally {
        a.dispose();
    }
});
test("setAwake immediately publishes zero velocity for the whole sleeping island", async () => {
    const a = await app();
    try {
        const left = ball(a.world, -2),
            right = ball(a.world, 2);
        a.world.add(a.world.create(), WeldJoint, {
            a: left,
            b: right,
            localAnchorA: [2, 0, 0],
            localAnchorB: [-2, 0, 0],
        });
        a.world.tick();
        for (const eid of [left, right]) setLinearVelocity(a.world, eid, { x: 3, y: 2, z: 1 });
        setAwake(a.world, left, false);
        for (const eid of [left, right]) {
            expect(solver(a.world, eid)).toEqual({ x: 0, y: 0, z: 0 });
            expect(published(a.world, eid)).toEqual(solver(a.world, eid));
        }
    } finally {
        a.dispose();
    }
});
for (const [name, change] of [
    ["angular impulse", (world, eid) => applyAngularImpulse(world, eid, { x: 100, y: 50, z: 25 })],
    ["angular velocity", (world, eid) => setAngularVelocity(world, eid, { x: 3, y: 2, z: 1 })],
] as const satisfies readonly [
    string,
    (world: import("../../engine").World, eid: number) => void,
][]) {
    test(`${name} changes no published linear velocity`, async () => {
        const a = await app();
        try {
            const eid = ball(a.world, 0);
            a.world.tick();
            setLinearVelocity(a.world, eid, { x: 1, y: 2, z: 3 });
            change(a.world, eid);
            expect(getAngularVelocity(a.world, eid)!.x).toBeGreaterThan(0);
            expect(published(a.world, eid)).toEqual({ x: 1, y: 2, z: 3 });
            expect(published(a.world, eid)).toEqual(solver(a.world, eid));
        } finally {
            a.dispose();
        }
    });
}
