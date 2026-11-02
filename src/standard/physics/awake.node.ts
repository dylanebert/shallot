import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { expectWarning } from "../../../scripts/test-console";
import { Body, BodyType, Shape, ShapeKind, Sleeping } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine";
import {
    isAwake,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    setAwake,
    setLinearVelocity,
} from ".";

import { kernel } from "./kernel/kernel";
import { physicsBodyIndex, physicsState } from "./runtime";
import { destroyBody } from "./world/body";

setDefaultTimeout(TIMEOUT.node);

test("many sleep crossings retain only one pending bound body", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic });
        world.tick();
        for (let i = 0; i < 10000; i++) {
            setAwake(world, body, false);
            setAwake(world, body, true);
        }
        const state = physicsState(world)!;
        const k = kernel(world);
        let count = 0;
        for (
            let eid = k.bodySleepChange(state.worldId);
            eid !== -1;
            eid = k.bodySleepChange(state.worldId)
        ) {
            expect(eid).toBe(body);
            count++;
        }
        expect(count).toBe(1);
        world.tick();
        expect(world.has(body, Sleeping)).toBe(false);
        setAwake(world, body, false);
        world.tick();
        expect(world.has(body, Sleeping)).toBe(true);
    } finally {
        app.dispose();
    }
});

test("destroying a pending body retires its entry before its eid and solver slot are reused", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic });
        world.tick();
        setAwake(world, body, false);
        const state = physicsState(world)!;
        destroyBody(state, physicsBodyIndex(world, body));
        expect(kernel(world).bodySleepChange(state.worldId)).toBe(-1);
        world.destroy(body);
        const replacement = world.create();
        expect(replacement).toBe(body);
        world.add(replacement, Body, { type: BodyType.Dynamic });
        world.tick();
        expect(world.has(replacement, Sleeping)).toBe(false);
        setAwake(world, replacement, false);
        world.tick();
        expect(world.has(replacement, Sleeping)).toBe(true);
        setAwake(world, replacement, true);
        world.tick();
        expect(world.has(replacement, Sleeping)).toBe(false);
    } finally {
        app.dispose();
    }
});

test("Sleeping follows natural sleep, contact wake and restored solver state at fixed sync", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic });
        const shape = world.create();
        world.add(shape, Shape, { body, kind: ShapeKind.Sphere, radius: 0.5 });
        for (let i = 0; i < 120; i++) world.tick();
        expect(isAwake(world, body)).toBe(false);
        expect(world.has(body, Sleeping)).toBe(true);
        expect(world.has(shape, Sleeping)).toBe(true);
        const saved = world.snapshot();
        const other = world.create();
        world.add(other, Transform, { translation: [-2, 0, 0] });
        world.add(other, Body, { type: BodyType.Dynamic });
        world.add(other, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
        world.tick();
        setLinearVelocity(world, other, { x: 5, y: 0, z: 0 });
        for (let i = 0; i < 20; i++) world.tick();
        expect(isAwake(world, body)).toBe(true);
        expect(world.has(body, Sleeping)).toBe(false);
        expect(world.has(shape, Sleeping)).toBe(false);
        world.restore(saved);
        expect(world.has(body, Sleeping)).toBe(true);
        world.tick();
        expect(world.has(body, Sleeping)).toBe(true);
        setAwake(world, body, true);
        const pending = world.snapshot();
        world.tick();
        expect(world.has(body, Sleeping)).toBe(false);
        world.restore(pending);
        expect(world.has(body, Sleeping)).toBe(true);
        expect(world.has(shape, Sleeping)).toBe(true);
        world.tick();
        expect(world.has(body, Sleeping)).toBe(false);
        expect(world.has(shape, Sleeping)).toBe(false);
    } finally {
        app.dispose();
    }
});

test("attached shapes inherit sleep through attachment, wake, rebinding, detachment, destruction and restore", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic });
        const awake = world.create();
        world.add(awake, Body, { type: BodyType.Dynamic });
        world.tick();
        setAwake(world, body, false);
        world.tick();
        const shape = world.create();
        world.add(shape, Shape, { body, kind: ShapeKind.Sphere, radius: 0.5 });
        world.tick();
        expect(world.has(body, Sleeping)).toBe(true);
        expect(world.has(shape, Sleeping)).toBe(true);
        const saved = world.snapshot();
        setAwake(world, body, true);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(false);
        setAwake(world, body, false);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(true);
        world.storage(Shape).body.set(shape, awake);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(false);
        world.storage(Shape).body.set(shape, body);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(true);
        world.storage(Shape).body.set(shape, 0);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(false);
        world.restore(saved);
        expect(world.has(shape, Sleeping)).toBe(true);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(true);
        setAwake(world, body, true);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(false);
        setAwake(world, body, false);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(true);
        world.destroy(body);
        expectWarning(/body reference is no longer valid/);
        world.tick();
        expect(world.has(shape, Sleeping)).toBe(false);
    } finally {
        app.dispose();
    }
});

test("isAwake reads live solver state, not the spawn definition", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic });
        expect(isAwake(world, body)).toBeNull();
        world.tick();
        expect(isAwake(world, body)).toBe(true);
        setAwake(world, body, false);
        expect(isAwake(world, body)).toBe(false);
        expect(world.storage(Body).isAwake.get(body)).toBe(1);
        world.tick();
        expect(world.has(body, Sleeping)).toBe(true);
        setAwake(world, body, true);
        expect(isAwake(world, body)).toBe(true);
        world.tick();
        expect(world.has(body, Sleeping)).toBe(false);
        const ground = world.create();
        world.add(ground, Body, { type: BodyType.Static });
        world.tick();
        expect(world.has(ground, Sleeping)).toBe(false);
        world.destroy(body);
        world.tick();
        expect(isAwake(world, body)).toBeNull();
    } finally {
        app.dispose();
    }
});
