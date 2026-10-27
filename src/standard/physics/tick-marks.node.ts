import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import type { System } from "../../engine";
import { createApp } from "../../engine/app";
import { StandardPhysicsPlugin } from ".";
import { defaultSurfaceMaterial } from "./common/types";
import { BodyField, bodyField } from "./kernel/bodyrecords";
import { physicsBodyIndex, physicsState, StepPhysicsSystem } from "./runtime";
import { getShapeMaterial, readShapeSphere } from "./shapes/shape";

setDefaultTimeout(TIMEOUT.node);

for (const kind of ["scale", "teleport", "body"] as const)
    test(`a ${kind} write after the physics step survives CPU tick mark clearing and reaches the kernel`, async () => {
        let eid = -1;
        let pending = false;
        const writer: System = {
            name: "late-writer",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                if (!pending) return;
                pending = false;
                if (kind === "scale") world.storage(Transform).scale.set(eid, 3, 3, 3);
                else if (kind === "teleport")
                    world.storage(Transform).translation.set(eid, 5, 0, 0);
                else world.storage(Body).gravityScale.set(eid, 1);
            },
        };
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin, { name: "LateWriter", systems: [writer] }],
        });
        try {
            const world = app.world;
            eid = world.create();
            world.add(eid, Transform);
            world.add(eid, Body, { type: BodyType.Dynamic, gravityScale: 0 });
            world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
            world.tick();
            pending = true;
            world.tick();
            for (const entry of world.storageEntries())
                for (const field of entry.fields.values())
                    expect(field.dirty.every((word) => word === 0)).toBe(true);
            world.tick();
            const state = physicsState(world)!;
            if (kind === "scale") {
                const id = bodyField(state, physicsBodyIndex(world, eid), BodyField.headShapeId);
                expect(readShapeSphere(state, id).radius).toBe(1.5);
            } else if (kind === "teleport") {
                expect(world.storage(GlobalTransform).translation.x.get(eid)).toBeCloseTo(5);
            } else expect(world.storage(GlobalTransform).translation.y.get(eid)).toBeLessThan(0);
        } finally {
            app.dispose();
        }
    });

test("a Shape write after the physics step of the tick that creates it reaches the kernel the next tick", async () => {
    let eid = -1;
    let written = false;
    const writer: System = {
        name: "same-tick-writer",
        group: "fixed",
        after: [StepPhysicsSystem],
        update(world) {
            if (written) return;
            written = true;
            // The spawn's friction mark is still set: only the write's change tick tells it apart.
            world.storage(Shape).friction.set(eid, 0.875);
        },
    };
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, { name: "SameTickWriter", systems: [writer] }],
    });
    try {
        const world = app.world;
        eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5, friction: 0.25 });
        world.tick();
        const state = physicsState(world)!;
        const id = bodyField(state, physicsBodyIndex(world, eid), BodyField.headShapeId);
        const material = { ...defaultSurfaceMaterial() };
        expect(getShapeMaterial(state, id, material).friction).toBe(0.25);
        world.tick();
        expect(getShapeMaterial(state, id, material).friction).toBe(0.875);
    } finally {
        app.dispose();
    }
});

test("after restoring an earlier image, its pending mark and a later write each reach the kernel", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5, friction: 0.25 });
        world.tick();
        const friction = world.storage(Shape).friction;
        friction.set(eid, 0.5);
        const saved = world.snapshot();
        // Captures advance their last tick past everything the image holds.
        for (let i = 0; i < 3; i++) {
            friction.set(eid, 0.625 + i / 8);
            world.tick();
        }
        world.restore(saved);
        world.tick();
        const state = physicsState(world)!;
        const id = bodyField(state, physicsBodyIndex(world, eid), BodyField.headShapeId);
        const material = { ...defaultSurfaceMaterial() };
        expect(getShapeMaterial(state, id, material).friction).toBe(0.5);
        friction.set(eid, 0.125);
        world.tick();
        expect(getShapeMaterial(state, id, material).friction).toBe(0.125);
    } finally {
        app.dispose();
    }
});
