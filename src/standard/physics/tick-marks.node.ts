import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import type { System } from "../../engine";
import { createApp } from "../../engine/app";
import { StandardPhysicsPlugin } from ".";
import { BodyField, bodyField } from "./kernel/bodyrecords";
import { physicsBodyIndex, physicsState, StepPhysicsSystem } from "./runtime";
import { readShapeSphere } from "./shapes/shape";

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
