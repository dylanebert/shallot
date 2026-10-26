import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { expectWarning } from "../../../scripts/test-console";
import { Body, RevoluteJoint, Shape, SphericalJoint } from "../../core/physics";
import { createApp } from "../../engine";
import * as authoring from "./authoring";
import * as records from "./kernel/jointrecords";
import * as marshal from "./marshal-shape";
import { physicsState, StandardPhysicsPlugin } from "./runtime";
import * as shapeSync from "./shape-sync";
import { worldCounters } from "./world/world";

setDefaultTimeout(TIMEOUT.node);

test("one bare Body added beside 1000 warm Body+Shape entities visits no unrelated shapes or joints", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const bodies: number[] = [];
    for (let i = 0; i < 1000; i++) {
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape);
        bodies.push(eid);
    }
    world.step(1 / 60);
    world.step(1 / 60);
    const shape = spyOn(shapeSync, "shapeNeedsRecreation");
    const field = spyOn(records, "jointField");
    try {
        const bare = world.create();
        world.add(bare, Body);
        world.step(1 / 60);
        expect(shape).toHaveBeenCalledTimes(0);
        for (const body of bodies) {
            const joint = world.create();
            world.add(joint, RevoluteJoint, { a: body, b: bare });
        }
        world.step(1 / 60);
        world.step(1 / 60);
        shape.mockClear();
        field.mockClear();
        const other = world.create();
        world.add(other, Body);
        world.step(1 / 60);
        expect(shape).toHaveBeenCalledTimes(0);
        expect(field).toHaveBeenCalledTimes(0);
    } finally {
        shape.mockRestore();
        field.mockRestore();
        await app.dispose();
    }
});

test("body re-add visits only its dependents after retargeting and reusable snapshot recovery", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const bodies = Array.from({ length: 3 }, () => {
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape);
        return eid;
    });
    const [a, b, c] = bodies as [number, number, number];
    const child = world.create();
    world.add(child, Shape, { body: a });
    const hinge = world.create();
    world.add(hinge, RevoluteJoint, { a, b: c });
    const ball = world.create();
    world.add(ball, SphericalJoint, { a: b, b: c });
    world.step(1 / 60);
    world.step(1 / 60);
    const shape = spyOn(marshal, "marshalShape");
    const joint = spyOn(authoring, "jointDef");
    const checked = spyOn(shapeSync, "shapeNeedsRecreation");
    function readd(body: number): void {
        shape.mockClear();
        joint.mockClear();
        checked.mockClear();
        world.remove(body, Body);
        world.add(body, Body);
        world.step(1 / 60);
    }
    const shapes = () => shape.mock.calls.map((call) => call[3]);
    const joints = () => joint.mock.calls.map((call) => call[2]);
    try {
        readd(a);
        expect(shapes()).toEqual([a, child]);
        expect(joints()).toEqual([hinge]);
        expect(checked).toHaveBeenCalledTimes(0);
        world.storage(Shape).body.set(child, b);
        world.storage(RevoluteJoint).a.set(hinge, b);
        world.step(1 / 60);
        readd(a);
        expect(shapes()).toEqual([a]);
        expect(joints()).toEqual([]);
        const image = world.snapshot();
        world.storage(Shape).body.set(child, c);
        world.storage(RevoluteJoint).b.set(hinge, a);
        world.step(1 / 60);
        for (let i = 0; i < 2; i++) {
            world.restore(image);
            readd(b);
            // Body destruction makes both shapes new creations, queued in list join order.
            expect(shapes()).toEqual([b, child]);
            expect(joints()).toEqual([hinge, ball]);
            expect(checked).toHaveBeenCalledTimes(0);
        }
    } finally {
        shape.mockRestore();
        joint.mockRestore();
        checked.mockRestore();
        await app.dispose();
    }
});

test("re-adding a Body realizes its separate shapes in dependency join order", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const body = world.create();
    world.add(body, Body);
    const first = world.create();
    world.add(first, Shape, { body });
    const second = world.create();
    world.add(second, Shape, { body });
    world.step(1 / 60);
    world.step(1 / 60);
    const shape = spyOn(marshal, "marshalShape");
    try {
        world.remove(body, Body);
        world.add(body, Body);
        world.step(1 / 60);
        expect(shape.mock.calls.map((call) => call[3])).toEqual([first, second]);
    } finally {
        shape.mockRestore();
        await app.dispose();
    }
});

test("retargets and re-added Shapes join the destination list at its end through restore", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const a = world.create(),
        b = world.create();
    world.add(a, Body);
    world.add(b, Body);
    const first = world.create();
    world.add(first, Shape, { body: a });
    const second = world.create();
    world.add(second, Shape, { body: b });
    world.step(1 / 60);
    world.step(1 / 60);
    const shape = spyOn(marshal, "marshalShape");
    function readd(): number[] {
        shape.mockClear();
        world.remove(b, Body);
        world.add(b, Body);
        world.step(1 / 60);
        return shape.mock.calls.map((call) => call[3]);
    }
    try {
        world.storage(Shape).body.set(first, b);
        world.step(1 / 60);
        world.step(1 / 60);
        expect(readd()).toEqual([second, first]);
        const image = world.snapshot();
        world.storage(Shape).body.set(second, a);
        world.step(1 / 60);
        world.storage(Shape).body.set(second, b);
        world.step(1 / 60);
        world.step(1 / 60);
        expect(readd()).toEqual([first, second]);
        world.remove(first, Shape);
        world.add(first, Shape, { body: b });
        world.step(1 / 60);
        world.step(1 / 60);
        expect(readd()).toEqual([second, first]);
        for (let i = 0; i < 2; i++) {
            world.restore(image);
            expect(readd()).toEqual([second, first]);
        }
    } finally {
        shape.mockRestore();
        await app.dispose();
    }
});

test("adding a missing Body satisfies indexed shape and joint dependencies, including after restore", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    try {
        const anchor = world.create();
        world.add(anchor, Body);
        const target = world.create();
        const child = world.create();
        world.add(child, Shape, { body: target });
        const link = world.create();
        world.add(link, RevoluteJoint, { a: anchor, b: target });
        expectWarning(new RegExp(`^\\[physics\\] RevoluteJoint ${link} endpoint unavailable`));
        world.step(1 / 60);
        expect(worldCounters(physicsState(world)!).shapeCount).toBe(0);
        expect(worldCounters(physicsState(world)!).jointCount).toBe(0);
        const pending = world.snapshot();
        for (let i = 0; i < 2; i++) {
            world.restore(pending);
            world.add(target, Body);
            world.step(1 / 60);
            expect(worldCounters(physicsState(world)!).shapeCount).toBe(1);
            expect(worldCounters(physicsState(world)!).jointCount).toBe(1);
        }
    } finally {
        await app.dispose();
    }
});
