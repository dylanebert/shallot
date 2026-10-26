import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, RevoluteJoint, Shape, SphericalJoint } from "../../core/physics";
import { createApp, type World } from "../../engine";
import * as marshal from "./marshal";
import * as shapes from "./marshal-shape";
import { StandardPhysicsPlugin } from "./runtime";
import * as revolute from "./solver/revoluteJoint";
import * as spherical from "./solver/sphericalJoint";

setDefaultTimeout(TIMEOUT.node);

test("fixed sync preserves cross-kind authoring order in setup, after eid reuse and after restore", async () => {
    const calls: string[] = [];
    const body = marshal.marshalBody;
    const shape = shapes.marshalShape;
    const hinge = revolute.createRevoluteJoint;
    const ball = spherical.createSphericalJoint;
    const spies = [
        spyOn(marshal, "marshalBody").mockImplementation((...args) => {
            calls.push(`body ${args[2]}`);
            return body(...args);
        }),
        spyOn(shapes, "marshalShape").mockImplementation((...args) => {
            calls.push(`shape ${args[3]}`);
            return shape(...args);
        }),
        spyOn(revolute, "createRevoluteJoint").mockImplementation((...args) => {
            calls.push("revolute");
            return hinge(...args);
        }),
        spyOn(spherical, "createSphericalJoint").mockImplementation((...args) => {
            calls.push("spherical");
            return ball(...args);
        }),
    ];
    let anchor = 0;
    let first = 0;
    const expected: string[] = [];
    const joints: number[] = [];
    function author(world: World): void {
        first = world.create();
        world.add(first, Body, { type: BodyType.Dynamic });
        world.add(first, Shape);
        expected.push(`body ${first}`, `shape ${first}`);
        const joint = world.create();
        joints.push(joint);
        world.add(joint, RevoluteJoint, { a: anchor, b: first });
        expected.push("revolute");
        const second = world.create();
        world.add(second, Body, { type: BodyType.Dynamic });
        world.add(second, Shape);
        expected.push(`body ${second}`, `shape ${second}`);
        const other = world.create();
        joints.push(other);
        world.add(other, SphericalJoint, { a: first, b: second });
        expected.push("spherical");
    }
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            anchor = world.create();
            world.add(anchor, Body);
            expected.push(`body ${anchor}`);
            author(world);
        },
    });
    try {
        const world = app.world;
        world.tick();
        expect(calls).toEqual(expected);
        const recycled = first;
        for (const joint of joints) world.destroy(joint);
        joints.length = 0;
        world.destroy(recycled);
        calls.length = expected.length = 0;
        author(world);
        expect(first).toBe(recycled);
        const pending = world.snapshot();
        world.tick();
        expect(calls).toEqual(expected);
        calls.length = 0;
        world.restore(pending);
        world.tick();
        expect(calls).toEqual(expected);

        calls.length = 0;
        world.remove(first, Body);
        world.add(first, Body, { type: BodyType.Dynamic });
        world.tick();
        expect(calls).toContain(`body ${first}`);
        expect(calls.filter((call) => call === `body ${first}`)).toHaveLength(1);

        calls.length = 0;
        world.remove(first, Shape);
        world.add(first, Shape);
        world.tick();
        expect(calls).toEqual([`shape ${first}`]);

        calls.length = 0;
        world.remove(joints[0]!, RevoluteJoint);
        world.add(joints[0]!, RevoluteJoint, { a: anchor, b: first });
        world.tick();
        expect(calls).toEqual(["revolute"]);
    } finally {
        await app.dispose();
        for (const spy of spies) spy.mockRestore();
    }
});
