import { expect, setDefaultTimeout, test } from "bun:test";
import { BaseColor, Body, BodyType, createApp, Shape } from "@dylanebert/shallot";
import { StandardPhysicsPlugin, setAwake } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { EventSample } from "../src/events";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { drivenBodies, setReaction, shapeMaterial, shapeRole } from "../src/shape-color";

setDefaultTimeout(TIMEOUT.node);

test("reaction begin and end recolor every attached shape without frame reconciliation", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic, gravityScale: 0 });
        const shapes = [world.create(), world.create()];
        for (const shape of shapes) {
            world.add(shape, Shape, { body });
            world.add(shape, BaseColor, { color: [0.72, 0.4, 0.05, 1] });
        }
        world.tick();
        const reactions = world.resource(EventSample).reactions;
        const ref = world.ref(body);
        setReaction(world, reactions, ref);
        for (const shape of shapes)
            expect(Array.from(world.get(shape, BaseColor)!.color)).toEqual(
                [1, 0.22, 0.035, 1].map(Math.fround),
            );
        setAwake(world, body, false);
        world.tick();
        for (const shape of shapes)
            expect(Array.from(world.get(shape, BaseColor)!.color)).toEqual(
                [1, 0.22, 0.035, 1].map(Math.fround),
            );
        setReaction(world, reactions, ref, false);
        for (const shape of shapes)
            expect(Array.from(world.get(shape, BaseColor)!.color)).toEqual(
                [0.72, 0.4, 0.05, 1].map(Math.fround),
            );
    } finally {
        app.dispose();
    }
});

test("body-type controls update authored instance colors at the control change", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const bodies = createScene(world, "bodyType");
        for (const eid of world.query([Shape]))
            world.add(eid, BaseColor, {
                color: shapeMaterial(world, shapeRole(world, eid, drivenBodies(world)), eid)
                    .baseColor,
            });
        const values = controlValues("bodyType");
        for (const type of [BodyType.Static, BodyType.Kinematic, BodyType.Dynamic]) {
            changeControl(world, "bodyType", bodies, values, "Type", type);
            for (const eid of world.query([Shape, BaseColor])) {
                const role = shapeRole(world, eid, drivenBodies(world));
                expect(Array.from(world.get(eid, BaseColor)!.color)).toEqual(
                    Array.from(shapeMaterial(world, role, eid).baseColor),
                );
            }
        }
    } finally {
        app.dispose();
    }
});
