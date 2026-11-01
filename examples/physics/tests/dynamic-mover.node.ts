import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    GlobalTransform,
    hashPhysics,
    MoverJoint,
    Shape,
    Transform,
} from "@dylanebert/shallot";
import { setLinearVelocity, setTransform } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { DynamicMover } from "../src/dynamic-mover";
import { dynamicMoverDrawing } from "../src/dynamic-mover-presentation";
import { DynamicMoverPlugin, DynamicMoverState } from "../src/dynamic-mover-scene";
import { shapeRole } from "../src/presentation";
import { changeControl, controlValues, createScene, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
test("Dynamic Mover takes the sample's defaults and exposes its simulation controls without a camera toggle", async () => {
    const app = await createApp({ defaults: false, plugins: [DynamicMoverPlugin] });
    try {
        const bodies = createScene(app.world, "dynamicMover"),
            s = app.world.resource(DynamicMoverState),
            values = controlValues("dynamicMover");
        expect([...app.world.query([Body])]).toHaveLength(8585);
        expect([...app.world.query([MoverJoint])]).toHaveLength(1);
        expect(s.mover).toMatchObject({
            jumpSpeed: 7,
            maxSpeed: 6,
            stopSpeed: 3,
            accelerate: 20,
            airSteer: 0.25,
            friction: 8,
            gravityScale: 1.5,
            maxGroundForce: 6000,
            maxAirForce: 1500,
            pogoHertz: 3,
            pogoDampingRatio: Math.fround(0.8),
            pogoRestLength: Math.fround(0.9),
            pogoCompressionScale: 100,
            pogoTensionScale: 100,
            enableSkirtPogo: true,
            platformMaterialId: 7n,
        });
        expect(Object.keys(scenes.dynamicMover.controls)).toEqual([
            "Skirt Pogo",
            "Draw Probes",
            "Box Mover",
            "Jump Speed",
            "Max Speed",
            "Accelerate",
            "Friction",
            "Air Steer",
            "Pogo Hertz",
            "Pogo Damping",
            "Ground Force",
            "Gravity Scale",
            "Ball Speed",
            "Snap Position",
        ]);
        changeControl(app.world, "dynamicMover", bodies, values, "Gravity Scale", 2);
        expect(app.world.storage(Body).gravityScale.get(s.mover.body)).toBe(2);
        changeControl(app.world, "dynamicMover", bodies, values, "Box Mover", 1);
        app.world.tick();
        expect(s.useBox).toBe(true);
        expect(s.pendingSwap).toBe(false);
        setTransform(app.world, s.mover.body, [-0.5, 1.5, 0.5], [0, 0, 0, 1]);
        changeControl(app.world, "dynamicMover", bodies, values, "Snap Position", 1);
        app.world.tick();
        const p = app.world.storage(GlobalTransform).translation;
        expect(p.x.get(s.mover.body)).toBe(-1);
        expect(p.z.get(s.mover.body)).toBe(1);
    } finally {
        app.dispose();
    }
});
test("Dynamic Mover's body roles and optional probes use the example's affordances", async () => {
    const app = await createApp({ defaults: false, plugins: [DynamicMoverPlugin] });
    try {
        createScene(app.world, "dynamicMover");
        app.world.tick();
        const s = app.world.resource(DynamicMoverState),
            shapes = app.world.storage(Shape);
        for (const eid of app.world.query([Shape])) {
            const body = shapes.body.get(eid),
                type = app.world.storage(Body).type.get(body);
            expect(shapeRole(app.world, eid, new Set())).toBe(
                shapes.isSensor.get(eid)
                    ? "sensor"
                    : body === s.mover.body || type === BodyType.Kinematic
                      ? "motorized"
                      : type === BodyType.Static
                        ? "ground"
                        : "subject",
            );
        }
        expect(dynamicMoverDrawing(app.world)).toHaveLength(s.mover.probes.length);
        s.drawProbes = false;
        expect(dynamicMoverDrawing(app.world)).toHaveLength(0);
        s.keys.Aim = 1;
        expect(dynamicMoverDrawing(app.world)).toHaveLength(1);
    } finally {
        app.dispose();
    }
});
test("Dynamic Mover does not inherit velocity from a destroyed surface whose entity is reused", async () => {
    const app = await createApp({ defaults: false, plugins: [DynamicMoverPlugin] });
    try {
        createScene(app.world, "dynamicMover");
        const s = app.world.resource(DynamicMoverState);
        s.active = false;
        const surface = app.world.create();
        app.world.add(surface, Transform);
        app.world.add(surface, Body, { type: BodyType.Dynamic });
        app.world.tick();
        s.mover.castResult.body = app.world.ref(surface);
        app.world.destroy(surface);
        const replacement = app.world.create();
        expect(replacement).toBe(surface);
        app.world.add(replacement, Transform);
        app.world.add(replacement, Body, { type: BodyType.Dynamic });
        app.world.tick();
        setLinearVelocity(app.world, replacement, { x: 0, y: 50, z: 0 });
        setLinearVelocity(app.world, s.mover.body, { x: 0, y: 0, z: 0 });
        s.mover.onGround = s.mover.walkable = true;
        expect(s.mover.jump(app.world)).toBe(true);
        expect(app.world.storage(GlobalTransform).linearVelocity.y.get(s.mover.body)).toBeCloseTo(
            7,
            5,
        );
    } finally {
        app.dispose();
    }
});

test("Dynamic Mover restores its vendored class and replays the same next step", async () => {
    const app = await createApp({ defaults: false, plugins: [DynamicMoverPlugin] });
    try {
        createScene(app.world, "dynamicMover");
        for (let i = 0; i < 15; i++) app.world.tick();
        const snapshot = app.world.snapshot();
        app.world.tick();
        const expected = hashPhysics(app.world);
        app.world.resource(DynamicMoverState).mover.pogoHertz = 10;
        app.world.restore(snapshot);
        expect(app.world.resource(DynamicMoverState).mover).toBeInstanceOf(DynamicMover);
        app.world.tick();
        expect(hashPhysics(app.world)).toBe(expected);
    } finally {
        app.dispose();
    }
});
