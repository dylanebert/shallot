import { expect } from "bun:test";
import { check } from "../../harness/check";
import { BodyType, makeBoxHull, type Pos, World } from "./api";

const identity = (x = 0, y = 0, z = 0) => ({
    p: { x, y, z },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});

function dynamicBox(world: World, position: Pos) {
    const body = world.createBody({ type: BodyType.Dynamic, position });
    const shape = body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
    return { body, shape };
}

check(
    "contact begin events carry the solved normal impulse",
    {
        claim: "a contact begin event omits its solved normal impulse, so an impact-driven consumer cannot distinguish a forceful contact from a grazing one",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const floor = world.createBody({ position: { x: 0, y: -0.5, z: 0 } });
        floor.createHull({}, makeBoxHull(10, 0.5, 10));
        const falling = dynamicBox(world, { x: 0, y: 3, z: 0 });
        falling.shape.enableContactEvents(true);

        let begin: ReturnType<World["getContactEvents"]>["beginEvents"][number] | undefined;
        for (let i = 0; i < 120 && !begin; ++i) {
            world.step(1 / 60, 4);
            begin = world.getContactEvents().beginEvents[0];
        }
        expect(begin).toBeDefined();
        expect(begin!.normalImpulse).toBeGreaterThan(0);
        world.destroy();
    },
);

check(
    "joint force thresholds publish joint events",
    {
        claim: "a joint over its configured break threshold produces no public event, so a breakable constraint cannot react to overload",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = dynamicBox(world, { x: 1, y: 5, z: 0 }).body;
        const joint = world.createRevoluteJoint(anchor, arm, {
            localFrameA: identity(),
            localFrameB: identity(-1, 0, 0),
            forceThreshold: 0,
        });
        world.step(1 / 60, 4);
        const events = world.getJointEvents();
        expect(
            events.some(
                (event) => event.joint.isValid() && event.joint.getType() === joint.getType(),
            ),
        ).toBe(true);
        world.destroy();
    },
);

check(
    "a soft joint follows a movable world anchor",
    {
        claim: "a soft joint requires a fabricated user body for its fixed endpoint, so a grab anchor cannot move in world space through the public API",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: 0, z: 0 }, enableContinuous: false });
        const { body } = dynamicBox(world, { x: 2, y: 0, z: 0 });
        const joint = world.createSoftJoint(
            body,
            { x: 0, y: 0, z: 0 },
            { hertz: 8, dampingRatio: 0.7 },
        );
        joint.setAnchor({ x: 0, y: 1, z: 0 });
        for (let i = 0; i < 30; ++i) world.step(1 / 60, 4);
        expect(joint.getAnchor()).toEqual({ x: 0, y: 1, z: 0 });
        expect(body.getPosition().x).toBeLessThan(2);
        joint.destroy();
        expect(joint.isValid()).toBe(false);
        world.destroy();
    },
);

check(
    "setting a motor speed wakes its sleeping island",
    {
        claim: "changing a joint motor speed leaves a sleeping body asleep, so the first input after a parked vehicle settles is ignored",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: 0, z: 0 }, enableContinuous: false });
        const anchor = world.createBody({ position: { x: 0, y: 0, z: 0 } });
        const { body } = dynamicBox(world, { x: 1, y: 0, z: 0 });
        const joint = world.createRevoluteJoint(anchor, body, {
            localFrameA: identity(),
            localFrameB: identity(-1, 0, 0),
            enableMotor: true,
            maxMotorTorque: 10,
        });
        body.setAwake(false);
        expect(body.isAwake()).toBe(false);
        joint.setMotorSpeed(4);
        expect(body.isAwake()).toBe(true);
        world.destroy();
    },
);
