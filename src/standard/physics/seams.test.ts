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
    "a joint setter wakes its sleeping island and the joint responds",
    {
        claim: "a revolute motor speed, wheel steering target or wheel spin motor target set on a sleeping island leaves it asleep, so the first input after a parked vehicle settles is lost",
    },
    () => {
        const wheelPair = (world: World) => {
            const chassis = world.createBody({ type: BodyType.Dynamic });
            chassis.createHull({ density: 1 }, makeBoxHull(1, 0.25, 0.5));
            const wheel = world.createBody({ type: BodyType.Dynamic });
            wheel.createHull({ density: 1 }, makeBoxHull(0.25, 0.25, 0.5));
            return { chassis, wheel };
        };
        const cases = [
            {
                name: "revolute motor speed",
                arrange(world: World) {
                    const anchor = world.createBody({ position: { x: 0, y: 0, z: 0 } });
                    const { body } = dynamicBox(world, { x: 1, y: 0, z: 0 });
                    const joint = world.createRevoluteJoint(anchor, body, {
                        localFrameA: identity(),
                        localFrameB: identity(-1, 0, 0),
                        enableMotor: true,
                        maxMotorTorque: 10,
                    });
                    return {
                        sleepers: [body],
                        set: () => joint.setMotorSpeed(4),
                        responded: () => true,
                    };
                },
            },
            {
                name: "wheel steering target",
                arrange(world: World) {
                    const { chassis, wheel } = wheelPair(world);
                    const joint = world.createWheelJoint(chassis, wheel, {
                        enableSteering: true,
                        steeringHertz: 10,
                        steeringDampingRatio: 0.5,
                        maxSteeringTorque: 100,
                    });
                    return {
                        sleepers: [chassis, wheel],
                        set: () => joint.setTargetSteeringAngle(0.5),
                        responded: () => joint.getSteeringAngle() > 0.1,
                    };
                },
            },
            {
                name: "wheel spin motor target",
                arrange(world: World) {
                    const { chassis, wheel } = wheelPair(world);
                    const joint = world.createWheelJoint(chassis, wheel, {
                        enableSpinMotor: true,
                        spinSpeed: 10,
                        maxSpinTorque: 100,
                    });
                    return {
                        sleepers: [chassis, wheel],
                        set: () => joint.setSpinMotorSpeed(10),
                        responded: () => joint.getSpinSpeed() > 1,
                    };
                },
            },
        ];
        for (const c of cases) {
            const world = new World({ gravity: { x: 0, y: 0, z: 0 }, enableContinuous: false });
            const { sleepers, set, responded } = c.arrange(world);
            for (const body of sleepers) body.setAwake(false);
            expect(sleepers[0].isAwake(), `${c.name}: asleep before`).toBe(false);
            set();
            expect(sleepers[0].isAwake(), `${c.name}: awake after`).toBe(true);
            for (let i = 0; i < 10; ++i) world.step(1 / 60, 8);
            expect(responded(), `${c.name}: joint responds`).toBe(true);
            world.destroy();
        }
    },
);
