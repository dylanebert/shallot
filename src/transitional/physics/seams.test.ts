import { expect, test } from "bun:test";
import { BodyType, makeBoxHull, PhysicsWorld, type Pos } from "./api";

const identity = (x = 0, y = 0, z = 0) => ({
    p: { x, y, z },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});

function dynamicBox(physicsWorld: PhysicsWorld, position: Pos) {
    const body = physicsWorld.createBody({ type: BodyType.Dynamic, position });
    const shape = body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
    return { body, shape };
}

test("World.getGravity fills a supplied vector so a stepped consumer can read gravity without allocating", () => {
    const physicsWorld = new PhysicsWorld({ gravity: { x: 1, y: -10, z: 2 } });
    const out = { x: 0, y: 0, z: 0 };
    expect(physicsWorld.getGravity(out)).toBe(out);
    expect(out).toEqual({ x: 1, y: -10, z: 2 });
    physicsWorld.destroy();
});

test("a contact begin event omits its solved normal impulse, so an impact-driven consumer cannot distinguish a forceful contact from a grazing one", () => {
    const physicsWorld = new PhysicsWorld({
        gravity: { x: 0, y: -10, z: 0 },
        enableContinuous: false,
    });
    const floor = physicsWorld.createBody({ position: { x: 0, y: -0.5, z: 0 } });
    floor.createHull({}, makeBoxHull(10, 0.5, 10));
    const falling = dynamicBox(physicsWorld, { x: 0, y: 3, z: 0 });
    falling.shape.enableContactEvents(true);

    let begin: ReturnType<PhysicsWorld["getContactEvents"]>["beginEvents"][number] | undefined;
    for (let i = 0; i < 120 && !begin; ++i) {
        physicsWorld.step(1 / 60, 4);
        begin = physicsWorld.getContactEvents().beginEvents[0];
    }
    expect(begin).toBeDefined();
    expect(begin!.normalImpulse).toBeGreaterThan(0);
    physicsWorld.destroy();
});

test("a joint over its configured break threshold produces no public event, so a breakable constraint cannot react to overload", () => {
    const physicsWorld = new PhysicsWorld({
        gravity: { x: 0, y: -10, z: 0 },
        enableContinuous: false,
    });
    const anchor = physicsWorld.createBody({ position: { x: 0, y: 5, z: 0 } });
    const arm = dynamicBox(physicsWorld, { x: 1, y: 5, z: 0 }).body;
    const joint = physicsWorld.createRevoluteJoint(anchor, arm, {
        localFrameA: identity(),
        localFrameB: identity(-1, 0, 0),
        forceThreshold: 0,
    });
    physicsWorld.step(1 / 60, 4);
    const events = physicsWorld.getJointEvents();
    expect(
        events.some((event) => event.joint.isValid() && event.joint.getType() === joint.getType()),
    ).toBe(true);
    physicsWorld.destroy();
});

test("a soft joint requires a fabricated user body for its fixed endpoint, so a grab anchor cannot move in world space through the public API", () => {
    const physicsWorld = new PhysicsWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableContinuous: false,
    });
    const { body } = dynamicBox(physicsWorld, { x: 2, y: 0, z: 0 });
    const joint = physicsWorld.createSoftJoint(
        body,
        { x: 0, y: 0, z: 0 },
        { hertz: 8, dampingRatio: 0.7 },
    );
    joint.setAnchor({ x: 0, y: 1, z: 0 });
    for (let i = 0; i < 30; ++i) physicsWorld.step(1 / 60, 4);
    expect(joint.getAnchor()).toEqual({ x: 0, y: 1, z: 0 });
    expect(body.getPosition().x).toBeLessThan(2);
    joint.destroy();
    expect(joint.isValid()).toBe(false);
    physicsWorld.destroy();
});

test("a revolute motor speed, wheel steering target or wheel spin motor target set on a sleeping island leaves it asleep, so the first input after a parked vehicle settles is lost", () => {
    const wheelPair = (physicsWorld: PhysicsWorld) => {
        const chassis = physicsWorld.createBody({ type: BodyType.Dynamic });
        chassis.createHull({ density: 1 }, makeBoxHull(1, 0.25, 0.5));
        const wheel = physicsWorld.createBody({ type: BodyType.Dynamic });
        wheel.createHull({ density: 1 }, makeBoxHull(0.25, 0.25, 0.5));
        return { chassis, wheel };
    };
    const cases = [
        {
            name: "revolute motor speed",
            arrange(physicsWorld: PhysicsWorld) {
                const anchor = physicsWorld.createBody({ position: { x: 0, y: 0, z: 0 } });
                const { body } = dynamicBox(physicsWorld, { x: 1, y: 0, z: 0 });
                const joint = physicsWorld.createRevoluteJoint(anchor, body, {
                    localFrameA: identity(),
                    localFrameB: identity(-1, 0, 0),
                    enableMotor: true,
                    maxMotorTorque: 100000,
                });
                return {
                    sleepers: [body],
                    set: () => joint.setMotorSpeed(4),
                    responded: () => Math.abs(joint.getAngle()) > 0.05,
                };
            },
        },
        {
            name: "wheel steering target",
            arrange(physicsWorld: PhysicsWorld) {
                const { chassis, wheel } = wheelPair(physicsWorld);
                const joint = physicsWorld.createWheelJoint(chassis, wheel, {
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
            arrange(physicsWorld: PhysicsWorld) {
                const { chassis, wheel } = wheelPair(physicsWorld);
                const joint = physicsWorld.createWheelJoint(chassis, wheel, {
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
        const physicsWorld = new PhysicsWorld({
            gravity: { x: 0, y: 0, z: 0 },
            enableContinuous: false,
        });
        const { sleepers, set, responded } = c.arrange(physicsWorld);
        for (const body of sleepers) body.setAwake(false);
        expect(sleepers[0].isAwake(), `${c.name}: asleep before`).toBe(false);
        set();
        expect(sleepers[0].isAwake(), `${c.name}: awake after`).toBe(true);
        for (let i = 0; i < 10; ++i) physicsWorld.step(1 / 60, 8);
        expect(responded(), `${c.name}: joint responds`).toBe(true);
        physicsWorld.destroy();
    }
});
