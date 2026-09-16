// Fast-tier joint behavior (Box3D's test_joint.c essence + the collideConnected filter). The
// bit-exact solve math is covered by step.fixture.ts; here we exercise the public handle lifecycle,
// the joint-connected contact filter, and that each joint type actually constrains its bodies.

import { expect } from "bun:test";
import { check } from "../../../harness/check";
import {
    type Body,
    BodyType,
    type Joint,
    JointType,
    makeBoxHull,
    makeCubeHull,
    type Vec3,
    World,
} from "../api/index";
import { LINEAR_SLOP } from "../common/constants";
import { f32, PI } from "../common/math";

function frame(x: number, y: number, z: number) {
    return { p: { x, y, z }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
}

const len = (v: Vec3): number => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

function pendulum(): { world: World; joint: Joint } {
    const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
    const arm = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
    arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
    const joint = world.createRevoluteJoint(anchor, arm, {
        localFrameA: frame(0, 0, 0),
        localFrameB: frame(-1, 0, 0),
    });
    return { world, joint };
}

check(
    "a recycled joint slot invalidates the stale generation",
    {
        claim: "a joint handle kept past its destroy resolves again once its slot is reused, so a stale reference would silently drive somebody else's constraint",
    },
    () => {
        const { world, joint } = pendulum();
        const anchor = joint.getBodies()[0];
        joint.destroy();
        // Recreate: same slot, bumped generation — the stale handle must not resolve.
        const arm2 = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
        arm2.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createRevoluteJoint(anchor, arm2, { localFrameA: frame(0, 0, 0) });
        expect(joint.isValid()).toBe(false);
        world.destroy();
    },
);

check(
    "collideConnected decides whether jointed bodies also collide",
    {
        claim: "the joint-connected contact filter ignores collideConnected, so two jointed touching bodies would either fight their own contact or pass through each other against the caller's setting",
    },
    () => {
        // Two adjacent dynamic boxes that touch. With collideConnected off (default) the joint must
        // filter the contact between them; on, the contact is created.
        function twoTouching(collide: boolean): number {
            const world = new World({ gravity: { x: 0, y: 0, z: 0 }, enableContinuous: false });
            const a = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
            a.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
            const b = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
            b.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
            world.createRevoluteJoint(a, b, {
                localFrameA: frame(0.5, 0, 0),
                localFrameB: frame(-0.5, 0, 0),
                collideConnected: collide,
            });
            world.step(1 / 60, 4);
            const contacts = world.getCounters().contactCount;
            world.destroy();
            return contacts;
        }

        expect(twoTouching(false)).toBe(0);
        expect(twoTouching(true)).toBe(1);
    },
);

check(
    "a revolute joint holds the arm at its hinge and reacts against gravity",
    {
        claim: "a revolute joint does not hold its hinge point, so a pendulum arm would drift away from its anchor or carry no constraint force",
    },
    () => {
        const { world, joint } = pendulum();
        for (let i = 0; i < 60; ++i) {
            world.step(1 / 60, 4);
        }
        // The arm's pivot (its local (-1,0,0)) must stay near the anchor at (0,5,0): the arm swings
        // down but the hinge point holds, so the arm center stays ~1 unit from the anchor.
        const arm = joint.getBodies()[1];
        const c = arm.getPosition();
        const dist = len({ x: c.x - 0, y: c.y - 5, z: c.z - 0 });
        expect(dist).toBeGreaterThan(0.9);
        expect(dist).toBeLessThan(1.1);
        // Under gravity the hinge carries load, so the constraint force is non-zero.
        const force = len(joint.getConstraintForce());
        expect(force).toBeGreaterThan(0);
        world.destroy();
    },
);

check(
    "a weld joint rigidly holds a box fixed to a static anchor against gravity",
    {
        claim: "a weld joint lets its body sag, so a welded box would fall away from its start pose under gravity",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const box = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        box.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        const joint = world.createWeldJoint(anchor, box, {
            localFrameA: frame(0, -1, 0),
            localFrameB: frame(0, 0, 0),
        });
        for (let i = 0; i < 90; ++i) world.step(1 / 60, 4);
        // Welded to the static anchor, the box hangs at its start pose; the weld carries its weight.
        const p = box.getPosition();
        expect(len({ x: p.x, y: p.y - 4, z: p.z })).toBeLessThan(0.05);
        expect(len(joint.getConstraintForce())).toBeGreaterThan(0);
        world.destroy();
    },
);

check(
    "a parallel joint applies a corrective torque against a body spinning off-axis",
    {
        claim: "a parallel joint carries no constraint torque, so a body spinning off its partner's axis would never be pulled back into alignment",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const a = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        a.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularVelocity: { x: 3, y: 2, z: 0 },
        });
        b.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
        const joint = world.createParallelJoint(a, b, { maxTorque: 5 });
        for (let i = 0; i < 30; ++i) world.step(1 / 60, 4);
        // The joint resists the off-axis spin, so it carries a non-zero constraint torque.
        expect(joint.getType()).toBe(JointType.Parallel);
        expect(len(joint.getConstraintTorque())).toBeGreaterThan(0);
        world.destroy();
    },
);

check(
    "a motor joint drives the arm's spin in the direction of the target angular velocity",
    {
        claim: "a motor joint's target angular velocity does not set which way its body turns, so a driven arm would spin the same way whatever sign the caller asks for",
    },
    () => {
        // The arm's absolute spin mixes the motor drive with the gravity-driven swing, so assert the
        // motor's *effect*: flipping the target's sign flips which way the arm ends up spinning.
        function armSpinZ(targetZ: number): number {
            const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
            const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
            const arm = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 1, y: 5, z: 0 },
            });
            arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
            world.createMotorJoint(anchor, arm, {
                localFrameA: frame(0, 0, 0),
                localFrameB: frame(-1, 0, 0),
                maxVelocityForce: 1000,
                angularVelocity: { x: 0, y: 0, z: targetZ },
                maxVelocityTorque: 500,
            });
            for (let i = 0; i < 60; ++i) world.step(1 / 60, 4);
            const wz = arm.getAngularVelocity().z;
            world.destroy();
            return wz;
        }

        expect(armSpinZ(5)).toBeGreaterThan(armSpinZ(-5));
    },
);

check(
    "a prismatic motor drives the slider to its upper limit while the line holds it off-axis",
    {
        claim: "a prismatic joint ignores its translation limit or lets gravity pull the slider off its axis, so a driven slider would overshoot or drop off its rail",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const slider = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        slider.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        const joint = world.createPrismaticJoint(anchor, slider, {
            enableMotor: true,
            motorSpeed: 3,
            maxMotorForce: 1000,
            enableLimit: true,
            lowerTranslation: -2,
            upperTranslation: 2,
        });
        for (let i = 0; i < 180; ++i) world.step(1 / 60, 4);
        // The motor drives it along body A's x to the upper limit (~2); the point-to-line + rotation
        // constraints hold it on the axis, so gravity never drops it off y = 5.
        const p = slider.getPosition();
        expect(p.x).toBeGreaterThan(1.8);
        expect(p.x).toBeLessThan(2.05);
        expect(Math.abs(p.y - 5)).toBeLessThan(0.1);
        expect(len(joint.getConstraintForce())).toBeGreaterThan(0);
        world.destroy();
    },
);

check(
    "a spherical joint pins the arm's pivot at the anchor as it swings",
    {
        claim: "a spherical joint does not pin its pivot point, so a ball-jointed arm would drift off its socket while it swings",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        const joint = world.createSphericalJoint(anchor, arm, { localFrameB: frame(-1, 0, 0) });
        for (let i = 0; i < 90; ++i) world.step(1 / 60, 4);
        const c = arm.getPosition();
        const dist = len({ x: c.x, y: c.y - 5, z: c.z });
        expect(dist).toBeGreaterThan(0.9);
        expect(dist).toBeLessThan(1.1);
        expect(len(joint.getConstraintForce())).toBeGreaterThan(0);
        world.destroy();
    },
);

check(
    "a wheel joint's spin motor drives the wheel in the target direction",
    {
        claim: "a wheel joint's spin motor ignores the sign of its target speed, so a vehicle wheel would turn the same way in forward and reverse",
    },
    () => {
        function wheelSpin(speed: number): number {
            const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
            const chassis = world.createBody({ position: { x: 0, y: 5, z: 0 } });
            const wheel = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: 5, z: 0 },
            });
            wheel.createHull({}, makeBoxHull(0.3, 0.3, 0.3));
            world.createWheelJoint(chassis, wheel, {
                enableSpinMotor: true,
                spinSpeed: speed,
                maxSpinTorque: 50,
            });
            for (let i = 0; i < 60; ++i) world.step(1 / 60, 4);
            const wz = wheel.getAngularVelocity().z;
            world.destroy();
            return wz;
        }

        expect(wheelSpin(10)).toBeGreaterThan(wheelSpin(-10));
    },
);

check(
    "a filter joint suppresses the contact between its two bodies",
    {
        claim: "a filter joint fails to suppress the contact between its pair, so two overlapping bodies deliberately excluded from each other would still push apart",
    },
    () => {
        // A filter joint carries no constraint; it exists only to suppress the contact between its bodies.
        function twoTouching(filtered: boolean): number {
            const world = new World({ gravity: { x: 0, y: 0, z: 0 }, enableContinuous: false });
            const a = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
            a.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
            const b = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
            b.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
            if (filtered) {
                const joint = world.createFilterJoint(a, b);
                expect(joint.getType()).toBe(JointType.Filter);
            }
            world.step(1 / 60, 4);
            const contacts = world.getCounters().contactCount;
            world.destroy();
            return contacts;
        }

        expect(twoTouching(true)).toBe(0);
        expect(twoTouching(false)).toBe(1);
    },
);

check(
    "a mixed spherical/revolute ragdoll island settles and sleeps as a unit",
    {
        claim: "a multi-joint island never reaches sleep, so a settled ragdoll would keep every one of its bodies awake and burning solver time forever",
    },
    () => {
        // A mixed-joint island (torso + 2 spherical-shouldered arms + 2 revolute-hipped legs) falls
        // onto the ground, settles, and sleeps as a unit.
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const ground = world.createBody({ position: { x: 0, y: -1, z: 0 } });
        ground.createHull({}, makeBoxHull(20, 1, 20));
        const torso = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
            angularDamping: 0.5,
            linearDamping: 0.5,
        });
        torso.createHull({}, makeBoxHull(0.25, 0.5, 0.25));
        const mkLimb = (x: number, y: number, box: [number, number, number]) => {
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x, y, z: 0 },
                angularDamping: 0.5,
                linearDamping: 0.5,
            });
            b.createHull({}, makeBoxHull(box[0], box[1], box[2]));
            return b;
        };
        const armL = mkLimb(-0.75, 3.5, [0.5, 0.125, 0.125]);
        const armR = mkLimb(0.75, 3.5, [0.5, 0.125, 0.125]);
        const legL = mkLimb(-0.25, 2.0, [0.125, 0.5, 0.125]);
        const legR = mkLimb(0.25, 2.0, [0.125, 0.5, 0.125]);
        world.createSphericalJoint(torso, armL, {
            localFrameA: frame(-0.25, 0.5, 0),
            localFrameB: frame(0.5, 0, 0),
        });
        world.createSphericalJoint(torso, armR, {
            localFrameA: frame(0.25, 0.5, 0),
            localFrameB: frame(-0.5, 0, 0),
        });
        world.createRevoluteJoint(torso, legL, {
            localFrameA: frame(-0.25, -0.5, 0),
            localFrameB: frame(0, 0.5, 0),
        });
        world.createRevoluteJoint(torso, legR, {
            localFrameA: frame(0.25, -0.5, 0),
            localFrameB: frame(0, 0.5, 0),
        });
        for (let i = 0; i < 300; ++i) world.step(1 / 60, 4);
        // The whole ragdoll sleeps as one island, so every dynamic body reports not-awake.
        expect(torso.isAwake()).toBe(false);
        expect(armL.isAwake()).toBe(false);
        expect(legR.isAwake()).toBe(false);
        world.destroy();
    },
);

check(
    "a distance joint holds a rigid length between the anchors as the ball swings down",
    {
        claim: "a rigid distance joint lets its length change under load, so a swinging ball on a fixed rope would stretch away from its anchor",
    },
    () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const ball = world.createBody({ type: BodyType.Dynamic, position: { x: 2, y: 5, z: 0 } });
        ball.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
        const joint = world.createDistanceJoint(anchor, ball, { length: 2 });
        for (let i = 0; i < 120; ++i) world.step(1 / 60, 4);
        // The rigid joint pins the ball at length 2 from the anchor regardless of where it swings.
        const p = ball.getPosition();
        const dist = len({ x: p.x - 0, y: p.y - 5, z: p.z - 0 });
        expect(dist).toBeGreaterThan(1.9);
        expect(dist).toBeLessThan(2.1);
        expect(len(joint.getConstraintForce())).toBeGreaterThan(0);
        world.destroy();
    },
);

// Port of test_joint.c: one check per joint type. Each creates the joint, exercises the shared base
// API plus every type-specific accessor, then steps to make sure it solves without tripping a
// validation assert. Values are compared through f32() because the C literals are f32.

function frameP(x: number, y: number, z: number) {
    return { p: { x, y, z }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
}

// Static ground + a dynamic box, anchored so a point-coincident joint starts satisfied. Gravity off
// so the body stays put across the handful of steps each check takes.
function fixture() {
    const world = new World({ gravity: { x: 0, y: 0, z: 0 } });
    const ground = world.createBody({});
    const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
    body.createHull({ density: 1 }, makeCubeHull(0.5));
    return { world, ground, body };
}

// Anchor at the dynamic body so both local frames map to the same world point.
const commonFrames = {
    localFrameA: frameP(0, 4, 0),
    localFrameB: frameP(0, 0, 0),
};

function finish(joint: Joint, world: World) {
    for (let i = 0; i < 8; ++i) world.step(1 / 60, 4);
    joint.destroy(true);
    expect(joint.isValid()).toBe(false);
    world.destroy();
}

// Exercise the API shared by every joint type. Frames are saved and restored.
function exerciseBase(joint: Joint, world: World, expectedType: JointType) {
    expect(joint.isValid()).toBe(true);
    expect(joint.getType()).toBe(expectedType);
    const [a, b] = joint.getBodies();
    expect(a.getType()).toBe(BodyType.Static);
    expect(b.getType()).toBe(BodyType.Dynamic);
    expect(joint.getWorld().state).toBe(world.state);

    const originalA = joint.getLocalFrameA();
    const originalB = joint.getLocalFrameB();

    joint.setLocalFrameA(frameP(0.1, 0.2, 0.3));
    const gotA = joint.getLocalFrameA();
    expect(gotA.p.x).toBe(f32(0.1));
    expect(gotA.p.y).toBe(f32(0.2));
    expect(gotA.p.z).toBe(f32(0.3));

    joint.setLocalFrameB(frameP(-0.4, 0.5, -0.6));
    const gotB = joint.getLocalFrameB();
    expect(gotB.p.x).toBe(f32(-0.4));
    expect(gotB.p.y).toBe(f32(0.5));
    expect(gotB.p.z).toBe(f32(-0.6));

    joint.setCollideConnected(true);
    expect(joint.getCollideConnected()).toBe(true);
    joint.setCollideConnected(false);
    expect(joint.getCollideConnected()).toBe(false);

    const userData = {};
    joint.setUserData(userData);
    expect(joint.getUserData()).toBe(userData);

    joint.setConstraintTuning(90, 3);
    const tuning = joint.getConstraintTuning();
    expect(tuning.hertz).toBe(90);
    expect(tuning.dampingRatio).toBe(3);

    joint.setForceThreshold(100);
    expect(joint.getForceThreshold()).toBe(100);
    joint.setTorqueThreshold(200);
    expect(joint.getTorqueThreshold()).toBe(200);

    joint.wakeBodies();

    joint.setLocalFrameA(originalA);
    joint.setLocalFrameB(originalB);
}

type Accessor = [setter: string, args: unknown[], expected: Record<string, unknown>];

const f32v = (x: number, y: number, z: number) => ({ x: f32(x), y: f32(y), z: f32(z) });
const spring = (hertz: number, ratio: number): Accessor[] => [
    ["enableSpring", [true], { isSpringEnabled: true }],
    ["setSpringHertz", [hertz], { getSpringHertz: hertz }],
    ["setSpringDampingRatio", [ratio], { getSpringDampingRatio: f32(ratio) }],
];
const motor = (speed: number): Accessor[] => [
    ["enableMotor", [true], { isMotorEnabled: true }],
    ["setMotorSpeed", [speed], { getMotorSpeed: speed }],
];

const accessorTable: {
    type: JointType;
    create: (world: World, ground: Body, body: Body) => Joint;
    accessors: Accessor[];
}[] = [
    {
        type: JointType.Parallel,
        create: (w, g, b) =>
            w.createParallelJoint(g, b, {
                ...commonFrames,
                hertz: 2,
                dampingRatio: 0.5,
                maxTorque: 100,
            }),
        accessors: [
            ["setSpringHertz", [5], { getSpringHertz: 5 }],
            ["setSpringDampingRatio", [0.7], { getSpringDampingRatio: f32(0.7) }],
            ["setMaxTorque", [250], { getMaxTorque: 250 }],
        ],
    },
    {
        type: JointType.Distance,
        create: (w, g, b) => w.createDistanceJoint(g, b, { ...commonFrames, length: 2 }),
        accessors: [
            ["setLength", [3], { getLength: 3 }],
            [
                "setSpringForceRange",
                [-50, 75],
                { getSpringForceRange: { lowerForce: -50, upperForce: 75 } },
            ],
            ...spring(4, 0.6),
            ["enableLimit", [true], { isLimitEnabled: true }],
            ["setLengthRange", [1, 5], { getMinLength: 1, getMaxLength: 5 }],
            ...motor(1.5),
            ["setMaxMotorForce", [25], { getMaxMotorForce: 25 }],
        ],
    },
    { type: JointType.Filter, create: (w, g, b) => w.createFilterJoint(g, b), accessors: [] },
    {
        type: JointType.Motor,
        create: (w, g, b) => w.createMotorJoint(g, b, { ...commonFrames }),
        accessors: [
            [
                "setLinearVelocity",
                [{ x: 1, y: 2, z: 3 }],
                { getLinearVelocity: { x: 1, y: 2, z: 3 } },
            ],
            [
                "setAngularVelocity",
                [{ x: 0.1, y: 0.2, z: 0.3 }],
                { getAngularVelocity: f32v(0.1, 0.2, 0.3) },
            ],
            ["setMaxVelocityForce", [500], { getMaxVelocityForce: 500 }],
            ["setMaxVelocityTorque", [600], { getMaxVelocityTorque: 600 }],
            ["setLinearHertz", [3], { getLinearHertz: 3 }],
            ["setLinearDampingRatio", [0.8], { getLinearDampingRatio: f32(0.8) }],
            ["setAngularHertz", [4], { getAngularHertz: 4 }],
            ["setAngularDampingRatio", [0.9], { getAngularDampingRatio: f32(0.9) }],
            ["setMaxSpringForce", [700], { getMaxSpringForce: 700 }],
            ["setMaxSpringTorque", [800], { getMaxSpringTorque: 800 }],
        ],
    },
    {
        type: JointType.Prismatic,
        create: (w, g, b) => w.createPrismaticJoint(g, b, { ...commonFrames }),
        accessors: [
            ...spring(5, 0.5),
            ["setTargetTranslation", [1], { getTargetTranslation: 1 }],
            ["enableLimit", [true], { isLimitEnabled: true }],
            ["setLimits", [-2, 2], { getLowerLimit: -2, getUpperLimit: 2 }],
            ...motor(1.5),
            ["setMaxMotorForce", [30], { getMaxMotorForce: 30 }],
        ],
    },
    {
        type: JointType.Revolute,
        create: (w, g, b) => w.createRevoluteJoint(g, b, { ...commonFrames }),
        accessors: [
            ...spring(5, 0.5),
            ["setTargetAngle", [0.5], { getTargetAngle: 0.5 }],
            ["enableLimit", [true], { isLimitEnabled: true }],
            ["setLimits", [-1, 1], { getLowerLimit: -1, getUpperLimit: 1 }],
            ...motor(2),
            ["setMaxMotorTorque", [40], { getMaxMotorTorque: 40 }],
        ],
    },
    {
        type: JointType.Spherical,
        create: (w, g, b) => w.createSphericalJoint(g, b, { ...commonFrames }),
        accessors: [
            ["enableConeLimit", [true], { isConeLimitEnabled: true }],
            ["setConeLimit", [0.5], { getConeLimit: 0.5 }],
            ["enableTwistLimit", [true], { isTwistLimitEnabled: true }],
            ["setTwistLimits", [-0.5, 0.5], { getLowerTwistLimit: -0.5, getUpperTwistLimit: 0.5 }],
            ...spring(5, 0.5),
            [
                "setTargetRotation",
                [{ v: { x: 0, y: 0, z: Math.SQRT1_2 }, s: Math.SQRT1_2 }],
                { getTargetRotation: { v: f32v(0, 0, Math.SQRT1_2), s: f32(Math.SQRT1_2) } },
            ],
            ["enableMotor", [true], { isMotorEnabled: true }],
            [
                "setMotorVelocity",
                [{ x: 0.1, y: 0.2, z: 0.3 }],
                { getMotorVelocity: f32v(0.1, 0.2, 0.3) },
            ],
            ["setMaxMotorTorque", [50], { getMaxMotorTorque: 50 }],
        ],
    },
    {
        type: JointType.Weld,
        create: (w, g, b) => w.createWeldJoint(g, b, { ...commonFrames }),
        accessors: [
            ["setLinearHertz", [3], { getLinearHertz: 3 }],
            ["setLinearDampingRatio", [0.5], { getLinearDampingRatio: 0.5 }],
            ["setAngularHertz", [4], { getAngularHertz: 4 }],
            ["setAngularDampingRatio", [0.7], { getAngularDampingRatio: f32(0.7) }],
        ],
    },
    {
        type: JointType.Wheel,
        create: (w, g, b) => w.createWheelJoint(g, b, { ...commonFrames }),
        accessors: [
            ["enableSuspension", [true], { isSuspensionEnabled: true }],
            ["setSuspensionHertz", [5], { getSuspensionHertz: 5 }],
            ["setSuspensionDampingRatio", [0.5], { getSuspensionDampingRatio: 0.5 }],
            ["enableSuspensionLimit", [true], { isSuspensionLimitEnabled: true }],
            [
                "setSuspensionLimits",
                [-1, 1],
                { getLowerSuspensionLimit: -1, getUpperSuspensionLimit: 1 },
            ],
            ["enableSpinMotor", [true], { isSpinMotorEnabled: true }],
            ["setSpinMotorSpeed", [6], { getSpinMotorSpeed: 6 }],
            ["setMaxSpinTorque", [35], { getMaxSpinTorque: 35 }],
            ["enableSteering", [true], { isSteeringEnabled: true }],
            ["setSteeringHertz", [7], { getSteeringHertz: 7 }],
            ["setSteeringDampingRatio", [0.8], { getSteeringDampingRatio: f32(0.8) }],
            ["setMaxSteeringTorque", [45], { getMaxSteeringTorque: 45 }],
            ["enableSteeringLimit", [true], { isSteeringLimitEnabled: true }],
            [
                "setSteeringLimits",
                [-0.6, 0.6],
                { getLowerSteeringLimit: f32(-0.6), getUpperSteeringLimit: f32(0.6) },
            ],
            ["setTargetSteeringAngle", [0.25], { getTargetSteeringAngle: 0.25 }],
        ],
    },
];

check(
    "every joint type's accessors round-trip and the joint steps",
    {
        claim: "a joint type's type-specific setter does not reach the storage its getter reads, so retuning a hinge, slider, rope, motor, weld, wheel or ball socket from script would silently keep the old value",
    },
    () => {
        for (const row of accessorTable) {
            const { world, ground, body } = fixture();
            const joint = row.create(world, ground, body);
            exerciseBase(joint, world, row.type);
            const methods = joint as unknown as Record<string, (...args: unknown[]) => unknown>;
            for (const [setter, args, expected] of row.accessors) {
                methods[setter](...args);
                for (const [getter, value] of Object.entries(expected)) {
                    expect({ type: row.type, getter, value: methods[getter]() }).toEqual({
                        type: row.type,
                        getter,
                        value,
                    });
                }
            }
            finish(joint, world);
        }
    },
);

// The accessor table above round-trips every accessor with in-order, in-bounds values, so they step
// past the ordering + clamp branches in the limit/length/force setters. These pin those branches,
// each observable through the public getters (Box3D's b3*_SetLimits / SetLength / SetMaxSpring*).

function rig() {
    const world = new World({ gravity: { x: 0, y: 0, z: 0 } });
    const a = world.createBody({});
    const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
    b.createHull({ density: 1 }, makeCubeHull(0.5));
    return { world, a, b };
}

check(
    "revolute setLimits orders reversed args and clamps to plus/minus 0.99 pi",
    {
        claim: "revolute setLimits stores its arguments unordered or unclamped, so a hinge given reversed or out-of-range angles would hold an empty or wrapped limit range",
    },
    () => {
        const { world, a, b } = rig();
        const j = world.createRevoluteJoint(a, b);
        j.setLimits(1, -1); // reversed, so ordered low..high
        expect(j.getLowerLimit()).toBe(-1);
        expect(j.getUpperLimit()).toBe(1);
        const bound = f32(0.99 * PI);
        j.setLimits(-10, 10); // out of range, so clamped
        expect(j.getLowerLimit()).toBe(-bound);
        expect(j.getUpperLimit()).toBe(bound);
        world.destroy();
    },
);

check(
    "prismatic setLimits orders reversed args with no clamp",
    {
        claim: "prismatic setLimits stores its arguments unordered, so a slider given reversed translations would hold an empty limit range and jam",
    },
    () => {
        const { world, a, b } = rig();
        const j = world.createPrismaticJoint(a, b);
        j.setLimits(2, -2);
        expect(j.getLowerLimit()).toBe(-2);
        expect(j.getUpperLimit()).toBe(2);
        world.destroy();
    },
);

check(
    "spherical setTwistLimits orders reversed args and clamps to plus/minus 0.99 pi",
    {
        claim: "spherical setTwistLimits stores its arguments unordered or unclamped, so a ball socket given reversed or out-of-range twist angles would hold an empty or wrapped range",
    },
    () => {
        const { world, a, b } = rig();
        const j = world.createSphericalJoint(a, b);
        j.setTwistLimits(0.5, -0.5);
        expect(j.getLowerTwistLimit()).toBe(-0.5);
        expect(j.getUpperTwistLimit()).toBe(0.5);
        const bound = f32(0.99 * PI);
        j.setTwistLimits(-10, 10);
        expect(j.getLowerTwistLimit()).toBe(-bound);
        expect(j.getUpperTwistLimit()).toBe(bound);
        world.destroy();
    },
);

check(
    "distance setLength clamps below the linear slop",
    {
        claim: "distance setLength accepts a zero or negative length, so a collapsed rope would divide by its own degenerate axis in the solver",
    },
    () => {
        const { world, a, b } = rig();
        const j = world.createDistanceJoint(a, b, { length: 2 });
        j.setLength(0);
        expect(j.getLength()).toBe(LINEAR_SLOP);
        world.destroy();
    },
);

check(
    "distance setLengthRange clamps each value then orders the pair",
    {
        claim: "distance setLengthRange orders before clamping or skips one of the two, so a reversed range with a degenerate low end would survive into the solver",
    },
    () => {
        const { world, a, b } = rig();
        const j = world.createDistanceJoint(a, b, { length: 2 });
        // Reversed, and the smaller falls below the slop: clamp each to [slop, huge], then order.
        j.setLengthRange(5, 0);
        expect(j.getMinLength()).toBe(LINEAR_SLOP);
        expect(j.getMaxLength()).toBe(5);
        world.destroy();
    },
);

check(
    "motor clamps a negative max spring force and torque to zero",
    {
        claim: "motor setMaxSpringForce and setMaxSpringTorque keep a negative cap, so a mis-set motor would drive its spring backwards instead of disabling it",
    },
    () => {
        const { world, a, b } = rig();
        const j = world.createMotorJoint(a, b);
        j.setMaxSpringForce(-5);
        expect(j.getMaxSpringForce()).toBe(0);
        j.setMaxSpringTorque(-7);
        expect(j.getMaxSpringTorque()).toBe(0);
        j.setMaxSpringForce(12); // a positive value passes through unclamped
        expect(j.getMaxSpringForce()).toBe(12);
        world.destroy();
    },
);
