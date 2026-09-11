// Fast-tier joint behavior (Box3D's test_joint.c essence + the collideConnected filter). The
// bit-exact solve math is covered by step.fixture.ts; here we exercise the public handle lifecycle,
// the joint-connected contact filter, and that each joint type actually constrains its bodies.

import { expect } from "bun:test";
import { check } from "../../../harness/check";
import {
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
    "a created joint exposes its type, its bodies and the world's joint count",
    {
        claim: "a freshly created joint handle reports the wrong type or the wrong pair of bodies, so a caller could not tell which constraint it just made",
    },
    () => {
        const { world, joint } = pendulum();
        expect(joint.getType()).toBe(JointType.Revolute);
        const [a, b] = joint.getBodies();
        expect(a.getPosition().y).toBe(5);
        expect(b.getType()).toBe(BodyType.Dynamic);
        expect(world.getCounters().jointCount).toBe(1);
        world.destroy();
    },
);

check(
    "destroying a joint invalidates its handle and frees the id",
    {
        claim: "a destroyed joint still reports itself valid or leaves its id counted, so the joint pool would leak slots across a scene teardown",
    },
    () => {
        const { world, joint } = pendulum();
        expect(joint.isValid()).toBe(true);
        joint.destroy();
        expect(joint.isValid()).toBe(false);
        expect(world.getCounters().jointCount).toBe(0);
        world.destroy();
    },
);

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

    // No stable value to assert before the first step; call for coverage (must not throw / NaN).
    expect(Number.isFinite(joint.getLinearSeparation())).toBe(true);
    joint.getConstraintForce();
    joint.getConstraintTorque();
    // Wheel angular separation is an unimplemented todo in the C reference.
    if (expectedType !== JointType.Wheel) {
        expect(Number.isFinite(joint.getAngularSeparation())).toBe(true);
    }

    joint.setLocalFrameA(originalA);
    joint.setLocalFrameB(originalB);
}

check(
    "parallel joint accessors round-trip and the joint steps",
    {
        claim: "a parallel joint's spring and max-torque accessors do not round-trip through the solver's storage, so tuning a parallel joint from script would silently keep the old value",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createParallelJoint(ground, body, {
            ...commonFrames,
            hertz: 2,
            dampingRatio: 0.5,
            maxTorque: 100,
        });
        exerciseBase(joint, world, JointType.Parallel);

        joint.setSpringHertz(5);
        expect(joint.getSpringHertz()).toBe(5);
        joint.setSpringDampingRatio(0.7);
        expect(joint.getSpringDampingRatio()).toBe(f32(0.7));
        joint.setMaxTorque(250);
        expect(joint.getMaxTorque()).toBe(250);

        finish(joint, world);
    },
);

check(
    "distance joint accessors round-trip and the joint steps",
    {
        claim: "a distance joint's length, spring, limit and motor accessors do not round-trip, so retuning a rope or spring from script would silently keep the old value",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createDistanceJoint(ground, body, { ...commonFrames, length: 2 });
        exerciseBase(joint, world, JointType.Distance);

        joint.setLength(3);
        expect(joint.getLength()).toBe(3);
        joint.enableSpring(true);
        expect(joint.isSpringEnabled()).toBe(true);
        joint.setSpringForceRange(-50, 75);
        const range = joint.getSpringForceRange();
        expect(range.lowerForce).toBe(-50);
        expect(range.upperForce).toBe(75);
        joint.setSpringHertz(4);
        expect(joint.getSpringHertz()).toBe(4);
        joint.setSpringDampingRatio(0.6);
        expect(joint.getSpringDampingRatio()).toBe(f32(0.6));
        joint.enableLimit(true);
        expect(joint.isLimitEnabled()).toBe(true);
        joint.setLengthRange(1, 5);
        expect(joint.getMinLength()).toBe(1);
        expect(joint.getMaxLength()).toBe(5);
        expect(Number.isFinite(joint.getCurrentLength())).toBe(true);
        joint.enableMotor(true);
        expect(joint.isMotorEnabled()).toBe(true);
        joint.setMotorSpeed(1.5);
        expect(joint.getMotorSpeed()).toBe(1.5);
        joint.setMaxMotorForce(25);
        expect(joint.getMaxMotorForce()).toBe(25);
        expect(Number.isFinite(joint.getMotorForce())).toBe(true);

        finish(joint, world);
    },
);

check(
    "filter joint carries the shared base API with no type-specific accessors",
    {
        claim: "a filter joint does not answer the shared joint API or refuses to step, so a contact-suppression joint would break the code paths every other joint type shares",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createFilterJoint(ground, body);
        exerciseBase(joint, world, JointType.Filter);
        finish(joint, world);
    },
);

check(
    "motor joint accessors round-trip and the joint steps",
    {
        claim: "a motor joint's velocity targets, force caps and spring tuning do not round-trip, so retargeting a motor from script would silently keep the old drive",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createMotorJoint(ground, body, { ...commonFrames });
        exerciseBase(joint, world, JointType.Motor);

        joint.setLinearVelocity({ x: 1, y: 2, z: 3 });
        expect(joint.getLinearVelocity()).toEqual({ x: 1, y: 2, z: 3 });
        joint.setAngularVelocity({ x: 0.1, y: 0.2, z: 0.3 });
        const w = joint.getAngularVelocity();
        expect(w.x).toBe(f32(0.1));
        expect(w.y).toBe(f32(0.2));
        expect(w.z).toBe(f32(0.3));
        joint.setMaxVelocityForce(500);
        expect(joint.getMaxVelocityForce()).toBe(500);
        joint.setMaxVelocityTorque(600);
        expect(joint.getMaxVelocityTorque()).toBe(600);
        joint.setLinearHertz(3);
        expect(joint.getLinearHertz()).toBe(3);
        joint.setLinearDampingRatio(0.8);
        expect(joint.getLinearDampingRatio()).toBe(f32(0.8));
        joint.setAngularHertz(4);
        expect(joint.getAngularHertz()).toBe(4);
        joint.setAngularDampingRatio(0.9);
        expect(joint.getAngularDampingRatio()).toBe(f32(0.9));
        joint.setMaxSpringForce(700);
        expect(joint.getMaxSpringForce()).toBe(700);
        joint.setMaxSpringTorque(800);
        expect(joint.getMaxSpringTorque()).toBe(800);

        finish(joint, world);
    },
);

check(
    "prismatic joint accessors round-trip and the joint steps",
    {
        claim: "a prismatic joint's spring, limit and motor accessors do not round-trip, so retuning a slider from script would silently keep the old rail settings",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createPrismaticJoint(ground, body, { ...commonFrames });
        exerciseBase(joint, world, JointType.Prismatic);

        joint.enableSpring(true);
        expect(joint.isSpringEnabled()).toBe(true);
        joint.setSpringHertz(5);
        expect(joint.getSpringHertz()).toBe(5);
        joint.setSpringDampingRatio(0.5);
        expect(joint.getSpringDampingRatio()).toBe(0.5);
        joint.setTargetTranslation(1);
        expect(joint.getTargetTranslation()).toBe(1);
        joint.enableLimit(true);
        expect(joint.isLimitEnabled()).toBe(true);
        joint.setLimits(-2, 2);
        expect(joint.getLowerLimit()).toBe(-2);
        expect(joint.getUpperLimit()).toBe(2);
        joint.enableMotor(true);
        expect(joint.isMotorEnabled()).toBe(true);
        joint.setMotorSpeed(1.5);
        expect(joint.getMotorSpeed()).toBe(1.5);
        joint.setMaxMotorForce(30);
        expect(joint.getMaxMotorForce()).toBe(30);
        expect(Number.isFinite(joint.getMotorForce())).toBe(true);
        expect(Number.isFinite(joint.getTranslation())).toBe(true);
        expect(Number.isFinite(joint.getSpeed())).toBe(true);

        finish(joint, world);
    },
);

check(
    "revolute joint accessors round-trip and the joint steps",
    {
        claim: "a revolute joint's target angle, limits and motor accessors do not round-trip, so retuning a hinge from script would silently keep the old settings",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createRevoluteJoint(ground, body, { ...commonFrames });
        exerciseBase(joint, world, JointType.Revolute);

        joint.enableSpring(true);
        expect(joint.isSpringEnabled()).toBe(true);
        joint.setSpringHertz(5);
        expect(joint.getSpringHertz()).toBe(5);
        joint.setSpringDampingRatio(0.5);
        expect(joint.getSpringDampingRatio()).toBe(0.5);
        joint.setTargetAngle(0.5);
        expect(joint.getTargetAngle()).toBe(0.5);
        expect(Number.isFinite(joint.getAngle())).toBe(true);
        joint.enableLimit(true);
        expect(joint.isLimitEnabled()).toBe(true);
        joint.setLimits(-1, 1);
        expect(joint.getLowerLimit()).toBe(-1);
        expect(joint.getUpperLimit()).toBe(1);
        joint.enableMotor(true);
        expect(joint.isMotorEnabled()).toBe(true);
        joint.setMotorSpeed(2);
        expect(joint.getMotorSpeed()).toBe(2);
        joint.setMaxMotorTorque(40);
        expect(joint.getMaxMotorTorque()).toBe(40);
        expect(Number.isFinite(joint.getMotorTorque())).toBe(true);

        finish(joint, world);
    },
);

check(
    "spherical joint accessors round-trip and the joint steps",
    {
        claim: "a spherical joint's cone, twist, spring and motor accessors do not round-trip, so retuning a ball socket from script would silently keep the old limits",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createSphericalJoint(ground, body, { ...commonFrames });
        exerciseBase(joint, world, JointType.Spherical);

        joint.enableConeLimit(true);
        expect(joint.isConeLimitEnabled()).toBe(true);
        joint.setConeLimit(0.5);
        expect(joint.getConeLimit()).toBe(0.5);
        expect(Number.isFinite(joint.getConeAngle())).toBe(true);
        joint.enableTwistLimit(true);
        expect(joint.isTwistLimitEnabled()).toBe(true);
        joint.setTwistLimits(-0.5, 0.5);
        expect(joint.getLowerTwistLimit()).toBe(-0.5);
        expect(joint.getUpperTwistLimit()).toBe(0.5);
        expect(Number.isFinite(joint.getTwistAngle())).toBe(true);
        joint.enableSpring(true);
        expect(joint.isSpringEnabled()).toBe(true);
        joint.setSpringHertz(5);
        expect(joint.getSpringHertz()).toBe(5);
        joint.setSpringDampingRatio(0.5);
        expect(joint.getSpringDampingRatio()).toBe(0.5);

        // 90 degrees about z: a unit quaternion (0, 0, sin45, cos45) that round-trips through f32.
        const target = { v: { x: 0, y: 0, z: Math.SQRT1_2 }, s: Math.SQRT1_2 };
        joint.setTargetRotation(target);
        const got = joint.getTargetRotation();
        expect(Math.abs(got.v.x - target.v.x)).toBeLessThan(1e-5);
        expect(Math.abs(got.v.y - target.v.y)).toBeLessThan(1e-5);
        expect(Math.abs(got.v.z - target.v.z)).toBeLessThan(1e-5);
        expect(Math.abs(got.s - target.s)).toBeLessThan(1e-5);

        joint.enableMotor(true);
        expect(joint.isMotorEnabled()).toBe(true);
        joint.setMotorVelocity({ x: 0.1, y: 0.2, z: 0.3 });
        const mv = joint.getMotorVelocity();
        expect(mv.x).toBe(f32(0.1));
        expect(mv.y).toBe(f32(0.2));
        expect(mv.z).toBe(f32(0.3));
        joint.setMaxMotorTorque(50);
        expect(joint.getMaxMotorTorque()).toBe(50);
        const mt = joint.getMotorTorque();
        expect(Number.isFinite(mt.x)).toBe(true);

        finish(joint, world);
    },
);

check(
    "weld joint accessors round-trip and the joint steps",
    {
        claim: "a weld joint's linear and angular softness accessors do not round-trip, so softening a weld from script would silently keep it rigid",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createWeldJoint(ground, body, { ...commonFrames });
        exerciseBase(joint, world, JointType.Weld);

        joint.setLinearHertz(3);
        expect(joint.getLinearHertz()).toBe(3);
        joint.setLinearDampingRatio(0.5);
        expect(joint.getLinearDampingRatio()).toBe(0.5);
        joint.setAngularHertz(4);
        expect(joint.getAngularHertz()).toBe(4);
        joint.setAngularDampingRatio(0.7);
        expect(joint.getAngularDampingRatio()).toBe(f32(0.7));

        finish(joint, world);
    },
);

check(
    "wheel joint accessors round-trip and the joint steps",
    {
        claim: "a wheel joint's suspension, spin-motor and steering accessors do not round-trip, so tuning a vehicle wheel from script would silently keep the old suspension or steering",
    },
    () => {
        const { world, ground, body } = fixture();
        const joint = world.createWheelJoint(ground, body, { ...commonFrames });
        exerciseBase(joint, world, JointType.Wheel);

        joint.enableSuspension(true);
        expect(joint.isSuspensionEnabled()).toBe(true);
        joint.setSuspensionHertz(5);
        expect(joint.getSuspensionHertz()).toBe(5);
        joint.setSuspensionDampingRatio(0.5);
        expect(joint.getSuspensionDampingRatio()).toBe(0.5);
        joint.enableSuspensionLimit(true);
        expect(joint.isSuspensionLimitEnabled()).toBe(true);
        joint.setSuspensionLimits(-1, 1);
        expect(joint.getLowerSuspensionLimit()).toBe(-1);
        expect(joint.getUpperSuspensionLimit()).toBe(1);
        joint.enableSpinMotor(true);
        expect(joint.isSpinMotorEnabled()).toBe(true);
        joint.setSpinMotorSpeed(6);
        expect(joint.getSpinMotorSpeed()).toBe(6);
        joint.setMaxSpinTorque(35);
        expect(joint.getMaxSpinTorque()).toBe(35);
        expect(Number.isFinite(joint.getSpinSpeed())).toBe(true);
        expect(Number.isFinite(joint.getSpinTorque())).toBe(true);
        joint.enableSteering(true);
        expect(joint.isSteeringEnabled()).toBe(true);
        joint.setSteeringHertz(7);
        expect(joint.getSteeringHertz()).toBe(7);
        joint.setSteeringDampingRatio(0.8);
        expect(joint.getSteeringDampingRatio()).toBe(f32(0.8));
        joint.setMaxSteeringTorque(45);
        expect(joint.getMaxSteeringTorque()).toBe(45);
        joint.enableSteeringLimit(true);
        expect(joint.isSteeringLimitEnabled()).toBe(true);
        joint.setSteeringLimits(-0.6, 0.6);
        expect(joint.getLowerSteeringLimit()).toBe(f32(-0.6));
        expect(joint.getUpperSteeringLimit()).toBe(f32(0.6));
        joint.setTargetSteeringAngle(0.25);
        expect(joint.getTargetSteeringAngle()).toBe(0.25);
        expect(Number.isFinite(joint.getSteeringAngle())).toBe(true);
        expect(Number.isFinite(joint.getSteeringTorque())).toBe(true);

        finish(joint, world);
    },
);

// The parity checks above round-trip every accessor with in-order, in-bounds values, so they step
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
