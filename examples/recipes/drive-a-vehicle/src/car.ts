import {
    Body,
    Color,
    Inputs,
    Part,
    type Plugin,
    ShapeKind,
    type State,
    type System,
    Tumble,
} from "@dylanebert/shallot";

// a drivable vehicle on wheel joints. each wheel joint gives one wheel a suspension spring (frame A's x is
// the suspension axis, pointing up), a spin axis, and — on the front wheels — a steering DOF. the rear wheels
// carry a spin motor (the throttle), the front wheels a steering target. the chassis + wheels are substrate
// `Body` entities (their friction is authored right on the body); the wheel joints, and a soft parallel joint
// to the ground that keeps the car upright, ride `Tumble.world`, the escape hatch past the substrate's
// `Spring`/`Joint`. drive with W/S (throttle) and A/D (steer). sphere wheels sidestep wheel-orientation
// bookkeeping, so the joint frames stay simple.
//
// wheel joints (suspension + spin motor + steering) have no published substrate-surface equivalent yet, so
// this recipe rides the escape hatch; the gym twin `joints-driving` is the oracle-gated gold.

type V3 = { x: number; y: number; z: number };
type Wheel = ReturnType<NonNullable<typeof Tumble.world>["createWheelJoint"]>;

const THROTTLE = 14; // rad/s spin speed at full throttle
const STEER = Math.PI / 5; // rad steering lock

// shortest-arc quaternion rotating unit `from` onto unit `to`; used to build the wheel joint's frame
// rotations (suspension axis, spin axis) and the wheels' own rest orientation.
function quatBetween(from: V3, to: V3): { v: V3; s: number } {
    const v = {
        x: from.y * to.z - from.z * to.y,
        y: from.z * to.x - from.x * to.z,
        z: from.x * to.y - from.y * to.x,
    };
    const s = 1 + (from.x * to.x + from.y * to.y + from.z * to.z);
    const len = Math.hypot(v.x, v.y, v.z, s) || 1;
    return { v: { x: v.x / len, y: v.y / len, z: v.z / len }, s: s / len };
}

const AXIS_X = { x: 1, y: 0, z: 0 };
const AXIS_Y = { x: 0, y: 1, z: 0 };
const AXIS_Z = { x: 0, y: 0, z: 1 };
const Q_SUSP = quatBetween(AXIS_X, AXIS_Y); // frame A: local x → world up (the suspension axis)
const Q_SPIN = quatBetween(AXIS_Z, AXIS_Y); // frame B: local z is the spin axis
const WHEEL_ROT = quatBetween(AXIS_Y, AXIS_Z); // wheel rest orientation so the spin axis lands horizontal

let groundEid = -1;
let chassisEid = -1;
let wheelEids: number[] = [];
let rear: Wheel[] = [];
let front: Wheel[] = [];
let wired = false;

function part(state: State, eid: number, color: [number, number, number]): void {
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
}

function build(state: State): void {
    groundEid = state.create();
    state.add(groundEid, Body);
    Body.pos.set(groundEid, 0, 0, 0, 0);
    Body.halfExtents.set(groundEid, 30, 0.5, 30, 0);
    Body.mass.set(groundEid, 0);
    Body.friction.set(groundEid, 1);
    part(state, groundEid, [0.4, 0.42, 0.46]);

    chassisEid = state.create();
    state.add(chassisEid, Body);
    Body.pos.set(chassisEid, 0, 2.5, 0, 0);
    Body.halfExtents.set(chassisEid, 2, 0.5, 1, 0);
    Body.mass.set(chassisEid, 4);
    part(state, chassisEid, [0.5, 0.55, 0.85]);

    wheelEids = [];
    for (const [sx, sz] of [
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
    ]) {
        const eid = state.create();
        state.add(eid, Body);
        Body.shape.set(eid, ShapeKind.Sphere);
        // the wheels sit BELOW the chassis (top y 1.9 clears its bottom y 2.0), never inside it: a wheel
        // that overlaps the chassis collides for the one fixed tick before its joint is wired, and the
        // persistent contact then fights the suspension so the car won't drive (the joint wires a step after
        // the bodies marshal). collideConnected filters the pair only from wiring on, not that first tick.
        Body.pos.set(eid, 1.5 * sx, 1.5, 0.8 * sz, 0);
        Body.quat.set(eid, WHEEL_ROT.v.x, WHEEL_ROT.v.y, WHEEL_ROT.v.z, WHEEL_ROT.s);
        Body.halfExtents.set(eid, 0, 0, 0, 0.4); // sphere: radius rides the w lane
        Body.mass.set(eid, 0.5);
        Body.friction.set(eid, 3);
        part(state, eid, [0.2, 0.22, 0.26]);
        wheelEids.push(eid);
    }
    state.addSystem(driver);
}

// wire the four wheel joints + the parallel upright joint once every body has marshaled. the front two
// (indices 0, 1) steer; the rear two (2, 3) carry the spin motor.
function wire(): void {
    const world = Tumble.world;
    if (!world || wired) return;
    const ground = Tumble.body(groundEid);
    const chassis = Tumble.body(chassisEid);
    const wheels = wheelEids.map((e) => Tumble.body(e));
    if (!ground || !chassis || wheels.some((w) => !w)) return;

    rear = [];
    front = [];
    const layout: [number, number, boolean][] = [
        [1, 1, true],
        [1, -1, true],
        [-1, 1, false],
        [-1, -1, false],
    ];
    for (let i = 0; i < wheels.length; i++) {
        const [sx, sz, isFront] = layout[i];
        const suspension = {
            // frame A sits at the wheel's rest centre — chassis-local y = wheelY(1.5) − chassisY(2.5) = −1
            localFrameA: { p: { x: 1.5 * sx, y: -1, z: 0.8 * sz }, q: Q_SUSP },
            localFrameB: { p: { x: 0, y: 0, z: 0 }, q: Q_SPIN },
            enableSuspensionSpring: true,
            suspensionHertz: 4,
            suspensionDampingRatio: 0.7,
            enableSuspensionLimit: true,
            lowerSuspensionLimit: -0.2,
            upperSuspensionLimit: 0.2,
        };
        const j = isFront
            ? world.createWheelJoint(chassis, wheels[i]!, {
                  ...suspension,
                  enableSteering: true,
                  steeringHertz: 10,
                  steeringDampingRatio: 0.7,
                  maxSteeringTorque: 5,
                  targetSteeringAngle: 0,
                  enableSteeringLimit: true,
                  lowerSteeringLimit: -STEER,
                  upperSteeringLimit: STEER,
              })
            : world.createWheelJoint(chassis, wheels[i]!, {
                  ...suspension,
                  enableSpinMotor: true,
                  spinSpeed: 0,
                  maxSpinTorque: 5,
              });
        (isFront ? front : rear).push(j);
    }

    world.createParallelJoint(ground, chassis, {
        localFrameA: { p: { x: 0, y: 0, z: 0 }, q: Q_SPIN },
        localFrameB: { p: { x: 0, y: 0, z: 0 }, q: Q_SPIN },
        hertz: 0.5,
        dampingRatio: 1,
        collideConnected: true,
    });
    wired = true;
}

// read the throttle + steer each frame and push them to the joints: the rear spin motors get the throttle
// speed, the front steering targets the lock angle.
const driver: System = {
    name: "car-driver",
    group: "simulation",
    update() {
        if (!wired) {
            wire();
            return;
        }
        let throttle = 0;
        if (Inputs.isKeyDown("KeyW")) throttle -= THROTTLE;
        if (Inputs.isKeyDown("KeyS")) throttle += THROTTLE;
        let steer = 0;
        if (Inputs.isKeyDown("KeyA")) steer += STEER;
        if (Inputs.isKeyDown("KeyD")) steer -= STEER;
        for (const j of rear) j.setSpinMotorSpeed(throttle);
        for (const j of front) j.setTargetSteeringAngle(steer);
        // a parked car sleeps, and setting a motor speed does NOT wake a sleeping body — so wake the wheels +
        // chassis on any driver input, or the first throttle after the car settles would be ignored.
        if (throttle !== 0 || steer !== 0) {
            Tumble.body(chassisEid)?.setAwake(true);
            for (const e of wheelEids) Tumble.body(e)?.setAwake(true);
        }
    },
};

export const Car = {
    name: "Car",
    warm(state: State) {
        groundEid = -1;
        chassisEid = -1;
        wheelEids = [];
        rear = [];
        front = [];
        wired = false;
        build(state);
    },
} satisfies Plugin;

export default Car;
