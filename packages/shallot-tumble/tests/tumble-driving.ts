// The tumble.js `Driving` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A four-wheeled car on wheel joints: rear wheels driven by their spin motors,
// front wheels steered. A suspension spring at each wheel and a soft parallel joint to the ground keep it
// upright. Sphere wheels sidestep wheel-orientation bookkeeping.
//
// The sample's `throttle`/`steer` knobs are declared `live: true` — the base `Sample` only applies a live
// knob through `act()`, called from a slider drag or a button press, never from `step()`. The mint
// (`sample.step()`) never touches a knob, so `act()` never fires during the gold trajectory: `throttle`'s
// build-time value (baked into the rear wheels' `spinSpeed`) is the one the whole gold trajectory sees, and
// `steer` (default 0, applied nowhere in `build()`) never turns the front wheels either. No `update()`
// needed — `build()` alone reproduces the gold.
//
// Creation order is load-bearing for the hash: ground, chassis, the four wheels (front-left, front-right,
// rear-left, rear-right), then the chassis-to-ground parallel joint — the sample's exact order.

import {
    BodyType,
    defaultSurfaceMaterial,
    makeBoxHull,
    type Quat,
    type Vec3,
    type WheelJointConfig,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

const IDENT: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };

function at(x: number, y: number, z: number, q: Quat = IDENT) {
    return { p: { x, y, z }, q };
}

/** Shortest-arc quaternion rotating unit `from` onto unit `to` (half-vector form; axis pairs only). */
function quatBetween(from: Vec3, to: Vec3): Quat {
    const v = {
        x: from.y * to.z - from.z * to.y,
        y: from.z * to.x - from.x * to.z,
        z: from.x * to.y - from.y * to.x,
    };
    const s = 1 + (from.x * to.x + from.y * to.y + from.z * to.z);
    const len = Math.hypot(v.x, v.y, v.z, s) || 1;
    return { v: { x: v.x / len, y: v.y / len, z: v.z / len }, s: s / len };
}

/**
 * Author the Driving scene into `world`, reading the `throttle` knob (rear spin-motor speed baked into
 * `build()` — see the module note on why `steer`/live retarget never fire in the gold). A static ground
 * box, a dynamic chassis, four sphere wheels on wheel joints (front steered, rear spin-driven), and a soft
 * parallel joint holding the chassis level to the ground.
 */
export function buildDriving(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const chassis = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2.5, z: 0 } });
    chassis.createHull({ density: 0.5 }, makeBoxHull(2, 0.5, 1));

    const axisX = { x: 1, y: 0, z: 0 };
    const axisY = { x: 0, y: 1, z: 0 };
    const axisZ = { x: 0, y: 0, z: 1 };
    // Frame A's x is the suspension axis (world up); frame B's z is the spin axis. The wheel body is
    // pre-rotated y→z so its localFrameB.q cancels back to identity, landing the spin axis on world z (a
    // horizontal roll). Without the body rotation the spin axis points up and steering degenerates to a
    // divide-by-zero. Mirrors Box3D's Driving sample (sample_joint.cpp).
    const qA = quatBetween(axisX, axisY);
    const qB = quatBetween(axisZ, axisY);
    const wheelRot = quatBetween(axisY, axisZ);
    const throttle = params.throttle as number;

    const wheel = (sx: number, sz: number, isFront: boolean): void => {
        const w = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1.5 * sx, y: 2.0, z: 0.8 * sz },
            rotation: wheelRot,
            allowFastRotation: true,
        });
        w.createSphere(
            { density: 2, baseMaterial: { ...defaultSurfaceMaterial(), friction: 3 } },
            { center: { x: 0, y: 0, z: 0 }, radius: 0.4 },
        );
        const cfg: Partial<WheelJointConfig> = {
            localFrameA: at(1.5 * sx, -0.5, 0.8 * sz, qA),
            localFrameB: at(0, 0, 0, qB),
            enableSuspensionSpring: true,
            suspensionHertz: 4,
            suspensionDampingRatio: 0.7,
            enableSuspensionLimit: true,
            lowerSuspensionLimit: -0.2,
            upperSuspensionLimit: 0.2,
        };
        if (isFront) {
            cfg.enableSteering = true;
            cfg.steeringHertz = 10;
            cfg.steeringDampingRatio = 0.7;
            cfg.maxSteeringTorque = 5;
            cfg.targetSteeringAngle = 0;
            cfg.enableSteeringLimit = true;
            cfg.lowerSteeringLimit = -Math.PI / 4;
            cfg.upperSteeringLimit = Math.PI / 4;
        } else {
            cfg.enableSpinMotor = true;
            cfg.spinSpeed = throttle;
            cfg.maxSpinTorque = 5;
        }
        world.createWheelJoint(chassis, w, cfg);
    };

    wheel(1, 1, true);
    wheel(1, -1, true);
    wheel(-1, 1, false);
    wheel(-1, -1, false);

    world.createParallelJoint(ground, chassis, {
        localFrameA: at(0, 0, 0, qB),
        localFrameB: at(0, 0, 0, qB),
        hertz: 0.5,
        dampingRatio: 1,
        collideConnected: true,
    });
}
