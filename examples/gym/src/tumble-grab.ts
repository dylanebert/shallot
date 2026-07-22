// Mouse-grab for the tumble sample host, ported from tumble.js `samples/src/sample.ts` (Box3D's
// samples/sample.cpp MouseDown/MouseMove/Step). A pick ray → closest dynamic body → a spring motor joint
// on a kinematic anchor; the anchor is driven toward the cursor target each fixed tick, so a held-still
// mouse yields ~zero anchor velocity and the grab stays stable. Ray-based (no camera): the host converts
// the pointer + orbit camera to a world ray (`physics/core` `cursorRay`) and hands it in, keeping this pure
// engine-surface logic the grab-regression test drives headless.

import {
    type Body,
    BodyType,
    type Joint,
    type Vec3,
    type World,
} from "@dylanebert/shallot/tumble/core";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/** a live grab: the motor joint + its kinematic anchor, the stored ray depth, and the current world target
 *  the anchor is driven toward each fixed tick. `body` is the grabbed dynamic body (instrumentation reads its
 *  live pose; the physics never touches it through this field). */
export interface Grab {
    joint: Joint;
    anchor: Body;
    depth: number;
    target: Vec3;
    body: Body;
}

const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const len = (v: Vec3): number => Math.hypot(v.x, v.y, v.z);
function normalize(v: Vec3): Vec3 {
    const l = len(v) || 1;
    return { x: v.x / l, y: v.y / l, z: v.z / l };
}

/**
 * Begin a grab from a world pick ray (origin + full translation vector, `origin + translation` = the far
 * point). Raycasts the world, and if the closest hit is a **dynamic** body springs it to a fresh kinematic
 * anchor at the hit point. Returns the grab, or null on a miss / a non-dynamic hit (the caller orbits).
 *
 * The spring force scales to the body's weight (100× mg, mirroring sample.cpp), not a flat constant — a
 * flat force rips a light body through its neighbours (the regression the host's grab test pins). The joint
 * anchors at the click point in the body's local frame (zero initial offset — no yank on grab); the
 * velocity torque is angular friction so the grabbed body doesn't spin free.
 */
export function beginGrab(world: World, origin: Vec3, translation: Vec3): Grab | null {
    const r = world.castRayClosest(origin, translation);
    if (r.hit === false || r.shape === null) return null;
    const body = r.shape.getBody();
    if (body.getType() !== BodyType.Dynamic) return null;

    const dir = normalize(translation);
    const depth = dot(sub(r.point, origin), dir);
    const anchor = world.createBody({
        type: BodyType.Kinematic,
        position: r.point,
        enableSleep: false,
    });

    const md = body.getMassData();
    const mg = md.mass * len(world.getGravity());
    const trace = md.inertia.cx.x + md.inertia.cy.y + md.inertia.cz.z;
    const lever = md.mass > 0 ? Math.sqrt(trace / (3 * md.mass)) : 0;
    const joint = world.createMotorJoint(anchor, body, {
        localFrameB: { p: body.getLocalPoint(r.point), q: IDENT },
        linearHertz: 7.5,
        linearDampingRatio: 1,
        maxSpringForce: 100 * mg,
        maxVelocityTorque: 0.5 * lever * mg,
    });
    return { joint, anchor, depth, target: r.point, body };
}

/** Update the grab target from a new pick ray at the stored depth. The anchor is not moved here —
 *  {@link driveGrab} moves it toward this target each tick, so a held-still ray produces no drift. */
export function updateGrab(grab: Grab, origin: Vec3, dir: Vec3): void {
    const d = normalize(dir);
    grab.target = {
        x: origin.x + grab.depth * d.x,
        y: origin.y + grab.depth * d.y,
        z: origin.z + grab.depth * d.z,
    };
}

// A physical mouse-drag sweeps only a fraction of the viewport per frame. A one-frame full-viewport cursor
// jump — a frame hitch, or synthetic trusted input — is not reachable by a hand, and it drives the kinematic
// anchor to an unbounded velocity: setTargetTransform sets the anchor velocity to (target − anchor)/dt, so a
// far one-frame target makes the motor joint inject a huge impulse into the grabbed body AND its joint-chain
// neighbours in a single step, flinging them out of the camera frustum (the floor-vanish, spec tumble-inline
// 6b — a correct cull of grab-flung bodies). The tumble.js source `Sample` (samples/src/sample.ts step()) has
// NO such cap — it relies on the browser rate-limiting real pointermove events — so this bound is a documented
// deviation from the source, not a missed port.
//
// Cap the anchor's per-step move toward the target at a fraction of the pick depth. Cursor-pixel → world scale
// is proportional to viewing depth (a viewport spans ~2·depth·tan(fov/2) world units), so `MAX_DRAG_FRACTION ·
// depth` bounds the cursor to that fraction of the viewport per step, scene-scale-free. 0.25 sits ~7× above the
// fastest normal drag (the interaction gate's pyramid/pendulum grab-drag envelopes, ~0.03·depth per step) and
// ~4× below a full-viewport teleport (~1.15·depth), so normal drag is untouched and only the teleport is
// clamped. The anchor still tracks the true cursor point (`grab.target`) — it just catches up at bounded speed
// over the next frames, so a real drag feels identical and only a single-frame spike is bounded.
const MAX_DRAG_FRACTION = 0.25;

/** Drive the kinematic anchor toward the current target over `dt` (sample.cpp Step: `SetTargetTransform`
 *  every step, not on the pointer event — a stationary target yields ~zero velocity), clamping the per-step
 *  move to {@link MAX_DRAG_FRACTION}·depth so a one-frame cursor teleport can't inject unbounded velocity.
 *  Call before the world steps. */
export function driveGrab(grab: Grab, dt: number): void {
    const cur = grab.anchor.getPosition();
    const step = sub(grab.target, cur);
    const dist = len(step);
    const max = MAX_DRAG_FRACTION * grab.depth;
    const p =
        dist > max
            ? {
                  x: cur.x + (step.x / dist) * max,
                  y: cur.y + (step.y / dist) * max,
                  z: cur.z + (step.z / dist) * max,
              }
            : grab.target;
    grab.anchor.setTargetTransform({ p, q: IDENT }, dt, true);
}

/** Release the grab, destroying the joint and its anchor. */
export function endGrab(grab: Grab): void {
    grab.joint.destroy();
    grab.anchor.destroy();
}
