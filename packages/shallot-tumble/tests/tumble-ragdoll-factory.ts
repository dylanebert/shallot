// Shared ragdoll-building factory for the `ragdoll-ragdoll` and `determinism-falling-ragdolls` gym twins,
// ported near-verbatim from tumble.js's `samples/src/samples/human.ts`. Eleven axis-aligned capsule bones
// tied by the standard ragdoll joint pattern: spherical shoulders, hips, and neck (cone + twist limited,
// lightly motor-damped), revolute elbows and knees (one-way bend), and a filter joint between the thighs
// so the legs don't collide. Built upright at `origin`.

import { type Body, BodyType, type Vec3, type World } from "@dylanebert/shallot/tumble/core";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/** A capsule bone: endpoints `a`→`b` and radius `r`, in humanoid-local space (feet near y = 0). */
type Bone = { a: Vec3; b: Vec3; r: number };

const mid = (a: Vec3, b: Vec3): Vec3 => ({
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
});
const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });

// Bones laid out standing, ~1.85 units tall, feet at y ≈ 0.1.
const BONES: Record<string, Bone> = {
    pelvis: { a: { x: -0.13, y: 0.95, z: 0 }, b: { x: 0.13, y: 0.95, z: 0 }, r: 0.13 },
    chest: { a: { x: 0, y: 1.05, z: 0 }, b: { x: 0, y: 1.45, z: 0 }, r: 0.16 },
    head: { a: { x: 0, y: 1.6, z: 0 }, b: { x: 0, y: 1.75, z: 0 }, r: 0.13 },
    upperArmL: { a: { x: -0.18, y: 1.45, z: 0 }, b: { x: -0.45, y: 1.2, z: 0 }, r: 0.07 },
    lowerArmL: { a: { x: -0.45, y: 1.2, z: 0 }, b: { x: -0.6, y: 0.92, z: 0 }, r: 0.06 },
    upperArmR: { a: { x: 0.18, y: 1.45, z: 0 }, b: { x: 0.45, y: 1.2, z: 0 }, r: 0.07 },
    lowerArmR: { a: { x: 0.45, y: 1.2, z: 0 }, b: { x: 0.6, y: 0.92, z: 0 }, r: 0.06 },
    thighL: { a: { x: -0.12, y: 0.9, z: 0 }, b: { x: -0.16, y: 0.5, z: 0 }, r: 0.1 },
    calfL: { a: { x: -0.16, y: 0.5, z: 0 }, b: { x: -0.18, y: 0.12, z: 0 }, r: 0.08 },
    thighR: { a: { x: 0.12, y: 0.9, z: 0 }, b: { x: 0.16, y: 0.5, z: 0 }, r: 0.1 },
    calfR: { a: { x: 0.16, y: 0.5, z: 0 }, b: { x: 0.18, y: 0.12, z: 0 }, r: 0.08 },
};

/**
 * Build a ragdoll into `world`, its feet near `origin`. Returns the bone bodies in a fixed order —
 * creation order is load-bearing for the gold hash.
 * @example const bones = buildHuman(world, { x: 0, y: 2.5, z: 0 });
 */
export function buildHuman(world: World, origin: Vec3): Body[] {
    const bodies: Record<string, Body> = {};
    for (const name in BONES) {
        const bone = BONES[name];
        const center = mid(bone.a, bone.b);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: add(origin, center),
            angularDamping: 0.3,
            linearDamping: 0.1,
        });
        body.createCapsule(
            { density: 1 },
            { center1: sub(bone.a, center), center2: sub(bone.b, center), radius: bone.r },
        );
        bodies[name] = body;
    }

    const worldPivot = (p: Vec3): Vec3 => add(origin, p);
    const ball = (a: Body, b: Body, p: Vec3): void => {
        const pivot = worldPivot(p);
        world.createSphericalJoint(a, b, {
            localFrameA: { p: a.getLocalPoint(pivot), q: IDENT },
            localFrameB: { p: b.getLocalPoint(pivot), q: IDENT },
            enableConeLimit: true,
            coneAngle: 0.9,
            enableTwistLimit: true,
            lowerTwistAngle: -0.4,
            upperTwistAngle: 0.4,
            enableMotor: true,
            maxMotorTorque: 1.5,
            motorVelocity: { x: 0, y: 0, z: 0 },
        });
    };
    const hinge = (a: Body, b: Body, p: Vec3): void => {
        const pivot = worldPivot(p);
        world.createRevoluteJoint(a, b, {
            localFrameA: { p: a.getLocalPoint(pivot), q: IDENT },
            localFrameB: { p: b.getLocalPoint(pivot), q: IDENT },
            enableLimit: true,
            lowerAngle: -0.1,
            upperAngle: 2.2,
            enableMotor: true,
            maxMotorTorque: 1.5,
            motorSpeed: 0,
        });
    };

    ball(bodies.pelvis, bodies.chest, { x: 0, y: 1.0, z: 0 });
    ball(bodies.chest, bodies.head, { x: 0, y: 1.55, z: 0 });
    ball(bodies.chest, bodies.upperArmL, { x: -0.18, y: 1.45, z: 0 });
    ball(bodies.chest, bodies.upperArmR, { x: 0.18, y: 1.45, z: 0 });
    hinge(bodies.upperArmL, bodies.lowerArmL, { x: -0.45, y: 1.2, z: 0 });
    hinge(bodies.upperArmR, bodies.lowerArmR, { x: 0.45, y: 1.2, z: 0 });
    ball(bodies.pelvis, bodies.thighL, { x: -0.12, y: 0.9, z: 0 });
    ball(bodies.pelvis, bodies.thighR, { x: 0.12, y: 0.9, z: 0 });
    hinge(bodies.thighL, bodies.calfL, { x: -0.16, y: 0.5, z: 0 });
    hinge(bodies.thighR, bodies.calfR, { x: 0.16, y: 0.5, z: 0 });
    world.createFilterJoint(bodies.thighL, bodies.thighR);

    return Object.keys(BONES).map((name) => bodies[name]);
}
