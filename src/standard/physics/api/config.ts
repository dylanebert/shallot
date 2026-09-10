import type { Manifold } from "../collision/contact";
import type { PlaneResult } from "../collision/mover";
import type { EntityId } from "../common/ids";
import type { Pos, Quat, Transform, Vec3 } from "../common/math";
import type { Shape as ShapeRecord } from "../shapes/shape";
import type { JointDef, Joint as JointRecord } from "../solver/joint";
import type { WorldState } from "../world/world";
import type { Body } from "./body";
import type { Joint } from "./joint";
import type { Contact, Shape } from "./shape";

export function makeShapeId(world: WorldState, shape: ShapeRecord): EntityId {
    return { index1: shape.id + 1, world0: world.worldId, generation: shape.generation };
}

export function makeJointId(world: WorldState, joint: JointRecord): EntityId {
    return { index1: joint.jointId + 1, world0: world.worldId, generation: joint.generation };
}

/** Options shared by every joint (all optional; ported defaults fill the rest). */
export type BaseJointConfig = {
    localFrameA: Transform;
    localFrameB: Transform;
    collideConnected: boolean;
    constraintHertz: number;
    constraintDampingRatio: number;
    forceThreshold: number;
    torqueThreshold: number;
    drawScale: number;
    userData: unknown;
};

export function cloneQuat(q: Quat): Quat {
    return { v: { x: q.v.x, y: q.v.y, z: q.v.z }, s: q.s };
}

export function cloneTransform(t: Transform): Transform {
    return { p: { x: t.p.x, y: t.p.y, z: t.p.z }, q: cloneQuat(t.q) };
}

/** Resolve the shared base joint definition from a config, filling gaps from the ported defaults. */
export function baseJointDef(
    bodyA: Body,
    bodyB: Body,
    cfg: Partial<BaseJointConfig>,
    d: JointDef,
): JointDef {
    return {
        bodyIdA: bodyA.id.index1 - 1,
        bodyIdB: bodyB.id.index1 - 1,
        localFrameA: cfg.localFrameA ?? d.localFrameA,
        localFrameB: cfg.localFrameB ?? d.localFrameB,
        forceThreshold: cfg.forceThreshold ?? d.forceThreshold,
        torqueThreshold: cfg.torqueThreshold ?? d.torqueThreshold,
        constraintHertz: cfg.constraintHertz ?? d.constraintHertz,
        constraintDampingRatio: cfg.constraintDampingRatio ?? d.constraintDampingRatio,
        drawScale: cfg.drawScale ?? d.drawScale,
        collideConnected: cfg.collideConnected ?? d.collideConnected,
        userData: cfg.userData ?? d.userData,
    };
}

/** Revolute-specific options, on top of the shared base (all optional). */
export type RevoluteJointConfig = BaseJointConfig & {
    targetAngle: number;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    enableLimit: boolean;
    lowerAngle: number;
    upperAngle: number;
    enableMotor: boolean;
    maxMotorTorque: number;
    motorSpeed: number;
};

/** Weld-specific options, on top of the shared base (all optional). */
export type WeldJointConfig = BaseJointConfig & {
    linearHertz: number;
    linearDampingRatio: number;
    angularHertz: number;
    angularDampingRatio: number;
};

/** Parallel-specific options, on top of the shared base (all optional). */
export type ParallelJointConfig = BaseJointConfig & {
    hertz: number;
    dampingRatio: number;
    maxTorque: number;
};

/** Motor-specific options, on top of the shared base (all optional). */
export type MotorJointConfig = BaseJointConfig & {
    linearVelocity: Vec3;
    maxVelocityForce: number;
    angularVelocity: Vec3;
    maxVelocityTorque: number;
    linearHertz: number;
    linearDampingRatio: number;
    maxSpringForce: number;
    angularHertz: number;
    angularDampingRatio: number;
    maxSpringTorque: number;
};

/** Distance-specific options, on top of the shared base (all optional). */
export type DistanceJointConfig = BaseJointConfig & {
    length: number;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    lowerSpringForce: number;
    upperSpringForce: number;
    enableLimit: boolean;
    minLength: number;
    maxLength: number;
    enableMotor: boolean;
    maxMotorForce: number;
    motorSpeed: number;
};

/** Prismatic-specific options, on top of the shared base (all optional). */
export type PrismaticJointConfig = BaseJointConfig & {
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    targetTranslation: number;
    enableLimit: boolean;
    lowerTranslation: number;
    upperTranslation: number;
    enableMotor: boolean;
    maxMotorForce: number;
    motorSpeed: number;
};

/** Spherical-specific options, on top of the shared base (all optional). */
export type SphericalJointConfig = BaseJointConfig & {
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    targetRotation: Quat;
    enableConeLimit: boolean;
    coneAngle: number;
    enableTwistLimit: boolean;
    lowerTwistAngle: number;
    upperTwistAngle: number;
    enableMotor: boolean;
    maxMotorTorque: number;
    motorVelocity: Vec3;
};

/** Wheel-specific options, on top of the shared base (all optional). */
export type WheelJointConfig = BaseJointConfig & {
    enableSuspensionSpring: boolean;
    suspensionHertz: number;
    suspensionDampingRatio: number;
    enableSuspensionLimit: boolean;
    lowerSuspensionLimit: number;
    upperSuspensionLimit: number;
    enableSpinMotor: boolean;
    maxSpinTorque: number;
    spinSpeed: number;
    enableSteering: boolean;
    steeringHertz: number;
    steeringDampingRatio: number;
    targetSteeringAngle: number;
    maxSteeringTorque: number;
    enableSteeringLimit: boolean;
    lowerSteeringLimit: number;
    upperSteeringLimit: number;
};

/** A single ray/shape-cast hit reported to a query callback. */
export type CastHit = {
    shape: Shape;
    /** World hit point. */
    point: Pos;
    /** World surface normal at the hit. */
    normal: Vec3;
    /** Fraction of the input translation at the hit. */
    fraction: number;
    /** User material id at the hit (per-triangle for mesh/height-field/child-mesh shapes). */
    userMaterialId: bigint;
    /** Triangle index for mesh/height-field hits, else -1/0. */
    triangleIndex: number;
    /** Compound child index for compound hits, else 0. */
    childIndex: number;
};

/** The closest hit returned by {@link World.castRayClosest} (b3RayResult). `hit` false ⇒ invalid. */
export type RayResult = {
    shape: Shape | null;
    point: Pos;
    normal: Vec3;
    fraction: number;
    userMaterialId: bigint;
    triangleIndex: number;
    childIndex: number;
    hit: boolean;
};

/**
 * Called once per ray/shape-cast hit. Return the new max fraction to clip the query (typically
 * `hit.fraction` for closest-hit), 0 to stop immediately, 1 to continue, or -1 to ignore this shape.
 */
export type CastCallback = (hit: CastHit) => number;

/** Called once per overlapping shape; return false to stop the query. */
export type OverlapCallback = (shape: Shape) => boolean;

/** The closest hit returned by {@link Body.castRay}/{@link Body.castShape}. `hit` false ⇒ invalid. */
export type BodyCastHit = {
    shape: Shape | null;
    point: Pos;
    normal: Vec3;
    fraction: number;
    triangleIndex: number;
    userMaterialId: bigint;
    hit: boolean;
};

/**
 * Called once per shape a mover touches, with that shape and its collision planes (see
 * {@link solvePlanes}). Return false to stop the query. Used by {@link World.collideMover}.
 */
export type PlaneResultCallback = (shape: Shape, planes: PlaneResult[]) => boolean;

/** Per-shape accept filter for {@link World.castMover}; return false to skip the shape. */
export type MoverFilterCallback = (shape: Shape) => boolean;

/** One collision plane between a mover and a body shape, from {@link Body.collideMover}. */
export type BodyPlane = {
    shape: Shape;
    plane: PlaneResult;
};

/** A sensor begin- or end-touch event (b3SensorBeginTouchEvent / b3SensorEndTouchEvent). */
export type SensorTouchEvent = { sensor: Shape; visitor: Shape };

/** Sensor events since the last step (b3SensorEvents), from {@link World.getSensorEvents}. */
export type SensorEvents = { beginEvents: SensorTouchEvent[]; endEvents: SensorTouchEvent[] };

/** A contact begin- or end-touch event (b3ContactBeginTouchEvent / b3ContactEndTouchEvent). */
export type ContactTouchEvent = { shapeA: Shape; shapeB: Shape; contact: Contact };

/** A contact hit event (b3ContactHitEvent): a collision faster than the world hit threshold. */
export type ContactHitEvent = {
    shapeA: Shape;
    shapeB: Shape;
    contact: Contact;
    /** Mid-point between the two surfaces at the start of the step. */
    point: Pos;
    /** Normal pointing from shape A to shape B. */
    normal: Vec3;
    /** The speed the shapes approached at, always positive (m/s). */
    approachSpeed: number;
    userMaterialIdA: bigint;
    userMaterialIdB: bigint;
};

/** Contact events since the last {@link World.step} (b3ContactEvents). */
export type ContactEvents = {
    beginEvents: ContactTouchEvent[];
    endEvents: ContactTouchEvent[];
    hitEvents: ContactHitEvent[];
};

/** A body move event (b3BodyMoveEvent): a body that moved this step. */
export type BodyMoveEvent = {
    body: Body;
    transform: Transform;
    userData: unknown;
    /** Did the body fall asleep this step? Sleep the associated game object too. */
    fellAsleep: boolean;
};

/**
 * Body events since the last {@link World.step} (b3BodyEvents). `moveEvents` is a reused pool of
 * length `count` (only bodies that moved) — valid until the next step or the next `getBodyEvents`.
 */
export type BodyEvents = { moveEvents: BodyMoveEvent[]; count: number };

/** A joint event (b3JointEvent): an awake joint over its force/torque threshold. */
export type JointEvent = { joint: Joint; userData: unknown };

/** Contact data from {@link Contact.getData} (b3ContactData). Manifolds point to internal data. */
export type ContactData = { contact: Contact; shapeA: Shape; shapeB: Shape; manifolds: Manifold[] };
