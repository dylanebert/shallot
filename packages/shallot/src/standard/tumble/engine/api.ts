// The public surface: thin handle classes over the internal id/record model. A World/Body/Shape
// instance holds only an id and delegates to the internal free functions; all state lives in the
// solver-set columns. Stale handles self-invalidate through the generation stored in the id (the
// planck/rapier idiom). Definitions are plain partial data merged over the ported defaults.
//
// This is authoring ergonomics only — the internals stay op-for-op faithful to Box3D regardless.
// The step and the reads that depend on it (velocities, awake state) arrive with the solver stage.

import { NULL_INDEX } from "./array";
import {
    type BodyPlaneResult,
    type Body as BodyRecord,
    bodyApplyAngularImpulse,
    bodyApplyForce,
    bodyApplyForceToCenter,
    bodyApplyLinearImpulse,
    bodyApplyLinearImpulseToCenter,
    bodyApplyTorque,
    bodyCastRay,
    bodyCastShape,
    bodyCollideMover,
    bodyGetClosestPoint,
    bodyOverlapShape,
    bodySetAngularVelocity,
    bodySetAwake,
    bodySetLinearVelocity,
    bodySetTargetTransform,
    bodySetTransform,
    bodySetType,
    createBody,
    destroyBody,
    getBodySim,
    getBodyState,
    getBodyTransformQuick,
    getMassData,
    makeBodyId,
    updateBodyMassData,
} from "./body";
import type { CompoundData } from "./compound";
import type { Manifold, ManifoldPoint } from "./contact";
import { DEFAULT_MASK_BITS, HUGE, LINEAR_SLOP, SetType } from "./core";
import type { ShapeProxy } from "./distance";
import {
    createDistanceJoint,
    type DistanceJoint as DistanceJointData,
    type DistanceJointDef,
    defaultDistanceJointDef,
    distanceJointCurrentLength,
} from "./distanceJoint";
import { type DebugDraw, worldDraw } from "./draw";
import type { Capsule, MassData, Sphere } from "./geometry";
import type { HeightFieldData } from "./heightfield";
import type { HullData } from "./hull";
import type { EntityId } from "./ids";
import {
    createFilterJoint,
    defaultJointDef,
    destroyJointInternal,
    getJointAngularSeparation,
    getJointConstraintForce,
    getJointConstraintTorque,
    getJointLinearSeparation,
    getJointSim,
    type JointDef,
    type Joint as JointRecord,
    type JointSim,
    type JointType,
    setJointCollideConnected,
    wakeJointBodies,
} from "./joint";
import {
    type AABB,
    clampf,
    f32,
    froundConfig,
    invTransformWorldPoint,
    PI,
    type Pos,
    type Quat,
    type Transform,
    type Vec3,
    vec3,
    type WorldTransform,
} from "./math";
import type { MeshData } from "./mesh";
import {
    createMotorJoint,
    defaultMotorJointDef,
    type MotorJoint as MotorJointData,
    type MotorJointDef,
} from "./motorJoint";
import type { PlaneResult } from "./mover";
import {
    createParallelJoint,
    defaultParallelJointDef,
    type ParallelJoint as ParallelJointData,
    type ParallelJointDef,
} from "./parallelJoint";
import {
    createPrismaticJoint,
    defaultPrismaticJointDef,
    type PrismaticJoint as PrismaticJointData,
    type PrismaticJointDef,
    prismaticJointSpeed,
    prismaticJointTranslation,
} from "./prismaticJoint";
import type { Profile } from "./profile";
import {
    castMover as castMoverInternal,
    castRayClosest as castRayClosestInternal,
    castRay as castRayInternal,
    castShape as castShapeInternal,
    collideMover as collideMoverInternal,
    overlapAABB as overlapAABBInternal,
    overlapShapeQuery,
} from "./query";
import {
    createRevoluteJoint,
    defaultRevoluteJointDef,
    type RevoluteJoint as RevoluteJointData,
    type RevoluteJointDef,
    revoluteJointAngle,
} from "./revoluteJoint";
import {
    computeShapeMass,
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
    destroyShape,
    getSensorData,
    isSensorShape,
    type Shape as ShapeRecord,
} from "./shape";
import {
    createSphericalJoint,
    defaultSphericalJointDef,
    type SphericalJoint as SphericalJointData,
    type SphericalJointDef,
    sphericalJointConeAngle,
    sphericalJointTwistAngle,
} from "./sphericalJoint";
import { step as stepWorld } from "./step";
import type { TreeStats } from "./tree";
import {
    type BodyDef,
    type BodyType,
    defaultBodyDef,
    defaultQueryFilter,
    defaultShapeDef,
    defaultWorldDef,
    type QueryFilter,
    type ShapeDef,
    type ShapeType,
    type WorldDef,
} from "./types";
import {
    createWeldJoint,
    defaultWeldJointDef,
    type WeldJoint as WeldJointData,
    type WeldJointDef,
} from "./weldJoint";
import {
    createWheelJoint,
    defaultWheelJointDef,
    type WheelJoint as WheelJointData,
    type WheelJointDef,
    wheelJointSpinSpeed,
    wheelJointSteeringAngle,
} from "./wheelJoint";
import {
    type Counters,
    createWorld,
    destroyWorld,
    getWorld,
    type WorldId,
    type WorldState,
    worldCounters,
    worldIsValid,
    worldProfile,
} from "./world";

function makeShapeId(world: WorldState, shape: ShapeRecord): EntityId {
    return { index1: shape.id + 1, world0: world.worldId, generation: shape.generation };
}

function makeJointId(world: WorldState, joint: JointRecord): EntityId {
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

function cloneQuat(q: Quat): Quat {
    return { v: { x: q.v.x, y: q.v.y, z: q.v.z }, s: q.s };
}

function cloneTransform(t: Transform): Transform {
    return { p: { x: t.p.x, y: t.p.y, z: t.p.z }, q: cloneQuat(t.q) };
}

/** Resolve the shared base joint definition from a config, filling gaps from the ported defaults. */
function baseJointDef(
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

/**
 * A transient contact handle (b3ContactId) carried by contact events. A contact may be destroyed by
 * a world modification or a step, so call {@link isValid} before {@link getData}.
 */
export class Contact {
    /** @internal */
    readonly world: WorldState;
    /** @internal */
    readonly id: EntityId;

    /** @internal carried by contact events */
    constructor(world: WorldState, id: EntityId) {
        this.world = world;
        this.id = id;
    }

    /** @returns whether this contact still exists (b3Contact_IsValid). */
    isValid(): boolean {
        if (this.world.inUse === false) {
            return false;
        }
        const contactId = this.id.index1 - 1;
        if (contactId < 0 || contactId >= this.world.contacts.length) {
            return false;
        }
        const contact = this.world.contacts[contactId];
        if (contact.contactId === NULL_INDEX) {
            return false;
        }
        return this.id.generation === contact.generation;
    }

    /** @returns the two shapes and current manifold(s) of this contact (b3Contact_GetData). */
    getData(): ContactData {
        const world = this.world;
        const contact = world.contacts[this.id.index1 - 1];
        const shapeA = world.shapes[contact.shapeIdA];
        const shapeB = world.shapes[contact.shapeIdB];
        return {
            contact: this,
            shapeA: new Shape(world, makeShapeId(world, shapeA)),
            shapeB: new Shape(world, makeShapeId(world, shapeB)),
            // The contact's manifolds are column-resident views over wasm memory the next step
            // overwrites; snapshot them into plain objects the caller can hold (b3Contact_GetData).
            manifolds: snapshotManifolds(contact.manifolds, contact.manifoldCount),
        };
    }
}

/** Copy `count` column-backed manifold views into detached plain `Manifold` objects. */
function snapshotManifolds(manifolds: Manifold[], count: number): Manifold[] {
    const out: Manifold[] = new Array(count);
    for (let m = 0; m < count; ++m) {
        const src = manifolds[m];
        const pc = src.pointCount;
        const points: ManifoldPoint[] = new Array(pc);
        for (let p = 0; p < pc; ++p) {
            const sp = src.points[p];
            points[p] = {
                anchorA: sp.anchorA,
                anchorB: sp.anchorB,
                separation: sp.separation,
                baseSeparation: sp.baseSeparation,
                normalImpulse: sp.normalImpulse,
                totalNormalImpulse: sp.totalNormalImpulse,
                normalVelocity: sp.normalVelocity,
                featureId: sp.featureId,
                triangleIndex: sp.triangleIndex,
                persisted: sp.persisted,
            };
        }
        out[m] = {
            points,
            normal: src.normal,
            twistImpulse: src.twistImpulse,
            frictionImpulse: src.frictionImpulse,
            rollingImpulse: src.rollingImpulse,
            pointCount: pc,
        };
    }
    return out;
}

/** A simulation world: bodies, shapes, and the broad-phase. */
export class World {
    /** @internal the underlying world state */
    readonly state: WorldState;
    private readonly _worldId: WorldId;
    // Reused wrappers over the internal move-event pool, so getBodyEvents allocates nothing in steady
    // state (matching the internal pool). Rebuilt lazily; valid until the next step or getBodyEvents.
    private readonly _moveEventPool: BodyMoveEvent[] = [];

    constructor(def: Partial<WorldDef> = {}) {
        this._worldId = createWorld({ ...defaultWorldDef(), ...def });
        // getWorld succeeds immediately after creation.
        this.state = getWorld(this._worldId) as WorldState;
    }

    /** @internal wrap an existing world state as a handle (e.g. Joint.getWorld). */
    static _wrap(state: WorldState): World {
        const world = Object.create(World.prototype) as {
            state: WorldState;
            _worldId: WorldId;
        };
        world.state = state;
        world._worldId = { index1: state.worldId + 1, generation: state.generation };
        return world as unknown as World;
    }

    /** @returns whether this world has not been destroyed. */
    isValid(): boolean {
        return worldIsValid(this._worldId);
    }

    /** Destroy this world and every body and shape in it. */
    destroy(): void {
        destroyWorld(this.state);
    }

    /** Create a body from a (partial) definition. */
    createBody(def: Partial<BodyDef> = {}): Body {
        const bodyId = createBody(this.state, { ...defaultBodyDef(), ...def });
        return new Body(this.state, makeBodyId(this.state, bodyId));
    }

    /**
     * Advance the simulation by `timeStep` seconds, split into `subStepCount` solver sub-steps.
     * @example world.step(1 / 60, 4)
     */
    step(timeStep: number, subStepCount = 4): void {
        stepWorld(this.state, timeStep, subStepCount);
    }

    /**
     * Walk every shape and joint whose fat AABB overlaps `draw.drawingBounds`, resolving each to the
     * typed callbacks on `draw` (b3World_Draw). A renderer supplies the callbacks; a headless caller
     * can count draws. Read-only — never advances the simulation.
     * @example const d = { ...defaultDebugDraw(), drawShapes: true, drawSolidSphere }; world.draw(d)
     */
    draw(draw: DebugDraw, maskBits: bigint = DEFAULT_MASK_BITS): void {
        worldDraw(this.state, draw, maskBits);
    }

    /**
     * Sensor begin/end touch events accumulated during the last {@link step} (b3World_GetSensorEvents).
     * Valid until the next step; end events read from the previous buffer, so they survive one step.
     * @example for (const e of world.getSensorEvents().beginEvents) onEnter(e.sensor, e.visitor)
     */
    getSensorEvents(): SensorEvents {
        const state = this.state;
        const wrap = (e: {
            sensorShapeId: EntityId;
            visitorShapeId: EntityId;
        }): SensorTouchEvent => ({
            sensor: new Shape(state, e.sensorShapeId),
            visitor: new Shape(state, e.visitorShapeId),
        });
        // Careful to read the previous end-event buffer (the swap already happened this step).
        const endEvents = state.sensorEndEvents[1 - state.endEventArrayIndex];
        return {
            beginEvents: state.sensorBeginEvents.map(wrap),
            endEvents: endEvents.map(wrap),
        };
    }

    /**
     * Contact begin/end/hit events from the last {@link step} (b3World_GetContactEvents). Begin/end
     * carry {@link Contact} handles (validate before use); hit events carry the impact point, normal,
     * and approach speed. End events read the previous buffer, so they survive one step.
     * @example for (const e of world.getContactEvents().hitEvents) spark(e.point, e.approachSpeed)
     */
    getContactEvents(): ContactEvents {
        const state = this.state;
        const wrapTouch = (e: {
            shapeIdA: EntityId;
            shapeIdB: EntityId;
            contactId: EntityId;
        }): ContactTouchEvent => ({
            shapeA: new Shape(state, e.shapeIdA),
            shapeB: new Shape(state, e.shapeIdB),
            contact: new Contact(state, e.contactId),
        });
        // Careful to read the previous end-event buffer (the swap already happened this step).
        const endEvents = state.contactEndEvents[1 - state.endEventArrayIndex];
        return {
            beginEvents: state.contactBeginEvents.map(wrapTouch),
            endEvents: endEvents.map(wrapTouch),
            hitEvents: state.contactHitEvents.map((e) => ({
                shapeA: new Shape(state, e.shapeIdA),
                shapeB: new Shape(state, e.shapeIdB),
                contact: new Contact(state, e.contactId),
                point: { ...e.point },
                normal: { ...e.normal },
                approachSpeed: e.approachSpeed,
                userMaterialIdA: e.userMaterialIdA,
                userMaterialIdB: e.userMaterialIdB,
            })),
        };
    }

    /**
     * Body move events from the last {@link step} (b3World_GetBodyEvents) — every body that moved,
     * for bulk-syncing game object transforms (cheaper than per-body {@link Body.getTransform}). The
     * `moveEvents` array is a reused pool valid until the next step; keep `userData` to route each.
     * @example for (const e of world.getBodyEvents().moveEvents) sync(e.userData, e.transform)
     */
    getBodyEvents(): BodyEvents {
        const state = this.state;
        const count = state.bodyMoveCount;
        const pool = this._moveEventPool;
        while (pool.length < count) {
            pool.push({
                body: new Body(state, { index1: 0, world0: state.worldId, generation: 0 }),
                transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
                userData: null,
                fellAsleep: false,
            });
        }
        pool.length = count;
        for (let i = 0; i < count; ++i) {
            const rec = state.bodyMoveEvents[i];
            const ev = pool[i];
            ev.body.id.index1 = rec.bodyId + 1;
            ev.body.id.generation = rec.generation;
            // Reference the internal pooled transform directly (valid until the next step).
            ev.transform = rec.transform;
            ev.userData = rec.userData;
            ev.fellAsleep = rec.fellAsleep;
        }
        return { moveEvents: pool, count };
    }

    /**
     * Joint events from the last {@link step} (b3World_GetJointEvents): awake joints whose force or
     * torque exceeded the threshold set via {@link Joint.setForceThreshold}/{@link Joint.setTorqueThreshold}.
     */
    getJointEvents(): JointEvent[] {
        const state = this.state;
        return state.jointEvents.map((e) => ({
            joint: new Joint(state, e.jointId),
            userData: e.userData,
        }));
    }

    /** @returns the collision speed above which a contact reports a hit event (b3World_GetHitEventThreshold). */
    getHitEventThreshold(): number {
        return this.state.hitEventThreshold;
    }

    /** Set the collision speed above which a contact reports a hit event (b3World_SetHitEventThreshold). */
    setHitEventThreshold(value: number): void {
        this.state.hitEventThreshold = f32(value);
    }

    /**
     * Connect two bodies with a revolute (hinge) joint.
     * @example world.createRevoluteJoint(anchor, arm, { localFrameA: { p, q } })
     */
    createRevoluteJoint(
        bodyA: Body,
        bodyB: Body,
        cfg: Partial<RevoluteJointConfig> = {},
    ): RevoluteJoint {
        cfg = froundConfig(cfg);
        const d = defaultRevoluteJointDef(defaultJointDef());
        const def: RevoluteJointDef = {
            base: baseJointDef(bodyA, bodyB, cfg, d.base),
            targetAngle: cfg.targetAngle ?? d.targetAngle,
            enableSpring: cfg.enableSpring ?? d.enableSpring,
            hertz: cfg.hertz ?? d.hertz,
            dampingRatio: cfg.dampingRatio ?? d.dampingRatio,
            enableLimit: cfg.enableLimit ?? d.enableLimit,
            lowerAngle: cfg.lowerAngle ?? d.lowerAngle,
            upperAngle: cfg.upperAngle ?? d.upperAngle,
            enableMotor: cfg.enableMotor ?? d.enableMotor,
            maxMotorTorque: cfg.maxMotorTorque ?? d.maxMotorTorque,
            motorSpeed: cfg.motorSpeed ?? d.motorSpeed,
        };
        const { joint } = createRevoluteJoint(this.state, def);
        return new RevoluteJoint(this.state, makeJointId(this.state, joint));
    }

    /**
     * Rigidly fix two bodies (position + orientation), optionally softened by linear/angular springs.
     * @example world.createWeldJoint(a, b, { localFrameA, localFrameB })
     */
    createWeldJoint(bodyA: Body, bodyB: Body, cfg: Partial<WeldJointConfig> = {}): WeldJoint {
        cfg = froundConfig(cfg);
        const d = defaultWeldJointDef(defaultJointDef());
        const def: WeldJointDef = {
            base: baseJointDef(bodyA, bodyB, cfg, d.base),
            linearHertz: cfg.linearHertz ?? d.linearHertz,
            linearDampingRatio: cfg.linearDampingRatio ?? d.linearDampingRatio,
            angularHertz: cfg.angularHertz ?? d.angularHertz,
            angularDampingRatio: cfg.angularDampingRatio ?? d.angularDampingRatio,
        };
        const { joint } = createWeldJoint(this.state, def);
        return new WeldJoint(this.state, makeJointId(this.state, joint));
    }

    /**
     * Hold two bodies' local-z frames collinear (a soft angular constraint capped by maxTorque).
     * @example world.createParallelJoint(a, b, { hertz: 2 })
     */
    createParallelJoint(
        bodyA: Body,
        bodyB: Body,
        cfg: Partial<ParallelJointConfig> = {},
    ): ParallelJoint {
        cfg = froundConfig(cfg);
        const d = defaultParallelJointDef(defaultJointDef());
        const def: ParallelJointDef = {
            base: baseJointDef(bodyA, bodyB, cfg, d.base),
            hertz: cfg.hertz ?? d.hertz,
            dampingRatio: cfg.dampingRatio ?? d.dampingRatio,
            maxTorque: cfg.maxTorque ?? d.maxTorque,
        };
        const { joint } = createParallelJoint(this.state, def);
        return new ParallelJoint(this.state, makeJointId(this.state, joint));
    }

    /**
     * Drive two bodies toward target relative linear/angular velocities (each capped by a max effort).
     * @example world.createMotorJoint(a, b, { angularVelocity: { x: 0, y: 0, z: 1 }, maxVelocityTorque: 100 })
     */
    createMotorJoint(bodyA: Body, bodyB: Body, cfg: Partial<MotorJointConfig> = {}): MotorJoint {
        cfg = froundConfig(cfg);
        const d = defaultMotorJointDef(defaultJointDef());
        const def: MotorJointDef = {
            base: baseJointDef(bodyA, bodyB, cfg, d.base),
            linearVelocity: cfg.linearVelocity ?? d.linearVelocity,
            maxVelocityForce: cfg.maxVelocityForce ?? d.maxVelocityForce,
            angularVelocity: cfg.angularVelocity ?? d.angularVelocity,
            maxVelocityTorque: cfg.maxVelocityTorque ?? d.maxVelocityTorque,
            linearHertz: cfg.linearHertz ?? d.linearHertz,
            linearDampingRatio: cfg.linearDampingRatio ?? d.linearDampingRatio,
            maxSpringForce: cfg.maxSpringForce ?? d.maxSpringForce,
            angularHertz: cfg.angularHertz ?? d.angularHertz,
            angularDampingRatio: cfg.angularDampingRatio ?? d.angularDampingRatio,
            maxSpringTorque: cfg.maxSpringTorque ?? d.maxSpringTorque,
        };
        const { joint } = createMotorJoint(this.state, def);
        return new MotorJoint(this.state, makeJointId(this.state, joint));
    }

    /**
     * Constrain the distance between two anchor points — rigid, or a soft spring with limits + motor.
     * @example world.createDistanceJoint(a, b, { length: 2, enableSpring: true, hertz: 4, dampingRatio: 0.5 })
     */
    createDistanceJoint(
        bodyA: Body,
        bodyB: Body,
        cfg: Partial<DistanceJointConfig> = {},
    ): DistanceJoint {
        cfg = froundConfig(cfg);
        const d = defaultDistanceJointDef(defaultJointDef());
        const def: DistanceJointDef = {
            base: baseJointDef(bodyA, bodyB, cfg, d.base),
            length: cfg.length ?? d.length,
            enableSpring: cfg.enableSpring ?? d.enableSpring,
            hertz: cfg.hertz ?? d.hertz,
            dampingRatio: cfg.dampingRatio ?? d.dampingRatio,
            lowerSpringForce: cfg.lowerSpringForce ?? d.lowerSpringForce,
            upperSpringForce: cfg.upperSpringForce ?? d.upperSpringForce,
            enableLimit: cfg.enableLimit ?? d.enableLimit,
            minLength: cfg.minLength ?? d.minLength,
            maxLength: cfg.maxLength ?? d.maxLength,
            enableMotor: cfg.enableMotor ?? d.enableMotor,
            maxMotorForce: cfg.maxMotorForce ?? d.maxMotorForce,
            motorSpeed: cfg.motorSpeed ?? d.motorSpeed,
        };
        const { joint } = createDistanceJoint(this.state, def);
        return new DistanceJoint(this.state, makeJointId(this.state, joint));
    }

    /**
     * Constrain two bodies to slide along body A's local x-axis, with optional spring/motor/limits.
     * @example world.createPrismaticJoint(base, slider, { enableLimit: true, upperTranslation: 2 })
     */
    createPrismaticJoint(
        bodyA: Body,
        bodyB: Body,
        cfg: Partial<PrismaticJointConfig> = {},
    ): PrismaticJoint {
        cfg = froundConfig(cfg);
        const d = defaultPrismaticJointDef(defaultJointDef());
        const def: PrismaticJointDef = {
            base: baseJointDef(bodyA, bodyB, cfg, d.base),
            enableSpring: cfg.enableSpring ?? d.enableSpring,
            hertz: cfg.hertz ?? d.hertz,
            dampingRatio: cfg.dampingRatio ?? d.dampingRatio,
            targetTranslation: cfg.targetTranslation ?? d.targetTranslation,
            enableLimit: cfg.enableLimit ?? d.enableLimit,
            lowerTranslation: cfg.lowerTranslation ?? d.lowerTranslation,
            upperTranslation: cfg.upperTranslation ?? d.upperTranslation,
            enableMotor: cfg.enableMotor ?? d.enableMotor,
            maxMotorForce: cfg.maxMotorForce ?? d.maxMotorForce,
            motorSpeed: cfg.motorSpeed ?? d.motorSpeed,
        };
        const { joint } = createPrismaticJoint(this.state, def);
        return new PrismaticJoint(this.state, makeJointId(this.state, joint));
    }

    /**
     * Ball-and-socket joint: hold two anchor points together, with optional cone/twist limits + drive.
     * @example world.createSphericalJoint(a, b, { enableConeLimit: true, coneAngle: 0.5 })
     */
    createSphericalJoint(
        bodyA: Body,
        bodyB: Body,
        cfg: Partial<SphericalJointConfig> = {},
    ): SphericalJoint {
        cfg = froundConfig(cfg);
        const d = defaultSphericalJointDef(defaultJointDef());
        const def: SphericalJointDef = {
            base: baseJointDef(bodyA, bodyB, cfg, d.base),
            enableSpring: cfg.enableSpring ?? d.enableSpring,
            hertz: cfg.hertz ?? d.hertz,
            dampingRatio: cfg.dampingRatio ?? d.dampingRatio,
            targetRotation: cfg.targetRotation ?? d.targetRotation,
            enableConeLimit: cfg.enableConeLimit ?? d.enableConeLimit,
            coneAngle: cfg.coneAngle ?? d.coneAngle,
            enableTwistLimit: cfg.enableTwistLimit ?? d.enableTwistLimit,
            lowerTwistAngle: cfg.lowerTwistAngle ?? d.lowerTwistAngle,
            upperTwistAngle: cfg.upperTwistAngle ?? d.upperTwistAngle,
            enableMotor: cfg.enableMotor ?? d.enableMotor,
            maxMotorTorque: cfg.maxMotorTorque ?? d.maxMotorTorque,
            motorVelocity: cfg.motorVelocity ?? d.motorVelocity,
        };
        const { joint } = createSphericalJoint(this.state, def);
        return new SphericalJoint(this.state, makeJointId(this.state, joint));
    }

    /**
     * Car-suspension joint: slide along body A's x (suspension), spin about body B's z, optional steering.
     * @example world.createWheelJoint(chassis, wheel, { enableSpinMotor: true, spinSpeed: 20 })
     */
    createWheelJoint(bodyA: Body, bodyB: Body, cfg: Partial<WheelJointConfig> = {}): WheelJoint {
        cfg = froundConfig(cfg);
        const d = defaultWheelJointDef(defaultJointDef());
        const def: WheelJointDef = {
            base: baseJointDef(bodyA, bodyB, cfg, d.base),
            enableSuspensionSpring: cfg.enableSuspensionSpring ?? d.enableSuspensionSpring,
            suspensionHertz: cfg.suspensionHertz ?? d.suspensionHertz,
            suspensionDampingRatio: cfg.suspensionDampingRatio ?? d.suspensionDampingRatio,
            enableSuspensionLimit: cfg.enableSuspensionLimit ?? d.enableSuspensionLimit,
            lowerSuspensionLimit: cfg.lowerSuspensionLimit ?? d.lowerSuspensionLimit,
            upperSuspensionLimit: cfg.upperSuspensionLimit ?? d.upperSuspensionLimit,
            enableSpinMotor: cfg.enableSpinMotor ?? d.enableSpinMotor,
            maxSpinTorque: cfg.maxSpinTorque ?? d.maxSpinTorque,
            spinSpeed: cfg.spinSpeed ?? d.spinSpeed,
            enableSteering: cfg.enableSteering ?? d.enableSteering,
            steeringHertz: cfg.steeringHertz ?? d.steeringHertz,
            steeringDampingRatio: cfg.steeringDampingRatio ?? d.steeringDampingRatio,
            targetSteeringAngle: cfg.targetSteeringAngle ?? d.targetSteeringAngle,
            maxSteeringTorque: cfg.maxSteeringTorque ?? d.maxSteeringTorque,
            enableSteeringLimit: cfg.enableSteeringLimit ?? d.enableSteeringLimit,
            lowerSteeringLimit: cfg.lowerSteeringLimit ?? d.lowerSteeringLimit,
            upperSteeringLimit: cfg.upperSteeringLimit ?? d.upperSteeringLimit,
        };
        const { joint } = createWheelJoint(this.state, def);
        return new WheelJoint(this.state, makeJointId(this.state, joint));
    }

    /**
     * Connect two bodies solely to suppress collision between them (no constraint).
     * @example world.createFilterJoint(a, b)
     */
    createFilterJoint(bodyA: Body, bodyB: Body, cfg: Partial<BaseJointConfig> = {}): Joint {
        cfg = froundConfig(cfg);
        const def = baseJointDef(bodyA, bodyB, cfg, defaultJointDef());
        const { joint } = createFilterJoint(this.state, def);
        return new Joint(this.state, makeJointId(this.state, joint));
    }

    /** @returns entity counts (bodies, shapes, contacts, joints, islands). */
    getCounters(): Counters {
        return worldCounters(this.state);
    }

    /**
     * @returns the last step's per-phase timings in milliseconds (b3World_GetProfile).
     * @example world.step(1 / 60, 4); console.log(world.getProfile().collide)
     */
    getProfile(): Profile {
        return worldProfile(this.state);
    }

    /** @returns the gravity vector. */
    getGravity(): Vec3 {
        return { ...this.state.gravity };
    }

    /** Set the gravity vector. */
    setGravity(gravity: Vec3): void {
        this.state.gravity = froundConfig({ x: gravity.x, y: gravity.y, z: gravity.z });
    }

    /**
     * Cast a ray from `origin` along `translation`, returning the single closest hit.
     * @example const r = world.castRayClosest(eye, dir); if (r.hit) console.log(r.point)
     */
    castRayClosest(
        origin: Pos,
        translation: Vec3,
        filter: QueryFilter = defaultQueryFilter(),
    ): RayResult {
        const r = castRayClosestInternal(this.state, origin, translation, filter);
        return {
            shape: r.hit ? new Shape(this.state, r.shapeId) : null,
            point: r.point,
            normal: r.normal,
            fraction: r.fraction,
            userMaterialId: r.userMaterialId,
            triangleIndex: r.triangleIndex,
            childIndex: r.childIndex,
            hit: r.hit,
        };
    }

    /**
     * Cast a ray, reporting every hit to `fcn`. Return from `fcn` the new max fraction (see
     * {@link CastCallback}). @returns broad-phase traversal counts.
     */
    castRay(
        origin: Pos,
        translation: Vec3,
        fcn: CastCallback,
        filter: QueryFilter = defaultQueryFilter(),
    ): TreeStats {
        return castRayInternal(
            this.state,
            origin,
            translation,
            filter,
            (id, point, normal, fraction, userMaterialId, triangleIndex, childIndex) =>
                fcn({
                    shape: new Shape(this.state, id),
                    point,
                    normal,
                    fraction,
                    userMaterialId,
                    triangleIndex,
                    childIndex,
                }),
        );
    }

    /** Report every shape whose fat AABB overlaps `box`; return false from `fcn` to stop. */
    overlapAABB(
        box: AABB,
        fcn: OverlapCallback,
        filter: QueryFilter = defaultQueryFilter(),
    ): TreeStats {
        return overlapAABBInternal(this.state, box, filter, (id) => fcn(new Shape(this.state, id)));
    }

    /**
     * Report every shape whose geometry overlaps the convex `proxy` placed at `origin`; return false
     * from `fcn` to stop.
     */
    overlapShape(
        origin: Pos,
        proxy: ShapeProxy,
        fcn: OverlapCallback,
        filter: QueryFilter = defaultQueryFilter(),
    ): TreeStats {
        return overlapShapeQuery(this.state, origin, proxy, filter, (id) =>
            fcn(new Shape(this.state, id)),
        );
    }

    /**
     * Cast the convex `proxy` from `origin` along `translation`, reporting every hit to `fcn` (see
     * {@link CastCallback}). @returns broad-phase traversal counts.
     */
    castShape(
        origin: Pos,
        proxy: ShapeProxy,
        translation: Vec3,
        fcn: CastCallback,
        filter: QueryFilter = defaultQueryFilter(),
    ): TreeStats {
        return castShapeInternal(
            this.state,
            origin,
            proxy,
            translation,
            filter,
            (id, point, normal, fraction, userMaterialId, triangleIndex, childIndex) =>
                fcn({
                    shape: new Shape(this.state, id),
                    point,
                    normal,
                    fraction,
                    userMaterialId,
                    triangleIndex,
                    childIndex,
                }),
        );
    }

    /**
     * Collide a capsule `mover` at `origin` against the world, reporting each touched shape's collision
     * planes to `fcn`. Feed the gathered planes to {@link solvePlanes} to resolve character movement.
     */
    collideMover(
        origin: Pos,
        mover: Capsule,
        fcn: PlaneResultCallback,
        filter: QueryFilter = defaultQueryFilter(),
    ): void {
        collideMoverInternal(this.state, origin, mover, filter, (id, planes) =>
            fcn(new Shape(this.state, id), planes),
        );
    }

    /**
     * Cast a capsule `mover` from `origin` along `translation`, returning the earliest fraction of
     * contact (1 when the path is clear). `fcn` optionally skips shapes per-hit.
     */
    castMover(
        origin: Pos,
        mover: Capsule,
        translation: Vec3,
        filter: QueryFilter = defaultQueryFilter(),
        fcn: MoverFilterCallback | null = null,
    ): number {
        return castMoverInternal(
            this.state,
            origin,
            mover,
            translation,
            filter,
            fcn === null ? null : (id) => fcn(new Shape(this.state, id)),
        );
    }
}

/** A rigid body handle. */
export class Body {
    /** @internal */
    readonly world: WorldState;
    /** @internal */
    readonly id: EntityId;

    /** @internal use World.createBody */
    constructor(world: WorldState, id: EntityId) {
        this.world = world;
        this.id = id;
    }

    private record(): BodyRecord {
        return this.world.bodies[this.id.index1 - 1];
    }

    /** @returns whether this body has not been destroyed and its world is alive. */
    isValid(): boolean {
        if (this.world.inUse === false) {
            return false;
        }
        const i = this.id.index1 - 1;
        if (i < 0 || i >= this.world.bodies.length) {
            return false;
        }
        const body = this.world.bodies[i];
        if (body.setIndex === NULL_INDEX) {
            return false;
        }
        return body.generation === this.id.generation;
    }

    /** Destroy this body, its shapes, contacts, and joints. */
    destroy(): void {
        destroyBody(this.world, this.record());
    }

    /** Attach a sphere shape. */
    createSphere(def: Partial<ShapeDef>, sphere: Sphere): Shape {
        const shape = createSphereShape(
            this.world,
            this.record(),
            { ...defaultShapeDef(), ...def },
            sphere,
        );
        return new Shape(this.world, makeShapeId(this.world, shape as ShapeRecord));
    }

    /** Attach a capsule shape. */
    createCapsule(def: Partial<ShapeDef>, capsule: Capsule): Shape {
        const shape = createCapsuleShape(
            this.world,
            this.record(),
            { ...defaultShapeDef(), ...def },
            capsule,
        );
        return new Shape(this.world, makeShapeId(this.world, shape as ShapeRecord));
    }

    /** Attach a convex-hull shape. */
    createHull(def: Partial<ShapeDef>, hull: HullData): Shape {
        const shape = createHullShape(
            this.world,
            this.record(),
            { ...defaultShapeDef(), ...def },
            hull,
        );
        return new Shape(this.world, makeShapeId(this.world, shape as ShapeRecord));
    }

    /** Attach a static triangle-mesh shape. `mesh` is caller-owned and may be shared across shapes. */
    createMesh(def: Partial<ShapeDef>, mesh: MeshData, scale: Vec3 = { x: 1, y: 1, z: 1 }): Shape {
        const shape = createMeshShape(
            this.world,
            this.record(),
            { ...defaultShapeDef(), ...def },
            mesh,
            scale,
        );
        return new Shape(this.world, makeShapeId(this.world, shape as ShapeRecord));
    }

    /** Attach a static height-field shape. `heightField` is caller-owned and may be shared. */
    createHeightField(def: Partial<ShapeDef>, heightField: HeightFieldData): Shape {
        const shape = createHeightFieldShape(
            this.world,
            this.record(),
            { ...defaultShapeDef(), ...def },
            heightField,
        );
        return new Shape(this.world, makeShapeId(this.world, shape as ShapeRecord));
    }

    /**
     * Attach a static compound shape (a container of child shapes). `compound` is caller-owned and may
     * be shared; the compound's own materials drive contacts, so the def's materials are ignored.
     */
    createCompound(def: Partial<ShapeDef>, compound: CompoundData): Shape {
        const shape = createCompoundShape(
            this.world,
            this.record(),
            { ...defaultShapeDef(), ...def },
            compound,
        );
        return new Shape(this.world, makeShapeId(this.world, shape as ShapeRecord));
    }

    /** @returns the body type (static / kinematic / dynamic). */
    getType(): BodyType {
        return this.record().type;
    }

    /**
     * @returns the body origin position in world space. Pass `out` to fill it instead of allocating
     * (the three.js `getWorldPosition(target)` idiom) for zero-allocation reads in a hot loop.
     */
    getPosition(out?: Pos): Pos {
        const p = getBodyTransformQuick(this.world, this.record()).p;
        if (out === undefined) {
            return { x: p.x, y: p.y, z: p.z };
        }
        out.x = p.x;
        out.y = p.y;
        out.z = p.z;
        return out;
    }

    /** @returns the body rotation. Pass `out` to fill it instead of allocating. */
    getRotation(out?: Quat): Quat {
        const q = getBodyTransformQuick(this.world, this.record()).q;
        if (out === undefined) {
            return { v: { x: q.v.x, y: q.v.y, z: q.v.z }, s: q.s };
        }
        out.v.x = q.v.x;
        out.v.y = q.v.y;
        out.v.z = q.v.z;
        out.s = q.s;
        return out;
    }

    /** @returns the body world transform. Pass `out` to fill it instead of allocating. */
    getTransform(out?: WorldTransform): WorldTransform {
        const t = getBodyTransformQuick(this.world, this.record());
        if (out === undefined) {
            return { p: { x: t.p.x, y: t.p.y, z: t.p.z }, q: { v: { ...t.q.v }, s: t.q.s } };
        }
        out.p.x = t.p.x;
        out.p.y = t.p.y;
        out.p.z = t.p.z;
        out.q.v.x = t.q.v.x;
        out.q.v.y = t.q.v.y;
        out.q.v.z = t.q.v.z;
        out.q.s = t.q.s;
        return out;
    }

    /** @returns the world-space center of mass. */
    getWorldCenterOfMass(): Pos {
        return { ...getBodySim(this.world, this.record()).center };
    }

    /**
     * @returns `worldPoint` expressed in the body's local frame.
     * @example const local = body.getLocalPoint(hit.point);
     */
    getLocalPoint(worldPoint: Pos): Vec3 {
        return invTransformWorldPoint(getBodyTransformQuick(this.world, this.record()), worldPoint);
    }

    /** @returns the body's linear velocity (zero when the body is not awake). */
    getLinearVelocity(): Vec3 {
        const state = getBodyState(this.world, this.record());
        return state === null ? { x: 0, y: 0, z: 0 } : { ...state.linearVelocity };
    }

    /** @returns the body's angular velocity (zero when the body is not awake). */
    getAngularVelocity(): Vec3 {
        const state = getBodyState(this.world, this.record());
        return state === null ? { x: 0, y: 0, z: 0 } : { ...state.angularVelocity };
    }

    /** Set the body's linear velocity, waking it when nonzero. */
    setLinearVelocity(velocity: Vec3): void {
        bodySetLinearVelocity(this.world, this.record(), froundConfig(velocity));
    }

    /** Set the body's angular velocity (locked axes masked out), waking it when nonzero. */
    setAngularVelocity(velocity: Vec3): void {
        bodySetAngularVelocity(this.world, this.record(), froundConfig(velocity));
    }

    /**
     * Drive the body toward a target transform over `timeStep` by setting the velocity that reaches it.
     * For kinematic bodies animated along a path. Pass `wake` to wake a sleeping body.
     */
    setTargetTransform(target: WorldTransform, timeStep: number, wake = false): void {
        bodySetTargetTransform(this.world, this.record(), target, timeStep, wake);
    }

    /**
     * Teleport the body to a new pose, recomputing its center of mass and broadphase proxies. Velocity
     * is unchanged. Prefer `setTargetTransform` to animate a kinematic body along a path.
     * @example body.setTransform({ x: 0, y: 5, z: 0 }, quat.identity());
     */
    setTransform(position: Pos, rotation: Quat): void {
        bodySetTransform(this.world, this.record(), froundConfig(position), froundConfig(rotation));
    }

    /**
     * Change the body type (static / kinematic / dynamic), rebuilding its solver-set membership, island,
     * contacts, joints, and proxies. Not supported for bodies with a compound or height-field shape when
     * the target type is non-static.
     */
    setType(type: BodyType): void {
        bodySetType(this.world, this.record(), type);
    }

    /** Force the body awake, or put its whole island to sleep. */
    setAwake(awake: boolean): void {
        bodySetAwake(this.world, this.record(), awake);
    }

    /**
     * Accumulate a world-space force at a world-space point over the next step; an off-center point also
     * produces a torque. `wake` wakes a sleeping body first. @example body.applyForce(f, hit, true);
     */
    applyForce(force: Vec3, point: Pos, wake = true): void {
        bodyApplyForce(this.world, this.record(), froundConfig(force), froundConfig(point), wake);
    }

    /** Accumulate a world-space force at the center of mass over the next step (no torque). */
    applyForceToCenter(force: Vec3, wake = true): void {
        bodyApplyForceToCenter(this.world, this.record(), froundConfig(force), wake);
    }

    /** Accumulate a torque about the center of mass over the next step. */
    applyTorque(torque: Vec3, wake = true): void {
        bodyApplyTorque(this.world, this.record(), froundConfig(torque), wake);
    }

    /**
     * Apply an instantaneous world-space impulse at a world-space point, changing velocity immediately;
     * an off-center point also changes angular velocity. @example body.applyLinearImpulse(j, hit, true);
     */
    applyLinearImpulse(impulse: Vec3, point: Pos, wake = true): void {
        bodyApplyLinearImpulse(
            this.world,
            this.record(),
            froundConfig(impulse),
            froundConfig(point),
            wake,
        );
    }

    /** Apply an instantaneous impulse at the center of mass, changing linear velocity immediately. */
    applyLinearImpulseToCenter(impulse: Vec3, wake = true): void {
        bodyApplyLinearImpulseToCenter(this.world, this.record(), froundConfig(impulse), wake);
    }

    /** Apply an instantaneous angular impulse, changing angular velocity immediately. */
    applyAngularImpulse(impulse: Vec3, wake = true): void {
        bodyApplyAngularImpulse(this.world, this.record(), froundConfig(impulse), wake);
    }

    /** @returns whether the body is in the awake solver set. */
    isAwake(): boolean {
        return this.record().setIndex === SetType.Awake;
    }

    /** @returns the body mass. */
    getMass(): number {
        return this.record().mass;
    }

    /** @returns the mass, local center of mass, and rotational inertia. */
    getMassData(): MassData {
        return getMassData(this.world, this.record());
    }

    /** Recompute mass properties from the attached shapes. */
    applyMassFromShapes(): void {
        updateBodyMassData(this.world, this.record());
    }

    /** @returns the number of attached shapes. */
    getShapeCount(): number {
        return this.record().shapeCount;
    }

    /** @returns the user data attached to this body. */
    getUserData(): unknown {
        return this.record().userData;
    }

    /** Attach arbitrary user data to this body. */
    setUserData(userData: unknown): void {
        this.record().userData = userData;
    }

    /**
     * Cast a ray at this body's shapes using `bodyTransform` as the pose (not the body's stored
     * transform), returning the closest hit. Re-centered on `origin` for far-from-origin precision.
     * @example const h = body.castRay(eye, dir, body.getTransform()); if (h.hit) ...
     */
    castRay(
        origin: Pos,
        translation: Vec3,
        bodyTransform: Transform,
        filter: QueryFilter = defaultQueryFilter(),
        maxFraction = 1,
    ): BodyCastHit {
        const r = bodyCastRay(
            this.world,
            this.record(),
            origin,
            translation,
            filter,
            maxFraction,
            bodyTransform,
        );
        return {
            shape: r.hit ? new Shape(this.world, r.shapeId) : null,
            point: r.point,
            normal: r.normal,
            fraction: r.fraction,
            triangleIndex: r.triangleIndex,
            userMaterialId: r.userMaterialId,
            hit: r.hit,
        };
    }

    /**
     * Cast a convex `proxy` at this body's shapes using `bodyTransform` as the pose, returning the
     * closest hit. @example body.castShape(origin, proxy, dir, xf)
     */
    castShape(
        origin: Pos,
        proxy: ShapeProxy,
        translation: Vec3,
        bodyTransform: Transform,
        filter: QueryFilter = defaultQueryFilter(),
        maxFraction = 1,
        canEncroach = false,
    ): BodyCastHit {
        const r = bodyCastShape(
            this.world,
            this.record(),
            origin,
            proxy,
            translation,
            filter,
            maxFraction,
            canEncroach,
            bodyTransform,
        );
        return {
            shape: r.hit ? new Shape(this.world, r.shapeId) : null,
            point: r.point,
            normal: r.normal,
            fraction: r.fraction,
            triangleIndex: r.triangleIndex,
            userMaterialId: r.userMaterialId,
            hit: r.hit,
        };
    }

    /** True if `proxy` overlaps this body's shapes at `bodyTransform` (b3Body_OverlapShape). */
    overlapShape(
        origin: Pos,
        proxy: ShapeProxy,
        bodyTransform: Transform,
        filter: QueryFilter = defaultQueryFilter(),
    ): boolean {
        return bodyOverlapShape(this.world, this.record(), origin, proxy, filter, bodyTransform);
    }

    /**
     * Closest point on this body's convex shapes to `target`, in world space, and its distance
     * (b3Body_GetClosestPoint). Uses the body's stored transform.
     */
    getClosestPoint(target: Vec3): { point: Vec3; distance: number } {
        return bodyGetClosestPoint(this.world, this.record(), target);
    }

    /**
     * Collide a capsule `mover` at `origin` against this body's convex shapes (sphere/capsule/hull),
     * using `bodyTransform` as the pose, returning one plane per touched shape up to `capacity`
     * (b3Body_CollideMover). Mesh/height-field/compound shapes are skipped.
     */
    collideMover(
        origin: Pos,
        mover: Capsule,
        bodyTransform: WorldTransform,
        capacity = 4,
        filter: QueryFilter = defaultQueryFilter(),
    ): BodyPlane[] {
        const results = bodyCollideMover(
            this.world,
            this.record(),
            capacity,
            origin,
            mover,
            filter,
            bodyTransform,
        );
        return results.map((r: BodyPlaneResult) => ({
            shape: new Shape(this.world, r.shapeId),
            plane: r.result,
        }));
    }
}

/** A joint handle connecting two bodies. */
export class Joint {
    /** @internal */
    readonly world: WorldState;
    /** @internal */
    readonly id: EntityId;

    /** @internal use World.createRevoluteJoint */
    constructor(world: WorldState, id: EntityId) {
        this.world = world;
        this.id = id;
    }

    /** @internal */
    protected record(): JointRecord {
        return this.world.joints[this.id.index1 - 1];
    }

    /** @internal the live simulation payload (graph color when awake, else the solver set). */
    protected sim(): JointSim {
        return getJointSim(this.world, this.record());
    }

    /** @returns whether this joint has not been destroyed and its world is alive. */
    isValid(): boolean {
        if (this.world.inUse === false) {
            return false;
        }
        const i = this.id.index1 - 1;
        if (i < 0 || i >= this.world.joints.length) {
            return false;
        }
        const joint = this.world.joints[i];
        if (joint.setIndex === NULL_INDEX) {
            return false;
        }
        return joint.generation === this.id.generation;
    }

    /** Destroy this joint. Pass `false` to leave the attached bodies asleep. */
    destroy(wakeBodies = true): void {
        destroyJointInternal(this.world, this.record(), wakeBodies);
    }

    /** @returns the joint kind. */
    getType(): JointType {
        return this.record().type;
    }

    /** @returns the two bodies this joint connects. */
    getBodies(): [Body, Body] {
        const joint = this.record();
        return [
            new Body(this.world, makeBodyId(this.world, joint.edges[0].bodyId)),
            new Body(this.world, makeBodyId(this.world, joint.edges[1].bodyId)),
        ];
    }

    /** @returns the constraint force this joint currently applies (world units). */
    getConstraintForce(): Vec3 {
        const sim = getJointSim(this.world, this.record());
        return getJointConstraintForce(this.world, sim);
    }

    /** @returns the constraint torque this joint currently applies (world units). */
    getConstraintTorque(): Vec3 {
        const sim = getJointSim(this.world, this.record());
        return getJointConstraintTorque(this.world, sim);
    }

    /** @returns the user data attached to this joint. */
    getUserData(): unknown {
        return this.record().userData;
    }

    /** Attach arbitrary user data to this joint. */
    setUserData(userData: unknown): void {
        this.record().userData = userData;
    }

    /** @returns a handle to the world this joint belongs to. */
    getWorld(): World {
        return World._wrap(this.world);
    }

    /** @returns body A's local joint frame. */
    getLocalFrameA(): Transform {
        return cloneTransform(this.sim().localFrameA);
    }

    /** Set body A's local joint frame. */
    setLocalFrameA(frame: Transform): void {
        // froundConfig returns a fresh deep copy, so the caller's object is never aliased.
        this.sim().localFrameA = froundConfig(frame);
    }

    /** @returns body B's local joint frame. */
    getLocalFrameB(): Transform {
        return cloneTransform(this.sim().localFrameB);
    }

    /** Set body B's local joint frame. */
    setLocalFrameB(frame: Transform): void {
        this.sim().localFrameB = froundConfig(frame);
    }

    /** @returns whether the two connected bodies collide. */
    getCollideConnected(): boolean {
        return this.record().collideConnected;
    }

    /** Toggle whether the two connected bodies collide (updates the broad-phase). */
    setCollideConnected(shouldCollide: boolean): void {
        setJointCollideConnected(this.world, this.record(), shouldCollide);
    }

    /** @returns the joint's constraint softness tuning (hertz + damping ratio). */
    getConstraintTuning(): { hertz: number; dampingRatio: number } {
        const sim = this.sim();
        return { hertz: sim.constraintHertz, dampingRatio: sim.constraintDampingRatio };
    }

    /** Set the joint's constraint softness (hertz + damping ratio). */
    setConstraintTuning(hertz: number, dampingRatio: number): void {
        const sim = this.sim();
        sim.constraintHertz = f32(hertz);
        sim.constraintDampingRatio = f32(dampingRatio);
    }

    /** @returns the force at which this joint reports as over-stressed. */
    getForceThreshold(): number {
        return this.sim().forceThreshold;
    }

    /** Set the force at which this joint reports as over-stressed. */
    setForceThreshold(threshold: number): void {
        this.sim().forceThreshold = f32(threshold);
    }

    /** @returns the torque at which this joint reports as over-stressed. */
    getTorqueThreshold(): number {
        return this.sim().torqueThreshold;
    }

    /** Set the torque at which this joint reports as over-stressed. */
    setTorqueThreshold(threshold: number): void {
        this.sim().torqueThreshold = f32(threshold);
    }

    /** Wake both bodies this joint connects. */
    wakeBodies(): void {
        wakeJointBodies(this.world, this.record());
    }

    /** @returns the current linear separation error at the joint anchors. */
    getLinearSeparation(): number {
        return getJointLinearSeparation(this.world, this.record());
    }

    /** @returns the current angular separation error at the joint. */
    getAngularSeparation(): number {
        return getJointAngularSeparation(this.world, this.record());
    }
}

/** A revolute (hinge) joint handle. */
export class RevoluteJoint extends Joint {
    private data(): RevoluteJointData {
        return this.sim().data as RevoluteJointData;
    }

    /** Enable/disable the angular limit. */
    enableLimit(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableLimit) {
            j.lowerImpulse = 0;
            j.upperImpulse = 0;
        }
        j.enableLimit = enable;
    }

    /** @returns whether the angular limit is enabled. */
    isLimitEnabled(): boolean {
        return this.data().enableLimit;
    }

    /** @returns the lower angle limit (radians). */
    getLowerLimit(): number {
        return this.data().lowerAngle;
    }

    /** @returns the upper angle limit (radians). */
    getUpperLimit(): number {
        return this.data().upperAngle;
    }

    /** Set the angle limits (radians), clamped to ±0.99π. */
    setLimits(lower: number, upper: number): void {
        const lo = f32(lower);
        const hi = f32(upper);
        const lowerAngle = lo < hi ? lo : hi;
        const upperAngle = lo > hi ? lo : hi;
        const bound = f32(0.99 * PI);
        const j = this.data();
        j.lowerAngle = clampf(lowerAngle, -bound, bound);
        j.upperAngle = clampf(upperAngle, -bound, bound);
    }

    /** @returns the current hinge angle (radians). */
    getAngle(): number {
        return revoluteJointAngle(this.world, this.sim());
    }

    /** Enable/disable the drive spring. */
    enableSpring(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableSpring) {
            j.springImpulse = 0;
        }
        j.enableSpring = enable;
    }

    /** @returns whether the drive spring is enabled. */
    isSpringEnabled(): boolean {
        return this.data().enableSpring;
    }

    /** Set the spring target angle (radians). */
    setTargetAngle(target: number): void {
        this.data().targetAngle = f32(target);
    }

    /** @returns the spring target angle (radians). */
    getTargetAngle(): number {
        return this.data().targetAngle;
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        this.data().hertz = f32(hertz);
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return this.data().hertz;
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        this.data().dampingRatio = f32(dampingRatio);
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return this.data().dampingRatio;
    }

    /**
     * Enable/disable the motor.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableMotor) {
            j.motorImpulse = 0;
        }
        j.enableMotor = enable;
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return this.data().enableMotor;
    }

    /**
     * Set the motor target speed (radians/second).
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMotorSpeed(speed: number): void {
        this.data().motorSpeed = f32(speed);
    }

    /** @returns the motor target speed (radians/second). */
    getMotorSpeed(): number {
        return this.data().motorSpeed;
    }

    /**
     * Set the maximum motor torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorTorque(torque: number): void {
        this.data().maxMotorTorque = f32(torque);
    }

    /** @returns the maximum motor torque. */
    getMaxMotorTorque(): number {
        return this.data().maxMotorTorque;
    }

    /** @returns the torque the motor applied last step. */
    getMotorTorque(): number {
        return f32(this.world.invH * this.data().motorImpulse);
    }
}

/** A distance joint handle. */
export class DistanceJoint extends Joint {
    private data(): DistanceJointData {
        return this.sim().data as DistanceJointData;
    }

    /** Set the rest length, clamped to [linear slop, huge]; resets accumulated impulses. */
    setLength(length: number): void {
        const j = this.data();
        j.length = clampf(f32(length), LINEAR_SLOP, HUGE);
        j.impulse = 0;
        j.lowerImpulse = 0;
        j.upperImpulse = 0;
    }

    /** @returns the rest length. */
    getLength(): number {
        return this.data().length;
    }

    /** Enable/disable the length limit. */
    enableLimit(enable: boolean): void {
        this.data().enableLimit = enable;
    }

    /** @returns whether the length limit is enabled. */
    isLimitEnabled(): boolean {
        return this.data().enableLimit;
    }

    /** Set the min/max length range, each clamped to [linear slop, huge]; resets impulses. */
    setLengthRange(minLength: number, maxLength: number): void {
        const lo = clampf(f32(minLength), LINEAR_SLOP, HUGE);
        const hi = clampf(f32(maxLength), LINEAR_SLOP, HUGE);
        const j = this.data();
        j.minLength = lo < hi ? lo : hi;
        j.maxLength = lo > hi ? lo : hi;
        j.impulse = 0;
        j.lowerImpulse = 0;
        j.upperImpulse = 0;
    }

    /** @returns the minimum length. */
    getMinLength(): number {
        return this.data().minLength;
    }

    /** @returns the maximum length. */
    getMaxLength(): number {
        return this.data().maxLength;
    }

    /** @returns the current distance between the anchor points. */
    getCurrentLength(): number {
        return distanceJointCurrentLength(this.world, this.sim());
    }

    /** Enable/disable the spring. */
    enableSpring(enable: boolean): void {
        this.data().enableSpring = enable;
    }

    /** @returns whether the spring is enabled. */
    isSpringEnabled(): boolean {
        return this.data().enableSpring;
    }

    /** Set the spring reaction-force range. */
    setSpringForceRange(lowerForce: number, upperForce: number): void {
        const j = this.data();
        j.lowerSpringForce = f32(lowerForce);
        j.upperSpringForce = f32(upperForce);
    }

    /** @returns the spring reaction-force range. */
    getSpringForceRange(): { lowerForce: number; upperForce: number } {
        const j = this.data();
        return { lowerForce: j.lowerSpringForce, upperForce: j.upperSpringForce };
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        this.data().hertz = f32(hertz);
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return this.data().hertz;
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        this.data().dampingRatio = f32(dampingRatio);
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return this.data().dampingRatio;
    }

    /**
     * Enable/disable the motor; resets the motor impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableMotor) {
            j.enableMotor = enable;
            j.motorImpulse = 0;
        }
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return this.data().enableMotor;
    }

    /**
     * Set the motor target speed.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMotorSpeed(speed: number): void {
        this.data().motorSpeed = f32(speed);
    }

    /** @returns the motor target speed. */
    getMotorSpeed(): number {
        return this.data().motorSpeed;
    }

    /** @returns the force the motor applied last step. */
    getMotorForce(): number {
        return f32(this.world.invH * this.data().motorImpulse);
    }

    /**
     * Set the maximum motor force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorForce(force: number): void {
        this.data().maxMotorForce = f32(force);
    }

    /** @returns the maximum motor force. */
    getMaxMotorForce(): number {
        return this.data().maxMotorForce;
    }
}

/** A prismatic (slider) joint handle. */
export class PrismaticJoint extends Joint {
    private data(): PrismaticJointData {
        return this.sim().data as PrismaticJointData;
    }

    /** Enable/disable the translation limit; resets limit impulses on change. */
    enableLimit(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableLimit) {
            j.lowerImpulse = 0;
            j.upperImpulse = 0;
        }
        j.enableLimit = enable;
    }

    /** @returns whether the translation limit is enabled. */
    isLimitEnabled(): boolean {
        return this.data().enableLimit;
    }

    /** @returns the lower translation limit. */
    getLowerLimit(): number {
        return this.data().lowerTranslation;
    }

    /** @returns the upper translation limit. */
    getUpperLimit(): number {
        return this.data().upperTranslation;
    }

    /** Set the translation limits (ordered low..high). */
    setLimits(lower: number, upper: number): void {
        const lo = f32(lower);
        const hi = f32(upper);
        const j = this.data();
        j.lowerTranslation = lo < hi ? lo : hi;
        j.upperTranslation = lo > hi ? lo : hi;
    }

    /** @returns the current translation along the joint axis. */
    getTranslation(): number {
        return prismaticJointTranslation(this.world, this.sim());
    }

    /** Enable/disable the spring; resets the spring impulse on change. */
    enableSpring(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableSpring) {
            j.springImpulse = 0;
        }
        j.enableSpring = enable;
    }

    /** @returns whether the spring is enabled. */
    isSpringEnabled(): boolean {
        return this.data().enableSpring;
    }

    /** Set the spring target translation. */
    setTargetTranslation(target: number): void {
        this.data().targetTranslation = f32(target);
    }

    /** @returns the spring target translation. */
    getTargetTranslation(): number {
        return this.data().targetTranslation;
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        this.data().hertz = f32(hertz);
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return this.data().hertz;
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        this.data().dampingRatio = f32(dampingRatio);
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return this.data().dampingRatio;
    }

    /**
     * Enable/disable the motor; resets the motor impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableMotor) {
            j.motorImpulse = 0;
        }
        j.enableMotor = enable;
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return this.data().enableMotor;
    }

    /**
     * Set the motor target speed.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMotorSpeed(speed: number): void {
        this.data().motorSpeed = f32(speed);
    }

    /** @returns the motor target speed. */
    getMotorSpeed(): number {
        return this.data().motorSpeed;
    }

    /**
     * Set the maximum motor force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorForce(force: number): void {
        this.data().maxMotorForce = f32(force);
    }

    /** @returns the maximum motor force. */
    getMaxMotorForce(): number {
        return this.data().maxMotorForce;
    }

    /** @returns the force the motor applied last step. */
    getMotorForce(): number {
        return f32(this.world.invH * this.data().motorImpulse);
    }

    /** @returns the current translation speed along the joint axis. */
    getSpeed(): number {
        return prismaticJointSpeed(this.world, this.sim());
    }
}

/** A spherical (ball-and-socket) joint handle. */
export class SphericalJoint extends Joint {
    private data(): SphericalJointData {
        return this.sim().data as SphericalJointData;
    }

    /** Enable/disable the cone (swing) limit; resets the swing impulse on change. */
    enableConeLimit(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableConeLimit) {
            j.swingImpulse = 0;
        }
        j.enableConeLimit = enable;
    }

    /** @returns whether the cone limit is enabled. */
    isConeLimitEnabled(): boolean {
        return this.data().enableConeLimit;
    }

    /** @returns the cone half-angle limit (radians). */
    getConeLimit(): number {
        return this.data().coneAngle;
    }

    /** Set the cone half-angle limit (radians). */
    setConeLimit(angle: number): void {
        this.data().coneAngle = f32(angle);
    }

    /** @returns the current swing (cone) angle (radians). */
    getConeAngle(): number {
        return sphericalJointConeAngle(this.world, this.sim());
    }

    /** Enable/disable the twist limit; resets twist impulses on change. */
    enableTwistLimit(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableTwistLimit) {
            j.lowerTwistImpulse = 0;
            j.upperTwistImpulse = 0;
        }
        j.enableTwistLimit = enable;
    }

    /** @returns whether the twist limit is enabled. */
    isTwistLimitEnabled(): boolean {
        return this.data().enableTwistLimit;
    }

    /** @returns the lower twist limit (radians). */
    getLowerTwistLimit(): number {
        return this.data().lowerTwistAngle;
    }

    /** @returns the upper twist limit (radians). */
    getUpperTwistLimit(): number {
        return this.data().upperTwistAngle;
    }

    /** Set the twist limits (radians), clamped to ±0.99π. */
    setTwistLimits(lower: number, upper: number): void {
        const lo = f32(lower);
        const hi = f32(upper);
        const lowerAngle = lo < hi ? lo : hi;
        const upperAngle = lo > hi ? lo : hi;
        const bound = f32(0.99 * PI);
        const j = this.data();
        j.lowerTwistAngle = clampf(lowerAngle, -bound, bound);
        j.upperTwistAngle = clampf(upperAngle, -bound, bound);
    }

    /** @returns the current twist angle (radians). */
    getTwistAngle(): number {
        return sphericalJointTwistAngle(this.world, this.sim());
    }

    /** Enable/disable the orientation spring; resets the spring impulse on change. */
    enableSpring(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableSpring) {
            j.springImpulse = { x: 0, y: 0, z: 0 };
        }
        j.enableSpring = enable;
    }

    /** @returns whether the orientation spring is enabled. */
    isSpringEnabled(): boolean {
        return this.data().enableSpring;
    }

    /** Set the spring target relative rotation. */
    setTargetRotation(target: Quat): void {
        this.data().targetRotation = froundConfig(target);
    }

    /** @returns the spring target relative rotation. */
    getTargetRotation(): Quat {
        return cloneQuat(this.data().targetRotation);
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        this.data().hertz = f32(hertz);
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return this.data().hertz;
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        this.data().dampingRatio = f32(dampingRatio);
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return this.data().dampingRatio;
    }

    /**
     * Enable/disable the motor; resets the motor impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableMotor(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableMotor) {
            j.motorImpulse = { x: 0, y: 0, z: 0 };
        }
        j.enableMotor = enable;
    }

    /** @returns whether the motor is enabled. */
    isMotorEnabled(): boolean {
        return this.data().enableMotor;
    }

    /**
     * Set the motor target angular velocity.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMotorVelocity(velocity: Vec3): void {
        this.data().motorVelocity = froundConfig(velocity);
    }

    /** @returns the motor target angular velocity. */
    getMotorVelocity(): Vec3 {
        return { ...this.data().motorVelocity };
    }

    /**
     * Set the maximum motor torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxMotorTorque(torque: number): void {
        this.data().maxMotorTorque = f32(torque);
    }

    /** @returns the maximum motor torque. */
    getMaxMotorTorque(): number {
        return this.data().maxMotorTorque;
    }

    /** @returns the torque the motor applied last step. */
    getMotorTorque(): Vec3 {
        return vec3.scale(this.world.invH, this.data().motorImpulse);
    }
}

/** A weld joint handle. */
export class WeldJoint extends Joint {
    private data(): WeldJointData {
        return this.sim().data as WeldJointData;
    }

    /** Set the linear spring frequency (Hz). */
    setLinearHertz(hertz: number): void {
        this.data().linearHertz = f32(hertz);
    }

    /** @returns the linear spring frequency (Hz). */
    getLinearHertz(): number {
        return this.data().linearHertz;
    }

    /** Set the linear spring damping ratio. */
    setLinearDampingRatio(dampingRatio: number): void {
        this.data().linearDampingRatio = f32(dampingRatio);
    }

    /** @returns the linear spring damping ratio. */
    getLinearDampingRatio(): number {
        return this.data().linearDampingRatio;
    }

    /** Set the angular spring frequency (Hz). */
    setAngularHertz(hertz: number): void {
        this.data().angularHertz = f32(hertz);
    }

    /** @returns the angular spring frequency (Hz). */
    getAngularHertz(): number {
        return this.data().angularHertz;
    }

    /** Set the angular spring damping ratio. */
    setAngularDampingRatio(dampingRatio: number): void {
        this.data().angularDampingRatio = f32(dampingRatio);
    }

    /** @returns the angular spring damping ratio. */
    getAngularDampingRatio(): number {
        return this.data().angularDampingRatio;
    }
}

/** A motor joint handle. */
export class MotorJoint extends Joint {
    private data(): MotorJointData {
        return this.sim().data as MotorJointData;
    }

    /**
     * Set the target relative linear velocity.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setLinearVelocity(velocity: Vec3): void {
        this.data().linearVelocity = froundConfig(velocity);
    }

    /** @returns the target relative linear velocity. */
    getLinearVelocity(): Vec3 {
        return { ...this.data().linearVelocity };
    }

    /**
     * Set the target relative angular velocity.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setAngularVelocity(velocity: Vec3): void {
        this.data().angularVelocity = froundConfig(velocity);
    }

    /** @returns the target relative angular velocity. */
    getAngularVelocity(): Vec3 {
        return { ...this.data().angularVelocity };
    }

    /**
     * Set the maximum velocity-drive torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxVelocityTorque(maxTorque: number): void {
        this.data().maxVelocityTorque = f32(maxTorque);
    }

    /** @returns the maximum velocity-drive torque. */
    getMaxVelocityTorque(): number {
        return this.data().maxVelocityTorque;
    }

    /**
     * Set the maximum velocity-drive force.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxVelocityForce(maxForce: number): void {
        this.data().maxVelocityForce = f32(maxForce);
    }

    /** @returns the maximum velocity-drive force. */
    getMaxVelocityForce(): number {
        return this.data().maxVelocityForce;
    }

    /** Set the linear spring frequency (Hz). */
    setLinearHertz(hertz: number): void {
        this.data().linearHertz = f32(hertz);
    }

    /** @returns the linear spring frequency (Hz). */
    getLinearHertz(): number {
        return this.data().linearHertz;
    }

    /** Set the linear spring damping ratio. */
    setLinearDampingRatio(dampingRatio: number): void {
        this.data().linearDampingRatio = f32(dampingRatio);
    }

    /** @returns the linear spring damping ratio. */
    getLinearDampingRatio(): number {
        return this.data().linearDampingRatio;
    }

    /** Set the angular spring frequency (Hz). */
    setAngularHertz(hertz: number): void {
        this.data().angularHertz = f32(hertz);
    }

    /** @returns the angular spring frequency (Hz). */
    getAngularHertz(): number {
        return this.data().angularHertz;
    }

    /** Set the angular spring damping ratio. */
    setAngularDampingRatio(dampingRatio: number): void {
        this.data().angularDampingRatio = f32(dampingRatio);
    }

    /** @returns the angular spring damping ratio. */
    getAngularDampingRatio(): number {
        return this.data().angularDampingRatio;
    }

    /** Set the maximum spring force (clamped ≥ 0). */
    setMaxSpringForce(maxForce: number): void {
        const v = f32(maxForce);
        this.data().maxSpringForce = 0 > v ? 0 : v;
    }

    /** @returns the maximum spring force. */
    getMaxSpringForce(): number {
        return this.data().maxSpringForce;
    }

    /** Set the maximum spring torque (clamped ≥ 0). */
    setMaxSpringTorque(maxTorque: number): void {
        const v = f32(maxTorque);
        this.data().maxSpringTorque = 0 > v ? 0 : v;
    }

    /** @returns the maximum spring torque. */
    getMaxSpringTorque(): number {
        return this.data().maxSpringTorque;
    }
}

/** A parallel joint handle. */
export class ParallelJoint extends Joint {
    private data(): ParallelJointData {
        return this.sim().data as ParallelJointData;
    }

    /** Set the spring frequency (Hz). */
    setSpringHertz(hertz: number): void {
        this.data().hertz = f32(hertz);
    }

    /** @returns the spring frequency (Hz). */
    getSpringHertz(): number {
        return this.data().hertz;
    }

    /** Set the spring damping ratio. */
    setSpringDampingRatio(dampingRatio: number): void {
        this.data().dampingRatio = f32(dampingRatio);
    }

    /** @returns the spring damping ratio. */
    getSpringDampingRatio(): number {
        return this.data().dampingRatio;
    }

    /** Set the maximum corrective torque. */
    setMaxTorque(maxTorque: number): void {
        this.data().maxTorque = f32(maxTorque);
    }

    /** @returns the maximum corrective torque. */
    getMaxTorque(): number {
        return this.data().maxTorque;
    }
}

/** A wheel joint handle. */
export class WheelJoint extends Joint {
    private data(): WheelJointData {
        return this.sim().data as WheelJointData;
    }

    /** Enable/disable the suspension spring; resets the suspension impulse on change. */
    enableSuspension(enable: boolean): void {
        const j = this.data();
        if (enable !== j.enableSuspensionSpring) {
            j.enableSuspensionSpring = enable;
            j.suspensionSpringImpulse = 0;
        }
    }

    /** @returns whether the suspension spring is enabled. */
    isSuspensionEnabled(): boolean {
        return this.data().enableSuspensionSpring;
    }

    /** Set the suspension spring frequency (Hz). */
    setSuspensionHertz(hertz: number): void {
        this.data().suspensionHertz = f32(hertz);
    }

    /** @returns the suspension spring frequency (Hz). */
    getSuspensionHertz(): number {
        return this.data().suspensionHertz;
    }

    /** Set the suspension spring damping ratio. */
    setSuspensionDampingRatio(dampingRatio: number): void {
        this.data().suspensionDampingRatio = f32(dampingRatio);
    }

    /** @returns the suspension spring damping ratio. */
    getSuspensionDampingRatio(): number {
        return this.data().suspensionDampingRatio;
    }

    /** Enable/disable the suspension limit; resets limit impulses on change. */
    enableSuspensionLimit(enable: boolean): void {
        const j = this.data();
        if (j.enableSuspensionLimit !== enable) {
            j.lowerSuspensionImpulse = 0;
            j.upperSuspensionImpulse = 0;
            j.enableSuspensionLimit = enable;
        }
    }

    /** @returns whether the suspension limit is enabled. */
    isSuspensionLimitEnabled(): boolean {
        return this.data().enableSuspensionLimit;
    }

    /** @returns the lower suspension limit. */
    getLowerSuspensionLimit(): number {
        return this.data().lowerSuspensionLimit;
    }

    /** @returns the upper suspension limit. */
    getUpperSuspensionLimit(): number {
        return this.data().upperSuspensionLimit;
    }

    /** Set the suspension limits; resets limit impulses when changed. */
    setSuspensionLimits(lower: number, upper: number): void {
        const lo = f32(lower);
        const hi = f32(upper);
        const j = this.data();
        if (lo !== j.lowerSuspensionLimit || hi !== j.upperSuspensionLimit) {
            j.lowerSuspensionLimit = lo;
            j.upperSuspensionLimit = hi;
            j.lowerSuspensionImpulse = 0;
            j.upperSuspensionImpulse = 0;
        }
    }

    /**
     * Enable/disable the spin motor; resets the spin impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableSpinMotor(enable: boolean): void {
        const j = this.data();
        if (j.enableSpinMotor !== enable) {
            j.spinImpulse = 0;
            j.enableSpinMotor = enable;
        }
    }

    /** @returns whether the spin motor is enabled. */
    isSpinMotorEnabled(): boolean {
        return this.data().enableSpinMotor;
    }

    /**
     * Set the spin motor target speed.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setSpinMotorSpeed(speed: number): void {
        this.data().spinSpeed = f32(speed);
    }

    /** @returns the spin motor target speed. */
    getSpinMotorSpeed(): number {
        return this.data().spinSpeed;
    }

    /**
     * Set the maximum spin torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxSpinTorque(torque: number): void {
        this.data().maxSpinTorque = f32(torque);
    }

    /** @returns the maximum spin torque. */
    getMaxSpinTorque(): number {
        return this.data().maxSpinTorque;
    }

    /** @returns the current spin speed about the spin axis. */
    getSpinSpeed(): number {
        return wheelJointSpinSpeed(this.world, this.sim());
    }

    /** @returns the spin torque applied last step. */
    getSpinTorque(): number {
        return f32(this.world.invH * this.data().spinImpulse);
    }

    /**
     * Enable/disable steering; resets the steering angular impulse on change.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    enableSteering(enable: boolean): void {
        const j = this.data();
        if (j.enableSteering !== enable) {
            j.angularImpulse = { x: 0, y: 0 };
            j.enableSteering = enable;
        }
    }

    /** @returns whether steering is enabled. */
    isSteeringEnabled(): boolean {
        return this.data().enableSteering;
    }

    /** Set the steering spring frequency (Hz). */
    setSteeringHertz(hertz: number): void {
        this.data().steeringHertz = f32(hertz);
    }

    /** @returns the steering spring frequency (Hz). */
    getSteeringHertz(): number {
        return this.data().steeringHertz;
    }

    /** Set the steering spring damping ratio. */
    setSteeringDampingRatio(dampingRatio: number): void {
        this.data().steeringDampingRatio = f32(dampingRatio);
    }

    /** @returns the steering spring damping ratio. */
    getSteeringDampingRatio(): number {
        return this.data().steeringDampingRatio;
    }

    /**
     * Set the maximum steering torque.
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setMaxSteeringTorque(maxTorque: number): void {
        this.data().maxSteeringTorque = f32(maxTorque);
    }

    /** @returns the maximum steering torque. */
    getMaxSteeringTorque(): number {
        return this.data().maxSteeringTorque;
    }

    /** Enable/disable the steering limit; resets limit impulses on change. */
    enableSteeringLimit(enable: boolean): void {
        const j = this.data();
        if (j.enableSteeringLimit !== enable) {
            j.lowerSteeringImpulse = 0;
            j.upperSteeringImpulse = 0;
            j.enableSteeringLimit = enable;
        }
    }

    /** @returns whether the steering limit is enabled. */
    isSteeringLimitEnabled(): boolean {
        return this.data().enableSteeringLimit;
    }

    /** @returns the lower steering limit (radians). */
    getLowerSteeringLimit(): number {
        return this.data().lowerSteeringLimit;
    }

    /** @returns the upper steering limit (radians). */
    getUpperSteeringLimit(): number {
        return this.data().upperSteeringLimit;
    }

    /** Set the steering limits (radians). */
    setSteeringLimits(lower: number, upper: number): void {
        const j = this.data();
        j.lowerSteeringLimit = f32(lower);
        j.upperSteeringLimit = f32(upper);
    }

    /**
     * Set the steering spring target angle (radians).
     * A sleeping body ignores this until `setAwake(true)`: the setter is a pure data write and does not wake the body.
     */
    setTargetSteeringAngle(radians: number): void {
        this.data().targetSteeringAngle = f32(radians);
    }

    /** @returns the steering spring target angle (radians). */
    getTargetSteeringAngle(): number {
        return this.data().targetSteeringAngle;
    }

    /** @returns the current steering angle (radians). */
    getSteeringAngle(): number {
        return wheelJointSteeringAngle(this.world, this.sim());
    }

    /** @returns the steering torque applied last step. */
    getSteeringTorque(): number {
        return f32(this.world.invH * this.data().steeringSpringImpulse);
    }
}

/** A shape handle. */
export class Shape {
    /** @internal */
    readonly world: WorldState;
    /** @internal */
    readonly id: EntityId;

    /** @internal use Body.createSphere/createCapsule/createHull */
    constructor(world: WorldState, id: EntityId) {
        this.world = world;
        this.id = id;
    }

    private record(): ShapeRecord {
        return this.world.shapes[this.id.index1 - 1];
    }

    /** @returns whether this shape has not been destroyed and its world is alive. */
    isValid(): boolean {
        if (this.world.inUse === false) {
            return false;
        }
        const i = this.id.index1 - 1;
        if (i < 0 || i >= this.world.shapes.length) {
            return false;
        }
        const shape = this.world.shapes[i];
        if (shape.id === NULL_INDEX) {
            return false;
        }
        return shape.generation === this.id.generation;
    }

    /** Destroy this shape. Pass `false` to skip recomputing the body mass. */
    destroy(updateBodyMass = true): void {
        destroyShape(this.world, this.record(), updateBodyMass);
    }

    /** @returns the shape type. */
    getType(): ShapeType {
        return this.record().type;
    }

    /** @returns the body this shape is attached to. */
    getBody(): Body {
        const bodyId = this.record().bodyId;
        return new Body(this.world, makeBodyId(this.world, bodyId));
    }

    /** @returns the mass, center, and inertia this shape contributes at its density. */
    computeMassData(): MassData {
        return computeShapeMass(this.record());
    }

    /** @returns the shape's world AABB (as of the last proxy update). */
    getAABB(): AABB {
        const a = this.record().aabb;
        return { lowerBound: { ...a.lowerBound }, upperBound: { ...a.upperBound } };
    }

    /** @returns the shape density. */
    getDensity(): number {
        return this.record().density;
    }

    /** @returns the user data attached to this shape. */
    getUserData(): unknown {
        return this.record().userData;
    }

    /** Attach arbitrary user data to this shape. */
    setUserData(userData: unknown): void {
        this.record().userData = userData;
    }

    /** @returns whether this shape is a sensor (b3Shape_IsSensor). */
    isSensor(): boolean {
        return isSensorShape(this.record());
    }

    /**
     * The shapes currently overlapping this sensor as of the last {@link World.step}
     * (b3Shape_GetSensorData). Empty if this shape is not a sensor.
     */
    getSensorOverlaps(): Shape[] {
        const state = this.world;
        return getSensorData(state, this.record()).map(
            (r) =>
                new Shape(state, {
                    index1: r.shapeId + 1,
                    world0: state.worldId,
                    generation: r.generation,
                }),
        );
    }

    /**
     * Enable or disable sensor overlap events for this shape (b3Shape_EnableSensorEvents). On a sensor
     * this gates its own detection; on any shape it gates whether sensors detect it. Takes effect next step.
     */
    enableSensorEvents(flag: boolean): void {
        this.record().enableSensorEvents = flag;
    }

    /** @returns whether sensor events are enabled for this shape (b3Shape_AreSensorEventsEnabled). */
    areSensorEventsEnabled(): boolean {
        return this.record().enableSensorEvents;
    }

    /**
     * Enable or disable contact begin/end touch events for this shape (b3Shape_EnableContactEvents).
     * Either shape in a pair enabling this reports the pair. Takes effect on the next contact update.
     */
    enableContactEvents(flag: boolean): void {
        this.record().enableContactEvents = flag;
    }

    /** @returns whether contact events are enabled for this shape (b3Shape_AreContactEventsEnabled). */
    areContactEventsEnabled(): boolean {
        return this.record().enableContactEvents;
    }

    /**
     * Enable or disable hit events for this shape (b3Shape_EnableHitEvents). A hit event fires when a
     * contact this shape is part of collides faster than {@link World.setHitEventThreshold}.
     */
    enableHitEvents(flag: boolean): void {
        this.record().enableHitEvents = flag;
    }

    /** @returns whether hit events are enabled for this shape (b3Shape_AreHitEventsEnabled). */
    areHitEventsEnabled(): boolean {
        return this.record().enableHitEvents;
    }
}
