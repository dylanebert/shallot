// The public surface: thin handle classes over the internal id/record model. A World/Body/Shape
// instance holds only an id and delegates to the internal free functions; all state lives in the
// solver-set columns. Stale handles self-invalidate through the generation stored in the id (the
// planck/rapier idiom). Definitions are plain partial data merged over the ported defaults.
//
// This is authoring ergonomics only — the internals stay op-for-op faithful to Box3D regardless.
// The step and the reads that depend on it (velocities, awake state) arrive with the solver stage.

import type { ShapeProxy } from "../collision/distance";
import {
    castMover as castMoverInternal,
    castRayClosest as castRayClosestInternal,
    castRay as castRayInternal,
    castShape as castShapeInternal,
    collideMover as collideMoverInternal,
    overlapAABB as overlapAABBInternal,
    overlapShapeQuery,
} from "../collision/query";
import type { TreeStats } from "../collision/tree";
import { DEFAULT_MASK_BITS } from "../common/constants";
import type { EntityId } from "../common/ids";
import { type AABB, f32, froundConfig, type Pos, type Vec3 } from "../common/math";
import {
    type BodyDef,
    defaultBodyDef,
    defaultQueryFilter,
    defaultWorldDef,
    type QueryFilter,
    type WorldDef,
} from "../common/types";
import type { Capsule } from "../shapes/geometry";
import {
    createDistanceJoint,
    type DistanceJointDef,
    defaultDistanceJointDef,
} from "../solver/distanceJoint";
import { createFilterJoint, defaultJointDef } from "../solver/joint";
import { createMotorJoint, defaultMotorJointDef, type MotorJointDef } from "../solver/motorJoint";
import {
    createParallelJoint,
    defaultParallelJointDef,
    type ParallelJointDef,
} from "../solver/parallelJoint";
import {
    createPrismaticJoint,
    defaultPrismaticJointDef,
    type PrismaticJointDef,
} from "../solver/prismaticJoint";
import {
    createRevoluteJoint,
    defaultRevoluteJointDef,
    type RevoluteJointDef,
} from "../solver/revoluteJoint";
import {
    createSphericalJoint,
    defaultSphericalJointDef,
    type SphericalJointDef,
} from "../solver/sphericalJoint";
import { step as stepWorld } from "../solver/step";
import { createWeldJoint, defaultWeldJointDef, type WeldJointDef } from "../solver/weldJoint";
import { createWheelJoint, defaultWheelJointDef, type WheelJointDef } from "../solver/wheelJoint";
import { createBody, makeBodyId } from "../world/body";
import { type DebugDraw, worldDraw } from "../world/draw";
import type { Profile } from "../world/profile";
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
} from "../world/world";
import { Body } from "./body";
import {
    type BaseJointConfig,
    type BodyEvents,
    type BodyMoveEvent,
    baseJointDef,
    type CastCallback,
    type ContactEvents,
    type ContactTouchEvent,
    type DistanceJointConfig,
    type JointEvent,
    type MotorJointConfig,
    type MoverFilterCallback,
    makeJointId,
    type OverlapCallback,
    type ParallelJointConfig,
    type PlaneResultCallback,
    type PrismaticJointConfig,
    type RayResult,
    type RevoluteJointConfig,
    type SensorEvents,
    type SensorTouchEvent,
    type SphericalJointConfig,
    type WeldJointConfig,
    type WheelJointConfig,
} from "./config";
import { DistanceJoint, Joint, PrismaticJoint, RevoluteJoint } from "./joint";
import { MotorJoint, ParallelJoint, SphericalJoint, WeldJoint, WheelJoint } from "./joints";
import { Contact, Shape } from "./shape";

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
