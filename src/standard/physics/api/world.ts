import { ShapeField, shapeField } from "../kernel/shaperecords";
import type { CustomFilterCallback, PreSolveCallback } from "./config";
// The public surface: thin handle classes over the internal id/record model. A World/Body/Shape
// instance holds only an id and delegates to the internal free functions; all state lives in the
// solver-set columns. Stale handles self-invalidate through the generation stored in the id (the
// planck/rapier idiom). Definitions are plain partial data merged over the ported defaults.
//
// This is authoring ergonomics only — the internals stay op-for-op faithful to Box3D regardless.
// The step and the reads that depend on it (velocities, awake state) arrive with the solver stage.

import type { ShapeProxy } from "../collision/distance";
import type { PlaneResult } from "../collision/mover";
import { DEFAULT_MASK_BITS, SetType } from "../common/constants";
import { type AABB, f32, froundConfig, maxf, type Pos, type Vec3 } from "../common/math";
import {
    type BodyDef,
    BodyType,
    defaultBodyDef,
    defaultQueryFilter,
    defaultWorldDef,
    type MixCallback,
    type QueryFilter,
    type WorldDef,
} from "../common/types";
import { EventKind, eventCount, eventId } from "../kernel/eventbuffers";
import { readJointEventUserData } from "../kernel/jointcolumns";
import { kernel, rethrowQueryError, setQueryCallback } from "../kernel/kernel";
import { queryColumns } from "../kernel/querycolumns";
import {
    setBodyCount,
    solverSetCount,
    solverSetIndex,
    wakeSolverSet,
} from "../kernel/solversetcolumns";
import type { TreeStats } from "../kernel/treecolumns";
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
import type { StepProfile } from "../world/profile";
import {
    type Counters,
    createWorld,
    defaultFrictionCallback,
    defaultRestitutionCallback,
    destroyWorld,
    getWorld,
    type WorldId,
    type WorldState,
    worldCounters,
    worldIsValid,
    worldProfile,
} from "../world/world";
import type {
    WorldCustomFilterCallback,
    WorldMixCallback,
    WorldPreSolveCallback,
} from "../world-definition";
import { Body } from "./body";
import {
    type BaseJointConfig,
    type BodyEvents,
    type BodyMoveEvent,
    baseJointDef,
    type CastCallback,
    type CastHit,
    type ContactEvents,
    type ContactHitEvent,
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
    type SoftJointConfig,
    type SphericalJointConfig,
    type WeldJointConfig,
    type WheelJointConfig,
} from "./config";
import { DistanceJoint, Joint, PrismaticJoint, RevoluteJoint } from "./joint";
import {
    MotorJoint,
    ParallelJoint,
    SoftJoint,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
} from "./joints";
import { Contact, Shape } from "./shape";
import {
    type PhysicsSnapshot,
    restore as restoreWorld,
    snapshot as snapshotWorld,
} from "./snapshot";

function queryShape(world: WorldState, id: number): Shape {
    return new Shape(world, {
        index1: id + 1,
        world0: world.worldId,
        generation: shapeField(world, id, ShapeField.generation),
    });
}

function castHit(world: WorldState, id: number, f: Float32Array, n: number, origin: Pos): CastHit {
    return {
        shape: queryShape(world, id),
        point: { x: origin.x + f[n + 2], y: origin.y + f[n + 3], z: origin.z + f[n + 4] },
        normal: { x: f[n + 5], y: f[n + 6], z: f[n + 7] },
        fraction: f[n + 1],
        userMaterialId: world.shapeStore.materialUserIdAt(id, f[n + 11]),
        triangleIndex: f[n + 9],
        childIndex: f[n + 10],
    };
}

// The event getters refill per-world lists in place; each event and handle is fresh, so only a
// list, never an event, is shared across steps.
function fillSensorTouches(world: WorldState, out: SensorTouchEvent[], kind: number): void {
    out.length = 0;
    for (let i = 0, count = eventCount(world, kind); i < count; ++i) {
        out.push({
            sensor: new Shape(world, eventId(world, kind, i, 0)),
            visitor: new Shape(world, eventId(world, kind, i, 2)),
        });
    }
}

function fillContactTouches(world: WorldState, out: ContactTouchEvent[], kind: number): void {
    out.length = 0;
    for (let i = 0, count = eventCount(world, kind); i < count; ++i) {
        out.push({
            shapeA: new Shape(world, eventId(world, kind, i, 0)),
            shapeB: new Shape(world, eventId(world, kind, i, 2)),
            contact: new Contact(world, eventId(world, kind, i, 4)),
            normalImpulse:
                kind === EventKind.ContactBegin
                    ? kernel(world.ecsState).eventFloat(world.worldId, kind, i, 7)
                    : 0,
        });
    }
}

function fillContactHits(world: WorldState, out: ContactHitEvent[]): void {
    out.length = 0;
    const k = kernel(world.ecsState),
        kind = EventKind.ContactHit,
        id = world.worldId;
    for (let i = 0, count = eventCount(world, kind); i < count; ++i) {
        out.push({
            shapeA: new Shape(world, eventId(world, kind, i, 0)),
            shapeB: new Shape(world, eventId(world, kind, i, 2)),
            contact: new Contact(world, eventId(world, kind, i, 4)),
            point: {
                x: k.eventFloat(id, kind, i, 7),
                y: k.eventFloat(id, kind, i, 8),
                z: k.eventFloat(id, kind, i, 9),
            },
            normal: {
                x: k.eventFloat(id, kind, i, 10),
                y: k.eventFloat(id, kind, i, 11),
                z: k.eventFloat(id, kind, i, 12),
            },
            approachSpeed: k.eventFloat(id, kind, i, 13),
            userMaterialIdA:
                BigInt(k.eventWord(id, kind, i, 14) >>> 0) |
                (BigInt(k.eventWord(id, kind, i, 15) >>> 0) << 32n),
            userMaterialIdB:
                BigInt(k.eventWord(id, kind, i, 16) >>> 0) |
                (BigInt(k.eventWord(id, kind, i, 17) >>> 0) << 32n),
        });
    }
}

function fillJointEvents(world: WorldState, out: JointEvent[]): void {
    const count = eventCount(world, EventKind.Joint);
    out.length = 0;
    if (count === 0) return;
    readJointEventUserData(world);
    for (let i = 0; i < count; ++i) {
        out.push({
            joint: new Joint(world, eventId(world, EventKind.Joint, i, 0)),
            userData: world.jointEventUserData[i],
        });
    }
}

/**
 * Query callbacks may query this world again, but querying another world on the same kernel is
 * refused. Callback exceptions are rethrown after the kernel traversal returns normally.
 */
export class PhysicsWorld {
    /** @internal the underlying world state */
    readonly state: WorldState;
    private readonly _worldId: WorldId;
    private readonly _bodyForEntity?: (eid: number) => Body | null;
    // Only the output array is reused; move events and their body handles are fresh.
    private readonly _moveEventPool: BodyMoveEvent[] = [];
    private readonly _bodyEvents: BodyEvents = { moveEvents: this._moveEventPool, count: 0 };
    private readonly _sensorEvents: SensorEvents = { beginEvents: [], endEvents: [] };
    private readonly _contactEvents: ContactEvents = {
        beginEvents: [],
        endEvents: [],
        hitEvents: [],
    };
    private readonly _jointEvents: JointEvent[] = [];

    constructor(
        def: Partial<WorldDef> = {},
        world?: import("../../../engine").World,
        /** @internal Resolve authored entities from the simulation owner's existing body map. */
        bodyForEntity?: (eid: number) => Body | null,
    ) {
        this._bodyForEntity = bodyForEntity;
        this._worldId = createWorld(world, { ...defaultWorldDef(), ...def });
        // getWorld succeeds immediately after creation.
        this.state = getWorld(this._worldId) as WorldState;
    }

    /** @internal wrap an existing world state as a handle (e.g. Joint.getWorld). */
    static _wrap(state: WorldState): PhysicsWorld {
        const world = Object.create(PhysicsWorld.prototype) as {
            state: WorldState;
            _worldId: WorldId;
            _sensorEvents: SensorEvents;
            _contactEvents: ContactEvents;
            _jointEvents: JointEvent[];
            _moveEventPool: BodyMoveEvent[];
            _bodyEvents: BodyEvents;
        };
        world.state = state;
        world._worldId = { index1: state.worldId + 1, generation: state.generation };
        world._sensorEvents = { beginEvents: [], endEvents: [] };
        world._contactEvents = { beginEvents: [], endEvents: [], hitEvents: [] };
        world._jointEvents = [];
        world._moveEventPool = [];
        world._bodyEvents = { moveEvents: world._moveEventPool, count: 0 };
        return world as unknown as PhysicsWorld;
    }

    /** @returns whether this world has not been destroyed. */
    isValid(): boolean {
        return worldIsValid(this._worldId);
    }

    /** Destroy this world and every body and shape in it. Ignored during a step. */
    destroy(): void {
        if (!this.isValid() || this.state.locked) return;
        destroyWorld(this.state);
    }

    /** Capture the complete wasm-backed state for deterministic resimulation. */
    snapshot(): PhysicsSnapshot {
        return snapshotWorld(this);
    }

    /** Restore into a live compatible World; refuses if another live World shares its kernel. */
    restore(snapshot: PhysicsSnapshot): void {
        restoreWorld(this, snapshot);
    }

    /**
     * Live solver handle for an authored `Body` entity, or null before marshaling,
     * after removal, or for an entity without `Body`. Standalone solver worlds return null.
     * The handle belongs to this world and expires when its body is removed or the world is destroyed.
     */
    getBody(eid: number): Body | null {
        if (!this.isValid()) return null;
        const body = this._bodyForEntity?.(eid);
        return body?.isValid() ? body : null;
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
     * Callbacks see the world as it was when `draw` was called and must not mutate it.
     * Flags and `drawingBounds` are read once at entry.
     */
    draw(draw: DebugDraw, maskBits: bigint = DEFAULT_MASK_BITS): void {
        worldDraw(this.state, draw, maskBits);
    }

    /**
     * Sensor begin/end touch events accumulated during the last {@link step} (b3World_GetSensorEvents).
     * End events read from the previous buffer, so they survive one step. The returned object and its
     * arrays are reused: the next step or the next call overwrites them, so copy an array to keep it.
     * Each event and its handles are fresh. Empty while the world is stepping.
     * @example for (const e of world.getSensorEvents().beginEvents) onEnter(e.sensor, e.visitor)
     */
    getSensorEvents(): SensorEvents {
        const state = this.state;
        const events = this._sensorEvents;
        if (state.locked) {
            events.beginEvents.length = 0;
            events.endEvents.length = 0;
            return events;
        }
        fillSensorTouches(state, events.beginEvents, EventKind.SensorBegin);
        // Careful to read the previous end-event buffer (the swap already happened this step).
        fillSensorTouches(state, events.endEvents, EventKind.SensorEnd);
        return events;
    }

    /**
     * Contact begin/end/hit events from the last {@link step} (b3World_GetContactEvents). Begin/end
     * carry {@link Contact} handles (validate before use); hit events carry the impact point, normal,
     * and approach speed. End events read the previous buffer, so they survive one step. The returned
     * object and its arrays are reused: the next step or the next call overwrites them, so copy an
     * array to keep it. Each event and its handles are fresh. Empty while the world is stepping.
     * @example for (const e of world.getContactEvents().hitEvents) spark(e.point, e.approachSpeed)
     */
    getContactEvents(): ContactEvents {
        const state = this.state;
        const events = this._contactEvents;
        if (state.locked) {
            events.beginEvents.length = 0;
            events.endEvents.length = 0;
            events.hitEvents.length = 0;
            return events;
        }
        fillContactTouches(state, events.beginEvents, EventKind.ContactBegin);
        // Careful to read the previous end-event buffer (the swap already happened this step).
        fillContactTouches(state, events.endEvents, EventKind.ContactEnd);
        fillContactHits(state, events.hitEvents);
        return events;
    }

    /**
     * Body move events from the last {@link step} (b3World_GetBodyEvents), bridged from the kernel's
     * retained finalization records. The returned object and move-event array are reused; each call
     * creates fresh events, transforms and body handles. Only the first `count` array entries are valid.
     * Empty while the world is stepping.
     * @example const ev = world.getBodyEvents(); for (let i = 0; i < ev.count; i++) sync(ev.moveEvents[i].userData, ev.moveEvents[i].transform)
     */
    getBodyEvents(): BodyEvents {
        const state = this.state;
        if (state.locked) {
            this._bodyEvents.count = 0;
            this._moveEventPool.length = 0;
            return this._bodyEvents;
        }
        const count = eventCount(state, EventKind.BodyMove);
        const pool = this._moveEventPool;
        const k = kernel(state.ecsState),
            world = state.worldId,
            kind = EventKind.BodyMove;
        if (pool.length < count) pool.length = count;
        for (let i = 0; i < count; ++i) {
            pool[i] = {
                body: new Body(state, eventId(state, kind, i, 8)),
                transform: {
                    p: {
                        x: k.eventFloat(world, kind, i, 1),
                        y: k.eventFloat(world, kind, i, 2),
                        z: k.eventFloat(world, kind, i, 3),
                    },
                    q: {
                        v: {
                            x: k.eventFloat(world, kind, i, 4),
                            y: k.eventFloat(world, kind, i, 5),
                            z: k.eventFloat(world, kind, i, 6),
                        },
                        s: k.eventFloat(world, kind, i, 7),
                    },
                },
                userData: state.bodyUserData[k.eventWord(world, kind, i, 0)],
                fellAsleep: k.eventWord(world, kind, i, 10) !== 0,
            };
        }
        const events = this._bodyEvents;
        events.count = count;
        return events;
    }

    /**
     * Joint events from the last {@link step} (b3World_GetJointEvents): awake joints whose force or
     * torque exceeded the threshold set via {@link Joint.setForceThreshold}/{@link Joint.setTorqueThreshold}.
     * The returned array is reused: the next step or the next call overwrites it, so copy it to keep
     * it. Each event and its handle are fresh. Empty while the world is stepping.
     */
    getJointEvents(): JointEvent[] {
        const events = this._jointEvents;
        if (this.state.locked) {
            events.length = 0;
            return events;
        }
        fillJointEvents(this.state, events);
        return events;
    }

    /** @returns the collision speed above which a contact reports a hit event (b3World_GetHitEventThreshold). */
    getHitEventThreshold(): number {
        return this.state.hitEventThreshold;
    }

    /** Set the collision speed above which a contact reports a hit event (b3World_SetHitEventThreshold). */
    setHitEventThreshold(value: number): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.hitEventThreshold = maxf(0, f32(value));
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
     * Attach a body to a fixed world point with a soft distance spring.
     * The implementation owns an unshaped static anchor body; callers move the anchor with
     * {@link SoftJoint.setAnchor} and never need to manufacture a second body.
     * @example world.createSoftJoint(body, { x: 0, y: 3, z: 0 }, { hertz: 5, dampingRatio: 0.7 })
     */
    createSoftJoint(body: Body, anchor: Pos, cfg: Partial<SoftJointConfig> = {}): SoftJoint {
        cfg = froundConfig(cfg);
        const anchorBody = this.createBody({
            type: BodyType.Static,
            position: froundConfig(anchor),
        });
        const bodyAnchor = body.getLocalPoint(anchor);
        const bodyPosition = body.getPosition();
        const restLength =
            cfg.length ??
            Math.hypot(
                bodyPosition.x + bodyAnchor.x - anchor.x,
                bodyPosition.y + bodyAnchor.y - anchor.y,
                bodyPosition.z + bodyAnchor.z - anchor.z,
            );
        const d = defaultDistanceJointDef(defaultJointDef());
        const def: DistanceJointDef = {
            base: baseJointDef(anchorBody, body, cfg, d.base),
            length: f32(restLength),
            enableSpring: true,
            hertz: cfg.hertz ?? cfg.stiffness ?? 4,
            dampingRatio: cfg.dampingRatio ?? cfg.damping ?? 1,
            lowerSpringForce: d.lowerSpringForce,
            upperSpringForce: d.upperSpringForce,
            enableLimit: false,
            minLength: d.minLength,
            maxLength: d.maxLength,
            enableMotor: false,
            maxMotorForce: 0,
            motorSpeed: 0,
        };
        const { joint } = createDistanceJoint(this.state, def);
        return new SoftJoint(this.state, makeJointId(this.state, joint), anchorBody);
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
     * @returns a copy of the last step's per-phase timings in milliseconds (b3World_GetProfile).
     * Resolution is the host's `performance.now()`, coarsened in pages without cross-origin isolation.
     */
    getProfile(): StepProfile {
        return worldProfile(this.state);
    }

    /** @returns the gravity vector. Pass `out` to fill it in place instead of allocating. */
    getGravity(out?: Vec3): Vec3 {
        if (out === undefined) return { ...this.state.gravity };
        out.x = this.state.gravity.x;
        out.y = this.state.gravity.y;
        out.z = this.state.gravity.z;
        return out;
    }

    /** Install the collision-pair filter (b3World_SetCustomFilterCallback), or clear it with null.
     * Consulted for new contacts, continuous candidates and sensor overlaps when either shape
     * enables custom filtering; existing discrete contacts are not refiltered. The world retains
     * the function until replaced, cleared or disposed. It runs synchronously in the stepping realm at each
     * phase's serial point. Refuses changes during a step. Errors are rethrown after Rust returns. */
    setCustomFilterCallback(callback: CustomFilterCallback | null): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.customFilterCallback = callback;
        kernel(this.state.ecsState).worldSetCustomFilterCallback(
            this.state.worldId,
            callback !== null || this.state.worldCustomFilterCallback !== null,
        );
    }

    /** Install contact inspection (b3World_SetPreSolveCallback), or clear it with null.
     * Enabled convex contact updates and continuous impacts run on the stepping realm at serial
     * points. Returning false disables the contact for this step or rejects the continuous impact.
     * Sensors do not call it. The world retains the function until replaced, cleared or disposed.
     * Refuses changes during a step. Errors are rethrown after Rust returns. */
    setPreSolveCallback(callback: PreSolveCallback | null): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.preSolveCallback = callback;
        kernel(this.state.ecsState).worldSetPreSolveCallback(
            this.state.worldId,
            callback !== null || this.state.worldPreSolveCallback !== null,
        );
    }

    /** Install or clear the ECS world's custom filter callback. @internal */
    setWorldCustomFilterCallback(callback: WorldCustomFilterCallback | null): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.worldCustomFilterCallback = callback;
        kernel(this.state.ecsState).worldSetCustomFilterCallback(
            this.state.worldId,
            callback !== null || this.state.customFilterCallback !== null,
        );
    }

    /** Install or clear the ECS world's pre-solve callback. @internal */
    setWorldPreSolveCallback(callback: WorldPreSolveCallback | null): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.worldPreSolveCallback = callback;
        kernel(this.state.ecsState).worldSetPreSolveCallback(
            this.state.worldId,
            callback !== null || this.state.preSolveCallback !== null,
        );
    }

    /** Install or clear the ECS world's friction mixing callback. @internal */
    setWorldFrictionCallback(callback: WorldMixCallback | null): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.worldFrictionCallback = callback;
    }

    /** Install or clear the ECS world's restitution mixing callback. @internal */
    setWorldRestitutionCallback(callback: WorldMixCallback | null): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.worldRestitutionCallback = callback;
    }

    /** Set the gravity vector (b3World_SetGravity). */
    setGravity(gravity: Vec3): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.gravity = froundConfig({ x: gravity.x, y: gravity.y, z: gravity.z });
    }

    /** Set restitution and hit-event speed thresholds (b3World_SetRestitutionThreshold). */
    setRestitutionThreshold(value: number): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.restitutionThreshold = maxf(0, f32(value));
    }

    /** Set contact stiffness, damping and overlap-recovery speed (b3World_SetContactTuning). */
    setContactTuning(hertz: number, dampingRatio: number, contactSpeed: number): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.contactHertz = maxf(0, f32(hertz));
        this.state.contactDampingRatio = maxf(0, f32(dampingRatio));
        this.state.contactSpeed = maxf(0, f32(contactSpeed));
    }

    /** Set the maximum distance at which contacts are recycled (b3World_SetContactRecycleDistance). */
    setContactRecycleDistance(distance: number): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.contactRecycleDistance = maxf(0, f32(distance));
    }

    /** Set the speed cap for dynamic bodies (b3World_SetMaximumLinearSpeed). */
    setMaximumLinearSpeed(speed: number): void {
        if (!(speed > 0)) throw new RangeError("physics: maximum linear speed must be positive");
        if (!this.isValid() || this.state.locked) return;
        this.state.maxLinearSpeed = f32(speed);
    }

    /** Enable or disable body sleeping (b3World_EnableSleeping). */
    enableSleeping(enabled: boolean): void {
        if (!this.isValid() || this.state.locked || this.state.enableSleep === enabled) return;
        this.state.enableSleep = enabled;
        if (!enabled) {
            const count = solverSetCount(this.state);
            for (let set = SetType.FirstSleeping; set < count; set++) {
                if (solverSetIndex(this.state, set) >= 0 && setBodyCount(this.state, set) > 0)
                    wakeSolverSet(this.state, set);
            }
        }
    }

    /** Enable or disable continuous collision (b3World_EnableContinuous). */
    enableContinuous(enabled: boolean): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.enableContinuous = enabled;
    }

    /** Enable or disable impulse warm starting (b3World_EnableWarmStarting). */
    enableWarmStarting(enabled: boolean): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.enableWarmStarting = enabled;
    }

    /** Enable or disable world speculative mesh contacts (b3World_EnableSpeculative). */
    enableSpeculative(enabled: boolean): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.enableSpeculative = enabled;
    }

    /** Install or clear Box3D's optional friction mixing callback. */
    setFrictionCallback(callback: MixCallback | null): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.frictionCallback = callback ?? defaultFrictionCallback;
    }

    /** Install or clear Box3D's optional restitution mixing callback. */
    setRestitutionCallback(callback: MixCallback | null): void {
        if (!this.isValid() || this.state.locked) return;
        this.state.restitutionCallback = callback ?? defaultRestitutionCallback;
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
        if (!this.isValid() || this.state.locked)
            return {
                shape: null,
                point: { x: 0, y: 0, z: 0 },
                normal: { x: 0, y: 0, z: 0 },
                fraction: 0,
                userMaterialId: 0n,
                triangleIndex: 0,
                childIndex: 0,
                hit: false,
            };
        const q = queryColumns(this.state);
        const k = q.prepare(origin, filter);
        q.translation(translation);
        k.worldQuery(this.state.worldId, 3, 0);
        if (q.resultU[0] === 0xffffffff)
            return {
                shape: null,
                point: { x: 0, y: 0, z: 0 },
                normal: { x: 0, y: 0, z: 0 },
                fraction: 0,
                userMaterialId: 0n,
                triangleIndex: 0,
                childIndex: 0,
                hit: false,
            };
        return { ...castHit(this.state, q.resultU[0], q.resultF, 4, origin), hit: true };
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
        if (!this.isValid() || this.state.locked) return { nodeVisits: 0, leafVisits: 0 };
        const q = queryColumns(this.state);
        const k = q.prepare(origin, filter);
        q.translation(translation);
        const previous = setQueryCallback(this.state.ecsState, (_kind, id, data) => {
            this.state.shapeStore.refreshViews();
            return fcn(
                castHit(this.state, id, this.state.shapeStore.materialF, data >>> 2, origin),
            );
        });
        try {
            k.worldQuery(this.state.worldId, 2, 1);
            rethrowQueryError(this.state.ecsState);
        } finally {
            setQueryCallback(this.state.ecsState, previous);
        }
        return { nodeVisits: q.resultU[1], leafVisits: q.resultU[2] };
    }

    /** Report shapes whose fat AABB overlaps `box`. Returning false stops the current body-type
     * tree (static, kinematic, dynamic); the next tree is still queried. */
    overlapAABB(
        box: AABB,
        fcn: OverlapCallback,
        filter: QueryFilter = defaultQueryFilter(),
    ): TreeStats {
        if (!this.isValid() || this.state.locked) return { nodeVisits: 0, leafVisits: 0 };
        const q = queryColumns(this.state);
        const k = q.prepare({ x: 0, y: 0, z: 0 }, filter);
        q.bounds(box);
        const previous = setQueryCallback(this.state.ecsState, (_kind, id) =>
            Number(fcn(queryShape(this.state, id))),
        );
        try {
            k.worldQuery(this.state.worldId, 0, 1);
            rethrowQueryError(this.state.ecsState);
        } finally {
            setQueryCallback(this.state.ecsState, previous);
        }
        return { nodeVisits: q.resultU[1], leafVisits: q.resultU[2] };
    }

    /**
     * Report shapes overlapping the convex `proxy` at `origin`. Returning false stops the current
     * body-type tree (static, kinematic, dynamic); the next tree is still queried.
     */
    overlapShape(
        origin: Pos,
        proxy: ShapeProxy,
        fcn: OverlapCallback,
        filter: QueryFilter = defaultQueryFilter(),
    ): TreeStats {
        if (!this.isValid() || this.state.locked) return { nodeVisits: 0, leafVisits: 0 };
        const q = queryColumns(this.state);
        const k = q.prepare(origin, filter);
        q.proxy(proxy);
        const previous = setQueryCallback(this.state.ecsState, (_kind, id) =>
            Number(fcn(queryShape(this.state, id))),
        );
        try {
            k.worldQuery(this.state.worldId, 1, 1);
            rethrowQueryError(this.state.ecsState);
        } finally {
            setQueryCallback(this.state.ecsState, previous);
        }
        return { nodeVisits: q.resultU[1], leafVisits: q.resultU[2] };
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
        if (!this.isValid() || this.state.locked) return { nodeVisits: 0, leafVisits: 0 };
        const q = queryColumns(this.state);
        const k = q.prepare(origin, filter);
        q.proxy(proxy);
        q.translation(translation);
        const previous = setQueryCallback(this.state.ecsState, (_kind, id, data) => {
            this.state.shapeStore.refreshViews();
            return fcn(
                castHit(this.state, id, this.state.shapeStore.materialF, data >>> 2, origin),
            );
        });
        try {
            k.worldQuery(this.state.worldId, 4, 1);
            rethrowQueryError(this.state.ecsState);
        } finally {
            setQueryCallback(this.state.ecsState, previous);
        }
        return { nodeVisits: q.resultU[1], leafVisits: q.resultU[2] };
    }

    /**
     * Collide a capsule `mover` at `origin` against the world, reporting each touched shape's collision
     * planes to `fcn`. Returning false stops the current body-type tree (static, kinematic, dynamic);
     * the next tree is still queried. Feed the planes to {@link solvePlanes} to resolve movement.
     */
    collideMover(
        origin: Pos,
        mover: Capsule,
        fcn: PlaneResultCallback,
        filter: QueryFilter = defaultQueryFilter(),
    ): void {
        if (!this.isValid() || this.state.locked) return;
        const q = queryColumns(this.state);
        const k = q.prepare(origin, filter);
        q.mover(mover.center1, mover.center2, mover.radius);
        const previous = setQueryCallback(this.state.ecsState, (_kind, id, data, count) => {
            this.state.shapeStore.refreshViews();
            const f = this.state.shapeStore.materialF;
            const base = data >>> 2;
            const planes: PlaneResult[] = [];
            for (let i = 0; i < count; ++i) {
                const n = base + i * 10;
                planes.push({
                    plane: { normal: { x: f[n], y: f[n + 1], z: f[n + 2] }, offset: f[n + 3] },
                    point: { x: f[n + 4], y: f[n + 5], z: f[n + 6] },
                    triangleIndex: f[n + 7],
                    childIndex: f[n + 8],
                    materialIndex: f[n + 9],
                });
            }
            return Number(fcn(queryShape(this.state, id), planes));
        });
        try {
            k.worldQuery(this.state.worldId, 5, 1);
            rethrowQueryError(this.state.ecsState);
        } finally {
            setQueryCallback(this.state.ecsState, previous);
        }
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
        if (!this.isValid() || this.state.locked) return 1;
        const q = queryColumns(this.state);
        const k = q.prepare(origin, filter);
        q.mover(mover.center1, mover.center2, mover.radius);
        q.translation(translation);
        const previous = setQueryCallback(
            this.state.ecsState,
            fcn === null ? null : (_kind, id) => Number(fcn(queryShape(this.state, id))),
        );
        try {
            k.worldQuery(this.state.worldId, 6, Number(fcn !== null));
            rethrowQueryError(this.state.ecsState);
        } finally {
            setQueryCallback(this.state.ecsState, previous);
        }
        return q.resultF[3];
    }
}
