import type { Manifold, ManifoldPoint } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import type { EntityId } from "../common/ids";
import type { AABB } from "../common/math";
import type { ShapeType } from "../common/types";
import type { MassData } from "../shapes/geometry";
import {
    computeShapeMass,
    destroyShape,
    getSensorData,
    isSensorShape,
    type Shape as ShapeRecord,
} from "../shapes/shape";
import { makeBodyId } from "../world/body";
import type { WorldState } from "../world/world";
import { Body } from "./body";
import { type ContactData, makeShapeId } from "./config";

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
