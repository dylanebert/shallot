import type { ShapeProxy } from "../collision/distance";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/core";
import type { EntityId } from "../common/ids";
import {
    froundConfig,
    invTransformWorldPoint,
    type Pos,
    type Quat,
    type Transform,
    type Vec3,
    type WorldTransform,
} from "../common/math";
import {
    type BodyType,
    defaultQueryFilter,
    defaultShapeDef,
    type QueryFilter,
    type ShapeDef,
} from "../common/types";
import type { CompoundData } from "../shapes/compound";
import type { Capsule, MassData, Sphere } from "../shapes/geometry";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import type { MeshData } from "../shapes/mesh";
import {
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
    type Shape as ShapeRecord,
} from "../shapes/shape";
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
    destroyBody,
    getBodySim,
    getBodyState,
    getBodyTransformQuick,
    getMassData,
    updateBodyMassData,
} from "../world/body";
import type { WorldState } from "../world/world";
import { type BodyCastHit, type BodyPlane, makeShapeId } from "./config";
import { Shape } from "./shape";

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
