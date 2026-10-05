import type { ShapeProxy } from "../collision/distance";
import { SetType } from "../common/constants";
import type { EntityId } from "../common/ids";
import {
    f32,
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
import { readSimTransform, readStateLinearVelocity } from "../kernel/bodycolumns";
import { bodyType } from "../kernel/filtercolumns";
import { kernel, setQueryCallback } from "../kernel/kernel";
import { type QueryColumns, queryColumns } from "../kernel/querycolumns";
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
    getShapeMaterials,
    type Shape as ShapeRecord,
} from "../shapes/shape";
import {
    type Body as BodyRecord,
    bodyApplyAngularImpulse,
    bodyApplyForce,
    bodyApplyForceToCenter,
    bodyApplyLinearImpulse,
    bodyApplyLinearImpulseToCenter,
    bodyApplyTorque,
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

// Registers the pose reads stage through and the rounded writes hand the solver, which copies out of
// them; never live across calls.
const poseRead: WorldTransform = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
const poseWrite: WorldTransform = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
const velocityWrite: Vec3 = { x: 0, y: 0, z: 0 };

function bodyHit(world: WorldState, q: QueryColumns, origin: Pos): BodyCastHit {
    const id = q.resultU[0];
    if (id === 0xffffffff)
        return {
            shape: null,
            point: { x: 0, y: 0, z: 0 },
            normal: { x: 0, y: 0, z: 0 },
            fraction: 0,
            triangleIndex: 0,
            userMaterialId: 0n,
            hit: false,
        };
    const record = world.shapes[id];
    const f = q.resultF;
    const materials = getShapeMaterials(world.ecsState, record);
    return {
        shape: new Shape(world, {
            index1: id + 1,
            world0: world.worldId,
            generation: record.generation,
        }),
        point: { x: origin.x + f[6], y: origin.y + f[7], z: origin.z + f[8] },
        normal: { x: f[9], y: f[10], z: f[11] },
        fraction: f[5],
        triangleIndex: f[13],
        userMaterialId:
            materials[Math.max(0, Math.min(materials.length - 1, f[15]))].userMaterialId,
        hit: true,
    };
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
        if (kernel(this.world.ecsState).bodyAlive(this.world.worldId, i) === 0) {
            return false;
        }
        return (
            kernel(this.world.ecsState).bodyGeneration(this.world.worldId, i) === this.id.generation
        );
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
        return bodyType(this.world, this.record().id);
    }

    /**
     * @returns the body origin position in world space. Pass `out` to fill it instead of allocating
     * (the three.js `getWorldPosition(target)` idiom) for zero-allocation reads in a hot loop.
     */
    getPosition(out?: Pos): Pos {
        const p = readSimTransform(getBodySim(this.world, this.record()), poseRead).p;
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
        const q = readSimTransform(getBodySim(this.world, this.record()), poseRead).q;
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
        const t = readSimTransform(getBodySim(this.world, this.record()), poseRead);
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

    /** @returns the body's linear velocity (zero when the body is not awake). Pass `out` to fill it instead of allocating. */
    getLinearVelocity(out?: Vec3): Vec3 {
        const state = getBodyState(this.world, this.record());
        if (out === undefined) {
            return state === null ? { x: 0, y: 0, z: 0 } : { ...state.linearVelocity };
        }
        if (state !== null) return readStateLinearVelocity(state, out);
        out.x = 0;
        out.y = 0;
        out.z = 0;
        return out;
    }

    /** @returns the body's angular velocity (zero when the body is not awake). */
    getAngularVelocity(): Vec3 {
        const state = getBodyState(this.world, this.record());
        return state === null ? { x: 0, y: 0, z: 0 } : { ...state.angularVelocity };
    }

    /** Set the body's linear velocity, waking it when nonzero. */
    setLinearVelocity(velocity: Vec3): void {
        const v = velocityWrite;
        v.x = f32(velocity.x);
        v.y = f32(velocity.y);
        v.z = f32(velocity.z);
        bodySetLinearVelocity(this.world, this.record(), v);
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
        const p = poseWrite.p;
        const q = poseWrite.q;
        p.x = f32(position.x);
        p.y = f32(position.y);
        p.z = f32(position.z);
        q.v.x = f32(rotation.v.x);
        q.v.y = f32(rotation.v.y);
        q.v.z = f32(rotation.v.z);
        q.s = f32(rotation.s);
        bodySetTransform(this.world, this.record(), p, q);
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
        const q = queryColumns(this.world);
        const k = q.prepare(origin, filter);
        q.placement(bodyTransform, origin);
        q.translation(translation);
        q.input[12] = maxFraction;
        k.bodyQuery(this.world.worldId, 0, this.record().headShapeId, 0);
        return bodyHit(this.world, q, origin);
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
        const q = queryColumns(this.world);
        const k = q.prepare(origin, filter);
        q.placement(bodyTransform, origin);
        q.proxy(proxy);
        q.translation(translation);
        q.input[12] = maxFraction;
        q.input[13] = Number(canEncroach);
        k.bodyQuery(this.world.worldId, 1, this.record().headShapeId, 0);
        return bodyHit(this.world, q, origin);
    }

    /** True if `proxy` overlaps this body's shapes at `bodyTransform` (b3Body_OverlapShape). */
    overlapShape(
        origin: Pos,
        proxy: ShapeProxy,
        bodyTransform: Transform,
        filter: QueryFilter = defaultQueryFilter(),
    ): boolean {
        const q = queryColumns(this.world);
        const k = q.prepare(origin, filter);
        q.placement(bodyTransform, origin);
        q.proxy(proxy);
        k.bodyQuery(this.world.worldId, 2, this.record().headShapeId, 0);
        return q.resultU[0] !== 0xffffffff;
    }

    /**
     * Closest point on this body's convex shapes to `target`, in world space, and its distance
     * (b3Body_GetClosestPoint). Uses the body's stored transform.
     */
    getClosestPoint(target: Vec3): { point: Vec3; distance: number } {
        const origin = { x: 0, y: 0, z: 0 };
        const q = queryColumns(this.world);
        const k = q.prepare(origin);
        q.placement(getBodyTransformQuick(this.world, this.record()), origin);
        q.proxy({ points: [target], count: 1, radius: 0 });
        k.bodyQuery(this.world.worldId, 3, this.record().headShapeId, 0);
        return {
            point: { x: q.resultF[6], y: q.resultF[7], z: q.resultF[8] },
            distance: q.resultF[3],
        };
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
        const q = queryColumns(this.world);
        const k = q.prepare(origin, filter);
        q.placement(bodyTransform, origin);
        q.mover(mover.center1, mover.center2, mover.radius);
        const results: BodyPlane[] = [];
        const previous = setQueryCallback(this.world.ecsState, (_kind, id, data) => {
            const f = new Float32Array(k.memory.buffer, data, 10);
            results.push({
                shape: new Shape(this.world, {
                    index1: id + 1,
                    world0: this.world.worldId,
                    generation: this.world.shapes[id].generation,
                }),
                plane: {
                    plane: { normal: { x: f[0], y: f[1], z: f[2] }, offset: f[3] },
                    point: { x: f[4], y: f[5], z: f[6] },
                },
            });
            return 1;
        });
        try {
            k.bodyQuery(this.world.worldId, 4, this.record().headShapeId, Math.max(0, capacity));
        } finally {
            setQueryCallback(this.world.ecsState, previous);
        }
        return results;
    }
}
