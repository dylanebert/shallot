// World spatial queries: ray cast, shape cast, and overlap against the broad-phase trees.
// Ported op-for-op from Box3D's physics_world.c query section (Erin Catto, MIT). Recording
// trampolines are omitted (recording is deferred); the arithmetic is otherwise faithful.
//
// Every query is origin-relative: the caller passes a world origin, each candidate shape's body
// transform is re-differenced against it, and the hit point is lifted back by the origin. In the
// single-precision port this collapses to plain float math, but the subtract/add ordering is kept
// so the result is bit-exact with the C build (and a later large-world build has one set of seams).

import { getBodyTransformQuick } from "./body";
import type { BroadPhase } from "./broadphase";
import type { RayCastInput, ShapeCastInput, ShapeProxy } from "./distance";
import type { Capsule } from "./geometry";
import type { EntityId } from "./ids";
import {
    aabb,
    clampInt,
    offsetPos,
    type Pos,
    toRelativeTransform,
    toVec3,
    type Vec3,
} from "./math";
import type { PlaneResult } from "./mover";
import {
    collideMover as collideMoverShape,
    getShapeMaterials,
    overlapShape,
    rayCastShape,
    type Shape,
    shapeCastShape,
} from "./shape";
import type { TreeStats } from "./tree";
import * as tree from "./tree";
import {
    type QueryFilter,
    type QueryFilterBits,
    shouldQueryCollide,
    toQueryFilterBits,
} from "./types";
import type { WorldState } from "./world";

const BODY_TYPE_COUNT = 3;

/** Reported to an overlap query per hit; return false to stop the query (b3OverlapResultFcn). */
export type OverlapResultFcn = (shapeId: EntityId) => boolean;

/**
 * Reported to a ray/shape cast per hit (b3CastResultFcn). Return the new max fraction to clip the
 * query, 0 to stop, or -1 to ignore this shape and keep the current clip.
 */
export type CastResultFcn = (
    shapeId: EntityId,
    point: Pos,
    normal: Vec3,
    fraction: number,
    userMaterialId: bigint,
    triangleIndex: number,
    childIndex: number,
) => number;

/** The single closest hit of a ray cast (b3RayResult). */
export type RayResult = {
    shapeId: EntityId;
    point: Pos;
    normal: Vec3;
    userMaterialId: bigint;
    fraction: number;
    triangleIndex: number;
    childIndex: number;
    nodeVisits: number;
    leafVisits: number;
    hit: boolean;
};

const shapeId = (world: WorldState, shape: Shape): EntityId => ({
    index1: shape.id + 1,
    world0: world.worldId,
    generation: shape.generation,
});

// --- overlap --------------------------------------------------------------------------------

/** Report every shape whose fat AABB overlaps `box` (b3World_OverlapAABB). */
export function overlapAABB(
    world: WorldState,
    box: { lowerBound: Vec3; upperBound: Vec3 },
    filter: QueryFilter,
    fcn: OverlapResultFcn,
): TreeStats {
    const treeStats: TreeStats = { nodeVisits: 0, leafVisits: 0 };
    (world.broadPhase as BroadPhase).store.refreshIfStale();
    const trees = (world.broadPhase as BroadPhase).trees;
    const filterBits = toQueryFilterBits(filter);

    for (let i = 0; i < BODY_TYPE_COUNT; ++i) {
        const treeResult = tree.query(
            trees[i],
            box,
            filterBits.maskHi,
            filterBits.maskLo,
            false,
            (_proxyId, userData) => {
                const shape = world.shapes[userData];
                if (shouldQueryCollide(shape.filter, filterBits) === false) {
                    return true;
                }
                return fcn(shapeId(world, shape));
            },
        );
        treeStats.nodeVisits += treeResult.nodeVisits;
        treeStats.leafVisits += treeResult.leafVisits;
    }

    return treeStats;
}

/** Report every shape whose geometry overlaps the swept proxy at `origin` (b3World_OverlapShape). */
export function overlapShapeQuery(
    world: WorldState,
    origin: Pos,
    proxy: ShapeProxy,
    filter: QueryFilter,
    fcn: OverlapResultFcn,
): TreeStats {
    const treeStats: TreeStats = { nodeVisits: 0, leafVisits: 0 };
    (world.broadPhase as BroadPhase).store.refreshIfStale();
    const trees = (world.broadPhase as BroadPhase).trees;
    const filterBits = toQueryFilterBits(filter);

    // Bound the proxy in origin-relative space, then lift to a conservative world float box.
    const box = aabb.offset(aabb.make(proxy.points, proxy.count, proxy.radius), toVec3(origin));

    for (let i = 0; i < BODY_TYPE_COUNT; ++i) {
        const treeResult = tree.query(
            trees[i],
            box,
            filterBits.maskHi,
            filterBits.maskLo,
            false,
            (_proxyId, userData) => {
                const shape = world.shapes[userData];
                if (shouldQueryCollide(shape.filter, filterBits) === false) {
                    return true;
                }
                // Re-center on the query origin so the overlap test stays in float precision far out.
                const body = world.bodies[shape.bodyId];
                const transform = toRelativeTransform(getBodyTransformQuick(world, body), origin);
                if (overlapShape(shape, transform, proxy) === false) {
                    return true;
                }
                return fcn(shapeId(world, shape));
            },
        );
        treeStats.nodeVisits += treeResult.nodeVisits;
        treeStats.leafVisits += treeResult.leafVisits;
    }

    return treeStats;
}

// --- ray cast -------------------------------------------------------------------------------

type RayCastContext = {
    world: WorldState;
    fcn: CastResultFcn;
    filter: QueryFilterBits;
    fraction: number;
    origin: Pos;
};

function rayCastCallback(ctx: RayCastContext, input: RayCastInput, userData: number): number {
    const world = ctx.world;
    const shape = world.shapes[userData];
    if (shouldQueryCollide(shape.filter, ctx.filter) === false) {
        return input.maxFraction;
    }

    const body = world.bodies[shape.bodyId];
    const transform = toRelativeTransform(getBodyTransformQuick(world, body), ctx.origin);

    const localInput: RayCastInput = {
        origin: { x: 0, y: 0, z: 0 },
        translation: input.translation,
        maxFraction: input.maxFraction,
    };
    const output = rayCastShape(shape, transform, localInput);

    if (output.hit) {
        const point = offsetPos(ctx.origin, output.point);
        const materialIndex = clampInt(output.materialIndex, 0, shape.materialCount - 1);
        const userMaterialId = getShapeMaterials(shape)[materialIndex].userMaterialId;
        const fraction = ctx.fcn(
            shapeId(world, shape),
            point,
            output.normal,
            output.fraction,
            userMaterialId,
            output.triangleIndex,
            output.childIndex,
        );
        // The user may return -1 to skip this shape.
        if (fraction >= 0 && fraction <= 1) {
            ctx.fraction = fraction;
        }
        return fraction;
    }

    return input.maxFraction;
}

/** Cast a ray, reporting each hit to `fcn` (b3World_CastRay). */
export function castRay(
    world: WorldState,
    origin: Pos,
    translation: Vec3,
    filter: QueryFilter,
    fcn: CastResultFcn,
): TreeStats {
    const treeStats: TreeStats = { nodeVisits: 0, leafVisits: 0 };
    (world.broadPhase as BroadPhase).store.refreshIfStale();
    const trees = (world.broadPhase as BroadPhase).trees;
    const filterBits = toQueryFilterBits(filter);

    const input: RayCastInput = { origin: toVec3(origin), translation, maxFraction: 1 };
    const ctx: RayCastContext = { world, fcn, filter: filterBits, fraction: 1, origin };

    for (let i = 0; i < BODY_TYPE_COUNT; ++i) {
        const treeResult = tree.rayCast(
            trees[i],
            input,
            filterBits.maskHi,
            filterBits.maskLo,
            false,
            (subInput, _id, userData) => rayCastCallback(ctx, subInput, userData),
        );
        treeStats.nodeVisits += treeResult.nodeVisits;
        treeStats.leafVisits += treeResult.leafVisits;

        if (ctx.fraction === 0) {
            break;
        }
        input.maxFraction = ctx.fraction;
    }

    return treeStats;
}

/** Cast a ray, returning the single closest hit (b3World_CastRayClosest). */
export function castRayClosest(
    world: WorldState,
    origin: Pos,
    translation: Vec3,
    filter: QueryFilter,
): RayResult {
    const result: RayResult = {
        shapeId: { index1: 0, world0: 0, generation: 0 },
        point: { x: 0, y: 0, z: 0 },
        normal: { x: 0, y: 0, z: 0 },
        userMaterialId: 0n,
        fraction: 0,
        triangleIndex: 0,
        childIndex: 0,
        nodeVisits: 0,
        leafVisits: 0,
        hit: false,
    };
    (world.broadPhase as BroadPhase).store.refreshIfStale();
    const trees = (world.broadPhase as BroadPhase).trees;
    const filterBits = toQueryFilterBits(filter);

    const input: RayCastInput = { origin: toVec3(origin), translation, maxFraction: 1 };
    // The closest-hit callback: ignore initial overlap, record the hit, clip to its fraction.
    const closestFcn: CastResultFcn = (
        id,
        point,
        normal,
        fraction,
        userMaterialId,
        triangleIndex,
        childIndex,
    ) => {
        if (fraction === 0) {
            return -1;
        }
        result.shapeId = id;
        result.point = point;
        result.normal = normal;
        result.fraction = fraction;
        result.userMaterialId = userMaterialId;
        result.triangleIndex = triangleIndex;
        result.childIndex = childIndex;
        result.hit = true;
        return fraction;
    };
    const ctx: RayCastContext = { world, fcn: closestFcn, filter: filterBits, fraction: 1, origin };

    for (let i = 0; i < BODY_TYPE_COUNT; ++i) {
        const treeResult = tree.rayCast(
            trees[i],
            input,
            filterBits.maskHi,
            filterBits.maskLo,
            false,
            (subInput, _id, userData) => rayCastCallback(ctx, subInput, userData),
        );
        result.nodeVisits += treeResult.nodeVisits;
        result.leafVisits += treeResult.leafVisits;

        if (ctx.fraction === 0) {
            break;
        }
        input.maxFraction = ctx.fraction;
    }

    return result;
}

// --- shape cast -----------------------------------------------------------------------------

/** Cast a swept convex proxy, reporting each hit to `fcn` (b3World_CastShape). */
export function castShape(
    world: WorldState,
    origin: Pos,
    proxy: ShapeProxy,
    translation: Vec3,
    filter: QueryFilter,
    fcn: CastResultFcn,
): TreeStats {
    const treeStats: TreeStats = { nodeVisits: 0, leafVisits: 0 };
    (world.broadPhase as BroadPhase).store.refreshIfStale();
    const trees = (world.broadPhase as BroadPhase).trees;
    const filterBits = toQueryFilterBits(filter);

    // The origin-relative shape cast input, rebuilt per candidate from the tree's advancing fraction.
    const castInput: ShapeCastInput = {
        proxy,
        translation,
        maxFraction: 1,
        canEncroach: false,
    };
    const ctx = { fraction: 1 };

    const callback = (boxMaxFraction: number, userData: number): number => {
        const shape = world.shapes[userData];
        if (shouldQueryCollide(shape.filter, filterBits) === false) {
            return boxMaxFraction;
        }

        // Rebuild from the origin-relative input, taking only the advancing fraction from the tree.
        const localInput: ShapeCastInput = {
            proxy: castInput.proxy,
            translation: castInput.translation,
            maxFraction: boxMaxFraction,
            canEncroach: castInput.canEncroach,
        };

        const body = world.bodies[shape.bodyId];
        const transform = toRelativeTransform(getBodyTransformQuick(world, body), origin);

        const output = shapeCastShape(shape, transform, localInput);
        if (output.hit) {
            const materialIndex = clampInt(output.materialIndex, 0, shape.materialCount - 1);
            const userMaterialId = getShapeMaterials(shape)[materialIndex].userMaterialId;
            const fraction = fcn(
                shapeId(world, shape),
                offsetPos(origin, output.point),
                output.normal,
                output.fraction,
                userMaterialId,
                output.triangleIndex,
                output.childIndex,
            );
            // The user may return -1 to skip this shape.
            if (fraction >= 0 && fraction <= 1) {
                ctx.fraction = fraction;
            }
            return fraction;
        }

        return boxMaxFraction;
    };

    // Bound the proxy in origin-relative space then lift to a conservative world float box.
    const localBox = aabb.make(proxy.points, proxy.count, proxy.radius);
    const treeInput = { box: aabb.offset(localBox, toVec3(origin)), translation, maxFraction: 1 };

    for (let i = 0; i < BODY_TYPE_COUNT; ++i) {
        const treeResult = tree.boxCast(
            trees[i],
            treeInput,
            filterBits.maskHi,
            filterBits.maskLo,
            false,
            (boxInput, _id, userData) => callback(boxInput.maxFraction, userData),
        );
        treeStats.nodeVisits += treeResult.nodeVisits;
        treeStats.leafVisits += treeResult.leafVisits;

        if (ctx.fraction === 0) {
            break;
        }
        treeInput.maxFraction = ctx.fraction;
    }

    return treeStats;
}

// --- character mover ------------------------------------------------------------------------

/** Reported to a collide-mover query per touched shape (b3PlaneResultFcn); return false to stop. */
export type PlaneResultFcn = (shapeId: EntityId, planes: PlaneResult[]) => boolean;

/** Per-shape accept filter for a mover cast (b3MoverFilterFcn); return false to skip the shape. */
export type MoverFilterFcn = (shapeId: EntityId) => boolean;

// The per-shape plane buffer matches the C's fixed b3PlaneResult buffer[64].
const MOVER_PLANE_CAPACITY = 64;

/**
 * Collide a capsule `mover` (at `origin`) against the world, reporting the collision planes of each
 * touched shape to `fcn` (b3World_CollideMover). The planes can be fed to {@link solvePlanes} to
 * resolve character movement.
 */
export function collideMover(
    world: WorldState,
    origin: Pos,
    mover: Capsule,
    filter: QueryFilter,
    fcn: PlaneResultFcn,
): void {
    (world.broadPhase as BroadPhase).store.refreshIfStale();
    const trees = (world.broadPhase as BroadPhase).trees;
    const filterBits = toQueryFilterBits(filter);

    // Bound the mover in origin-relative space, then lift to a conservative world float box.
    const relBox = aabb.make([mover.center1, mover.center2], 2, mover.radius);
    const box = aabb.offset(relBox, toVec3(origin));

    for (let i = 0; i < BODY_TYPE_COUNT; ++i) {
        tree.query(
            trees[i],
            box,
            filterBits.maskHi,
            filterBits.maskLo,
            false,
            (_proxyId, userData) => {
                const shape = world.shapes[userData];
                if (shouldQueryCollide(shape.filter, filterBits) === false) {
                    return true;
                }

                // Re-center on the query origin: the mover and the resulting planes are origin-relative.
                const body = world.bodies[shape.bodyId];
                const transform = toRelativeTransform(getBodyTransformQuick(world, body), origin);

                const planes = collideMoverShape(shape, transform, mover, MOVER_PLANE_CAPACITY);
                if (planes.length > 0) {
                    return fcn(shapeId(world, shape), planes);
                }
                return true;
            },
        );
    }
}

/**
 * Cast a capsule `mover` from `origin` along `translation`, returning the earliest fraction of contact
 * (b3World_CastMover, 1 when clear). `fcn` optionally accepts/skips shapes per-hit. Overlapping shapes
 * (fraction 0) are ignored.
 */
export function castMover(
    world: WorldState,
    origin: Pos,
    mover: Capsule,
    translation: Vec3,
    filter: QueryFilter,
    fcn: MoverFilterFcn | null,
): number {
    (world.broadPhase as BroadPhase).store.refreshIfStale();
    const trees = (world.broadPhase as BroadPhase).trees;
    const filterBits = toQueryFilterBits(filter);

    // The origin-relative shape cast input, rebuilt per candidate from the tree's advancing fraction.
    const castInput: ShapeCastInput = {
        proxy: { points: [mover.center1, mover.center2], count: 2, radius: mover.radius },
        translation,
        maxFraction: 1,
        canEncroach: mover.radius > 0,
    };
    const ctx = { fraction: 1 };

    const callback = (boxMaxFraction: number, userData: number): number => {
        const shape = world.shapes[userData];
        if (shouldQueryCollide(shape.filter, filterBits) === false) {
            return ctx.fraction;
        }

        if (fcn !== null) {
            if (fcn(shapeId(world, shape)) === false) {
                return ctx.fraction;
            }
        }

        // Rebuild from the origin-relative input, taking only the advancing fraction from the tree.
        const localInput: ShapeCastInput = {
            proxy: castInput.proxy,
            translation: castInput.translation,
            maxFraction: boxMaxFraction,
            canEncroach: castInput.canEncroach,
        };

        const body = world.bodies[shape.bodyId];
        const transform = toRelativeTransform(getBodyTransformQuick(world, body), origin);

        const output = shapeCastShape(shape, transform, localInput);
        if (output.fraction === 0) {
            // Ignore overlapping shapes.
            return ctx.fraction;
        }

        ctx.fraction = output.fraction;
        return output.fraction;
    };

    // Bound the capsule in origin-relative space then lift to a conservative world float box.
    const localBox = aabb.make([mover.center1, mover.center2], 2, mover.radius);
    const treeInput = { box: aabb.offset(localBox, toVec3(origin)), translation, maxFraction: 1 };

    for (let i = 0; i < BODY_TYPE_COUNT; ++i) {
        tree.boxCast(
            trees[i],
            treeInput,
            filterBits.maskHi,
            filterBits.maskLo,
            false,
            (boxInput, _id, userData) => callback(boxInput.maxFraction, userData),
        );

        if (ctx.fraction === 0) {
            break;
        }
        treeInput.maxFraction = ctx.fraction;
    }

    return ctx.fraction;
}
