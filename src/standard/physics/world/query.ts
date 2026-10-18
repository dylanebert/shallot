import type { ShapeProxy } from "../collision/distance";
import type { PlaneResult } from "../collision/mover";
import type { EntityId } from "../common/ids";
import type { AABB, Pos, Transform, Vec3, WorldTransform } from "../common/records";
import { defaultQueryFilter, type QueryFilter } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { rethrowQueryError, setQueryCallback } from "../kernel/kernel";
import { queryColumns } from "../kernel/querycolumns";
import type { TreeStats } from "../kernel/treecolumns";
import type { Capsule } from "../shapes/geometry";
import { getShapeMaterialCount } from "../shapes/shape";
import { bodyGetTransform, bodyIsValid } from "./body";
import type { WorldState } from "./world";

export type WorldCastHit = {
    shape: number;
    userData: unknown;
    point: Vec3;
    normal: Vec3;
    fraction: number;
    userMaterialId: bigint;
    triangleIndex: number;
    childIndex: number;
};
export type ClosestCastHit = Omit<WorldCastHit, "shape" | "userData"> & {
    shape: number | null;
    userData: unknown;
    hit: boolean;
};
export type BodyCastHit = {
    shape: number | null;
    userData: unknown;
    point: Vec3;
    normal: Vec3;
    fraction: number;
    triangleIndex: number;
    userMaterialId: bigint;
    hit: boolean;
};
export type BodyPlaneHit = { shape: number; plane: PlaneResult };

const emptyCastHit = (): ClosestCastHit => ({
    shape: null,
    userData: undefined,
    point: { x: 0, y: 0, z: 0 },
    normal: { x: 0, y: 0, z: 0 },
    fraction: 0,
    userMaterialId: 0n,
    triangleIndex: 0,
    childIndex: 0,
    hit: false,
});

const emptyBodyCastHit = (): BodyCastHit => ({
    shape: null,
    userData: undefined,
    point: { x: 0, y: 0, z: 0 },
    normal: { x: 0, y: 0, z: 0 },
    fraction: 0,
    triangleIndex: 0,
    userMaterialId: 0n,
    hit: false,
});

function validWorld(world: WorldState): boolean {
    return world.inUse && !world.locked;
}

function queryCastHit(
    world: WorldState,
    id: number,
    f: Float32Array,
    n: number,
    origin: Pos,
): WorldCastHit {
    return {
        shape: id,
        userData: world.shapeUserData[id],
        point: { x: origin.x + f[n + 2], y: origin.y + f[n + 3], z: origin.z + f[n + 4] },
        normal: { x: f[n + 5], y: f[n + 6], z: f[n + 7] },
        fraction: f[n + 1],
        userMaterialId: world.shapeStore.userMaterialIdAt(id, f[n + 11]),
        triangleIndex: f[n + 9],
        childIndex: f[n + 10],
    };
}

function bodyCastHit(
    world: WorldState,
    q: ReturnType<typeof queryColumns>,
    origin: Pos,
): BodyCastHit {
    const id = q.resultU[0];
    if (id === 0xffffffff) return emptyBodyCastHit();
    const f = q.resultF;
    const materialCount = getShapeMaterialCount(world, id);
    return {
        shape: id,
        userData: world.shapeUserData[id],
        point: { x: origin.x + f[6], y: origin.y + f[7], z: origin.z + f[8] },
        normal: { x: f[9], y: f[10], z: f[11] },
        fraction: f[5],
        triangleIndex: f[13],
        userMaterialId: world.shapeStore.userMaterialIdAt(
            id,
            Math.max(0, Math.min(materialCount - 1, f[15])),
        ),
        hit: true,
    };
}

export function worldCastRayClosest(
    world: WorldState,
    origin: Pos,
    translation: Vec3,
    filter: QueryFilter = defaultQueryFilter(),
): ClosestCastHit {
    if (!validWorld(world)) return emptyCastHit();
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.translation(translation);
    k.worldQuery(world.worldId, 3, 0);
    const id = q.resultU[0];
    if (id === 0xffffffff) return emptyCastHit();
    const hit = queryCastHit(world, id, q.resultF, 4, origin);
    return { ...hit, hit: true };
}

export function worldCastRay(
    world: WorldState,
    origin: Pos,
    translation: Vec3,
    callback: (hit: WorldCastHit) => number,
    filter: QueryFilter = defaultQueryFilter(),
): TreeStats {
    if (!validWorld(world)) return { nodeVisits: 0, leafVisits: 0 };
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.translation(translation);
    const previous = setQueryCallback(world.ecsState, (_kind, id, data) => {
        world.shapeStore.refreshViews();
        return callback(queryCastHit(world, id, world.shapeStore.materialF, data >>> 2, origin));
    });
    try {
        k.worldQuery(world.worldId, 2, 1);
        rethrowQueryError(world.ecsState);
    } finally {
        setQueryCallback(world.ecsState, previous);
    }
    return { nodeVisits: q.resultU[1], leafVisits: q.resultU[2] };
}

export function worldOverlapAABB(
    world: WorldState,
    box: AABB,
    callback: (shape: number) => boolean,
    filter: QueryFilter = defaultQueryFilter(),
): TreeStats {
    if (!validWorld(world)) return { nodeVisits: 0, leafVisits: 0 };
    const q = queryColumns(world);
    const k = q.prepare({ x: 0, y: 0, z: 0 }, filter);
    q.bounds(box);
    const previous = setQueryCallback(world.ecsState, (_kind, id) => Number(callback(id)));
    try {
        k.worldQuery(world.worldId, 0, 1);
        rethrowQueryError(world.ecsState);
    } finally {
        setQueryCallback(world.ecsState, previous);
    }
    return { nodeVisits: q.resultU[1], leafVisits: q.resultU[2] };
}

export function worldOverlapShape(
    world: WorldState,
    origin: Pos,
    proxy: ShapeProxy,
    callback: (shape: number) => boolean,
    filter: QueryFilter = defaultQueryFilter(),
): TreeStats {
    if (!validWorld(world)) return { nodeVisits: 0, leafVisits: 0 };
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.proxy(proxy);
    const previous = setQueryCallback(world.ecsState, (_kind, id) => Number(callback(id)));
    try {
        k.worldQuery(world.worldId, 1, 1);
        rethrowQueryError(world.ecsState);
    } finally {
        setQueryCallback(world.ecsState, previous);
    }
    return { nodeVisits: q.resultU[1], leafVisits: q.resultU[2] };
}

export function worldCastShape(
    world: WorldState,
    origin: Pos,
    proxy: ShapeProxy,
    translation: Vec3,
    callback: (hit: WorldCastHit) => number,
    filter: QueryFilter = defaultQueryFilter(),
): TreeStats {
    if (!validWorld(world)) return { nodeVisits: 0, leafVisits: 0 };
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.proxy(proxy);
    q.translation(translation);
    const previous = setQueryCallback(world.ecsState, (_kind, id, data) => {
        world.shapeStore.refreshViews();
        return callback(queryCastHit(world, id, world.shapeStore.materialF, data >>> 2, origin));
    });
    try {
        k.worldQuery(world.worldId, 4, 1);
        rethrowQueryError(world.ecsState);
    } finally {
        setQueryCallback(world.ecsState, previous);
    }
    return { nodeVisits: q.resultU[1], leafVisits: q.resultU[2] };
}

export function worldCollideMover(
    world: WorldState,
    origin: Pos,
    mover: Capsule,
    callback: (shape: number, planes: PlaneResult[]) => boolean,
    filter: QueryFilter = defaultQueryFilter(),
): void {
    if (!validWorld(world)) return;
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.mover(mover.center1, mover.center2, mover.radius);
    const previous = setQueryCallback(world.ecsState, (_kind, id, data, count) => {
        world.shapeStore.refreshViews();
        const f = world.shapeStore.materialF;
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
        return Number(callback(id, planes));
    });
    try {
        k.worldQuery(world.worldId, 5, 1);
        rethrowQueryError(world.ecsState);
    } finally {
        setQueryCallback(world.ecsState, previous);
    }
}

export function worldCastMover(
    world: WorldState,
    origin: Pos,
    mover: Capsule,
    translation: Vec3,
    filter: QueryFilter = defaultQueryFilter(),
    callback: ((shape: number) => boolean) | null = null,
): number {
    if (!validWorld(world)) return 1;
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.mover(mover.center1, mover.center2, mover.radius);
    q.translation(translation);
    const previous = setQueryCallback(
        world.ecsState,
        callback === null ? null : (_kind, id) => Number(callback(id)),
    );
    try {
        k.worldQuery(world.worldId, 6, Number(callback !== null));
        rethrowQueryError(world.ecsState);
    } finally {
        setQueryCallback(world.ecsState, previous);
    }
    return q.resultF[3];
}

function validBody(world: WorldState, body: EntityId): boolean {
    return validWorld(world) && bodyIsValid(world, body);
}

export function bodyCastRay(
    world: WorldState,
    body: EntityId,
    origin: Pos,
    translation: Vec3,
    bodyTransform: Transform,
    filter: QueryFilter = defaultQueryFilter(),
    maxFraction = 1,
): BodyCastHit {
    if (!validBody(world, body)) return emptyBodyCastHit();
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.placement(bodyTransform, origin);
    q.translation(translation);
    q.input[12] = maxFraction;
    k.bodyQuery(world.worldId, 0, bodyField(world, body.index1 - 1, BodyField.headShapeId), 0);
    return bodyCastHit(world, q, origin);
}

export function bodyCastShape(
    world: WorldState,
    body: EntityId,
    origin: Pos,
    proxy: ShapeProxy,
    translation: Vec3,
    bodyTransform: Transform,
    filter: QueryFilter = defaultQueryFilter(),
    maxFraction = 1,
    canEncroach = false,
): BodyCastHit {
    if (!validBody(world, body)) return emptyBodyCastHit();
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.placement(bodyTransform, origin);
    q.proxy(proxy);
    q.translation(translation);
    q.input[12] = maxFraction;
    q.input[13] = Number(canEncroach);
    k.bodyQuery(world.worldId, 1, bodyField(world, body.index1 - 1, BodyField.headShapeId), 0);
    return bodyCastHit(world, q, origin);
}

export function bodyOverlapShape(
    world: WorldState,
    body: EntityId,
    origin: Pos,
    proxy: ShapeProxy,
    bodyTransform: Transform,
    filter: QueryFilter = defaultQueryFilter(),
): boolean {
    if (!validBody(world, body)) return false;
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.placement(bodyTransform, origin);
    q.proxy(proxy);
    k.bodyQuery(world.worldId, 2, bodyField(world, body.index1 - 1, BodyField.headShapeId), 0);
    return q.resultU[0] !== 0xffffffff;
}

export function bodyGetClosestPoint(
    world: WorldState,
    body: EntityId,
    target: Vec3,
): { point: Vec3; distance: number } {
    if (!validBody(world, body)) return { point: { x: 0, y: 0, z: 0 }, distance: 0 };
    const bodyPoseScratch: WorldTransform = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const origin = { x: 0, y: 0, z: 0 };
    const q = queryColumns(world);
    const k = q.prepare(origin);
    q.placement(bodyGetTransform(world, body.index1 - 1, bodyPoseScratch), origin);
    q.proxy({ points: [target], count: 1, radius: 0 });
    k.bodyQuery(world.worldId, 3, bodyField(world, body.index1 - 1, BodyField.headShapeId), 0);
    return { point: { x: q.resultF[6], y: q.resultF[7], z: q.resultF[8] }, distance: q.resultF[3] };
}

export function bodyCollideMover(
    world: WorldState,
    body: EntityId,
    origin: Pos,
    mover: Capsule,
    bodyTransform: WorldTransform,
    capacity = 4,
    filter: QueryFilter = defaultQueryFilter(),
): BodyPlaneHit[] {
    if (!validBody(world, body)) return [];
    const q = queryColumns(world);
    const k = q.prepare(origin, filter);
    q.placement(bodyTransform, origin);
    q.mover(mover.center1, mover.center2, mover.radius);
    const results: BodyPlaneHit[] = [];
    const previous = setQueryCallback(world.ecsState, (_kind, id, data) => {
        const f = new Float32Array(k.memory.buffer, data, 10);
        results.push({
            shape: id,
            plane: {
                plane: { normal: { x: f[0], y: f[1], z: f[2] }, offset: f[3] },
                point: { x: f[4], y: f[5], z: f[6] },
                triangleIndex: f[7],
                childIndex: f[8],
                materialIndex: f[9],
            },
        });
        return 1;
    });
    try {
        k.bodyQuery(
            world.worldId,
            4,
            bodyField(world, body.index1 - 1, BodyField.headShapeId),
            Math.max(0, capacity),
        );
    } finally {
        setQueryCallback(world.ecsState, previous);
    }
    return results;
}
