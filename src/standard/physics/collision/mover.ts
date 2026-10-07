import type { Plane, Vec3 } from "../common/math";
import { kernel } from "../kernel/kernel";

let buffer: ArrayBufferLike | undefined;
let capacity = 0;
let values = new Float32Array(0);
let result = new Float32Array(0);

function upload(planes: CollisionPlane[], count: number) {
    const k = kernel(undefined);
    const ptr = k.moverPlanesPtr(count);
    if (buffer !== k.memory.buffer || capacity < count) {
        buffer = k.memory.buffer;
        capacity = count;
        values = new Float32Array(buffer, ptr, count * 7);
        result = new Float32Array(buffer, k.moverOutputPtr(), 4);
    } else if (values.byteOffset !== ptr) {
        values = new Float32Array(buffer, ptr, capacity * 7);
    }
    for (let i = 0; i < count; ++i) {
        const p = planes[i];
        const n = i * 7;
        values[n] = p.plane.normal.x;
        values[n + 1] = p.plane.normal.y;
        values[n + 2] = p.plane.normal.z;
        values[n + 3] = p.plane.offset;
        values[n + 4] = p.pushLimit;
        values[n + 5] = p.push;
        // The ABI flag is a u32; use its f32 bit representation without a second view.
        values[n + 6] = p.clipVelocity ? 1.401298464324817e-45 : 0;
    }
    return k;
}

/** The plane between a mover and a shape, plus the closest point on that shape (b3PlaneResult). */
export type PlaneResult = {
    plane: Plane;
    point: Vec3;
    triangleIndex?: number;
    childIndex?: number;
    materialIndex?: number;
};

/**
 * A collision plane the mover solver resolves against (b3CollisionPlane). `pushLimit` FLT_MAX makes
 * the plane rigid; lower values soften it. `push` is filled by {@link solvePlanes}. `clipVelocity`
 * false leaves the plane out of {@link clipVector} (soft collisions).
 */
export type CollisionPlane = {
    plane: Plane;
    pushLimit: number;
    push: number;
    clipVelocity: boolean;
};

/** Result of {@link solvePlanes}: the resolved relative motion and the iteration count (b3PlaneSolverResult). */
export type PlaneSolverResult = {
    delta: Vec3;
    iterationCount: number;
};

/**
 * Resolve `targetDelta` against the collision planes, accumulating a clamped push per plane until the
 * motion no longer drives into any of them (b3SolvePlanes). Mutates each plane's `push`.
 * Writes into caller-owned `out` when supplied; otherwise returns a fresh result and delta.
 * @returns the resolved delta and the iterations used (for diagnostics).
 */
export function solvePlanes(
    targetDelta: Vec3,
    planes: CollisionPlane[],
    count: number,
    out: PlaneSolverResult = { delta: { x: 0, y: 0, z: 0 }, iterationCount: 0 },
): PlaneSolverResult {
    const k = upload(planes, count);
    k.moverSolve(0, targetDelta.x, targetDelta.y, targetDelta.z, count);
    for (let i = 0; i < count; ++i) planes[i].push = values[i * 7 + 5];
    out.delta.x = result[0];
    out.delta.y = result[1];
    out.delta.z = result[2];
    out.iterationCount = result[3];
    return out;
}

/**
 * Remove the into-plane component of `vector` for every plane that got a push and opts into velocity
 * clipping (b3ClipVector). Used to project the mover's velocity along the surfaces it hit.
 * Writes into caller-owned `out` when supplied (which may alias `vector`); otherwise returns a fresh vector.
 */
export function clipVector(
    vector: Vec3,
    planes: CollisionPlane[],
    count: number,
    out: Vec3 = { x: 0, y: 0, z: 0 },
): Vec3 {
    const k = upload(planes, count);
    k.moverSolve(1, vector.x, vector.y, vector.z, count);
    out.x = result[0];
    out.y = result[1];
    out.z = result[2];
    return out;
}
