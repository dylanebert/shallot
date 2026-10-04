// Character mover: the plane solver that pushes a capsule mover out of a set of collision planes,
// and the velocity clip that projects motion along them. Ported op-for-op from Box3D's mover.c
// (Erin Catto, MIT). fround discipline (every f32 result rounds through Math.fround, keeping bit-exact f32 parity).
//
// The collision planes fed here come from b3CollideMover (collideMover in shape.ts), which gathers a
// b3PlaneResult per touched shape. The caller turns those into b3CollisionPlanes (adding a pushLimit
// and a clipVelocity flag), runs solvePlanes to resolve the target motion, then clipVector to remove
// the into-plane velocity component.

import { LINEAR_SLOP } from "../common/constants";
import { absf, clampf, f32, minf, type Plane, type Vec3 } from "../common/math";

/** The plane between a mover and a shape, plus the closest point on that shape (b3PlaneResult). */
export type PlaneResult = {
    plane: Plane;
    point: Vec3;
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
    for (let i = 0; i < count; ++i) {
        planes[i].push = 0;
    }

    let x = targetDelta.x;
    let y = targetDelta.y;
    let z = targetDelta.z;
    const tolerance = LINEAR_SLOP;

    let iteration = 0;
    for (; iteration < 20; ++iteration) {
        let totalPush = 0;
        for (let planeIndex = 0; planeIndex < count; ++planeIndex) {
            const pl = planes[planeIndex];

            // Add slop to prevent jitter
            const n = pl.plane.normal;
            const dot = f32(f32(f32(n.x * x) + f32(n.y * y)) + f32(n.z * z));
            const separation = f32(f32(dot - pl.plane.offset) + LINEAR_SLOP);

            let push = -separation;

            // Clamp accumulated push
            const accumulatedPush = pl.push;
            pl.push = clampf(f32(pl.push + push), 0, pl.pushLimit);
            push = f32(pl.push - accumulatedPush);
            x = f32(x + f32(push * n.x));
            y = f32(y + f32(push * n.y));
            z = f32(z + f32(push * n.z));

            // Track total push for convergence
            totalPush = f32(totalPush + absf(push));
        }

        if (totalPush < tolerance) {
            break;
        }
    }

    out.delta.x = x;
    out.delta.y = y;
    out.delta.z = z;
    out.iterationCount = iteration;
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
    let x = vector.x;
    let y = vector.y;
    let z = vector.z;

    for (let planeIndex = 0; planeIndex < count; ++planeIndex) {
        const pl = planes[planeIndex];
        if (pl.push === 0 || pl.clipVelocity === false) {
            continue;
        }

        const n = pl.plane.normal;
        const s = minf(0, f32(f32(f32(x * n.x) + f32(y * n.y)) + f32(z * n.z)));
        x = f32(x - f32(s * n.x));
        y = f32(y - f32(s * n.y));
        z = f32(z - f32(s * n.z));
    }

    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}
