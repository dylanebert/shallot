// Character mover: the plane solver that pushes a capsule mover out of a set of collision planes,
// and the velocity clip that projects motion along them. Ported op-for-op from Box3D's mover.c
// (Erin Catto, MIT). fround discipline per the README.
//
// The collision planes fed here come from b3CollideMover (collideMover in shape.ts), which gathers a
// b3PlaneResult per touched shape. The caller turns those into b3CollisionPlanes (adding a pushLimit
// and a clipVelocity flag), runs solvePlanes to resolve the target motion, then clipVector to remove
// the into-plane velocity component.

import { LINEAR_SLOP } from "./core";
import { absf, clampf, f32, minf, type Plane, plane, type Vec3, vec3 } from "./math";

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
 * @returns the resolved delta and the iterations used (for diagnostics).
 */
export function solvePlanes(
    targetDelta: Vec3,
    planes: CollisionPlane[],
    count: number,
): PlaneSolverResult {
    for (let i = 0; i < count; ++i) {
        planes[i].push = 0;
    }

    let delta = targetDelta;
    const tolerance = LINEAR_SLOP;

    let iteration = 0;
    for (; iteration < 20; ++iteration) {
        let totalPush = 0;
        for (let planeIndex = 0; planeIndex < count; ++planeIndex) {
            const pl = planes[planeIndex];

            // Add slop to prevent jitter
            const separation = f32(plane.separation(pl.plane, delta) + LINEAR_SLOP);

            let push = -separation;

            // Clamp accumulated push
            const accumulatedPush = pl.push;
            pl.push = clampf(f32(pl.push + push), 0, pl.pushLimit);
            push = f32(pl.push - accumulatedPush);
            delta = vec3.mulAdd(delta, push, pl.plane.normal);

            // Track total push for convergence
            totalPush = f32(totalPush + absf(push));
        }

        if (totalPush < tolerance) {
            break;
        }
    }

    return { delta, iterationCount: iteration };
}

/**
 * Remove the into-plane component of `vector` for every plane that got a push and opts into velocity
 * clipping (b3ClipVector). Used to project the mover's velocity along the surfaces it hit.
 */
export function clipVector(vector: Vec3, planes: CollisionPlane[], count: number): Vec3 {
    let v = vector;

    for (let planeIndex = 0; planeIndex < count; ++planeIndex) {
        const pl = planes[planeIndex];
        if (pl.push === 0 || pl.clipVelocity === false) {
            continue;
        }

        v = vec3.mulSub(v, minf(0, vec3.dot(v, pl.plane.normal)), pl.plane.normal);
    }

    return v;
}
