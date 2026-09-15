import { ShapeKind } from "./index";

// Pure render-interpolation math for the physics backend's `compose` . Factored out so the shortest-arc nlerp + per-shape render scale are
// unit-testable without a GPU device or a live physics World.

/** shortest-arc nlerp from the quaternion at `at` in `prev` to the one at `at` in `curr`, at `t`: flip
 *  `prev` into `curr`'s hemisphere, lerp, renormalize, into `out`. Writes the identity quat if the blend
 *  degenerates. */
export function nlerpShortest<Out extends { [i: number]: number }>(
    prev: ArrayLike<number>,
    curr: ArrayLike<number>,
    t: number,
    out: Out = [0, 0, 0, 0] as unknown as Out,
    at = 0,
): Out {
    const dot =
        prev[at] * curr[at] +
        prev[at + 1] * curr[at + 1] +
        prev[at + 2] * curr[at + 2] +
        prev[at + 3] * curr[at + 3];
    const flip = dot < 0 ? -1 : 1;
    const x = prev[at] * flip * (1 - t) + curr[at] * t;
    const y = prev[at + 1] * flip * (1 - t) + curr[at + 1] * t;
    const z = prev[at + 2] * flip * (1 - t) + curr[at + 2] * t;
    const w = prev[at + 3] * flip * (1 - t) + curr[at + 3] * t;
    const len = Math.sqrt(x * x + y * y + z * z + w * w);
    if (len > 1e-12) {
        out[0] = x / len;
        out[1] = y / len;
        out[2] = z / len;
        out[3] = w / len;
    } else {
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        out[3] = 1;
    }
    return out;
}

/** the render scale mapping a `Body`'s collider to its unit render mesh, into `out` : box/hull → `2·halfExtents`, sphere → uniform
 *  `2·radius`, capsule → `(2·radius, halfExtents.y + radius, 2·radius)` (the caps distort under a
 *  non-proportional ratio — render-only; the collider stays exact). */
export function renderScale<Out extends { [i: number]: number }>(
    shape: number,
    halfExtents: ArrayLike<number>,
    radius: number,
    out: Out = [0, 0, 0] as unknown as Out,
): Out {
    if (shape === ShapeKind.Sphere) {
        out[0] = 2 * radius;
        out[1] = 2 * radius;
        out[2] = 2 * radius;
    } else if (shape === ShapeKind.Capsule) {
        out[0] = 2 * radius;
        out[1] = halfExtents[1] + radius;
        out[2] = 2 * radius;
    } else {
        out[0] = 2 * halfExtents[0];
        out[1] = 2 * halfExtents[1];
        out[2] = 2 * halfExtents[2];
    }
    return out;
}
