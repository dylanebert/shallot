import { ShapeKind } from "../physics";

// Pure render-interpolation math for the tumble backend's `compose` — the CPU twin of AVBD's
// `COMPOSE_PASS_WGSL` (avbd/step.ts). Factored out so the shortest-arc nlerp + per-shape render scale are
// unit-testable without a GPU device or a live tumble World.

/** shortest-arc nlerp from `prev` to `curr` at `t` (`XFORM_WGSL`'s `nlerpShortest`, ported to CPU): flip
 *  `prev` into `curr`'s hemisphere, lerp, renormalize. Returns the identity quat if the blend degenerates. */
export function nlerpShortest(
    prev: readonly [number, number, number, number],
    curr: readonly [number, number, number, number],
    t: number,
): [number, number, number, number] {
    const dot = prev[0] * curr[0] + prev[1] * curr[1] + prev[2] * curr[2] + prev[3] * curr[3];
    const flip = dot < 0 ? -1 : 1;
    const x = prev[0] * flip * (1 - t) + curr[0] * t;
    const y = prev[1] * flip * (1 - t) + curr[1] * t;
    const z = prev[2] * flip * (1 - t) + curr[2] * t;
    const w = prev[3] * flip * (1 - t) + curr[3] * t;
    const len = Math.sqrt(x * x + y * y + z * z + w * w);
    return len > 1e-12 ? [x / len, y / len, z / len, w / len] : [0, 0, 0, 1];
}

/** the render scale mapping a `Body`'s collider to its unit render mesh (avbd/step.ts `COMPOSE_PASS_WGSL`,
 *  physics.md "Storage + the Body / Transform contract"): box/hull → `2·halfExtents`, sphere → uniform
 *  `2·radius`, capsule → `(2·radius, halfExtents.y + radius, 2·radius)` (the caps distort under a
 *  non-proportional ratio — render-only; the collider stays exact). */
export function renderScale(
    shape: number,
    halfExtents: readonly [number, number, number],
    radius: number,
): [number, number, number] {
    if (shape === ShapeKind.Sphere) {
        return [2 * radius, 2 * radius, 2 * radius];
    }
    if (shape === ShapeKind.Capsule) {
        return [2 * radius, halfExtents[1] + radius, 2 * radius];
    }
    return [2 * halfExtents[0], 2 * halfExtents[1], 2 * halfExtents[2]];
}
