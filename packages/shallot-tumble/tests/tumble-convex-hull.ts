// The tumble.js `ConvexHull` sample (`samples/src/samples/geometry.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. Random point clouds turned into convex polytopes via `createHull(points, 16)` —
// a grid of unique hull bodies. The `count` knob (4-24, default 12) picks the hull count.
//
// Creation order is load-bearing for the hash: ground, then each hull left to right / row by row — the
// sample's exact order, drawing from the same `rng(1234)` stream across the whole loop.

import {
    BodyType,
    createHull,
    makeBoxHull,
    type Vec3,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/** Deterministic PRNG (mulberry32) — matches the sample's own generator bit-for-bit. */
function rng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Author the Convex Hull scene into `world`, reading the `count` knob (4-24, default 12): `count` random
 * 20-point clouds each reduced to a 16-vertex convex hull, arranged in a 4-column grid.
 */
export function buildConvexHull(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const count = params.count as number;
    const rand = rng(1234);
    const cols = 4;
    for (let i = 0; i < count; ++i) {
        const points: Vec3[] = [];
        for (let p = 0; p < 20; ++p) {
            points.push({
                x: (rand() - 0.5) * 1.2,
                y: (rand() - 0.5) * 1.2,
                z: (rand() - 0.5) * 1.2,
            });
        }
        const hull = createHull(points, 16);
        if (hull === null) continue;
        const col = i % cols;
        const row = Math.floor(i / cols);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: (col - (cols - 1) / 2) * 1.6, y: 2 + row * 1.6, z: 0 },
        });
        body.createHull({ density: 1 }, hull);
    }
}
