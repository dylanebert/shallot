// The tumble.js `HullReduction` sample (`samples/src/samples/geometry.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. A sphere point cloud (60 points) per body, reduced to `createHull(points,
// maxVerts)` — the `vertices` knob (4-20, default 8) caps the hull, `count` (1-12, default 6) the row length.
//
// Creation order is load-bearing for the hash: ground, then each body left to right — the sample's exact
// order, drawing from the same `rng(42)` stream across the whole loop.

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
 * Author the Hull Reduction scene into `world`, reading the `vertices` knob (4-20, max hull vertex count)
 * and `count` knob (1-12, row length): `count` sphere-sampled point clouds (60 points each) reduced to
 * `vertices`-vertex hulls.
 */
export function buildHullReduction(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const maxVerts = params.vertices as number;
    const count = params.count as number;
    const rand = rng(42);
    for (let i = 0; i < count; ++i) {
        const points: Vec3[] = [];
        for (let p = 0; p < 60; ++p) {
            let x = rand() - 0.5;
            let y = rand() - 0.5;
            let z = rand() - 0.5;
            const len = Math.hypot(x, y, z) || 1;
            x /= len;
            y /= len;
            z /= len;
            points.push({ x, y, z });
        }
        const hull = createHull(points, maxVerts);
        if (hull === null) continue;
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: (i - (count - 1) / 2) * 2.2, y: 3, z: 0 },
        });
        body.createHull({ density: 1 }, hull);
    }
}
