// The tumble.js `TileFloor` sample (`samples/src/samples/compound.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. A 10x10 grid of box tiles at randomized heights baked into one static
// compound — an uneven floor from a single body — with a 6x6 grid of boxes dropped over it. No knobs, no
// `update()`. No ground body: the tile compound itself is the floor.
//
// Creation order is load-bearing for the hash: the compound body (its 100 tiles built in nested i/j loop
// order, heights drawn from one `rng(99)` stream), then the 36 dropped boxes in nested i/j loop order — the
// sample's exact order.

import {
    BodyType,
    type CompoundHullDef,
    createCompound,
    defaultSurfaceMaterial,
    makeBoxHull,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/** Deterministic PRNG (mulberry32) so the port draws the same sequence as the sample. */
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
 * Author the Tile Floor scene into `world`. No knobs — `params` is unused, kept for the shared
 * `SampleBuild` signature.
 */
export function buildTileFloor(world: World, _params: SampleParams): void {
    const material = defaultSurfaceMaterial();
    const a = 2;
    const grid = 10;
    const tile = makeBoxHull(a, 0.5 * a, a);
    const rand = rng(99);
    const hulls: CompoundHullDef[] = [];
    for (let i = 0; i < grid; ++i) {
        for (let j = 0; j < grid; ++j) {
            hulls.push({
                hull: tile,
                transform: {
                    p: {
                        x: (2 * i - grid) * a,
                        y: (rand() - 0.5) * a,
                        z: (2 * j - grid) * a,
                    },
                    q: IDENT,
                },
                material,
            });
        }
    }
    const compound = createCompound({ hulls });
    if (compound !== null) {
        world.createBody({ type: BodyType.Static }).createCompound({}, compound);
    }

    const box = makeBoxHull(0.6, 0.6, 0.6);
    for (let i = 0; i < 6; ++i) {
        for (let j = 0; j < 6; ++j) {
            world
                .createBody({
                    type: BodyType.Dynamic,
                    position: { x: (i - 2.5) * 3, y: 8 + 0.4 * j, z: (j - 2.5) * 3 },
                })
                .createHull({ density: 1 }, box);
        }
    }
}
