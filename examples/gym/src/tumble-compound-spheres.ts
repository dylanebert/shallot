// The tumble.js `CompoundSpheres` sample (`samples/src/samples/compound.ts`) reproduced near-verbatim
// through the escape-hatch `World` API. A lumpy boulder of eighteen spheres baked into one static compound,
// pelted by sixteen dynamic boxes. No knobs, no `update()`.
//
// Creation order is load-bearing for the hash: ground, the compound body (its eighteen spheres drawn from
// one `rng(7)` stream), then the sixteen boxes drawn from the SAME stream continuing where the spheres left
// off — the sample's exact order and RNG sequencing.

import {
    BodyType,
    type CompoundSphereDef,
    createCompound,
    defaultSurfaceMaterial,
    makeBoxHull,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

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
 * Author the Compound Spheres scene into `world`. No knobs — `params` is unused, kept for the shared
 * `SampleBuild` signature.
 */
export function buildCompoundSpheres(world: World, _params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const material = defaultSurfaceMaterial();
    const rand = rng(7);
    const spheres: CompoundSphereDef[] = [];
    for (let i = 0; i < 18; ++i) {
        spheres.push({
            sphere: {
                center: {
                    x: (rand() - 0.5) * 4,
                    y: 3 + (rand() - 0.5) * 3,
                    z: (rand() - 0.5) * 4,
                },
                radius: 0.4 + rand() * 0.4,
            },
            material,
        });
    }
    const compound = createCompound({ spheres });
    if (compound !== null) {
        world.createBody({ type: BodyType.Static }).createCompound({}, compound);
    }

    const box = makeBoxHull(0.3, 0.3, 0.3);
    for (let i = 0; i < 16; ++i) {
        world
            .createBody({
                type: BodyType.Dynamic,
                position: { x: (rand() - 0.5) * 3, y: 7 + i * 0.4, z: (rand() - 0.5) * 3 },
            })
            .createHull({ density: 1 }, box);
    }
}
