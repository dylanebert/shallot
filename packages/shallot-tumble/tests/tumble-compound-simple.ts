// The tumble.js `SimpleCompound` sample (`samples/src/samples/compound.ts`) reproduced near-verbatim
// through the escape-hatch `World` API. Four tilted slabs baked into one static compound body form a
// descending staircase; five loose spheres cascade down it onto a ground plane. No knobs, no `update()`.
//
// Creation order is load-bearing for the hash: the compound body (its four tilted-slab hulls built in loop
// order), then the five spheres, then the ground — the sample's exact order.

import {
    BodyType,
    type CompoundHullDef,
    createCompound,
    defaultSurfaceMaterial,
    makeBoxHull,
    type Transform,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

// A rotation of `angle` radians about the z axis (an exactly-halved quat via plain trig — the sample's own
// authoring math, not solver arithmetic, so it needs no f32 discipline).
function zRot(angle: number): Transform["q"] {
    return { v: { x: 0, y: 0, z: Math.sin(angle / 2) }, s: Math.cos(angle / 2) };
}

/**
 * Author the Simple Compound scene into `world`. No knobs — `params` is unused, kept for the shared
 * `SampleBuild` signature.
 */
export function buildCompoundSimple(world: World, _params: SampleParams): void {
    const material = defaultSurfaceMaterial();
    const slab = makeBoxHull(1.6, 0.2, 3);
    const hulls: CompoundHullDef[] = [];
    for (let i = 0; i < 4; ++i) {
        hulls.push({
            hull: slab,
            transform: { p: { x: i * 2.6 - 4, y: 5 - i * 1.3, z: 0 }, q: zRot(-0.2) },
            material,
        });
    }
    const compound = createCompound({ hulls });
    if (compound !== null) {
        world.createBody({ type: BodyType.Static }).createCompound({}, compound);
    }

    for (let i = 0; i < 5; ++i) {
        world
            .createBody({ type: BodyType.Dynamic, position: { x: -4, y: 7, z: (i - 2) * 0.9 } })
            .createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.35 });
    }

    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));
}
