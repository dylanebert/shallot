// The tumble.js `Torus` sample (`samples/src/samples/mesh.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A flat torus mesh (rotated so its hole faces up) floats above the ground; a
// ring of balls at the tube radius balances on it, a cluster over the hole drops straight through. No
// `update()`, no knobs.
//
// Creation order is load-bearing for the hash: ground, torus, then the ring of 12 followed by the cluster
// of 4 — the sample's exact order.

import {
    BodyType,
    createTorusMesh,
    makeBoxHull,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/** Author the Torus scene into `world`. No knobs. */
export function buildTorus(world: World, _params: SampleParams): void {
    world.createBody({ type: BodyType.Static }).createHull({}, makeBoxHull(50, 1, 50));

    // The torus is generated in the xy-plane (hole axis = z); rotate +90° about x so the hole points up
    // and gravity pulls balls through it.
    const mesh = createTorusMesh(28, 14, 3, 0.8);
    const s = Math.SQRT1_2;
    world
        .createBody({
            type: BodyType.Static,
            position: { x: 0, y: 4, z: 0 },
            rotation: { v: { x: s, y: 0, z: 0 }, s },
        })
        .createMesh({}, mesh);

    // A ring of droppers at the tube radius (land on the ring) plus a cluster over the hole.
    const jitter = (i: number): number => 0.3 * Math.sin(i * 12.9898);
    for (let i = 0; i < 12; i++) {
        const a = (i / 12) * Math.PI * 2;
        world
            .createBody({
                type: BodyType.Dynamic,
                position: { x: 3 * Math.cos(a), y: 8 + jitter(i), z: 3 * Math.sin(a) },
            })
            .createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.35 });
    }
    for (let i = 0; i < 4; i++) {
        world
            .createBody({
                type: BodyType.Dynamic,
                position: { x: (i % 2) * 0.6 - 0.3, y: 7 + i * 0.8, z: (i >> 1) * 0.6 - 0.3 },
            })
            .createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.3 });
    }
}
