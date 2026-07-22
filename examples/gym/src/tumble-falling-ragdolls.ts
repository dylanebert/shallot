// The tumble.js `FallingRagdolls` sample (`samples/src/samples/ragdoll.ts`) reproduced near-verbatim
// through the escape-hatch `World` API. A grid of ragdolls (the `grid` knob, 1-4) dropped onto the ground
// and left to settle into a pile. The sample's `update()` only tallies `getBodyEvents()` into an internal
// sleep-detection hash for its own HUD text (`render()`) — no world mutation — so it's outside the gold
// contract per the `events-hit` precedent and is not ported; `build()` only.
//
// Creation order is load-bearing for the hash: the static ground, then each grid cell's humanoid
// (`buildHuman`, shared with `ragdoll-ragdoll` via `tumble-ragdoll-factory.ts`) in row-major (ix outer, iz
// inner) order — the sample's exact order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";
import { buildHuman } from "./tumble-ragdoll-factory";

/** Author the Falling Ragdolls scene into `world`, reading the `grid` knob (1-4, the grid side length). */
export function buildFallingRagdolls(world: World, params: SampleParams): void {
    world.createBody({ type: BodyType.Static }).createHull({}, makeBoxHull(50, 1, 50));

    const n = params.grid as number;
    const spacing = 1.6;
    for (let ix = 0; ix < n; ix++) {
        for (let iz = 0; iz < n; iz++) {
            const origin = {
                x: (ix - (n - 1) / 2) * spacing,
                y: 2.6,
                z: (iz - (n - 1) / 2) * spacing,
            };
            buildHuman(world, origin);
        }
    }
}
