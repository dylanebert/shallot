// The tumble.js `Ragdoll` sample (`samples/src/samples/ragdoll.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A single humanoid dropped from a height — its cone/twist-limited joints let it
// fold and settle like a rag doll. No `update()`, no knobs. The humanoid factory (`buildHuman`) is shared
// with `determinism-falling-ragdolls`, ported into `tumble-ragdoll-factory.ts`.
//
// Creation order is load-bearing for the hash: the static ground, then the ragdoll's own bone + joint
// order (`buildHuman`) — the sample's exact order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";
import { buildHuman } from "./tumble-ragdoll-factory";

/** Author the Ragdoll scene into `world`. No knobs — `params` is unused, kept for the shared
 *  `SampleBuild` signature. */
export function buildRagdoll(world: World, _params: SampleParams): void {
    world.createBody({ type: BodyType.Static }).createHull({}, makeBoxHull(50, 1, 50));
    buildHuman(world, { x: 0, y: 2.6, z: 0 });
}
