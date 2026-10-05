import { SetType } from "../common/constants";
import type { Body } from "../world/body";
import type { WorldState } from "../world/world";

// Keep static rows above every possible awake row: contact linking can wake sleeping sets.
export function stageBodies(world: WorldState): void {
    const offsets = world.stagedBodyOffsets;
    let cursor = world.solverSets[SetType.Awake].bodySims.length;
    for (let set = SetType.FirstSleeping; set < world.solverSets.length; ++set) {
        offsets[set] = cursor;
        for (const sim of world.solverSets[set].bodySims) world.bodyStore.writeSim(cursor++, sim);
    }
    const sims = world.solverSets[SetType.Static].bodySims;
    offsets[SetType.Static] = world.bodies.length - sims.length;
    for (let i = 0; i < sims.length; ++i)
        world.bodyStore.writeSim(offsets[SetType.Static] + i, sims[i]);
}

export function bodyColumnIndex(world: WorldState, body: Body): number {
    return body.setIndex === SetType.Awake
        ? body.localIndex
        : world.stagedBodyOffsets[body.setIndex] + body.localIndex;
}
