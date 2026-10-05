import { SetType } from "../common/constants";
import { type Body, getBodySim } from "../world/body";
import type { WorldState } from "../world/world";

export function beginBodyStaging(world: WorldState): void {
    ++world.bodyStagingEpoch;
    world.bodyStagingCursor = world.solverSets[SetType.Awake].bodySims.length;
}

export function bodyColumnIndex(world: WorldState, body: Body): number {
    if (body.setIndex === SetType.Awake) return body.localIndex;
    if (world.bodyStagingStamps[body.id] !== world.bodyStagingEpoch) {
        // Static rows must survive contact linking waking any number of sleeping bodies.
        // Sleeping rows are consumed before linking; joint prepare only references awake/static sims.
        const index =
            body.setIndex === SetType.Static
                ? world.bodies.length -
                  world.solverSets[SetType.Static].bodySims.length +
                  body.localIndex
                : world.bodyStagingCursor++;
        if (world.bodyStore.stale) world.bodyStore.refreshViews();
        world.bodyStore.writeSim(index, getBodySim(world, body));
        world.bodyStagingIndices[body.id] = index;
        world.bodyStagingStamps[body.id] = world.bodyStagingEpoch;
    }
    return world.bodyStagingIndices[body.id];
}
