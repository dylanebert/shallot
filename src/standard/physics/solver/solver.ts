import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import { solveBullets } from "../kernel/continuouscolumns";
import { setSplitIslandCandidate, splitIslandCandidate } from "../kernel/islandcolumns";
import { collectJointEvents } from "../kernel/jointcolumns";
import { kernel, runPool, workers } from "../kernel/kernel";
import { setArrayCount, setArrayGet, setBodyCount } from "../kernel/solversetcolumns";
import { splitIsland } from "../world/island";
import { trySleepIsland } from "../world/solverset";
import type { WorldState } from "../world/world";
import type { StepContext } from "./contactsolver";

export function solve(world: WorldState, context: StepContext): void {
    world.stepIndex += 1;
    const k = kernel(world.ecsState);
    const count = setBodyCount(world, SetType.Awake);
    if (count === 0) {
        k.eventUpdateBeginImpulses(world.worldId);
        return;
    }
    context.bodyCount = count;
    const pool = workers(world.ecsState);
    const profile = world.profile;
    const constraintsStart = performance.now();
    k.stepSolveBuild(
        (pool?.size ?? 0) + 1,
        context.subStepCount,
        world.gravity.x,
        world.gravity.y,
        world.gravity.z,
        context.maxLinearVelocity,
        world.contactSpeed,
        world.enableWarmStarting,
        context.restitutionThreshold,
        world.hitEventThreshold,
        world.enableContinuous,
        world.enableSleep,
    );
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    if (pool) runPool(world.ecsState, pool, k.runMt);
    else k.runMt();
    const candidate = splitIslandCandidate(world);
    if (candidate !== NULL_INDEX) splitIsland(world, candidate);
    setSplitIslandCandidate(world, NULL_INDEX);
    profile.constraints = performance.now() - constraintsStart;
    let start = performance.now();
    const bullets = k.stepFinalize(count, context.dt, world.enableSleep);
    profile.transforms = performance.now() - start;
    k.eventUpdateBeginImpulses(world.worldId);
    start = performance.now();
    collectJointEvents(world);
    profile.jointEvents = performance.now() - start;
    start = performance.now();
    k.eventBuildHits(world.worldId, world.hitEventThreshold);
    profile.hitEvents = performance.now() - start;
    if (bullets) {
        start = performance.now();
        solveBullets(world, count);
        profile.bullets = performance.now() - start;
    }
    world.bodyStore.refreshViews();
    world.bodyStore.syncCount = k.bodySyncMoved(k.eventCount(world.worldId, 6));
    if (world.enableSleep) {
        start = performance.now();
        for (let index = setArrayCount(world, SetType.Awake, 1) - 1; index >= 0; --index) {
            const island = setArrayGet(world, SetType.Awake, 1, index);
            if (k.islandCanSleep(island)) trySleepIsland(world, island);
        }
        profile.sleepIslands = performance.now() - start;
    }
}
