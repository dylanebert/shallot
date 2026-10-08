import { readJointEventUserData } from "../kernel/jointcolumns";
import { kernel, threads } from "../kernel/kernel";
import {
    defaultFrictionCallback,
    defaultRestitutionCallback,
    type WorldState,
} from "../world/world";
import { solve } from "./solver";

/** Advance the kernel world; TypeScript supplies settings and drives its yields. */
export function step(world: WorldState, timeStep: number, subStepCount: number): void {
    if (world.locked) return;
    world.locked = true;
    try {
        world.broadPhase.store.initialize();
        world.bodyStore.syncCount = 0;
        world.jointEventUserDataCount = 0;
        const k = kernel(world.ecsState);
        k.stepBegin(
            world.worldId,
            timeStep,
            subStepCount,
            threads(world.ecsState),
            world.gravity.x,
            world.gravity.y,
            world.gravity.z,
            world.contactHertz,
            world.contactDampingRatio,
            world.maxLinearSpeed,
            world.contactSpeed,
            world.restitutionThreshold,
            world.hitEventThreshold,
            world.contactRecycleDistance,
            world.enableWarmStarting,
            world.enableContinuous,
            world.enableSleep,
            world.frictionCallback === defaultFrictionCallback &&
                world.restitutionCallback === defaultRestitutionCallback,
        );
        solve(world);
        readJointEventUserData(world);
        world.invDt = k.stepInvDt();
        world.invH = k.stepInvH();
        if (timeStep > 0) ++world.stepIndex;
        world.bodyStore.syncCount = k.bodySyncCount();
        world.bodyStore.refreshViews();
        world.manifoldStore.refreshViews();
        world.shapeStore.refreshViews();
        world.broadPhase.store.refreshIfStale();
    } finally {
        world.locked = false;
    }
}
