import { awakeContactCount } from "../collision/contact";
// The per-step driver — Box3D's b3World_Step (physics_world.c, Erin Catto, MIT). One step updates
// the broad-phase pairs, runs narrow-phase collision, then solves and integrates. The world-state
// hash (the regression contract) is taken by the caller after the step returns.
//

import { collide } from "../collision/collide";
import { updateBroadPhasePairs } from "../collision/pairs";
import { maxInt } from "../common/math";
import { reserveBodies } from "../kernel/bodycolumns";
import { kernel } from "../kernel/kernel";
import { resetStepProfile } from "../world/profile";
import { overlapSensors } from "../world/sensor";
import type { WorldState } from "../world/world";
import type { StepContext } from "./contactsolver";

import { solve } from "./solver";

/** Build the reusable per-step solver context shell. Its scalar fields are rewritten and its collections
 * cleared at the top of every `step`; one context lives per world (`world.stepContext`) and dies with it. */
function newStepContext(world: WorldState): StepContext {
    return {
        world,
        bodyCount: 0,
        dt: 0,
        invDt: 0,
        h: 0,
        invH: 0,
        subStepCount: 1,
        contactSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        staticSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        restitutionThreshold: 0,
        maxLinearVelocity: 0,
        enableWarmStarting: false,
        awakeIslands: [],
        splitIslandId: -1,
        splitSleepTime: 0,
        bulletBodies: [],
    };
}

let contextView = new Float32Array(0);
function readStepContext(world: WorldState, context: StepContext): void {
    const k = kernel(world.ecsState);
    const ptr = k.stepContext(
        context.dt,
        context.subStepCount,
        world.contactHertz,
        world.contactDampingRatio,
    );
    if (contextView.buffer !== k.memory.buffer || contextView.byteOffset !== ptr)
        contextView = new Float32Array(k.memory.buffer, ptr, 10);
    context.dt = contextView[0];
    context.invDt = contextView[1];
    context.h = contextView[2];
    context.invH = contextView[3];
    context.contactSoftness.biasRate = contextView[4];
    context.contactSoftness.massScale = contextView[5];
    context.contactSoftness.impulseScale = contextView[6];
    context.staticSoftness.biasRate = contextView[7];
    context.staticSoftness.massScale = contextView[8];
    context.staticSoftness.impulseScale = contextView[9];
}

/** Advance the world by one time step, sub-stepped `subStepCount` times (b3World_Step). */
export function step(world: WorldState, timeStep: number, subStepCount: number): void {
    world.locked = true;
    world.broadPhase.store.initialize();
    kernel(world.ecsState).bodySetActiveWorld(world.worldId);
    kernel(world.ecsState).shapeSetActiveWorld(world.worldId);
    const profile = world.profile;
    resetStepProfile(profile);
    const stepStart = performance.now();
    let phaseStart: number;

    kernel(world.ecsState).eventBeginStep(world.worldId);
    world.bodyStore.syncCount = 0;
    world.jointEventUserData.fill(null);

    // Update collision pairs and create contacts.
    phaseStart = performance.now();
    updateBroadPhasePairs(world);
    profile.pairs = performance.now() - phaseStart;

    // Reuse the per-world context across steps: rewrite every scalar field and clear the collections so no
    // stale per-step data is observable. `sims` is assigned inside solve(); `awakeIslands`
    // is reassigned to a scratch by finalize — none are read before those points, so they need no reset here.
    const context = world.stepContext ?? (world.stepContext = newStepContext(world));
    context.dt = timeStep;
    context.invDt = 0;
    context.h = 0;
    context.invH = 0;
    context.subStepCount = maxInt(1, subStepCount);
    context.restitutionThreshold = world.restitutionThreshold;
    context.maxLinearVelocity = world.maxLinearSpeed;
    context.enableWarmStarting = world.enableWarmStarting;
    context.splitIslandId = -1;
    context.splitSleepTime = 0;
    context.bulletBodies.length = 0;

    readStepContext(world, context);
    world.invH = context.invH;
    world.invDt = context.invDt;

    // Reserve for the body high-water, not the awake set: a mid-step wake must not allocate.
    if (reserveBodies(world.ecsState, kernel(world.ecsState).bodyLength(world.worldId))) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
        world.shapeStore.refreshViews();
    }

    // Narrow phase: update contacts.
    phaseStart = performance.now();
    if (awakeContactCount(world) !== 0) collide(context);
    profile.collide = performance.now() - phaseStart;

    // Integrate velocities, solve velocity constraints, integrate positions.
    if (timeStep > 0) {
        phaseStart = performance.now();
        solve(world, context);
        profile.solve = performance.now() - phaseStart;
    }

    // Refresh sensor overlaps and publish begin/end touch events (after solve, so continuous hits
    // from this step are already recorded).
    phaseStart = performance.now();
    overlapSensors(world);
    profile.sensors = performance.now() - phaseStart;

    kernel(world.ecsState).eventEndStep(world.worldId);

    profile.step = performance.now() - stepStart;
    world.locked = false;
}
