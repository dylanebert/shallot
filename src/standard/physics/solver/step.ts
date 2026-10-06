import { awakeContactCount } from "../collision/contact";
// The per-step driver — Box3D's b3World_Step (physics_world.c, Erin Catto, MIT). One step updates
// the broad-phase pairs, runs narrow-phase collision, then solves and integrates. The world-state
// hash (the regression contract) is taken by the caller after the step returns.
//
// No recording. Single-threaded and serial, so the parallel task orchestration collapses to
// straight-line calls. fround discipline (every f32 result rounds through Math.fround, keeping bit-exact f32 parity).

import { collide } from "../collision/collide";
import { updateBroadPhasePairs } from "../collision/pairs";
import { f32, maxInt, minf } from "../common/math";
import { reserveBodies } from "../kernel/bodycolumns";
import { kernel } from "../kernel/kernel";
import { resetStepProfile } from "../world/profile";
import { overlapSensors } from "../world/sensor";
import type { WorldState } from "../world/world";
import type { StepContext } from "./contactsolver";
import { writeSoft } from "./softness";
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

function writeStepSoftness(world: WorldState, context: StepContext): void {
    const contactHertz = minf(world.contactHertz, f32(0.125 * context.invH));
    writeSoft(context.contactSoftness, contactHertz, world.contactDampingRatio, context.h);
    writeSoft(
        context.staticSoftness,
        f32(2.0 * contactHertz),
        f32(0.5 * world.contactDampingRatio),
        context.h,
    );
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

    if (timeStep > 0) {
        context.invDt = f32(1.0 / timeStep);
        context.h = f32(timeStep / context.subStepCount);
        context.invH = f32(context.subStepCount * context.invDt);
    }

    world.invH = context.invH;
    world.invDt = context.invDt;

    // Contact softness. Hertz is reduced for large time steps. Written in place into the reused objects.
    writeStepSoftness(world, context);

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
