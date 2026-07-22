// The per-step driver — Box3D's b3World_Step (physics_world.c, Erin Catto, MIT). One step updates
// the broad-phase pairs, runs narrow-phase collision, then solves and integrates. The world-state
// hash (the regression contract) is taken by the caller after the step returns.
//
// No recording. Single-threaded and serial, so the parallel task orchestration collapses to
// straight-line calls. fround discipline per the README.

import { claimResident, reserveBodies } from "./bodycolumns";
import { collide } from "./collide";
import type { StepContext } from "./contactsolver";
import { SetType } from "./core";
import { rebuildGeometry } from "./geocolumns";
import { f32, maxInt, minf } from "./math";
import { updateBroadPhasePairs } from "./pairs";
import { elapsed, resetProfile, ticks } from "./profile";
import { overlapSensors } from "./sensor";
import { writeSoft } from "./softness";
import { solve } from "./solver";
import type { WorldState } from "./world";

/** Build the reusable per-step solver context shell. Its scalar fields are rewritten and its collections
 * cleared at the top of every `step`; one context lives per world (`world.stepContext`) and dies with it. */
function newStepContext(world: WorldState): StepContext {
    return {
        world,
        sims: [],
        states: world.solverSets[SetType.Awake].bodyStates,
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
        hitEventContacts: new Set(),
        jointEventFlags: new Set(),
    };
}

/** Advance the world by one time step, sub-stepped `subStepCount` times (b3World_Step). */
export function step(world: WorldState, timeStep: number, subStepCount: number): void {
    world.locked = true;
    // Claim the shared resident body region for this world — throws if another world took it over
    // (two live worlds can't be stepped interleaved over the singleton kernel memory).
    claimResident(world);
    const profile = world.profile;
    resetProfile(profile);

    // Reset per-step event buffers so a user never reads stale data on an early return. Truncate in
    // place (like the body move pool's valid-length reset) instead of re-minting: the API accessors map
    // these into fresh wrapped objects per call (api.ts), never expose the raw arrays, so a caller can
    // only ever hold copies and the reuse is unobservable.
    world.bodyMoveCount = 0;
    world.sensorBeginEvents.length = 0;
    world.contactBeginEvents.length = 0;
    world.contactHitEvents.length = 0;
    world.jointEvents.length = 0;

    const stepStart = ticks();

    // Update collision pairs and create contacts.
    const pairsStart = ticks();
    updateBroadPhasePairs(world);
    profile.pairs = elapsed(pairsStart);

    const awakeSet = world.solverSets[SetType.Awake];

    // Reuse the per-world context across steps: rewrite every scalar field and clear the collections so no
    // stale per-step data is observable. `sims` and `states` are (re)assigned inside solve(); `awakeIslands`
    // is reassigned to a scratch by finalize — none are read before those points, so they need no reset here.
    const context = world.stepContext ?? (world.stepContext = newStepContext(world));
    context.states = awakeSet.bodyStates;
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
    context.hitEventContacts.clear();
    context.jointEventFlags.clear();

    if (timeStep > 0) {
        context.invDt = f32(1.0 / timeStep);
        context.h = f32(timeStep / context.subStepCount);
        context.invH = f32(context.subStepCount * context.invDt);
    }

    world.invH = context.invH;
    world.invDt = context.invDt;

    // Contact softness. Hertz is reduced for large time steps. Written in place into the reused objects.
    const contactHertz = minf(world.contactHertz, f32(0.125 * context.invH));
    writeSoft(context.contactSoftness, contactHertz, world.contactDampingRatio, context.h);
    writeSoft(
        context.staticSoftness,
        f32(2.0 * contactHertz),
        f32(0.5 * world.contactDampingRatio),
        context.h,
    );

    // Size the persistent body region to the total-body high-water first, since it sits below the
    // fat-AABB + shape + manifold + geometry regions and a grow relocates them in place (detaching every
    // view). Refresh the stores before anything reads through them, including the shape store the finalize
    // refit reads. (Usually a no-op — createBody already sized it.) The fat-AABB + shape regions size
    // themselves at shape create (fataabbcolumns / shapecolumns), so no step-top reserve is needed for them.
    if (reserveBodies(world.bodies.length)) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
        world.shapeStore.refreshViews();
    }

    // Size the persistent manifold columns to this step's contacts before the geometry region, which
    // sits after them and shifts when they grow (a manifold-region grow forces a geometry re-upload).
    // A manifold grow can `memory.grow`, detaching the body store's views — re-derive them.
    if (world.manifoldStore.flush()) {
        world.geometryDirty = true;
        world.bodyStore.refreshViews();
    }

    // Flush any pending hull uploads into the kernel's static geometry columns before the narrowphase
    // reads them (deferred from shape creation so `init()` is guaranteed to have run by now).
    if (world.geometryDirty) {
        rebuildGeometry(world);
        world.geometryDirty = false;
        world.bodyStore.refreshViews();
    }

    // Narrow phase: update contacts.
    const collideStart = ticks();
    collide(context);
    profile.collide = elapsed(collideStart);

    // A mid-narrowphase manifold-pool grow moved the geometry region (and the solver columns) that sit
    // after it past the old GEO_END. Re-upload the geometry now — before solve reserves its columns from
    // GEO_END — so the solver base lands past the grown pool instead of overlapping it, and refresh the
    // store views that the re-upload's grow detached.
    if (world.manifoldStore.grew) {
        rebuildGeometry(world);
        world.manifoldStore.grew = false;
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
    }

    // Integrate velocities, solve velocity constraints, integrate positions.
    if (timeStep > 0) {
        const solveStart = ticks();
        solve(world, context);
        profile.solve = elapsed(solveStart);
    }

    // Refresh sensor overlaps and publish begin/end touch events (after solve, so continuous hits
    // from this step are already recorded).
    const sensorStart = ticks();
    overlapSensors(world);
    profile.sensors = elapsed(sensorStart);

    // Swap the double-buffered end-event arrays.
    world.endEventArrayIndex = 1 - world.endEventArrayIndex;
    world.contactEndEvents[world.endEventArrayIndex] = [];
    world.sensorEndEvents[world.endEventArrayIndex] = [];

    profile.step = elapsed(stepStart);
    world.locked = false;
}
