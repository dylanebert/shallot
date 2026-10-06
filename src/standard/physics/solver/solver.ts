import { setSplitIslandCandidate, splitIslandCandidate } from "../kernel/islandcolumns";
import { setArrayCount, setArrayGet, setBodyCount } from "../kernel/solversetcolumns";
// The soft-step solver loop — Box3D's solver.c b3Solve + the body integration tasks (Erin Catto,
// MIT). The port runs the canonical colored constraint schedule, with a real overflow fallback:
// prepare each selected color, then for each substep integrate velocities, warm-start, solve (bias),
// integrate positions, and relax (no bias); after the substeps apply restitution and store impulses;
// finally advance the bodies and re-fit their broad-phase AABBs.
//
// After finalize, a deferred island split runs (b3SplitIsland), then the bullet CCD stage sweeps any
// fast bullet bodies, then the island-sleep pass moves islands with no still-moving body into sleeping
// sets. Fast non-bullet bodies are swept inline during kernel finalize (continuous.rs). Every op is
// fround-wrapped; see the README.

import { NULL_INDEX } from "../common/array";
import { OVERFLOW_INDEX, SetType } from "../common/constants";
import { reserveColumns, S2_FLAGS, SIM2_STRIDE } from "../kernel/columns";
import { consumeContinuous, prepareContinuous, solveBullets } from "../kernel/continuouscolumns";
import { collectJointEvents, jointArrayCount, jointSpans } from "../kernel/jointcolumns";
import { kernel, runPool, workers } from "../kernel/kernel";
import { BodyFlags } from "../world/body";
import { splitIsland } from "../world/island";
import { trySleepIsland } from "../world/solverset";
import type { WorldState } from "../world/world";
import { computeLayout, type StepContext, writeColorSpans, writeSlots } from "./contactsolver";

function finalizeBodies(context: StepContext): void {
    const world = context.world;
    const count = context.bodyCount;
    consumeContinuous(world, count, false);
    const sim2U = world.bodyStore.sim2U;
    for (let i = 0; i < count; ++i) {
        const flags = sim2U[i * SIM2_STRIDE + S2_FLAGS];
        if (flags & BodyFlags.isFast && flags & BodyFlags.isBullet) context.bulletBodies.push(i);
    }
    const k = kernel(world.ecsState);
    context.splitIslandId = k.bodyFinish(count, context.dt, world.enableSleep);
    k.treeEnlargePass(count, 0);
}

// --- Solve -----------------------------------------------------------------------------------

/** Run the full substep solve and advance the bodies (b3Solve). */
export function solve(world: WorldState, context: StepContext): void {
    // Only count steps that advance the simulation
    world.stepIndex += 1;

    const awakeSet = SetType.Awake;
    const awakeBodyCount = setBodyCount(world, awakeSet);
    if (awakeBodyCount === 0) {
        kernel(world.ecsState).eventUpdateBeginImpulses(world.worldId);
        return;
    }

    context.bodyCount = awakeBodyCount;
    // The layout fixes the contact and joint ranges.
    const layout = computeLayout(world);
    const pool = workers(world.ecsState);
    const cols = reserveColumns(
        world.ecsState,
        awakeBodyCount,
        layout.contacts,
        layout.manifolds,
        layout.points,
        layout.wide,
        layout.colors.length,
    );
    // reserveColumns may replace wasm memory; refresh the owners before writeSlots or finalization
    // reads their views.
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    writeSlots(world);
    writeColorSpans(cols, layout);

    const gravity = world.gravity;
    const h = context.h;
    const invH = context.invH;
    const contactSpeed = world.contactSpeed;
    const cs = context.contactSoftness;
    const ss = context.staticSoftness;
    const warmStartScale = world.enableWarmStarting ? 1 : 0;

    const profile = world.profile;
    let phaseStart: number;

    // One crossing owns the phase schedule; its per-phase profile split remains zero.
    const constraintsStart = performance.now();

    const k = kernel(world.ecsState);
    const restThreshold = context.restitutionThreshold;
    const hitThreshold = world.hitEventThreshold;
    const subStepCount = context.subStepCount;

    const jointTotal = jointSpans(world, layout, cols.colorSpan);
    prepareContinuous(world, context.bodyCount);
    k.solveBuild(
        (pool?.size ?? 0) + 1,
        subStepCount,
        layout.wideTotal,
        layout.meshStart,
        layout.meshTotal,
        layout.overflowStart,
        layout.overflowCount,
        jointTotal,
        jointArrayCount(world, OVERFLOW_INDEX),
        gravity.x,
        gravity.y,
        gravity.z,
        h,
        invH,
        context.dt,
        context.invDt,
        context.maxLinearVelocity,
        contactSpeed,
        cs.biasRate,
        cs.massScale,
        cs.impulseScale,
        ss.biasRate,
        ss.massScale,
        ss.impulseScale,
        warmStartScale,
        restThreshold,
        hitThreshold,
        world.enableContinuous ? 1 : 0,
    );
    if (pool) runPool(world.ecsState, pool, k.runMt);
    else k.runMt();

    // Split a deferred island (candidate collected in the previous step's sleep stage) before
    // finalize reads island indices. In C this runs as a task alongside the solve; serially it must
    // complete before finalize.
    const candidate = splitIslandCandidate(world);
    if (candidate !== NULL_INDEX) splitIsland(world, candidate);
    setSplitIslandCandidate(world, NULL_INDEX);
    profile.constraints = performance.now() - constraintsStart;

    // Kernel finalization (pose, continuous and refit) ran inside the solve crossing.
    // `profile.constraints` absorbs that task; `transforms` times the retained serial tail.
    phaseStart = performance.now();

    finalizeBodies(context);
    profile.transforms = performance.now() - phaseStart;

    // The contact-begin records are created during collision detection, but their normal impulses are
    // only authoritative after the velocity solve has stored the warm-start columns.
    k.eventUpdateBeginImpulses(world.worldId);

    // Report joint and hit events (b3Solve, after finalize, before the bullet stage).
    phaseStart = performance.now();
    collectJointEvents(world);
    profile.jointEvents = performance.now() - phaseStart;
    phaseStart = performance.now();
    k.eventBuildHits(world.worldId, world.hitEventThreshold);
    profile.hitEvents = performance.now() - phaseStart;

    // Deferred bullet CCD: fast bullet bodies sweep the dynamic + kinematic trees, which are only
    // fully enlarged once finalize has refit every non-bullet proxy (b3World_Step's bullet stage).
    if (context.bulletBodies.length > 0) {
        phaseStart = performance.now();
        solveBullets(world, context.bodyCount);
        profile.bullets = performance.now() - phaseStart;
    }

    // Publish the finalized, CCD-clipped pose before sleep compacts the resident body columns.
    world.bodyStore.refreshViews();
    world.bodyStore.syncCount = kernel(world.ecsState).bodySyncMoved(
        kernel(world.ecsState).eventCount(world.worldId, 6),
    );

    // Island sleeping — must be last, because sleeping invalidates the enlarged-body bookkeeping.
    if (world.enableSleep) {
        phaseStart = performance.now();
        // Collect the split-island candidate for the next step (single worker → no cross-worker reduction).
        if (context.splitIslandId !== NULL_INDEX) {
            setSplitIslandCandidate(world, context.splitIslandId);
        }

        // Reverse order because sleeping an island swap-removes it from the awake islandSims.
        const count = setArrayCount(world, awakeSet, 1);
        for (let islandIndex = count - 1; islandIndex >= 0; --islandIndex) {
            const island = setArrayGet(world, awakeSet, 1, islandIndex);
            if (kernel(world.ecsState).islandCanSleep(island)) trySleepIsland(world, island);
        }
        profile.sleepIslands = performance.now() - phaseStart;
    }
}
