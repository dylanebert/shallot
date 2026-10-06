import { ContactField, contactField } from "../collision/contact";
import { contactTotalImpulse, readContactManifolds } from "../collision/manifoldstore";
import { shapeBodyId } from "../kernel/filtercolumns";
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
import { OVERFLOW_INDEX, SetType, TIME_TO_SLEEP } from "../common/constants";
import { f32, type Vec3, vec3 } from "../common/math";
import {
    type Columns,
    FIN_OUT_STRIDE,
    reserveColumns,
    S2_BODY_ID,
    S2_FLAGS,
    SIM2_STRIDE,
} from "../kernel/columns";
import { consumeContinuous, prepareContinuous, solveBullets } from "../kernel/continuouscolumns";
import { collectJointEvents, jointArrayCount, stageJointBodies } from "../kernel/jointcolumns";
import { kernel, runPool, workers } from "../kernel/kernel";
import { getShapeUserMaterialId } from "../shapes/shape";
import { BODY_TRANSIENT_FLAGS, BodyFlags, getBodySim } from "../world/body";

import { splitIsland } from "../world/island";
import { trySleepIsland } from "../world/solverset";
import type { WorldState } from "../world/world";
import {
    computeLayout,
    readbackHitEvents,
    type StepContext,
    writeColorSpans,
    writeSlots,
} from "./contactsolver";

const SPEED_CAPPED = BodyFlags.isSpeedCapped;
const TOI = BodyFlags.hadTimeOfImpact;

// Per-island awake marks, reused across steps (grow-only; valid prefix reset in `solve` before
// finalize). Module scratch is safe across sequential worlds: the buffer is write-before-read within
// one synchronous `solve` and never read across steps.
const awakeIslandsScratch: boolean[] = [];

// --- Finalize --------------------------------------------------------------------------------

/** Publish continuous hits and apply the retained body/island metadata policy after kernel finalization. */
function finalizeBodies(context: StepContext, cols: Columns): void {
    const world = context.world;
    const sims = context.sims;
    const enableSleep = world.enableSleep;
    const timeStep = context.dt;
    const count = sims.length;

    // Kernel finalization publishes one retained move record per awake body. Keep only its valid
    // prefix count here; the public World bridge reads the wasm records after the step.
    world.bodyMoveCount = count;

    const store = world.bodyStore;
    const sim2U = store.sim2U;
    const flagsU = store.flagsU;
    const outCol = cols.finOut;
    consumeContinuous(world, count, false);

    for (let simIndex = 0; simIndex < count; ++simIndex) {
        const s2o = simIndex * SIM2_STRIDE;
        const simFlags = sim2U[s2o + S2_FLAGS];
        const fast = simFlags & BodyFlags.isFast;
        if (fast && simFlags & BodyFlags.isBullet) context.bulletBodies.push(sims[simIndex]);

        const body = world.bodies[sim2U[s2o + S2_BODY_ID]];
        body.bodyMoveIndex = simIndex;

        // Kernel finalization already published this body's move identity. Keep the local index only
        // so the TS sleep policy can mark the retained kernel record if the body falls asleep.

        body.flags &= ~BODY_TRANSIENT_FLAGS;
        body.flags |= simFlags & (SPEED_CAPPED | TOI);
        const stateFlags = flagsU[simIndex];
        body.flags |= stateFlags & (SPEED_CAPPED | TOI);
        sim2U[s2o + S2_FLAGS] = (simFlags & ~BODY_TRANSIENT_FLAGS) | fast;
        flagsU[simIndex] = stateFlags & ~BODY_TRANSIENT_FLAGS;

        const sleepVelocity = outCol[simIndex * FIN_OUT_STRIDE];

        if (
            enableSleep === false ||
            (body.flags & BodyFlags.enableSleep) === 0 ||
            sleepVelocity > body.sleepThreshold
        ) {
            body.sleepTime = 0;
        } else {
            body.sleepTime = f32(body.sleepTime + timeStep);
        }

        // Any single body in an island can keep it awake; a sleepy body in a split-pending island is
        // tracked as a split candidate (ties broken by island id for determinism).
        const island = world.islands[body.islandId];
        if (body.sleepTime < TIME_TO_SLEEP) {
            context.awakeIslands[island.localIndex] = true;
        } else if (island.constraintRemoveCount > 0) {
            if (
                body.sleepTime > context.splitSleepTime ||
                (body.sleepTime === context.splitSleepTime && body.islandId > context.splitIslandId)
            ) {
                context.splitIslandId = body.islandId;
                context.splitSleepTime = body.sleepTime;
            }
        }
    }
    kernel(world.ecsState).treeEnlargePass(count, 0);
}

// --- Event build passes ----------------------------------------------------------------------

/** Emit a joint event for each joint flagged over its threshold, in ascending id order (b3Solve). */
/** Fill begin events after the solve has written the per-point normal impulses. */
function updateBeginContactImpulses(world: WorldState): void {
    const events = world.contactBeginEvents;
    for (let i = 0; i < events.length; ++i) {
        const event = events[i];
        const contact = event.contactId.index1 - 1;
        if (
            contactField(world, contact, ContactField.contactId) === NULL_INDEX ||
            contactField(world, contact, ContactField.generation) !== event.contactId.generation
        )
            continue;
        event.normalImpulse = contactTotalImpulse(world, contact);
    }
}

function buildJointEvents(context: StepContext): void {
    if (context.jointEventFlags.size === 0) {
        return;
    }
    const world = context.world;
    const worldId = world.worldId;
    const ids = [...context.jointEventFlags].sort((a, b) => a - b);
    for (const jointId of ids) {
        const joint = world.joints[jointId];
        world.jointEvents.push({
            jointId: { index1: jointId + 1, world0: worldId, generation: joint.generation },
            userData: joint.userData,
        });
    }
}

/**
 * Build the hit event for each flagged contact, in ascending id order (b3Solve's hit-event pass).
 * A contact's fastest-approaching point above the threshold with a confirmed impulse wins; the point
 * is the mid-anchor offset from the two bodies' mid-center.
 */
function buildHitEvents(context: StepContext): void {
    const world = context.world;
    const worldId = world.worldId;
    const threshold = world.hitEventThreshold;
    const ids = [...context.hitEventContacts].sort((a, b) => a - b);

    for (const contactId of ids) {
        const contact = contactId;
        const shapeA = world.shapes[contactField(world, contact, ContactField.shapeIdA)];
        const shapeB = world.shapes[contactField(world, contact, ContactField.shapeIdB)];
        const simA = getBodySim(world, world.bodies[shapeBodyId(world, shapeA.id)]);
        const simB = getBodySim(world, world.bodies[shapeBodyId(world, shapeB.id)]);
        const midCenter = vec3.lerp(simA.center, simB.center, f32(0.5));

        let approachSpeed = threshold;
        let found = false;
        let point: Vec3 = { x: 0, y: 0, z: 0 };
        let normal: Vec3 = { x: 0, y: 0, z: 0 };
        let triangleIndex = 0;

        const manifolds = readContactManifolds(world, contact);
        for (const manifold of manifolds) {
            for (let p = 0; p < manifold.pointCount; ++p) {
                const mp = manifold.points[p];
                const speed = f32(-mp.normalVelocity);
                // A speculative point may not be colliding, so require a confirmed impulse.
                if (speed > approachSpeed && mp.totalNormalImpulse > 0) {
                    approachSpeed = speed;
                    point = vec3.add(midCenter, vec3.lerp(mp.anchorA, mp.anchorB, f32(0.5)));
                    normal = manifold.normal;
                    triangleIndex = mp.triangleIndex;
                    found = true;
                }
            }
        }

        if (found) {
            world.contactHitEvents.push({
                shapeIdA: { index1: shapeA.id + 1, world0: worldId, generation: shapeA.generation },
                shapeIdB: { index1: shapeB.id + 1, world0: worldId, generation: shapeB.generation },
                contactId: {
                    index1: contact + 1,
                    world0: worldId,
                    generation: contactField(world, contact, ContactField.generation),
                },
                point,
                normal: { x: normal.x, y: normal.y, z: normal.z },
                approachSpeed,
                // shapeB is never a compound (b3CreateContact), so its childIndex is irrelevant.
                userMaterialIdA: getShapeUserMaterialId(
                    world.ecsState,
                    shapeA,
                    contactField(world, contact, ContactField.childIndex),
                    triangleIndex,
                ),
                userMaterialIdB: getShapeUserMaterialId(world.ecsState, shapeB, 0, triangleIndex),
            });
        }
    }
}

// --- Solve -----------------------------------------------------------------------------------

/** Run the full substep solve and advance the bodies (b3Solve). */
export function solve(world: WorldState, context: StepContext): void {
    // Only count steps that advance the simulation
    world.stepIndex += 1;

    const awakeSet = world.solverSets[SetType.Awake];
    const awakeBodyCount = awakeSet.bodySims.length;
    if (awakeBodyCount === 0) {
        updateBeginContactImpulses(world);
        return;
    }

    context.sims = awakeSet.bodySims;
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
    // reserveColumns may have grown wasm memory, detaching every view; re-derive the manifold store's
    // (writeSlots writes contact rows through them) and the body store's (body staging and the
    // finalize tail read resident sim/state columns through them) before either is touched. The body
    // columns are resident (bodycolumns.ts) — the awake `BodySim`/`BodyState` are views over them, so no
    // per-step marshal runs; the kernel reads them where they already live.
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    writeSlots(cols, world, layout);
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

    const jointTotal = stageJointBodies(world, layout, cols.colorSpan);
    prepareContinuous(world, context.sims);
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
    collectJointEvents(world, layout, context.jointEventFlags);
    readbackHitEvents(world, layout, context);

    // Split a deferred island (candidate collected in the previous step's sleep stage) before
    // finalize reads island indices. In C this runs as a task alongside the solve; serially it must
    // complete before finalize.
    if (world.splitIslandId !== NULL_INDEX) {
        splitIsland(world, world.splitIslandId);
    }
    world.splitIslandId = NULL_INDEX;
    profile.constraints = performance.now() - constraintsStart;

    // Kernel finalization (pose, continuous and refit) ran inside the solve crossing.
    // `profile.constraints` absorbs that task; `transforms` times the retained serial tail.
    phaseStart = performance.now();

    // Reset the per-step sleep bookkeeping (the C per-worker b3TaskContext reset before finalize).
    // The island marks reuse a grow-only module buffer — valid prefix = this step's awake island
    // count, cleared here; never read across steps (finalize writes it, the sleep pass below reads it).
    const islandCount = awakeSet.islandSims.length;
    while (awakeIslandsScratch.length < islandCount) awakeIslandsScratch.push(false);
    for (let i = 0; i < islandCount; ++i) awakeIslandsScratch[i] = false;
    context.awakeIslands = awakeIslandsScratch;
    context.splitIslandId = NULL_INDEX;
    context.splitSleepTime = 0;

    finalizeBodies(context, cols);
    profile.transforms = performance.now() - phaseStart;

    // The contact-begin records are created during collision detection, but their normal impulses are
    // only authoritative after the velocity solve has stored the warm-start columns.
    updateBeginContactImpulses(world);

    // Report joint and hit events (b3Solve, after finalize, before the bullet stage).
    phaseStart = performance.now();
    buildJointEvents(context);
    profile.jointEvents = performance.now() - phaseStart;
    phaseStart = performance.now();
    if (context.hitEventContacts.size > 0) buildHitEvents(context);
    profile.hitEvents = performance.now() - phaseStart;

    // Deferred bullet CCD: fast bullet bodies sweep the dynamic + kinematic trees, which are only
    // fully enlarged once finalize has refit every non-bullet proxy (b3World_Step's bullet stage).
    if (context.bulletBodies.length > 0) {
        phaseStart = performance.now();
        solveBullets(world, context.sims);
        profile.bullets = performance.now() - phaseStart;
    }

    // Publish the finalized, CCD-clipped pose before sleep compacts the resident body columns.
    world.bodyStore.refreshViews();
    world.bodyStore.syncCount = kernel(world.ecsState).bodySyncMoved(world.bodyMoveCount);

    // Island sleeping — must be last, because sleeping invalidates the enlarged-body bookkeeping.
    if (world.enableSleep) {
        phaseStart = performance.now();
        // Collect the split-island candidate for the next step (single worker → no cross-worker reduction).
        if (context.splitIslandId !== NULL_INDEX) {
            world.splitIslandId = context.splitIslandId;
        }

        // Reverse order because sleeping an island swap-removes it from the awake islandSims.
        const islands = awakeSet.islandSims;
        for (let islandIndex = islands.length - 1; islandIndex >= 0; --islandIndex) {
            if (context.awakeIslands[islandIndex]) {
                continue;
            }
            trySleepIsland(world, islands[islandIndex].islandId);
        }
        profile.sleepIslands = performance.now() - phaseStart;
    }
}
