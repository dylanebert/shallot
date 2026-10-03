// The soft-step solver loop — Box3D's solver.c b3Solve + the body integration tasks (Erin Catto,
// MIT). The port runs the canonical colored constraint schedule, with a real overflow fallback:
// prepare each selected color, then for each substep integrate velocities, warm-start, solve (bias),
// integrate positions, and relax (no bias); after the substeps apply restitution and store impulses;
// finally advance the bodies and re-fit their broad-phase AABBs.
//
// After finalize, a deferred island split runs (b3SplitIsland), then the bullet CCD stage sweeps any
// fast bullet bodies, then the island-sleep pass moves islands with no still-moving body into sleeping
// sets. Fast non-bullet bodies are swept inline during finalize (continuous.ts). Every op is
// fround-wrapped; see the README.

import * as bp from "../collision/broadphase";
import { NULL_INDEX } from "../common/array";
import { OVERFLOW_INDEX, SetType, SPECULATIVE_DISTANCE, TIME_TO_SLEEP } from "../common/constants";
import {
    type AABB,
    aabb,
    f32,
    type Mat3,
    mat3,
    type Quat,
    type Vec3,
    vec3,
    type WorldTransform,
} from "../common/math";
import { BodyType } from "../common/types";
import {
    type Columns,
    FIN_OUT_STRIDE,
    FIN_STRIDE,
    reserveColumns,
    S2_CENTER0,
    S2_ROTATION0,
    SIM_STRIDE,
    SIM2_STRIDE,
} from "../kernel/columns";
import { countJoints, marshalJoints, readbackJointImpulses } from "../kernel/jointcolumns";
import { kernel, runPool, workers } from "../kernel/kernel";
import { isConvexRefit, S_CAND, S_ESCAPED, SHAPE_STRIDE } from "../kernel/shapecolumns";
import { computeFatShapeAABBOut, getShapeUserMaterialId, type Shape } from "../shapes/shape";
import { BODY_TRANSIENT_FLAGS, BodyFlags, type BodyState, getBodySim } from "../world/body";
import { CONSTRAINTS_SLOT, SOLVE_PHASE_SLOT } from "../world/clock";
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
import {
    bufferFastBulletMoves,
    enlargeFastProxies,
    solveBullets,
    solveContinuous,
} from "./continuous";
import { flagJointEvent } from "./joint";

const SPEED_CAPPED = BodyFlags.isSpeedCapped;
const TOI = BodyFlags.hadTimeOfImpact;

// Scratch for finalizeBodies, all reused per body (never live across bodies) so the per-body loop over
// the resident columns allocates nothing — the awake `ResidentBodySim` getters would allocate a
// Vec3/Quat/Mat3 per pose/inertia field, so finalize indexes the columns raw instead.
const finRotation = mat3.zero();
const finRotationT = mat3.zero();
const finInertiaTmp = mat3.zero(); // R · I⁻¹ (world-inertia update)
const finInvIWorld = mat3.zero(); // R · I⁻¹ · Rᵀ, staged before the column write
const finInvILocal = mat3.zero(); // invInertiaLocal, staged from the column
const finQuat: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 }; // the current rotation, read from the column
const finTransform: WorldTransform = {
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
}; // the advanced pose, for the move event + shape refit

// Per-island awake marks, reused across steps (grow-only; valid prefix reset in `solve` before
// finalize). Module scratch is safe across sequential worlds: the buffer is write-before-read within
// one synchronous `solve` and never read across steps.
const awakeIslandsScratch: boolean[] = [];

/** Read a Mat3 out of `col` at `o` into `out` (kernel row order cx, cy, cz — read_sim, body.rs). */
function readMat3(col: Float32Array, o: number, out: Mat3): void {
    out.cx.x = col[o];
    out.cx.y = col[o + 1];
    out.cx.z = col[o + 2];
    out.cy.x = col[o + 3];
    out.cy.y = col[o + 4];
    out.cy.z = col[o + 5];
    out.cz.x = col[o + 6];
    out.cz.y = col[o + 7];
    out.cz.z = col[o + 8];
}

/** Set a body's sweep base (center0, rotation0 in the sim2 column) to its just-advanced pose (center in
 * the fin column, transform.q in the sim column) — b3FinalizeBodies' `center0 = center; rotation0 = q`
 * for a discretely-advanced (non-fast) body, written column-to-column with no allocation. */
function setSweepBase(
    sim2F: Float32Array,
    s2o: number,
    finF: Float32Array,
    fo: number,
    simF: Float32Array,
    so: number,
): void {
    sim2F[s2o + S2_CENTER0] = finF[fo];
    sim2F[s2o + S2_CENTER0 + 1] = finF[fo + 1];
    sim2F[s2o + S2_CENTER0 + 2] = finF[fo + 2];
    sim2F[s2o + S2_ROTATION0] = simF[so + 28];
    sim2F[s2o + S2_ROTATION0 + 1] = simF[so + 29];
    sim2F[s2o + S2_ROTATION0 + 2] = simF[so + 30];
    sim2F[s2o + S2_ROTATION0 + 3] = simF[so + 31];
}

// --- Finalize --------------------------------------------------------------------------------

/**
 * Commit an escaped shape's refit: margin-inflate the (speculative) tight box into `shape.fatAABB`,
 * mirror it into the resident fat-AABB column (the in-kernel recycle + finalize escape tests read it),
 * and enlarge the broad-phase proxy (b3BroadPhase_EnlargeProxy). `box` is the shape's just-updated tight
 * AABB — the kernel candidate for a convex shape, the TS-computed one for a fallback shape. The tail of
 * `finalizeBodies`, factored out because both branches share it; the fat-column view is refreshed once at
 * the top of the pass, so the write here is raw (no reserve, no per-shape refresh).
 */
function commitRefit(world: WorldState, shape: Shape, box: AABB): void {
    const margin = shape.aabbMargin;
    const fat = shape.fatAABB;
    fat.lowerBound.x = f32(box.lowerBound.x - margin);
    fat.lowerBound.y = f32(box.lowerBound.y - margin);
    fat.lowerBound.z = f32(box.lowerBound.z - margin);
    fat.upperBound.x = f32(box.upperBound.x + margin);
    fat.upperBound.y = f32(box.upperBound.y + margin);
    fat.upperBound.z = f32(box.upperBound.z + margin);
    world.shapeStore.writeFatAabb(shape.id, fat);
    bp.enlargeProxy(world.broadPhase, shape.proxyKey, fat);
}

/**
 * Advance body transforms from the solved deltas and re-fit broad-phase AABBs (b3FinalizeBodies). The
 * substep solve already ran over the resident columns, so finalize consumes them directly: kernel
 * `finalize` does the per-body pose-advance arithmetic straight into the columns, and the TS tail (this
 * function) owns everything touching world state — move events, sleep, CCD, refit, island bookkeeping.
 * The pose advance already ran as the staged solve's terminal stage. The awake
 * `sim`/`fin`/`sim2` + `state` are all resident (bodycolumns.ts), so nothing marshals in or back out.
 * This loop reads/writes the pose/inertia columns **raw** (by localIndex) rather than through the
 * `ResidentBodySim` view whose vector getters would allocate per field per body; the view (`sim`) is
 * used only for scalar fields (no allocation) and the `solveContinuous` handoff. `states` is the
 * awake-set body-state view array.
 */
function finalizeBodies(context: StepContext, cols: Columns, states: BodyState[]): void {
    const world = context.world;
    const sims = context.sims;
    const enableSleep = world.enableSleep;
    const enableContinuous = world.enableContinuous;
    const timeStep = context.dt;
    const speculativeScalar = SPECULATIVE_DISTANCE;
    const count = sims.length;

    // Kernel finalization publishes one retained move record per awake body. Keep only its valid
    // prefix count here; the public World bridge reads the wasm records after the step.
    world.bodyMoveCount = count;

    const store = world.bodyStore;
    const simF = store.simF;
    const finF = store.finF;
    const sim2F = store.sim2F;
    const outCol = cols.finOut;

    // The kernel finalize wrote each convex shape's candidate AABB + escaped flag into the resident shape
    // column; refresh the shape + fat views (the solve's column reserve, or a step-top body grow, may have
    // relocated/detached them — the finalize refit is their one per-step read, so refresh here, before it).
    world.shapeStore.refreshViews();
    const shapeF = world.shapeStore.shapeF;
    const shapeU = world.shapeStore.shapeU;

    for (let simIndex = 0; simIndex < count; ++simIndex) {
        const state = states[simIndex];
        const sim = sims[simIndex];
        const so = simIndex * SIM_STRIDE;
        const fo = simIndex * FIN_STRIDE;
        const s2o = simIndex * SIM2_STRIDE;

        // The kernel finalize advanced the pose (center/rotation/transform.p), rebuilt the world inertia,
        // and zeroed the deltas + force/torque straight into the resident columns. Read the advanced pose
        // into scratch (no per-field alloc) for the move event + shape refit below; `state` (velocity +
        // reset deltas + the transient IS_SPEED_CAPPED flag) is already resident for the next step.
        finTransform.p.x = finF[fo + 9];
        finTransform.p.y = finF[fo + 10];
        finTransform.p.z = finF[fo + 11];
        finTransform.q.v.x = simF[so + 28];
        finTransform.q.v.y = simF[so + 29];
        finTransform.q.v.z = simF[so + 30];
        finTransform.q.s = simF[so + 31];

        const body = world.bodies[sim.bodyId];
        body.bodyMoveIndex = simIndex;

        // Kernel finalization already published this body's move identity. Keep the local index only
        // so the TS sleep policy can mark the retained kernel record if the body falls asleep.

        body.flags &= ~BODY_TRANSIENT_FLAGS;
        body.flags |= sim.flags & (SPEED_CAPPED | TOI);
        body.flags |= state.flags & (SPEED_CAPPED | TOI);
        sim.flags &= ~BODY_TRANSIENT_FLAGS;
        state.flags &= ~BODY_TRANSIENT_FLAGS;

        // The kernel emits the two sleep/continuous decision scalars; TS owns the branches.
        const oo = simIndex * FIN_OUT_STRIDE;
        const sleepVelocity = outCol[oo];
        const maxMotion = outCol[oo + 1];

        if (
            enableSleep === false ||
            (body.flags & BodyFlags.enableSleep) === 0 ||
            sleepVelocity > body.sleepThreshold
        ) {
            // Body is not sleepy
            body.sleepTime = 0;

            const safetyFactor = f32(0.5);
            if (
                body.type === BodyType.Dynamic &&
                enableContinuous &&
                maxMotion > f32(safetyFactor * sim.minExtent)
            ) {
                // Fast body: sweep it to its first impact instead of the discrete advance. The isFast
                // flag is retained for the refit branch below (and for debug draw). Bullets are
                // deferred to the post-finalize stage (they must sweep the enlarged dynamic proxies);
                // non-bullets sweep static geometry now, which no one mutates mid-finalize. (solveContinuous
                // writes the impact pose through the view; the inertia recompute below re-reads it raw.)
                sim.flags |= BodyFlags.isFast;
                if (sim.flags & BodyFlags.isBullet) {
                    context.bulletBodies.push(sim);
                } else {
                    solveContinuous(world, sim);
                }
            }
            // else: the body advances discretely. Its sweep base (center0 = center, rotation0 = q) was
            // already written by the kernel finalize for every non-fast body — no TS copy needed here.
        } else {
            // Body is safe to advance and is falling asleep. The kernel wrote the sweep base for a
            // non-fast body, but not for a fast candidate (skipped there), and a sleepy fast candidate
            // still needs one — so write it here, the sleepy branch's own copy (harmlessly redundant
            // with the kernel's for a non-fast sleepy body).
            setSweepBase(sim2F, s2o, finF, fo, simF, so);
            body.sleepTime = f32(body.sleepTime + timeStep);
        }

        // Update world-space inverse inertia tensor. The kernel finalize already wrote it (from the
        // finalize rotation) for every body, so a discretely-advanced body needs no re-derivation. A
        // CCD-clipped fast body's transform.q was just changed by solveContinuous, so C recomputes it
        // from the post-sweep rotation (solver.c b3FinalizeBodiesTask, after b3SolveContinuous); match
        // that for fast bodies only — read the current rotation + invInertiaLocal and write
        // R · invInertiaLocal · Rᵀ back raw.
        if (sim.flags & BodyFlags.isFast) {
            finQuat.v.x = simF[so + 28];
            finQuat.v.y = simF[so + 29];
            finQuat.v.z = simF[so + 30];
            finQuat.s = simF[so + 31];
            mat3.fromQuatOut(finQuat, finRotation);
            readMat3(simF, so + 10, finInvILocal);
            mat3.mulOut(finRotation, finInvILocal, finInertiaTmp);
            mat3.transposeOut(finRotation, finRotationT);
            mat3.mulOut(finInertiaTmp, finRotationT, finInvIWorld);
            simF[so + 19] = finInvIWorld.cx.x;
            simF[so + 20] = finInvIWorld.cx.y;
            simF[so + 21] = finInvIWorld.cx.z;
            simF[so + 22] = finInvIWorld.cy.x;
            simF[so + 23] = finInvIWorld.cy.y;
            simF[so + 24] = finInvIWorld.cy.z;
            simF[so + 25] = finInvIWorld.cz.x;
            simF[so + 26] = finInvIWorld.cz.y;
            simF[so + 27] = finInvIWorld.cz.z;
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

        // Update shape AABBs and re-fit enlarged proxies in place (b3Solve's refit stage, folded into
        // finalize so the move buffer stays in ascending sim order). Fast bodies already had their
        // AABBs computed in the sweep: a non-bullet enlarges its proxies now; a bullet only buffers
        // the move (its dynamic-tree proxy is enlarged later, in the bullet stage).
        if (sim.flags & BodyFlags.isFast) {
            if (sim.flags & BodyFlags.isBullet) {
                bufferFastBulletMoves(world, body);
            } else {
                enlargeFastProxies(world, body);
            }
            continue;
        }

        // Non-fast body: commit the refit the kernel computed. For each convex shape the kernel wrote its
        // candidate AABB (the tight box + speculative margin) and an escaped flag into the shape column;
        // copy the candidate into `shape.aabb` (the mesh narrowphase + sensors read it next step) and
        // enlarge the proxy only if it escaped its cached fat margin. Fallback shapes (mesh/height-field/
        // compound) the kernel skips — compute them here at their list position, exactly as before. The
        // walk stays in ascending sim / head→next order, so the buffered moves keep their world-hash order.
        let shapeId = body.headShapeId;
        while (shapeId !== -1) {
            const shape = world.shapes[shapeId];
            if (isConvexRefit(shape.type)) {
                const c = shape.id * SHAPE_STRIDE + S_CAND;
                const box = shape.aabb;
                box.lowerBound.x = shapeF[c];
                box.lowerBound.y = shapeF[c + 1];
                box.lowerBound.z = shapeF[c + 2];
                box.upperBound.x = shapeF[c + 3];
                box.upperBound.y = shapeF[c + 4];
                box.upperBound.z = shapeF[c + 5];
                if (shapeU[shape.id * SHAPE_STRIDE + S_ESCAPED] !== 0) {
                    commitRefit(world, shape, box);
                }
            } else {
                // In-place refit: shape.aabb/fatAABB are shape-owned (the tree clones on enlarge).
                const box = computeFatShapeAABBOut(
                    shape,
                    finTransform,
                    speculativeScalar,
                    shape.aabb,
                );
                if (aabb.contains(shape.fatAABB, box) === false) {
                    commitRefit(world, shape, box);
                }
            }
            shapeId = shape.nextShapeId;
        }
    }
}

// --- Event build passes ----------------------------------------------------------------------

/** Emit a joint event for each joint flagged over its threshold, in ascending id order (b3Solve). */
/** Fill begin events after the solve has written the per-point normal impulses. */
function updateBeginContactImpulses(world: WorldState): void {
    const events = world.contactBeginEvents;
    for (let i = 0; i < events.length; ++i) {
        const event = events[i];
        const contact = world.contacts[event.contactId.index1 - 1];
        if (!contact || contact.generation !== event.contactId.generation) continue;
        let impulse = 0;
        for (let m = 0; m < contact.manifoldCount; ++m) {
            const manifold = contact.manifolds[m];
            for (let p = 0; p < manifold.pointCount; ++p) {
                impulse = f32(impulse + manifold.points[p].totalNormalImpulse);
            }
        }
        event.normalImpulse = impulse;
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
    if (context.hitEventContacts.size === 0) {
        return;
    }
    const world = context.world;
    const worldId = world.worldId;
    const threshold = world.hitEventThreshold;
    const ids = [...context.hitEventContacts].sort((a, b) => a - b);

    for (const contactId of ids) {
        const contact = world.contacts[contactId];
        const shapeA = world.shapes[contact.shapeIdA];
        const shapeB = world.shapes[contact.shapeIdB];
        const simA = getBodySim(world, world.bodies[shapeA.bodyId]);
        const simB = getBodySim(world, world.bodies[shapeB.bodyId]);
        const midCenter = vec3.lerp(simA.center, simB.center, f32(0.5));

        let approachSpeed = threshold;
        let found = false;
        let point: Vec3 = { x: 0, y: 0, z: 0 };
        let normal: Vec3 = { x: 0, y: 0, z: 0 };
        let triangleIndex = 0;

        for (let m = 0; m < contact.manifoldCount; ++m) {
            const manifold = contact.manifolds[m];
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
                    index1: contact.contactId + 1,
                    world0: worldId,
                    generation: contact.generation,
                },
                point,
                normal: { x: normal.x, y: normal.y, z: normal.z },
                approachSpeed,
                // shapeB is never a compound (b3CreateContact), so its childIndex is irrelevant.
                userMaterialIdA: getShapeUserMaterialId(
                    world.ecsState,
                    shapeA,
                    contact.childIndex,
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
    // Body `state` (velocity/delta/flags) is resident: `bodyStates` are offset-backed views over the
    // persistent column the kernel runs over directly. The layout fixes the contact and joint ranges.
    const persistentStates = awakeSet.bodyStates;
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
        countJoints(world, layout),
    );
    // reserveColumns may have grown wasm memory, detaching every view; re-derive the manifold store's
    // (writeSlots writes contact rows through them) and the body store's (joint marshaling and the
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

    const clock = world.clock;

    // One crossing owns the phase schedule; its per-phase profile split remains zero.
    clock.mark(CONSTRAINTS_SLOT);

    const k = kernel(world.ecsState);
    const restThreshold = context.restitutionThreshold;
    const hitThreshold = world.hitEventThreshold;
    const subStepCount = context.subStepCount;

    const jointTotal = marshalJoints(world, layout, cols);
    k.solveBuild(
        (pool?.size ?? 0) + 1,
        subStepCount,
        layout.wideTotal,
        layout.meshStart,
        layout.meshTotal,
        layout.overflowStart,
        layout.overflowCount,
        jointTotal,
        jointTotal,
        world.constraintGraph.colors[OVERFLOW_INDEX].jointSims.length,
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
    readbackJointImpulses(world, layout, cols);
    for (const span of layout.colors) {
        for (const sim of span.color.jointSims) flagJointEvent(sim, context);
    }
    for (const sim of world.constraintGraph.colors[OVERFLOW_INDEX].jointSims) {
        flagJointEvent(sim, context);
    }
    readbackHitEvents(world, layout, context);

    // Split a deferred island (candidate collected in the previous step's sleep stage) before
    // finalize reads island indices. In C this runs as a task alongside the solve; serially it must
    // complete before finalize.
    if (world.splitIslandId !== NULL_INDEX) {
        splitIsland(world, world.splitIslandId);
    }
    world.splitIslandId = NULL_INDEX;
    clock.span("constraints", CONSTRAINTS_SLOT);

    // Finalize: advance transforms, re-fit AABBs (the port folds refit in, so its cost lands here).
    // The kernel pose advance already ran inside the solve crossing, so
    // `profile.constraints` absorbs it and `transforms` times only the serial TS tail.
    clock.mark(SOLVE_PHASE_SLOT);

    // Reset the per-step sleep bookkeeping (the C per-worker b3TaskContext reset before finalize).
    // The island marks reuse a grow-only module buffer — valid prefix = this step's awake island
    // count, cleared here; never read across steps (finalize writes it, the sleep pass below reads it).
    const islandCount = awakeSet.islandSims.length;
    while (awakeIslandsScratch.length < islandCount) awakeIslandsScratch.push(false);
    for (let i = 0; i < islandCount; ++i) awakeIslandsScratch[i] = false;
    context.awakeIslands = awakeIslandsScratch;
    context.splitIslandId = NULL_INDEX;
    context.splitSleepTime = 0;

    finalizeBodies(context, cols, persistentStates);
    clock.span("transforms", SOLVE_PHASE_SLOT);

    // The contact-begin records are created during collision detection, but their normal impulses are
    // only authoritative after the velocity solve has stored the warm-start columns.
    updateBeginContactImpulses(world);

    // Report joint and hit events (b3Solve, after finalize, before the bullet stage).
    clock.mark(SOLVE_PHASE_SLOT);
    buildJointEvents(context);
    clock.span("jointEvents", SOLVE_PHASE_SLOT);
    clock.mark(SOLVE_PHASE_SLOT);
    buildHitEvents(context);
    clock.span("hitEvents", SOLVE_PHASE_SLOT);

    // Deferred bullet CCD: fast bullet bodies sweep the dynamic + kinematic trees, which are only
    // fully enlarged once finalize has refit every non-bullet proxy (b3World_Step's bullet stage).
    if (context.bulletBodies.length > 0) {
        clock.mark(SOLVE_PHASE_SLOT);
        solveBullets(world, context.bulletBodies);
        clock.span("bullets", SOLVE_PHASE_SLOT);
    }

    // Publish the finalized, CCD-clipped pose before sleep compacts the resident body columns.
    world.bodyStore.refreshViews();
    world.bodyStore.syncCount = kernel(world.ecsState).bodySyncMoved(world.bodyMoveCount);

    // Island sleeping — must be last, because sleeping invalidates the enlarged-body bookkeeping.
    if (world.enableSleep) {
        clock.mark(SOLVE_PHASE_SLOT);
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
        clock.span("sleepIslands", SOLVE_PHASE_SLOT);
    }
}
