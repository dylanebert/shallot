// Continuous collision detection (CCD) — Box3D's solver.c b3SolveContinuous + b3ContinuousQueryCallback
// (Erin Catto, MIT). A body that would move more than half its smallest extent in one step is "fast":
// instead of the discrete advance it is swept from its start pose (center0/rotation0) to its solved
// pose, and the earliest time of impact against static geometry (or, for bullets, kinematic + dynamic
// too) stops it there, preventing tunnelling.
//
// The sweep is re-centered on the body's start position so the TOI stays in float precision far from
// the origin. Non-bullet fast bodies are solved inline during finalize (they query only the static
// tree, which no one mutates mid-step); bullets are deferred to a stage after the dynamic proxies have
// been enlarged, so their swept query sees everyone's final AABB. Every op is fround-wrapped per
// the README. A static mesh or height-field target sweeps per-triangle (b3MeshTimeOfImpactFcn); a compound
// target sweeps per-child through its inner tree (b3CompoundTimeOfImpactFcn), each child dispatched by
// type (the mesh child reuses the per-triangle path).

import { NULL_INDEX } from "./array";
import { type Body, BodyFlags, type BodySim, getBodySim } from "./body";
import * as bp from "./broadphase";
import { type CompoundData, getCompoundChild, queryCompound } from "./compound";
import { ALL_BITS_HI, ALL_BITS_LO, LINEAR_SLOP, SPECULATIVE_DISTANCE } from "./core";
import { type Sweep, type TOIInput, type TOIOutput, TOIState, timeOfImpact } from "./distance";
import type { Capsule, Sphere } from "./geometry";
import { type HeightFieldData, queryHeightField } from "./heightfield";
import type { HullData } from "./hull";
import {
    type AABB,
    aabb,
    f32,
    maxf,
    quat,
    type Transform,
    type Vec3,
    vec3,
    type WorldTransform,
    xf,
} from "./math";
import { type Mesh, queryMesh } from "./mesh";
import { shouldBodiesCollide, shouldShapesCollide } from "./pairs";
import { recordSensorHit } from "./sensor";
import {
    computeFatShapeAABB,
    computeShapeAABB,
    computeShapeExtent,
    computeSweptShapeAABB,
    getShapeCentroid,
    makeShapeProxy,
    type Shape,
} from "./shape";
import * as tree from "./tree";
import { BodyType, ShapeType } from "./types";
import { setMoveTransform, type WorldState } from "./world";

/** Max continuous sensor hits recorded per fast body (B2_MAX_CONTINUOUS_SENSOR_HITS). */
const MAX_CONTINUOUS_SENSOR_HITS = 8;

/** A zeroed TOI output (b3TOIOutput{0}): no hit until a triangle registers one. */
function zeroTOIOutput(): TOIOutput {
    return {
        state: TOIState.Unknown,
        point: vec3.zero(),
        normal: vec3.zero(),
        fraction: 0,
        distance: 0,
        distanceIterations: 0,
        pushBackIterations: 0,
        rootIterations: 0,
        usedFallback: false,
    };
}

/**
 * Per-triangle TOI callback for a convex shape swept against a static mesh (b3MeshTimeOfImpactFcn).
 * The mesh is assumed static: the triangle is a fixed 3-point proxy for shape A, and shape B (the fast
 * shape) sweeps toward it. An early out skips triangles the centroid starts behind or finishes clear
 * of; a fraction-0 hit falls back to a small sphere around the fast shape's centroid. `ctx.input`
 * carries the swept query and its shrinking `maxFraction`; the earliest hit lands in `ctx.output`.
 */
type MeshImpactContext = {
    input: TOIInput;
    output: TOIOutput;
    localCentroidB: Vec3;
    meshLocalCentroidB1: Vec3;
    meshLocalCentroidB2: Vec3;
    fallbackRadius: number;
    isSensor: boolean;
};

function meshTimeOfImpactFcn(ctx: MeshImpactContext, a: Vec3, b: Vec3, c: Vec3): boolean {
    // Early out for parallel movement: project the swept centroid onto the triangle plane.
    const c1 = ctx.meshLocalCentroidB1;
    const c2 = ctx.meshLocalCentroidB2;

    const n = vec3.normalize(vec3.cross(vec3.sub(b, a), vec3.sub(c, a)));
    const offset1 = vec3.dot(n, vec3.sub(c1, a));
    const offset2 = vec3.dot(n, vec3.sub(c2, a));

    // Started behind the triangle.
    if (offset1 < 0) {
        return true;
    }

    // Finished in front of the triangle without crossing the fallback band.
    if (
        ctx.isSensor === false &&
        f32(offset1 - offset2) < ctx.fallbackRadius &&
        offset2 > ctx.fallbackRadius
    ) {
        return true;
    }

    ctx.input.proxyA = { points: [a, b, c], count: 3, radius: 0 };
    let output = timeOfImpact(ctx.input);

    // A hit at fraction == 0 is possible; retry with a small sphere around the fast centroid.
    if (0 < output.fraction && output.fraction < ctx.input.maxFraction) {
        ctx.output = output;
        ctx.input.maxFraction = output.fraction;
    } else if (output.fraction === 0) {
        const fallbackInput: TOIInput = {
            proxyA: ctx.input.proxyA,
            proxyB: {
                points: [ctx.localCentroidB],
                count: 1,
                radius: f32(ctx.fallbackRadius + LINEAR_SLOP),
            },
            sweepA: ctx.input.sweepA,
            sweepB: ctx.input.sweepB,
            maxFraction: ctx.input.maxFraction,
        };
        output = timeOfImpact(fallbackInput);

        if (0 < output.fraction && output.fraction < ctx.input.maxFraction) {
            ctx.output = { ...output, usedFallback: true };
            ctx.input.maxFraction = output.fraction;
        }
    }

    // Continue the query.
    return true;
}

/**
 * Time of impact between a target shape and the fast shape, along their sweeps (b3ShapeTimeOfImpact).
 * The fast shape B is always convex; the static target A dispatches by type — mesh/height sweep
 * per-triangle, compound sweeps per-child, everything else is a plain convex-vs-convex TOI.
 */
function shapeTimeOfImpact(
    shapeA: Shape,
    shapeB: Shape,
    sweepA: Sweep,
    sweepB: Sweep,
    maxFraction: number,
): TOIOutput {
    if (shapeA.type === ShapeType.Mesh || shapeA.type === ShapeType.HeightField) {
        return meshTimeOfImpact(shapeA, shapeB, sweepA, sweepB, maxFraction);
    }
    if (shapeA.type === ShapeType.Compound) {
        return compoundTimeOfImpact(shapeA, shapeB, sweepA, sweepB, maxFraction);
    }
    return timeOfImpact({
        proxyA: makeShapeProxy(shapeA),
        proxyB: makeShapeProxy(shapeB),
        sweepA,
        sweepB,
        maxFraction,
    });
}

/**
 * TOI of the fast convex shape B swept against a static mesh or height-field shape A (b3ShapeTimeOfImpact
 * mesh/height branch — one code path, differing only in the triangle query). The target is assumed
 * static; only the triangles overlapping shape B's swept AABB are tested, each via
 * {@link meshTimeOfImpactFcn}. Returns the earliest impact, or a zero output if B clears the target.
 */
function meshTimeOfImpact(
    shapeA: Shape,
    shapeB: Shape,
    sweepA: Sweep,
    sweepB: Sweep,
    maxFraction: number,
): TOIOutput {
    const localCentroidB = getShapeCentroid(shapeB);

    const input: TOIInput = {
        proxyA: { points: [], count: 3, radius: 0 },
        proxyB: makeShapeProxy(shapeB),
        sweepA,
        sweepB,
        maxFraction,
    };

    // The mesh is static, so its start transform xfA maps the swept centroid/bounds into mesh space.
    const xfA: Transform = {
        p: vec3.sub(sweepA.c1, quat.rotate(sweepA.q1, sweepA.localCenter)),
        q: sweepA.q1,
    };
    const xfB1: Transform = {
        p: vec3.sub(sweepB.c1, quat.rotate(sweepB.q1, sweepB.localCenter)),
        q: sweepB.q1,
    };
    const xfB2: Transform = {
        p: vec3.sub(sweepB.c2, quat.rotate(sweepB.q2, sweepB.localCenter)),
        q: sweepB.q2,
    };

    const ctx: MeshImpactContext = {
        input,
        output: zeroTOIOutput(),
        localCentroidB,
        meshLocalCentroidB1: xf.invPoint(xfA, xf.point(xfB1, localCentroidB)),
        meshLocalCentroidB2: xf.invPoint(xfA, xf.point(xfB2, localCentroidB)),
        fallbackRadius: maxf(
            f32(0.5 * computeShapeExtent(shapeB, localCentroidB).minExtent),
            LINEAR_SLOP,
        ),
        isSensor: false,
    };

    // Swept bounds of shape B, expressed in the target's local frame.
    const bounds = computeSweptShapeAABB(shapeB, sweepB, maxFraction);
    const localBounds = aabb.transform(xf.invert(xfA), bounds);

    if (shapeA.type === ShapeType.Mesh) {
        queryMesh(shapeA.mesh as Mesh, localBounds, (a, b, c) => meshTimeOfImpactFcn(ctx, a, b, c));
    } else {
        queryHeightField(shapeA.heightField as HeightFieldData, localBounds, (a, b, c) => {
            meshTimeOfImpactFcn(ctx, a, b, c);
        });
    }

    return ctx.output;
}

/**
 * Sweep of a static compound child, in the compound's frame (b3MakeCompoundChildSweep). The child is
 * baked relative to the compound and both are static, so the sweep is degenerate: start pose = end pose.
 */
function makeCompoundChildSweep(compoundTransform: Transform, childTransform: Transform): Sweep {
    const t = xf.mul(compoundTransform, childTransform);
    return { localCenter: vec3.zero(), c1: t.p, c2: t.p, q1: t.q, q2: t.q };
}

/** Per-child sweep of the fast shape B against one child of a static compound (b3CompoundImpactContext). */
type CompoundImpactContext = {
    input: TOIInput;
    output: TOIOutput;
    compoundTransform: Transform;
    localSweepBoundsB: AABB;
    localCentroidB: Vec3;
    fallbackRadius: number;
};

/**
 * Per-child TOI callback for a fast convex shape swept against a static compound (b3CompoundTimeOfImpactFcn).
 * Each child is resolved to a temporary shape and dispatched by type: convex children sweep as a plain
 * proxy; the mesh child reuses the per-triangle {@link meshTimeOfImpactFcn} path in child-local space.
 * The earliest hit across children lands in `ctx.output`, shrinking `ctx.input.maxFraction` as it goes.
 */
function compoundTimeOfImpactFcn(
    ctx: CompoundImpactContext,
    compound: CompoundData,
    childIndex: number,
): boolean {
    const child = getCompoundChild(compound, childIndex);

    let output = zeroTOIOutput();
    ctx.input.sweepA = makeCompoundChildSweep(ctx.compoundTransform, child.transform);

    switch (child.type) {
        case ShapeType.Capsule: {
            const c = child.capsule as Capsule;
            ctx.input.proxyA = { points: [c.center1, c.center2], count: 2, radius: c.radius };
            output = timeOfImpact(ctx.input);
            break;
        }
        case ShapeType.Hull: {
            const h = child.hull as HullData;
            ctx.input.proxyA = { points: h.points, count: h.vertexCount, radius: 0 };
            output = timeOfImpact(ctx.input);
            break;
        }
        case ShapeType.Mesh: {
            const sweepB = ctx.input.sweepB;
            const meshWorldTransform = xf.mul(ctx.compoundTransform, child.transform);
            const xfB1: Transform = {
                p: vec3.sub(sweepB.c1, quat.rotate(sweepB.q1, sweepB.localCenter)),
                q: sweepB.q1,
            };
            const xfB2: Transform = {
                p: vec3.sub(sweepB.c2, quat.rotate(sweepB.q2, sweepB.localCenter)),
                q: sweepB.q2,
            };

            // A private copy of the input: the mesh path shrinks its own maxFraction across triangles,
            // and only its returned output (not that internal shrink) feeds the compound's maxFraction.
            const meshCtx: MeshImpactContext = {
                input: { ...ctx.input },
                output: zeroTOIOutput(),
                localCentroidB: ctx.localCentroidB,
                meshLocalCentroidB1: xf.invPoint(
                    meshWorldTransform,
                    xf.point(xfB1, ctx.localCentroidB),
                ),
                meshLocalCentroidB2: xf.invPoint(
                    meshWorldTransform,
                    xf.point(xfB2, ctx.localCentroidB),
                ),
                fallbackRadius: ctx.fallbackRadius,
                isSensor: false,
            };

            // The child's bounds query is in child-local space; localSweepBoundsB is compound-local.
            const localBounds = aabb.transform(xf.invert(child.transform), ctx.localSweepBoundsB);
            queryMesh(child.mesh as Mesh, localBounds, (a, b, c) =>
                meshTimeOfImpactFcn(meshCtx, a, b, c),
            );
            output = meshCtx.output;
            break;
        }
        case ShapeType.Sphere: {
            const s = child.sphere as Sphere;
            ctx.input.proxyA = { points: [s.center], count: 1, radius: s.radius };
            output = timeOfImpact(ctx.input);
            break;
        }
        default:
            throw new Error(`tumble: compound child TOI unknown type ${child.type}`);
    }

    if (output.fraction > 0 && output.fraction < ctx.input.maxFraction) {
        ctx.output = output;
        ctx.input.maxFraction = output.fraction;
    }

    // Continue the query.
    return true;
}

/**
 * TOI of the fast convex shape B swept against a static compound shape A (b3ShapeTimeOfImpact compound
 * branch). The compound is assumed static; only the children overlapping B's swept AABB are tested, each
 * via {@link compoundTimeOfImpactFcn}. Returns the earliest impact, or a zero output if B clears it.
 */
function compoundTimeOfImpact(
    shapeA: Shape,
    shapeB: Shape,
    sweepA: Sweep,
    sweepB: Sweep,
    maxFraction: number,
): TOIOutput {
    const localCentroidB = getShapeCentroid(shapeB);

    // The compound's start pose (its localCenter is zero — static, children baked to its origin).
    const compoundTransform: Transform = { p: sweepA.c1, q: sweepA.q1 };

    const input: TOIInput = {
        proxyA: { points: [], count: 0, radius: 0 },
        proxyB: makeShapeProxy(shapeB),
        sweepA,
        sweepB,
        maxFraction,
    };

    // Swept bounds of shape B, expressed in the compound's local frame.
    const bounds = computeSweptShapeAABB(shapeB, sweepB, maxFraction);
    const localBounds = aabb.transform(xf.invert(compoundTransform), bounds);

    const ctx: CompoundImpactContext = {
        input,
        output: zeroTOIOutput(),
        compoundTransform,
        localSweepBoundsB: localBounds,
        localCentroidB,
        fallbackRadius: maxf(
            f32(0.75 * computeShapeExtent(shapeB, localCentroidB).minExtent),
            SPECULATIVE_DISTANCE,
        ),
    };

    const compound = shapeA.compound as CompoundData;
    queryCompound(compound, localBounds, (childIndex) => {
        compoundTimeOfImpactFcn(ctx, compound, childIndex);
        return true;
    });

    return ctx.output;
}

/** Sweep of a body's center of mass relative to a base position (b3MakeRelativeSweep). */
function makeRelativeSweep(sim: BodySim, base: Vec3): Sweep {
    return {
        c1: vec3.sub(sim.center0, base),
        c2: vec3.sub(sim.center, base),
        q1: sim.rotation0,
        q2: sim.transform.q,
        localCenter: sim.localCenter,
    };
}

/**
 * Sweep a fast body from its start pose to its solved pose and stop it at the earliest impact
 * (b3SolveContinuous). Faithful to C: this only advances the body and flags its shapes' AABBs as
 * enlarged (shape.enlargedAABB + sim.enlargeBounds); the caller enlarges the broad-phase proxies (so
 * the deterministic move-buffer order is preserved). `isBullet` also sweeps kinematic + dynamic trees.
 */
export function solveContinuous(world: WorldState, sim: BodySim): void {
    // Re-center the sweep on the fast body so the TOI and the swept query stay in float precision.
    const base = sim.center0;
    const sweep = makeRelativeSweep(sim, base);

    // The start transform xf1 and the per-shape centroid1/centroid2 the C computes are dead (written
    // to the context, never read); only xf2 (the end transform, for the end AABB) is live.
    const xf2: Transform = {
        q: sweep.q2,
        p: vec3.sub(sweep.c2, quat.rotate(sweep.q2, sweep.localCenter)),
    };

    // Re-derive the resident tree views if a solve-column reserve detached them before this swept-query
    // pass (O(1) when fresh); the queries below read the tree pools directly.
    world.broadPhase.store.refreshIfStale();
    const staticTree = world.broadPhase.trees[BodyType.Static];
    const kinematicTree = world.broadPhase.trees[BodyType.Kinematic];
    const dynamicTree = world.broadPhase.trees[BodyType.Dynamic];
    const fastBody = world.bodies[sim.bodyId];

    const isBullet = (sim.flags & BodyFlags.isBullet) !== 0;

    // The earliest impact fraction across every shape of the fast body, shared by the query callback.
    const context = { fraction: 1 };

    // Sensor shapes swept over this step are recorded (not resolved) and, after the final impact
    // fraction is known, the ones that occurred before it are reported to their sensor (b3Solve).
    const sensorHits: { sensorId: number; visitorId: number; fraction: number }[] = [];

    const callback = (_proxyId: number, shapeId: number, fastShape: Shape): boolean => {
        // Skip same shape.
        if (shapeId === fastShape.id) {
            return true;
        }

        const shape = world.shapes[shapeId];

        // Skip same body.
        if (shape.bodyId === fastShape.bodyId) {
            return true;
        }

        // Skip sensors unless both shapes want sensor events (a sensor is detected, never resolved).
        const isSensor = shape.sensorIndex !== NULL_INDEX;
        if (
            isSensor &&
            (shape.enableSensorEvents === false || fastShape.enableSensorEvents === false)
        ) {
            return true;
        }

        // Skip filtered shapes.
        if (shouldShapesCollide(fastShape.filter, shape.filter) === false) {
            return true;
        }

        const body = world.bodies[shape.bodyId];
        const bodySim = getBodySim(world, body);

        // Skip bullets — bullet-vs-bullet is never resolved by CCD.
        if (bodySim.flags & BodyFlags.isBullet) {
            return true;
        }

        // Skip filtered bodies (a joint with collideConnected off).
        if (shouldBodiesCollide(world, fastBody, body) === false) {
            return true;
        }

        // No custom filtering / pre-solve events yet (their stages).
        const sweepA = makeRelativeSweep(bodySim, base);
        const output = shapeTimeOfImpact(shape, fastShape, sweepA, sweep, context.fraction);

        if (isSensor) {
            // Record a sensor hit only if it precedes the current solid impact and there is room.
            if (
                output.fraction <= context.fraction &&
                sensorHits.length < MAX_CONTINUOUS_SENSOR_HITS
            ) {
                sensorHits.push({
                    sensorId: shape.id,
                    visitorId: fastShape.id,
                    fraction: output.fraction,
                });
            }
        } else if (output.fraction > 0 && output.fraction < context.fraction) {
            sim.flags |= BodyFlags.hadTimeOfImpact;
            context.fraction = output.fraction;
        }

        // Continue query.
        return true;
    };

    let shapeId = fastBody.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const fastShape = world.shapes[shapeId];
        shapeId = fastShape.nextShapeId;

        const box1 = fastShape.aabb;
        // xf2 is relative to the base, so translate the end box back to world space.
        const box2 = aabb.offset(computeShapeAABB(fastShape, xf2), base);

        // Store this to avoid recomputing it in the no-impact case below.
        fastShape.aabb = box2;

        // No continuous collision for mesh/height fast shapes, or sensors — none exist yet.

        const sweptBox = aabb.union(box1, box2);
        tree.query(staticTree, sweptBox, ALL_BITS_HI, ALL_BITS_LO, false, (id, data) =>
            callback(id, data, fastShape),
        );

        if (isBullet) {
            tree.query(kinematicTree, sweptBox, ALL_BITS_HI, ALL_BITS_LO, false, (id, data) =>
                callback(id, data, fastShape),
            );
            tree.query(dynamicTree, sweptBox, ALL_BITS_HI, ALL_BITS_LO, false, (id, data) =>
                callback(id, data, fastShape),
            );
        }
    }

    // Report sensor hits that occurred before the final solid impact (b3Solve's sensor-hit pass).
    for (const hit of sensorHits) {
        if (hit.fraction < context.fraction) {
            recordSensorHit(world, hit.sensorId, hit.visitorId);
        }
    }

    if (context.fraction < 1) {
        // Time of impact: interpolate the pose to the impact and re-add the base to return to world.
        const q = quat.nlerp(sweep.q1, sweep.q2, context.fraction);
        const c = vec3.lerp(sweep.c1, sweep.c2, context.fraction);
        const origin = vec3.sub(c, quat.rotate(q, sweep.localCenter));

        const transform: WorldTransform = { p: vec3.add(base, origin), q };
        const center = vec3.add(base, c);
        sim.transform = transform;
        sim.center = center;
        // Component copies, not references: finalize mutates transform.q/center in place each step,
        // and the sweep base (rotation0/center0) must not follow it.
        sim.rotation0 = { v: { ...q.v }, s: q.s };
        sim.center0 = { ...center };

        // The move event was written at the pre-CCD pose in finalize; correct it with the impact pose.
        const body = world.bodies[sim.bodyId];
        if (body.bodyMoveIndex !== NULL_INDEX) {
            setMoveTransform(world.bodyMoveEvents[body.bodyMoveIndex], transform);
        }

        // Recompute AABBs at the interpolated transform and flag any that grew for enlargement.
        enlargeFastShapes(world, sim, fastBody, transform);
    } else {
        // No impact — advance to the solved pose (already in sim.transform/center) and flag AABBs.
        // Component copies for the same reason as the impact branch above.
        sim.rotation0 = { v: { ...sim.transform.q.v }, s: sim.transform.q.s };
        sim.center0 = { ...sim.center };
        // shape.aabb is still the tight end box (box2) from the loop above; C does NOT re-fat it here
        // (unlike the impact case), so pass recompute=false and test containment against that box.
        enlargeFastShapes(world, sim, fastBody, sim.transform, false);
    }
}

/**
 * Recompute each fast shape's fat AABB and flag it enlarged if it escaped its cached margin. Mirrors
 * b3SolveContinuous's per-shape AABB loop: the proxy itself is enlarged by the caller, not here.
 * `recompute` re-derives shape.aabb (the impact case); otherwise shape.aabb is already the end box.
 */
function enlargeFastShapes(
    world: WorldState,
    sim: BodySim,
    fastBody: Body,
    transform: WorldTransform,
    recompute = true,
): void {
    const speculativeScalar = SPECULATIVE_DISTANCE;
    let shapeId = fastBody.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];

        if (recompute) {
            shape.aabb = computeFatShapeAABB(shape, transform, speculativeScalar);
        }

        if (aabb.contains(shape.fatAABB, shape.aabb) === false) {
            const margin = shape.aabbMargin;
            shape.fatAABB = {
                lowerBound: {
                    x: f32(shape.aabb.lowerBound.x - margin),
                    y: f32(shape.aabb.lowerBound.y - margin),
                    z: f32(shape.aabb.lowerBound.z - margin),
                },
                upperBound: {
                    x: f32(shape.aabb.upperBound.x + margin),
                    y: f32(shape.aabb.upperBound.y + margin),
                    z: f32(shape.aabb.upperBound.z + margin),
                },
            };
            shape.enlargedAABB = true;
            sim.flags |= BodyFlags.enlargeBounds;
            world.fatAabbStore.write(shape.id, shape.fatAABB);
        }

        shapeId = shape.nextShapeId;
    }
}

/**
 * Enlarge a non-bullet fast body's broad-phase proxies for the shapes b3SolveContinuous flagged
 * (b3BroadPhase_EnlargeProxy — buffers a move). Runs at the body's finalize position so the move
 * buffer stays in ascending sim order.
 */
export function enlargeFastProxies(world: WorldState, fastBody: Body): void {
    let shapeId = fastBody.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        if (shape.enlargedAABB) {
            bp.enlargeProxy(world.broadPhase, shape.proxyKey, shape.fatAABB);
            shape.enlargedAABB = false;
        }
        shapeId = shape.nextShapeId;
    }
}

/**
 * Buffer a fast-bullet body's proxies into the move array (b3BufferMove) at its finalize position, so
 * the move buffer stays in deterministic order; the tree proxy itself is enlarged later in the bullet
 * stage, after all dynamic proxies are known.
 */
export function bufferFastBulletMoves(world: WorldState, fastBody: Body): void {
    let shapeId = fastBody.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        bp.bufferMove(world.broadPhase, shape.proxyKey);
        shapeId = shape.nextShapeId;
    }
}

/**
 * The deferred bullet stage (b3World_Step's bullet loop): sweep each bullet, then enlarge its already-
 * buffered dynamic-tree proxies (b3DynamicTree_EnlargeProxy — no second buffer). Bullets skip other
 * bullets, so their sweeps are order-independent.
 */
export function solveBullets(world: WorldState, bullets: BodySim[]): void {
    // The per-bullet sweeps re-fit fast shapes into the resident fat-AABB column; refresh the view once
    // before the loop (a solve column reserve may have detached it), so the raw writes below land right.
    world.fatAabbStore.refreshViews();
    for (const sim of bullets) {
        solveContinuous(world, sim);
    }

    world.broadPhase.store.refreshIfStale();
    const dynamicTree = world.broadPhase.trees[BodyType.Dynamic];
    for (const sim of bullets) {
        if ((sim.flags & BodyFlags.enlargeBounds) === 0) {
            continue;
        }
        sim.flags &= ~BodyFlags.enlargeBounds;

        const fastBody = world.bodies[sim.bodyId];
        let shapeId = fastBody.headShapeId;
        while (shapeId !== NULL_INDEX) {
            const shape = world.shapes[shapeId];
            if (shape.enlargedAABB) {
                shape.enlargedAABB = false;
                tree.enlargeProxy(dynamicTree, bp.proxyId(shape.proxyKey), shape.fatAABB);
            }
            shapeId = shape.nextShapeId;
        }
    }
}
