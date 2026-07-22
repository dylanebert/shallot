// Sensor overlap tracking (Box3D's sensor.c, Erin Catto, MIT). A sensor shape detects other shapes
// overlapping it and reports begin/end touch events, in deterministic order, without producing any
// contact response. Sensors never create contacts (the pair phase skips them, pairs.ts) and never
// perturb dynamics, so the world-state hash is unaffected; correctness is behavioral, not bit-exact.
//
// Single-threaded and serial: the C's per-worker sensor task + the event-publish pass collapse into
// one loop, and the eventBits optimization drops out (a sensor whose overlaps didn't change emits no
// events regardless, so the diff always runs). fround discipline per the README.

import { NULL_INDEX } from "./array";
import { getBodyTransformQuick } from "./body";
import { MAX_SHAPE_CAST_POINTS, SetType } from "./core";
import type { EntityId } from "./ids";
import { minInt, type Transform, toRelativeTransform, type Vec3, xf } from "./math";
import { shouldShapesCollide } from "./pairs";
import { makeShapeProxy, overlapShape, type Shape } from "./shape";
import * as tree from "./tree";
import { BodyType, ShapeType } from "./types";
import type { WorldState } from "./world";

/** A tracked overlap: the visitor shape's id and generation (b3Visitor). */
export type Visitor = { shapeId: number; generation: number };

/**
 * Per-sensor overlap state (b3Sensor). `overlaps2` is the current frame's overlaps (double-buffered
 * with `overlaps1`, the previous frame's, so begin/end events are a sorted diff); `hits` collects
 * time-of-impact detections from the continuous solver, appended into `overlaps2` each step.
 */
export type Sensor = {
    hits: Visitor[];
    overlaps1: Visitor[];
    overlaps2: Visitor[];
    shapeId: number;
};

/** A begin-touch event between a sensor and a visitor shape (b3SensorBeginTouchEvent). */
export type SensorBeginTouchEvent = { sensorShapeId: EntityId; visitorShapeId: EntityId };

const WORLD_ORIGIN = { x: 0, y: 0, z: 0 };

/** A fresh sensor bound to `shapeId` (b3CreateShape's sensor branch). */
export function createSensor(shapeId: number): Sensor {
    return { hits: [], overlaps1: [], overlaps2: [], shapeId };
}

/** A shape's public id from its slot index and generation (b3ShapeId). */
const shapeEntityId = (world: WorldState, index: number, generation: number): EntityId => ({
    index1: index + 1,
    world0: world.worldId,
    generation,
});

/**
 * Record a continuous (time-of-impact) sensor hit (b3Solve's sensor-hit report). Called from the
 * continuous solver after it has filtered hits against the body's final impact fraction; resolves the
 * visitor's generation and appends to the sensor's hit list, which the next overlap pass folds in.
 */
export function recordSensorHit(world: WorldState, sensorId: number, visitorId: number): void {
    const sensorShape = world.shapes[sensorId];
    const visitor = world.shapes[visitorId];
    const sensor = world.sensors[sensorShape.sensorIndex];
    sensor.hits.push({ shapeId: visitorId, generation: visitor.generation });
}

/**
 * True if a visitor shape overlaps a sensor shape (b3OverlapSensor). The visitor's convex proxy is
 * pulled into the sensor's local frame and tested against the sensor geometry at identity — the same
 * arithmetic the C runs, so the boolean matches at overlap boundaries.
 */
function overlapSensor(
    sensorShape: Shape,
    sensorTransform: Transform,
    visitorShape: Shape,
    visitorTransform: Transform,
): boolean {
    const proxy = makeShapeProxy(visitorShape);
    const relative = xf.invMul(sensorTransform, visitorTransform);

    const count = minInt(proxy.count, MAX_SHAPE_CAST_POINTS);
    const points: Vec3[] = new Array(count);
    for (let i = 0; i < count; ++i) {
        points[i] = xf.point(relative, proxy.points[i]);
    }
    return overlapShape(sensorShape, xf.identity(), { points, count, radius: proxy.radius });
}

/**
 * Refresh every sensor's overlaps and publish begin/end events (b3OverlapSensors + b3SensorTask,
 * merged for the serial path). Runs after the solver, so continuous hits are already recorded.
 */
export function overlapSensors(world: WorldState): void {
    const sensorCount = world.sensors.length;
    if (sensorCount === 0) {
        return;
    }

    // Re-derive the resident tree views if the solve's column reserve grew memory and detached them
    // before this post-solve query pass (O(1) when fresh).
    world.broadPhase.store.refreshIfStale();
    const trees = world.broadPhase.trees;

    for (let sensorIndex = 0; sensorIndex < sensorCount; ++sensorIndex) {
        const sensor = world.sensors[sensorIndex];
        const sensorShape = world.shapes[sensor.shapeId];

        // Swap overlap buffers, seed the new frame with this step's time-of-impact hits.
        sensor.overlaps1 = sensor.overlaps2;
        sensor.overlaps2 = sensor.hits;
        sensor.hits = [];
        const overlaps2 = sensor.overlaps2;

        const body = world.bodies[sensorShape.bodyId];
        const disabled =
            body.setIndex === SetType.Disabled || sensorShape.enableSensorEvents === false;

        if (disabled === false) {
            const transform = toRelativeTransform(getBodyTransformQuick(world, body), WORLD_ORIGIN);
            const bounds = sensorShape.aabb;
            const maskHi = sensorShape.filter.maskHi;
            const maskLo = sensorShape.filter.maskLo;

            const callback = (_proxyId: number, shapeId: number): boolean => {
                if (shapeId === sensorShape.id) {
                    return true;
                }
                const other = world.shapes[shapeId];

                // Mesh vs mesh (or height field) has no overlap test — skip if both are non-convex.
                const sensorNonConvex =
                    sensorShape.type === ShapeType.Mesh ||
                    sensorShape.type === ShapeType.HeightField;
                const otherNonConvex =
                    other.type === ShapeType.Mesh || other.type === ShapeType.HeightField;
                if (sensorNonConvex && otherNonConvex) {
                    return true;
                }

                if (other.enableSensorEvents === false) {
                    return true;
                }
                if (other.bodyId === sensorShape.bodyId) {
                    return true;
                }
                if (shouldShapesCollide(sensorShape.filter, other.filter) === false) {
                    return true;
                }
                // Custom user filtering lands with its own stage (no customFilterFcn yet).

                const otherTransform = toRelativeTransform(
                    getBodyTransformQuick(world, world.bodies[other.bodyId]),
                    WORLD_ORIGIN,
                );
                if (overlapSensor(sensorShape, transform, other, otherTransform) === false) {
                    return true;
                }

                overlaps2.push({ shapeId, generation: other.generation });
                return true;
            };

            tree.query(trees[BodyType.Static], bounds, maskHi, maskLo, false, callback);
            tree.query(trees[BodyType.Kinematic], bounds, maskHi, maskLo, false, callback);
            tree.query(trees[BodyType.Dynamic], bounds, maskHi, maskLo, false, callback);

            // Sort by shape id, then drop duplicates (a hit may repeat a queried overlap).
            overlaps2.sort((a, b) => a.shapeId - b.shapeId);
            let uniqueCount = 0;
            for (let i = 0; i < overlaps2.length; ++i) {
                if (
                    uniqueCount === 0 ||
                    overlaps2[i].shapeId !== overlaps2[uniqueCount - 1].shapeId
                ) {
                    overlaps2[uniqueCount] = overlaps2[i];
                    uniqueCount += 1;
                }
            }
            overlaps2.length = uniqueCount;
        }

        emitSensorEvents(world, sensorShape, sensor.overlaps1, overlaps2);
    }
}

/**
 * Publish begin/end events by walking the two sorted overlap lists in lock-step (b3OverlapSensors's
 * per-sensor diff): a shape present last frame but gone this frame ends; a new shape begins; a
 * matching shape whose generation changed ends the old and begins the new.
 */
function emitSensorEvents(
    world: WorldState,
    sensorShape: Shape,
    refs1: Visitor[],
    refs2: Visitor[],
): void {
    const sensorId = shapeEntityId(world, sensorShape.id, sensorShape.generation);
    const beginEvents = world.sensorBeginEvents;
    const endEvents = world.sensorEndEvents[world.endEventArrayIndex];

    const visitorId = (r: Visitor): EntityId => shapeEntityId(world, r.shapeId, r.generation);
    const begin = (r: Visitor): void => {
        beginEvents.push({ sensorShapeId: sensorId, visitorShapeId: visitorId(r) });
    };
    const end = (r: Visitor): void => {
        endEvents.push({ sensorShapeId: sensorId, visitorShapeId: visitorId(r) });
    };

    const count1 = refs1.length;
    const count2 = refs2.length;
    let index1 = 0;
    let index2 = 0;
    while (index1 < count1 && index2 < count2) {
        const r1 = refs1[index1];
        const r2 = refs2[index2];
        if (r1.shapeId === r2.shapeId) {
            if (r1.generation < r2.generation) {
                end(r1);
                index1 += 1;
            } else if (r1.generation > r2.generation) {
                begin(r2);
                index2 += 1;
            } else {
                index1 += 1;
                index2 += 1;
            }
        } else if (r1.shapeId < r2.shapeId) {
            end(r1);
            index1 += 1;
        } else {
            begin(r2);
            index2 += 1;
        }
    }
    while (index1 < count1) {
        end(refs1[index1]);
        index1 += 1;
    }
    while (index2 < count2) {
        begin(refs2[index2]);
        index2 += 1;
    }
}

/**
 * Destroy a sensor when its shape is destroyed (b3DestroySensor). Emits an end-touch event for every
 * current overlap, then swap-removes the sensor from the dense array and fixes up the moved sensor's
 * back-reference so `shape.sensorIndex` stays valid.
 */
export function destroySensor(world: WorldState, sensorShape: Shape): void {
    const sensorIndex = sensorShape.sensorIndex;
    const sensor = world.sensors[sensorIndex];
    const sensorId = shapeEntityId(world, sensorShape.id, sensorShape.generation);
    const endEvents = world.sensorEndEvents[world.endEventArrayIndex];
    for (const ref of sensor.overlaps2) {
        endEvents.push({
            sensorShapeId: sensorId,
            visitorShapeId: shapeEntityId(world, ref.shapeId, ref.generation),
        });
    }

    // Swap-remove from the dense sensor array; repoint the moved sensor's shape.
    const last = world.sensors.length - 1;
    if (sensorIndex !== last) {
        const moved = world.sensors[last];
        world.sensors[sensorIndex] = moved;
        world.shapes[moved.shapeId].sensorIndex = sensorIndex;
    }
    world.sensors.pop();
    sensorShape.sensorIndex = NULL_INDEX;
}
