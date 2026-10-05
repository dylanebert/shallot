// Sensor overlap tracking (Box3D's sensor.c, Erin Catto, MIT). A sensor shape detects other shapes
// overlapping it and reports begin/end touch events, in deterministic order, without producing any
// contact response. Sensors never create contacts (the pair phase skips them, pairs.ts) and never
// perturb dynamics, so the world-state hash is unaffected; correctness is behavioral, not bit-exact.
//
// Single-threaded and serial: the C's per-worker sensor task + the event-publish pass collapse into
// one loop, and the eventBits optimization drops out (a sensor whose overlaps didn't change emits no
// events regardless, so the diff always runs). fround discipline (every f32 result rounds through Math.fround, keeping bit-exact f32 parity).

import { NULL_INDEX, qsort } from "../common/array";
import { SetType } from "../common/constants";
import type { EntityId } from "../common/ids";
import type { Kernel } from "../kernel/kernel";
import { type QueryColumns, queryColumns } from "../kernel/querycolumns";
import { readShapeAabb, SHAPE_STRIDE } from "../kernel/shapecolumns";

const sensorBounds = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };

import type { Shape } from "../shapes/shape";
import type { WorldState } from "./world";

/** A tracked overlap: the visitor shape's id and generation (b3Visitor). */
export type Visitor = { shapeId: number; generation: number };

/**
 * Per-sensor overlap state (b3Sensor). `overlaps2` is the current frame's overlaps (double-buffered
 * with `overlaps1`, the previous frame's, so begin/end events are a sorted diff); `hits` collects
 * time-of-impact detections from the continuous solver, appended into `overlaps2` each step.
 */
export type Sensor = {
    hits: VisitorArray;
    overlaps1: VisitorArray;
    overlaps2: VisitorArray;
    shapeId: number;
};

/** Visitors in the first `count` slots of retained records (b3Array(b3Visitor)); slots past it are spare. */
export type VisitorArray = { data: Visitor[]; count: number };

/** Append a visitor, reusing a spare record when one exists (b3Array_Emplace). */
function emplace(array: VisitorArray, shapeId: number, generation: number): void {
    if (array.count === array.data.length) array.data.push({ shapeId, generation });
    else {
        const r = array.data[array.count];
        r.shapeId = shapeId;
        r.generation = generation;
    }
    array.count += 1;
}

/** A begin-touch event between a sensor and a visitor shape (b3SensorBeginTouchEvent). */
export type SensorBeginTouchEvent = { sensorShapeId: EntityId; visitorShapeId: EntityId };

const WORLD_ORIGIN = { x: 0, y: 0, z: 0 };

/** A fresh sensor bound to `shapeId` (b3CreateShape's sensor branch). */
export function createSensor(shapeId: number): Sensor {
    return {
        hits: { data: [], count: 0 },
        overlaps1: { data: [], count: 0 },
        overlaps2: { data: [], count: 0 },
        shapeId,
    };
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
    emplace(sensor.hits, visitorId, visitor.generation);
}

// qsort's index callbacks over the array being sorted; swapping fields keeps every record distinct.
let sorting: Visitor[] = [];
const lessShapeId = (i: number, j: number): boolean => sorting[i].shapeId < sorting[j].shapeId;
const swapVisitors = (i: number, j: number): void => {
    const a = sorting[i];
    const b = sorting[j];
    const shapeId = a.shapeId;
    const generation = a.generation;
    a.shapeId = b.shapeId;
    a.generation = b.generation;
    b.shapeId = shapeId;
    b.generation = generation;
};

/**
 * Refresh every sensor's overlaps and publish begin/end events (b3OverlapSensors + b3SensorTask,
 * merged for the serial path). Runs after the solver, so continuous hits are already recorded.
 */
export function overlapSensors(world: WorldState): void {
    const sensorCount = world.sensors.length;
    if (sensorCount === 0) {
        return;
    }

    const q = queryColumns(world);
    const k = q.prepare(WORLD_ORIGIN);

    for (let sensorIndex = 0; sensorIndex < sensorCount; ++sensorIndex)
        refreshSensor(world, world.sensors[sensorIndex], q, k);
}

// Refresh one sensor's overlaps and publish its events (b3SensorTask, then its diff). Kept out of
// overlapSensors so a sensorless world's early return stays small enough to tier up during warm-up.
function refreshSensor(world: WorldState, sensor: Sensor, q: QueryColumns, k: Kernel): void {
    const sensorShape = world.shapes[sensor.shapeId];

    // Swap overlap buffers, seed the new frame with this step's time-of-impact hits. The retired
    // previous-frame buffer becomes the empty hit list; nothing outside this pass holds it.
    const retired = sensor.overlaps1;
    sensor.overlaps1 = sensor.overlaps2;
    sensor.overlaps2 = sensor.hits;
    retired.count = 0;
    sensor.hits = retired;
    const overlaps2 = sensor.overlaps2;

    const body = world.bodies[sensorShape.bodyId];
    const disabled = body.setIndex === SetType.Disabled || sensorShape.enableSensorEvents === false;

    if (disabled === false) {
        q.bounds(readShapeAabb(world, sensorShape.id, sensorBounds));
        let shapeId = k.sensorQuery(world.worldId, sensor.shapeId) >>> 0;
        while (shapeId !== 0xffffffff) {
            emplace(overlaps2, shapeId, world.shapes[shapeId].generation);
            shapeId = world.shapeStore.shapeU[shapeId * SHAPE_STRIDE + 33];
        }

        // Sort by shape id, then drop duplicates (a hit may repeat a queried overlap).
        sorting = overlaps2.data;
        qsort(overlaps2.count, lessShapeId, swapVisitors);
        const data = overlaps2.data;
        let uniqueCount = 0;
        for (let i = 0; i < overlaps2.count; ++i) {
            if (uniqueCount === 0 || data[i].shapeId !== data[uniqueCount - 1].shapeId) {
                data[uniqueCount].shapeId = data[i].shapeId;
                data[uniqueCount].generation = data[i].generation;
                uniqueCount += 1;
            }
        }
        overlaps2.count = uniqueCount;
    }

    emitSensorEvents(world, sensorShape, sensor.overlaps1, overlaps2);
}

/**
 * Publish begin/end events by walking the two sorted overlap lists in lock-step (b3OverlapSensors's
 * per-sensor diff): a shape present last frame but gone this frame ends; a new shape begins; a
 * matching shape whose generation changed ends the old and begins the new.
 */
function emitSensorEvents(
    world: WorldState,
    sensorShape: Shape,
    array1: VisitorArray,
    array2: VisitorArray,
): void {
    const refs1 = array1.data;
    const refs2 = array2.data;
    const begin = (r: Visitor): void => {
        world.sensorBeginEvents.push({
            sensorShapeId: shapeEntityId(world, sensorShape.id, sensorShape.generation),
            visitorShapeId: shapeEntityId(world, r.shapeId, r.generation),
        });
    };
    const end = (r: Visitor): void => {
        world.sensorEndEvents[world.endEventArrayIndex].push({
            sensorShapeId: shapeEntityId(world, sensorShape.id, sensorShape.generation),
            visitorShapeId: shapeEntityId(world, r.shapeId, r.generation),
        });
    };

    const count1 = array1.count;
    const count2 = array2.count;
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
    for (let i = 0; i < sensor.overlaps2.count; i++) {
        const ref = sensor.overlaps2.data[i];
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
