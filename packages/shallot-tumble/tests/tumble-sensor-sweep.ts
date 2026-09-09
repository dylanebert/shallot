// The tumble.js `SensorSweep` sample (`samples/src/samples/events.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A kinematic sensor box sweeps vertically through a stack of five resting boxes
// via a sine-driven `setTargetTransform` in `update()` — a world-mutating drive, load-bearing for the gold.
// The sample's `render()` overlay ({@link renderSensorSweep}) outlines every box currently inside the sensor
// and tallies the begin/end events — both read-only queries, outside the gold contract.
//
// Creation order is load-bearing for the hash: ground, the five stacked boxes bottom to top, then the
// sensor — the sample's exact order.

import {
    type Body,
    BodyType,
    makeBoxHull,
    type Shape,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleUpdate } from "./tumble-oracle";
import type { Overlay, SampleRender } from "./tumble-overlay";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

let sensorBody: Body | null = null;
let sensor: Shape | null = null;
// the begin/end event tally, accumulated in the render (a read-only event read, never touching the world so
// the gold stays untouched — the sample counts in update(), but here update() is the oracle path).
let begins = 0;
let ends = 0;

/**
 * Author the Sensor Sweep scene into `world`: a stack of five density-1 boxes, and a kinematic sensor box
 * (`isSensor`, `enableSensorEvents`) starting at y=1.
 */
export function buildSensorSweep(world: World): void {
    sensorBody = null;
    sensor = null;
    begins = 0;
    ends = 0;
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const box = makeBoxHull(0.6, 0.4, 0.6);
    for (let i = 0; i < 5; ++i) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 0.4 + 0.85 * i, z: 0 },
        });
        body.createHull({ density: 1, enableSensorEvents: true }, box);
    }

    sensorBody = world.createBody({
        type: BodyType.Kinematic,
        position: { x: 0, y: 1, z: 0 },
    });
    sensor = sensorBody.createHull(
        { isSensor: true, enableSensorEvents: true },
        makeBoxHull(1.2, 0.5, 1.2),
    );
}

/**
 * Sweep the sensor vertically through the stack (the sample's `update()`): a sine-driven
 * `setTargetTransform` toward the swept y.
 */
export const updateSensorSweep: SampleUpdate = (
    _world: World,
    _params,
    dt: number,
    stepCount: number,
) => {
    if (sensorBody === null) return;
    const y = 3 + 2.6 * Math.sin(1.2 * stepCount * dt);
    sensorBody.setTargetTransform({ p: { x: 0, y, z: 0 }, q: IDENT }, dt, true);
};

/**
 * Outline every box currently inside the sensor and tally the begin/end events (the sample's `render()`).
 * The tally accumulates here (the render path) so the oracle-run `update()` stays a pure world drive.
 */
export const renderSensorSweep: SampleRender = (draw: Overlay, world: World) => {
    if (sensor === null) return;
    const events = world.getSensorEvents();
    begins += events.beginEvents.length;
    ends += events.endEvents.length;

    for (const shape of sensor.getSensorOverlaps()) {
        const p = shape.getBody().getPosition();
        draw.aabb(
            { x: p.x - 0.65, y: p.y - 0.45, z: p.z - 0.65 },
            { x: p.x + 0.65, y: p.y + 0.45, z: p.z + 0.65 },
            0xffd040,
        );
    }
    draw.text(`begin ${begins}   end ${ends}`);
};
