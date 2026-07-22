// Stage 13 (sensors) tests. Sensors detect overlaps and report begin/end touch events without any
// contact response, so — like queries — they never feed the world-state hash and the contract is
// behavioral, not bit-exact. The overlap geometry reuses the batch-1 overlap primitives (gold-tested
// already), so these pin the sensor-specific logic: the sorted begin/end diff, the double buffer,
// the enable/filter/same-body gates, destruction fixup, continuous (time-of-impact) hits, and — the
// one place sensors touch the stepped world — that a sensor never perturbs the bodies around it.
// There is no upstream test_sensor.c; the determinism suite covers sensors upstream, out of v1 scope.

import { describe, expect, test } from "bun:test";
import { BodyType, World } from "./index";

const SENSOR = { isSensor: true, enableSensorEvents: true };
const VISIBLE = { enableSensorEvents: true };
const sphere = (radius: number) => ({ center: { x: 0, y: 0, z: 0 }, radius });

/** Step `n` times, accumulating every begin/end event across all steps. */
function run(world: World, n: number, dt = 1 / 60) {
    const begins: { sensor: number; visitor: number }[] = [];
    const ends: { sensor: number; visitor: number }[] = [];
    for (let i = 0; i < n; ++i) {
        world.step(dt);
        const ev = world.getSensorEvents();
        for (const e of ev.beginEvents) {
            begins.push({ sensor: e.sensor.id.index1, visitor: e.visitor.id.index1 });
        }
        for (const e of ev.endEvents) {
            ends.push({ sensor: e.sensor.id.index1, visitor: e.visitor.id.index1 });
        }
    }
    return { begins, ends };
}

describe("sensor overlap events", () => {
    test("a visitor passing through fires one begin then one end", () => {
        const world = new World();
        const sb = world.createBody();
        sb.createSphere(SENSOR, sphere(1));
        const vb = world.createBody({
            type: BodyType.Kinematic,
            position: { x: -3, y: 0, z: 0 },
            linearVelocity: { x: 3, y: 0, z: 0 },
        });
        vb.createSphere(VISIBLE, sphere(0.5));

        // -3 → +3 over 120 steps: enters near x=-1.5, exits near x=+1.5, exactly once each.
        const { begins, ends } = run(world, 120);
        expect(begins.length).toBe(1);
        expect(ends.length).toBe(1);
        expect(begins[0].sensor).toBe(ends[0].sensor);
        expect(begins[0].visitor).toBe(ends[0].visitor);

        world.destroy();
    });

    test("a resident visitor fires begin once and never repeats", () => {
        const world = new World();
        const sb = world.createBody();
        sb.createSphere(SENSOR, sphere(1));
        const vb = world.createBody({ type: BodyType.Kinematic, position: { x: 0, y: 0, z: 0 } });
        vb.createSphere(VISIBLE, sphere(0.25));

        const { begins, ends } = run(world, 10);
        expect(begins.length).toBe(1);
        expect(ends.length).toBe(0);

        world.destroy();
    });

    test("no events when nothing overlaps", () => {
        const world = new World();
        const sb = world.createBody();
        sb.createSphere(SENSOR, sphere(1));
        const vb = world.createBody({ type: BodyType.Kinematic, position: { x: 5, y: 0, z: 0 } });
        vb.createSphere(VISIBLE, sphere(0.25));

        const { begins, ends } = run(world, 10);
        expect(begins.length).toBe(0);
        expect(ends.length).toBe(0);

        world.destroy();
    });

    test("getSensorOverlaps reflects the current overlaps", () => {
        const world = new World();
        const sb = world.createBody();
        const sensor = sb.createSphere(SENSOR, sphere(1));
        const vb = world.createBody({ type: BodyType.Kinematic, position: { x: 0, y: 0, z: 0 } });
        const v = vb.createSphere(VISIBLE, sphere(0.25));

        world.step(1 / 60);
        const overlaps = sensor.getSensorOverlaps();
        expect(overlaps.length).toBe(1);
        expect(overlaps[0].id.index1).toBe(v.id.index1);

        world.destroy();
    });
});

describe("sensor gating", () => {
    test("a visitor without sensor events enabled is not detected", () => {
        const world = new World();
        const sb = world.createBody();
        const sensor = sb.createSphere(SENSOR, sphere(1));
        const vb = world.createBody({ type: BodyType.Kinematic, position: { x: 0, y: 0, z: 0 } });
        vb.createSphere({}, sphere(0.25)); // enableSensorEvents defaults to false

        const { begins } = run(world, 5);
        expect(begins.length).toBe(0);
        expect(sensor.getSensorOverlaps().length).toBe(0);

        world.destroy();
    });

    test("a sensor with events disabled detects nothing", () => {
        const world = new World();
        const sb = world.createBody();
        sb.createSphere({ isSensor: true, enableSensorEvents: false }, sphere(1));
        const vb = world.createBody({ type: BodyType.Kinematic, position: { x: 0, y: 0, z: 0 } });
        vb.createSphere(VISIBLE, sphere(0.25));

        const { begins, ends } = run(world, 5);
        expect(begins.length).toBe(0);
        expect(ends.length).toBe(0);

        world.destroy();
    });

    test("shapes on the sensor's own body are not detected", () => {
        const world = new World();
        const sb = world.createBody();
        sb.createSphere(SENSOR, sphere(1));
        sb.createSphere(VISIBLE, sphere(0.25)); // same body, overlaps trivially

        const { begins } = run(world, 5);
        expect(begins.length).toBe(0);

        world.destroy();
    });

    test("filtered-out shapes are not detected", () => {
        const world = new World();
        const sb = world.createBody();
        // Sensor in category 1, masks only category 2.
        sb.createSphere(
            { ...SENSOR, filter: { categoryBits: 0b01n, maskBits: 0b10n, groupIndex: 0 } },
            sphere(1),
        );
        const vb = world.createBody({ type: BodyType.Kinematic, position: { x: 0, y: 0, z: 0 } });
        // Visitor in category 1 (not 2) → filtered out.
        vb.createSphere(
            { ...VISIBLE, filter: { categoryBits: 0b01n, maskBits: 0b01n, groupIndex: 0 } },
            sphere(0.25),
        );

        const { begins } = run(world, 5);
        expect(begins.length).toBe(0);

        world.destroy();
    });

    test("two visitors are both detected, sorted by shape id", () => {
        const world = new World();
        const sb = world.createBody();
        const sensor = sb.createSphere(SENSOR, sphere(2));
        const a = world.createBody({ type: BodyType.Kinematic, position: { x: -0.5, y: 0, z: 0 } });
        const av = a.createSphere(VISIBLE, sphere(0.25));
        const b = world.createBody({ type: BodyType.Kinematic, position: { x: 0.5, y: 0, z: 0 } });
        const bv = b.createSphere(VISIBLE, sphere(0.25));

        const { begins } = run(world, 1);
        expect(begins.length).toBe(2);
        const overlaps = sensor.getSensorOverlaps().map((s) => s.id.index1);
        expect(overlaps).toEqual([av.id.index1, bv.id.index1].sort((x, y) => x - y));

        world.destroy();
    });
});

describe("sensor destruction", () => {
    test("destroying a sensor shape emits end events for its overlaps and fixes up the moved sensor", () => {
        const world = new World();
        // Sensor A (dense index 0) with a resident visitor.
        const sa = world.createBody();
        const sensorA = sa.createSphere(SENSOR, sphere(1));
        const va = world.createBody({ type: BodyType.Kinematic, position: { x: 0, y: 0, z: 0 } });
        const vaShape = va.createSphere(VISIBLE, sphere(0.25));
        // Sensor B (dense index 1) with its own resident visitor, elsewhere.
        const sbBody = world.createBody({ position: { x: 10, y: 0, z: 0 } });
        const sensorB = sbBody.createSphere(SENSOR, sphere(1));
        const vbBody = world.createBody({
            type: BodyType.Kinematic,
            position: { x: 10, y: 0, z: 0 },
        });
        const vb = vbBody.createSphere(VISIBLE, sphere(0.25));

        run(world, 2); // both sensors now hold their overlaps

        // Destroy A. It pushes an end event for va into the current end-event buffer; the swap-remove
        // moves B into A's slot and repoints B's sensorIndex. The event surfaces after the next step
        // (the swap makes that buffer the one getSensorEvents reads).
        sensorA.destroy();
        world.step(1 / 60);
        const ev = world.getSensorEvents();
        expect(ev.endEvents.some((e) => e.visitor.id.index1 === vaShape.id.index1)).toBe(true);

        // B survived the swap-remove fixup and still detects its visitor.
        expect(sensorB.isValid()).toBe(true);
        expect(sensorB.getSensorOverlaps().map((s) => s.id.index1)).toEqual([vb.id.index1]);

        world.destroy();
    });
});

describe("sensor is inert to dynamics", () => {
    test("a dynamic body falls through a sensor identically to no sensor", () => {
        const fall = (withSensor: boolean): number => {
            const world = new World();
            const floor = world.createBody({ position: { x: 0, y: -3, z: 0 } });
            floor.createSphere({}, sphere(1));
            if (withSensor) {
                const s = world.createBody({ position: { x: 0, y: 0, z: 0 } });
                s.createSphere(SENSOR, sphere(1.5));
            }
            const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 3, z: 0 } });
            b.createSphere(VISIBLE, sphere(0.5));
            for (let i = 0; i < 120; ++i) {
                world.step(1 / 60);
            }
            const y = b.getPosition().y;
            world.destroy();
            return y;
        };
        // Bit-identical: the sensor must not alter the falling body's trajectory at all.
        expect(fall(true)).toBe(fall(false));
    });
});

describe("continuous sensor hits", () => {
    test("a fast body tunnelling through a static sensor still fires a begin event", () => {
        const world = new World({ gravity: { x: 0, y: 0, z: 0 } });
        const sb = world.createBody();
        sb.createSphere(SENSOR, sphere(0.5));
        // Small, fast body: in one 1/60 step it moves 2 units from x=-1 to x=+1, fully past the sensor,
        // so the discrete post-step overlap misses it — only the time-of-impact hit detects it.
        const vb = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -1, y: 0, z: 0 },
            linearVelocity: { x: 120, y: 0, z: 0 },
        });
        vb.createSphere(VISIBLE, sphere(0.1));

        world.step(1 / 60);
        // Ended past the sensor (center 0, reach 0.6) with no discrete overlap → only TOI could detect.
        expect(vb.getPosition().x).toBeGreaterThan(0.6);
        const ev = world.getSensorEvents();
        expect(ev.beginEvents.length).toBe(1);

        world.destroy();
    });
});
