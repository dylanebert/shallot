import { readContactManifolds } from "../collision/manifoldstore";
// Active scene parity runs the immutable official command/bundle corpus. The historical 53 scene
// fixtures remain under fixtures/ as migration evidence, but their predecessor hashes are not the
// current authority and are intentionally not asserted here.

import { afterAll, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { PhysicsWorld } from "../api/world";
import { shutdown, threads, workers } from "../kernel/kernel";

const pooled = process.env.PHYSICS_CORPUS_POOL === "1";
if (pooled) {
    if (threads(undefined) !== 4 || workers(undefined) === null) {
        throw new Error("the corpus requires a live four-thread pool");
    }
}

import { BodyType } from "../common/types";
import { loadConsumerCorpus, runCommonInput } from "../oracle/consumer";
import { loadScenarioCorpus, runScenario } from "../oracle/scenario";
import { compareCase } from "../oracle/strict";
import { makeBoxHull } from "../shapes/hull";

test("sensor tasks keep query scratch independent for sensors sharing every visitor", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableContinuous: false });
    try {
        const sensorBody = world.createBody({ type: BodyType.Static });
        const sensors = Array.from({ length: 64 }, () =>
            sensorBody.createSphere(
                { isSensor: true, enableSensorEvents: true },
                { center: { x: 0, y: 0, z: 0 }, radius: 1 },
            ),
        );
        const visitorBody = world.createBody({ type: BodyType.Kinematic });
        const visitors = Array.from({ length: 16 }, () =>
            visitorBody.createSphere(
                { enableSensorEvents: true },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.1 },
            ),
        );
        world.step(1 / 60, 1);
        const ids = visitors.map((shape) => shape.id.index1);
        for (const sensor of sensors)
            expect(sensor.getSensorOverlaps().map((shape) => shape.id.index1)).toEqual(ids);
        expect(world.getSensorEvents().beginEvents.length).toBe(64 * 16);
        world.step(1 / 60, 1);
        expect(world.getSensorEvents().beginEvents.length).toBe(0);
        for (const sensor of sensors)
            expect(sensor.getSensorOverlaps().map((shape) => shape.id.index1)).toEqual(ids);
    } finally {
        world.destroy();
    }
});

test("the active collision route changes the symmetric face-B feature order, the pinned CCD and sensor intermediate bits, or the public body move record identity", () => {
    const physicsWorld = new PhysicsWorld({
        gravity: { x: 0, y: -10, z: 0 },
        enableContinuous: false,
    });
    const body = physicsWorld.createBody({
        type: BodyType.Dynamic,
        position: { x: 0, y: 1, z: 0 },
    });
    body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    body.applyMassFromShapes();
    physicsWorld.step(1 / 60, 1);
    const moves = physicsWorld.getBodyEvents();
    expect(moves.count).toBeGreaterThan(0);
    const event = moves.moveEvents[0];
    expect(event.body.id.index1).toBe(body.id.index1);
    expect(event.body.id.generation).toBe(body.id.generation);
    expect(event.fellAsleep).toBe(false);

    const { corpus, digest } = loadScenarioCorpus();
    const ccd = corpus.scenarios.find((scenario) => scenario.name === "ccd-bullet");
    const sensor = corpus.scenarios.find((scenario) => scenario.name === "sensor");
    if (!ccd || !sensor) throw new Error("required CCD/sensor scenarios are missing");
    const bits = (value: number): string => {
        const view = new DataView(new ArrayBuffer(4));
        view.setFloat32(0, value);
        return view.getUint32(0).toString(16).padStart(8, "0");
    };
    const features = [0x0112010b, 0x01070112, 0x01100107, 0x010b0110];
    let ccdChecked = false;
    const ccdResult = runScenario(ccd, digest, false, (physicsWorld, step) => {
        if (step !== 0) return;
        const manifold = readContactManifolds(physicsWorld.state, 0)[0];
        if (!manifold) throw new Error("CCD step 0 has no contact manifold");
        expect(manifold.points.slice(0, 4).map((point) => point.featureId)).toEqual(features);
        expect(bits(manifold.points[0].anchorA.x)).toBe("3d4ccc00");
        ccdChecked = true;
    });
    expect(ccdChecked).toBe(true);
    expect(ccdResult.hashes.find((hash) => hash.step === 0)?.value).toBe("0xd4475ed2139c0e45");

    let sensorChecked = false;
    const sensorResult = runScenario(sensor, digest, false, (physicsWorld, step) => {
        if (step !== 14) return;
        const manifold = readContactManifolds(physicsWorld.state, 0)[0];
        if (!manifold) throw new Error("sensor step 14 has no face-B contact manifold");
        expect(bits(manifold.normal.x)).toBe("00000000");
        expect(bits(manifold.normal.y)).toBe("3f800000");
        expect(bits(manifold.normal.z)).toBe("00000000");
        expect(manifold.points.slice(0, 4).map((point) => point.featureId)).toEqual(features);
        expect(bits(manifold.points[0].separation)).toBe("3ba3d800");
        sensorChecked = true;
    });
    expect(sensorChecked).toBe(true);
    expect(sensorResult.hashes.find((hash) => hash.step === 14)?.value).toBe("0x31240e262d32461b");
});

test("a Shallot physics result diverges from the Box3D reference on any case of the immutable v6 corpus, including the official scenes", () => {
    const corpus = loadConsumerCorpus();
    for (const item of corpus.cases) {
        const result = compareCase(item, runCommonInput(item));
        if (result.status !== "pass")
            throw new Error(`${item.id}: ${result.firstDifference?.path ?? "mismatch"}`);
    }
    console.log(JSON.stringify({ corpus: "immutable box3d v6", cases: corpus.cases.length }));
});

test("a weld reports a biased-pass threshold crossing even when relaxation removes the force", () => {
    const world = new PhysicsWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    try {
        const anchor = world.createBody({});
        const box = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 1, z: 0 } });
        box.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        const joint = world.createWeldJoint(anchor, box, { forceThreshold: 100 });
        world.step(1 / 60, 4);
        expect(world.getJointEvents().length).toBe(1);
        const force = joint.getConstraintForce();
        expect(Math.hypot(force.x, force.y, force.z)).toBeLessThan(100);
    } finally {
        world.destroy();
    }
});

if (pooled) {
    afterAll(() => shutdown(undefined));
} else {
    test("the Box3D corpus passes on a live pool in its own process", async () => {
        const child = Bun.spawn(
            [
                process.execPath,
                "test",
                "--preload",
                `${import.meta.dir}/step.pool.ts`,
                import.meta.filename,
            ],
            {
                env: { ...process.env, PHYSICS_CORPUS_POOL: "1" },
                stdout: "inherit",
                stderr: "inherit",
            },
        );
        expect(await child.exited).toBe(0);
    });
}
