import { bunChild } from "../../../../scripts/bun-child";
import { readContactManifolds } from "../collision/manifoldstore";
// Active scene parity runs the immutable official command/bundle corpus. The historical 53 scene
// fixtures remain under fixtures/ as migration evidence, but their predecessor hashes are not the
// current authority and are intentionally not asserted here.

import { afterAll, expect, setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { splitAllocationSubject } from "../joint-allocation.fixture";
import { EventKind, eventCount, eventId } from "../kernel/eventbuffers";
import { kernel, shutdown, threads, workers } from "../kernel/kernel";
import { createHullShape, createSphereShape, getSensorData } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { makeBodyId } from "../world/body";
import { worldCounters, worldProfile } from "../world/world";
import { defaultJointDef, getJointConstraintForce } from "./joint";
import { createWeldJoint, defaultWeldJointDef } from "./weldJoint";

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

test("a split candidate completes before finalize on the serial and live-pool step", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        const split = splitAllocationSubject(world);
        for (let i = 0; i < 20; ++i) split();
        expect(worldCounters(world).islandCount).toBe(1);
        expect(worldProfile(world).sleepIslands).toBe(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("sensor tasks keep query scratch independent for sensors sharing every visitor", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableContinuous: false });
    try {
        const sensorBody = createTestBody(world, { type: BodyType.Static });
        const sensorDef = testShapeDef({ isSensor: true, enableSensorEvents: true });
        const sensors = Array.from(
            { length: 64 },
            () =>
                createSphereShape(world, sensorBody, sensorDef, {
                    center: { x: 0, y: 0, z: 0 },
                    radius: 1,
                })!,
        );
        const visitorBody = createTestBody(world, { type: BodyType.Kinematic });
        const visitorDef = testShapeDef({ enableSensorEvents: true });
        const visitors = Array.from(
            { length: 16 },
            () =>
                createSphereShape(world, visitorBody, visitorDef, {
                    center: { x: 0, y: 0, z: 0 },
                    radius: 0.1,
                })!,
        );
        stepTestWorld(world, 1 / 60, 1);
        const ids = visitors.map((shape) => shape + 1);
        for (const sensor of sensors)
            expect(getSensorData(world, sensor).map((shape) => shape.shapeId + 1)).toEqual(ids);
        expect(eventCount(world, EventKind.SensorBegin)).toBe(64 * 16);
        stepTestWorld(world, 1 / 60, 1);
        expect(eventCount(world, EventKind.SensorBegin)).toBe(0);
        for (const sensor of sensors)
            expect(getSensorData(world, sensor).map((shape) => shape.shapeId + 1)).toEqual(ids);
    } finally {
        destroyTestWorld(world);
    }
});

test("the active collision route changes the symmetric face-B feature order, the pinned CCD and sensor intermediate bits, or the public body move record identity", () => {
    const physicsWorld = createTestWorld({
        gravity: { x: 0, y: -10, z: 0 },
        enableContinuous: false,
    });
    try {
        const body = createTestBody(physicsWorld, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 1, z: 0 },
        });
        createSphereShape(physicsWorld, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        stepTestWorld(physicsWorld, 1 / 60, 1);
        const moves = eventCount(physicsWorld, EventKind.BodyMove);
        expect(moves).toBeGreaterThan(0);
        const eventBody = eventId(physicsWorld, EventKind.BodyMove, 0, 8);
        const expectedBody = makeBodyId(physicsWorld, body);
        expect(eventBody.index1).toBe(expectedBody.index1);
        expect(eventBody.generation).toBe(expectedBody.generation);
        expect(
            kernel(physicsWorld.ecsState).eventWord(
                physicsWorld.worldId,
                EventKind.BodyMove,
                0,
                10,
            ),
        ).toBe(0);

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
            const manifold = readContactManifolds(physicsWorld, 0)[0];
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
            const manifold = readContactManifolds(physicsWorld, 0)[0];
            if (!manifold) throw new Error("sensor step 14 has no face-B contact manifold");
            expect(bits(manifold.normal.x)).toBe("00000000");
            expect(bits(manifold.normal.y)).toBe("3f800000");
            expect(bits(manifold.normal.z)).toBe("00000000");
            expect(manifold.points.slice(0, 4).map((point) => point.featureId)).toEqual(features);
            expect(bits(manifold.points[0].separation)).toBe("3ba3d800");
            sensorChecked = true;
        });
        expect(sensorChecked).toBe(true);
        expect(sensorResult.hashes.find((hash) => hash.step === 14)?.value).toBe(
            "0x31240e262d32461b",
        );
    } finally {
        destroyTestWorld(physicsWorld);
    }
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
    const world = createTestWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    try {
        const anchor = createTestBody(world);
        const box = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 1, z: 0 },
        });
        createHullShape(world, box, testShapeDef(), makeBoxHull(0.25, 0.25, 0.25));
        const base = { ...defaultJointDef(), bodyIdA: anchor, bodyIdB: box, forceThreshold: 100 };
        const joint = createWeldJoint(world, { ...defaultWeldJointDef(base), base }).joint;
        stepTestWorld(world, 1 / 60, 4);
        expect(eventCount(world, EventKind.Joint)).toBe(1);
        const force = getJointConstraintForce(world, joint);
        expect(Math.hypot(force.x, force.y, force.z)).toBeLessThan(100);
    } finally {
        destroyTestWorld(world);
    }
});

if (pooled) {
    afterAll(() => shutdown(undefined));
} else {
    test("the Box3D corpus passes on a live pool in its own process", async () => {
        const child = Bun.spawn({
            ...bunChild([
                "test",
                "--preload",
                `${import.meta.dir}/step.pool.ts`,
                import.meta.filename,
            ]),
            env: { ...process.env, PHYSICS_CORPUS_POOL: "1" },
            stdout: "inherit",
            stderr: "inherit",
        });
        expect(await child.exited).toBe(0);
    });
}
