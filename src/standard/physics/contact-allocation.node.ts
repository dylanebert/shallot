import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";
import { countedKernelBytes } from "../../../scripts/counted-kernel";
import { BodyType } from "./common/types";
import { clockImport, type Kernel, kernelState } from "./kernel/kernel";
import { ShapeField, shapeField } from "./kernel/shaperecords";
import { makeBoxHull } from "./shapes/build-hull";
import { createHullShape } from "./shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "./testing";

setDefaultTimeout(TIMEOUT.node);
const entry = resolve(import.meta.dir, "fixtures/contact-allocation.entry.ts");
let kernelBytes: ArrayBuffer | undefined;
beforeAll(async () => {
    kernelBytes = await countedKernelBytes();
}, TIMEOUT.startup);

test("fixed-tick contact and sensor reads retain their records without steady JavaScript allocation", async () => {
    const sample = await sampleAllocation(entry, { warm: 6000, frames: 600 });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});

test("contact, sensor and child-event reads allocate no WASM heap, with an allocating control", async () => {
    const bytes = kernelBytes;
    if (!bytes) throw new Error("counting kernel was not built");
    const runtime = kernelState(undefined);
    const previous = runtime.instance;
    let world: ReturnType<typeof createTestWorld> | undefined;
    try {
        const built = await WebAssembly.instantiate(bytes, {
            env: {
                now: clockImport(() => k.memory),
                queryCallback() {
                    throw new Error("unexpected query callback");
                },
                materialCallback() {
                    throw new Error("unexpected material callback");
                },
                collisionCallback() {
                    throw new Error("unexpected collision callback");
                },
            },
        });
        const k = built.instance.exports as unknown as Kernel & {
            allocationCount(): number;
            allocationControl(): void;
        };
        runtime.instance = k;
        world = createTestWorld();
        const a = createTestBody(world, { type: BodyType.Static });
        const b = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 1, z: 0 },
        });
        const hull = makeBoxHull(0.5, 0.5, 0.5);
        const shapeA = createHullShape(world, a, testShapeDef(), hull)!;
        const shapeB = createHullShape(
            world,
            b,
            testShapeDef({ enableContactEvents: true, enableSensorEvents: true }),
            hull,
        )!;
        const sensorBody = createTestBody(world, { type: BodyType.Static });
        const sensor = createHullShape(
            world,
            sensorBody,
            testShapeDef({ isSensor: true, enableSensorEvents: true }),
            makeBoxHull(3, 3, 3),
        )!;
        stepTestWorld(world, 1 / 60);
        const sensorIndex = shapeField(world, sensor, ShapeField.sensorIndex);
        const read = () => {
            if (
                k.sensorVisitorCount(world!.worldId, sensorIndex) !== 1 ||
                k.sensorVisitorWord(world!.worldId, sensorIndex, 0, 0) !== shapeB
            )
                throw new Error("wrong allocation sensor visitor");
            if (!k.contactQuery(world!.worldId, shapeA, shapeB, 0))
                throw new Error("missing allocation contact");
            if (k.eventChildIndex(world!.worldId, 2, 0) !== 0)
                throw new Error("wrong allocation child");
        };
        for (let i = 0; i < 1200; i++) read();
        const before = k.allocationCount();
        for (let i = 0; i < 600; i++) read();
        expect(k.allocationCount() - before).toBe(0);
        k.allocationControl();
        expect(k.allocationCount() - before).toBeGreaterThan(0);
    } finally {
        if (world) destroyTestWorld(world);
        runtime.instance = previous;
    }
});

test("the fixed-tick contact allocation oracle rejects an allocating control", async () => {
    const sample = await sampleAllocation(entry, { warm: 1200, frames: 600, input: "allocating" });
    expect(sample.control.length).toBeGreaterThan(0);
    expect(allocationFailure(sample)).toContain("steady play allocated JavaScript heap");
});
