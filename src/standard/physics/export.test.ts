import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import {
    BodyType,
    createParallelJoint,
    createPool,
    createRevoluteJoint,
    createSoftJoint,
    createSphericalJoint,
    createWheelJoint,
    getContactEvents,
    getJointEvents,
    JointType,
    maxWorkers,
    type Pool,
    type WorkerReady,
    World,
} from "@dylanebert/shallot/physics";

function poolSize(pool: Pool): number {
    return pool.size;
}

function workerIndex(worker: WorkerReady): number {
    return worker.index;
}

check(
    "physics: State-scoped physics seams are public",
    {
        claim: "the wheel, parallel, hinge, cone/twist, soft-anchor, contact-event and joint-event seams are absent from the published physics subpath",
    },
    () => {
        expect(BodyType).toBeDefined();
        expect(JointType).toBeDefined();
        expect(World).toBeDefined();
        expect(createWheelJoint).toBeFunction();
        expect(createParallelJoint).toBeFunction();
        expect(createRevoluteJoint).toBeFunction();
        expect(createSphericalJoint).toBeFunction();
        expect(createSoftJoint).toBeFunction();
        expect(getContactEvents).toBeFunction();
        expect(getJointEvents).toBeFunction();
    },
);

check(
    "physics: pool helpers are public",
    {
        claim: "the physics pool helpers and worker-ready types compile through the public physics export",
    },
    () => {
        expect(typeof createPool).toBe("function");
        expect(typeof maxWorkers).toBe("function");
        expect(poolSize).toBeDefined();
        expect(workerIndex).toBeDefined();
    },
);
