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

// Type-only evidence: this file stops compiling if the public subpath drops either pool type.
export type PublicPoolTypes = [Pool["size"], WorkerReady["index"]];

check(
    "physics: State-scoped physics seams are public",
    {
        claim: "the wheel, parallel, hinge, cone/twist, soft-anchor, contact-event and joint-event seams are absent from the published physics subpath",
    },
    () => {
        expect(World).toBeFunction();
        expect(typeof BodyType.Dynamic).toBe("number");
        expect(typeof JointType.Wheel).toBe("number");
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
    },
);
