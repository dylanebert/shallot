import { expect, test } from "bun:test";
import * as physics from "@dylanebert/shallot/physics";
import * as standard from "@dylanebert/shallot/standard/physics";

test("core physics exports shared authoring data without engine aliases or unused observation helpers", () => {
    expect(Object.keys(physics).sort()).toEqual([
        "Body",
        "Hulls",
        "Joint",
        "PhysicsPlugin",
        "ShapeKind",
        "Spring",
        "UNIT_CUBE_ID",
    ]);
});

test("standard physics exports consumed world operations and profiling without solver internals", () => {
    expect(Object.keys(standard).sort()).toEqual([
        "CLOCK_SLOTS",
        "PhysicsWorld",
        "StandardPhysicsPlugin",
        "StepPhysicsSystem",
        "hashPhysics",
        "physicsWorld",
        "readBody",
        "restorePhysics",
        "setKinematic",
        "setVelocity",
        "snapshotPhysics",
        "zeroStepProfile",
    ]);
});
