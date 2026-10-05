import { expect, test } from "bun:test";
import * as physics from "@dylanebert/shallot/physics";
import * as standard from "@dylanebert/shallot/standard/physics";

test("core physics exports shared authoring data without engine aliases or analytic shape helpers", () => {
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
        "Character",
        "CharacterPlugin",
        "GroundState",
        "PhysicsWorld",
        "StandardPhysicsPlugin",
        "StepPhysicsSystem",
        "clipVector",
        "hashPhysics",
        "physicsWorld",
        "readBody",
        "restorePhysics",
        "setKinematic",
        "setVelocity",
        "snapshotPhysics",
        "solvePlanes",
    ]);
});
