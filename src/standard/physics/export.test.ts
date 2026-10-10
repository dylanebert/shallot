import { expect, test } from "bun:test";
import * as physics from "@dylanebert/shallot/physics";
import * as standard from "@dylanebert/shallot/standard/physics";

test("core physics exports shared authoring data without engine aliases or analytic shape helpers", () => {
    expect(Object.keys(physics).sort()).toEqual([
        "Body",
        "BodyMotionLock",
        "BodyType",
        "Compounds",
        "DistanceJoint",
        "FilterJoint",
        "HeightFields",
        "Hulls",
        "MotorJoint",
        "ParallelJoint",
        "PhysicsMeshes",
        "PhysicsPlugin",
        "PrismaticJoint",
        "RevoluteJoint",
        "Shape",
        "ShapeKind",
        "ShapeMaterials",
        "SphericalJoint",
        "UNIT_CUBE_ID",
        "WeldJoint",
        "WheelJoint",
    ]);
});

test("standard physics exports consumed world operations and profiling without solver internals", () => {
    expect(Object.keys(standard).sort()).toEqual([
        "Character",
        "CharacterPlugin",
        "GroundState",
        "PhysicsWorld",
        "PhysicsWorldDefinition",
        "StandardPhysicsPlugin",
        "StepPhysicsSystem",
        "applyAngularImpulse",
        "applyForce",
        "applyForceToCenter",
        "applyLinearImpulse",
        "applyLinearImpulseToCenter",
        "applyTorque",
        "clipVector",
        "hashPhysics",
        "physicsWorld",
        "setAngularVelocity",
        "setAwake",
        "setLinearVelocity",
        "setTargetTransform",
        "setTransform",
        "solvePlanes",
    ]);
});
