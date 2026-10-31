import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "../../src/engine";
import { hashPhysics, PhysicsWorldDefinition, StandardPhysicsPlugin } from "../../src/standard/physics";
import { BodyType, init } from "../../src/standard/physics/api";
import { createSphereShape } from "../../src/standard/physics/shapes/shape";
import { bodyIsAwake, setMassData } from "../../src/standard/physics/world/body";
import { hashWorldState } from "../../src/standard/physics/world/hash";
import {
    worldEnableSleeping,
    worldSetGravity,
    worldSetIsotropicInertiaTolerance,
} from "../../src/standard/physics/world/world";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld, testShapeDef } from "../../src/standard/physics/testing";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(180_000);
await init(undefined, { threads: 0 });
await assertPublicOracleKernel();
const native = BigInt(nativeSseOutput("world-definition.c", "").trim());
const nativeGravity = BigInt(nativeSseOutput("world-gravity.c", "").trim());
const nativeSleeping = BigInt(nativeSseOutput("world-sleeping.c", "").trim());

test("a non-default isotropic inertia tolerance hashes equal native Box3D", () => {
    const world = createTestWorld({ isotropicInertiaTolerance: 0.9, gravity: { x: 0, y: 0, z: 0 } });
    try {
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            angularVelocity: { x: 1, y: -2, z: 3 },
        });
        setMassData(world, body, {
            mass: 1,
            center: { x: 0, y: 0, z: 0 },
            inertia: {
                cx: { x: 0.4, y: 0, z: 0 },
                cy: { x: 0, y: 0.5, z: 0 },
                cz: { x: 0, y: 0, z: 0.6 },
            },
        });
        for (let i = 0; i < 10; i++) stepTestWorld(world, 1 / 60, 4);
        expect(hashWorldState(world)).toBe(BigInt(nativeSseOutput("gyroscopic-tolerance.c", "").trim()));
        const hashes = [hashWorldState(world)];
        worldSetIsotropicInertiaTolerance(world, 0.1);
        for (let i = 0; i < 10; i++) stepTestWorld(world, 1 / 60, 4);
        hashes.push(hashWorldState(world));
        setMassData(world, body, {
            mass: 1,
            center: { x: 0, y: 0, z: 0 },
            inertia: {
                cx: { x: 0.4, y: 0, z: 0 },
                cy: { x: 0, y: 0.5, z: 0 },
                cz: { x: 0, y: 0, z: 0.6 },
            },
        });
        for (let i = 0; i < 10; i++) stepTestWorld(world, 1 / 60, 4);
        hashes.push(hashWorldState(world));
        expect(hashes).toEqual(nativeSseOutput("gyroscopic-tolerance.c", "1\n").trim().split("\n").map(BigInt));
    } finally {
        destroyTestWorld(world);
    }
});

test("default ECS world definition hashes equal native Box3D", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const definition = app.world.resource(PhysicsWorldDefinition);
        expect(definition.gravity).toEqual({ x: 0, y: -10, z: 0 });
        expect(definition.subStepCount).toBe(4);
        expect(hashPhysics(app.world)).toBe(native);
    } finally {
        app.dispose();
    }
});

test("a live gravity setter hashes equal to native Box3D", () => {
    const world = createTestWorld();
    try {
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 10, z: 0 },
        });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        stepTestWorld(world, 1 / 60, 4);
        worldSetGravity(world, { x: 0, y: -2, z: 0 });
        stepTestWorld(world, 1 / 60, 4);
        expect(hashWorldState(world)).toBe(nativeGravity);
    } finally {
        destroyTestWorld(world);
    }
});

test("disabling sleeping wakes existing bodies like native Box3D", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        for (let i = 0; i < 60; i++) stepTestWorld(world, 1 / 60, 4);
        expect(bodyIsAwake(world, body)).toBe(false);
        worldEnableSleeping(world, false);
        expect(bodyIsAwake(world, body)).toBe(true);
        expect(hashWorldState(world)).toBe(nativeSleeping);
    } finally {
        destroyTestWorld(world);
    }
});
