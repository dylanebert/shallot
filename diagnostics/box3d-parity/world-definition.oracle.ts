import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "../../src/engine";
import { hashPhysics, PhysicsWorldDefinition, StandardPhysicsPlugin } from "../../src/standard/physics";
import { BodyType, hash, init, PhysicsWorld } from "../../src/standard/physics/api";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(180_000);
await init(undefined, { threads: 0 });
await assertPublicOracleKernel();
const native = BigInt(nativeSseOutput("world-definition.c", "").trim());
const nativeGravity = BigInt(nativeSseOutput("world-gravity.c", "").trim());
const nativeSleeping = BigInt(nativeSseOutput("world-sleeping.c", "").trim());

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
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 10, z: 0 },
        });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        world.step(1 / 60, 4);
        world.setGravity({ x: 0, y: -2, z: 0 });
        world.step(1 / 60, 4);
        expect(hash(world)).toBe(nativeGravity);
    } finally {
        world.destroy();
    }
});

test("disabling sleeping wakes existing bodies like native Box3D", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        for (let i = 0; i < 60; i++) world.step(1 / 60, 4);
        expect(body.isAwake()).toBe(false);
        world.enableSleeping(false);
        expect(body.isAwake()).toBe(true);
        expect(hash(world)).toBe(nativeSleeping);
    } finally {
        world.destroy();
    }
});
