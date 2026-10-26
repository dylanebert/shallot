import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import {
    defaultExplosionDef,
    explode,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);

test("explosion defaults are independent Box3D definitions", () => {
    const a = defaultExplosionDef();
    expect(a).toEqual({
        maskBits: 0xffffffffffffffffn,
        position: { x: 0, y: 0, z: 0 },
        radius: 0,
        falloff: 0,
        impulsePerArea: 0,
    });
    a.position.x = 7;
    expect(defaultExplosionDef().position.x).toBe(0);
});

test("explode acts immediately on bound shapes and not bodies awaiting their first sync", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        },
    });
    try {
        const world = app.world;
        const ball = world.create();
        world.add(ball, Transform, { translation: [2, 0, 0] });
        world.add(ball, Body, { type: BodyType.Dynamic });
        world.add(ball, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
        const def = { ...defaultExplosionDef(), radius: 4, impulsePerArea: 1000 };
        explode(world, def);
        world.tick();
        expect(world.get(ball, GlobalTransform)!.linearVelocity[0]).toBe(0);
        explode(world, { ...def, maskBits: 0n });
        world.tick();
        expect(world.get(ball, GlobalTransform)!.linearVelocity[0]).toBe(0);
        explode(world, def);
        // The kernel changes immediately; placement publication remains on the next fixed step.
        expect(world.get(ball, GlobalTransform)!.linearVelocity[0]).toBe(0);
        world.tick();
        expect(world.get(ball, GlobalTransform)!.linearVelocity[0]).toBeGreaterThan(0);
    } finally {
        app.dispose();
    }
});
