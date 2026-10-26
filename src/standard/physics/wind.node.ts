import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { applyWind, PhysicsWorldDefinition, StandardPhysicsPlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);

test("wind addresses a shape entity and accumulates only after its first sync until the next step", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        const body = world.create();
        world.add(body, Transform);
        world.add(body, Body, { type: BodyType.Dynamic });
        const shape = world.create();
        world.add(shape, Shape, { body, kind: ShapeKind.Sphere, radius: 0.5 });
        const wind = { x: 10, y: 0, z: 0 };
        applyWind(world, shape, wind, 1, 1, 10, true);
        world.tick();
        expect(world.get(body, GlobalTransform)!.linearVelocity[0]).toBe(0);
        applyWind(world, body, wind, 1, 1, 10, true);
        world.tick();
        expect(world.get(body, GlobalTransform)!.linearVelocity[0]).toBe(0);
        applyWind(world, shape, wind, 1, 1, 10, true);
        expect(world.get(body, GlobalTransform)!.linearVelocity[0]).toBe(0);
        world.tick();
        expect(world.get(body, GlobalTransform)!.linearVelocity[0]).toBeGreaterThan(0);
        world.remove(shape, Shape);
        expect(() => applyWind(world, shape, wind, 1, 1, 10, true)).not.toThrow();
        world.destroy(shape);
        const recycled = world.create();
        expect(recycled).toBe(shape);
        world.add(recycled, Shape, { body, kind: ShapeKind.Sphere, radius: 0.5 });
        const velocity = world.get(body, GlobalTransform)!.linearVelocity[0];
        applyWind(world, recycled, wind, 1, 1, 10, true);
        world.tick();
        expect(world.get(body, GlobalTransform)!.linearVelocity[0]).toBe(velocity);
    } finally {
        app.dispose();
    }
});
