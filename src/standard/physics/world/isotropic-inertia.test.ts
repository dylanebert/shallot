import { expect, test } from "bun:test";
import { createApp } from "../../../engine";
import { PhysicsWorldDefinition, StandardPhysicsPlugin } from "..";
import { restore, snapshot } from "../api/snapshot";
import { kernel } from "../kernel/kernel";
import { physicsState } from "../runtime";
import { createTestWorld, destroyTestWorld } from "../testing";
import { worldGetIsotropicInertiaTolerance, worldSetIsotropicInertiaTolerance } from "./world";

test("isotropic inertia tolerance applies at fixed sync by comparing applied kernel state", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const definition = world.resource(PhysicsWorldDefinition);
        const state = physicsState(world)!;
        expect(definition.isotropicInertiaTolerance).toBe(Math.fround(0.2));
        definition.isotropicInertiaTolerance = 0.9;
        expect(worldGetIsotropicInertiaTolerance(state)).toBe(Math.fround(0.2));
        world.tick();
        expect(worldGetIsotropicInertiaTolerance(state)).toBe(Math.fround(0.9));
        worldSetIsotropicInertiaTolerance(state, 0.1);
        world.tick();
        expect(worldGetIsotropicInertiaTolerance(state)).toBe(Math.fround(0.9));
        state.locked = true;
        worldSetIsotropicInertiaTolerance(state, 0);
        expect(worldGetIsotropicInertiaTolerance(state)).toBe(Math.fround(0.9));
        state.locked = false;
        worldSetIsotropicInertiaTolerance(state, -1);
        expect(worldGetIsotropicInertiaTolerance(state)).toBe(0);
        worldSetIsotropicInertiaTolerance(state, 2);
        expect(worldGetIsotropicInertiaTolerance(state)).toBe(1);
    } finally {
        app.dispose();
    }
});

test("isotropic inertia tolerance round-trips in the kernel snapshot without changing a sibling", () => {
    const world = createTestWorld({ isotropicInertiaTolerance: 0.7 });
    const sibling = createTestWorld({ isotropicInertiaTolerance: 0.4 });
    try {
        const saved = snapshot(world);
        worldSetIsotropicInertiaTolerance(world, 0.1);
        restore(world, saved);
        expect(worldGetIsotropicInertiaTolerance(world)).toBe(Math.fround(0.7));
        expect(worldGetIsotropicInertiaTolerance(sibling)).toBe(Math.fround(0.4));
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(sibling);
    }
});

test("resetting the Bodies region returns its isotropic inertia tolerance to the default", () => {
    const world = createTestWorld({ isotropicInertiaTolerance: 0.8 });
    const k = kernel(world.ecsState);
    const id = world.worldId;
    destroyTestWorld(world);
    expect(k.worldGetIsotropicInertiaTolerance(id)).toBe(Math.fround(0.2));
});
