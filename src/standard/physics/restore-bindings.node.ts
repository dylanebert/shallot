import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { createApp, Time, type World } from "@dylanebert/shallot";
import { Body, ShapeKind, Spring } from "@dylanebert/shallot/physics";
import {
    hashPhysics,
    physicsWorld,
    readBody,
    restorePhysics,
    StandardPhysicsPlugin,
    setKinematic,
    snapshotPhysics,
} from "@dylanebert/shallot/standard/physics";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

function addBox(world: World, x: number, y: number, mass: number): number {
    const eid = world.create();
    world.add(eid, Body);
    world.storage(Body).shape.set(eid, ShapeKind.Box);
    world.storage(Body).halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
    world.storage(Body).position.set(eid, x, y, 0, 0);
    world.storage(Body).rotation.set(eid, 0, 0, 0, 1);
    world.storage(Body).mass.set(eid, mass);
    return eid;
}

async function scene() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    return { app, world: app.world };
}

test("a body spawned between snapshot and restore marshals after the restore", async () => {
    const { app, world } = await scene();
    try {
        addBox(world, 0, 2, 1);
        world.step(Time.FIXED_DT);
        const saved = snapshotPhysics(world);
        const spawned = addBox(world, 3, 2, 1);
        world.step(Time.FIXED_DT);
        restorePhysics(world, saved);
        world.step(Time.FIXED_DT);
        const before = readBody(world, spawned)!.position[1];
        world.step(Time.FIXED_DT);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(2);
        expect(physicsWorld(world)!.getBody(spawned)?.isValid()).toBe(true);
        expect(readBody(world, spawned)!.position[1]).toBeLessThan(before);
    } finally {
        app.dispose();
    }
});

test("a body despawned between snapshot and restore leaves no orphan solver body", async () => {
    const { app, world } = await scene();
    try {
        addBox(world, 0, 2, 1);
        const despawned = addBox(world, 3, 2, 1);
        world.step(Time.FIXED_DT);
        const saved = snapshotPhysics(world);
        world.destroy(despawned);
        world.step(Time.FIXED_DT);
        restorePhysics(world, saved);
        world.step(Time.FIXED_DT);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
    } finally {
        app.dispose();
    }
});

test("a spring added between snapshot and restore returns after it, and despawning it removes it", async () => {
    const { app, world } = await scene();
    try {
        const anchor = addBox(world, 0, 2, 0);
        const bob = addBox(world, 0, -2, 1);
        world.step(Time.FIXED_DT);
        const saved = snapshotPhysics(world);
        const spring = world.create();
        world.add(spring, Spring, { a: anchor, b: bob, rest: 4, stiffness: 100 });
        world.step(Time.FIXED_DT);
        restorePhysics(world, saved);
        world.step(Time.FIXED_DT);
        expect(physicsWorld(world)!.getCounters().jointCount).toBe(1);
        world.destroy(spring);
        world.step(Time.FIXED_DT);
        expect(physicsWorld(world)!.getCounters().jointCount).toBe(0);
    } finally {
        app.dispose();
    }
});

test("a kinematic body driven without velocity replays every tick's hash after a restore", async () => {
    const { app, world } = await scene();
    try {
        const platform = addBox(world, 0, 0, 0);
        addBox(world, 0.2, 1, 1);
        const place = (tick: number) =>
            setKinematic(world, platform, [tick * 0.05, 0, 0], [0, 0, 0, 1]);
        for (let tick = 0; tick < 5; tick++) {
            place(tick);
            world.step(Time.FIXED_DT);
        }
        const saved = snapshotPhysics(world);
        const original: bigint[] = [];
        for (let tick = 5; tick < 9; tick++) {
            place(tick);
            world.step(Time.FIXED_DT);
            original.push(hashPhysics(world));
        }
        restorePhysics(world, saved);
        const replay: bigint[] = [];
        for (let tick = 5; tick < 9; tick++) {
            place(tick);
            world.step(Time.FIXED_DT);
            replay.push(hashPhysics(world));
        }
        expect(replay).toEqual(original);
    } finally {
        app.dispose();
    }
});

test("restorePhysics refuses a solver snapshot without bindings before changing the world", async () => {
    const { app, world } = await scene();
    try {
        addBox(world, 0, 2, 1);
        world.step(Time.FIXED_DT);
        const solverOnly = physicsWorld(world)!.snapshot();
        world.step(Time.FIXED_DT);
        const hash = hashPhysics(world);
        expect(() => restorePhysics(world, solverOnly)).toThrow(
            "physics: restorePhysics needs a snapshot from snapshotPhysics",
        );
        expect(hashPhysics(world)).toBe(hash);
    } finally {
        app.dispose();
    }
});
