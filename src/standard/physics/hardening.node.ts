import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { createApp, Time, type World } from "@dylanebert/shallot";
import { Body, BodyType, ShapeKind } from "@dylanebert/shallot/physics";
import {
    hashPhysics,
    type PhysicsSnapshot,
    physicsWorld,
    StandardPhysicsPlugin,
    setLinearVelocity,
} from "@dylanebert/shallot/standard/physics";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { physicsCounters } from "./runtime";

await setupGlobals();

function addBody(
    world: World,
    data: {
        shape: number;
        pos: [number, number, number];
        halfExtents: [number, number, number, number];
        mass: number;
        friction?: number;
        quat?: [number, number, number, number];
    },
): number {
    const eid = world.create();
    world.add(eid, Body);
    world.storage(Body).shape.set(eid, data.shape);
    world.storage(Body).halfExtents.set(eid, ...data.halfExtents);
    world.storage(Body).position.set(eid, data.pos[0], data.pos[1], data.pos[2], 0);
    world.storage(Body).rotation.set(eid, ...(data.quat ?? [0, 0, 0, 1]));
    world.storage(Body).type.set(eid, BodyType.Dynamic);
    world.storage(Body).mass.set(eid, data.mass);
    world.storage(Body).friction.set(eid, data.friction ?? 0.5);
    return eid;
}

async function cleanState() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const body = addBody(app.world, {
        shape: ShapeKind.Box,
        pos: [0, 2, 0],
        halfExtents: [0.5, 0.5, 0.5, 0],
        mass: 1,
    });
    return { app, world: app.world, body };
}

test("sequential clean physics Worlds and an owner-world snapshot replay one fixed action stream, so rollback reproduces a confirmed tick without overlapping global slabs", async () => {
    const left = await cleanState();
    const leftHashes: string[] = [];
    const leftAfterSaved: string[] = [];
    try {
        for (let tick = 0; tick < 6; tick++) {
            setLinearVelocity(left.world, left.body, { x: 1, y: 0, z: 0 });
            left.world.step(Time.FIXED_DT);
            leftHashes.push(hashPhysics(left.world).toString(16));
            if (tick > 2) leftAfterSaved.push(hashPhysics(left.world).toString(16));
        }
    } finally {
        left.app.dispose();
    }

    const right = await cleanState();
    const rightHashes: string[] = [];
    try {
        for (let tick = 0; tick < 6; tick++) {
            setLinearVelocity(right.world, right.body, { x: 1, y: 0, z: 0 });
            right.world.step(Time.FIXED_DT);
            rightHashes.push(hashPhysics(right.world).toString(16));
        }
    } finally {
        right.app.dispose();
    }
    expect(leftHashes).toEqual(rightHashes);

    const replay = await cleanState();
    try {
        for (let tick = 0; tick < 3; tick++) {
            setLinearVelocity(replay.world, replay.body, { x: 1, y: 0, z: 0 });
            replay.world.step(Time.FIXED_DT);
        }
        const saved = replay.world.snapshot();
        const savedHash = hashPhysics(replay.world);
        const before = savedHash;
        replay.world.step(Time.FIXED_DT);
        replay.world.restore(saved);
        expect(hashPhysics(replay.world)).toBe(savedHash);
        setLinearVelocity(replay.world, replay.body, { x: 1, y: 0, z: 0 });
        replay.world.step(Time.FIXED_DT);
        expect(hashPhysics(replay.world).toString(16)).toBe(leftAfterSaved[0]);
        expect(hashPhysics(replay.world)).not.toBe(before);
    } finally {
        replay.app.dispose();
    }
});

interface SnapshotRefs {
    snapshot: WeakRef<PhysicsSnapshot>;
    logical: WeakRef<object>;
    bytes: WeakRef<Uint8Array>;
}

function snapshotRefs(saved: PhysicsSnapshot): SnapshotRefs {
    return {
        snapshot: new WeakRef(saved),
        logical: new WeakRef(saved.state as object),
        bytes: new WeakRef(saved.bytes),
    };
}

function droppedSnapshotRefs(world: World, registry: FinalizationRegistry<string>): SnapshotRefs {
    const saved = physicsWorld(world)!.snapshot();
    registry.register(saved, "dropped");
    return snapshotRefs(saved);
}

function retainedSnapshot(
    world: World,
    registry: FinalizationRegistry<string>,
): {
    saved: PhysicsSnapshot;
    refs: SnapshotRefs;
} {
    const saved = physicsWorld(world)!.snapshot();
    registry.register(saved, "control", saved);
    return { saved, refs: snapshotRefs(saved) };
}

test("a solver snapshot restores into a fresh compatible PhysicsWorld with an equivalent hash", async () => {
    const source = await cleanState();
    const target = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        for (let tick = 0; tick < 4; tick++) {
            setLinearVelocity(source.world, source.body, { x: 1, y: 0, z: 0 });
            source.world.step(Time.FIXED_DT);
        }
        const saved = physicsWorld(source.world)!.snapshot();
        const expected = hashPhysics(source.world);
        const targetWorld = physicsWorld(target.world);
        expect(targetWorld).not.toBeNull();
        targetWorld!.restore(saved);
        expect(hashPhysics(target.world)).toBe(expected);
    } finally {
        target.dispose();
        source.app.dispose();
    }
});

test("snapshot logical state and WASM bytes are collectable after the caller drops them", async () => {
    const subject = await cleanState();
    try {
        const finalized = new Set<string>();
        const registry = new FinalizationRegistry<string>((tag) => finalized.add(tag));
        const dropped = droppedSnapshotRefs(subject.world, registry);
        const control = retainedSnapshot(subject.world, registry);

        const deadline = Date.now() + 1_000;
        while (Date.now() < deadline) {
            // A deref keeps its target alive for this job; yield before the next collection.
            await new Promise((resolve) => setTimeout(resolve, 0));
            Bun.gc(true);
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (
                finalized.has("dropped") &&
                dropped.snapshot.deref() === undefined &&
                dropped.logical.deref() === undefined &&
                dropped.bytes.deref() === undefined
            )
                break;
        }

        expect(dropped.logical.deref()).toBeUndefined();
        expect(dropped.bytes.deref()).toBeUndefined();
        expect(dropped.snapshot.deref()).toBeUndefined();
        expect(finalized.has("dropped")).toBe(true);
        expect(finalized.has("control")).toBe(false);
        expect(control.refs.snapshot.deref()).toBe(control.saved);
        // Keep the registry alive through collection, not just its callback's result set.
        expect(registry.unregister(control.saved)).toBe(true);
    } finally {
        subject.app.dispose();
    }
});

test("physics reports the same body visit count for every scene, so a body-content mutation can hide an omitted visit from the budget row", async () => {
    const one = await cleanState();
    try {
        one.world.step(Time.FIXED_DT);
        const oneCount = physicsCounters(one.world).bodiesVisited;
        expect(oneCount).toBe(1);
        expect(one.world.has(one.body, Body)).toBe(true);

        const second = addBody(one.world, {
            shape: ShapeKind.Box,
            pos: [2, 2, 0],
            halfExtents: [0.5, 0.5, 0.5, 0],
            mass: 1,
        });
        one.world.step(Time.FIXED_DT);
        const changed = physicsCounters(one.world);
        expect(changed.bodiesVisited).toBe(2);
        expect(one.world.has(second, Body)).toBe(true);
    } finally {
        one.app.dispose();
    }
});
