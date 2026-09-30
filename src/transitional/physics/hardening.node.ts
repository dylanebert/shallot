import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { build, type World, Time } from "@dylanebert/shallot";
import {
    Body,
    hash,
    PhysicsPlugin,
    physicsCounters,
    physicsWorld,
    readBody,
    restore,
    ShapeKind,
    setVelocity,
    snapshot,
    type WorldSnapshot,
} from "@dylanebert/shallot/physics";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

function addBody(
    state: World,
    data: {
        shape: number;
        pos: [number, number, number];
        halfExtents: [number, number, number, number];
        mass: number;
        friction?: number;
        quat?: [number, number, number, number];
    },
): number {
    const eid = state.create();
    state.add(eid, Body);
    state.of(Body).shape.set(eid, data.shape);
    state.of(Body).halfExtents.set(eid, ...data.halfExtents);
    state.of(Body).pos.set(eid, data.pos[0], data.pos[1], data.pos[2], 0);
    state.of(Body).quat.set(eid, ...(data.quat ?? [0, 0, 0, 1]));
    state.of(Body).mass.set(eid, data.mass);
    state.of(Body).friction.set(eid, data.friction ?? 0.5);
    return eid;
}

async function cleanState() {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin] });
    const body = addBody(app.state, {
        shape: ShapeKind.Box,
        pos: [0, 2, 0],
        halfExtents: [0.5, 0.5, 0.5, 0],
        mass: 1,
    });
    return { app, state: app.state, body };
}

test("sequential clean physics States and an owner-world snapshot replay one fixed action stream, so rollback reproduces a confirmed tick without overlapping global slabs", async () => {
    const left = await cleanState();
    const leftHashes: string[] = [];
    const leftAfterSaved: string[] = [];
    try {
        for (let tick = 0; tick < 6; tick++) {
            setVelocity(left.state, left.body, 1, 0, 0);
            left.state.step(Time.FIXED_DT);
            leftHashes.push(hash(left.state).toString(16));
            if (tick > 2) leftAfterSaved.push(hash(left.state).toString(16));
        }
    } finally {
        left.app.dispose();
    }

    const right = await cleanState();
    const rightHashes: string[] = [];
    try {
        for (let tick = 0; tick < 6; tick++) {
            setVelocity(right.state, right.body, 1, 0, 0);
            right.state.step(Time.FIXED_DT);
            rightHashes.push(hash(right.state).toString(16));
        }
    } finally {
        right.app.dispose();
    }
    expect(leftHashes).toEqual(rightHashes);

    const replay = await cleanState();
    try {
        for (let tick = 0; tick < 3; tick++) {
            setVelocity(replay.state, replay.body, 1, 0, 0);
            replay.state.step(Time.FIXED_DT);
        }
        const saved = snapshot(replay.state);
        const savedHash = hash(replay.state);
        const before = savedHash;
        replay.state.step(Time.FIXED_DT);
        restore(replay.state, saved);
        expect(hash(replay.state)).toBe(savedHash);
        setVelocity(replay.state, replay.body, 1, 0, 0);
        replay.state.step(Time.FIXED_DT);
        expect(hash(replay.state).toString(16)).toBe(leftAfterSaved[0]);
        expect(hash(replay.state)).not.toBe(before);
    } finally {
        replay.app.dispose();
    }
});

interface SnapshotRefs {
    snapshot: WeakRef<WorldSnapshot>;
    logical: WeakRef<object>;
    bytes: WeakRef<Uint8Array>;
}

function snapshotRefs(saved: WorldSnapshot): SnapshotRefs {
    return {
        snapshot: new WeakRef(saved),
        logical: new WeakRef(saved.state as object),
        bytes: new WeakRef(saved.bytes),
    };
}

function droppedSnapshotRefs(state: World, registry: FinalizationRegistry<string>): SnapshotRefs {
    const saved = snapshot(state);
    registry.register(saved, "dropped");
    return snapshotRefs(saved);
}

function retainedSnapshot(
    state: World,
    registry: FinalizationRegistry<string>,
): {
    saved: WorldSnapshot;
    refs: SnapshotRefs;
} {
    const saved = snapshot(state);
    registry.register(saved, "control");
    return { saved, refs: snapshotRefs(saved) };
}

test("a snapshot restores into a fresh compatible World with an equivalent hash", async () => {
    const source = await cleanState();
    const target = await build({ defaults: false, plugins: [PhysicsPlugin] });
    try {
        for (let tick = 0; tick < 4; tick++) {
            setVelocity(source.state, source.body, 1, 0, 0);
            source.state.step(Time.FIXED_DT);
        }
        const saved = snapshot(source.state);
        const expected = hash(source.state);
        const targetWorld = physicsWorld(target.state);
        expect(targetWorld).not.toBeNull();
        targetWorld!.restore(saved);
        expect(hash(target.state)).toBe(expected);
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
        const dropped = droppedSnapshotRefs(subject.state, registry);
        const control = retainedSnapshot(subject.state, registry);

        const deadline = Date.now() + 1_000;
        while (!finalized.has("dropped") && Date.now() < deadline) {
            Bun.gc(true);
            await new Promise((resolve) => setTimeout(resolve, 0));
        }

        expect(finalized.has("dropped")).toBe(true);
        expect(finalized.has("control")).toBe(false);
        expect(control.refs.snapshot.deref()).toBe(control.saved);
        expect(dropped.snapshot.deref()).toBeUndefined();
        expect(dropped.logical.deref()).toBeUndefined();
        expect(dropped.bytes.deref()).toBeUndefined();
    } finally {
        subject.app.dispose();
    }
});

test("physics reports the same body visit count for every scene, so a body-content mutation can hide an omitted visit from the budget row", async () => {
    const one = await cleanState();
    try {
        one.state.step(Time.FIXED_DT);
        const oneCount = physicsCounters(one.state).bodiesVisited;
        expect(oneCount).toBe(1);
        expect(readBody(one.state, one.body)).not.toBeNull();

        const second = addBody(one.state, {
            shape: ShapeKind.Box,
            pos: [2, 2, 0],
            halfExtents: [0.5, 0.5, 0.5, 0],
            mass: 1,
        });
        one.state.step(Time.FIXED_DT);
        const changed = physicsCounters(one.state);
        expect(changed.bodiesVisited).toBe(2);
        expect(readBody(one.state, second)).not.toBeNull();
    } finally {
        one.app.dispose();
    }
});
