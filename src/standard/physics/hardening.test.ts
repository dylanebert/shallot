import { expect } from "bun:test";
import { build, type State, Time } from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import {
    Body,
    hash,
    PhysicsPlugin,
    physicsCounters,
    readBody,
    restore,
    ShapeKind,
    setVelocity,
    snapshot,
} from "@dylanebert/shallot/physics";

function addBody(
    state: State,
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
    Body.shape.set(eid, data.shape);
    Body.halfExtents.set(eid, ...data.halfExtents);
    Body.pos.set(eid, data.pos[0], data.pos[1], data.pos[2], 0);
    Body.quat.set(eid, ...(data.quat ?? [0, 0, 0, 1]));
    Body.mass.set(eid, data.mass);
    Body.friction.set(eid, data.friction ?? 0.5);
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

check(
    "physics: twin states and restore replay the same fixed action stream",
    {
        claim: "sequential clean physics States and a restored wasm world replay one fixed action stream, so rollback reproduces a confirmed tick without overlapping global slabs",
    },
    async () => {
        const left = await cleanState();
        const leftHashes: string[] = [];
        let saved = snapshot(left.state);
        let savedHash = hash(left.state);
        const leftAfterSaved: string[] = [];
        try {
            for (let tick = 0; tick < 6; tick++) {
                setVelocity(left.state, left.body, 1, 0, 0);
                left.state.step(Time.FIXED_DT);
                leftHashes.push(hash(left.state).toString(16));
                if (tick === 2) {
                    saved = snapshot(left.state);
                    savedHash = hash(left.state);
                }
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
            const before = hash(replay.state);
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
    },
);

check(
    "physics: counters follow authored body content",
    {
        claim: "physics reports the same body visit count for every scene, so a body-content mutation can hide an omitted visit from the budget row",
    },
    async () => {
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
            expect(changed.bytesUploaded).toBe(0);
        } finally {
            one.app.dispose();
        }
    },
);
