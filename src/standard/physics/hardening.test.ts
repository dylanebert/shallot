import { expect } from "bun:test";
import { Time } from "../../engine";
import { check } from "../../harness/check";
import { addBody, headlessPhysicsState } from "./headless.fixture";
import {
    hash,
    PhysicsPlugin,
    physicsCounters,
    readBody,
    restore,
    ShapeKind,
    setVelocity,
    snapshot,
} from "./index";

async function cleanState() {
    const state = await headlessPhysicsState();
    const body = addBody(state, {
        shape: ShapeKind.Box,
        pos: [0, 2, 0],
        halfExtents: [0.5, 0.5, 0.5, 0],
        mass: 1,
    });
    return { state, body };
}

check(
    "physics: twin states and restore replay the same fixed action stream",
    {
        claim: "two clean physics States and a restored wasm world diverge under one fixed action stream, so rollback cannot reproduce a confirmed tick",
    },
    async () => {
        const left = await cleanState();
        const right = await cleanState();
        try {
            const leftHashes: string[] = [];
            const rightHashes: string[] = [];
            let saved = snapshot(left.state);
            let savedHash = hash(left.state);
            const leftAfterSaved: string[] = [];
            for (let tick = 0; tick < 6; tick++) {
                setVelocity(left.state, left.body, 1, 0, 0);
                setVelocity(right.state, right.body, 1, 0, 0);
                left.state.step(Time.FIXED_DT);
                right.state.step(Time.FIXED_DT);
                leftHashes.push(hash(left.state).toString(16));
                rightHashes.push(hash(right.state).toString(16));
                if (tick === 2) {
                    saved = snapshot(left.state);
                    savedHash = hash(left.state);
                }
                if (tick > 2) leftAfterSaved.push(hash(left.state).toString(16));
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
                PhysicsPlugin.dispose?.(replay.state);
            }
        } finally {
            PhysicsPlugin.dispose?.(left.state);
            PhysicsPlugin.dispose?.(right.state);
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
            PhysicsPlugin.dispose?.(one.state);
        }
    },
);
