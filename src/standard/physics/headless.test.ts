import { afterAll, afterEach, expect } from "bun:test";
import { Time } from "../../engine";
import { check } from "../../harness/check";
import { hashWorldState, shutdown } from "./api";
import { addBody, headlessPhysicsState } from "./headless.fixture";
import { Physics, PhysicsPlugin, ShapeKind } from "./index";

// The premise every other physics check rests on: the solver is CPU-native wasm, so a `State` with
// `PhysicsPlugin` warms and steps in Bun with no GPU device and no browser.

let live: Awaited<ReturnType<typeof headlessPhysicsState>> | null = null;

afterEach(() => {
    if (live) PhysicsPlugin.dispose?.(live);
    live = null;
});
afterAll(shutdown);

check(
    "a headless State warms PhysicsPlugin and steps it with no GPU",
    {
        claim: "physics stops stepping without a GPU device, so every step-tier physics check would be unrunnable in Bun",
    },
    async () => {
        const state = await headlessPhysicsState();
        live = state;
        expect(Physics.world).not.toBeNull();

        addBody(state, {
            shape: ShapeKind.Box,
            pos: [0, 0, 0],
            halfExtents: [10, 0.5, 10, 0],
            mass: 0,
        });
        const falling = addBody(state, {
            shape: ShapeKind.Box,
            pos: [0, 5, 0],
            halfExtents: [0.5, 0.5, 0.5, 0],
            mass: 1,
        });

        const hashes: bigint[] = [];
        for (let i = 0; i < 30; i++) {
            state.step(Time.FIXED_DT);
            hashes.push(hashWorldState(Physics.world!.state));
        }

        // the world advanced: 30 distinct states, and the dynamic body fell under gravity while the
        // static floor held.
        expect(new Set(hashes.map(String)).size).toBe(30);
        const handle = Physics.body(falling);
        expect(handle).not.toBeNull();
        const y = handle!.getPosition().y;
        expect(y).toBeLessThan(5);
        expect(y).toBeGreaterThan(0);
    },
);
