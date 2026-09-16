import { afterEach, expect } from "bun:test";
import { build, type State, Time } from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { Body, body, PhysicsPlugin, physicsWorld, ShapeKind } from "@dylanebert/shallot/physics";

// The solver is CPU-native wasm: the public PhysicsPlugin composition warms and steps in Bun with no
// GPU device. Keep authoring in this check so it proves the public component writes rather than a fixture seam.

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

let live: Awaited<ReturnType<typeof build>> | null = null;

afterEach(() => {
    live?.dispose();
    live = null;
});
check(
    "a headless State warms PhysicsPlugin and steps it with no GPU",
    {
        claim: "physics stops stepping without a GPU device, so every step-tier physics check would be unrunnable in Bun",
    },
    async () => {
        expect(globalThis.navigator?.gpu).toBeUndefined();
        live = await build({ defaults: false, plugins: [PhysicsPlugin] });
        const { state } = live;
        expect(physicsWorld(state)).not.toBeNull();

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

        for (let i = 0; i < 30; i++) state.step(Time.FIXED_DT);

        // the dynamic body fell under gravity while the static floor held.
        const handle = body(state, falling);
        expect(handle).not.toBeNull();
        const y = handle!.getPosition().y;
        expect(y).toBeLessThan(5);
        expect(y).toBeGreaterThan(0);
    },
);
