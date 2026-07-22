import { afterAll, afterEach, describe, expect, test } from "bun:test";
import { attach } from "../../../tests/helpers";
import { State, Time } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import {
    Body,
    bodyTraits,
    Joint,
    jointTraits,
    Physics,
    ShapeKind,
    Spring,
    springTraits,
} from "../physics";
import { Slab } from "../slab";
import { shutdown } from "./engine";
import { Tumble, TumblePlugin } from "./index";

// World lifecycle conformance (specs/tumble-shallot.md "World lifecycle is exclusive and disposal is
// mandatory"): the wasm kernel is a singleton with ONE resident region, so a leaked world on a rebuild is a
// hard failure, not a slow leak — the build→step→dispose ×2 roster entry this file is. No device needed:
// TumblePlugin's own warm() never touches Compute (CPU-native), so this runs at the fast `bun test` tier —
// bypasses `build()`/`app()` (register + Slab.collect + the lifecycle hooks directly), the orbit.test.ts shape.

// The wasm kernel is a process singleton (engine/kernel.ts). warm() runs init(), which boots the
// multithreaded worker pool wherever the host holds shared memory (bun does). Release it at file teardown
// so the pool's solve path doesn't leak into sibling engine test files that assume the single-thread kernel.
afterAll(shutdown);

async function buildTumble(): Promise<State> {
    clear();
    const state = new State();
    register("body", Body, bodyTraits);
    register("spring", Spring, springTraits);
    register("joint", Joint, jointTraits);
    Slab.collect();
    TumblePlugin.initialize?.(state);
    await TumblePlugin.warm?.(state);
    attach(state, TumblePlugin);
    return state;
}

describe("TumblePlugin lifecycle", () => {
    let state: State;

    afterEach(() => {
        TumblePlugin.dispose?.(state);
    });

    test("warm installs the backend and creates a world", async () => {
        state = await buildTumble();
        expect(Physics.backend).not.toBeNull();
        expect(Tumble.world).not.toBeNull();
    });

    test("dispose uninstalls the backend and destroys the world", async () => {
        state = await buildTumble();
        TumblePlugin.dispose?.(state);
        expect(Physics.backend).toBeNull();
        expect(Tumble.world).toBeNull();
    });

    test("build → step → dispose survives two full cycles (reload conformance)", async () => {
        for (let cycle = 0; cycle < 2; cycle++) {
            state = await buildTumble();
            const eid = state.create();
            state.add(eid, Body);
            Body.shape.set(eid, ShapeKind.Sphere);
            Body.halfExtents.set(eid, 0, 0, 0, 0.5);
            Body.pos.set(eid, 0, 5, 0, 0);
            Body.mass.set(eid, 1);
            for (let i = 0; i < 5; i++) state.step(Time.FIXED_DT);
            const live = Physics.backend?.readBody(eid);
            expect(live).not.toBeNull();
            expect(Number.isFinite(live?.pos[1])).toBe(true);
            TumblePlugin.dispose?.(state);
        }
    });

    test("a fresh world builds cleanly after a prior world was destroyed", async () => {
        state = await buildTumble();
        TumblePlugin.dispose?.(state);
        state = await buildTumble();
        expect(Tumble.world).not.toBeNull();
        expect(() => state.step(Time.FIXED_DT)).not.toThrow();
    });
});
