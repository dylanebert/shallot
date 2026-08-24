import { afterAll, afterEach, describe, expect, spyOn, test } from "bun:test";
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
import { Hulls } from "../physics/core";
import { Slab } from "../slab";
import { shutdown } from "./engine";
import { Tumble, TumblePlugin } from "./index";

// World lifecycle conformance: the wasm kernel is a singleton with ONE resident region, so a leaked world on a rebuild is a
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

// Late-marshal constraint retry: a constraint whose endpoint body hasn't marshaled yet (its hull registers
// a tick later — the normal async-hull path) used to be dropped permanently. `createJoint` returned null
// with no retry, and `ConstraintSystem` re-uploads on an authored-signature change only, none of which
// moves when a deferred body finally marshals. The fix is a retained authored def set plus a `SyncSystem`
// re-invoke pump (index.ts / joints.ts). These two arms pin both halves.
//
// Manufacturing a deferred hull: register a `__unit_cube__` clone under a new name, then `Hulls.delete` it
// — `Registry.delete` frees the value and keeps the id slot, so the later re-register reuses the SAME id
// and grows `Hulls.size`, which is the key the marshal `failed` ledger retries on.

const LATE_HULL = "__late_marshal_arm__";

/** reserve a hull id whose hull is not registered — a body carrying it fails to marshal until {@link registerLateHull}. */
function reserveLateHullId(): number {
    const id = Hulls.register({ ...Hulls.get("__unit_cube__")!, name: LATE_HULL });
    Hulls.delete(LATE_HULL);
    return id;
}

const registerLateHull = (): void => {
    Hulls.register({ ...Hulls.get("__unit_cube__")!, name: LATE_HULL });
};

// the scene: a mass-0 Box anchor at y=5, a mass-1 Hull body at y=4 carrying the reserved (unregistered)
// hull id in halfExtents.w, and a spherical pin (stiffnessAng 0) between them with rA one unit below the
// anchor — so a live joint holds the bob at y≈4 and a dropped one free-falls it past y=0.
function pinnedDeferredScene(state: State): { anchor: number; bob: number } {
    const anchor = state.create();
    state.add(anchor, Body);
    Body.shape.set(anchor, ShapeKind.Box);
    Body.halfExtents.set(anchor, 0.5, 0.5, 0.5, 0);
    Body.pos.set(anchor, 0, 5, 0, 0);
    Body.mass.set(anchor, 0);

    const bob = state.create();
    state.add(bob, Body);
    Body.shape.set(bob, ShapeKind.Hull);
    Body.halfExtents.set(bob, 1, 1, 1, reserveLateHullId());
    Body.pos.set(bob, 0, 4, 0, 0);
    Body.mass.set(bob, 1);

    const j = state.create();
    state.add(j, Joint);
    Joint.a.set(j, anchor);
    Joint.b.set(j, bob);
    Joint.rA.set(j, 0, -1, 0, 0);
    Joint.rB.set(j, 0, 0, 0, 0);
    Joint.stiffnessAng.set(j, 0);

    return { anchor, bob };
}

describe("TumblePlugin late-marshal constraints", () => {
    let state: State;

    afterEach(() => {
        TumblePlugin.dispose?.(state);
        Hulls.delete(LATE_HULL);
    });

    test("a joint dropped against a not-yet-marshaled body is retried once it marshals", async () => {
        state = await buildTumble();
        const { bob } = pinnedDeferredScene(state);

        // one tick: bob's marshal fails (its hull id has no hull), so the joint's create finds no endpoint
        state.step(Time.FIXED_DT);
        // the hull arrives — Hulls.size grows, so the marshal ledger retries bob on the next tick
        registerLateHull();
        for (let i = 0; i < 60; i++) state.step(Time.FIXED_DT);

        const live = Physics.backend?.readBody(bob);
        expect(live).not.toBeNull();
        // POSITION, not readBody's non-nullness: the body's own marshal retry already worked before this
        // fix, so an arm keyed on readBody was green while the constraint was permanently lost. A live pin
        // holds the bob at y≈4; the dropped joint free-fell it to ≈-1.29.
        expect(live?.pos[1]).toBeGreaterThan(3);
    });

    test("the skip warning names the deferred marshal, not a non-Body reference", async () => {
        state = await buildTumble();
        pinnedDeferredScene(state);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            state.step(Time.FIXED_DT);
            const jointWarn = warn.mock.calls
                .map((c) => String(c[0]))
                .find((m) => m.includes("joint"));
            expect(jointWarn).toBeDefined();
            expect(jointWarn).toContain("deferred");
            expect(jointWarn).not.toContain("non-Body entity");
        } finally {
            warn.mockRestore();
        }
    });
});
