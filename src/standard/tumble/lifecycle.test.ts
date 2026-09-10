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
// hull id in halfExtents.w, and a constraint between them — a spherical pin (stiffnessAng 0) with rA one
// unit below the anchor, or the spring twin (a stiff distance spring, anchor-to-anchor rest 1). Either way a
// live constraint holds the bob at y≈4 and a dropped one free-falls it past y=0. Returns the bob eid, the
// only handle an arm reads.
function pinnedDeferredScene(state: State, kind: "joint" | "spring"): number {
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

    const c = state.create();
    if (kind === "joint") {
        state.add(c, Joint);
        Joint.a.set(c, anchor);
        Joint.b.set(c, bob);
        Joint.rA.set(c, 0, -1, 0, 0);
        Joint.rB.set(c, 0, 0, 0, 0);
        Joint.stiffnessAng.set(c, 0);
    } else {
        state.add(c, Spring);
        Spring.a.set(c, anchor);
        Spring.b.set(c, bob);
        Spring.rA.set(c, 0, 0, 0, 0);
        Spring.rB.set(c, 0, 0, 0, 0);
        // stiff enough that the static droop (mg/k ≈ 0.01 m) stays far above the free-fall the arm rejects
        Spring.stiffness.set(c, 1000);
        Spring.rest.set(c, 1);
    }

    return bob;
}

describe("TumblePlugin late-marshal constraints", () => {
    let state: State;

    afterEach(() => {
        TumblePlugin.dispose?.(state);
        Hulls.delete(LATE_HULL);
    });

    test("a joint dropped against a not-yet-marshaled body is retried once it marshals", async () => {
        state = await buildTumble();
        const bob = pinnedDeferredScene(state, "joint");

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

    // the spring half of the same pump: `createSpring` has its own early-out and its own warn, so the joint
    // arm alone would leave `syncSprings`/`resyncConstraints`'s spring branch uncovered.
    test("a spring dropped against a not-yet-marshaled body is retried once it marshals", async () => {
        state = await buildTumble();
        const bob = pinnedDeferredScene(state, "spring");

        state.step(Time.FIXED_DT);
        registerLateHull();
        for (let i = 0; i < 60; i++) state.step(Time.FIXED_DT);

        const live = Physics.backend?.readBody(bob);
        expect(live).not.toBeNull();
        // a live spring holds the bob one rest-length below the anchor (y≈4, minus the mg/k droop); a dropped
        // one free-falls it well past y=0.
        expect(live?.pos[1]).toBeGreaterThan(3);
    });

    test("the skip warning names the deferred marshal, not a non-Body reference", async () => {
        state = await buildTumble();
        pinnedDeferredScene(state, "joint");

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

    // the mixed pair: `a` is truly not a `Body`, `b` is a deferred marshal. Classifying only the first missing
    // endpoint would name `a`'s cause and silently apply it to `b` — for the deferred half, exactly the
    // misdirection the discriminator exists to delete. Both endpoints get their own cause.
    test("a mixed pair names each endpoint's own cause", async () => {
        state = await buildTumble();

        const notABody = state.create(); // no Body component: permanently unsatisfiable
        const bob = state.create();
        state.add(bob, Body);
        Body.shape.set(bob, ShapeKind.Hull);
        Body.halfExtents.set(bob, 1, 1, 1, reserveLateHullId());
        Body.pos.set(bob, 0, 4, 0, 0);
        Body.mass.set(bob, 1);

        const j = state.create();
        state.add(j, Joint);
        Joint.a.set(j, notABody);
        Joint.b.set(j, bob);
        Joint.rA.set(j, 0, 0, 0, 0);
        Joint.rB.set(j, 0, 0, 0, 0);
        Joint.stiffnessAng.set(j, 0);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            state.step(Time.FIXED_DT);
            const jointWarn = warn.mock.calls
                .map((c) => String(c[0]))
                .find((m) => m.includes("joint endpoint unavailable"));
            expect(jointWarn).toBeDefined();
            expect(jointWarn).toContain(`a: ${notABody} is not a Body`);
            expect(jointWarn).toContain(`b: ${bob} is a deferred body`);
        } finally {
            warn.mockRestore();
        }
    });

    // The narrowing arm: a mixed pair where `a` is genuinely non-`Body` and `b` is a deferred `Body`.
    // On the authored upload, `endpoints()` warns naming both causes (a: not a Body; b: deferred). When `b`
    // marshals and the pump re-runs, `endpoints()` still returns null (`a` is still missing) and should warn
    // with the NARROWED cause naming only `a` — but under the old key (`${key}|endpoint`, ignoring `parts`)
    // the key was already banked, so the corrected diagnostic was swallowed. Folding `parts` into the key
    // makes a changed composition a distinct key, so the narrowed warning re-warns once.
    //
    // Pre-fix witnessed red: afterTen.length = 1 (Expected 2, Received 1) — the second (narrowed) warning
    // was swallowed by the stale `${key}|endpoint` key banked on the authored upload. The final count after
    // 30 churn ticks was also 1, so the narrowing was never observed.
    // Green: afterTen.length = 2, final count = 2 — the narrowed warning fires once, then stays put.
    //
    // If the key were keyed TOO finely (e.g. on something that changes every tick), afterTen.length would
    // grow past 2 and the final count would exceed afterTen.length — the never-thrash invariant breaks.
    // This arm catches that direction: the final-count-equals-afterTen assertion reds on per-tick growth.
    // The existing never-thrash arm ("a permanently unsatisfiable constraint warns once") catches the same
    // direction for the static (non-narrowing) case; this arm catches it for the narrowing case.
    test("a mixed pair re-warns with the narrowed cause once the deferred half marshals", async () => {
        state = await buildTumble();

        const notABody = state.create(); // no Body component: permanently unsatisfiable
        const bob = state.create();
        state.add(bob, Body);
        Body.shape.set(bob, ShapeKind.Hull);
        Body.halfExtents.set(bob, 1, 1, 1, reserveLateHullId());
        Body.pos.set(bob, 0, 4, 0, 0);
        Body.mass.set(bob, 1);

        const j = state.create();
        state.add(j, Joint);
        Joint.a.set(j, notABody);
        Joint.b.set(j, bob);
        Joint.rA.set(j, 0, 0, 0, 0);
        Joint.rB.set(j, 0, 0, 0, 0);
        Joint.stiffnessAng.set(j, 0);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        const endpointWarnings = (): string[] =>
            warn.mock.calls
                .map((c) => String(c[0]))
                .filter((m) => m.includes("joint endpoint unavailable"));

        const spawn = (i: number): void => {
            const eid = state.create();
            state.add(eid, Body);
            Body.shape.set(eid, ShapeKind.Sphere);
            Body.halfExtents.set(eid, 0, 0, 0, 0.5);
            Body.pos.set(eid, i * 2, 10, 0, 0);
            Body.mass.set(eid, 1);
        };

        try {
            // tick 0: authored upload — both endpoints missing, mixed causes
            state.step(Time.FIXED_DT);
            const firstWarnings = endpointWarnings();
            expect(firstWarnings.length).toBe(1);
            expect(firstWarnings[0]).toContain(`a: ${notABody} is not a Body`);
            expect(firstWarnings[0]).toContain(`b: ${bob} is a deferred body`);

            // hull arrives — bob marshals on the next tick
            registerLateHull();

            // churn: one spawn per tick fires the pump every tick
            for (let i = 0; i < 10; i++) {
                spawn(i);
                state.step(Time.FIXED_DT);
            }
            // the narrowed warning fires once, naming only the surviving cause (a's non-Body),
            // NOT b's deferred cause — b marshaled, so b is no longer missing
            const afterTen = endpointWarnings();
            expect(afterTen.length).toBe(2);
            expect(afterTen[1]).toContain(`a: ${notABody} is not a Body`);
            expect(afterTen[1]).not.toContain("deferred");
            expect(afterTen[1]).not.toContain(`b: ${bob}`);

            // count stays put across remaining churn — no thrash
            for (let i = 10; i < 30; i++) {
                spawn(i);
                state.step(Time.FIXED_DT);
            }
            expect(endpointWarnings().length).toBe(afterTen.length);
        } finally {
            warn.mockRestore();
        }
    });

    // The pump fires on ANY body-set change, so a continuously spawning scene re-syncs every tick. A retry
    // that re-warned would emit one warning per tick forever, breaking the same never-thrash-the-frame-loop
    // invariant the marshal `failed` ledger holds (index.ts). Counted across ticks, because no per-tick arm
    // can see a warning that only grows with tick count.
    test("a permanently unsatisfiable constraint warns once, not once per body-set change", async () => {
        state = await buildTumble();

        const notABody = state.create(); // no Body: this joint can never be satisfied
        const anchor = state.create();
        state.add(anchor, Body);
        Body.shape.set(anchor, ShapeKind.Box);
        Body.halfExtents.set(anchor, 0.5, 0.5, 0.5, 0);
        Body.pos.set(anchor, 0, 5, 0, 0);
        Body.mass.set(anchor, 0);

        const j = state.create();
        state.add(j, Joint);
        Joint.a.set(j, anchor);
        Joint.b.set(j, notABody);
        Joint.rA.set(j, 0, -1, 0, 0);
        Joint.rB.set(j, 0, 0, 0, 0);
        Joint.stiffnessAng.set(j, 0);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        const spawn = (i: number): void => {
            const eid = state.create();
            state.add(eid, Body);
            Body.shape.set(eid, ShapeKind.Sphere);
            Body.halfExtents.set(eid, 0, 0, 0, 0.5);
            Body.pos.set(eid, i * 2, 10, 0, 0);
            Body.mass.set(eid, 1);
        };
        const jointWarnings = (): number =>
            warn.mock.calls.filter((c) => String(c[0]).includes("joint")).length;

        try {
            // one body spawned per tick, so the body set changes (and the pump fires) on every tick
            for (let i = 0; i < 10; i++) {
                spawn(i);
                state.step(Time.FIXED_DT);
            }
            const afterTen = jointWarnings();
            for (let i = 10; i < 30; i++) {
                spawn(i);
                state.step(Time.FIXED_DT);
            }
            // the authored upload's one warning, and no growth with tick count
            expect(afterTen).toBe(1);
            expect(jointWarnings()).toBe(afterTen);
        } finally {
            warn.mockRestore();
        }
    });

    // The deferred-then-unsatisfiable class: a def whose endpoint was deferred at authored time resolves
    // into a both-static pair on retry. The authored upload's `endpoints()` warning names "deferred, will
    // retry" (not "unsatisfiable"), and the both-static guard sits AFTER `endpoints()` returns a pair — so
    // the retry is the only path that can evaluate it. Under `quiet` (S2's fix), the retry was silenced, so
    // the both-static warning was permanently unreachable: count 0. Under dedupe (S3's fix), the both-static
    // warning fires exactly once (deduped on the composite key `jointKey|both-static`, which is distinct
    // from the `jointKey|endpoint` the authored upload banked). Pre-fix witnessed red: count 0 (Expected 1,
    // Received 0). Green: count 1. Under the wrong narrowing (un-quieting the three guards), the count grows
    // past 1 — the pump fires on every body-set change (one spawn per tick), and the both-static warning
    // re-fires each tick with no dedupe.
    test("a deferred-then-both-static joint warns exactly once naming the unsatisfiable cause", async () => {
        state = await buildTumble();

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
        Body.mass.set(bob, 0); // mass-0: both-static when it marshals

        const j = state.create();
        state.add(j, Joint);
        Joint.a.set(j, anchor);
        Joint.b.set(j, bob);
        Joint.rA.set(j, 0, -1, 0, 0);
        Joint.rB.set(j, 0, 0, 0, 0);
        Joint.stiffnessAng.set(j, 0);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        const unsatisfiableWarnings = (): number =>
            warn.mock.calls.filter((c) => String(c[0]).includes("unsatisfiable")).length;

        const spawn = (i: number): void => {
            const eid = state.create();
            state.add(eid, Body);
            Body.shape.set(eid, ShapeKind.Sphere);
            Body.halfExtents.set(eid, 0, 0, 0, 0.5);
            Body.pos.set(eid, i * 2, 10, 0, 0);
            Body.mass.set(eid, 1);
        };

        try {
            // tick 0: bob's marshal fails (hull not registered), endpoints warns "deferred"
            state.step(Time.FIXED_DT);
            // hull arrives — bob will marshal on the next tick
            registerLateHull();
            // 30 ticks with one spawn per tick: the pump fires every tick
            for (let i = 0; i < 30; i++) {
                spawn(i);
                state.step(Time.FIXED_DT);
            }
            // exactly one warning naming the unsatisfiable cause (the both-static guard),
            // not zero (quiet) and not growing (wrong narrowing / un-quieting)
            expect(unsatisfiableWarnings()).toBe(1);
        } finally {
            warn.mockRestore();
        }
    });

    // The pump's own safety claim: `resyncConstraints` asserts "a no-op walk over the live joints — each
    // `isValid()` handle is reused, nothing is created or destroyed". Nothing gated this claim before — the
    // shipped warning-count arm reads no handle identity and no pose, so a future desync between `jointKey`
    // and the def content `syncSet` diffs on would silently destroy and recreate every constraint per tick
    // under churn (dropping warm-started impulses, symptom: mushy chains in busy scenes) with a green suite
    // throughout. This arm pins it: spy `createSphericalJoint`, settle a pin, then spawn one body per tick
    // for 30 ticks (the pump fires every tick). The create count must stay exactly 1 (the initial create)
    // and the bob's pose must stay within epsilon of settled. Green at the pre-fix state (the pump already
    // reuses handles correctly) — this is a pinning arm, not a bug-reproducing arm. The create-count half
    // is load-bearing: without it, the pose half alone can't detect a destroy+recreate that preserves the
    // rest pose.
    test("the pump reuses live joint handles across churn — create count is 1, pose holds", async () => {
        state = await buildTumble();

        const anchor = state.create();
        state.add(anchor, Body);
        Body.shape.set(anchor, ShapeKind.Box);
        Body.halfExtents.set(anchor, 0.5, 0.5, 0.5, 0);
        Body.pos.set(anchor, 0, 5, 0, 0);
        Body.mass.set(anchor, 0);

        const bob = state.create();
        state.add(bob, Body);
        Body.shape.set(bob, ShapeKind.Sphere);
        Body.halfExtents.set(bob, 0, 0, 0, 0.5);
        Body.pos.set(bob, 0, 4, 0, 0);
        Body.mass.set(bob, 1);

        const j = state.create();
        state.add(j, Joint);
        Joint.a.set(j, anchor);
        Joint.b.set(j, bob);
        Joint.rA.set(j, 0, -1, 0, 0);
        Joint.rB.set(j, 0, 0, 0, 0);
        Joint.stiffnessAng.set(j, 0);

        const createSpy = spyOn(Tumble.world!, "createSphericalJoint");
        try {
            // settle the pin
            for (let i = 0; i < 60; i++) state.step(Time.FIXED_DT);
            expect(createSpy).toHaveBeenCalledTimes(1);
            const settledY = Physics.backend!.readBody(bob)!.pos[1];

            // churn: one body per tick for 30 ticks — the pump fires every tick
            const spawn = (i: number): void => {
                const eid = state.create();
                state.add(eid, Body);
                Body.shape.set(eid, ShapeKind.Sphere);
                Body.halfExtents.set(eid, 0, 0, 0, 0.5);
                Body.pos.set(eid, i * 2, 10, 0, 0);
                Body.mass.set(eid, 1);
            };
            for (let i = 0; i < 30; i++) {
                spawn(i);
                state.step(Time.FIXED_DT);
            }
            expect(createSpy).toHaveBeenCalledTimes(1);
            const liveY = Physics.backend!.readBody(bob)!.pos[1];
            expect(Math.abs(liveY - settledY)).toBeLessThan(0.01);
        } finally {
            createSpy.mockRestore();
        }
    });
});
