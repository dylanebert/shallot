import { afterAll, afterEach, describe, expect, test } from "bun:test";
import { attach, stepFor } from "../../../tests/helpers";
import { State } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { Body, bodyTraits, Joint, jointTraits, Physics, Spring, springTraits } from "../physics";
import { Slab } from "../slab";
import { shutdown, type Joint as TumbleJoint } from "./engine";
import { TumblePlugin } from "./index";
import { stiffnessHertz, syncSet } from "./joints";

// The Spring/Joint → tumble mapping (joints.ts): the stiffness→hertz conversion law, the content-keyed
// diff semantics (kept slots), and the three behavioral mappings run end to end through TumblePlugin on
// a headless State — spring settles to the mg/k equilibrium (the conversion is load-bearing: a wrong
// hertz moves the rest pose), a spherical joint holds its pin length while swinging, a fixed joint holds
// the AUTHORED relative pose (the relRotation capture, not an axes-aligned snap).

describe("stiffnessHertz", () => {
    test("static anchor: m_eff is the dynamic side's mass", () => {
        // k = m·ω² ⇒ h = √(k/m)/2π
        expect(stiffnessHertz(100, 0, 8)).toBeCloseTo(Math.sqrt(100 / 8) / (2 * Math.PI), 12);
    });

    test("two dynamic endpoints: the reduced mass", () => {
        // m_eff = 2·2/(2+2) = 1
        expect(stiffnessHertz(100, 2, 2)).toBeCloseTo(Math.sqrt(100) / (2 * Math.PI), 12);
    });

    test("no dynamic endpoint (or no stiffness) yields 0 — the caller's skip signal", () => {
        expect(stiffnessHertz(100, 0, 0)).toBe(0);
        expect(stiffnessHertz(0, 1, 1)).toBe(0);
    });
});

describe("syncSet", () => {
    type Stub = { valid: boolean; destroyed: number };
    const stub = (): Stub => ({ valid: true, destroyed: 0 });
    const asJoint = (s: Stub): TumbleJoint =>
        ({
            isValid: () => s.valid,
            destroy: () => {
                s.destroyed++;
                s.valid = false;
            },
        }) as unknown as TumbleJoint;

    test("an unchanged def keeps its live joint; a removed def's joint is destroyed", () => {
        const live = new Map<string, TumbleJoint[]>();
        const a = stub();
        const b = stub();
        const made: Stub[] = [];
        const create = (): TumbleJoint => {
            const s = stub();
            made.push(s);
            return asJoint(s);
        };
        live.set("a", [asJoint(a)]);
        live.set("b", [asJoint(b)]);

        // re-upload with "a" kept and "b" dropped: no creates, only b destroyed
        syncSet(live, ["a"], (d) => d, create);
        expect(made.length).toBe(0);
        expect(a.destroyed).toBe(0);
        expect(b.destroyed).toBe(1);
        expect([...live.keys()]).toEqual(["a"]);
    });

    test("an invalidated handle (its body died) is recreated, not reused", () => {
        const live = new Map<string, TumbleJoint[]>();
        const dead = stub();
        dead.valid = false; // tumble cascaded the destroy — the handle is stale
        let created = 0;
        live.set("a", [asJoint(dead)]);

        syncSet(
            live,
            ["a"],
            (d) => d,
            () => {
                created++;
                return asJoint(stub());
            },
        );
        expect(created).toBe(1);
        expect(dead.destroyed).toBe(0); // invalid — never re-destroyed
    });

    test("duplicate defs each hold their own joint", () => {
        const live = new Map<string, TumbleJoint[]>();
        let created = 0;
        syncSet(
            live,
            ["a", "a"],
            (d) => d,
            () => {
                created++;
                return asJoint(stub());
            },
        );
        expect(created).toBe(2);
        expect(live.get("a")?.length).toBe(2);
    });

    test("a skipped def (create returns null) leaves no entry", () => {
        const live = new Map<string, TumbleJoint[]>();
        syncSet(
            live,
            ["a"],
            (d) => d,
            () => null,
        );
        expect(live.size).toBe(0);
    });
});

// ── behavioral: the mapping through TumblePlugin on a headless State ─────────────────────────────────

interface SceneBody {
    pos: [number, number, number];
    quat?: [number, number, number, number];
    half?: [number, number, number];
    mass: number;
}

// the live State of the current behavioral test — afterEach disposes it so a FAILED assert can't leak
// the installed backend (the single-backend guard would then throw in the next test's warm)
let liveState: State | null = null;
afterEach(() => {
    if (liveState) TumblePlugin.dispose?.(liveState);
    liveState = null;
});

// Release the multithreaded worker pool warm()/init() boots (the wasm kernel is a process singleton) at
// file teardown, so its solve path doesn't leak into sibling engine test files that assume single-thread.
afterAll(shutdown);

async function build(bodies: SceneBody[]): Promise<{ state: State; eids: number[] }> {
    clear();
    const state = new State();
    liveState = state;
    register("body", Body, bodyTraits);
    register("spring", Spring, springTraits);
    register("joint", Joint, jointTraits);
    Slab.collect();
    TumblePlugin.initialize?.(state);
    await TumblePlugin.warm?.(state);
    attach(state, TumblePlugin);
    const eids: number[] = [];
    for (const b of bodies) {
        const eid = state.create();
        state.add(eid, Body);
        const h = b.half ?? [0.5, 0.5, 0.5];
        Body.halfExtents.set(eid, h[0], h[1], h[2], 0);
        Body.pos.set(eid, b.pos[0], b.pos[1], b.pos[2], 0);
        const q = b.quat ?? [0, 0, 0, 1];
        Body.quat.set(eid, q[0], q[1], q[2], q[3]);
        Body.mass.set(eid, b.mass);
        eids.push(eid);
    }
    return { state, eids };
}

function addSpring(state: State, a: number, b: number, stiffness: number, rest: number): void {
    const e = state.create();
    state.add(e, Spring);
    Spring.a.set(e, a);
    Spring.b.set(e, b);
    Spring.stiffness.set(e, stiffness);
    Spring.rest.set(e, rest);
}

function addJoint(
    state: State,
    a: number,
    b: number,
    rA: [number, number, number],
    rB: [number, number, number],
    stiffnessAng = 0,
): void {
    const e = state.create();
    state.add(e, Joint);
    Joint.a.set(e, a);
    Joint.b.set(e, b);
    Joint.rA.set(e, rA[0], rA[1], rA[2], 0);
    Joint.rB.set(e, rB[0], rB[1], rB[2], 0);
    Joint.stiffnessAng.set(e, stiffnessAng);
}

describe("tumble constraint mapping", () => {
    test("a spring settles at the mg/k equilibrium — the stiffness→hertz law holds", async () => {
        // anchor at y=10, block (mass 8) hung on a rest-4 stiffness-100 spring: equilibrium extension
        // mg/k = 8·10/100 = 0.8 past rest ⇒ y = 10 − 4 − 0.8 = 5.2. Spawned AT rest length (y=6), the
        // critically-damped transient (ω = √(k/m) ≈ 3.54 rad/s) decays to ~2e-8 of the initial 0.8 m
        // displacement within 5 s; the residual band is solver slop (~5e-3), so ±0.02 is derived, not tuned.
        const { state, eids } = await build([
            { pos: [0, 10, 0], mass: 0, half: [0.1, 0.1, 0.1] },
            { pos: [0, 6, 0], mass: 8 },
        ]);
        addSpring(state, eids[0], eids[1], 100, 4);
        stepFor(state, 5);
        const body = Physics.backend?.readBody(eids[1]);
        expect(body).not.toBeNull();
        expect(Math.abs((body?.pos[1] ?? 0) - 5.2)).toBeLessThan(0.02);
    });

    test("a spherical joint holds its pin length while the bob swings", async () => {
        // bob (mass 1) hangs 2 m off a static pivot by a pin at the ROD END (rA at the pivot, rB the
        // bob-local offset back to it — pinning the bob's own center would just let it spin in place).
        // Released horizontal, it swings (y drops) while |bob − pivot| stays the rod length: the anchor
        // coincidence pins bob + rot(q, rB) AT the pivot, so the center is always |rB| away. Pin
        // tolerance: a 60 Hz soft constraint stretches g/ω² ≈ 7e-5 under one g — the ±0.02 band is
        // dominated by solver slop, not physics.
        const { state, eids } = await build([
            { pos: [0, 10, 0], mass: 0, half: [0.1, 0.1, 0.1] },
            { pos: [2, 10, 0], mass: 1, half: [0.25, 0.25, 0.25] },
        ]);
        addJoint(state, eids[0], eids[1], [0, 0, 0], [-2, 0, 0]);
        stepFor(state, 1);
        const bob = Physics.backend?.readBody(eids[1]);
        expect(bob).not.toBeNull();
        const dx = (bob?.pos[0] ?? 0) - 0;
        const dy = (bob?.pos[1] ?? 0) - 10;
        const dz = (bob?.pos[2] ?? 0) - 0;
        expect(Math.abs(Math.sqrt(dx * dx + dy * dy + dz * dz) - 2)).toBeLessThan(0.02);
        expect(bob?.pos[1] ?? 10).toBeLessThan(9.5); // it actually swung down
    });

    test("a fixed joint holds the AUTHORED relative pose, not an axes-aligned snap", async () => {
        // the arm spawns rotated 30° about z; a fixed joint must hold that spawn orientation under
        // gravity (relRotation captures it into the weld frame — a frame-q of identity would instead
        // torque the arm back to the base's axes).
        const s30 = Math.sin(Math.PI / 12);
        const c30 = Math.cos(Math.PI / 12);
        const { state, eids } = await build([
            { pos: [0, 2, 0], mass: 0 },
            { pos: [1.5, 2.5, 0], mass: 1, half: [0.25, 0.25, 0.25], quat: [0, 0, s30, c30] },
        ]);
        addJoint(state, eids[0], eids[1], [1.5, 0.5, 0], [0, 0, 0], Number.POSITIVE_INFINITY);
        stepFor(state, 2);
        const arm = Physics.backend?.readBody(eids[1]);
        expect(arm).not.toBeNull();
        expect(Math.abs((arm?.pos[0] ?? 0) - 1.5)).toBeLessThan(0.02);
        expect(Math.abs((arm?.pos[1] ?? 0) - 2.5)).toBeLessThan(0.02);
        // orientation held: |dot(q, q0)| ≈ 1 (sign-insensitive quat identity)
        const dot = (arm?.quat[2] ?? 0) * s30 + (arm?.quat[3] ?? 0) * c30;
        expect(Math.abs(dot)).toBeGreaterThan(0.9999);
    });

    test("a negative stiffnessAng must not produce a rigid weld (warn+skip, matching the spring path)", async () => {
        // stiffnessAng: -1 → stiffnessHertz returns 0 (stiffness <= 0) → angularHertz 0 →
        // weldJoint maps angularHertz === 0 → sim.constraintSoftness = RIGID. The weakest requested
        // angular stiffness yields the stiffest joint, silently — while the spring path warns+skips
        // the identical non-positive input (joints.ts createSpring). The swap-parity rule the file
        // states ("settle-to-equilibrium is the behavior that matches across the swap") admits
        // warn+skip: a non-positive stiffness has no equilibrium to settle to, so the matching
        // behavior is to skip (as the spring path does), not to silently substitute a different
        // constraint. This arm asserts the warn+skip direction: no joint is created, so the body
        // falls freely under gravity instead of being held at the authored pose.
        const s30 = Math.sin(Math.PI / 12);
        const c30 = Math.cos(Math.PI / 12);
        const { state, eids } = await build([
            { pos: [0, 2, 0], mass: 0 },
            { pos: [1.5, 2.5, 0], mass: 1, half: [0.25, 0.25, 0.25], quat: [0, 0, s30, c30] },
        ]);
        addJoint(state, eids[0], eids[1], [1.5, 0.5, 0], [0, 0, 0], -1);
        stepFor(state, 2);
        const arm = Physics.backend?.readBody(eids[1]);
        expect(arm).not.toBeNull();
        // warn+skip: no joint created, body falls freely — not held at the authored y=2.5
        expect(arm?.pos[1] ?? 2.5).toBeLessThan(2.0);
    });

    test("an intermediate stiffnessAng settles rather than oscillates (amplitude decay over 40 fixed ticks)", async () => {
        // angularDampingRatio: 0 with angularHertz > 0 gives makeSoft(hertz, 0, h) — an undamped
        // angular spring that rings, contradicting the file's stated swap-parity rule (dampingRatio: 1
        // because "settle-to-equilibrium is the behavior that matches across the swap"). This arm
        // asserts the settle: after N fixed ticks the body's angular displacement over a measurement
        // window is below tolerance — a damped spring has settled, an undamped one is still ringing.
        //
        // Setup: a body hangs from a static pivot by a 0.5 m offset pin (rB = [0.5, 0, 0]). Gravity
        // creates a torque around the pin; the angular spring (stiffnessAng = 1000 → angularHertz ≈
        // 5.03 Hz, ω ≈ 31.6 rad/s, period T ≈ 12 ticks) resists. Critically damped (the fix), the
        // body settles to equilibrium within ~30 ticks (observed: < 0.05° per 10-tick window past
        // tick 30). Undamped (the bug), it oscillates at T ≈ 12 ticks (observed: 6.4° per 10-tick
        // window at tick 30–40). The arm measures the angular displacement between tick 30 and tick
        // 40 (a 10-tick ≈ 5T/6 window): a settled body moves < tolerance, a ringing body moves ~6.4°.
        //
        // N = 40 ticks (30 settling + 10 measurement), tolerance = 0.02 rad (~1.1°). The minimum N
        // that witnesses the ring: 30 ticks is the critical-damping settling time (observed from the
        // damped response), and 10 ticks (≈ 5T/6) is the shortest window in which a ringing body
        // moves past tolerance while a settled one does not.
        const { state, eids } = await build([
            { pos: [0.5, 10, 0], mass: 0, half: [0.1, 0.1, 0.1] },
            { pos: [0, 10, 0], mass: 1, half: [0.25, 0.25, 0.25] },
        ]);
        addJoint(state, eids[0], eids[1], [0, 0, 0], [0.5, 0, 0], 1000);
        stepFor(state, 0.5); // 30 ticks — settling window
        const q1 = Physics.backend?.readBody(eids[1])?.quat ?? [0, 0, 0, 1];
        stepFor(state, 1 / 6); // 10 ticks — measurement window (≈ 5T/6)
        const q2 = Physics.backend?.readBody(eids[1])?.quat ?? [0, 0, 0, 1];
        const dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];
        const angle = 2 * Math.acos(Math.min(1, Math.abs(dot)));
        // N = 40 ticks, tolerance = 0.02 rad (~1.1°): a settled body moves < 0.02, a ringing body
        // moves ~0.11 rad in the 10-tick window.
        expect(angle).toBeLessThan(0.02);
    });
});
