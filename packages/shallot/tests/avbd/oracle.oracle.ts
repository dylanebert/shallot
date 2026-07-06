import { describe, expect, test } from "bun:test";
import { SPECULATIVE_DISTANCE } from "./collide";
import { fixtureSolver, framePos, loadFixture } from "./fixtures";
import { joint } from "./joint";
import { COLLISION_MARGIN, PENALTY_MIN } from "./manifold";
import { length, type Quat, rotate, scale, sub, transform, type Vec3 } from "./math";
import { type Body, body, capsule, massOf, sphere } from "./rigid";
import { makeSolver, type Schedule, type Solver, step } from "./solver";
import { spring } from "./spring";

// The phase-0 standing gate: the TS oracle is the executable AVBD spec. Tolerances are derived
// + checked (the values in the comments were measured against the oracle, not guessed), and the
// gate is laid out by the ladder in scratch.md "AVBD rebuild → Gate ladder": closed-form gates
// (tightest, no reference), fixture reproduction (deterministic parts only — the rest state +
// the energy invariant, never a long-horizon trajectory match), and the scheduler bridge.
//
// Why no full-trajectory match on chaotic scenes: two float implementations (f64 oracle, f32 C++)
// can't bit-match a stacking sim over 600 frames — contact-set membership flips at the margin and
// trajectories separate exponentially (Lyapunov divergence is physics, not a bug). So each thing
// is checked where it's deterministic: the rest state (same fixed point), and energy conservation
// (premise-valid every frame, settled or not — a blow-up is an energy injection).

// total mechanical energy in the solver's own convention (world-frame diagonal inertia, matching
// the reference's MAng): KE_lin + KE_ang + PE. A dropped pile starts at rest at max height, so
// E(0) is the supremum; a dissipative implicit solver only loses energy.
function energy(s: Solver): number {
    let E = 0;
    for (const b of s.bodies) {
        if (b.mass <= 0) continue;
        const [vx, vy, vz] = b.velLin;
        const [wx, wy, wz] = b.velAng;
        E += 0.5 * b.mass * (vx * vx + vy * vy + vz * vz);
        E += 0.5 * (b.moment[0] * wx * wx + b.moment[1] * wy * wy + b.moment[2] * wz * wz);
        E += b.mass * -s.params.gravity * b.posLin[1];
    }
    return E;
}

const maxSpeed = (s: Solver): number => {
    let m = 0;
    for (const b of s.bodies) m = Math.max(m, length(b.velLin));
    return m;
};

// Energy is recomputed each frame (no accumulation), so the only error is the round-off of one
// N-term f64 sum ≈ N·ε_machine ≈ 3e-14 relative; a numerical instability injects O(1). 1e-6 is a
// derived guard band 8 orders above the noise floor and 4 below a real failure. Measured: 0.
const ENERGY_TOL = 1e-6;

// The C++ fixtures run 600 frames, but the oracle only checks deterministic invariants: the energy
// bound (a per-step invariant — instability shows in the first frames, not the 500th), the early
// trajectory band, and the settled rest state. The non-chaotic scenes all settle well inside this
// cap, and the `maxSpeed < 0.05` rest assert self-guards it: a scene not yet at rest fails loudly,
// so the cap can't silently weaken a settle gate. Replaying the full 600 just re-steps a static pile.
const REPLAY = 300;

describe("AVBD oracle — closed-form gates", () => {
    test("free-fall matches the exact discrete symplectic-Euler trajectory", () => {
        // no ground ⇒ no contact: the body lands on its inertial target each step, giving
        // v_n = g·dt·n and x_n = x0 + g·dt²·n(n+1)/2 exactly. Measured max error 1.6e-12.
        const dt = 1 / 60;
        const g = -10;
        const y0 = 10;
        const s = makeSolver([body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, y0, 0])], {
            gravity: g,
            dt,
        });
        let maxErr = 0;
        for (let n = 1; n <= 120; n++) {
            step(s);
            const yExact = y0 + (g * dt * dt * (n * (n + 1))) / 2;
            const vExact = g * dt * n;
            maxErr = Math.max(
                maxErr,
                Math.abs(s.bodies[0].posLin[1] - yExact),
                Math.abs(s.bodies[0].velLin[1] - vExact),
            );
        }
        expect(maxErr).toBeLessThan(1e-9);
    });

    test("substeps = N is exactly N sub-steps of h = dt/N (the small-steps definition)", () => {
        // one step at substeps=N must equal N manual steps at dt/N with substeps=1 — every dt-bearing
        // term (inertial init, M/h², BDF1 velocity, the velocity-sweep band) uses h, and the manifold
        // map persists across sub-steps exactly as across frames, so the two paths are the same f64 ops.
        // A 3-box stack on the ground exercises contacts + multi-pair warmstart carry between sub-steps.
        const scene = () => [
            body([20, 1, 20], 0, 0.5, [0, 0, 0]),
            body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.0, 0]),
            body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0.02, 1.97, 0]),
        ];
        const N = 4;
        const internal = makeSolver(scene(), { substeps: N });
        const external = makeSolver(scene(), { substeps: 1, dt: 1 / 60 / N });
        for (let f = 0; f < 20; f++) {
            step(internal);
            for (let sub = 0; sub < N; sub++) step(external);
        }
        let maxErr = 0;
        for (let i = 1; i < 3; i++) {
            const a = internal.bodies[i];
            const b = external.bodies[i];
            maxErr = Math.max(
                maxErr,
                length(sub(a.posLin, b.posLin)),
                length(sub(a.velLin, b.velLin)),
            );
        }
        expect(maxErr).toBeLessThan(1e-12); // same f64 op sequence ⇒ bit-identical to round-off
    });

    test("resting box penetration = mg/(nc·K) (penalty layer, α=0)", () => {
        // Penalty layer with α=0: at rest the constraint is C = C0 and the dq terms vanish, so the
        // nc contact forces balance gravity as nc·K·|C0_n| = mg. With C0_n = (y−1) + MARGIN, the
        // penetration past the (1−MARGIN) margin-rest is mg/(nc·K). nc is read back, not assumed.
        const g = -10;
        const mg = 10; // m = 1, |g| = 10
        const restY = 1 - COLLISION_MARGIN; // the box-touching height the margin holds it at
        const rest = (K: number) => {
            const s = makeSolver(
                [
                    body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                    body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.5, 0]),
                ],
                { layer: "penalty", alpha: 0, penaltyStiffness: K, gravity: g },
            );
            for (let f = 0; f < 300; f++) step(s);
            const m = [...s.manifolds.values()][0];
            return {
                y: s.bodies[1].posLin[1],
                nc: m ? m.contacts.length : 0,
                speed: length(s.bodies[1].velLin),
            };
        };
        const a = rest(1000);
        const b = rest(2000);

        expect(a.speed).toBeLessThan(1e-6); // premise: the box is at rest
        expect(a.nc).toBeGreaterThan(0);
        expect(restY - a.y).toBeCloseTo(mg / (a.nc * 1000), 6); // measured rel error 2e-14
        expect(restY - b.y).toBeCloseTo((restY - a.y) / 2, 6); // mg/k signature: doubling K halves penetration
    });

    test("dual layer ramps the penalty to hold the box near the margin, not the seed floor", () => {
        // The Phase-2 gate (roadmap "resting penetration → 0"): λ accumulation + the within-frame
        // penalty ramp let the contact force balance gravity without a large fixed stiffness, so the
        // box rests near the margin where a *fixed* penalty at the same seed (PENALTY_MIN = 1) sinks
        // mg/(nc·1) ≈ O(1) deep (clean through the ground). The dual layer cold-starts λ each frame
        // (no cross-frame warmstart until Phase 3), so it needs the higher betaLin = 1e5 the engine
        // ships — the canonical 1e4 is calibrated for warmstart and under-ramps from cold here.
        const g = -10;
        const restY = 1 - COLLISION_MARGIN;
        const settle = (params: Parameters<typeof makeSolver>[1]) => {
            const s = makeSolver(
                [
                    body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                    body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.5, 0]),
                ],
                params,
            );
            for (let f = 0; f < 300; f++) step(s);
            return { y: s.bodies[1].posLin[1], speed: length(s.bodies[1].velLin) };
        };
        const dual = settle({
            layer: "dual",
            penaltyStiffness: PENALTY_MIN,
            betaLin: 1e5,
            alpha: 0.99,
            gravity: g,
        });
        const seedFloor = settle({
            layer: "penalty",
            penaltyStiffness: PENALTY_MIN,
            alpha: 0.99,
            gravity: g,
        });
        console.log(
            `[oracle] dual(b=1e5) rest y ${dual.y.toFixed(5)} (pen ${(restY - dual.y).toExponential(3)}), ` +
                `fixed-PENALTY_MIN rest y ${seedFloor.y.toFixed(3)} (pen ${(restY - seedFloor.y).toFixed(2)})`,
        );

        // dual rests AT the margin — |penetration| an order below the box scale, so the box center sits
        // in (restY − 1e-2, restY + 1e-2) = [0.98, 1.0]: between the mg/k penetration and the bare touch.
        // The speculative band (Phase 4.8.3) catches the box at the band, so it settles a hair above the
        // bare margin rest (measured −2.3e-4 past restY) rather than the overlap-only ~2e-3 below — either
        // way at the margin, not the deep seed floor, and never hovering above the touch (center > 1.0).
        const dualPen = restY - dual.y;
        expect(Math.abs(dualPen)).toBeLessThan(1e-2);
        // the fixed-PENALTY_MIN baseline sinks O(1) deep — the dual ramp recovers ~2 orders
        expect(restY - seedFloor.y).toBeGreaterThan(1);
    });

    test("static friction holds on a 30° ramp iff μ ≥ tan 30°", () => {
        // Coulomb cone: a box on a θ ramp stays static iff μ ≥ tan θ. Both surfaces friction = μ
        // (μ_eff = √(μ·μ) = μ). Warm up 60 frames to land on the ramp, then measure the slide —
        // below threshold it keeps sliding (kinetic), above it comes to rest. Threshold tan30 = 0.577.
        const angle = Math.PI / 6;
        const q: Quat = [0, 0, Math.sin(angle / 2), Math.cos(angle / 2)];
        const normal = rotate(q, [0, 1, 0]);
        const slideAfterLanding = (mu: number) => {
            const ramp = body([40, 1, 24], 0, mu, [0, 6, 0], [0, 0, 0], q);
            const pos: Vec3 = [
                ramp.posLin[0] + normal[0] * 1.05,
                ramp.posLin[1] + normal[1] * 1.05,
                ramp.posLin[2] + normal[2] * 1.05,
            ];
            const s = makeSolver(
                [
                    body([100, 1, 100], 0, mu, [0, 0, 0]),
                    ramp,
                    body([1, 1, 1], massOf([1, 1, 1], 1), mu, pos),
                ],
                { gravity: -10 },
            );
            for (let f = 0; f < 60; f++) step(s);
            const landed = [...s.bodies[2].posLin] as Vec3;
            for (let f = 0; f < 200; f++) step(s);
            return {
                slid: length(sub(s.bodies[2].posLin, landed)),
                endSpeed: length(s.bodies[2].velLin),
            };
        };

        const below = slideAfterLanding(0.45); // < tan30 → kinetic
        const above = slideAfterLanding(0.75); // > tan30 → static
        expect(below.slid).toBeGreaterThan(2); // measured 4.8
        expect(below.endSpeed).toBeGreaterThan(0.3); // measured 1.5 — still moving
        expect(above.slid).toBeLessThan(0.5); // measured 0.33
        expect(above.endSpeed).toBeLessThan(0.05); // measured 0.003 — at rest
    });

    test("speculative contact stops a fast in-band box at the surface (no penetration pop / tunnel)", () => {
        // Phase 4.8.3: the SAT generates a contact while the boxes are still separated by up to
        // SPECULATIVE_DISTANCE, carrying the true +gap in c0. The repulsion-only normal constraint then
        // limits the approach to close exactly that gap, so a body within the band at frame start lands
        // AT the surface in one step (Box2D / Firth speculative contacts). Overlap-only CD generates no
        // contact for a separated pair, so the box tunnels ~1.7 m below the ground in a single 1/60 s
        // step — the red this gate turns green (revert the testAxis abort to SPECULATIVE_DISTANCE to see).
        const g = -10;
        const touchingY = 1.0; // box-center height where the box (half 0.5) face meets the ground top (0.5)
        const gap = SPECULATIVE_DISTANCE * 0.75; // 0.03 — inside the band, so the contact fires
        const v = 100; // downward; v·dt ≈ 1.67 m ≫ the box, so overlap-only CD tunnels it in one step

        const s = makeSolver(
            [
                body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, touchingY + gap, 0], [0, -v, 0]),
            ],
            { gravity: g },
        );
        step(s);

        const box = s.bodies[1];
        const contacts = [...s.manifolds.values()][0]?.contacts.length ?? 0;
        // the speculative contact fired and caught the box AT the band: it stays near where it entered
        // (measured center ≈ 1.0296, gap barely closed under the α=0.99 stabilization) and does NOT pass
        // through. Overlap-only CD lands the center at ~ −0.64, deep through the ground — so > touchingY
        // − 0.05 is the tunnel discriminator; over later frames it settles to the margin rest (~0.99).
        expect(contacts).toBeGreaterThan(0);
        expect(box.posLin[1]).toBeGreaterThan(touchingY - 0.05); // > 0.95 — caught, not tunnelled (~ −0.64)
        expect(box.posLin[1]).toBeLessThan(touchingY + gap); // only closes the entry gap, never lifts off
        // the 100 m/s approach is killed to near zero in one step — the speculative contact absorbs it
        // (lands at contact, no penetration pop). measured |vy| ≈ 0.024 m/s; 1 m/s is the derived ceiling
        // (≪ the 100 m/s it entered with, so the box can't keep falling through the ground).
        expect(Math.abs(box.velLin[1])).toBeLessThan(1);
    });

    test("velocity sweep catches a fast box beyond the static band; a static box at the same gap is untouched", () => {
        // Phase 4.8.4: the static band (4.8.3) only generates a contact within SPECULATIVE_DISTANCE at
        // frame start, so a body crossing the whole contact between frames (v·dt ≫ the band) still
        // tunnels. The velocity sweep extends the per-axis SAT band by the closing displacement
        // max(0, dot(dRel, n)) (dRel = (vA−vB)·dt), so the frame-start SAT generates the swept contact
        // and the existing repulsion-only constraint limits the approach (Box2D/Bullet/Firth speculative
        // contacts). It is velocity-GATED: a static body at the same gap stays separated (the swept band
        // degenerates to 4.8.3 at vRel=0). Red without the sweep: gap 0.5 ≫ the 0.04 band ⇒ 0 contacts ⇒
        // the box free-falls ~1.67 m, ~ −0.17 deep through the ground spanning [−0.5, 0.5].
        const g = -10;
        const touchingY = 1.0; // box-center height where the box (half 0.5) face meets the ground top (0.5)
        const gap = 0.5; // ≫ SPECULATIVE_DISTANCE (0.04): the static band alone generates nothing
        const v = 100; // v·dt ≈ 1.67 m ≫ the gap, so the box would tunnel in one step without the sweep

        const s = makeSolver(
            [
                body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, touchingY + gap, 0], [0, -v, 0]),
            ],
            { gravity: g },
        );
        step(s);
        const box = s.bodies[1];
        const contacts = [...s.manifolds.values()][0]?.contacts.length ?? 0;
        // the swept contact fired and arrested the box: it stays near where it entered (closes only the
        // (1−α) stabilization fraction this step, ~0.005 m) well above the ground — it did NOT tunnel.
        expect(contacts).toBeGreaterThan(0);
        expect(box.posLin[1]).toBeGreaterThan(touchingY - 0.05); // caught above the surface, not tunnelled
        expect(box.posLin[1]).toBeLessThan(touchingY + gap + 1e-3); // only closes toward contact, never rises
        // the 100 m/s approach is killed in one step (measured |vy| ≈ 0.3 m/s ≪ 100 — it can't keep falling)
        expect(Math.abs(box.velLin[1])).toBeLessThan(1);

        // velocity-gated: the SAME 0.5 gap with a STATIC box is past the band → no swept contact (4.8.3).
        const stat = makeSolver(
            [
                body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, touchingY + gap, 0]),
            ],
            { gravity: g },
        );
        step(stat);
        const statContacts = [...stat.manifolds.values()][0]?.contacts.length ?? 0;
        expect(statContacts).toBe(0); // separated past the band, no contact generated
    });
});

describe("AVBD oracle — reproduces the C++ contact fixtures (sequential)", () => {
    // Non-chaotic settling scenes: the deterministic parts are the rest state (same fixed point as
    // C++) and the energy invariant. The transient is tracked only within the f32-vs-f64 round-off
    // band (an order below the box scale), not pinned frame-by-frame.
    for (const scene of ["ground", "two-boxes", "stack", "stack-ratio"]) {
        test(`${scene}: settles to the C++ rest state, energy non-increasing`, () => {
            const fx = loadFixture("canonical", scene);
            const s = fixtureSolver(fx);
            const E0 = energy(s);
            let maxExcess = 0;
            let maxTransient = 0;
            for (let f = 0; f < Math.min(fx.frames.length, REPLAY); f++) {
                step(s);
                maxExcess = Math.max(maxExcess, (energy(s) - E0) / Math.abs(E0));
                let fm = 0;
                for (let i = 0; i < s.bodies.length; i++) {
                    fm = Math.max(fm, length(sub(s.bodies[i].posLin, framePos(fx.frames[f], i))));
                }
                maxTransient = Math.max(maxTransient, fm);
            }
            expect(maxExcess).toBeLessThan(ENERGY_TOL);
            // f32(C++) vs f64(oracle) round-off band: an order below the box scale (1), above the
            // observed ~1e-3 impact-moment divergence. Catches a gross trajectory bug (O(1)).
            expect(maxTransient).toBeLessThan(1e-2);
            // premise: the scene settles (residual speed ≪ the 10 m/s fall speed). Then the pose
            // matches the C++ at the same frame to f32 equilibrium precision — both at the stable
            // rest fixed point (maxSpeed gate), where the f32/f64 paths reconverge tight (measured
            // < 1e-4), an order below the chaotic-impact transient band above.
            expect(maxSpeed(s)).toBeLessThan(0.05);
            const last = fx.frames[Math.min(fx.frames.length, REPLAY) - 1];
            let restErr = 0;
            for (let i = 0; i < s.bodies.length; i++) {
                restErr = Math.max(restErr, length(sub(s.bodies[i].posLin, framePos(last, i))));
            }
            expect(restErr).toBeLessThan(1e-3);
        });
    }

    // the suite's heaviest fixture (80 frames of high-valence contact solve) runs ~5.5s on a slow
    // machine, over bun's 5s default — a wall-time budget, not a correctness bound
    test("pyramid: energy never increases through the collapse (high-valence stability)", () => {
        // The high-valence stress (a base body carries several simultaneous contacts), checked at
        // the runtime-budget params (4 iters) — the harder stability regime and 2.5× cheaper. No
        // settle/trajectory claim: the pyramid never fully settles, so the premise is "it collides"
        // (speed peaks in free-fall then drops as contacts engage), verified in-test.
        const fx = loadFixture("budget", "pyramid");
        const s = fixtureSolver(fx);
        const E0 = energy(s);
        let maxExcess = 0;
        let peak = 0;
        let collided = false;
        for (let f = 0; f < 80; f++) {
            step(s);
            maxExcess = Math.max(maxExcess, (energy(s) - E0) / Math.abs(E0));
            const sp = maxSpeed(s);
            peak = Math.max(peak, sp);
            if (peak > 5 && sp < peak * 0.3) collided = true;
        }
        expect(peak).toBeGreaterThan(5); // free-fall peak (measured ~11)
        expect(collided).toBe(true); // contacts engaged and dissipated the peak (measured @ frame 68)
        expect(maxExcess).toBeLessThan(ENERGY_TOL);
    }, 15_000);

    test("dynamic-friction: tracks the C++ slide early, energy never increases", () => {
        // Starts with kinetic energy (boxes launched at vx=10); friction can only remove it. Energy
        // non-increasing holds the whole run, but it's a weak gate — a near-frictionless slide passes
        // it too (the kinetic-friction bug did). So also pin the early slide against the C++ where it's
        // deterministic (before the boxes diverge): the bug fades kinetic friction, so the boxes slide
        // far past the C++ within ~10 frames (measured ≥ 0.1 m by frame 8 with the bug; ≤ 1e-3 here).
        const fx = loadFixture("canonical", "dynamic-friction");
        const s = fixtureSolver(fx);
        const E0 = energy(s);
        let maxExcess = 0;
        let earlyErr = 0;
        // no settle assert here — only the energy bound + the early (f<100) trajectory band, so stop
        // once the early window is past plus a margin for the energy invariant. The tail is a static slide.
        for (let f = 0; f < 150; f++) {
            step(s);
            maxExcess = Math.max(maxExcess, (energy(s) - E0) / Math.abs(E0));
            if (f < 100)
                for (let i = 0; i < s.bodies.length; i++)
                    earlyErr = Math.max(
                        earlyErr,
                        length(sub(s.bodies[i].posLin, framePos(fx.frames[f], i))),
                    );
        }
        expect(maxExcess).toBeLessThan(ENERGY_TOL);
        // the first 100 frames (the slide, before the f32/f64 paths separate) track the C++ tightly —
        // measured < 1e-3; the friction-ramp bug drives it past 0.1 within 8 frames.
        expect(earlyErr).toBeLessThan(1e-3);
    });
});

describe("AVBD oracle — friction + rotation reproduce the C++ fixtures", () => {
    // The demo's own box scenes are all axis-aligned (zero rotation) and its friction scenes slide off
    // the 100-wide ground and free-fall (never settle). These three harness scenes (harness-dense.cpp,
    // mirroring tests/avbd/corpus.ts) cover the dynamics that were never gold-checked: kinetic friction
    // that stops a box, a box tipping vertex→face, and a stack toppling. Each is gated where it's
    // deterministic (scratch.md "Gate ladder"): a clean slide / a contained tip track the C++ over the
    // whole run; a chaotic topple only early + on the statistical band.

    test("friction-settle: tracks the C++ slide-to-rest + lower-μ-slides-farther", () => {
        // four boxes launched at 5 m/s, decelerated to rest by friction within the ground. NOT chaotic
        // (a straight slide to a stop), so the f64 oracle tracks the f32 C++ over the WHOLE run. This is
        // the gate that catches the kinetic-friction ramp bug: gating the tangent-penalty ramp on the
        // post-clamp force ramps a sliding contact's penalty unboundedly, fading friction to ~0 so the
        // box never stops and slides ~100 m off the ground (vs the C++ stop). Measured max err 6.6e-5.
        const fx = loadFixture("canonical", "friction-settle");
        const s = fixtureSolver(fx);
        let maxErr = 0;
        for (let f = 0; f < Math.min(fx.frames.length, REPLAY); f++) {
            step(s);
            for (let i = 0; i < s.bodies.length; i++)
                maxErr = Math.max(
                    maxErr,
                    length(sub(s.bodies[i].posLin, framePos(fx.frames[f], i))),
                );
        }
        // 1e-3 is a derived f32-vs-f64 band an order above the measured 6.6e-5, far below the box scale.
        expect(maxErr).toBeLessThan(1e-3);
        expect(maxSpeed(s)).toBeLessThan(0.05); // came to rest (the bug leaves it sliding ~4.6 m/s)
        // friction signature: boxes 1..4 carry μ 0.2..0.8; lower μ slides farther (monotonic stop). A
        // faded kinetic friction would let them all slide the same (off the edge), breaking the order.
        const slid = (i: number): number => Math.abs(s.bodies[i].posLin[0] - -6);
        expect(slid(1)).toBeGreaterThan(slid(2));
        expect(slid(2)).toBeGreaterThan(slid(3));
        expect(slid(3)).toBeGreaterThan(slid(4));
    });

    test("corner-rest: tracks the C++ vertex→face tip, energy non-increasing, settles flat", () => {
        // a unit box tilted 45° about x then z, dropped — lands on a vertex, tips through an edge to a
        // face. The tip is contained (one box, no scatter), so it tracks the C++ the whole run (measured
        // 2.5e-5). The kinetic-friction bug also corrupts this (the box slides during the tip): measured
        // 2.3e-2 with the bug, 2.5e-5 fixed.
        const fx = loadFixture("canonical", "corner-rest");
        const s = fixtureSolver(fx);
        const E0 = energy(s);
        let maxErr = 0;
        let maxExcess = 0;
        for (let f = 0; f < Math.min(fx.frames.length, REPLAY); f++) {
            step(s);
            maxExcess = Math.max(maxExcess, (energy(s) - E0) / Math.abs(E0));
            for (let i = 0; i < s.bodies.length; i++)
                maxErr = Math.max(
                    maxErr,
                    length(sub(s.bodies[i].posLin, framePos(fx.frames[f], i))),
                );
        }
        expect(maxErr).toBeLessThan(1e-3);
        // dropped from rest ⇒ E0 is the supremum; the penalty spring loads on the corner impact but
        // never lifts the box above the drop, so excess stays ≤ a small slack (1e-2, the impact band).
        expect(maxExcess).toBeLessThan(1e-2);
        // settled flat on a face at the box-touching rest height (~margin below 1.0), at rest
        expect(maxSpeed(s)).toBeLessThan(0.05);
        expect(s.bodies[1].posLin[1]).toBeGreaterThan(0.9);
        expect(s.bodies[1].posLin[1]).toBeLessThan(1.05);
    });

    test("leaning: early topple tracks the C++; energy non-increasing; settles flat on the ground", () => {
        // a 5-box stack offset 0.4 in +x — the upper COM is past the base, so it topples, scatters, and
        // settles. A topple is the chaotic regime, so gate the early fall against the C++ (deterministic
        // before the scatter — measured < 1e-3 through frame 125) and the tail on the statistical band:
        // energy never exceeds the drop supremum, comes to rest, every box ends on the ground.
        const fx = loadFixture("canonical", "leaning");
        const s = fixtureSolver(fx);
        const E0 = energy(s);
        let earlyErr = 0;
        let maxExcess = 0;
        let finite = true;
        for (let f = 0; f < Math.min(fx.frames.length, REPLAY); f++) {
            step(s);
            maxExcess = Math.max(maxExcess, (energy(s) - E0) / Math.abs(E0));
            if (!s.bodies.every((b) => b.posLin.every(Number.isFinite))) finite = false;
            if (f < 100)
                for (let i = 0; i < s.bodies.length; i++)
                    earlyErr = Math.max(
                        earlyErr,
                        length(sub(s.bodies[i].posLin, framePos(fx.frames[f], i))),
                    );
        }
        expect(finite).toBe(true);
        expect(earlyErr).toBeLessThan(1e-3); // the free-fall + first contacts, before the scatter
        expect(maxExcess).toBeLessThan(1e-2); // dissipative through the topple (measured ~0)
        expect(maxSpeed(s)).toBeLessThan(0.05); // settles
        // every dynamic box ends resting on the ground (toppled flat, none launched away)
        for (let i = 1; i < s.bodies.length; i++) {
            expect(s.bodies[i].posLin[1]).toBeGreaterThan(0.9);
            expect(s.bodies[i].posLin[1]).toBeLessThan(1.1);
        }
    });
});

describe("AVBD oracle — spring (the soft Force)", () => {
    // Phase 6.1: the first non-contact Force — a soft distance constraint (spring.ts, a port of
    // spring.cpp). C = ‖pA − pB‖ − rest, force f = stiffness·C, no dual (finite stiffness ⇒ λ = 0).
    // The scene is the demo's sceneSpring stripped to the spring alone (no ground, irrelevant here):
    // a static anchor at y=14, a 2×2×2 block (mass 8) hanging from a center-anchored spring (k=100,
    // rest=4). Center anchors ⇒ rWorld = 0 ⇒ zero torque ⇒ a clean 1-DOF vertical oscillator, so the
    // closed forms are exact: static extension ext = m|g|/k, frequency ω = √(k/m).

    const k = 100;
    const rest = 4;
    const anchorY = 14;
    const g = -10;
    const dt = 1 / 60;
    const size: Vec3 = [2, 2, 2];
    const m = massOf(size, 1); // 8
    const ext = (m * Math.abs(g)) / k; // mg/k = 0.8 — the static spring extension past rest
    const yEq = anchorY - rest - ext; // 9.2 — the block's equilibrium height
    const omega = Math.sqrt(k / m); // √(k/m) = 3.5355 rad/s
    const wh = omega * dt;
    const decay = 1 / Math.sqrt(1 + wh * wh); // BDF1 per-step amplitude factor |λ|

    const hang = (yStart: number, vy = 0): Solver => {
        const anchor = body([1, 1, 1], 0, 0.5, [0, anchorY, 0]);
        const block = body(size, m, 0.5, [0, yStart, 0], [0, vy, 0]);
        const s = makeSolver([anchor, block], { gravity: g, dt });
        s.springs.push(spring(anchor, block, [0, 0, 0], [0, 0, 0], k, rest));
        return s;
    };

    test("static extension is mg/k — the exact fixed point, and the attractor from a displaced start", () => {
        // Phase 1 — the equilibrium is the AVBD fixed point. Placed at yEq the spring force k·ext = mg
        // exactly cancels gravity, so dx = 0 every step (no transient): a direct check that the force
        // MAGNITUDE is k·C with the equilibrium at extension mg/k. A wrong force law (wrong sign/scale)
        // moves the fixed point, so this discriminates. Measured drift: ~1e-13.
        const atEq = hang(yEq);
        let maxDrift = 0;
        for (let f = 0; f < 200; f++) {
            step(atEq);
            maxDrift = Math.max(maxDrift, Math.abs(atEq.bodies[1].posLin[1] - yEq));
        }
        expect(maxDrift).toBeLessThan(1e-9);
        expect(length(atEq.bodies[1].velLin)).toBeLessThan(1e-9);

        // Phase 2 — yEq is the ATTRACTOR. From y=8 (the demo start, displaced 1.2 below eq) the lightly
        // damped oscillation decays toward yEq. After F frames the analytic envelope is A0·decay^F; F=5000
        // ⇒ ~2e-4, far below the ~0.4 m a wrong equilibrium would sit at, so 1e-2 both passes and pins mg/k.
        const F = 5000;
        const settling = hang(8);
        for (let f = 0; f < F; f++) step(settling);
        const env = 1.2 * decay ** F; // ~2e-4
        expect(env).toBeLessThan(1e-3); // sanity: the run is long enough to have damped
        expect(Math.abs(settling.bodies[1].posLin[1] - yEq)).toBeLessThan(1e-2);
        expect(length(settling.bodies[1].velLin)).toBeLessThan(5e-3); // came to rest at eq
    });

    test("oscillates at ω = √(k/m) — the exact BDF1 discrete period", () => {
        // Phase 2 closed form: displaced and released, the block is a damped harmonic oscillator. BDF1 on
        // u'' = −ω²u has discrete eigenvalue arg(λ) = atan(ωh), so the position zero-crossings about yEq
        // sit EXACTLY π/atan(ωh) frames apart (a decaying cosine crosses zero independent of its envelope).
        // ⇒ period Tdisc = h·2π/atan(ωh), which is the continuous Tcont = 2π√(m/k) pulled +0.12% by the
        // ωh/atan(ωh) discretization factor. Both reference √(k/m); the measured period pins the frequency.
        const Tdisc = (dt * 2 * Math.PI) / Math.atan(wh); // 1.77926 s — the exact discrete prediction
        const Tcont = (2 * Math.PI) / omega; // 1.77715 s — the continuous closed form

        const ys: number[] = [];
        const s = hang(8);
        for (let f = 0; f < 1100; f++) {
            step(s);
            ys.push(s.bodies[1].posLin[1]);
        }
        // interpolated zero-crossings of u = y − yEq (linear root between bracketing samples)
        const crossings: number[] = [];
        for (let f = 1; f < ys.length; f++) {
            const u0 = ys[f - 1] - yEq;
            const u1 = ys[f] - yEq;
            if (u0 < 0 !== u1 < 0) crossings.push(f - 1 + u0 / (u0 - u1));
        }
        let sumHalf = 0;
        for (let i = 1; i < crossings.length; i++) sumHalf += crossings[i] - crossings[i - 1];
        const avgHalf = sumHalf / (crossings.length - 1); // frames per half-period
        const measuredT = 2 * avgHalf * dt;
        console.log(
            `[oracle/spring] period measured ${measuredT.toFixed(6)}s, Tdisc ${Tdisc.toFixed(6)}s, ` +
                `Tcont ${Tcont.toFixed(6)}s, ${crossings.length} crossings`,
        );

        expect(crossings.length).toBeGreaterThan(20); // ~20 half-periods over 1100 frames — well sampled
        // crossings are exactly π/atan(ωh) frames apart, so the only error is the sub-frame interpolation
        // (~1e-6 rel, averaged over ~12 intervals) — measuredT matches Tdisc to <5e-4. And it lands within
        // the derived 0.12% of the continuous 2π√(m/k).
        expect(measuredT).toBeCloseTo(Tdisc, 3); // |Δ| < 5e-4 s
        expect(measuredT).toBeCloseTo(Tcont, 2); // |Δ| < 5e-3 s — the continuous limit, +0.12% off
    });

    test("reproduces the C++ spring + soft/stiff-ratio fixtures (whole-run tracking)", () => {
        // The harness spring scenes: sceneSpring (the hanging mass on the ground) and sceneSpringsRatio
        // (an 8-body chain of alternating soft k=10 / stiff k=10000 springs with offset anchors, so the
        // links rotate). A spring chain has no contact-set churn, so it isn't chaotic — the f64 oracle
        // tracks the f32 C++ over the WHOLE 600 frames, gated on a derived f32-vs-f64 round-off band an
        // order below the oscillation amplitude (cf. friction-settle's whole-run track). This is the
        // oracle == C++ rung; GPU == oracle is the later gym `springs` gate. Reconstructing the spring
        // also exercises the harness springs dump (harness-dense.cpp) → fixtures.ts loader path.
        const trackBand: Record<string, number> = { spring: 1e-3, "spring-ratio": 2e-2 };
        for (const scene of ["spring", "spring-ratio"]) {
            const fx = loadFixture("canonical", scene);
            expect((fx.springs ?? []).length).toBeGreaterThan(0); // the dump round-tripped
            const s = fixtureSolver(fx);
            let maxErr = 0;
            let finite = true;
            for (let f = 0; f < fx.frames.length; f++) {
                step(s);
                for (let i = 0; i < s.bodies.length; i++) {
                    if (s.bodies[i].mass <= 0) continue;
                    if (!s.bodies[i].posLin.every(Number.isFinite)) finite = false;
                    maxErr = Math.max(
                        maxErr,
                        length(sub(s.bodies[i].posLin, framePos(fx.frames[f], i))),
                    );
                }
            }
            expect(finite).toBe(true);
            expect(maxErr).toBeLessThan(trackBand[scene]); // measured 2.5e-4 (spring) / 7.7e-3 (ratio)
        }
    });
});

describe("AVBD oracle — joints (the hard Force)", () => {
    // Phase 6.2: the hard Force (joint.ts, a port of joint.cpp). Two stacked constraints — a linear anchor
    // pin (C = pA − pB) and an angular relative-orientation lock (C = (qA − qB)·torqueArm) — carrying
    // warmstartable λ + a per-iteration penalty ramp, with the rigid (∞-stiffness) form adding the explosive-
    // error stabilization C −= α·C₀. The two shipped types are configs of the SAME force: spherical
    // (stiffnessAng = 0 → the angular rows never activate, rotation free) and fixed (stiffnessAng = ∞). Without
    // a joint the body free-falls, so these gates pin the joint behavior, not gravity.

    const g = -10;
    const dt = 1 / 60;
    const size: Vec3 = [0.5, 0.5, 0.5];
    const m = massOf(size, 1); // 0.125

    test("spherical-joint pendulum swings at the physical-pendulum period 2π√(I_pivot/(m·g·d))", () => {
        // A bob pinned by a spherical joint d from its COM is a physical pendulum: gravity torques it about
        // the pin, rotation otherwise free (the spherical joint's linear+angular Jacobian coupling makes it
        // a pin). Small angle ⇒ θ'' = −ω²θ with ω² = m·g·d / I_pivot, I_pivot = I_com + m·d² (parallel axis
        // about the swing-plane z). BDF1 on that ODE crosses zero exactly π/atan(ωh) frames apart, so the
        // period is Tdisc = h·2π/atan(ωh) — the same discretization the spring test pins, and +0.26% of the
        // simple-pendulum 2π√(d/g) (the I_com/(m·d²) = 0.46% box-moment correction). The arm MUST coincide at
        // t=0: COM d below the pivot, rB = +d (local up to the pin), the body rotated by θ0 so rotate(qθ0, rB)
        // lands on the pivot — an offset start instead injects energy (the next test's footgun).
        const d = 3;
        const H = 8;
        const theta0 = 0.05;
        const iCom = (m * (size[0] * size[0] + size[1] * size[1])) / 12; // about z
        const iPivot = iCom + m * d * d;
        const omega = Math.sqrt((m * Math.abs(g) * d) / iPivot);
        const wh = omega * dt;
        const Tdisc = (dt * 2 * Math.PI) / Math.atan(wh); // 3.45046 s
        const Tsimple = 2 * Math.PI * Math.sqrt(d / Math.abs(g)); // 2π√(L/g) = 3.44144 s

        const qz: Quat = [0, 0, Math.sin(theta0 / 2), Math.cos(theta0 / 2)];
        const pivot = body(size, 0, 0.5, [0, H, 0]);
        const bob = body(
            size,
            m,
            0.5,
            [d * Math.sin(theta0), H - d * Math.cos(theta0), 0],
            [0, 0, 0],
            qz,
        );
        const s = makeSolver([pivot, bob], { gravity: g, dt });
        s.joints.push(joint(pivot, bob, [0, 0, 0], [0, d, 0]));
        // the arm coincides at t=0 — a clean hanging small-angle pendulum, not the energy-injecting offset
        const armErr = length(
            sub(
                transform(pivot.posLin, pivot.posAng, [0, 0, 0]),
                transform(bob.posLin, bob.posAng, [0, d, 0]),
            ),
        );
        expect(armErr).toBeLessThan(1e-12);

        const ang: number[] = [];
        for (let f = 0; f < 1400; f++) {
            step(s);
            ang.push(Math.atan2(bob.posLin[0], H - bob.posLin[1])); // swing angle from vertical
        }
        const crossings: number[] = [];
        for (let f = 1; f < ang.length; f++) {
            const a0 = ang[f - 1];
            const a1 = ang[f];
            if (a0 < 0 !== a1 < 0) crossings.push(f - 1 + a0 / (a0 - a1));
        }
        let sumHalf = 0;
        for (let i = 1; i < crossings.length; i++) sumHalf += crossings[i] - crossings[i - 1];
        const measuredT = (2 * sumHalf * dt) / (crossings.length - 1);

        // small angle stays small (no energy injected — the arm started coincident), so the linear period holds
        expect(Math.max(...ang.map(Math.abs))).toBeLessThan(theta0 * 1.01); // measured max 0.0500
        expect(crossings.length).toBeGreaterThan(10); // ~14 half-periods over 1400 frames
        // crossings sit exactly π/atan(ωh) apart, so the only error is sub-frame interpolation — measuredT
        // matches Tdisc to < 5e-4 (measured 2.6e-4) and the simple form to < 0.5% (measured 0.27%)
        expect(measuredT).toBeCloseTo(Tdisc, 3); // |Δ| < 5e-4 s
        expect(measuredT).toBeCloseTo(Tsimple, 1); // |Δ| < 0.05 s — the box-moment-corrected 2π√(L/g)
    });

    test("a spherical joint leaves rotation free; a fixed joint locks it (a static anchor pins a body rigid)", () => {
        // Same 2-body rig — a dynamic box pinned to a static anchor at a 2 m arm — under gravity. The angular
        // row triple is the ONLY difference: spherical (stiffnessAng 0) lets the box swing down (rotation free
        // → a pendulum); fixed (stiffnessAng ∞) locks the box's orientation to the anchor's, and with the
        // linear pin that leaves zero DOF → the box is held rigid in place. Isolates the fixed joint's angular
        // rows from the shared linear ones.
        const H = 5;
        const d = 2;
        const run = (stiffAng: number): Body => {
            const anchor = body(size, 0, 0.5, [0, H, 0]);
            const b = body(size, m, 0.5, [d, H, 0]);
            const s = makeSolver([anchor, b], { gravity: g, dt });
            s.joints.push(
                joint(anchor, b, [0, 0, 0], [-d, 0, 0], Number.POSITIVE_INFINITY, stiffAng),
            );
            for (let f = 0; f < 240; f++) step(s);
            return b;
        };
        const free = run(0); // spherical
        const locked = run(Number.POSITIVE_INFINITY); // fixed

        expect(free.posLin[1]).toBeLessThan(H - 1); // swings down ~2 m (measured 3.00, a drop of ~2)
        expect(Math.abs(free.posAng[2])).toBeGreaterThan(0.5); // rotated freely (measured |qz| 0.73)
        expect(length(sub(locked.posLin, [d, H, 0]))).toBeLessThan(1e-2); // held rigid (measured 2.5e-3)
        expect(Math.abs(locked.posAng[2])).toBeLessThan(1e-2); // no z-rotation (measured 5e-4)
    });

    test("a hung rigid chain settles straight, anchors held to the augmented-Lagrangian residual", () => {
        // Roadmap closed forms: a fixed-joint chain settles straight + a bounded steady anchor error. The
        // fixed chain is a horizontal cantilever fixed to a static anchor; the angular rows keep it rigid, so
        // it settles to rest with the links still on the anchor's y/z row. The rigid constraint isn't driven
        // to zero but to the augmented-Lagrangian residual at the canonical (α 0.99 / γ 0.999 / β 1e4 / 10
        // iter) schedule — the same satisfaction scale as the 0.01 m contact margin, and the exact value the
        // C++ reference settles to (the parity test pins the value; this gates the qualitative invariants).
        const fx = loadFixture("canonical", "joint-fixed-chain");
        const s = fixtureSolver(fx);
        for (let f = 0; f < fx.frames.length; f++) step(s);

        let maxAnchor = 0;
        for (const jt of s.joints) {
            const pA = jt.a ? transform(jt.a.posLin, jt.a.posAng, jt.rA) : jt.rA;
            const pB = transform(jt.b.posLin, jt.b.posAng, jt.rB);
            maxAnchor = Math.max(maxAnchor, length(sub(pA, pB)));
        }
        let maxDev = 0;
        for (const b of s.bodies) {
            if (b.mass <= 0) continue;
            maxDev = Math.max(maxDev, Math.abs(b.posLin[1] - 8), Math.abs(b.posLin[2])); // off the anchor row
        }
        expect(maxSpeed(s)).toBeLessThan(1e-2); // settled to rest (measured 9e-5)
        expect(maxAnchor).toBeLessThan(1e-3); // the AL residual, an order under the 0.01 m margin (measured 4.4e-4)
        expect(maxDev).toBeLessThan(1e-2); // straight: < 1 cm sag/lateral over 3 links (measured 6e-3)
    });

    test("reproduces the C++ joint fixtures — pendulum, spherical + fixed chains (whole-run tracking)", () => {
        // The harness joint scenes (scenes.h): a spherical-pin pendulum, a 3-link spherical chain swaying, a
        // 3-link fixed-joint cantilever. All contact-free + bounded + non-chaotic, so the f64 oracle tracks the
        // f32 C++ over the whole 600 frames, gated on a derived f32-vs-f64 round-off band an order below the
        // motion scale (cf. the spring fixtures). The oracle == C++ rung; GPU == oracle is the later gym gate.
        // Reconstructing the joints also exercises the harness joint dump → fixtures loader (the 1e30 → ∞ map).
        const trackBand: Record<string, number> = {
            "joint-pendulum": 2e-3,
            "joint-spherical-chain": 1e-3,
            "joint-fixed-chain": 1e-4,
        };
        for (const scene of ["joint-pendulum", "joint-spherical-chain", "joint-fixed-chain"]) {
            const fx = loadFixture("canonical", scene);
            expect((fx.joints ?? []).length).toBeGreaterThan(0); // the dump round-tripped
            const s = fixtureSolver(fx);
            let maxErr = 0;
            let finite = true;
            for (let f = 0; f < fx.frames.length; f++) {
                step(s);
                for (let i = 0; i < s.bodies.length; i++) {
                    if (s.bodies[i].mass <= 0) continue;
                    if (!s.bodies[i].posLin.every(Number.isFinite)) finite = false;
                    maxErr = Math.max(
                        maxErr,
                        length(sub(s.bodies[i].posLin, framePos(fx.frames[f], i))),
                    );
                }
            }
            expect(finite).toBe(true);
            expect(maxErr).toBeLessThan(trackBand[scene]); // measured 4.0e-4 / 2.0e-4 / 9.4e-6
        }
    });

    test("joints never inject energy — a swinging pendulum + chains stay E(t) ≤ E(0) (the rope-explosion guard)", () => {
        // The legacy solver injected energy into ropes; the α-stabilized rigid joint + BDF1 recovery must not.
        // Each joint fixture is driven only by gravity from its start pose, so E(0) is the supremum and a
        // dissipative solver only loses energy. A coincident-anchor joint conserves it (measured dE = 0 to f64
        // round-off); energy GROWING is the injection failure, caught loudly here. The non-coincident footgun
        // that DOES inject is blocked at construction — the next test.
        for (const scene of ["joint-pendulum", "joint-spherical-chain", "joint-fixed-chain"]) {
            const fx = loadFixture("canonical", scene);
            const s = fixtureSolver(fx);
            const e0 = energy(s);
            let maxE = e0;
            let finite = true;
            for (let f = 0; f < fx.frames.length; f++) {
                step(s);
                maxE = Math.max(maxE, energy(s));
                for (const b of s.bodies) if (!b.posLin.every(Number.isFinite)) finite = false;
            }
            expect(finite).toBe(true);
            // band = 1e-3·E0, three orders over f64 round-off and far under the +34% a real injection produces
            expect(maxE).toBeLessThanOrEqual(e0 * (1 + 1e-3));
        }
    });

    test("a grossly non-coincident joint fails loudly at construction (the rope-explosion footgun)", () => {
        // Reproduces the legacy rope bug + proves the guard catches it. A rigid joint whose anchors start far
        // apart recovers spurious velocity through BDF1 as it corrects (measured: a 4.2 m mismatch injects +34%
        // energy, 6.6 m/s). joint() throws when the anchors exceed the bodies' combined reach, so the mistake
        // surfaces at build time, not as an explosion 50 frames in. A coincident (or ≤ reach) start constructs
        // fine — the α-stabilization absorbs a small offset.
        const pivot = body(size, 0, 0.5, [0, 8, 0]);
        const bob = body(size, m, 0.5, [0, 5, 0]); // hangs 3 m below, identity orientation
        expect(() => joint(pivot, bob, [0, 0, 0], [0, 3, 0])).not.toThrow(); // rB = +y → anchors meet
        expect(() => joint(pivot, bob, [0, 0, 0], [-3, 0, 0])).toThrow(/coincident/); // 4.2 m mismatch → loud
    });

    test("a joint between two non-dynamic bodies fails loudly at construction (the both-static energy guard)", () => {
        // A joint no dynamic body can resolve — both endpoints mass ≤ 0 (static/kinematic) — is never satisfied
        // by the primal (both skip it, solver.ts), so its dual penalty + λ ramp unbounded: the joint analog of
        // the contact all-static dual guard (manifold.ts updateDual). The harm surfaces when such an endpoint is
        // later released (made dynamic) — the accumulated huge λ yanks it. joint() rejects it at construction;
        // the GPU jointInit deactivates + bumps counters[1]. A joint with ONE dynamic body (a grab / pendulum to
        // a static anchor) is the LEGITIMATE case and must NOT be rejected. Anchors coincident so the rope guard
        // can't mask the both-static one. (Red before the guard: this construction does not throw.)
        const staticA = body(size, 0, 0.5, [0, 8, 0]);
        const staticB = body(size, 0, 0.5, [0, 8, 0]);
        const dyn = body(size, m, 0.5, [0, 8, 0]);
        expect(() => joint(staticA, staticB, [0, 0, 0], [0, 0, 0])).toThrow(/non-dynamic/);
        expect(() => joint(staticA, dyn, [0, 0, 0], [0, 0, 0])).not.toThrow(); // static anchor + dynamic = fine
    });

    test("the grab dangles from a world anchor without injecting energy; a kinematic-anchor BODY flails", () => {
        // avbd-demo3d's mouse-drag grab pins the box to a WORLD-space anchor (a = null, rA the cursor point) by a
        // soft SPHERICAL joint (joint.cpp `bodyA == null`): the box dangles (rotation free), and with NO anchor
        // body there's no anchor↔box contact. Dragging pumps the pendulum; holding the anchor still does no work,
        // so the box can only DISSIPATE (a damped pendulum) — energy never grows, never NaN, never the chaotic
        // flail the report feared. Contrast: a kinematic-anchor BODY (the old workaround) embeds a sphere
        // in the box surface whose CONTACT keeps shoving it, so that one stays agitated — why the grab joints to
        // the world (no body), not a kinematic anchor.
        const dragThenHold = (
            worldAnchor: boolean,
        ): { early: number; late: number; finite: boolean } => {
            const box = body([0.8, 0.8, 0.8], 0.4, 0.5, [0, 3, 0]); // held in the air (no ground)
            const arm: Vec3 = [0.4, 0, 0]; // grabbed at a +x face-surface point → offset, so it swings
            const hit: Vec3 = [0.4, 3, 0];
            const bodies: Body[] = [box];
            let j: ReturnType<typeof joint>;
            if (worldAnchor) {
                j = joint(null, box, [...hit] as Vec3, arm, 5000, 0); // world anchor — no body, no contact
            } else {
                const anchor = sphere(0.05, 0, 0, [...hit] as Vec3); // a kinematic anchor BODY → a contact
                bodies.push(anchor);
                j = joint(anchor, box, [0, 0, 0], arm, 5000, 0);
            }
            const s = makeSolver(bodies, { gravity: -10, iterations: 4 });
            s.joints.push(j);
            const setAnchor = (p: Vec3): void => {
                if (worldAnchor) {
                    j.rA = p; // mutate the world anchor point (the reference's `drag->rA = …`)
                } else {
                    const a = bodies[1];
                    a.velLin = scale(sub(p, a.posLin), 60);
                    a.posLin = p;
                }
            };
            const base: Vec3 = [0.4, 3, 0];
            // whip the anchor around a fast circle (radius 0.5, ~4 m/s), then hold it dead still for 6 s
            for (let f = 0; f < 240; f++) {
                const t = f / 60;
                setAnchor([
                    base[0] + Math.cos(t * 8) * 0.5 - 0.5,
                    base[1],
                    base[2] + Math.sin(t * 8) * 0.5,
                ]);
                step(s);
            }
            const hold: Vec3 = worldAnchor ? ([...j.rA] as Vec3) : ([...bodies[1].posLin] as Vec3);
            let early = 0;
            let late = 0;
            let finite = true;
            for (let f = 0; f < 360; f++) {
                setAnchor(hold);
                step(s);
                const w = length(box.velAng);
                if (f >= 60 && f < 120) early = Math.max(early, w); // peak swing in hold window [1s, 2s]
                if (f >= 240 && f < 300) late = Math.max(late, w); //  …and in window [4s, 5s]
                if (!box.posLin.every(Number.isFinite)) finite = false;
            }
            return { early, late, finite };
        };

        const world = dragThenHold(true);
        expect(world.finite).toBe(true); // never explodes
        expect(world.early).toBeGreaterThan(0.1); // it DANGLES — a spherical joint leaves rotation free
        expect(world.late).toBeLessThan(world.early); // and DAMPS: the held pendulum dissipates, no injection
        expect(world.early).toBeLessThan(15); // bounded swing, never a chaotic blow-up

        // the kinematic-anchor BODY stays more agitated than the world anchor (its embedded-sphere contact keeps
        // feeding energy in) — the reason the grab joints to the world, not a body. Measured world ≈ 2, kin ≈ 4.
        const kin = dragThenHold(false);
        expect(kin.late).toBeGreaterThan(world.late);
    });
});

describe("AVBD oracle — scheduler equivalence", () => {
    // a small settling stack: ground + 4 boxes. Adjacent boxes share a contact.
    const smallStack = (): Body[] => {
        const bs = [body([100, 1, 100], 0, 0.5, [0, 0, 0])];
        for (let i = 0; i < 4; i++)
            bs.push(body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.0 + i, 0]));
        return bs;
    };
    const restOf = (schedule: Schedule): Vec3[] => {
        const s = makeSolver(smallStack());
        for (let f = 0; f < 250; f++) step(s, schedule);
        return s.bodies.map((b) => [...b.posLin] as Vec3);
    };

    test("colored with reverse-rank colors is bit-identical to sequential", () => {
        // colors[i] = n−1−i gives every body its own color; ascending color order then visits bodies
        // n−1 … 0 — the same reverse-creation order sequential GS uses, and a one-body color commits
        // immediately, so the deferred-commit colored path reduces to sequential exactly. Measured: 0.
        const seq = restOf({ kind: "sequential" });
        const n = smallStack().length;
        const colored = restOf({
            kind: "colored",
            colors: Array.from({ length: n }, (_, i) => n - 1 - i),
        });
        let maxDiff = 0;
        for (let i = 0; i < seq.length; i++)
            maxDiff = Math.max(maxDiff, length(sub(seq[i], colored[i])));
        expect(maxDiff).toBeLessThan(1e-12);
    });

    test("a valid coloring reaches the same rest as sequential (coloring-preserves-solution)", () => {
        // alternate boxes by height — no same-color pair shares a contact, so colored GS converges to
        // the sequential fixed point (transient differs, rest does not). Measured 8e-6.
        const seq = restOf({ kind: "sequential" });
        const colored = restOf({ kind: "colored", colors: [0, 1, 0, 1, 0] });
        let maxDiff = 0;
        for (let i = 0; i < seq.length; i++)
            maxDiff = Math.max(maxDiff, length(sub(seq[i], colored[i])));
        expect(maxDiff).toBeLessThan(1e-3);
    });
});

describe("AVBD oracle — warmstart (cross-frame persistence)", () => {
    // The Phase-3 warmstart crux at the spec level (scratch.md "AVBD rebuild" → Distributed cruxes →
    // Warmstart): the reference's force-list manifold persistence — `initManifold` merges this frame's
    // contacts onto last frame's by feature key, carrying λ/k with γ decay (manifold.ts). The fixtures
    // (oracle.test.ts above) exercise the merge on *settling* scenes; these close the documented gap —
    // a churning scene (flipping feature keys) and the positive "warmstart converges tighter" property
    // that proves the persisted state actually does work. The GPU reconstructs this merge (step.ts);
    // the gym `pile` stack-warmstart gate verifies GPU == this oracle on the real device.

    // a 5-box chain on the ground — the bottom contact carries the whole stack, the regime the paper
    // flags as wanting more iterations (a "series of connections"). Warmstart accumulates λ/k across
    // frames, so it converges the chain in fewer per-frame iterations than a cold reset.
    const chain = (): Body[] => {
        const bs = [body([100, 1, 100], 0, 0.5, [0, 0, 0])];
        for (let i = 0; i < 5; i++)
            bs.push(body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.0 + i, 0]));
        return bs;
    };
    const restY = 1 - COLLISION_MARGIN; // the bottom box's box-touching height

    const settle = (layer: "dual" | "warmstart", iterations: number) => {
        const s = makeSolver(chain(), {
            layer,
            iterations,
            betaLin: 1e4, // canonical — the same ramp for both, so warmstart's edge is persistence alone
            alpha: 0.99,
            gamma: 0.999,
            gravity: -10,
        });
        for (let f = 0; f < 240; f++) step(s);
        return { pen: restY - s.bodies[1].posLin[1], speed: maxSpeed(s) };
    };

    test("warmstart converges the chain in fewer iterations than a cold reset", () => {
        // At the canonical 1e4 ramp and 4 iters, the cold `dual` layer (λ/k reset every frame) leaves the
        // load-bearing bottom contact deeply penetrated — 4 iters can't propagate the 5-box load down a
        // cold chain. Warmstart carries the ramped λ/k across frames, so the same 4 iters converge far
        // tighter — and at least as tight as 10 cold iters (the "fewer iterations" property). Measured:
        // cold-4it pen ~0.73 m, warmstart-4it ~2.6e-3 m (~280× tighter, and < cold-10it).
        const cold4 = settle("dual", 4);
        const warm4 = settle("warmstart", 4);
        const cold10 = settle("dual", 10);
        console.log(
            `[oracle/warmstart] chain bottom penetration — cold(dual,4it) ${cold4.pen.toExponential(2)} m, ` +
                `warmstart(4it) ${warm4.pen.toExponential(2)} m, cold(dual,10it) ${cold10.pen.toExponential(2)} m`,
        );
        expect(warm4.speed).toBeLessThan(0.05); // premise: warmstart settles
        expect(warm4.pen).toBeGreaterThan(0); // still touching (positive penetration)
        expect(warm4.pen).toBeLessThan(cold4.pen * 0.5); // ≥2× tighter than equal-iter cold — persistence works
        expect(warm4.pen).toBeLessThan(cold10.pen + 1e-3); // 4 warmstart iters ≥ 10 cold ones (fewer iterations)
        expect(warm4.pen).toBeLessThan(1e-2); // and converged near the margin
    });

    test("churning contacts (tipping box): energy non-increasing, no λ blow-up, settles flat", () => {
        // A box tilted ~40° about z, dropped from rest onto the ground. It lands on an edge, tips, and
        // settles flat — the contact feature flips (edge-edge key → face manifold keys) as it tips,
        // churning the keys the warmstart merge is keyed on. A bad merge (λ carried onto the wrong
        // feature, the legacy instability) injects energy; the gate is that mechanical energy never
        // exceeds the drop's E0 (the supremum — the box starts at rest) within round-off, stays finite,
        // and the box settles flat (proving the edge→face churn actually happened, then persisted).
        const a = 0.7; // ~40° about z
        const q: Quat = [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
        const s = makeSolver(
            [
                body([100, 1, 100], 0, 0.6, [0, 0, 0]),
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.6, [0, 2.5, 0], [0, 0, 0], q),
            ],
            { layer: "warmstart", gravity: -10 },
        );
        const E0 = energy(s);
        let maxExcess = 0;
        let finite = true;
        for (let f = 0; f < 300; f++) {
            step(s);
            if (!s.bodies[1].posLin.every(Number.isFinite)) finite = false;
            maxExcess = Math.max(maxExcess, (energy(s) - E0) / Math.abs(E0));
        }
        console.log(
            `[oracle/warmstart] tipping-box max energy excess ${maxExcess.toExponential(2)}`,
        );
        expect(finite).toBe(true);
        // E0 is the supremum (dropped from rest); a dissipative solver only loses energy. The penalty
        // spring loads on impact but never lifts the box above its drop height, so E(t) ≤ E0 holds to a
        // small slack. Measured ~1e-3; 1e-2 is a derived margin (a real blow-up is O(1)+ within frames).
        expect(maxExcess).toBeLessThan(1e-2);
        // settled flat on a face: y near the box-touching rest, at rest
        expect(maxSpeed(s)).toBeLessThan(0.05);
        expect(s.bodies[1].posLin[1]).toBeGreaterThan(0.9);
        expect(s.bodies[1].posLin[1]).toBeLessThan(1.05);
    });
});

describe("AVBD oracle — the kinematic-pushing fix (roadmap §6.4)", () => {
    // The headline correctness fix: a kinematic body (a character, mass ≤ 0, moved by its controller)
    // pushed into an immovable surface (a static wall, also mass ≤ 0). Neither body's primal runs
    // (both mass ≤ 0), so the contact constraint C is never satisfied — it stays penetrating frame
    // after frame. The reference's dual update (solver.cpp:230 runs updateDual on EVERY force) then
    // ramps that contact's penalty `k += βLin·|C|` every iteration, every frame, with nothing ever
    // moving to relax C — the escalating constraint force the legacy stack blew up on (physics.md
    // "legacy antipatterns"). The fix is at the dual update: a contact NO dynamic body can resolve
    // must not ramp. Validated here (red without the gate: the penalty escalates), bounded with it.

    const maxNormalPenalty = (s: Solver): number => {
        let p = 0;
        for (const m of s.manifolds.values())
            for (const c of m.contacts) p = Math.max(p, c.penalty[0]);
        return p;
    };

    test("a kinematic capsule held into a static wall does NOT ramp the penalty (both mass ≤ 0)", () => {
        // A static box wall + a capsule character (mass 0 = kinematic) overlapping its +x face. Both are
        // mass ≤ 0, so the primal skips both — the capsule stays put, penetrating, and the contact can
        // never close. Without the gate the dual ramps this unsolvable contact's penalty unbounded
        // (the legacy escalation); with it the penalty holds at the PENALTY_MIN seed (the decay clamps
        // a never-ramped contact to the floor). The bodies don't move either way (no primal), so this
        // isolates the ramp pathology, not a position effect.
        const wall = body([1, 4, 4], 0, 0.5, [0, 1, 0]); // static, +x face at x = 0.5
        const char = capsule(0.5, 0.3, 0, 0.5, [0.6, 1, 0]); // kinematic (mass 0), surface ~0.2 m into the wall
        const s = makeSolver([wall, char], { layer: "warmstart", gravity: -10 });

        let penAt5 = 0;
        for (let f = 0; f < 300; f++) {
            step(s);
            if (f === 5) penAt5 = maxNormalPenalty(s);
        }
        const penFinal = maxNormalPenalty(s);
        console.log(
            `[oracle/§6.4] kin-vs-static normal penalty: frame 5 ${penAt5.toExponential(2)}, ` +
                `frame 300 ${penFinal.toExponential(2)} (seed ${PENALTY_MIN})`,
        );

        // premise: the contact exists and the capsule is genuinely penetrating (else there's nothing to
        // ramp and the test is vacuous)
        expect([...s.manifolds.values()].some((m) => m.contacts.length > 0)).toBe(true);
        // the invariant: the penalty never climbs off the seed floor (a never-ramped contact decays to
        // PENALTY_MIN). Without the fix it escalates into the 1e5–1e10 range — `< 2` cleanly separates the
        // bounded seed from any ramp. It must also not grow between an early and a late frame.
        expect(penFinal).toBeLessThan(PENALTY_MIN + 1);
        expect(penFinal).toBeLessThanOrEqual(penAt5 + 1e-6);
    });

    test("a DYNAMIC box on the same wall still ramps — the gate is specific to the all-static contact", () => {
        // The fix must not silence a real resting contact. A dynamic box settling on a static ground has a
        // dynamic body, so its dual ramp is untouched: the penalty climbs well off the seed to hold mg
        // (the Phase-2 behavior). This pins that the gate keys on "no dynamic body", not "is static
        // involved" — a dyn–static contact (one dynamic) keeps ramping.
        const s = makeSolver(
            [
                body([100, 1, 100], 0, 0.5, [0, 0, 0]), // static ground
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.5, 0]), // dynamic box
            ],
            { layer: "warmstart", gravity: -10 },
        );
        for (let f = 0; f < 300; f++) step(s);
        const pen = maxNormalPenalty(s);
        console.log(
            `[oracle/§6.4] dyn-vs-static normal penalty ${pen.toExponential(2)} (ramps to hold mg)`,
        );
        expect(pen).toBeGreaterThan(PENALTY_MIN + 1); // a real contact ramps
        expect(length(s.bodies[1].velLin)).toBeLessThan(1e-2); // and rests
    });
});
