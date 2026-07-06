import { describe, expect, test } from "bun:test";
import { joint } from "./joint";
import { body, massOf } from "./rigid";
import { makeSolver, step } from "./solver";

// The angular motor — a 1-DOF force-clamped drive on the hard `Force` (joint.ts), a port of
// avbd-demo2d motor.cpp (`C = deltaAngle − speed·dt`, `f = clamp(penalty·C + λ, ±maxTorque)`). The motor
// competes inside each solver iteration, so unlike the forced-velocity drive (consumed once per step by the
// inertial prediction) it HOLDS the target ω against a load up to its torque clamp, and yields past it.
//
// These gates are derivable in isolation (no confounding multi-body load): the drive reaches target ω, the
// clamp bounds the spin-up rate at maxTorque/I, and against a gravity load the clamp-vs-peak-torque threshold
// (the reference `sceneMotor`) decides lap-vs-stall. The forced-velocity-stalls-but-motor-holds witness under
// a real constraint load (the spindle's rope joints) is the gym `motor` scenario, on the real GPU.
//
// RED-FIRST: with the motor term removed (jointContrib's clamp/target), the rotor never spins — every gate
// here goes red on `velAng` ≈ 0 / swept ≈ 0, so they pin the term, not a restatement of it.

const dt = 1 / 60;

describe("AVBD oracle — angular motor (the force-clamped 1-DOF drive)", () => {
    test("a motor spins a free rotor up to its target ω and holds it", () => {
        // A unit box pinned at its COM by a world spherical joint (rotation free) + a Y motor, no gravity. The
        // motor (non-binding clamp) drives the spin to `speed` and holds it — the steady state where C → 0.
        const speed = 5;
        const rotor = body([1, 1, 1], massOf([1, 1, 1], 1), 0, [0, 2, 0]);
        const s = makeSolver([rotor], { gravity: 0, dt });
        s.joints.push(
            joint(null, rotor, [0, 2, 0], [0, 0, 0], Number.POSITIVE_INFINITY, 0, {
                axis: [0, 1, 0],
                speed,
                maxTorque: 50, // ≫ steady-state torque (≈ 0) — non-binding once at speed
            }),
        );

        let swept = 0;
        for (let f = 0; f < 180; f++) {
            step(s);
            swept += rotor.velAng[1] * dt;
        }
        // reached and holds the target (the spin-up is a couple of steps at this clamp), and genuinely rotated
        // (> a full turn) — both red without the motor term (velAng ≈ 0, swept ≈ 0)
        expect(rotor.velAng[1]).toBeCloseTo(speed, 1);
        expect(swept).toBeGreaterThan(2 * Math.PI);
    });

    test("the torque clamp bounds the spin-up rate at maxTorque / I", () => {
        // Same rotor, but a SMALL clamp. While behind target the force saturates at maxTorque, so the angular
        // accel is α = maxTorque / I_y (constant) and ω(t) ≈ α·t until it nears `speed`. Pins the clamp.
        const speed = 5;
        const maxTorque = 0.1;
        const mass = massOf([1, 1, 1], 1); // 1
        const iY = ((1 + 1) / 12) * mass; // boxMoment about Y = (sx²+sz²)/12·m = 0.16667
        const alpha = maxTorque / iY; // 0.6 rad/s²
        const t = 2;
        const rotor = body([1, 1, 1], mass, 0, [0, 2, 0]);
        const s = makeSolver([rotor], { gravity: 0, dt });
        s.joints.push(
            joint(null, rotor, [0, 2, 0], [0, 0, 0], Number.POSITIVE_INFINITY, 0, {
                axis: [0, 1, 0],
                speed,
                maxTorque,
            }),
        );

        for (let f = 0; f < Math.round(t / dt); f++) step(s);

        // still ramping (clamp-limited, well below target) and tracking the constant-torque ramp α·t = 1.2.
        // BDF1 + the penalty solve add a few % — bracket ±15% around the derived 1.2.
        expect(rotor.velAng[1]).toBeLessThan(speed * 0.5);
        expect(rotor.velAng[1]).toBeGreaterThan(alpha * t * 0.85);
        expect(rotor.velAng[1]).toBeLessThan(alpha * t * 1.15);
    });

    test("against a gravity load the clamp decides lap (strong) vs stall (weak) — sceneMotor", () => {
        // A bar pinned at one end (world spherical), motored about Z against gravity (the reference scene). Peak
        // gravity torque is τg = m·g·(L/2) at horizontal. A clamp above τg drives the bar over the top and laps;
        // a clamp below τg can't lift past the gravity balance, so it stalls (never completes a revolution). The
        // threshold maxTorque ≷ τg is the derivable fork.
        const L = 4;
        const mass = massOf([L, 0.5, 0.5], 1); // 1
        const g = -10;
        const tauG = mass * Math.abs(g) * (L / 2); // 20
        const speed = 4;

        const run = (maxTorque: number): number => {
            const bar = body([L, 0.5, 0.5], mass, 0, [L / 2, 8, 0]);
            const s = makeSolver([bar], { gravity: g, dt });
            s.joints.push(
                joint(null, bar, [0, 8, 0], [-L / 2, 0, 0], Number.POSITIVE_INFINITY, 0, {
                    axis: [0, 0, 1],
                    speed,
                    maxTorque,
                }),
            );
            let swept = 0;
            let maxAbs = 0;
            for (let f = 0; f < Math.round(4 / dt); f++) {
                step(s);
                swept += bar.velAng[2] * dt;
                maxAbs = Math.max(maxAbs, Math.abs(swept));
            }
            return maxTorque > tauG ? swept : maxAbs;
        };

        // strong (2·τg): laps the gravity load — net rotation past a full revolution
        expect(run(2 * tauG)).toBeGreaterThan(2 * Math.PI);
        // weak (0.4·τg < τg): stalls — never completes a revolution (bounded oscillation about the balance)
        expect(run(0.4 * tauG)).toBeLessThan(Math.PI);
    });

    test("a motor between two free bodies drives their relative ω, splitting it equal-and-opposite", () => {
        // Both endpoints dynamic (a hinge motor, the two-body path the world-anchor tests never reach: isA both
        // ways + the a-side increment). Two identical free rotors, gravity off, joined by a pure Y motor
        // (stiffnessLin 0 — no linear pin). With no other angular constraint the motor drives (ω_b − ω_a) to
        // `speed`; equal-and-opposite torque + equal moments split it ω_b = +speed/2, ω_a = −speed/2 (the angular
        // -momentum invariant ω_a + ω_b = 0). The Y-axis anchors are spin-invariant, so the bodies hold position.
        const speed = 6;
        const a = body([1, 1, 1], massOf([1, 1, 1], 1), 0, [0, 0, 0]);
        const b = body([1, 1, 1], massOf([1, 1, 1], 1), 0, [0, 2, 0]);
        const s = makeSolver([a, b], { gravity: 0, dt });
        s.joints.push(
            joint(a, b, [0, 1, 0], [0, -1, 0], 0, 0, {
                axis: [0, 1, 0],
                speed,
                maxTorque: 50,
            }),
        );

        for (let f = 0; f < 180; f++) step(s);

        // relative rate held at +speed (signed — catches a sign flip on the isA stamp)
        expect(b.velAng[1] - a.velAng[1]).toBeCloseTo(speed, 1);
        // equal-and-opposite split (catches an asymmetric two-body stamp). The sum holds to ~0.01, not 0 — the
        // BDF1 quaternion integrator's small-angle ω bleed (physics.md "residual ω decay"), asymmetric here.
        expect(a.velAng[1] + b.velAng[1]).toBeCloseTo(0, 1);
        expect(b.velAng[1]).toBeCloseTo(speed / 2, 1);
    });
});
