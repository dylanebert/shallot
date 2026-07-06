// Loader for the dense AVBD parity fixtures (tests/fixtures/avbd/{canonical,budget}/, the C++
// reference trajectories — see gen-fixtures.ts). Builds a Solver from a fixture's initial state
// + params, and exposes the per-frame reference state the oracle is checked against. Test
// scaffolding; the fixtures are f32 (C++ float), the oracle f64 — so trajectory comparison is
// tolerant (and chaotic scenes only on the statistical band — scratch.md "Gate ladder").

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { joint } from "./joint";
import type { Vec3 } from "./math";
import { body } from "./rigid";
import { makeSolver, type Solver } from "./solver";
import { spring } from "./spring";

export type ParamSet = "canonical" | "budget";

export interface FixtureParams {
    dt: number;
    gravity: number;
    iterations: number;
    alpha: number;
    betaLin: number;
    betaAng: number;
    gamma: number;
}

export interface FixtureBody {
    mass: number;
    friction: number;
    size: [number, number, number];
    initialPos: [number, number, number];
    initialQuat: [number, number, number, number];
    initialVel?: [number, number, number];
}

export interface FixtureSpring {
    a: number; // body index (creation order)
    b: number;
    rA: [number, number, number];
    rB: [number, number, number];
    stiffness: number;
    rest: number;
}

export interface FixtureJoint {
    a: number; // body index (creation order)
    b: number;
    rA: [number, number, number];
    rB: [number, number, number];
    stiffnessLin: number; // the 1e30 sentinel = Infinity (harness-dense.cpp)
    stiffnessAng: number;
}

export interface Frame {
    frame: number;
    pos: number[]; // 3 floats per body
    quat: number[]; // 4 floats per body
    vel: number[]; // 3 floats per body
    angVel: number[]; // 3 floats per body
}

export interface Fixture {
    scene: string;
    params: FixtureParams;
    bodyCount: number;
    bodies: FixtureBody[];
    springs?: FixtureSpring[];
    joints?: FixtureJoint[];
    frames: Frame[];
}

/** the harness emits 1e30 for an INFINITY stiffness (JSON can't carry Infinity) — map it back */
const unInf = (v: number): number => (v >= 1e29 ? Number.POSITIVE_INFINITY : v);

export function loadFixture(set: ParamSet, scene: string): Fixture {
    const path = resolve(import.meta.dir, "..", "fixtures", "avbd", set, `dense-${scene}.json`);
    return JSON.parse(readFileSync(path, "utf8")) as Fixture;
}

/** A solver seeded from the fixture's initial bodies + params, in the full-AVBD `warmstart` layer. */
export function fixtureSolver(fx: Fixture): Solver {
    const bodies = fx.bodies.map((b) =>
        body(b.size, b.mass, b.friction, b.initialPos, b.initialVel ?? [0, 0, 0], b.initialQuat),
    );
    const p = fx.params;
    const s = makeSolver(bodies, {
        dt: p.dt,
        gravity: p.gravity,
        iterations: p.iterations,
        alpha: p.alpha,
        betaLin: p.betaLin,
        betaAng: p.betaAng,
        gamma: p.gamma,
        layer: "warmstart",
    });
    for (const sp of fx.springs ?? [])
        s.springs.push(spring(bodies[sp.a], bodies[sp.b], sp.rA, sp.rB, sp.stiffness, sp.rest));
    for (const jt of fx.joints ?? [])
        s.joints.push(
            joint(
                bodies[jt.a],
                bodies[jt.b],
                jt.rA,
                jt.rB,
                unInf(jt.stiffnessLin),
                unInf(jt.stiffnessAng),
            ),
        );
    return s;
}

/** body `i`'s reference position at a frame */
export const framePos = (f: Frame, i: number): Vec3 => [
    f.pos[i * 3],
    f.pos[i * 3 + 1],
    f.pos[i * 3 + 2],
];
