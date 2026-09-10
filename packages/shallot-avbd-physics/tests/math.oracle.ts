import { describe, expect, test } from "bun:test";
import {
    dot,
    length,
    type Mat3,
    normalize,
    orthonormal,
    type Quat,
    qadd,
    qinverse,
    qmul,
    qnormalize,
    qsub,
    solve,
    type Vec3,
} from "./math";

// Only the index-heavy / boundary-crossing ports are tested — the bits where a
// transcription bug from maths.h would hide. The trivial componentwise ops restate
// their implementation, so they're left to the consumers (solver.test, sat.test).

// deterministic LCG so the SPD systems are reproducible
function lcg(seed: number): () => number {
    let s = seed >>> 0 || 1;
    return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    };
}

// full symmetric 6×6 matvec, the reference the block-storage solve must reproduce
function matvec6(a: number[][], x: number[]): number[] {
    const out = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) out[i] += a[i][j] * x[j];
    return out;
}

describe("math — quaternion operators", () => {
    test("q · q⁻¹ = identity for a unit quat", () => {
        const q = qnormalize([0.1, 0.7, -0.3, 0.4]);
        const id = qmul(q, qinverse(q));
        expect(id[0]).toBeCloseTo(0, 12);
        expect(id[1]).toBeCloseTo(0, 12);
        expect(id[2]).toBeCloseTo(0, 12);
        expect(id[3]).toBeCloseTo(1, 12);
    });

    test("qsub inverts qadd for a small angular increment", () => {
        // q ⊕ ω then ⊖ q recovers ω to first order; |ω|=2e-3 ⇒ O(|ω|²)≈4e-6 error
        const q: Quat = qnormalize([0.2, 0.5, -0.1, 0.9]);
        const omega: Vec3 = [1e-3, -1.5e-3, 0.8e-3];
        const recovered = qsub(qadd(q, omega), q);
        for (let i = 0; i < 3; i++) expect(recovered[i]).toBeCloseTo(omega[i], 5);
    });
});

describe("math — orthonormal basis", () => {
    for (const n of [
        normalize([0, 1, 0] as Vec3),
        normalize([1, 0, 0] as Vec3),
        normalize([0.3, 0.4, -0.87] as Vec3),
        normalize([-0.9, 0.1, 0.05] as Vec3),
    ]) {
        test(`rows orthonormal, first row = normal (${n.map((x) => x.toFixed(2))})`, () => {
            const m = orthonormal(n);
            expect(m[0]).toEqual(n);
            expect(length(m[1])).toBeCloseTo(1, 12);
            expect(length(m[2])).toBeCloseTo(1, 12);
            expect(dot(m[0], m[1])).toBeCloseTo(0, 12);
            expect(dot(m[0], m[2])).toBeCloseTo(0, 12);
            expect(dot(m[1], m[2])).toBeCloseTo(0, 12);
        });
    }
});

describe("math — 6×6 LDLᵀ solve", () => {
    // A = BBᵀ + I is SPD with eigenvalues ≥ 1 (well-conditioned). Build a known x,
    // form b = Ax, decompose A into the lin/ang/cross block storage the solve reads,
    // and check it recovers x. This exercises every index in the LDLᵀ transcription.
    test("recovers x from b = Ax across seeds", () => {
        for (let seed = 1; seed <= 8; seed++) {
            const rand = lcg(seed * 2654435761);
            const b: number[][] = Array.from({ length: 6 }, () =>
                Array.from({ length: 6 }, () => rand() * 2 - 1),
            );
            const a: number[][] = Array.from({ length: 6 }, (_, i) =>
                Array.from({ length: 6 }, (_, j) => {
                    let s = i === j ? 1 : 0;
                    for (let k = 0; k < 6; k++) s += b[i][k] * b[j][k];
                    return s;
                }),
            );

            const xTrue = Array.from({ length: 6 }, () => rand() * 4 - 2);
            const rhs = matvec6(a, xTrue);

            const aLin: Mat3 = [
                [a[0][0], a[0][1], a[0][2]],
                [a[1][0], a[1][1], a[1][2]],
                [a[2][0], a[2][1], a[2][2]],
            ];
            const aAng: Mat3 = [
                [a[3][3], a[3][4], a[3][5]],
                [a[4][3], a[4][4], a[4][5]],
                [a[5][3], a[5][4], a[5][5]],
            ];
            // cross[r][c] = A[3+r][c] (bottom-left block, rows 4-6 × cols 1-3)
            const aCross: Mat3 = [
                [a[3][0], a[3][1], a[3][2]],
                [a[4][0], a[4][1], a[4][2]],
                [a[5][0], a[5][1], a[5][2]],
            ];

            const { xLin, xAng } = solve(
                aLin,
                aAng,
                aCross,
                [rhs[0], rhs[1], rhs[2]],
                [rhs[3], rhs[4], rhs[5]],
            );
            const got = [...xLin, ...xAng];
            for (let i = 0; i < 6; i++) expect(got[i]).toBeCloseTo(xTrue[i], 9);
        }
    });
});
