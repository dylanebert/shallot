import { beforeEach, describe, expect, test } from "bun:test";
import { State } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { Slab } from "../slab";
import { composeTransform, Transform } from "./index";

// Pre-test gate for the transform-firehose decompose-on-read change (roadmap render-perf Stage 1).
// The firehose's content moves from a materialized mat4 to a decomposed {pos, scale, quat} struct every
// reader reconstructs (niagara's reconstruct-on-read). These pin the math spec the change must meet,
// against the production matrix compose (`composeTransform`) as the non-circular anchor:
//   1. the decomposed application reproduces the matrix path (lossless vs today), and
//   2. the decomposed normal transform `rotate(n / scale, q)` is the correct inverse-transpose normal —
//      which today's `(R·S)·n` is NOT under non-uniform scale (the latent bug the change fixes).
// The GPU twin (the WGSL `transform()` helper) is gated separately by the Sub-stage 1 render oracle on
// the real device; this is the CPU spec it implements. A gate, not a bug-fix TDD cycle, so it's green now.

// the decomposed reader's rotation: active quaternion rotation of v (the convention COMPOSE_WGSL /
// physics qRotateW use). Independent expression of the same rotation composeTransform bakes into a matrix.
function rotate(q: number[], v: number[]): number[] {
    const [x, y, z, w] = q;
    const tx = 2 * (y * v[2] - z * v[1]);
    const ty = 2 * (z * v[0] - x * v[2]);
    const tz = 2 * (x * v[1] - y * v[0]);
    return [
        v[0] + w * tx + y * tz - z * ty,
        v[1] + w * ty + z * tx - x * tz,
        v[2] + w * tz + x * ty - y * tx,
    ];
}

function normalize(v: number[]): number[] {
    const l = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / l, v[1] / l, v[2] / l];
}

// upper-left 3×3 of a column-major 16-float mat4, returned as rows
function mat3(m: Float32Array): number[][] {
    return [
        [m[0], m[4], m[8]],
        [m[1], m[5], m[9]],
        [m[2], m[6], m[10]],
    ];
}

function mul3(a: number[][], v: number[]): number[] {
    return [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ];
}

// the textbook normal matrix: inverse-transpose = cofactor / det. Generic (no R·S assumption), so it's
// an independent oracle for the closed-form `rotate(n / scale, q)` the decomposed reader will use.
function invTranspose3(a: number[][]): number[][] {
    const [a00, a01, a02] = a[0];
    const [a10, a11, a12] = a[1];
    const [a20, a21, a22] = a[2];
    const c00 = a11 * a22 - a12 * a21;
    const c01 = -(a10 * a22 - a12 * a20);
    const c02 = a10 * a21 - a11 * a20;
    const c10 = -(a01 * a22 - a02 * a21);
    const c11 = a00 * a22 - a02 * a20;
    const c12 = -(a00 * a21 - a01 * a20);
    const c20 = a01 * a12 - a02 * a11;
    const c21 = -(a00 * a12 - a02 * a10);
    const c22 = a00 * a11 - a01 * a10;
    const det = a00 * c00 + a01 * c01 + a02 * c02;
    return [
        [c00 / det, c01 / det, c02 / det],
        [c10 / det, c11 / det, c12 / det],
        [c20 / det, c21 / det, c22 / det],
    ];
}

// a non-axis-aligned rotation (~50° about (1,1,1)) — exercises every matrix entry, not just diagonals
const Q = (() => {
    const a = (50 * Math.PI) / 180;
    const s = Math.sin(a / 2) / Math.sqrt(3);
    return [s, s, s, Math.cos(a / 2)];
})();

describe("transform firehose decompose-on-read (pre-test gate)", () => {
    let state: State;

    beforeEach(() => {
        clear();
        register("Transform", Transform);
        Slab.collect();
        state = new State();
    });

    function pose(pos: number[], q: number[], scale: number[]): Float32Array {
        const eid = state.create();
        state.add(eid, Transform);
        Transform.pos.set(eid, pos[0], pos[1], pos[2], 0);
        Transform.rot.set(eid, q[0], q[1], q[2], q[3]);
        Transform.scale.set(eid, scale[0], scale[1], scale[2], 0);
        return composeTransform(eid, new Float32Array(16));
    }

    test("position: matrix apply == rotate(p·scale, q) + pos (reconstruct is lossless)", () => {
        const pos = [3, -2, 5];
        const scale = [2, 0.5, 1.5]; // non-uniform
        const m = pose(pos, Q, scale);
        for (const p of [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.3, -0.7, 0.2],
            [1, 1, 1],
        ]) {
            const mp = [
                m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
                m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
                m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
            ];
            const r = rotate(Q, [p[0] * scale[0], p[1] * scale[1], p[2] * scale[2]]);
            const dp = [r[0] + pos[0], r[1] + pos[1], r[2] + pos[2]];
            // precision 4 (|Δ| < 5e-5): `m` is f32 (Float32Array out), the rotate path is f64 — the gap is
            // f32 rounding of the matrix entries × O(10) coords, ~1e-5 abs (testing.md f32 tier)
            for (let i = 0; i < 3; i++) expect(dp[i]).toBeCloseTo(mp[i], 4);
        }
    });

    test("normal: rotate(n / scale, q) == generic inverse-transpose normal", () => {
        const scale = [2, 0.5, 1.5];
        const m = pose([0, 0, 0], Q, scale);
        const nt = invTranspose3(mat3(m));
        for (const n of [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.6, -0.8, 0],
        ]) {
            const fixed = normalize(rotate(Q, [n[0] / scale[0], n[1] / scale[1], n[2] / scale[2]]));
            const correct = normalize(mul3(nt, n));
            for (let i = 0; i < 3; i++) expect(fixed[i]).toBeCloseTo(correct[i], 5);
        }
    });

    test("normal: today's (R·S)·n is wrong under non-uniform scale, correct under uniform", () => {
        const n = [0.6, -0.8, 0];
        // non-uniform: the current worldNormal diverges hard from the inverse-transpose normal
        {
            const m = pose([0, 0, 0], Q, [2, 0.5, 1.5]);
            const current = normalize(mul3(mat3(m), n)); // (R·S)·n — sear's worldNormal today
            const correct = normalize(mul3(invTranspose3(mat3(m)), n));
            const dot = current[0] * correct[0] + current[1] * correct[1] + current[2] * correct[2];
            expect(dot).toBeLessThan(0.99); // measurably wrong (≈0.49 for this pose)
        }
        // uniform: a positive uniform scale normalizes out, so the current path already matches — no regression
        {
            const m = pose([0, 0, 0], Q, [1.3, 1.3, 1.3]);
            const current = normalize(mul3(mat3(m), n));
            const correct = normalize(mul3(invTranspose3(mat3(m)), n));
            for (let i = 0; i < 3; i++) expect(current[i]).toBeCloseTo(correct[i], 5);
        }
    });
});
