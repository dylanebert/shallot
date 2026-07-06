import { describe, expect, test } from "bun:test";
import { invert, lookAt, multiply, orthographic, perspective } from "../../engine";
import {
    type FogLight,
    type FogScatter,
    type FogSun,
    fogComposite,
    fogDensity,
    fogInScatter,
    fogSunInScatter,
    fogTransmittance,
    heightOpticalDepth,
    henyeyGreenstein,
    inScatterContribution,
    reconstruct,
    sunInScatter,
} from "./march";

// project a world point through a column-major viewProj → ndc (after the perspective divide).
function project(m: Float32Array, p: readonly [number, number, number]): [number, number, number] {
    const [x, y, z] = p;
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const cz = m[2] * x + m[6] * y + m[10] * z + m[14];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    return [cx / cw, cy / cw, cz / cw];
}

describe("fog reconstruct", () => {
    // round-trip: a world point projected to (uv, depth) reconstructs back to itself through invViewProj.
    // The matrices are Float32Array (the GPU's precision), so the round-trip error is f32 matrix storage
    // (~1e-7 relative per entry) amplified through the perspective divide — bounded ~1e-3 for a mid-frustum
    // point. A wrong y-flip or axis swap is O(|P|) and fails loudly past this; the actual orientation
    // against WebGPU's framebuffer convention is validated by the gym `render` fog probe height-fog screenshot.
    const Tol = 1e-3;

    test("perspective round-trip", () => {
        const view = lookAt(4, 3, 8, 0, 1, 0);
        const proj = perspective(60, 16 / 9, 0.1, 1000);
        const vp = multiply(proj, view);
        const inv = invert(vp);
        for (const p of [
            [0, 1, 0],
            [-2, 0.5, 1.5],
            [3, 2, -1],
        ] as const) {
            const ndc = project(vp, p);
            expect(ndc[2]).toBeGreaterThan(0);
            expect(ndc[2]).toBeLessThan(1);
            const u = (ndc[0] + 1) / 2;
            const v = (1 - ndc[1]) / 2;
            const got = reconstruct(inv, u, v, ndc[2]);
            expect(Math.abs(got[0] - p[0])).toBeLessThan(Tol);
            expect(Math.abs(got[1] - p[1])).toBeLessThan(Tol);
            expect(Math.abs(got[2] - p[2])).toBeLessThan(Tol);
        }
    });

    test("orthographic round-trip", () => {
        const view = lookAt(0, 6, 6, 0, 0, 0);
        const proj = orthographic(5, 16 / 9, 0.1, 100);
        const vp = multiply(proj, view);
        const inv = invert(vp);
        const p = [1, 0.5, -1] as const;
        const ndc = project(vp, p);
        expect(ndc[2]).toBeGreaterThan(0);
        expect(ndc[2]).toBeLessThan(1);
        const u = (ndc[0] + 1) / 2;
        const v = (1 - ndc[1]) / 2;
        const got = reconstruct(inv, u, v, ndc[2]);
        expect(Math.abs(got[0] - p[0])).toBeLessThan(Tol);
        expect(Math.abs(got[1] - p[1])).toBeLessThan(Tol);
        expect(Math.abs(got[2] - p[2])).toBeLessThan(Tol);
    });
});

describe("fog extinction", () => {
    // Beer-Lambert: uniform fog (falloff 0) integrates exactly to exp(-density·dist) for any step count or
    // sample offset — the midpoint sum of a constant is exact. This pins the per-step transmittance law.
    test("uniform fog is exp(-density·dist), step/offset invariant", () => {
        const density = 0.07;
        const dist = 23;
        const want = Math.exp(-density * dist);
        for (const steps of [1, 4, 32]) {
            for (const offset of [0, 0.5, 0.9]) {
                const got = fogTransmittance(5, -0.4, dist, density, 0, 0, steps, offset);
                expect(got).toBeCloseTo(want, 6);
            }
        }
    });

    // the midpoint march of exponential height fog converges to the closed-form optical depth within the
    // midpoint-rule error bound |E| ≤ L³/(24·n²)·max|f''|, f''(t) = a·k²·e^{-kt}. A derived tolerance:
    // the bound is the math, not a tuned number, and it shrinks ~1/n² as steps rise.
    test("height-fog march matches the closed form within the midpoint bound", () => {
        const configs = [
            { originY: 10, dirY: -0.5, dist: 40, density: 0.03, base: 0, falloff: 0.1 },
            { originY: 2, dirY: 0.35, dist: 30, density: 0.05, base: 1, falloff: 0.2 },
        ];
        for (const c of configs) {
            const a = c.density * Math.exp(-c.falloff * (c.originY - c.base));
            const k = c.falloff * c.dirY;
            const maxF2 = a * k * k * Math.max(1, Math.exp(-k * c.dist));
            let prev = Infinity;
            for (const steps of [8, 32, 128]) {
                const t = fogTransmittance(
                    c.originY,
                    c.dirY,
                    c.dist,
                    c.density,
                    c.base,
                    c.falloff,
                    steps,
                    0.5,
                );
                const tauMarch = -Math.log(t);
                const tauExact = heightOpticalDepth(
                    c.originY,
                    c.dirY,
                    c.dist,
                    c.density,
                    c.base,
                    c.falloff,
                );
                const bound = (c.dist ** 3 / (24 * steps * steps)) * maxF2;
                const err = Math.abs(tauMarch - tauExact);
                expect(err).toBeLessThanOrEqual(bound + 1e-9);
                expect(err).toBeLessThanOrEqual(prev);
                prev = err;
            }
        }
    });

    // a horizontal ray (dirY 0 → k 0) hits the closed form's degenerate branch: density is constant along
    // it, so τ = a·dist and the midpoint march is exact for any step count.
    test("horizontal ray integrates constant height-density exactly", () => {
        const originY = 5;
        const dist = 20;
        const density = 0.04;
        const base = 0;
        const falloff = 0.15;
        const wantTau = density * Math.exp(-falloff * (originY - base)) * dist;
        expect(heightOpticalDepth(originY, 0, dist, density, base, falloff)).toBeCloseTo(
            wantTau,
            10,
        );
        const t = fogTransmittance(originY, 0, dist, density, base, falloff, 16, 0.5);
        expect(-Math.log(t)).toBeCloseTo(wantTau, 6);
    });
});

describe("fog composite", () => {
    test("transmittance 1 keeps the scene, 0 is full haze, 0.5 lerps", () => {
        const scene = [0.8, 0.4, 0.2] as const;
        const haze = [0.5, 0.6, 0.7] as const;
        expect(fogComposite(scene, haze, 1)).toEqual([0.8, 0.4, 0.2]);
        expect(fogComposite(scene, haze, 0)).toEqual([0.5, 0.6, 0.7]);
        const mid = fogComposite(scene, haze, 0.5);
        expect(mid[0]).toBeCloseTo(0.65, 6);
        expect(mid[1]).toBeCloseTo(0.5, 6);
        expect(mid[2]).toBeCloseTo(0.45, 6);
    });
});

describe("fog phase", () => {
    // Henyey-Greenstein is a normalized phase function — these are its defining properties, independent of
    // the closed form's algebra.
    test("isotropic at g=0 is 1/4π for any angle", () => {
        for (const c of [-1, -0.3, 0, 0.7, 1]) {
            expect(henyeyGreenstein(0, c)).toBeCloseTo(1 / (4 * Math.PI), 12);
        }
    });

    test("symmetric under (g, cosθ) → (−g, −cosθ)", () => {
        for (const [g, c] of [
            [0.5, 0.3],
            [0.8, -0.6],
            [-0.4, 0.9],
        ] as const) {
            expect(henyeyGreenstein(g, c)).toBeCloseTo(henyeyGreenstein(-g, -c), 12);
        }
    });

    // a phase function integrates to 1 over the sphere: ∫ p dΩ = 2π ∫_{-1}^{1} p(μ) dμ = 1. Trapezoidal
    // quadrature over a smooth integrand (g away from ±1) — the error is the quadrature's, under 1e-3 at
    // N=20000, derived not tuned.
    test("normalizes to 1 over the sphere", () => {
        const N = 20000;
        for (const g of [0, 0.3, -0.6, 0.85]) {
            let sum = 0;
            for (let i = 0; i <= N; i++) {
                const mu = -1 + (2 * i) / N;
                const w = i === 0 || i === N ? 0.5 : 1;
                sum += w * henyeyGreenstein(g, mu);
            }
            const integral = 2 * Math.PI * sum * (2 / N);
            expect(integral).toBeCloseTo(1, 3);
        }
    });
});

describe("fog in-scatter", () => {
    const light: FogLight = {
        pos: [2, 9, 0],
        invRangeSq: 1 / (12 * 12),
        radius: 0.1,
        color: [1, 0.9, 0.7],
        coneAxis: [0, 0, 0],
        coneScale: 0,
        coneOffset: 1,
    };
    const fog: FogScatter = {
        density: 0.05,
        base: 0,
        falloff: 0.1,
        absorption: 0.2,
        gain: 1.5,
        anisotropy: 0.3,
    };
    const origin = [0, 1, 5] as const;
    const dir = [0, 0, -1] as const;
    const dist = 10;

    // a single step has transmittance-to-start 1, so the march reduces to the energy-conserving per-step
    // integral: albedo·gain·(1−e^{−σt·ds})·contribution, with ds = dist and σt the midpoint density. Pins the
    // analytic per-step assembly — a rectangle-rule regression (`σs·gain·ds`) fails here, since
    // (1−e^{−σt·ds}) ≠ σt·ds, as do dropped albedo/gain factors.
    test("a single step integrates the source analytically at T=1", () => {
        const got = fogInScatter(origin, dir, dist, fog, light, 1, 0.5);
        const p = [origin[0], origin[1], origin[2] + dir[2] * 0.5 * dist] as const;
        const d = fogDensity(p[1], fog.density, fog.base, fog.falloff);
        const w = (1 - fog.absorption) * fog.gain * (1 - Math.exp(-d * dist));
        const c = inScatterContribution(light, p, dir, fog.anisotropy);
        for (let i = 0; i < 3; i++) {
            expect(got[i]).toBeCloseTo(w * c[i], 12);
        }
    });

    // in-scatter is linear in the scatter gain — doubling `gain` doubles the radiance (a gain² or
    // partial-gain bug breaks this), and the transmittance weighting makes a multi-step march strictly dimmer
    // than the un-attenuated sum (the gain·2 case still scales exactly, T factors out of the gain).
    test("scales linearly with gain", () => {
        const a = fogInScatter(origin, dir, dist, fog, light, 16, 0.5);
        const b = fogInScatter(origin, dir, dist, { ...fog, gain: fog.gain * 2 }, light, 16, 0.5);
        for (let i = 0; i < 3; i++) expect(b[i]).toBeCloseTo(2 * a[i], 12);
        expect(a[1]).toBeGreaterThan(0);
    });

    // the energy-conserving form folds the within-step extinction in exactly, leaving only the midpoint
    // sampling of the (smooth — this light has coneScale 0, no cone discontinuity) source, which is
    // second-order: each doubling of the step count quarters the error (err ∝ 1/n²). A rectangle-rule `·ds`
    // march is only first-order — its O(σt·ds) within-step over-brightening halves per doubling, ~2× — so the
    // ≥3.5× bound (between the first-order 2 and second-order 4) fails for it. The single-step test pins the
    // per-step form; this pins the convergence order, which a rectangle regression can't fake.
    test("in-scatter converges second-order in step count", () => {
        const ref = fogInScatter(origin, dir, dist, fog, light, 2048, 0.5);
        const err = (steps: number) => {
            const got = fogInScatter(origin, dir, dist, fog, light, steps, 0.5);
            return Math.hypot(got[0] - ref[0], got[1] - ref[1], got[2] - ref[2]);
        };
        for (const n of [8, 16, 32]) {
            expect(err(n) / err(2 * n)).toBeGreaterThanOrEqual(3.5);
        }
    });
});

describe("fog sun in-scatter", () => {
    // a downward sun (travel direction); toward-light is its negation, (0, 1, 0)
    const sun: FogSun = { direction: [0, -1, 0], color: [1, 0.95, 0.8] };
    const fog: FogScatter = {
        density: 0.05,
        base: 0,
        falloff: 0.1,
        absorption: 0.2,
        gain: 1.5,
        anisotropy: 0.4,
    };
    const origin = [0, 1, 5] as const;
    const dir = [0, 0, -1] as const;
    const dist = 10;

    // the sun contribution is color · HG(g, dot(dir, -sunDir)); with no atten/cone it's just the phase
    // toward the light. Pins the twin against the separately-tested henyeyGreenstein.
    test("contribution is color · HG(g, dot(dir, -sunDir))", () => {
        const viewDir = [0.3, 0.6, -0.74] as const; // arbitrary unit-ish ray
        const got = sunInScatter(sun.color, sun.direction, viewDir, fog.anisotropy);
        const cosTheta = -(
            viewDir[0] * sun.direction[0] +
            viewDir[1] * sun.direction[1] +
            viewDir[2] * sun.direction[2]
        );
        const phase = henyeyGreenstein(fog.anisotropy, cosTheta);
        for (let i = 0; i < 3; i++) expect(got[i]).toBeCloseTo(sun.color[i] * phase, 12);
    });

    // forward anisotropy (g>0) brightens the shaft when the view ray points toward the light vs away — the
    // crepuscular halo. A ray straight toward the light (dir = -sunDir) scatters more than straight away.
    test("forward-peaked toward the light at g > 0", () => {
        const toward = sunInScatter(sun.color, sun.direction, [0, 1, 0], 0.6);
        const away = sunInScatter(sun.color, sun.direction, [0, -1, 0], 0.6);
        expect(toward[0]).toBeGreaterThan(away[0]);
    });

    // a single step has transmittance-to-start 1, so the march reduces to the same energy-conserving per-step
    // integral as the clustered march — albedo·gain·(1−e^{−σt·ds})·contribution — with the sun integrand. A
    // rectangle-rule `·ds` regression or a dropped albedo/gain factor goes red.
    test("a single step integrates the source analytically at T=1", () => {
        const got = fogSunInScatter(origin, dir, dist, fog, sun, 1, 0.5);
        const py = origin[1] + dir[1] * 0.5 * dist;
        const d = fogDensity(py, fog.density, fog.base, fog.falloff);
        const w = (1 - fog.absorption) * fog.gain * (1 - Math.exp(-d * dist));
        const c = sunInScatter(sun.color, sun.direction, dir, fog.anisotropy);
        for (let i = 0; i < 3; i++) expect(got[i]).toBeCloseTo(w * c[i], 12);
    });

    // linear in gain, and the transmittance weighting makes a multi-step march strictly positive (the ray is
    // lit) — the gain·2 case scales exactly (T factors out of gain).
    test("scales linearly with gain", () => {
        const a = fogSunInScatter(origin, dir, dist, fog, sun, 16, 0.5);
        const b = fogSunInScatter(origin, dir, dist, { ...fog, gain: fog.gain * 2 }, sun, 16, 0.5);
        for (let i = 0; i < 3; i++) expect(b[i]).toBeCloseTo(2 * a[i], 12);
        expect(a[0]).toBeGreaterThan(0);
    });
});
