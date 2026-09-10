import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { invert, lookAt, multiply, orthographic, perspective } from "../../engine";
import { octEncode } from "../../engine/utils/core";
import { distanceAttenuation, PointLightGpu, spotFactor, spotParams } from "../render/core";
import {
    FOG_BYTES,
    FOG_FLOATS,
    FOG_MAX_STEPS,
    FOG_PARAMS,
    type FogScatter,
    type FogSun,
    fogComposite,
    fogDensity,
    fogInScatter,
    fogInScatterWgsl,
    fogMarchWgsl,
    fogStructWgsl,
    fogSunInScatter,
    fogTransmittance,
    heightOpticalDepth,
    henyeyGreenstein,
    inScatterContribution,
    reconstructWorld,
    sunInScatter,
} from "./march";

// Every march primitive is one TGSL function, so these tests exercise the SAME source the shader runs — its
// CPU arm. That makes each schema-typed argument and return f32-rounded, so a tolerance against an f64 closed
// form is bounded by f32 epsilon (2⁻²⁴ ≈ 6e-8 relative), not by the arithmetic. Where the only difference is
// the f32 rounding of a final product, the expectation is `Math.fround`ed and the assertion stays exact.
const F32_REL = 4 * 2 ** -24; // a few f32 ulps — the derived bound for "the f64 closed form, in f32"

// a column-major Float32Array (the engine's matrix form) as the `mat4x4f` the TGSL reconstruct takes
const mat = (m: Float32Array) =>
    d.mat4x4f(
        d.vec4f(m[0], m[1], m[2], m[3]),
        d.vec4f(m[4], m[5], m[6], m[7]),
        d.vec4f(m[8], m[9], m[10], m[11]),
        d.vec4f(m[12], m[13], m[14], m[15]),
    );

// the f32 whose BITS are `enc` — how a compacted light carries its oct-packed cone axis in `params.y`
// (`spotFactor` bitcasts it back), so the CPU arm decodes the same axis the shader does
function axisLane(enc: number): number {
    const f = new Float32Array(1);
    new Uint32Array(f.buffer)[0] = enc;
    return f[0];
}

// project a world point through a column-major viewProj → ndc (after the perspective divide).
function project(m: Float32Array, p: readonly [number, number, number]): [number, number, number] {
    const [x, y, z] = p;
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const cz = m[2] * x + m[6] * y + m[10] * z + m[14];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    return [cx / cw, cy / cw, cz / cw];
}

describe("fog uniform layout", () => {
    // the numbers a raw binding and the staging writer are sized by. The schema is their one source, so this
    // pins what the WGSL struct actually is — a field reorder or a widened lane shows up here
    test("sizes and packFog's write indices derive from the schema", () => {
        expect(FOG_BYTES).toBe(48);
        expect(FOG_FLOATS).toBe(12);
        expect(FOG_PARAMS).toEqual({ color: 0, march: 4, extra: 8 });
    });
});

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
        const inv = mat(invert(vp));
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
            const got = reconstructWorld(inv, d.vec2f(u, v), ndc[2]);
            expect(Math.abs(got.x - p[0])).toBeLessThan(Tol);
            expect(Math.abs(got.y - p[1])).toBeLessThan(Tol);
            expect(Math.abs(got.z - p[2])).toBeLessThan(Tol);
        }
    });

    test("orthographic round-trip", () => {
        const view = lookAt(0, 6, 6, 0, 0, 0);
        const proj = orthographic(5, 16 / 9, 0.1, 100);
        const vp = multiply(proj, view);
        const inv = mat(invert(vp));
        const p = [1, 0.5, -1] as const;
        const ndc = project(vp, p);
        expect(ndc[2]).toBeGreaterThan(0);
        expect(ndc[2]).toBeLessThan(1);
        const u = (ndc[0] + 1) / 2;
        const v = (1 - ndc[1]) / 2;
        const got = reconstructWorld(inv, d.vec2f(u, v), ndc[2]);
        expect(Math.abs(got.x - p[0])).toBeLessThan(Tol);
        expect(Math.abs(got.y - p[1])).toBeLessThan(Tol);
        expect(Math.abs(got.z - p[2])).toBeLessThan(Tol);
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
                const got = fogTransmittance(
                    d.vec3f(0, 5, 0),
                    d.vec3f(0, -0.4, 0),
                    dist,
                    density,
                    0,
                    0,
                    steps,
                    offset,
                );
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
                    d.vec3f(0, c.originY, 0),
                    d.vec3f(0, c.dirY, 0),
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
        const t = fogTransmittance(
            d.vec3f(0, originY, 0),
            d.vec3f(0, 0, 0),
            dist,
            density,
            base,
            falloff,
            16,
            0.5,
        );
        expect(-Math.log(t)).toBeCloseTo(wantTau, 6);
    });
});

describe("fog composite", () => {
    test("transmittance 1 keeps the scene, 0 is full haze, 0.5 lerps", () => {
        const scene = d.vec3f(0.8, 0.4, 0.2);
        const haze = d.vec3f(0.5, 0.6, 0.7);
        // T = 1 / 0 are the exact rails: the scene (or haze) survives bit-for-bit through the f32 lanes
        const keep = fogComposite(scene, haze, 1);
        expect([keep.x, keep.y, keep.z]).toEqual([scene.x, scene.y, scene.z]);
        const full = fogComposite(scene, haze, 0);
        expect([full.x, full.y, full.z]).toEqual([haze.x, haze.y, haze.z]);
        const mid = fogComposite(scene, haze, 0.5);
        expect(mid.x).toBeCloseTo(0.65, 6);
        expect(mid.y).toBeCloseTo(0.5, 6);
        expect(mid.z).toBeCloseTo(0.45, 6);
    });
});

describe("fog phase", () => {
    // Henyey-Greenstein is a normalized phase function — these are its defining properties, independent of
    // the closed form's algebra.
    test("isotropic at g=0 is 1/4π for any angle", () => {
        const want = 1 / (4 * Math.PI);
        for (const c of [-1, -0.3, 0, 0.7, 1]) {
            expect(Math.abs(henyeyGreenstein(0, c) - want) / want).toBeLessThan(F32_REL);
        }
    });

    test("symmetric under (g, cosθ) → (−g, −cosθ)", () => {
        // negating both operands negates only the sign of `2·g·cosθ`'s factors, so the f32 arithmetic is
        // bit-identical — the symmetry is exact, not approximate
        for (const [g, c] of [
            [0.5, 0.3],
            [0.8, -0.6],
            [-0.4, 0.9],
        ] as const) {
            expect(henyeyGreenstein(g, c)).toBe(henyeyGreenstein(-g, -c));
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

describe("fog in-scatter contribution", () => {
    // a spot light 4 units straight up from the march point: `toLight` is (0, 4, 0) and inverseSqrt(16) is
    // exactly 0.25, so the function's internal L is exactly (0, 1, 0) and the geometry is known without
    // reimplementing it. The cone axis is oct-packed into params.y (bitcast), the shipped carry.
    const AxisEnc = octEncode(0.6, -0.8, 0);
    const { scale, offset } = spotParams(30, 55);
    const Range = 12;
    const Radius = 0.5;
    const light = PointLightGpu({
        posRange: d.vec4f(0, 4, 0, 1 / (Range * Range)),
        color: d.vec4f(1, 0.9, 0.7, 0),
        // params.x is NEGATIVE: the sign is the Volumetric flag, the magnitude the source radius
        params: d.vec4f(-Radius, axisLane(AxisEnc), scale, offset),
    });
    const p = d.vec3f(0, 0, 0);
    const L = d.vec3f(0, 1, 0);

    // the contribution IS color · distanceAttenuation · spotFactor · phase — each factor separately tested
    // (lighting.test.ts for the first two, "fog phase" above for the third). Composition, not a second copy.
    test("is the product of the attenuation, the cone factor and the phase", () => {
        const atten = distanceAttenuation(16, light.posRange.w, Radius * Radius);
        const spot = spotFactor(light, L);
        expect(spot).toBeGreaterThan(0);
        expect(spot).toBeLessThan(1); // the sample sits between the inner and outer cone
        for (const dir of [d.vec3f(0, 1, 0), d.vec3f(0, 0, -1), d.vec3f(0, -1, 0)]) {
            for (const g of [0, 0.6, -0.4]) {
                const phase = henyeyGreenstein(g, dir.y); // dot(dir, (0,1,0)) = dir.y
                const got = inScatterContribution(light, p, dir, g);
                const f = atten * spot * phase;
                expect(got.x).toBe(Math.fround(light.color.x * f));
                expect(got.y).toBe(Math.fround(light.color.y * f));
                expect(got.z).toBe(Math.fround(light.color.z * f));
            }
        }
    });

    // params.x's sign is the Volumetric opt-in flag and the radius is squared on read, so the two signs must
    // give identical radiance — the convention the compact pass encodes. The march point sits INSIDE the
    // source sphere (dist 1, radius 3), so `max(distSq, radiusSq)` selects the radius term — the only regime
    // where an unsquared params.x read is observable at all.
    const bulb = (radius: number) =>
        PointLightGpu({
            posRange: d.vec4f(0, 1, 0, 1 / (Range * Range)),
            color: d.vec4f(1, 0.9, 0.7, 0),
            params: d.vec4f(radius, axisLane(AxisEnc), scale, offset),
        });

    test("params.x's sign is a flag, not part of the radius", () => {
        const a = inScatterContribution(bulb(-3), p, d.vec3f(0, 0, -1), 0.4);
        const b = inScatterContribution(bulb(3), p, d.vec3f(0, 0, -1), 0.4);
        expect(a.x).toBeGreaterThan(0);
        expect([a.x, a.y, a.z]).toEqual([b.x, b.y, b.z]);
    });
});

describe("fog in-scatter", () => {
    const light = PointLightGpu({
        posRange: d.vec4f(2, 9, 0, 1 / (12 * 12)),
        color: d.vec4f(1, 0.9, 0.7, 0),
        // a plain point light: params.z = 0 is spotFactor's early-out, so the cone factor is 1
        params: d.vec4f(0.1, 0, 0, 1),
    });
    const fog: FogScatter = {
        density: 0.05,
        base: 0,
        falloff: 0.1,
        absorption: 0.2,
        gain: 1.5,
        anisotropy: 0.3,
    };
    const origin = d.vec3f(0, 1, 5);
    const dir = d.vec3f(0, 0, -1);
    const dist = 10;

    // a single step has transmittance-to-start 1, so the march reduces to the energy-conserving per-step
    // integral: albedo·gain·(1−e^{−σt·ds})·contribution, with ds = dist and σt the midpoint density. Pins the
    // analytic per-step assembly — a rectangle-rule regression (`σs·gain·ds`) fails here, since
    // (1−e^{−σt·ds}) ≠ σt·ds, as do dropped albedo/gain factors.
    test("a single step integrates the source analytically at T=1", () => {
        const got = fogInScatter(origin, dir, dist, fog, light, 1, 0.5);
        const p = d.vec3f(origin.x, origin.y, origin.z + dir.z * 0.5 * dist);
        const dens = fogDensity(p, fog.density, fog.base, fog.falloff);
        const w = (1 - fog.absorption) * fog.gain * (1 - Math.exp(-dens * dist));
        const c = inScatterContribution(light, p, dir, fog.anisotropy);
        expect(got[0]).toBeCloseTo(w * c.x, 12);
        expect(got[1]).toBeCloseTo(w * c.y, 12);
        expect(got[2]).toBeCloseTo(w * c.z, 12);
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
    // sampling of the (smooth — this light has no cone, so no discontinuity) source, which is second-order:
    // each doubling of the step count quarters the error (err ∝ 1/n²). A rectangle-rule `·ds` march is only
    // first-order — its O(σt·ds) within-step over-brightening halves per doubling, ~2× — so the ≥3.5× bound
    // (between the first-order 2 and second-order 4) fails for it. The single-step test pins the per-step
    // form; this pins the convergence order, which a rectangle regression can't fake.
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
    const sun: FogSun = { direction: d.vec3f(0, -1, 0), color: d.vec3f(1, 0.95, 0.8) };
    const fog: FogScatter = {
        density: 0.05,
        base: 0,
        falloff: 0.1,
        absorption: 0.2,
        gain: 1.5,
        anisotropy: 0.4,
    };
    const origin = d.vec3f(0, 1, 5);
    const dir = d.vec3f(0, 0, -1);
    const dist = 10;

    // the sun contribution is color · HG(g, dot(dir, -sunDir)); with no atten/cone it's just the phase
    // toward the light. Pins the composition against the separately-tested henyeyGreenstein.
    test("contribution is color · HG(g, dot(dir, -sunDir))", () => {
        const viewDir = d.vec3f(0.3, 0.6, -0.74); // arbitrary unit-ish ray
        const got = sunInScatter(sun.color, sun.direction, viewDir, fog.anisotropy);
        // dot(viewDir, -sunDir) with sunDir = (0, -1, 0) is exactly viewDir.y
        const phase = henyeyGreenstein(fog.anisotropy, viewDir.y);
        expect(got.x).toBe(Math.fround(sun.color.x * phase));
        expect(got.y).toBe(Math.fround(sun.color.y * phase));
        expect(got.z).toBe(Math.fround(sun.color.z * phase));
    });

    // forward anisotropy (g>0) brightens the shaft when the view ray points toward the light vs away — the
    // crepuscular halo. A ray straight toward the light (dir = -sunDir) scatters more than straight away.
    test("forward-peaked toward the light at g > 0", () => {
        const toward = sunInScatter(sun.color, sun.direction, d.vec3f(0, 1, 0), 0.6);
        const away = sunInScatter(sun.color, sun.direction, d.vec3f(0, -1, 0), 0.6);
        expect(toward.x).toBeGreaterThan(away.x);
    });

    // a single step has transmittance-to-start 1, so the march reduces to the same energy-conserving per-step
    // integral as the clustered march — albedo·gain·(1−e^{−σt·ds})·contribution — with the sun integrand. A
    // rectangle-rule `·ds` regression or a dropped albedo/gain factor goes red.
    test("a single step integrates the source analytically at T=1", () => {
        const got = fogSunInScatter(origin, dir, dist, fog, sun, 1, 0.5);
        const p = d.vec3f(origin.x, origin.y, origin.z + dir.z * 0.5 * dist);
        const dens = fogDensity(p, fog.density, fog.base, fog.falloff);
        const w = (1 - fog.absorption) * fog.gain * (1 - Math.exp(-dens * dist));
        const c = sunInScatter(sun.color, sun.direction, dir, fog.anisotropy);
        expect(got[0]).toBeCloseTo(w * c.x, 12);
        expect(got[1]).toBeCloseTo(w * c.y, 12);
        expect(got[2]).toBeCloseTo(w * c.z, 12);
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

// The chunks are what a raw-WGSL splice site (the production `FogSystem`, the gym probe) calls into, so the
// emitted names ARE the contract — a rename or a suffix (`Fog_1`) only a shader compile would report.
describe("fog WGSL chunks", () => {
    const defs = (wgsl: string): string[] =>
        [...wgsl.matchAll(/^(?:fn|struct)\s+([A-Za-z0-9_]+)/gm)].map((m) => m[1]);

    test("the uniform struct emits as `Fog`", () => {
        expect(defs(fogStructWgsl())).toEqual(["Fog"]);
    });

    test("the march chunk emits the authored names", () => {
        const src = defs(fogMarchWgsl());
        for (const name of [
            "fogDensity",
            "fogTransmittance",
            "fogComposite",
            "reconstructWorld",
            "ign",
        ])
            expect(src).toContain(name);
    });

    // DXC chokes on a fully-dynamic loop bound (gpu.md), so the march must keep the FOG_MAX_STEPS constant
    // as the bound with a runtime `break` — the captured JS constant has to fold to a literal, not become a
    // uniform read or the dynamic `steps`.
    test("the march loop keeps the folded FOG_MAX_STEPS bound and the dynamic break", () => {
        const src = fogMarchWgsl();
        expect(src).toMatch(new RegExp(`i < ${FOG_MAX_STEPS}u`));
        expect(src).toMatch(/if \(\(i >= steps\)\) \{\s*break;/);
        expect(src).not.toMatch(/while \(\(i < steps\)\)/);
    });

    test("the in-scatter chunk emits its own names and leans on lightEvalWgsl for the shared ones", () => {
        const src = fogInScatterWgsl();
        expect(defs(src)).toEqual(["henyeyGreenstein", "inScatterContribution", "sunInScatter"]);
        // `distanceAttenuation` / `spotFactor` / `PointLightGpu` belong to the base chunks the consumer
        // splices ahead of this one; emitting them here would be a duplicate declaration
        expect(src).not.toContain("fn distanceAttenuation");
        expect(src).not.toContain("fn spotFactor");
        expect(src).not.toContain("struct PointLightGpu");
        // PI is a captured JS constant now, so it folds to a literal — the `FOG_PI` module const is gone
        expect(src).not.toContain("FOG_PI");
    });
});
