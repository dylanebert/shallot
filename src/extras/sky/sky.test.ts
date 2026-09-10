import { beforeEach, describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { body, flat, noIntegerDivision } from "../../../tests/wgsl";
import { State } from "../..";
import { clear, register } from "../../engine/ecs/core";
import { srgbToLinear } from "../../engine/utils/color";
import { Backgrounds } from "../../standard/sear/contract";
import { backgroundWgsl } from "../../standard/sear/pipelines";
import { Sky, SkyPlugin } from ".";
import { packSky } from "./pack";
import { SKY_AT, SKY_BYTES, SKY_FLOATS, SkyGpu, sampleSky, skyBackground } from "./shader";

const config = (overrides: Partial<d.Infer<typeof SkyGpu>> = {}) =>
    SkyGpu({
        hazeDensity: 0,
        horizonBand: 0,
        pad0: 0,
        pad1: 0,
        hazeColor: d.vec4f(0),
        skyZenith: d.vec4f(0.8, 0.9, 1, 0),
        skyHorizon: d.vec4f(0.2, 0.3, 0.4, 0),
        starParams: d.vec4f(0),
        cloudParams: d.vec4f(0, 0, 4, 0),
        cloudColor: d.vec4f(0),
        sunParams: d.vec4f(0),
        sunVisualColor: d.vec4f(1),
        ...overrides,
    });

type SkyConfig = d.Infer<typeof SkyGpu>;
type Vec2 = ReturnType<typeof d.vec2f>;
type Vec3 = ReturnType<typeof d.vec3f>;
type StarStats = { continued: number; evaluated: number; hits: number };

// biome-ignore lint/suspicious/noApproximativeNumericConstant: the reference pins the shipped WGSL literal
const REFERENCE_STAR_PI = 3.14159;
const REFERENCE_STAR_HALF_PI = 1.5708;

// Independent, test-local transcription of the shipped raw-WGSL recipe. It deliberately does not call
// any production sky helper: the differential catches control-flow, operand, or layer-routing drift in
// the TGSL graph while using TypeGPU's CPU stdlib for WGSL-compatible f32/vector primitives.
function referenceHash2(p: Vec2): number {
    let p3 = d.vec3f(std.fract(std.mul(d.vec3f(p.x, p.y, p.x), 0.1031)));
    p3 = d.vec3f(std.add(p3, std.dot(p3, std.add(p3.yzx, 33.33))));
    return std.fract((p3.x + p3.y) * p3.z);
}

function referenceSimplex2(p: Vec2): number {
    const K1 = d.f32(0.366025404);
    const K2 = d.f32(0.211324865);
    const i = std.floor(std.add(p, (p.x + p.y) * K1));
    const a = std.add(std.sub(p, i), (i.x + i.y) * K2);
    const o = std.select(d.vec2f(0, 1), d.vec2f(1, 0), a.x > a.y);
    const b = std.add(std.sub(a, o), K2);
    const c = std.add(std.sub(a, d.vec2f(1)), 2 * K2);
    const h = std.max(
        std.sub(d.vec3f(0.5), d.vec3f(std.dot(a, a), std.dot(b, b), std.dot(c, c))),
        d.vec3f(0),
    );
    const h4 = std.mul(std.mul(std.mul(h, h), h), h);
    const n = d.vec3f(
        std.dot(
            a,
            d.vec2f(referenceHash2(i) * 2 - 1, referenceHash2(std.add(i, d.vec2f(0, 1))) * 2 - 1),
        ),
        std.dot(
            b,
            d.vec2f(
                referenceHash2(std.add(i, o)) * 2 - 1,
                referenceHash2(std.add(std.add(i, o), d.vec2f(0, 1))) * 2 - 1,
            ),
        ),
        std.dot(
            c,
            d.vec2f(
                referenceHash2(std.add(i, d.vec2f(1))) * 2 - 1,
                referenceHash2(std.add(i, d.vec2f(1, 2))) * 2 - 1,
            ),
        ),
    );
    return std.dot(h4, n) * 70;
}

function referenceFbm2(p: Vec2): number {
    let value = d.f32(0);
    let amplitude = d.f32(0.5);
    let frequency = d.f32(1);
    for (let i = 0; i < 5; i++) {
        value = value + amplitude * referenceSimplex2(std.mul(p, frequency));
        amplitude = amplitude * 0.5;
        frequency = frequency * 2;
    }
    return value;
}

function referenceHash2Star(p: Vec2): Vec2 {
    let p3 = d.vec3f(std.fract(std.mul(d.vec3f(p.x, p.y, p.x), d.vec3f(0.1031, 0.103, 0.0973))));
    p3 = d.vec3f(std.add(p3, std.dot(p3, std.add(p3.yzx, 33.33))));
    return std.fract(std.mul(std.add(p3.xx, p3.yz), p3.zy));
}

function referenceStars(dir: Vec3, intensity: number, amount: number, stats?: StarStats): Vec3 {
    if (dir.y < 0) return d.vec3f(0);
    const theta = std.atan2(dir.z, dir.x);
    const phi = std.asin(std.clamp(dir.y, -1, 1));
    const gridSize = std.mix(20, 100, amount);
    const cell = d.vec2f(
        (theta * gridSize) / REFERENCE_STAR_PI,
        (phi * gridSize) / REFERENCE_STAR_HALF_PI,
    );
    const cellId = std.floor(cell);
    let starColor = d.vec3f(0);
    for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
            const neighbor = std.add(cellId, d.vec2f(dx, dy));
            const starHash = referenceHash2(neighbor);
            if (starHash > amount * 0.7) {
                if (stats) stats.continued++;
                continue;
            }
            if (stats) stats.evaluated++;
            const starPos = referenceHash2Star(neighbor);
            const starCenter = std.add(neighbor, starPos);
            const dist = std.length(std.sub(cell, starCenter));
            const brightness = referenceHash2(std.add(neighbor, d.vec2f(100)));
            const radius = 0.02 + brightness * 0.03;
            if (dist < radius) {
                if (stats) stats.hits++;
                const twinkle = 0.8 + 0.2 * std.sin(brightness * 100);
                const strength = intensity * brightness * twinkle;
                const falloff = 1 - std.smoothstep(0, radius, dist);
                const temp = referenceHash2(std.add(neighbor, d.vec2f(200)));
                const tint = std.mix(d.vec3f(1, 0.9, 0.8), d.vec3f(0.8, 0.9, 1), temp);
                starColor = d.vec3f(std.max(starColor, std.mul(std.mul(tint, strength), falloff)));
            }
        }
    }
    return starColor;
}

function referenceClouds(
    dir: Vec3,
    params: ReturnType<typeof d.vec4f>,
    color: ReturnType<typeof d.vec4f>,
) {
    if (dir.y < 0.01) return d.vec4f(0);
    const t = params.z / std.max(dir.y, 0.001);
    const uv = std.mul(dir.xz, t);
    let n = referenceFbm2(uv);
    n = std.smoothstep(1 - params.x, 1, n * 0.5 + 0.5) * params.y;
    n = n * std.smoothstep(0, 0.15, dir.y);
    return d.vec4f(color.xyz, n);
}

function referenceSky(sky: SkyConfig, dir: Vec3, sunDirection: Vec3): Vec3 {
    const t = std.pow(std.clamp(dir.y, 0, 1), 0.25);
    let color = d.vec3f(std.mix(sky.skyHorizon.xyz, sky.skyZenith.xyz, t));
    if (sky.horizonBand > 0) {
        const horizonBlend = std.pow(1 - std.abs(dir.y), 32) * sky.horizonBand;
        color = d.vec3f(std.mix(color, std.mul(sky.skyHorizon.xyz, 1.5), horizonBlend));
    }
    color = d.vec3f(std.add(color, referenceStars(dir, sky.starParams.x, sky.starParams.y)));
    const clouds = referenceClouds(dir, sky.cloudParams, sky.cloudColor);
    color = d.vec3f(std.mix(color, clouds.xyz, clouds.w));

    const sunDot = std.dot(dir, std.neg(sunDirection));
    const sunVisualColor = sky.sunVisualColor.xyz;
    const glowStrength = sky.sunParams.w;
    if (glowStrength > 0) {
        const g = d.f32(0.76);
        const gg = g * g;
        const mie = (1 - gg) / std.pow(1 + gg - 2 * g * sunDot, 1.5);
        color = d.vec3f(
            std.add(color, std.mul(std.mul(std.mul(sunVisualColor, mie), glowStrength), 0.025)),
        );
        const angle = std.max(0, sunDot);
        const corona = std.pow(angle, 512) * 0.4 + std.pow(angle, 128) * 0.06;
        color = d.vec3f(
            std.add(
                color,
                std.mul(
                    std.mul(std.mul(d.vec3f(1, 0.9, 0.7), sunVisualColor), corona),
                    glowStrength,
                ),
            ),
        );
    }

    // A zero f32 span means no disk, including when a tiny positive size rounds the span to zero.
    const baseSunSize = d.f32(0.9995);
    const sunThreshold = d.f32(1 - (1 - baseSunSize) * sky.sunParams.x);
    const sunSpan = d.f32(1 - sunThreshold);
    if (sunSpan > 0) {
        const sunEdgeWidth = sunSpan * 0.15;
        const diskBlend = std.smoothstep(
            sunThreshold - sunEdgeWidth,
            sunThreshold + sunEdgeWidth,
            sunDot,
        );
        if (diskBlend > 0) {
            const radial = std.saturate((sunDot - sunThreshold) / sunSpan);
            const r = 1 - radial;
            const mu = std.sqrt(1 - r * r);
            const limbDarken = 1 - 0.6 * (1 - mu);
            color = d.vec3f(
                std.add(color, std.mul(std.mul(sunVisualColor, limbDarken), diskBlend)),
            );
            const edgeDist = 1 - std.smoothstep(0, 1, radial);
            const fringe = d.vec3f(
                std.smoothstep(0.3, 0.7, edgeDist),
                std.smoothstep(0.5, 0.9, edgeDist),
                std.smoothstep(0.7, 1, edgeDist),
            );
            color = d.vec3f(
                std.add(
                    color,
                    std.mul(
                        std.mul(std.mul(std.mul(fringe, sunVisualColor), 0.15), diskBlend),
                        1 - radial,
                    ),
                ),
            );
        }
    }
    if (sky.hazeDensity > 0) {
        const horizonFactor = 1 - std.clamp(dir.y, 0, 1);
        const hazeAmount = std.pow(horizonFactor, 2) * std.saturate(sky.hazeDensity * 5);
        color = d.vec3f(std.mix(color, sky.hazeColor.xyz, hazeAmount));
    }
    return color;
}

function expectVecClose(actual: Vec3, expected: Vec3, digits = 5): void {
    expect(actual.x).toBeCloseTo(expected.x, digits);
    expect(actual.y).toBeCloseTo(expected.y, digits);
    expect(actual.z).toBeCloseTo(expected.z, digits);
}

function delta(a: Vec3, b: Vec3): number {
    return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.z - b.z));
}

function activeStarDirection(amount: number): { dir: Vec3; stats: StarStats } {
    const gridSize = std.mix(20, 100, amount);
    for (let y = 1; y < 20; y++) {
        for (let x = -20; x < 20; x++) {
            const neighbor = d.vec2f(x, y);
            if (referenceHash2(neighbor) > amount * 0.7) continue;
            const starPos = referenceHash2Star(neighbor);
            const theta = ((x + starPos.x) * REFERENCE_STAR_PI) / gridSize;
            const phi = ((y + starPos.y) * REFERENCE_STAR_HALF_PI) / gridSize;
            const cosPhi = Math.cos(phi);
            const dir = d.vec3f(cosPhi * Math.cos(theta), Math.sin(phi), cosPhi * Math.sin(theta));
            const stats = { continued: 0, evaluated: 0, hits: 0 };
            const color = referenceStars(dir, 1, amount, stats);
            if (
                stats.continued > 0 &&
                stats.evaluated > 0 &&
                stats.hits > 0 &&
                Math.max(color.x, color.y, color.z) > 0.01
            ) {
                return { dir, stats };
            }
        }
    }
    throw new Error("reference star search found no active mixed-continue cell");
}

describe("Sky typed shader", () => {
    test("SkyPlugin registration follows its State without an old build deleting the new owner", () => {
        Backgrounds.clear();
        const first = new State();
        SkyPlugin.initialize?.(first);
        const firstSpec = Backgrounds.get("sky");
        expect(firstSpec).toBeDefined();

        const second = new State();
        SkyPlugin.initialize?.(second);
        const secondSpec = Backgrounds.get("sky");
        expect(secondSpec).toBeDefined();
        expect(secondSpec).not.toBe(firstSpec);

        first.dispose();
        expect(Backgrounds.get("sky")).toBe(secondSpec);
        second.dispose();
        expect(Backgrounds.get("sky")).toBeUndefined();
    });

    test("the schema preserves the 144-byte uniform contract and every packer offset", () => {
        expect(SKY_BYTES).toBe(144);
        expect(d.sizeOf(SkyGpu)).toBe(SKY_BYTES);
        expect(d.alignmentOf(SkyGpu)).toBe(16);
        expect(SKY_FLOATS).toBe(36);
        expect(SKY_AT).toEqual({
            hazeDensity: 0,
            horizonBand: 1,
            hazeColor: 4,
            skyZenith: 8,
            skyHorizon: 12,
            starParams: 16,
            cloudParams: 20,
            cloudColor: 24,
            sunParams: 28,
            sunVisualColor: 32,
        });
    });

    test("sampleSky keeps the elevation gradient exact when optional layers are disabled", () => {
        const sky = config();
        const sunDirection = d.vec3f(1, 0, 0);
        const horizon = sampleSky(sky, d.vec3f(0, 0, 1), sunDirection);
        const zenith = sampleSky(sky, d.vec3f(0, 1, 0), sunDirection);
        expect([horizon.x, horizon.y, horizon.z]).toEqual([
            sky.skyHorizon.x,
            sky.skyHorizon.y,
            sky.skyHorizon.z,
        ]);
        expect([zenith.x, zenith.y, zenith.z]).toEqual([
            sky.skyZenith.x,
            sky.skyZenith.y,
            sky.skyZenith.z,
        ]);
    });

    test("sampleSky reads the directional-light travel direction for the sun glow", () => {
        const sky = config({
            skyZenith: d.vec4f(0),
            skyHorizon: d.vec4f(0),
            sunParams: d.vec4f(0, 0, 0, 1),
        });
        const dir = d.vec3f(0, 1, 0);
        const toward = sampleSky(sky, dir, d.vec3f(0, -1, 0));
        const away = sampleSky(sky, dir, d.vec3f(0, 1, 0));
        expect(toward.x).toBeGreaterThan(away.x * 10);
        expect(toward.y).toBeGreaterThan(away.y * 10);
        expect(toward.z).toBeGreaterThan(away.z * 10);
    });

    test("a zero f32 disk span disables zero and tiny-positive sizes before smoothstep or division", () => {
        const sky = config({
            skyZenith: d.vec4f(0),
            skyHorizon: d.vec4f(0),
            sunParams: d.vec4f(0),
        });
        const color = sampleSky(sky, d.vec3f(0, 1, 0), d.vec3f(0, -1, 0));
        expect([color.x, color.y, color.z].every(Number.isFinite)).toBe(true);
        expectVecClose(color, d.vec3f(0));

        // This is the first positive f32 neighbourhood that exposed the gap: size itself is positive,
        // but the shader's f32 threshold rounds to 1, so its computed denominator/edge width is zero.
        const tinySize = d.f32(0.00005);
        const sub = (a: number, b: number) => Math.fround(a - b);
        const mul = (a: number, b: number) => Math.fround(a * b);
        const threshold = sub(1, mul(sub(1, d.f32(0.9995)), tinySize));
        const span = sub(1, threshold);
        expect(tinySize).toBeGreaterThan(0);
        expect(threshold).toBe(1);
        expect(span).toBe(0);
        const tiny = sampleSky(
            config({
                skyZenith: d.vec4f(0),
                skyHorizon: d.vec4f(0),
                sunParams: d.vec4f(tinySize, 0, 0, 0),
            }),
            d.vec3f(0, 1, 0),
            d.vec3f(0, -1, 0),
        );
        expectVecClose(tiny, d.vec3f(0));

        const wgsl = backgroundWgsl(skyBackground);
        const sample = wgsl.slice(wgsl.indexOf("fn sampleSky("), wgsl.indexOf("fn skyFs_1("));
        const guardedDisk = body(sample, "if ((sunSpan > 0f))");
        expect(guardedDisk).toContain("smoothstep(");
        expect(guardedDisk).toMatch(/let radial = saturate\(.* \/ sunSpan\)/);
    });

    test("the TGSL graph differentially preserves every raw-WGSL layer and both star continue arms", () => {
        const black = {
            skyZenith: d.vec4f(0),
            skyHorizon: d.vec4f(0),
        };
        const star = activeStarDirection(0.5);
        expect(star.stats.continued).toBeGreaterThan(0);
        expect(star.stats.evaluated).toBeGreaterThan(0);
        expect(star.stats.hits).toBeGreaterThan(0);

        const cases: {
            name: string;
            sky: SkyConfig;
            inactive: SkyConfig;
            dir: Vec3;
            sunDirection: Vec3;
        }[] = [
            {
                name: "horizon band",
                sky: config({ horizonBand: 0.8 }),
                inactive: config(),
                dir: std.normalize(d.vec3f(1, 0.02, 0)),
                sunDirection: d.vec3f(1, 0, 0),
            },
            {
                name: "nested-grid stars",
                sky: config({ ...black, starParams: d.vec4f(1, 0.5, 0, 0) }),
                inactive: config({ ...black, starParams: d.vec4f(0, 0.5, 0, 0) }),
                dir: star.dir,
                sunDirection: d.vec3f(1, 0, 0),
            },
            {
                name: "FBM clouds",
                sky: config({
                    ...black,
                    cloudParams: d.vec4f(0.8, 0.7, 4, 0),
                    cloudColor: d.vec4f(0.7, 0.6, 0.5, 0),
                }),
                inactive: config({
                    ...black,
                    cloudParams: d.vec4f(0.8, 0, 4, 0),
                    cloudColor: d.vec4f(0.7, 0.6, 0.5, 0),
                }),
                dir: std.normalize(d.vec3f(0.4, 0.8, 0.4)),
                sunDirection: d.vec3f(1, 0, 0),
            },
            {
                name: "sun glow and corona",
                sky: config({ ...black, sunParams: d.vec4f(0, 0, 0, 0.6) }),
                inactive: config({ ...black, sunParams: d.vec4f(0) }),
                dir: d.vec3f(0, 1, 0),
                sunDirection: d.vec3f(0, -1, 0),
            },
            {
                name: "limb-darkened sun disk",
                sky: config({ ...black, sunParams: d.vec4f(0.7, 0, 0, 0) }),
                inactive: config({ ...black, sunParams: d.vec4f(0) }),
                dir: d.vec3f(0, 1, 0),
                sunDirection: d.vec3f(0, -1, 0),
            },
            {
                name: "horizon haze",
                sky: config({ hazeDensity: 0.1, hazeColor: d.vec4f(0.9, 0.1, 0.2, 0) }),
                inactive: config(),
                dir: std.normalize(d.vec3f(1, 0.04, 0)),
                sunDirection: d.vec3f(1, 0, 0),
            },
        ];

        for (const c of cases) {
            const actual = sampleSky(c.sky, c.dir, c.sunDirection);
            const expected = referenceSky(c.sky, c.dir, c.sunDirection);
            expectVecClose(actual, expected, 4);
            const inactive = sampleSky(c.inactive, c.dir, c.sunDirection);
            expect(delta(actual, inactive), `${c.name} must contribute`).toBeGreaterThan(1e-4);
        }
    });

    test("the typed background resolves the sky uniform, lighting accessor, and nested star loop", () => {
        const wgsl = backgroundWgsl(skyBackground);
        expect(wgsl).toContain("struct Sky");
        expect(wgsl).toMatch(/@group\(2\) @binding\(0\) var<uniform> sky: Sky/);
        expect(wgsl).toMatch(/@group\(0\) @binding\(2\) var<uniform> lighting: Lighting/);
        expect(wgsl).toContain("fn sampleSky(");
        expect(wgsl).toContain("fn sampleStars(");
        expect(wgsl).toContain("for (var dy = -1i; (dy <= 1i); dy = (dy + 1i))");
        expect(wgsl).toContain("continue;");
        noIntegerDivision(wgsl);
    });

    test("no two emitted sky functions share a body", () => {
        // `hashStar` was a byte-identical second registration of `hash2`, so the sky shipped the same
        // hash twice under two names. Compare bodies rather than names: a duplicate reintroduced under
        // any third name is the same defect, and the name is exactly what a reader trusts instead.
        const wgsl = backgroundWgsl(skyBackground);
        const seen = new Map<string, string>();
        for (const [, name] of wgsl.matchAll(/fn (\w+)\s*\(/g)) {
            const src = flat(body(wgsl, `fn ${name}(`));
            const decl = src.slice(src.indexOf("{"));
            const prior = seen.get(decl);
            expect(prior, `fn ${name} duplicates fn ${prior}`).toBeUndefined();
            seen.set(decl, name);
        }
        expect(seen.size).toBeGreaterThan(4); // the walk reached real functions, not zero
    });
});

// packSky crosses the CPU→GPU boundary: it lays a `Sky` singleton into `SkyGpu`. The contract is the
// field→offset map (a layout bug puts a value one slot off and the sky shades wrong) and the hex→linear
// decode. The real-GPU render lives in the gym `render` `sky` mode.
describe("packSky", () => {
    let state: State;

    beforeEach(() => {
        clear();
        state = new State();
        register("Sky", Sky, SkyPlugin.traits?.Sky);
    });

    test("lays scalars and hex-decoded colors into the std140 Sky offsets", () => {
        const eid = state.create();
        state.add(eid, Sky); // defaults applied, so every field is set
        Sky.hazeDensity.set(eid, 0.01);
        Sky.band.set(eid, 0.5);
        Sky.zenith.set(eid, 0xff8000);
        Sky.starIntensity.set(eid, 0.3);
        Sky.starAmount.set(eid, 0.6);
        Sky.cloudHeight.set(eid, 4);
        Sky.sunSize.set(eid, 0.7);
        Sky.sunGlow.set(eid, 0.2);
        Sky.sunColor.set(eid, 0x000000);

        const out = new Float32Array(SKY_FLOATS);
        packSky(eid, out);

        // leading f32 lanes
        expect(out[0]).toBeCloseTo(0.01, 6); // hazeDensity
        expect(out[1]).toBeCloseTo(0.5, 6); // band
        // skyZenith at the third vec4 (float 8): 0xff8000 → linear rgb, blue 0
        expect(out[8]).toBeCloseTo(srgbToLinear(1), 6);
        expect(out[9]).toBeCloseTo(srgbToLinear(0x80 / 255), 6);
        expect(out[10]).toBe(0);
        // starParams = (intensity, amount, _, _)
        expect(out[16]).toBeCloseTo(0.3, 6);
        expect(out[17]).toBeCloseTo(0.6, 6);
        // cloudParams = (coverage, density, height, _)
        expect(out[22]).toBe(4);
        // sunParams = (size, _, _, glow)
        expect(out[28]).toBeCloseTo(0.7, 6);
        expect(out[31]).toBeCloseTo(0.2, 6);
        // sunVisualColor at float 32: 0x000000 → 0
        expect(out[32]).toBe(0);
    });
});
