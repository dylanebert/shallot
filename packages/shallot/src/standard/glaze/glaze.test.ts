import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { body, integerDiscipline } from "../../../tests/wgsl";
import { linearToSrgbWgsl } from "../render/core";
import {
    applyGrade,
    applySaturation,
    applyVignette,
    bayer4,
    ditherPosterizeL,
    GlazeConfig,
    glazeWgsl,
    oklabL,
    WORKGROUP,
} from "./composite";
import {
    Tonemap,
    tmAces,
    tmAgx,
    tmLuma,
    tmNeutral,
    tmReinhard,
    tmReinhardLuminance,
    tmSomewhatBoring,
    tonemap,
    tonemapWgsl,
} from "./tonemap";

// The postfx chain is TGSL, so every operator runs on the CPU here — the same function the composite
// resolves, not a second implementation. Real-GPU truth (the composited pixels) is the gym `render`
// scenario's framebuffer probes; what's assertable without a device is the arithmetic and the emitted
// WGSL's shape.
//
// The expected values for the matrix-heavy operators are frozen literals derived by hand-evaluating the
// **shipped pre-port WGSL** in f64 (a throwaway transcription of the deleted `TONEMAP_WGSL` text, run
// once — deliberately not imported here, or the test would restate the implementation). Tolerances are
// derived from f32's 6e-8 relative step: ~5 digits for the two-matrix operators, one fewer for AgX,
// whose degree-6 polynomial cancels terms of magnitude ~10 down to ~0.79 (a ~13× error amplification)
// before a `pow(·, 2.2)` scales it again.

const PROBE = d.vec3f(0.5, 0.25, 0.75);
const GREY = d.vec3f(1, 1, 1);

const zeroed = () =>
    GlazeConfig({
        exposure: 0,
        vignetteStrength: 0,
        vignetteInner: 0,
        vignetteOuter: 0,
        posterizeBands: 0,
        ditherStrength: 0,
        tonemapMode: 0,
        saturation: 0,
        slope: d.vec4f(),
        offset: d.vec4f(),
        power: d.vec4f(),
    });

describe("Khronos Neutral — the zero-config default", () => {
    test("a shadow-clean color passes through untouched", () => {
        // the desaturation offset is `select(0.04, x - 6.25x², x < 0.08)` over the darkest channel, so a
        // color whose minimum is 0 gets offset 0; with its peak under the 0.76 knee nothing else applies
        const out = tmNeutral(d.vec3f(0, 0.5, 0.7));
        expect(out.x).toBe(0);
        expect(out.y).toBe(0.5);
        expect(out.z).toBeCloseTo(0.7, 6);
    });

    test("the toe branches meet at x = 0.08", () => {
        // 0.08 - 6.25·0.08² = 0.08 - 0.04 = 0.04, the constant branch's value: the offset is continuous
        // across the switchover, so a pixel sweeping through it has no visible seam
        const below = tmNeutral(d.vec3f(0.0799999, 0.4, 0.4));
        const above = tmNeutral(d.vec3f(0.0800001, 0.4, 0.4));
        expect(above.y - below.y).toBeCloseTo(0, 6);
    });

    test("a highlight is compressed under 1 and the ramp stays monotone", () => {
        let last = -1;
        for (const v of [0.5, 0.8, 1, 2, 4, 16, 64]) {
            const out = tmNeutral(d.vec3f(v, v, v));
            expect(out.x).toBeGreaterThan(last);
            expect(out.x).toBeLessThan(1);
            last = out.x;
        }
    });

    test("index 0 and any unknown mode are Neutral", () => {
        for (const mode of [Tonemap.Neutral, 7, 99]) {
            const out = tonemap(mode, PROBE);
            const want = tmNeutral(PROBE);
            expect([out.x, out.y, out.z]).toEqual([want.x, want.y, want.z]);
        }
    });
});

describe("the operator set", () => {
    test("None is a passthrough", () => {
        const out = tonemap(Tonemap.None, PROBE);
        expect([out.x, out.y, out.z]).toEqual([0.5, 0.25, 0.75]);
    });

    test("luma is the Rec. 709 primaries", () => {
        expect(tmLuma(d.vec3f(1, 0, 0))).toBeCloseTo(0.2126, 7);
        expect(tmLuma(d.vec3f(0, 1, 0))).toBeCloseTo(0.7152, 7);
        expect(tmLuma(d.vec3f(0, 0, 1))).toBeCloseTo(0.0722, 7);
        expect(tmLuma(GREY)).toBeCloseTo(1, 6);
    });

    test("Reinhard is c/(1+c) per channel", () => {
        const out = tmReinhard(PROBE);
        expect(out.x).toBeCloseTo(0.5 / 1.5, 6);
        expect(out.y).toBeCloseTo(0.25 / 1.25, 6);
        expect(out.z).toBeCloseTo(0.75 / 1.75, 6);
    });

    test("Reinhard-luminance scales all channels by one factor, so hue is preserved", () => {
        const out = tmReinhardLuminance(PROBE);
        // frozen from the shipped WGSL: L = 0.4693, L/(1+L) = 0.319399, ratio 0.6806866
        expect(out.x).toBeCloseTo(0.3733432891, 6);
        expect(out.y).toBeCloseTo(0.1866716446, 6);
        expect(out.z).toBeCloseTo(0.5600149337, 6);
        // one common factor: the channel ratios are the input's
        expect(out.y / out.x).toBeCloseTo(0.5, 6);
    });

    test("ACES Fitted matches the shipped RRT+ODT fit", () => {
        const grey = tmAces(GREY);
        expect(grey.x).toBeCloseTo(0.6191154269, 5);
        // both ACES matrices are row-normalized to 1, so grey stays grey — except the output matrix's
        // third row sums to 0.99999, which is the whole of this channel's departure
        expect(grey.y).toBeCloseTo(grey.x, 6);
        expect(grey.z).toBeCloseTo(0.6191092358, 5);
        const out = tmAces(PROBE);
        expect(out.x).toBeCloseTo(0.3712522994, 5);
        expect(out.y).toBeCloseTo(0.1778461902, 5);
        expect(out.z).toBeCloseTo(0.5046957987, 5);
    });

    test("AgX matches the shipped analytic curve", () => {
        const grey = tmAgx(GREY);
        expect(grey.x).toBeCloseTo(0.5902293768, 4);
        expect(grey.y).toBeCloseTo(0.590136055, 4);
        expect(grey.z).toBeCloseTo(0.5901022786, 4);
        const out = tmAgx(PROBE);
        expect(out.x).toBeCloseTo(0.4229065159, 4);
        expect(out.y).toBeCloseTo(0.3179647863, 4);
        expect(out.z).toBeCloseTo(0.536365196, 4);
    });

    test("AgX bottoms out at exactly zero", () => {
        // the 1e-10 log floor puts the log-encoded value below minEv, so it saturates to 0; the contrast
        // polynomial's constant term (-0.00232) then makes the outset product negative, `max(·, 0)` zeroes
        // it, and `pow(0, 2.2)` is 0 — a true black stays black rather than lifting
        const out = tmAgx(d.vec3f(0, 0, 0));
        expect([out.x, out.y, out.z]).toEqual([0, 0, 0]);
    });

    test("SomewhatBoring matches the shipped transform", () => {
        // grey has zero chroma, so `bt` is 0 and the whole desaturation branch drops out: the result is
        // `cin · sbCurve(luma)/luma · 0.97` = (1 - e⁻¹) · 0.97
        expect(tmSomewhatBoring(GREY).x).toBeCloseTo((1 - Math.exp(-1)) * 0.97, 6);
        const out = tmSomewhatBoring(PROBE);
        expect(out.x).toBeCloseTo(0.4054838574, 5);
        expect(out.y).toBeCloseTo(0.2073971674, 5);
        expect(out.z).toBeCloseTo(0.5963195401, 5);
    });

    test("every operator index routes somewhere distinct", () => {
        const seen = new Set<string>();
        for (const mode of Object.values(Tonemap)) {
            const out = tonemap(mode, PROBE);
            seen.add(`${out.x},${out.y},${out.z}`);
        }
        expect(seen.size).toBe(Object.keys(Tonemap).length);
    });
});

describe("the OkLab quantize axis", () => {
    test("lightness is monotone in luminance and normalized at white", () => {
        // the three LMS rows each sum to 1, so white lands at L = 0.2104542553 + 0.7936177850 -
        // 0.0040720468 ≈ 1 — the axis is normalized, which is what makes `posterize` a band *count*
        expect(oklabL(GREY)).toBeCloseTo(1, 6);
        expect(oklabL(d.vec3f(0, 0, 0))).toBe(0);
        let last = -1;
        for (const v of [0, 0.05, 0.2, 0.5, 0.8, 1]) {
            const L = oklabL(d.vec3f(v, v, v));
            expect(L).toBeGreaterThan(last);
            last = L;
        }
    });

    test("the 4×4 Bayer matrix covers all 16 offsets", () => {
        const seen = new Set<number>();
        for (let y = 0; y < 4; y++) {
            for (let x = 0; x < 4; x++) seen.add(bayer4(d.vec2f(x, y)));
        }
        expect(seen.size).toBe(16);
        expect(Math.min(...seen)).toBe(-0.5);
        expect(Math.max(...seen)).toBe(0.4375); // 15/16 - 0.5
        for (const v of seen) expect((v + 0.5) * 16).toBeCloseTo(Math.round((v + 0.5) * 16), 10);
    });

    test("posterize quantizes to band centers, dither offsets before it", () => {
        const bands = { ...zeroed(), posterizeBands: 4 };
        expect(ditherPosterizeL(0.3, d.vec2f(), bands)).toBeCloseTo(0.25, 6);
        expect(ditherPosterizeL(0.4, d.vec2f(), bands)).toBeCloseTo(0.5, 6);
        expect(ditherPosterizeL(0.9, d.vec2f(), bands)).toBeCloseTo(1, 6);
        // dither alone shifts L by the pixel's Bayer offset × the amplitude
        const dither = { ...zeroed(), ditherStrength: 0.5 };
        expect(ditherPosterizeL(0.6, d.vec2f(0, 0), dither)).toBeCloseTo(0.6 - 0.25, 6);
        // both off: identity, so a camera with no Glaze pays nothing
        expect(ditherPosterizeL(0.42, d.vec2f(2, 3), zeroed())).toBe(Math.fround(0.42));
        // dither before posterize: the same L lands in a different band once the offset is applied
        const both = { ...zeroed(), posterizeBands: 4, ditherStrength: 0.5 };
        // the darkest Bayer cell (-0.5) pulls 0.3 down into band 0, the brightest (+0.4375) up into band 2
        expect(ditherPosterizeL(0.3, d.vec2f(0, 0), both)).toBeCloseTo(0, 6);
        expect(ditherPosterizeL(0.3, d.vec2f(0, 3), both)).toBeCloseTo(0.5, 6);
    });
});

describe("grade, saturation, vignette", () => {
    const identity = {
        ...zeroed(),
        exposure: 1,
        saturation: 1,
        slope: d.vec4f(1, 1, 1, 0),
        power: d.vec4f(1, 1, 1, 0),
    };

    test("the no-Glaze grade identities are a no-op", () => {
        const out = applyGrade(PROBE, identity);
        expect(out.x).toBeCloseTo(0.5, 6);
        expect(out.y).toBeCloseTo(0.25, 6);
        expect(out.z).toBeCloseTo(0.75, 6);
        const sat = applySaturation(PROBE, identity);
        expect([sat.x, sat.y, sat.z]).toEqual([0.5, 0.25, 0.75]);
    });

    test("CDL is (in·slope + offset)^power, then exposure", () => {
        const graded = applyGrade(d.vec3f(0.25, 0.25, 0.25), {
            ...identity,
            exposure: 2,
            slope: d.vec4f(2, 1, 1, 0),
            offset: d.vec4f(0, 0.75, 0, 0),
            power: d.vec4f(1, 1, 0.5, 0),
        });
        expect(graded.x).toBeCloseTo(0.5 * 2, 6);
        expect(graded.y).toBeCloseTo(1 * 2, 6);
        expect(graded.z).toBeCloseTo(0.5 * 2, 6); // sqrt(0.25)
    });

    test("saturation 0 collapses to luma, and the grade clamps negatives before the power", () => {
        const grey = applySaturation(PROBE, { ...identity, saturation: 0 });
        const luma = tmLuma(PROBE);
        expect(grey.x).toBeCloseTo(luma, 6);
        expect(grey.z).toBeCloseTo(luma, 6);
        // a negative CDL result would make a fractional power NaN, so it's floored at 0 first
        const lifted = applyGrade(d.vec3f(0.1, 0.1, 0.1), {
            ...identity,
            offset: d.vec4f(-1, -1, -1, 0),
            power: d.vec4f(0.5, 0.5, 0.5, 0),
        });
        expect([lifted.x, lifted.y, lifted.z]).toEqual([0, 0, 0]);
    });

    test("vignette is identity at strength 0 and at the screen center", () => {
        const off = applyVignette(PROBE, d.vec2f(0.9, 0.9), zeroed());
        expect([off.x, off.y, off.z]).toEqual([0.5, 0.25, 0.75]);
        const full = { ...zeroed(), vignetteStrength: 1, vignetteInner: 0, vignetteOuter: 1 };
        const center = applyVignette(PROBE, d.vec2f(0.5, 0.5), full);
        expect(center.x).toBeCloseTo(0.5, 6);
    });

    test("the corner darkens by the γ-2.2 factor", () => {
        // dist = √0.5 = 0.7071, smoothstep(0, 1, 0.7071) = 0.79289, v = 0.20711, v^2.2 = 0.031306. The
        // 2.2 power is what makes `strength` mean *perceived* darkness through the sRGB encode that follows
        const full = { ...zeroed(), vignetteStrength: 1, vignetteInner: 0, vignetteOuter: 1 };
        const corner = applyVignette(GREY, d.vec2f(0, 0), full);
        expect(corner.x).toBeCloseTo(0.0313059943, 6);
    });
});

describe("the GlazeConfig layout", () => {
    test("the schema is the 80-byte uniform the packer writes through", () => {
        expect(d.sizeOf(GlazeConfig)).toBe(80);
        const at = (field: (g: d.Infer<typeof GlazeConfig>) => unknown) =>
            d.memoryLayoutOf(GlazeConfig, field).offset;
        expect(at((g) => g.exposure)).toBe(0);
        expect(at((g) => g.tonemapMode)).toBe(24);
        expect(at((g) => g.saturation)).toBe(28);
        // the three vec4 grade lanes align to 16, so the eight leading scalars fill exactly one row
        expect(at((g) => g.slope)).toBe(32);
        expect(at((g) => g.offset)).toBe(48);
        expect(at((g) => g.power)).toBe(64);
    });
});

describe("the emitted composite WGSL", () => {
    const kernel = glazeWgsl();

    test("the swapchain format is baked into the storage-texture type", () => {
        expect(kernel).toContain("var output: texture_storage_2d<bgra8unorm, write>");
        expect(glazeWgsl("rgba8unorm")).toContain(
            "var output: texture_storage_2d<rgba8unorm, write>",
        );
    });

    test("the entry point, workgroup size and dimension guard survive the port", () => {
        expect(kernel).toContain(`@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})`);
        expect(kernel).toContain("fn glazeComposite(@builtin(global_invocation_id) gid: vec3u)");
        expect(kernel).toContain("gid.x >= dim.x");
        expect(kernel).toContain("gid.y >= dim.y");
        expect(kernel).toContain("var input: texture_2d<f32>");
        expect(kernel).toContain("var<uniform> glaze: GlazeConfig");
    });

    test("every chain stage lands as its authored name", () => {
        for (const fn of [
            "applyGrade",
            "applySaturation",
            "oklabL",
            "bayer4",
            "ditherPosterizeL",
            "applyVignette",
            "linearToSrgb",
            "tonemap",
            "tmNeutral",
            "tmAces",
        ]) {
            expect(kernel).toContain(`fn ${fn}(`);
        }
        // typegpu names an unnamed kernel or factory-built schema `item`, which no diagnostic would
        // attribute back here
        expect(kernel).not.toMatch(/\b(fn|struct) item/);
    });

    test("the chain order is grade → tonemap → saturation → quantize → vignette → encode", () => {
        const main = body(kernel, "@compute");
        const order = [
            "tonemap(glaze.tonemapMode, applyGrade(",
            "applySaturation(",
            "ditherPosterizeL(",
            "applyVignette(",
            "textureStore(output, p, vec4f(linearToSrgb(",
        ].map((needle) => main.indexOf(needle));
        expect(order.every((i) => i >= 0)).toBe(true);
        expect(order).toEqual([...order].sort((a, b) => a - b));
    });

    test("no integer division and no signed integer leaks", () => {
        // TGSL transpiles `a / b` on integers to `f32(a) / f32(b)` — a fractional quotient. The composite
        // divides only floats, so an `f32(` conversion appearing at all is the signal
        for (const src of [kernel, tonemapWgsl(), linearToSrgbWgsl()]) {
            expect(src).not.toMatch(/\bf32\(/);
            integerDiscipline(src);
        }
    });
});

describe("the spliceable chunks", () => {
    test("the tonemap chunk emits every operator plus the total dispatch", () => {
        const wgsl = tonemapWgsl();
        for (const fn of [
            "tmLuma",
            "tmNeutral",
            "tmReinhard",
            "tmReinhardLuminance",
            "tmRrtOdtFit",
            "tmAces",
            "tmAgxContrast",
            "tmAgx",
            "tmRgbToYcbcr",
            "tmSbCurve",
            "tmSbCurve3",
            "tmSomewhatBoring",
            "tonemap",
        ]) {
            expect(wgsl).toContain(`fn ${fn}(`);
        }
        for (const mode of [1, 2, 3, 4, 5, 6]) expect(wgsl).toContain(`mode == ${mode}u`);
        // the dispatch is an if-chain (TGSL has no `switch`), so the Neutral fallthrough must be the
        // unconditional tail — index 0 AND any unknown index
        expect(wgsl.trimEnd().endsWith("return tmNeutral(c);\n}")).toBe(true);
    });

    test("AgX's EV range folds in f32, as the shipped `let`s did", () => {
        // `minEv` / `maxEv` are f32-typed, so WGSL const-folds `maxEv - minEv` in f32 → exactly 16.5. Left
        // as bare literals they would be abstract-float and fold at f64 → 16.499998, one f32 step low,
        // which shifts the log-range remap of every AgX pixel. The f32-rounded literals in the emitted
        // text are what prove the typing survived
        const wgsl = tonemapWgsl();
        expect(wgsl).toContain("const minEv = -12.473930358886719f;");
        expect(wgsl).toContain("const maxEv = 4.026069164276123f;");
    });

    test("the sRGB encode chunk emits the name a raw composite calls", () => {
        expect(linearToSrgbWgsl()).toContain("fn linearToSrgb(c: vec3f) -> vec3f");
    });
});
