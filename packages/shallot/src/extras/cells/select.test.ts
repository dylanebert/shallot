import { afterEach, describe, expect, test } from "bun:test";
import { vec3f } from "typegpu/data";
import { Compute } from "../../engine";
import { probeBuffer, requestGPU } from "../../engine/runtime";
import { unpackCell } from "./cell";
import { createCellGrid } from "./grid";
import { CELL_DIRECTIONAL_GLYPHS, CELL_FILL_GLYPHS, CELL_GLYPH_COUNT } from "./ramp";
import {
    BG_MATCH_EPSILON,
    directionalGlyphIndex,
    dispatchSelect,
    EDGE_MAGNITUDE_THRESHOLD,
    FACE_BOUNDARY_MAGNITUDE_THRESHOLD,
    fillIndexForLuma,
    localBoundaryMagnitude,
    luma,
    reinhard,
    resetSelectPipelines,
    selectWgsl,
} from "./select";

// directionalGlyphIndex is a typegpu dual CPU/GPU function (`packCell`'s own shape), callable directly
// with no device — the logic-truth surface for the tangent-bucket rotation (`testing.md`'s "CPU execution
// of pure TGSL kernels against deterministic references").
//
// gx/gy fixtures below are derived from concrete 3×3 luma neighborhoods run through selectKernel's own
// Sobel formula (`select.ts`), not picked as abstract numbers — the class of bug this arm exists to catch
// is a y-frame mix-up (grid y-down vs. the glyph labels' y-up visual convention), which an abstract
// `(gx, gy)` pair can't discriminate: the old version of this table asserted `gx=1,gy=1 -> "\\"`, which is
// what you get if you *don't* convert frames, and it passed for a full review round. Naming the luma neighborhood a case derives from ties the expectation to a real edge
// instead of to whatever the code under test currently computes.
describe("directionalGlyphIndex", () => {
    // l(dx, dy) = 0.5 + A*dx + B*dy over the 8 neighbors selectKernel reads (dx/dy in {-1,0,1}, own
    // excluded) — a smooth luma ramp, not a hard step, so no tap saturates and the tie-break at exactly
    // 45°/135° stays clean. Sobel of a linear field is exact: gx = 8*A, gy = 8*B (grid frame, y-down).
    function sobelOfLinearField(a: number, b: number): { gx: number; gy: number } {
        const l = (dx: number, dy: number) => 0.5 + a * dx + b * dy;
        const l00 = l(-1, -1);
        const l10 = l(0, -1);
        const l20 = l(1, -1);
        const l01 = l(-1, 0);
        const l21 = l(1, 0);
        const l02 = l(-1, 1);
        const l12 = l(0, 1);
        const l22 = l(1, 1);
        const gx = l20 + 2 * l21 + l22 - (l00 + 2 * l01 + l02);
        const gy = l02 + 2 * l12 + l22 - (l00 + 2 * l10 + l20);
        return { gx, gy };
    }

    const cases: { name: string; a: number; b: number; wantChar: string }[] = [
        // brightness rises rightward only (a horizontal ramp, top/bottom rows equal) -> the edge (the
        // ramp's own contour lines) runs vertically -> "|"
        { name: "brighter to the right (vertical edge)", a: 0.25, b: 0, wantChar: "|" },
        // brightness rises downward only (row-down) -> the edge runs horizontally -> "-"
        { name: "brighter toward the bottom (horizontal edge)", a: 0, b: 0.25, wantChar: "-" },
        // top-left darkest, bottom-right brightest: the contour tangent runs bottom-left-to-top-right
        { name: "darkest top-left, brightest bottom-right", a: 0.25, b: 0.25, wantChar: "/" },
        // top-right darkest, bottom-left brightest: the contour tangent runs top-left-to-bottom-right
        { name: "darkest bottom-left, brightest top-right", a: 0.25, b: -0.25, wantChar: "\\" },
    ];

    for (const { name, a, b, wantChar } of cases) {
        test(`${name} selects the perpendicular tangent glyph`, () => {
            const { gx, gy } = sobelOfLinearField(a, b);
            const index = directionalGlyphIndex(gx, gy);
            const char = CELL_DIRECTIONAL_GLYPHS[index - CELL_FILL_GLYPHS.length];
            expect(char).toBe(wantChar);
        });
    }

    // the two diagonal cases above are each other's y-frame-mix-up bug: dropping the grid-y-down ->
    // visual-y-up conversion swaps exactly these two and leaves the two axis-aligned cases untouched
    // (`select.ts`'s own docblock) — assert the swap explicitly so a regression that re-introduces it
    // reds here even if a future edit reorders or renames the table above.
    test("the two diagonal cases are not swapped", () => {
        const bl = sobelOfLinearField(0.25, 0.25);
        const br = sobelOfLinearField(0.25, -0.25);
        const blChar =
            CELL_DIRECTIONAL_GLYPHS[directionalGlyphIndex(bl.gx, bl.gy) - CELL_FILL_GLYPHS.length];
        const brChar =
            CELL_DIRECTIONAL_GLYPHS[directionalGlyphIndex(br.gx, br.gy) - CELL_FILL_GLYPHS.length];
        expect(blChar).toBe("/");
        expect(brChar).toBe("\\");
        expect(blChar).not.toBe(brChar);
    });

    test("indexes past the fill ramp — every directional index is >= CELL_FILL_GLYPHS.length", () => {
        for (const { a, b } of cases) {
            const { gx, gy } = sobelOfLinearField(a, b);
            expect(directionalGlyphIndex(gx, gy)).toBeGreaterThanOrEqual(CELL_FILL_GLYPHS.length);
        }
    });
});

describe("reinhard", () => {
    test("compresses toward 1 without ever reaching it", () => {
        const black = reinhard(vec3f(0, 0, 0));
        expect(black.x).toBe(0);
        expect(black.y).toBe(0);
        expect(black.z).toBe(0);
        const bright = reinhard(vec3f(9, 9, 9));
        expect(bright.x).toBeCloseTo(0.9, 5);
    });
});

describe("luma", () => {
    test("rec709 weights, white maps to 1", () => {
        expect(luma(vec3f(1, 1, 1))).toBeCloseTo(1, 5);
        expect(luma(vec3f(0, 0, 0))).toBe(0);
    });
});

describe("EDGE_MAGNITUDE_THRESHOLD", () => {
    // the kernel's true reachable maximum, derived rather than quoted: gx/gy are each linear in the 8
    // free luma taps (own excluded) selectKernel reads, each tap clamped to [0, 1], so a linear
    // objective (the Sobel magnitude, maximized over a direction) is maximized at a vertex of that
    // 8-cube — brute-forceable over all 256. `gx = gy = 4` (sqrt(32) ≈ 5.657) is NOT reachable: gx and
    // gy share taps with opposite sign (e.g. l02/l20), so driving gx to its own axis maximum caps what
    // gy can reach at the same time (`select.ts`'s own derivation).
    function sobelMagnitude(bits: number): number {
        const l = Array.from({ length: 8 }, (_, i) => (bits >> i) & 1);
        const [l00, l10, l20, l01, l21, l02, l12, l22] = l;
        const gx = l20 + 2 * l21 + l22 - (l00 + 2 * l01 + l02);
        const gy = l02 + 2 * l12 + l22 - (l00 + 2 * l10 + l20);
        return Math.hypot(gx, gy);
    }

    test("the kernel's reachable maximum is sqrt(20), not sqrt(32)", () => {
        let max = 0;
        for (let bits = 0; bits < 256; bits++) max = Math.max(max, sobelMagnitude(bits));
        expect(max).toBeCloseTo(Math.sqrt(20), 10);
    });

    test("sits strictly between zero and the Sobel kernel's reachable maximum", () => {
        let max = 0;
        for (let bits = 0; bits < 256; bits++) max = Math.max(max, sobelMagnitude(bits));
        // the threshold must gate somewhere inside that range or it can never fire, or always fires
        expect(EDGE_MAGNITUDE_THRESHOLD).toBeGreaterThan(0);
        expect(EDGE_MAGNITUDE_THRESHOLD).toBeLessThan(max);
    });
});

// rule 1's own repair proof ("a shared face boundary
// carries its own rule"): FACE_BOUNDARY_MAGNITUDE_THRESHOLD's own docblock claims a witnessed reproduction
// — a real yaw (2.4 rad) where a genuinely flat 4-connected neighborhood still read a nonzero full-Sobel
// magnitude solely from its diagonal taps landing in the next face over. This reproduces that shape
// directly (not the real device numbers, which live in the docblock's own prose) and proves
// localBoundaryMagnitude reads 0 where the diagonal-bearing Sobel construction would not.
describe("localBoundaryMagnitude / FACE_BOUNDARY_MAGNITUDE_THRESHOLD (rule 1's shared-face-boundary gate)", () => {
    test("a flat 4-connected neighborhood reads exactly 0, regardless of what the diagonal corners hold", () => {
        // own row/column all equal to the center's own luma (0.517) — a textbook-flat 4-neighborhood, the
        // exact shape the docblock's reproduction names.
        expect(localBoundaryMagnitude(0.517, 0.517, 0.517, 0.517)).toBe(0);
    });

    test("the full-Sobel construction this gate replaces would have fired FACE_BOUNDARY_MAGNITUDE_THRESHOLD on the same case, from diagonal taps alone", () => {
        // the reproduction's own numbers: own row/column flat at 0.517, but the diagonal corners (top-right,
        // bottom-right) sit in the next face over at 0.473 — a real, measured bled-Sobel input.
        const l00 = 0.517;
        const l10 = 0.517;
        const l20 = 0.473; // diagonal — the next face over
        const l01 = 0.517;
        const l21 = 0.517;
        const l02 = 0.517;
        const l12 = 0.517;
        const l22 = 0.473; // diagonal — the next face over
        const gx = l20 + 2 * l21 + l22 - (l00 + 2 * l01 + l02);
        const gy = l02 + 2 * l12 + l22 - (l00 + 2 * l10 + l20);
        const fullSobelMagnitude = Math.hypot(gx, gy);
        // the claim this reproduction rests on is that the replaced construction actually fires the
        // boundary gate on this input (|g| = 0.088), not merely that it reads nonzero — a bare `> 0`
        // passes even at 1e-12, far below FACE_BOUNDARY_MAGNITUDE_THRESHOLD (≈0.00581 at 87 fill
        // glyphs), which would not be the defect this rule exists to fix.
        expect(fullSobelMagnitude).toBeGreaterThan(FACE_BOUNDARY_MAGNITUDE_THRESHOLD);
        // and this gate's own localized metric reads 0 on the identical 4-connected neighbors
        expect(localBoundaryMagnitude(l01, l21, l10, l12)).toBe(0);
    });

    test("reads the target defect correctly — a real two-face luma step reads a nonzero, above-threshold magnitude", () => {
        // the amendment's own measured boundary: a ~0.012 luma step between two adjacent faces
        const magnitude = localBoundaryMagnitude(0.529, 0.517, 0.529, 0.529);
        expect(magnitude).toBeCloseTo(0.012, 5);
        expect(magnitude).toBeGreaterThan(FACE_BOUNDARY_MAGNITUDE_THRESHOLD);
    });

    test("sits strictly between zero and the metric's own reachable maximum (sqrt(2), a full 0-1 step on both axes)", () => {
        expect(FACE_BOUNDARY_MAGNITUDE_THRESHOLD).toBeGreaterThan(0);
        expect(FACE_BOUNDARY_MAGNITUDE_THRESHOLD).toBeLessThan(Math.SQRT2);
    });

    test("sits well below EDGE_MAGNITUDE_THRESHOLD — a second, lower gate, not a re-setting of the first", () => {
        expect(FACE_BOUNDARY_MAGNITUDE_THRESHOLD).toBeLessThan(EDGE_MAGNITUDE_THRESHOLD);
    });
});

describe("selectWgsl", () => {
    test("resolves both kernels with no device", () => {
        const wgsl = selectWgsl();
        expect(wgsl).toContain("fn");
        expect(wgsl.length).toBeGreaterThan(0);
    });
});

// The s3r item-8 repair's own regression guard: a real device dispatch (not the device-free structural
// arms above), because the defect this repair closes lived entirely in a runtime comparison
// (`BG_MATCH_EPSILON`'s distance gate against a live tonemapped average) no WGSL-text or pure-function
// arm can see. A non-black clear color's
// tonemapped luma never rounds to the ramp's zero-coverage entry, so a real scene's empty background read
// as a uniform field of `'` (glyph index 2) instead of blank — refuted twice at the wrong layer before
// this repair (the directional-glyph mechanism was never it; the fill computation itself was).
describe("background detection (s3r item 8's own repair)", () => {
    afterEach(() => {
        resetSelectPipelines();
        Object.assign(Compute, { root: undefined, device: undefined });
    });

    const Cols = 4;
    const Rows = 4;
    const Src = 16; // device px per axis, well over the 3x3 block-sample grid per cell

    // a uniform HDR source texture, every texel the same raw linear color — the shape a camera's
    // untouched clear-color background takes (no draw ever wrote a different value there).
    function uniformSourceTexture(color: readonly [number, number, number]): GPUTexture {
        const device = Compute.device;
        const texture = device.createTexture({
            label: "select-bg-test-src",
            size: { width: Src, height: Src },
            format: "rgba32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const data = new Float32Array(Src * Src * 4);
        for (let i = 0; i < Src * Src; i++) {
            data[i * 4] = color[0];
            data[i * 4 + 1] = color[1];
            data[i * 4 + 2] = color[2];
            data[i * 4 + 3] = 1;
        }
        device.queue.writeTexture(
            { texture },
            data,
            { bytesPerRow: Src * 16 },
            { width: Src, height: Src },
        );
        return texture;
    }

    // dispatch + one-shot readback (`probeBuffer`, the `tests/avbd/headless.tier.ts` pattern) — no Mirror
    // ring, no per-frame State, since this is a single dispatch against a hand-built texture.
    async function readGlyphs(
        color: readonly [number, number, number],
        bg?: readonly [number, number, number],
    ): Promise<number[]> {
        await requestGPU();
        // `Compute.span` is a module-level singleton `requestGPU` never clears (it only assigns
        // `device`/`root`/etc — `gpu.ts`'s own `Object.assign` list omits it), so a `ProfilePlugin`-
        // carrying test earlier in this same `bun test` process can leave a stale timestamp-writing
        // `span` installed on a device that never requested the `timestamp-query` feature. `recordSelect`
        // reads `Compute.span?.(...)` for its pass labels — a stale span makes the whole command buffer
        // invalid (a real GPU validation error, silently zeroing the readback), which reds this arm only
        // in full-suite order, never in isolation. This diagnostic dispatch needs no profiling.
        Compute.span = undefined;
        const texture = uniformSourceTexture(color);
        const grid = createCellGrid(Cols, Rows, CELL_GLYPH_COUNT);
        dispatchSelect(grid.buffer, Cols, Rows, texture.createView(), Src, Src, bg);
        const raw = Compute.root.unwrap(grid.buffer);
        const probe = await probeBuffer(Compute.device, raw, { label: "select-bg-test" });
        const glyphs: number[] = [];
        for (let i = 0; i < Cols * Rows; i++) glyphs.push(unpackCell(probe.bytes, i).glyph);
        texture.destroy();
        return glyphs;
    }

    test("a cell whose source region matches the caller's background reference selects the blank glyph", async () => {
        // reinhard(0.03) ≈ 0.0291 -> round(0.0291 * 90) = 3 absent the fix — a real, provably non-zero
        // fill index without the repair, so this arm reds first against the pre-fix mapping.
        const bg: [number, number, number] = [0.03, 0.03, 0.03];
        const glyphs = await readGlyphs(bg, bg);
        expect(glyphs.every((g) => g === 0)).toBe(true);
    });

    test("a mismatched background reference does not force the blank glyph — the gate discriminates rather than always firing", async () => {
        const glyphs = await readGlyphs([0.5, 0.5, 0.5], [0.03, 0.03, 0.03]);
        expect(glyphs.every((g) => g !== 0)).toBe(true);
    });

    test("omitting the background reference (NO_BACKGROUND) preserves the pre-repair mapping for a caller with no known background", async () => {
        // gym's own SelectPlugin scenario calls dispatchSelect this way (no bg argument) — its low-luma
        // SELECT_DARK regression guard expects a real ramp index, never the blank glyph, so the default
        // must stay inert.
        const glyphs = await readGlyphs([0.03, 0.03, 0.03]);
        expect(glyphs.every((g) => g > 0)).toBe(true);
    });

    test("BG_MATCH_EPSILON sits well inside the fill ramp's own per-index luma step", () => {
        // if the gate's tolerance ever grew past roughly half a ramp step, it would start swallowing a
        // real one-step-darker surface as background — this is the standing bound on that budget.
        const rampStep = 1 / (CELL_FILL_GLYPHS.length - 1);
        expect(BG_MATCH_EPSILON).toBeLessThan(rampStep / 2);
    });
});

// fillIndexForLuma's own in-range property, swept across its whole luma domain — device-free, unlike the
// real facade-ink measurement, which now lives entirely against a real device readback
// (`examples/gym/src/scenarios/cells.ts`'s `assertFacadeInk`, rendering `FACADE_BAND_LUMAS` through this
// mapping and the real draw pipeline, since only a real dispatch can measure rendered ink over a per-pixel
// threshold). The old version of this population was `FACADE_LUMA_SWEEP`, a 101-point sweep the facade-ink
// measurement itself averaged over; that measurement moved to the device (round-1's criterion-8 rejection:
// averaging over the whole ramp is not a facade), so the sweep's only remaining job is this file's own
// in-range check, local rather than exported.
describe("fillIndexForLuma (in-range property)", () => {
    const FullLumaDomainSweep: readonly number[] = Array.from({ length: 101 }, (_, i) => i / 100);

    test("the sweep is non-empty and spans the full luma domain — a non-vacuity control on the population itself", () => {
        expect(FullLumaDomainSweep.length).toBeGreaterThan(2);
        expect(Math.min(...FullLumaDomainSweep)).toBe(0);
        expect(Math.max(...FullLumaDomainSweep)).toBe(1);
    });

    test("fillIndexForLuma stays in range across the whole sweep", () => {
        for (const l of FullLumaDomainSweep) {
            const idx = fillIndexForLuma(l);
            expect(idx).toBeGreaterThanOrEqual(0);
            expect(idx).toBeLessThan(CELL_FILL_GLYPHS.length);
        }
    });
});

// fillIndexForLuma's own differential (`packages/shallot/tests/standards.ts` registry entry) — the
// range-check above proves the output stays in bounds, never that the rounding is correct, so this is
// the independent reference. WGSL's round() rounds ties to even, not JS's default half-away-from-zero,
// and `fillIndexForLuma`'s `luma` parameter is `d.f32`, so the input itself is quantized to float32
// precision before the ramp multiply — a boundary luma placed at an exact mathematical tie can land a
// few ULPs either side of it under that quantization and round the "wrong" way against naive
// double-precision arithmetic (reproduced directly against this file's own fixtures while deriving the
// epsilon below). 1e-4 sits many orders above that f32-rounding noise (~1e-7 relative), so "just below"
// / "just above" a boundary reads unambiguously regardless of which side an exact tie would land on.
function roundTiesToEven(v: number): number {
    const floor = Math.floor(v);
    return v === floor + 0.5 ? (floor % 2 === 0 ? floor : floor + 1) : Math.round(v);
}

// the reference mapping computed from scratch, never by calling `fillIndexForLuma` — a defect in the
// production rounding or clamping can't hide behind a shared implementation this way.
function referenceFillIndex(luma: number, fillCount: number): number {
    const sat = Math.min(1, Math.max(0, luma));
    const idx = roundTiesToEven(sat * (fillCount - 1));
    return Math.min(idx, fillCount - 1);
}

describe("fillIndexForLuma (the standards.ts differential)", () => {
    const fillCount = CELL_FILL_GLYPHS.length;
    const step = 1 / (fillCount - 1);

    test("exact index centers land on their own index, hand-computed", () => {
        for (const i of [0, 1, 2, Math.floor(fillCount / 2), fillCount - 2, fillCount - 1]) {
            const luma = i / (fillCount - 1);
            expect(fillIndexForLuma(luma)).toBe(referenceFillIndex(luma, fillCount));
            expect(fillIndexForLuma(luma)).toBe(i);
        }
    });

    test("out-of-domain luma clamps to the ramp's own endpoints", () => {
        expect(fillIndexForLuma(-0.5)).toBe(0);
        expect(fillIndexForLuma(1.5)).toBe(fillCount - 1);
    });

    test("crossing a rounding boundary flips the index by exactly one, on both sides", () => {
        // half a ramp step past an index center is where the mapping switches to its neighbor — the
        // same quantity FACE_BOUNDARY_MAGNITUDE_THRESHOLD derives (`select.ts`'s own docblock).
        const eps = 1e-4;
        for (const i of [0, 1, 10, Math.floor(fillCount / 2), fillCount - 12, fillCount - 2]) {
            const boundary = (i + 0.5) * step;
            const below = fillIndexForLuma(boundary - eps);
            const above = fillIndexForLuma(boundary + eps);
            expect(below).toBe(referenceFillIndex(boundary - eps, fillCount));
            expect(above).toBe(referenceFillIndex(boundary + eps, fillCount));
            expect(below).toBe(i);
            expect(above).toBe(i + 1);
        }
    });
});
