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
// what you get if you *don't* convert frames, and it passed for a full review round (`specs/shallot-tui.md`
// s3r item 1). Naming the luma neighborhood a case derives from ties the expectation to a real edge
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
// arm can see. `specs/shallot-tui.md`'s residue log names what this replaces: a non-black clear color's
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
