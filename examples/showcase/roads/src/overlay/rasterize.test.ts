import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../../../packages/shallot/tests/wgsl";
import { markingDistanceForSegment, type Segment, segmentDistance } from "./document";
import { encodeDistGpu, markingDistanceGpu, rasterizeWgsl, segmentDistanceGpu } from "./rasterize";
import { DIST_RANGE, decodeDist, encodeDist, TEXEL_SIZE } from "./tiles";

// Stage 5's differential oracle: `segmentDistanceGpu` (rasterize.ts, clamped-projection form, CPU-called
// through TypeGPU's dual execution) against `document.ts`'s `segmentDistance` (perpendicular-offset-via-
// cross-product form) — two independently written derivations of the same geometric quantity. The GPU
// side is additionally quantized through `encodeDistGpu` +
// `tiles.ts`'s `decodeDist`, the exact byte round-trip the real kernel's output goes through, so the
// tolerance is the r8unorm quantization step the spec's Approach names, not a raw-float tolerance.
const QUANT_STEP = (2 * DIST_RANGE) / 255;
const TOLERANCE = QUANT_STEP / 2 + 1e-6; // half a quantization step, plus f32-roundoff slack

/** the same CPU-side r8unorm quantization `encodeDist`/`encodeDistGpu` both perform, inlined once here. */
function quantizeCpu(metres: number): number {
    return decodeDist(
        Math.round(
            (Math.max(-DIST_RANGE, Math.min(DIST_RANGE, metres)) / (2 * DIST_RANGE) + 0.5) * 255,
        ),
    );
}

function mulberry32(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) | 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

describe("differential oracle — segmentDistanceGpu vs document.ts's segmentDistance", () => {
    test("agree within one quantization step over random segments and sample points", () => {
        const rng = mulberry32(7);
        const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
        for (let i = 0; i < 200; i++) {
            const seg: Segment = {
                ax: rand(-100, 100),
                az: rand(-100, 100),
                bx: rand(-100, 100),
                bz: rand(-100, 100),
                halfWidth: rand(0.5, 8),
            };
            const px = rand(-120, 120);
            const pz = rand(-120, 120);

            const cpu = segmentDistance(px, pz, seg);
            const gpuRaw = segmentDistanceGpu(
                px,
                pz,
                seg.ax,
                seg.az,
                seg.bx,
                seg.bz,
                seg.halfWidth,
            );
            const gpuQuantized = decodeDist(encodeDistGpu(gpuRaw));
            const cpuQuantized = quantizeCpu(cpu);
            expect(Math.abs(gpuQuantized - cpuQuantized)).toBeLessThanOrEqual(TOLERANCE);
            // and the raw (unquantized) forms agree far tighter — same geometric quantity, f32 roundoff only
            expect(Math.abs(gpuRaw - cpu)).toBeLessThan(1e-4);
        }
    });
});

describe("determinism", () => {
    test("segmentDistanceGpu is a pure function of its inputs — repeated calls agree exactly", () => {
        const a = segmentDistanceGpu(5, 3, -10, 0, 10, 0, 2);
        const b = segmentDistanceGpu(5, 3, -10, 0, 10, 0, 2);
        expect(a).toBe(b);
    });

    test("encodeDistGpu matches tiles.ts's real encodeDist within one f32 rounding ULP", () => {
        // the earlier title claimed this comparison without importing or calling tiles.ts's encodeDist —
        // fixed by actually calling it: the two codecs (TGSL f32 vs plain JS f64) round the same unit
        // fraction to a byte, so they can differ by at most one rounding step at a bin edge.
        for (const m of [-1, -0.5, 0, 0.3333, 0.9, 1]) {
            const gpu = encodeDistGpu(m);
            const cpu = encodeDist(m);
            expect(Math.abs(gpu - cpu)).toBeLessThanOrEqual(1);
        }
    });
});

describe("emitted WGSL — structural", () => {
    const wgsl = rasterizeWgsl();

    test("segmentDistanceGpu clamps the projection parameter to [0, 1]", () => {
        const fn = flat(body(wgsl, "fn segmentDistanceGpu"));
        expect(fn).toContain("clamp(");
        expect(fn).toContain("distance(");
    });

    test("the rasterize kernel dispatches 128×512 threads (GROUPS_PER_SIDE × TILE_RES) at workgroup 8×8", () => {
        const kernel = flat(body(wgsl, "@compute @workgroup_size(8, 8, 1) fn roadsrasterize"));
        expect(kernel).toContain("gx >= 128u");
        expect(kernel).toContain("gy >= 512u");
    });

    test("every index computation stays unsigned (integerDiscipline)", () => {
        integerDiscipline(wgsl);
    });

    test("the kernel packs 4 texels' distance bytes little-endian into one u32", () => {
        const kernel = flat(body(wgsl, "@compute @workgroup_size(8, 8, 1) fn roadsrasterize"));
        expect(kernel).toContain("word = (word | (byte << (k * 8u)))");
    });

    test("albedo rgb is a precomputed constant; the per-texel alpha carries the encoded marking distance", () => {
        expect(flat(wgsl)).toContain("const roadAlbedoRgb: u32 =");
        const kernel = flat(body(wgsl, "@compute @workgroup_size(8, 8, 1) fn roadsrasterize"));
        expect(kernel).toContain("albedoOut[albedoIdx] = (roadAlbedoRgb | (markingByte << 24u))");
    });

    describe("texel-group column ↔ world-x ↔ byte-column correspondence", () => {
        // The `toContain` checks above pin the *shift* (`k * 8u`) — which byte column `k` lands in — but
        // not which texel `k` actually reads its distance from. A horizontal mirror within the 4-texel
        // group (`texelX = gx*4 + (3-k)` instead of `gx*4 + k`) leaves every existing structural check
        // textually unchanged, since it touches neither the packing shift nor the dispatch bounds. Pin it
        // behaviorally instead: pull the *actual* emitted `texelX`/`worldX` formulas out of the resolved
        // WGSL and evaluate them for concrete (gx, k) pairs, against an independent hand computation —
        // gx*GROUP_TEXELS+k for texelX, origin + (texelX+0.5)*TEXEL_SIZE for worldX — so a mutated
        // formula evaluates to the wrong number here even though it satisfies a bare string match.
        const kernel = flat(body(wgsl, "@compute @workgroup_size(8, 8, 1) fn roadsrasterize"));
        const GroupTexels = 4; // rasterize.ts's own module constant — not exported, restated here as
        // the spec-given "4 x r8 = one u32" fact the module header names, not copied from the source

        /** the RHS expression of `let {name} = <expr>;` in `src`, matched by paren-depth rather than a
         *  fixed-shape regex — robust to the emitted expression's exact parenthesization. */
        function extractExpr(src: string, name: string): string {
            const marker = `let ${name} = `;
            const start = src.indexOf(marker);
            expect(start).toBeGreaterThanOrEqual(0);
            let i = start + marker.length;
            let depth = 0;
            const exprStart = i;
            for (; i < src.length; i++) {
                const c = src[i];
                if (c === "(") depth++;
                else if (c === ")") depth--;
                else if (c === ";" && depth === 0) break;
            }
            return src.slice(exprStart, i);
        }

        /** evaluate a small WGSL arithmetic expression as JS — strips `u`/`f` numeric suffixes and
         *  substitutes bare identifiers, then runs it through the JS arithmetic parser via `Function`. The
         *  expression comes straight out of the resolved WGSL, so this evaluates the kernel's *actual*
         *  current formula, not an assumption about its shape. */
        function evalWgslExpr(expr: string, vars: Record<string, number>): number {
            let js = expr.replace(/(\d+(?:\.\d+)?)[uf]\b/g, "$1");
            for (const [name, value] of Object.entries(vars)) {
                js = js.replace(new RegExp(`\\b${name}\\b`, "g"), String(value));
            }
            return Function(`"use strict"; return (${js});`)();
        }

        const texelXExpr = extractExpr(kernel, "texelX");
        const worldXExpr = extractExpr(kernel, "worldX");

        test("texelX(gx, k) === gx*GROUP_TEXELS + k for every (gx, k), independently hand-computed", () => {
            for (const gx of [0, 1, 5, 31, 64, 127]) {
                for (let k = 0; k < GroupTexels; k++) {
                    const actual = evalWgslExpr(texelXExpr, { gx, k });
                    expect(actual).toBe(gx * GroupTexels + k);
                }
            }
        });

        test("texelX is strictly increasing in k within one group (rules out any within-group reorder)", () => {
            for (const gx of [0, 3, 64]) {
                let prev = Number.NEGATIVE_INFINITY;
                for (let k = 0; k < GroupTexels; k++) {
                    const x = evalWgslExpr(texelXExpr, { gx, k });
                    expect(x).toBeGreaterThan(prev);
                    prev = x;
                }
            }
        });

        test("worldX(texelX) === origin.x + (texelX + 0.5) * TEXEL_SIZE, chained from the real texelX formula", () => {
            const origin = { x: -37.5 }; // an arbitrary non-zero tile origin — catches an origin-ignoring bug
            for (const gx of [0, 7, 96]) {
                for (let k = 0; k < GroupTexels; k++) {
                    const texelX = evalWgslExpr(texelXExpr, { gx, k });
                    const worldExpr = worldXExpr
                        .replace(/\(\*origin\)\.x/g, String(origin.x))
                        .replace(/f32\(texelX\)/g, String(texelX));
                    const actual = evalWgslExpr(worldExpr, {});
                    const expected = origin.x + (texelX + 0.5) * TEXEL_SIZE;
                    expect(actual).toBeCloseTo(expected, 9);
                }
            }
        });

        test("the packing expression itself (a cheap structural sentinel, in addition to the eval-based pins above — never the only one)", () => {
            expect(kernel).toContain("let texelX = ((gx * 4u) + k);");
        });
    });
});

// Stage 3's marking differential oracle: `markingDistanceGpu` (rasterize.ts, clamped-projection form,
// CPU-called through TypeGPU's dual execution) against `document.ts`'s `markingDistanceForSegment`
// (cross-product form) — two independently written derivations of the same geometric quantity, the
// signed distance to the nearest road marking (edge lines + broken centreline). The GPU side is
// quantized through `encodeDistGpu` + `tiles.ts`'s `decodeDist`, the exact byte round-trip the real
// kernel's output goes through, so the tolerance is the same r8unorm quantization step as the coverage
// differential above. `rasterize.ts` never imports `markingDistance` — the two derivations are
// independent by contract, because the differential arm is worthless if one side calls the other.
//
// RED-FIRST CONTROL: mutating one side's dash duty (DASH_DUTY) was witnessed red — changing the CPU
// side's DASH_DUTY to DASH_DUTY * 0.5 (halving the dash fraction) while leaving the GPU side unchanged
// produces disagreement at points inside a dash whose phase falls in the second half of the original
// duty cycle: the CPU side now reads those points as in a gap (marking distance positive) while the GPU
// side still reads them as in a dash (marking distance negative), exceeding the tolerance. This was
// confirmed by running the mutated comparison and observing the failure before restoring the real duty.
describe("marking differential oracle — markingDistanceGpu vs document.ts's markingDistanceForSegment", () => {
    test("agree within one quantization step over random segments and sample points", () => {
        const rng = mulberry32(42);
        const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
        for (let i = 0; i < 200; i++) {
            const seg: Segment = {
                ax: rand(-100, 100),
                az: rand(-100, 100),
                bx: rand(-100, 100),
                bz: rand(-100, 100),
                halfWidth: rand(1, 8),
            };
            const px = rand(-120, 120);
            const pz = rand(-120, 120);

            const cpu = markingDistanceForSegment(px, pz, seg);
            const gpuRaw = markingDistanceGpu(
                px,
                pz,
                seg.ax,
                seg.az,
                seg.bx,
                seg.bz,
                seg.halfWidth,
            );
            const gpuQuantized = decodeDist(encodeDistGpu(gpuRaw));
            const cpuQuantized = quantizeCpu(cpu);
            expect(Math.abs(gpuQuantized - cpuQuantized)).toBeLessThanOrEqual(TOLERANCE);
            // and the raw (unquantized) forms agree far tighter — same geometric quantity, f32 roundoff only
            expect(Math.abs(gpuRaw - cpu)).toBeLessThan(1e-4);
        }
    });

    test("agree on the centreline (where the dash phase is the nearest marking), inside a dash and in a gap", () => {
        // a horizontal road from (-100, 0) to (100, 0), halfWidth 4 — the standard chord's shape.
        // Points on the centreline (z=0) are where the centreline marking is nearest, so the dash
        // duty actually determines the distance — this is the axis the red-first control mutates.
        const seg: Segment = { ax: -100, az: 0, bx: 100, bz: 0, halfWidth: 4 };
        // station 100: phase = fract((100 + DASH_OFFSET) / DASH_PERIOD) ≈ 0.45 >= DASH_DUTY (0.25) → in a gap (offset shifts midpoint into gap)
        // station 120: phase = fract((120 + DASH_OFFSET) / DASH_PERIOD) ≈ 0.09 < DASH_DUTY → in a dash
        const inDash: Array<[number, number]> = [
            [20, 0], // station 120, on the centreline, in a dash
            [70, 0.01], // station 170, near the centreline, in a dash
        ];
        // station 100 (midpoint): phase ≈ 0.45 >= DASH_DUTY → in a gap (the offset's purpose)
        // station 50: phase = fract((50 + DASH_OFFSET) / DASH_PERIOD) ≈ 0.35 >= DASH_DUTY → in a gap
        const inGap: Array<[number, number]> = [
            [0, 0], // station 100, on the centreline, in a gap
            [-50, 0.01], // station 50, near the centreline, in a gap
        ];
        for (const [px, pz] of inDash) {
            const cpu = markingDistanceForSegment(px, pz, seg);
            const gpuRaw = markingDistanceGpu(
                px,
                pz,
                seg.ax,
                seg.az,
                seg.bx,
                seg.bz,
                seg.halfWidth,
            );
            expect(Math.abs(gpuRaw - cpu)).toBeLessThan(1e-4);
            // inside a dash → marking distance should be negative (inside the marking)
            expect(cpu).toBeLessThan(0);
        }
        for (const [px, pz] of inGap) {
            const cpu = markingDistanceForSegment(px, pz, seg);
            const gpuRaw = markingDistanceGpu(
                px,
                pz,
                seg.ax,
                seg.az,
                seg.bx,
                seg.bz,
                seg.halfWidth,
            );
            expect(Math.abs(gpuRaw - cpu)).toBeLessThan(1e-4);
            // in a gap → marking distance should be positive (outside the marking)
            expect(cpu).toBeGreaterThan(0);
        }
    });
});
