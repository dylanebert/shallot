import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../../../packages/shallot/tests/wgsl";
import {
    documentDistance,
    flattenSegments,
    type PolygonStamp,
    polygonDistance,
    type Segment,
    type StrokeDocument,
    segmentDistance,
} from "./document";
import {
    encodeDistGpu,
    polygonSignedDistanceGpu,
    rasterizeWgsl,
    segmentDistanceGpu,
} from "./rasterize";
import { DIST_RANGE, decodeDist, encodeDist, TEXEL_SIZE } from "./tiles";

// Stage 5's differential oracle: `segmentDistanceGpu` (rasterize.ts, clamped-projection form, CPU-called
// through TypeGPU's dual execution) against `document.ts`'s `segmentDistance` (perpendicular-offset-via-
// cross-product form) — two independently written derivations of the same geometric quantity. The GPU
// side is additionally quantized through `encodeDistGpu` +
// `tiles.ts`'s `decodeDist`, the exact byte round-trip the real kernel's output goes through, so the
// tolerance is the r8unorm quantization step the spec's Approach names, not a raw-float tolerance.
const QUANT_STEP = (2 * DIST_RANGE) / 255;
const TOLERANCE = QUANT_STEP / 2 + 1e-6; // half a quantization step, plus f32-roundoff slack

/** the same CPU-side r8unorm quantization `encodeDist`/`encodeDistGpu` both perform, inlined once here
 *  so both differential oracles (segment, polygon) key off one derivation rather than two copies. */
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

    test("a multi-primitive (segments + polygons) reduction agrees with documentDistance", () => {
        // documentDistanceGpu itself reads its segments/polygons out of bound storage buffers
        // (rasterLayout.$.segments / .polygons / .polyVerts), so it can't be CPU-called without a device —
        // the kernel's reduction loop is instead pinned structurally below ("emitted WGSL"). This exercises
        // the same reduction (min over every segment's
        // segmentDistanceGpu AND every polygon's winding+edge reduction) in plain JS, over the same
        // per-primitive kernel math the earlier tests already validated CPU-side — and, unlike the
        // segment-only reduction this replaces, carries a non-empty `polygons` array so
        // `documentDistanceGpu`'s polygon arm is exercised by more than the empty-array default every
        // other test in this file used.
        const doc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-50, 0],
                        [50, 0],
                    ],
                    halfWidth: 3,
                },
                {
                    points: [
                        [0, -50],
                        [0, 50],
                    ],
                    halfWidth: 2,
                },
            ],
            polygons: [
                {
                    points: [
                        [20, 20],
                        [35, 20],
                        [35, 32],
                        [20, 32],
                    ],
                },
            ],
        };
        const samples: Array<[number, number]> = [
            [0, 0],
            [10, 10],
            [-30, 5],
            [40, 40],
            [27, 26], // inside the polygon stamp
            [27, 40], // outside every primitive, nearest the polygon
        ];
        for (const [px, pz] of samples) {
            const cpu = documentDistance(px, pz, doc);
            let gpu = Number.POSITIVE_INFINITY;
            for (const seg of flattenSegments(doc)) {
                const d = segmentDistanceGpu(px, pz, seg.ax, seg.az, seg.bx, seg.bz, seg.halfWidth);
                if (d < gpu) gpu = d;
            }
            for (const poly of doc.polygons) {
                const d = jsPolygonSignedDistance(px, pz, poly.points);
                if (d < gpu) gpu = d;
            }
            expect(Math.abs(gpu - cpu)).toBeLessThan(1e-4);
        }
    });
});

/**
 * plain-JS mirror of `polygonDistanceGpu`'s winding + nearest-edge reduction — over `segmentDistanceGpu`
 * (the same CPU-callable primitive the real reduction calls per edge, kept as an independent derivation),
 * finishing with the real, exported `polygonSignedDistanceGpu` for the sign. The reduction loop
 * (winding parity, nearest-edge min) is a JS reimplementation — like `segmentsDistanceGpu`'s reduction
 * above, it reads a bound storage array GPU-side and can't be CPU-called directly — but the sign
 * convention itself, the exact axis a flipped `select` breaks, calls the real kernel code, not a copy.
 */
function jsPolygonSignedDistance(
    px: number,
    pz: number,
    pts: ReadonlyArray<readonly [number, number]>,
): number {
    let inside = false;
    let nearestEdge = Number.POSITIVE_INFINITY;
    for (let i = 0; i < pts.length; i++) {
        const [ax, az] = pts[i];
        const [bx, bz] = pts[(i + 1) % pts.length];
        if (az > pz !== bz > pz) {
            const xCross = ax + ((pz - az) / (bz - az)) * (bx - ax);
            if (px < xCross) inside = !inside;
        }
        const edge = segmentDistanceGpu(px, pz, ax, az, bx, bz, 0);
        if (edge < nearestEdge) nearestEdge = edge;
    }
    return polygonSignedDistanceGpu(nearestEdge, inside);
}

describe("differential oracle — polygonDistanceGpu's sign convention vs document.ts's polygonDistance", () => {
    test("agree within one quantization step over random polygons and sample points", () => {
        const rng = mulberry32(11);
        const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
        for (let i = 0; i < 150; i++) {
            const n = 3 + Math.floor(rng() * 5); // 3..7 vertices, roughly convex
            const cx = rand(-50, 50);
            const cz = rand(-50, 50);
            const r = rand(3, 20);
            const pts: Array<[number, number]> = [];
            for (let v = 0; v < n; v++) {
                const a = (v / n) * Math.PI * 2 + rand(-0.2, 0.2);
                pts.push([cx + Math.cos(a) * r, cz + Math.sin(a) * r]);
            }
            const poly: PolygonStamp = { points: pts };
            const px = rand(-70, 70);
            const pz = rand(-70, 70);

            const cpu = polygonDistance(px, pz, poly);
            const gpuRaw = jsPolygonSignedDistance(px, pz, pts);
            const gpuQuantized = decodeDist(encodeDistGpu(gpuRaw));
            const cpuQuantized = quantizeCpu(cpu);
            expect(Math.abs(gpuQuantized - cpuQuantized)).toBeLessThanOrEqual(TOLERANCE);
            // and the raw (unquantized) forms agree far tighter — same geometric quantity, f32 roundoff only
            expect(Math.abs(gpuRaw - cpu)).toBeLessThan(1e-4);
        }
    });

    test("the sign flips exactly at the boundary: a point moved just inside/outside a square flips sign", () => {
        const square: Array<[number, number]> = [
            [-10, -10],
            [10, -10],
            [10, 10],
            [-10, 10],
        ];
        const inside = jsPolygonSignedDistance(0, 0, square);
        const outside = jsPolygonSignedDistance(20, 20, square);
        expect(inside).toBeLessThan(0);
        expect(outside).toBeGreaterThan(0);
        // demonstrated mutation this pins: swapping polygonSignedDistanceGpu's select args would make
        // `inside` positive and `outside` negative — both assertions above would fail.
        expect(polygonSignedDistanceGpu(5, false)).toBe(5);
        expect(polygonSignedDistanceGpu(5, true)).toBe(-5);
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

    test("albedo is written once per texel with the precomputed constant word, not recomputed per-thread", () => {
        expect(flat(wgsl)).toContain("const roadAlbedoWord: u32 =");
        const kernel = flat(body(wgsl, "@compute @workgroup_size(8, 8, 1) fn roadsrasterize"));
        expect(kernel).toContain("albedoOut[albedoIdx] = roadAlbedoWord");
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
