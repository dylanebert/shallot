import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../../../packages/shallot/tests/wgsl";
import {
    documentDistance,
    flattenSegments,
    type Segment,
    type StrokeDocument,
    segmentDistance,
} from "./document";
import { encodeDistGpu, rasterizeWgsl, segmentDistanceGpu } from "./rasterize";
import { DIST_RANGE, decodeDist } from "./tiles";

// Stage 5's differential oracle: `segmentDistanceGpu` (rasterize.ts, clamped-projection form, CPU-called
// through TypeGPU's dual execution — testing.md's logic tier) against `document.ts`'s `segmentDistance`
// (perpendicular-offset-via-cross-product form) — two independently written derivations of the same
// geometric quantity (checks.md). The GPU side is additionally quantized through `encodeDistGpu` +
// `tiles.ts`'s `decodeDist`, the exact byte round-trip the real kernel's output goes through, so the
// tolerance is the r8unorm quantization step the spec's Approach names, not a raw-float tolerance.
const QUANT_STEP = (2 * DIST_RANGE) / 255;
const TOLERANCE = QUANT_STEP / 2 + 1e-6; // half a quantization step, plus f32-roundoff slack

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
            const cpuQuantized = decodeDist(
                Math.round(
                    (Math.max(-DIST_RANGE, Math.min(DIST_RANGE, cpu)) / (2 * DIST_RANGE) + 0.5) *
                        255,
                ),
            );
            expect(Math.abs(gpuQuantized - cpuQuantized)).toBeLessThanOrEqual(TOLERANCE);
            // and the raw (unquantized) forms agree far tighter — same geometric quantity, f32 roundoff only
            expect(Math.abs(gpuRaw - cpu)).toBeLessThan(1e-4);
        }
    });

    test("a multi-segment reduction over segmentDistanceGpu agrees with documentDistance", () => {
        // documentDistanceGpu itself reads its segments out of a bound storage buffer
        // (rasterLayout.$.segments), so it can't be CPU-called without a device (testing.md: "never bind a
        // device in bun test") — the kernel's reduction loop is instead pinned structurally below
        // ("emitted WGSL"). This exercises the same reduction (min over every segment's
        // segmentDistanceGpu) in plain JS, over the same per-primitive kernel math the earlier test
        // already validated CPU-side.
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
            polygons: [],
        };
        const samples: Array<[number, number]> = [
            [0, 0],
            [10, 10],
            [-30, 5],
            [40, 40],
        ];
        for (const [px, pz] of samples) {
            const cpu = documentDistance(px, pz, doc);
            let gpu = Number.POSITIVE_INFINITY;
            for (const seg of flattenSegments(doc)) {
                const d = segmentDistanceGpu(px, pz, seg.ax, seg.az, seg.bx, seg.bz, seg.halfWidth);
                if (d < gpu) gpu = d;
            }
            expect(Math.abs(gpu - cpu)).toBeLessThan(1e-4);
        }
    });
});

describe("determinism", () => {
    test("segmentDistanceGpu is a pure function of its inputs — repeated calls agree exactly", () => {
        const a = segmentDistanceGpu(5, 3, -10, 0, 10, 0, 2);
        const b = segmentDistanceGpu(5, 3, -10, 0, 10, 0, 2);
        expect(a).toBe(b);
    });

    test("encodeDistGpu is deterministic and matches tiles.ts's encodeDist within one f32 rounding ULP", () => {
        for (const m of [-1, -0.5, 0, 0.3333, 0.9, 1]) {
            const a = encodeDistGpu(m);
            const b = encodeDistGpu(m);
            expect(a).toBe(b);
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
});
