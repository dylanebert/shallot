import { describe, expect, test } from "bun:test";
import {
    BINDING_FLOOR,
    BYTES,
    CHUNK,
    CHUNK_CELLS,
    coord,
    DIM,
    faces,
    index,
    SLOTS,
    solid,
    TOTAL_CELLS,
} from "./grid";
import { checker, recenter, single, slab, solidChunk, sphere, tunnel } from "./patterns";

describe("capacity", () => {
    test("the active envelope fits one storage binding on the portable floor", () => {
        expect(TOTAL_CELLS).toBe(DIM.x * DIM.y * DIM.z);
        expect(BYTES).toBe(64 * 1024 * 1024);
        expect(BYTES).toBeLessThanOrEqual(BINDING_FLOOR);
    });
});

describe("addressing", () => {
    test("coord ∘ index round-trips at corners, chunk seams, and a random sample", () => {
        const cells: [number, number, number][] = [
            [0, 0, 0],
            [DIM.x - 1, DIM.y - 1, DIM.z - 1],
            [CHUNK - 1, CHUNK - 1, CHUNK - 1],
            [CHUNK, CHUNK, CHUNK],
            [CHUNK - 1, 0, 0],
            [CHUNK, 0, 0],
            [0, CHUNK - 1, 0],
            [0, CHUNK, 0],
            [0, 0, CHUNK - 1],
            [0, 0, CHUNK],
        ];
        for (const [x, y, z] of cells) expect(coord(index(x, y, z))).toEqual([x, y, z]);

        let s = 0x9e3779b9 >>> 0;
        const rand = () => {
            s = (s * 1664525 + 1013904223) >>> 0;
            return s / 0x100000000;
        };
        for (let i = 0; i < 4000; i++) {
            const x = Math.floor(rand() * DIM.x);
            const y = Math.floor(rand() * DIM.y);
            const z = Math.floor(rand() * DIM.z);
            expect(coord(index(x, y, z))).toEqual([x, y, z]);
        }
    });

    test("index endpoints span exactly [0, TOTAL_CELLS)", () => {
        expect(index(0, 0, 0)).toBe(0);
        expect(index(DIM.x - 1, DIM.y - 1, DIM.z - 1)).toBe(TOTAL_CELLS - 1);
    });

    test("each chunk occupies a contiguous CHUNK_CELLS range with no collisions (chunk-major)", () => {
        const chunks: [number, number, number][] = [
            [0, 0, 0],
            [1, 0, 0],
            [3, 5, 7],
            [SLOTS.x - 1, SLOTS.y - 1, SLOTS.z - 1],
        ];
        for (const [sx, sy, sz] of chunks) {
            const base = ((sz * SLOTS.y + sy) * SLOTS.x + sx) * CHUNK_CELLS;
            const seen = new Set<number>();
            for (let lz = 0; lz < CHUNK; lz++) {
                for (let ly = 0; ly < CHUNK; ly++) {
                    for (let lx = 0; lx < CHUNK; lx++) {
                        const off = index(sx * CHUNK + lx, sy * CHUNK + ly, sz * CHUNK + lz);
                        expect(off).toBeGreaterThanOrEqual(base);
                        expect(off).toBeLessThan(base + CHUNK_CELLS);
                        seen.add(off);
                    }
                }
            }
            expect(seen.size).toBe(CHUNK_CELLS);
        }
    });

    test("adjacent cells across a chunk seam land in different slots", () => {
        const a = index(CHUNK - 1, 10, 10);
        const b = index(CHUNK, 10, 10);
        expect(coord(a)).toEqual([CHUNK - 1, 10, 10]);
        expect(coord(b)).toEqual([CHUNK, 10, 10]);
        expect(Math.floor(a / CHUNK_CELLS)).not.toBe(Math.floor(b / CHUNK_CELLS));
    });
});

function count(data: Float32Array): number {
    let n = 0;
    for (let i = 0; i < data.length; i++) if (data[i] !== 0) n++;
    return n;
}

describe("canonical patterns", () => {
    test("single — one solid cell, neighbors air", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        expect(solid(data, 40, 50, 60)).toBe(true);
        expect(solid(data, 41, 50, 60)).toBe(false);
        expect(solid(data, 40, 51, 60)).toBe(false);
        expect(solid(data, 39, 50, 60)).toBe(false);
        expect(count(data)).toBe(1);
    });

    test("solid — a full chunk solid, exterior air", () => {
        const data = new Float32Array(TOTAL_CELLS);
        solidChunk(data, 1, 1, 1);
        expect(solid(data, 32, 32, 32)).toBe(true);
        expect(solid(data, 63, 63, 63)).toBe(true);
        expect(solid(data, 48, 48, 48)).toBe(true);
        expect(solid(data, 31, 48, 48)).toBe(false);
        expect(solid(data, 64, 48, 48)).toBe(false);
        expect(count(data)).toBe(CHUNK_CELLS);
    });

    test("checker — alternating solid / air by parity", () => {
        const data = new Float32Array(TOTAL_CELLS);
        checker(data, 0, 0, 0, 16, 16, 16);
        for (let z = 0; z < 16; z++) {
            for (let y = 0; y < 16; y++) {
                for (let x = 0; x < 16; x++) {
                    expect(solid(data, x, y, z)).toBe(((x + y + z) & 1) === 0);
                }
            }
        }
    });

    test("slab — ground plane solid below height", () => {
        const data = new Float32Array(TOTAL_CELLS);
        slab(data, 3);
        for (const [x, z] of [
            [0, 0],
            [128, 128],
            [DIM.x - 1, DIM.z - 1],
        ] as const) {
            for (let y = 0; y < 6; y++) expect(solid(data, x, y, z)).toBe(y < 3);
        }
    });

    test("tunnel — solid block with an inward-facing bored channel", () => {
        const data = new Float32Array(TOTAL_CELLS);
        tunnel(data, 10, 10, 10, 16);
        const c = 16 >> 1;
        for (let z = 0; z < 16; z++) expect(solid(data, 10 + c, 10 + c, 10 + z)).toBe(false);
        expect(solid(data, 10 + c + 1, 10 + c, 15)).toBe(true);
        expect(solid(data, 10, 10, 10)).toBe(true);
        expect(solid(data, 25, 25, 25)).toBe(true);
        expect(solid(data, 9, 10, 10)).toBe(false);
        expect(solid(data, 26, 10, 10)).toBe(false);
    });

    test("sphere — solid ball spanning multiple chunks, watertight occupancy", () => {
        const data = new Float32Array(TOTAL_CELLS);
        const cx = 128;
        const cy = 128;
        const cz = 128;
        const r = 40; // > CHUNK (32) → spans chunks
        sphere(data, cx, cy, cz, r);
        const r2 = r * r;
        const inside = (x: number, y: number, z: number) =>
            (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2;
        for (let z = cz - r - 2; z <= cz + r + 2; z += 3) {
            for (let y = cy - r - 2; y <= cy + r + 2; y += 3) {
                for (let x = cx - r - 2; x <= cx + r + 2; x += 3) {
                    expect(solid(data, x, y, z)).toBe(inside(x, y, z));
                }
            }
        }
        expect(solid(data, cx, cy, cz)).toBe(true);
        expect(solid(data, cx - 35, cy, cz)).toBe(true);
        expect(solid(data, cx + 35, cy, cz)).toBe(true);
        expect(Math.floor(index(cx - 35, cy, cz) / CHUNK_CELLS)).not.toBe(
            Math.floor(index(cx + 35, cy, cz) / CHUNK_CELLS),
        );
    });
});

describe("exposed-face oracle", () => {
    // `faces` is the analytic ground truth the GPU mesher is gated against (the voxel gate), so
    // validate it here against closed-form counts derivable by geometry. The oracle has one code path —
    // per solid cell, count air/out-of-bounds neighbours — so these cover it; the tunnel's inward faces
    // and the sphere's seam add GPU coverage, not a new oracle path, and are checked GPU-vs-oracle in the voxel gate.

    test("single isolated voxel — six faces", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        expect(faces(data)).toBe(6);
    });

    test("edge voxel — out-of-bounds neighbours count as exposed", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 0, 0, 0);
        expect(faces(data)).toBe(6); // all six neighbours are OOB → all six faces emit
    });

    test("solid chunk — only the 32³ shell, interior occluded", () => {
        const data = new Float32Array(TOTAL_CELLS);
        solidChunk(data, 1, 1, 1);
        expect(faces(data)).toBe(6 * CHUNK * CHUNK);
    });

    test("checker — every solid cell fully exposed (parity isolates it)", () => {
        const data = new Float32Array(TOTAL_CELLS);
        checker(data, 0, 0, 0, 16, 16, 16);
        expect(faces(data)).toBe(6 * (16 ** 3 / 2));
    });

    test("slab — two caps plus four sides over the full footprint", () => {
        const data = new Float32Array(TOTAL_CELLS);
        const h = 3;
        slab(data, h);
        expect(faces(data)).toBe(2 * DIM.x * DIM.z + 2 * h * DIM.x + 2 * h * DIM.z);
    });
});

describe("recenter", () => {
    // recentring is a rigid translation, so it must preserve both the solid count and the exposed-face
    // count (the mesher gate compares the GPU count to `faces()` on the recentred grid). The scenario
    // relies on that: it recentres every fixture for the orbit view without perturbing the correctness gate.
    test("centres a corner fixture, preserving solid + face count", () => {
        const corner = new Float32Array(TOTAL_CELLS);
        checker(corner, 0, 0, 0, 16, 16, 16); // jammed in the (0,0,0) octant
        const centred = recenter(corner);

        // a rigid translation: same number of solid cells, same exposed-face count
        expect(count(centred)).toBe(count(corner));
        expect(faces(centred)).toBe(faces(corner));

        // the 16³ box (cells [0..15]) shifts by +120 on each axis to straddle the grid centre [120..135]
        expect(solid(corner, 0, 0, 0)).toBe(true);
        expect(solid(centred, 0, 0, 0)).toBe(false);
        expect(solid(centred, 120, 120, 120)).toBe(true); // was (0,0,0)
        expect(solid(centred, 135, 135, 134)).toBe(true); // was (15,15,14), parity 0 → solid
    });

    test("leaves an already-centred fixture in place (no spurious shift)", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 128, 128, 128); // dead centre → zero shift; any erroneous shift moves the voxel
        const centred = recenter(data);
        expect(count(centred)).toBe(1);
        expect(solid(centred, 128, 128, 128)).toBe(true);
    });
});
