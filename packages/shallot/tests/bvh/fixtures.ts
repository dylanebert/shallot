// BVH fixture set — the shared input scenes for the CPU oracle (oracle.ts) and the
// real-GPU harness (the gym's `accel` scenario). Both must build the *same*
// prims, or the GPU-vs-oracle validation is meaningless, so the generators live here,
// dependency-free, and are imported by both sides. Test scaffolding, not engine code —
// kept out of the shipped src/ tree.
//
// AABB-only, matching kitchen/bvh/core: the builder's input is a primitive AABB array,
// and its query is ray-AABB. Consumer-specific scenes (a ray-triangle leaf, two-level
// instancing) would live with their consumer, building BVH input over this same Prims layout.
//
// Input layout matches the GPU builder's input buffer exactly: a primitive AABB array
// of two `vec4<f32>` per prim — `[min.x, min.y, min.z, _, max.x, max.y, max.z, _]`.
// Reading the whole AABB per pass coalesces (gpu.md quantization ledger), and a
// fixture `Float32Array` uploads to the GPU with a single `writeBuffer`.

/** floats per primitive AABB (2 × vec4: min.xyz+pad, max.xyz+pad) */
export const PRIM_F32 = 8;

/** a scene of primitive AABBs in the GPU input layout */
export interface Prims {
    count: number;
    /** length `count * PRIM_F32`; prim i at offset `i * PRIM_F32`. Backed by a
     * plain ArrayBuffer so it uploads via `writeBuffer` without a copy. */
    data: Float32Array<ArrayBuffer>;
}

/** min corner of prim i */
export function primMin(p: Prims, i: number): [number, number, number] {
    const o = i * PRIM_F32;
    return [p.data[o], p.data[o + 1], p.data[o + 2]];
}

/** max corner of prim i */
export function primMax(p: Prims, i: number): [number, number, number] {
    const o = i * PRIM_F32;
    return [p.data[o + 4], p.data[o + 5], p.data[o + 6]];
}

/** write prim i's AABB — exported so a consumer's triangle scenes can build their BVH input */
export function writePrim(
    p: Prims,
    i: number,
    min: [number, number, number],
    max: [number, number, number],
): void {
    const o = i * PRIM_F32;
    p.data[o] = min[0];
    p.data[o + 1] = min[1];
    p.data[o + 2] = min[2];
    p.data[o + 3] = 0;
    p.data[o + 4] = max[0];
    p.data[o + 5] = max[1];
    p.data[o + 6] = max[2];
    p.data[o + 7] = 0;
}

/** allocate an empty `count`-prim scene — exported for derived scenes (triAabbs) */
export function alloc(count: number): Prims {
    return { count, data: new Float32Array(count * PRIM_F32) };
}

// a deterministic LCG so a fixture is reproducible across the oracle and the GPU harness
// for a given seed.
export function lcg(seed: number): () => number {
    let s = seed >>> 0 || 1;
    return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    };
}

function box(
    center: [number, number, number],
    half: number,
): { min: [number, number, number]; max: [number, number, number] } {
    return {
        min: [center[0] - half, center[1] - half, center[2] - half],
        max: [center[0] + half, center[1] + half, center[2] + half],
    };
}

/** boxes scattered uniformly through a cube — the unstructured baseline */
export function uniformRandom(count: number, seed: number): Prims {
    const rng = lcg(seed);
    const p = alloc(count);
    for (let i = 0; i < count; i++) {
        const c: [number, number, number] = [
            (rng() - 0.5) * 20,
            (rng() - 0.5) * 20,
            (rng() - 0.5) * 20,
        ];
        const b = box(c, 0.1 + rng() * 0.4);
        writePrim(p, i, b.min, b.max);
    }
    return p;
}

/** boxes pulled toward a few tight centers — stresses locality + deep merges */
export function clustered(count: number, seed: number): Prims {
    const rng = lcg(seed);
    const clusters = 4;
    const centers: [number, number, number][] = [];
    for (let k = 0; k < clusters; k++) {
        centers.push([(rng() - 0.5) * 30, (rng() - 0.5) * 30, (rng() - 0.5) * 30]);
    }
    const p = alloc(count);
    for (let i = 0; i < count; i++) {
        const c = centers[i % clusters];
        // sum of three uniforms ≈ a tight bell around the center
        const jitter = (): number => (rng() + rng() + rng() - 1.5) * 1.5;
        const b = box([c[0] + jitter(), c[1] + jitter(), c[2] + jitter()], 0.1 + rng() * 0.2);
        writePrim(p, i, b.min, b.max);
    }
    return p;
}

/** boxes on the z=0 plane — zero extent on one axis, the degenerate Morton case */
export function coplanar(count: number, seed: number): Prims {
    const rng = lcg(seed);
    const p = alloc(count);
    for (let i = 0; i < count; i++) {
        const half = 0.1 + rng() * 0.4;
        const cx = (rng() - 0.5) * 20;
        const cy = (rng() - 0.5) * 20;
        // flat on z: min.z == max.z, so the scene's z extent is exactly zero
        writePrim(p, i, [cx - half, cy - half, 0], [cx + half, cy + half, 0]);
    }
    return p;
}

/**
 * every prim sharing one centroid (distinct box sizes) — all map to a single Morton
 * cell, so the whole scene is one equal-code run. The pure degenerate-Morton case the
 * clustering + the cross-workgroup merge must still resolve into a valid binary tree.
 */
export function coincident(count: number, seed: number): Prims {
    const rng = lcg(seed);
    const p = alloc(count);
    const c: [number, number, number] = [1.5, -2.3, 0.7];
    for (let i = 0; i < count; i++) {
        const h = 0.02 + rng() * 0.3; // distinct extents → non-trivial unions, identical centroid
        writePrim(p, i, [c[0] - h, c[1] - h, c[2] - h], [c[0] + h, c[1] + h, c[2] + h]);
    }
    return p;
}

/**
 * many tight clumps scattered through a wide, thin volume — within a clump the prims
 * share a centroid (one Morton cell), across clumps the codes spread. The dense-clump
 * stress a scene of many small objects (or a forest of foliage clusters) puts on
 * locally-ordered clustering and the build's cross-workgroup merge: long equal-code runs
 * concentrated across many workgroups, which a uniform-random scene never produces. At
 * scale this is the input that exposes a forward-progress / coherence gap a single small
 * build can't. `degenerate` mixes in zero-extent prims (collapsed geometry).
 */
export function clumps(
    clusterCount: number,
    perClump: number,
    seed: number,
    degenerate = false,
): Prims {
    const rng = lcg(seed);
    const p = alloc(clusterCount * perClump);
    let i = 0;
    for (let k = 0; k < clusterCount; k++) {
        const cx = (rng() - 0.5) * 400; // wide in x/z, thin in y — many cells apart per clump
        const cy = (rng() - 0.5) * 40;
        const cz = (rng() - 0.5) * 400;
        for (let j = 0; j < perClump; j++) {
            // most prims sit on the clump centroid (identical Morton); a few jitter within
            // the cell. Distinct extents keep each clump's internal unions non-trivial.
            const e = j % 4 === 0 ? 0.3 : 0;
            const px = cx + (rng() - 0.5) * e;
            const py = cy + (rng() - 0.5) * e;
            const pz = cz + (rng() - 0.5) * e;
            if (degenerate && j % 17 === 0) {
                writePrim(p, i++, [px, py, pz], [px, py, pz]); // zero-extent point
            } else {
                const h = 0.02 + rng() * 0.15;
                writePrim(p, i++, [px - h, py - h, pz - h], [px + h, py + h, pz + h]);
            }
        }
    }
    return p;
}

/**
 * one prim spanning a huge world AABB plus `count-1` sub-voxel-sized prims near the
 * origin — the scale-disparity case (Embree's "a primitive spanning the whole scene
 * plus a sub-voxel one"). The giant prim sets the scene extent, so every tiny prim's
 * centroid normalizes into the same 10-bit Morton cell: a scale-driven equal-code run
 * the f32 world-space node format (gpu.md ledger) and the cross-workgroup merge must
 * still resolve into a valid tree without quantization collapsing the tiny prims away.
 */
export function giantAndTiny(count: number, seed: number): Prims {
    const rng = lcg(seed);
    const p = alloc(count);
    writePrim(p, 0, [-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]);
    for (let i = 1; i < count; i++) {
        const c: [number, number, number] = [
            (rng() - 0.5) * 0.01,
            (rng() - 0.5) * 0.01,
            (rng() - 0.5) * 0.01,
        ];
        const b = box(c, 1e-4 + rng() * 1e-4);
        writePrim(p, i, b.min, b.max);
    }
    return p;
}

/**
 * thin, heavily-overlapping slabs threaded through the volume — each prim is near-flat
 * on a random axis and long on the others, so their AABBs overlap densely. The
 * Hairball-class quality stressor: high total surface area, the input where a Morton
 * builder's SAH is worst (H-PLOC 2024 — the LBVH-vs-optimum gap widens most on highly
 * varying primitive sizes). The SAH quality gap a uniform scene hides surfaces here,
 * while the build must still hold its invariants + ray agreement on adversarial overlap.
 */
export function slivers(count: number, seed: number): Prims {
    const rng = lcg(seed);
    const p = alloc(count);
    for (let i = 0; i < count; i++) {
        const c: [number, number, number] = [
            (rng() - 0.5) * 20,
            (rng() - 0.5) * 20,
            (rng() - 0.5) * 20,
        ];
        const flat = i % 3; // the near-zero axis
        const long = 2 + rng() * 6;
        const half: [number, number, number] = [
            flat === 0 ? 0.01 : long,
            flat === 1 ? 0.01 : long,
            flat === 2 ? 0.01 : long,
        ];
        writePrim(
            p,
            i,
            [c[0] - half[0], c[1] - half[1], c[2] - half[2]],
            [c[0] + half[0], c[1] + half[1], c[2] + half[2]],
        );
    }
    return p;
}

/** one prim — the leaf-is-root degenerate (node count 2N−1 = 1) */
export function singlePrim(): Prims {
    const p = alloc(1);
    writePrim(p, 0, [-1, -1, -1], [1, 1, 1]);
    return p;
}

/** two prims — the smallest tree with an internal node (3 nodes) */
export function twoPrim(): Prims {
    const p = alloc(2);
    writePrim(p, 0, [-2, -2, -2], [-1, -1, -1]);
    writePrim(p, 1, [1, 1, 1], [2, 2, 2]);
    return p;
}

/** the named fixture set the oracle and the GPU harness both iterate (valid finite scenes) */
export function allFixtures(): { name: string; prims: Prims }[] {
    return [
        { name: "single-prim", prims: singlePrim() },
        { name: "two-prim", prims: twoPrim() },
        { name: "uniform-16", prims: uniformRandom(16, 0x1111) },
        { name: "uniform-256", prims: uniformRandom(256, 0x2222) },
        { name: "uniform-1024", prims: uniformRandom(1024, 0x3333) },
        { name: "clustered-256", prims: clustered(256, 0x4444) },
        { name: "clustered-1024", prims: clustered(1024, 0x5555) },
        { name: "coplanar-256", prims: coplanar(256, 0x6666) },
        // adversarial: coincident Morton codes + dense clumps + degenerate prims — the
        // class of input (many small/coincident objects) that stresses clustering + the
        // cross-workgroup merge where uniform scenes don't. Scale-out lives in the GPU
        // repeated-build gate (bvh.ts); these moderate sizes broaden every check.
        { name: "coincident-512", prims: coincident(512, 0x9a9a) },
        { name: "clumps-2048", prims: clumps(64, 32, 0xb0b0) },
        { name: "clumps-degenerate", prims: clumps(64, 32, 0xc1c1, true) },
        // scale-disparity + adversarial-overlap robustness, valid finite geometry so the
        // oracle gates them too: giant+tiny (quantization collapse) and slivers (SAH worst case)
        { name: "giant-and-tiny-256", prims: giantAndTiny(256, 0xd2d2) },
        { name: "slivers-512", prims: slivers(512, 0xe3e3) },
    ];
}
