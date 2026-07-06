// CPU reference for the GPU incremental-greedy body coloring (step.ts COLORING_PASS_WGSL) — the
// Phase-4 coloring crux's executable spec (scratch.md "AVBD rebuild" → Distributed cruxes → Coloring).
// One "thread" per body, no atomics, reading a stable prior-frame snapshot: deterministic integer
// logic; the GPU reproduces it (gym `pile` coloring-conflict counter, the real-GPU home).
// Mirrors webphysics greedyBodyColorsShader: the higher-id symmetry break, the 32-wide usedMask, the
// keep-prior-color reuse, and the fold past the dispatched-color cap.

/** an undirected contact edge — the (a, b) body indices the GPU collide tags into each contact row */
export type Edge = [number, number];

/** the GPU's uncolored sentinel for a static body (`colors[bid] = 0xffffffff`) */
const STATIC = 0xffffffff;

/**
 * one incremental-greedy coloring sweep — the CPU twin of `colorize`, returning each body's color
 * (STATIC for a static body, which the primal skips). `scratch[i]` is body i's prior-frame color (the
 * GPU seeds it by copying `colors` before the pass). Each dynamic body avoids the prior colors of its
 * *higher-id dynamic neighbors* (the no-atomics symmetry break: an edge is resolved by its lower-id
 * endpoint), keeps its own prior color when still free, else takes the lowest free color, else folds to
 * `i % maxColors` past the cap (a tolerated same-color conflict).
 */
export function colorSweep(
    edges: Edge[],
    mass: number[],
    scratch: number[],
    maxColors: number,
): number[] {
    const n = mass.length;
    const colorsN = Math.max(1, Math.min(maxColors, 32));
    // each lower-id endpoint avoids its higher-id neighbor's prior color (the GPU scans every contact;
    // the resulting bit set is identical, and duplicate contacts on one pair OR the same bit twice)
    const higher: number[][] = Array.from({ length: n }, () => []);
    for (const [a, b] of edges) {
        if (a === b) continue;
        const lo = Math.min(a, b);
        const hi = Math.max(a, b);
        higher[lo].push(hi);
    }
    const colors = new Array<number>(n).fill(STATIC);
    for (let i = 0; i < n; i++) {
        if (mass[i] <= 0) continue; // static — uncolored
        let usedMask = 0;
        for (const other of higher[i]) {
            if (mass[other] <= 0) continue; // static neighbor: no scheduling constraint
            const pc = scratch[other];
            if (pc < 32) usedMask |= 1 << pc;
        }
        let chosen = scratch[i];
        let needsNew = chosen >= colorsN;
        if (!needsNew) needsNew = (usedMask & (1 << chosen)) !== 0;
        if (needsNew) {
            let found = false;
            for (let c = 0; c < colorsN; c++) {
                if ((usedMask & (1 << c)) === 0) {
                    chosen = c;
                    found = true;
                    break;
                }
            }
            if (!found) chosen = i % colorsN;
        }
        colors[i] = chosen >>> 0;
    }
    return colors;
}

/**
 * CPU twin of the GPU joint hard-conflict repair (step.ts REPAIR_PASS_WGSL, Phase 6.2) — the deterministic
 * spec for "a hard (dynamic-dynamic joint) pair must not end same-color." The greedy avoids ALL constraint
 * neighbors but tolerates a folded same-color pair; a soft spring survives that, a hard joint doesn't. Each
 * round snapshots the colors, then the lower-eid endpoint of any same-color joint pair recolors to the
 * lowest color free of all its constraint neighbors (else folds past the cap). Reading the stable snapshot
 * is what makes it race-free + deterministic. `soft` = spring/contact edges (avoidance only); `hard` = joint
 * edges (the recolor trigger). Returns the repaired colors (the greedy result, mutated).
 */
export function repairHardColors(
    soft: Edge[],
    hard: Edge[],
    colors: number[],
    mass: number[],
    maxColors: number,
    rounds = 2,
): number[] {
    const n = mass.length;
    const colorsN = Math.max(1, Math.min(maxColors, 32));
    const all: number[][] = Array.from({ length: n }, () => []); // avoidance neighbors (soft + hard)
    const hardAdj: number[][] = Array.from({ length: n }, () => []); // recolor-trigger neighbors (hard only)
    const link = (adj: number[][], a: number, b: number): void => {
        if (a !== b) {
            adj[a].push(b);
            adj[b].push(a);
        }
    };
    for (const [a, b] of soft) link(all, a, b);
    for (const [a, b] of hard) {
        link(all, a, b);
        link(hardAdj, a, b);
    }
    const out = colors.slice();
    for (let r = 0; r < rounds; r++) {
        const snap = out.slice();
        for (let i = 0; i < n; i++) {
            if (mass[i] <= 0) continue; // static — uncolored, never a mover
            const my = snap[i];
            // the lower-eid endpoint of a same-color joint pair is the one that recolors (higher-id stays)
            let hardConflict = false;
            for (const o of hardAdj[i])
                if (mass[o] > 0 && o > i && snap[o] === my) hardConflict = true;
            if (!hardConflict) continue;
            let usedMask = 0;
            for (const o of all[i]) if (mass[o] > 0 && snap[o] < 32) usedMask |= 1 << snap[o];
            let chosen = my;
            let found = false;
            for (let c = 0; c < colorsN; c++) {
                if ((usedMask & (1 << c)) === 0) {
                    chosen = c;
                    found = true;
                    break;
                }
            }
            if (!found) chosen = i % colorsN;
            out[i] = chosen >>> 0;
        }
    }
    return out;
}

/** same-color edges between two dynamic bodies, counted once per unique pair — the conflict invariant */
export function countConflicts(edges: Edge[], colors: number[], mass: number[]): number {
    const seen = new Set<number>();
    const n = mass.length;
    let conflicts = 0;
    for (const [a, b] of edges) {
        if (a === b || mass[a] <= 0 || mass[b] <= 0) continue;
        const key = a < b ? a * n + b : b * n + a;
        if (seen.has(key)) continue;
        seen.add(key);
        if (colors[a] === colors[b]) conflicts++;
    }
    return conflicts;
}
