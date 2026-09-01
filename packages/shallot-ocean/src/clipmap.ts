// The render mesh: a doubling clipmap, not a uniform grid. A uniform grid dense enough to resolve
// the finest displacement cascade at its full extent is far more triangles than the field needs
// away from the camera — a doubling clipmap keeps every ring at roughly the same vertex count
// regardless of its world size, coarsening spacing by exactly 2x per ring outward from a solid core.
//
// Every vertex uses the engine's `posU + normalV` authoring layout (position in the first 3 floats,
// VERTEX_FLOATS=8), so a consuming vertex/fragment shader only ever reads a vertex's world position,
// never the mesh topology.
//
// Pure module — no GPU import — so both the mesh builder and its own continuity checks read the
// same numbers from one source.

/** the two numbers that set an FFT cascade's texel size: N (grid resolution) and L (domain, meters). */
export interface CascadeTexel {
    N: number;
    L: number;
}

/** the smallest cascade texel size (m) — the field's finest resolvable feature. */
export function finestCascadeTexel(cascades: readonly CascadeTexel[]): number {
    let min = Number.POSITIVE_INFINITY;
    for (const c of cascades) {
        const texel = c.L / c.N;
        if (texel < min) min = texel;
    }
    return min;
}

/** one ring (or the solid core, when `rInner` is 0) of the clipmap: a square donut of quads at a
 *  fixed spacing between `rInner` and `rOuter` (world-space half-extents, meters). */
export interface ClipLevel {
    rInner: number;
    rOuter: number;
    spacing: number;
}

/**
 * Derive doubling clipmap levels: a solid core `[0, coreHalfExtent]` at `nearSpacing`, then donut
 * rings doubling both radius and spacing until `totalHalfExtent` is reached. `totalHalfExtent` must
 * be an exact power-of-two multiple of `coreHalfExtent` (checked) so every ring tiles exactly with
 * no gap or overlap, and `coreHalfExtent / nearSpacing` must be integral (checked) so the core grid
 * lands on integer steps — both violations throw rather than silently misbuild the mesh.
 */
export function buildClipLevels(opts: {
    coreHalfExtent: number;
    nearSpacing: number;
    totalHalfExtent: number;
}): ClipLevel[] {
    const { coreHalfExtent, nearSpacing, totalHalfExtent } = opts;
    const ratio = totalHalfExtent / coreHalfExtent;
    const doublings = Math.log2(ratio);
    if (!Number.isInteger(doublings)) {
        throw new Error(
            `buildClipLevels: totalHalfExtent/coreHalfExtent (${ratio}) is not a power of 2 — every ring must double the previous exactly`,
        );
    }
    if (!Number.isInteger(coreHalfExtent / nearSpacing)) {
        throw new Error(
            `buildClipLevels: coreHalfExtent/nearSpacing (${coreHalfExtent / nearSpacing}) is not integral — the core grid wouldn't tile exactly`,
        );
    }
    const levels: ClipLevel[] = [{ rInner: 0, rOuter: coreHalfExtent, spacing: nearSpacing }];
    let r = coreHalfExtent;
    let spacing = nearSpacing;
    for (let i = 0; i < doublings; i++) {
        const rNext = r * 2;
        levels.push({ rInner: r, rOuter: rNext, spacing: spacing * 2 });
        r = rNext;
        spacing *= 2;
    }
    return levels;
}

/** a uniform grid of `resolution` steps over `size` expressed as a one-level "clipmap" — lets the
 *  continuity checks run against a non-clipmap baseline for comparison. Spacing matches a plain
 *  `S / (G - 1)` grid step exactly. */
export function uniformGridLevel(resolution: number, size: number): ClipLevel {
    return { rInner: 0, rOuter: size / 2, spacing: size / (resolution - 1) };
}

/** the ocean clipmap configuration — near spacing is below half of cascade 1's texel size (the
 *  Nyquist-sampling bound for a mesh that must resolve every displacement wavelength the field
 *  carries). A 22.8 m core keeps the 0.12 m grid integral (190 steps), and its 364.8 m half-extent
 *  covers the represented band. */
export const OCEAN_CLIP_CONFIG = {
    coreHalfExtent: 22.8,
    nearSpacing: 0.12,
    totalHalfExtent: 364.8,
} as const;

export const OCEAN_CLIP_LEVELS: ClipLevel[] = buildClipLevels(OCEAN_CLIP_CONFIG);

// ── mesh geometry ─────────────────────────────────────────────────────────────

const VERTEX_FLOATS = 8; // px py pz u | nx ny nz v (posU + normalV authoring layout)

/** append one axis-aligned rectangular grid of quads (world-space x0..x1 by z0..z1, at `step`) to a
 *  shared vertex/index accumulator. `(x1-x0)/step` and `(z1-z0)/step` must be integral — the caller
 *  (a clipmap level's core square or one of its 4 ring strips) is built so they always are. */
function appendRectGrid(
    verts: number[],
    indices: number[],
    x0: number,
    x1: number,
    z0: number,
    z1: number,
    step: number,
): void {
    const nx = Math.round((x1 - x0) / step); // quads across x
    const nz = Math.round((z1 - z0) / step); // quads across z
    if (nx <= 0 || nz <= 0) return;
    const base = verts.length / VERTEX_FLOATS;
    for (let j = 0; j <= nz; j++) {
        const z = z0 + j * step;
        for (let i = 0; i <= nx; i++) {
            const x = x0 + i * step;
            verts.push(x, 0, z, i / nx, 0, 1, 0, j / nz);
        }
    }
    const stride = nx + 1;
    for (let j = 0; j < nz; j++) {
        for (let i = 0; i < nx; i++) {
            const v0 = base + j * stride + i;
            const v1 = base + j * stride + i + 1;
            const v2 = base + (j + 1) * stride + i;
            const v3 = base + (j + 1) * stride + i + 1;
            indices.push(v0, v2, v1, v1, v2, v3);
        }
    }
}

/** the core square, or (for `level.rInner > 0`) the 4 strips (top, bottom, left, right) that tile a
 *  square donut between `rInner` and `rOuter` exactly, with no shared-vertex dedup at the seams —
 *  fine for a mesh with no cross-seam skinning requirement. */
function appendClipLevel(verts: number[], indices: number[], level: ClipLevel): void {
    const { rInner, rOuter, spacing } = level;
    if (rInner === 0) {
        appendRectGrid(verts, indices, -rOuter, rOuter, -rOuter, rOuter, spacing);
        return;
    }
    appendRectGrid(verts, indices, -rOuter, rOuter, rInner, rOuter, spacing); // top
    appendRectGrid(verts, indices, -rOuter, rOuter, -rOuter, -rInner, spacing); // bottom
    appendRectGrid(verts, indices, -rOuter, -rInner, -rInner, rInner, spacing); // left
    appendRectGrid(verts, indices, rInner, rOuter, -rInner, rInner, spacing); // right
}

export function buildClipmapMesh(levels: readonly ClipLevel[]): {
    vertices: Float32Array;
    indices: Uint32Array;
} {
    const verts: number[] = [];
    const indices: number[] = [];
    for (const level of levels) appendClipLevel(verts, indices, level);
    return { vertices: new Float32Array(verts), indices: new Uint32Array(indices) };
}

// ── continuity check (pure arithmetic) ────────────────────────────────────────

export interface ContinuityFinding {
    ok: boolean;
    message: string;
}

/**
 * Mesh-vs-field sampling adequacy, asserted directly over grid spacing and cascade texel size —
 * no GPU, no rendered frame. Three checks:
 *   1. near-field (innermost level) spacing must resolve the finest cascade.
 *   2. each successive ring may only coarsen by at most 2x from its inner neighbor, and must abut it
 *      exactly (no gap, no overlap) — "coarsening with distance" bounded rather than unbounded.
 *   3. the outermost ring must still reach the represented band's required half-extent, so a fix
 *      can't silently satisfy (1) by shrinking the band instead of building a clipmap.
 */
export function checkContinuity(
    levels: readonly ClipLevel[],
    cascades: readonly CascadeTexel[],
    minBandHalfExtent: number,
): ContinuityFinding[] {
    const findings: ContinuityFinding[] = [];
    const finest = finestCascadeTexel(cascades);
    const nearBound = finest / 2;

    const near = levels[0];
    findings.push({
        ok: near.rInner === 0 && near.spacing <= nearBound,
        message: `near-field spacing ${near.spacing.toFixed(4)} m must be ≤ half the finest cascade texel (${finest.toFixed(4)} m / 2 = ${nearBound.toFixed(4)} m) — ratio ${(near.spacing / nearBound).toFixed(2)}x`,
    });

    for (let i = 1; i < levels.length; i++) {
        const prev = levels[i - 1];
        const cur = levels[i];
        findings.push({
            ok: cur.spacing >= prev.spacing && cur.spacing <= prev.spacing * 2,
            message: `ring ${i} spacing ${cur.spacing} m must coarsen monotonically from ring ${i - 1} (${prev.spacing} m) by at most 2x`,
        });
        findings.push({
            ok: cur.rInner === prev.rOuter,
            message: `ring ${i} must start exactly where ring ${i - 1} ends (${cur.rInner} m vs ${prev.rOuter} m) — no gap or overlap`,
        });
    }

    const outer = levels[levels.length - 1];
    findings.push({
        ok: outer.rOuter >= minBandHalfExtent,
        message: `represented band half-extent ${outer.rOuter} m must reach ≥ ${minBandHalfExtent} m`,
    });

    return findings;
}
