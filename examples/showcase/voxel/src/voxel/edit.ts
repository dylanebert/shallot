// Cursor pick + density brush — the data-level voxel edit primitives (roadmap Phase 5 carving). Pure over the
// CPU grid (`Voxels.data`), so `bun test` gates them device-free: `march` is Amanatides–Woo grid traversal
// (the cursor ray → first solid cell + the entry face), `brush` blends a smooth falloff WEIGHT into the
// density field (add or carve), returning the chunk slots whose occupancy flipped (so only those re-mesh).
// The interaction wiring (input → march → brush → commit) lives in the voxel scenario; these stay pure so
// the DDA + brush are unit-testable.
//
// Reference: gpucraft `voxel_mod.js` for the CPU-authoritative edit → re-upload model; the march is the
// canonical Amanatides–Woo "A Fast Voxel Traversal Algorithm" with an AABB pre-clip (the camera sits outside
// the grid, so the ray enters through a face); the additive-weight brush is the Astroneer / Planet-Coaster
// terrain-deform model (modify the density field with a radial falloff, then re-polygonize).

import { chunkSlot, DENSITY, DIM, get, ISO, set, VOXEL } from "./grid";

/** a cursor-ray hit: the first solid cell the ray enters, the air cell across the entry face (where added
 *  weight lands), and the world distance to that face. */
export interface Hit {
    cell: [number, number, number];
    place: [number, number, number];
    distance: number;
}

const EPS = 1e-9;

/**
 * the first solid cell `(origin, dir)` enters, marched cell-by-cell through the grid (Amanatides–Woo), or
 * null if the ray clears the grid without hitting one within `maxDist`. `dir` must be normalized; `distance`
 * is world units along it. Works in grid space (cell size 1, `p = world/VOXEL + DIM/2`) and scales `t` back
 * to world by VOXEL.
 */
export function march(
    data: Float32Array,
    origin: readonly [number, number, number],
    dir: readonly [number, number, number],
    maxDist: number,
): Hit | null {
    const dim = [DIM.x, DIM.y, DIM.z];
    // grid-space ray: position offset to cell coords, direction scaled by 1/VOXEL so `t` stays world units.
    const p = [
        origin[0] / VOXEL + dim[0] / 2,
        origin[1] / VOXEL + dim[1] / 2,
        origin[2] / VOXEL + dim[2] / 2,
    ];
    const d = [dir[0] / VOXEL, dir[1] / VOXEL, dir[2] / VOXEL];

    // clip the ray to the grid AABB [0, dim] (slab test) → the entry t + which face it enters through. The
    // camera orbits outside the grid, so without this the start cell floors out of bounds and the march bails.
    let tEnter = 0;
    let tExit = maxDist;
    let entryAxis = -1;
    for (let a = 0; a < 3; a++) {
        if (Math.abs(d[a]) < EPS) {
            if (p[a] < 0 || p[a] > dim[a]) return null; // parallel to this slab and outside it
            continue;
        }
        let t0 = (0 - p[a]) / d[a];
        let t1 = (dim[a] - p[a]) / d[a];
        if (t0 > t1) [t0, t1] = [t1, t0];
        if (t0 > tEnter) {
            tEnter = t0;
            entryAxis = a;
        }
        if (t1 < tExit) tExit = t1;
        if (tEnter > tExit) return null;
    }

    // start cell at the entry point; clamp a boundary-grazing floor back in bounds.
    const start = [p[0] + d[0] * tEnter, p[1] + d[1] * tEnter, p[2] + d[2] * tEnter];
    const c = [
        Math.min(dim[0] - 1, Math.max(0, Math.floor(start[0]))),
        Math.min(dim[1] - 1, Math.max(0, Math.floor(start[1]))),
        Math.min(dim[2] - 1, Math.max(0, Math.floor(start[2]))),
    ];

    // entry-face normal (points back toward the origin) → seeds `place`. -1 = the camera began inside.
    const n = [0, 0, 0];
    if (entryAxis >= 0) n[entryAxis] = d[entryAxis] > 0 ? -1 : 1;

    const step = [Math.sign(d[0]), Math.sign(d[1]), Math.sign(d[2])];
    const tMax = [0, 0, 0];
    const tDelta = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
        if (step[a] === 0) {
            tMax[a] = Number.POSITIVE_INFINITY;
            tDelta[a] = Number.POSITIVE_INFINITY;
        } else {
            const boundary = step[a] > 0 ? c[a] + 1 : c[a];
            tMax[a] = tEnter + (boundary - start[a]) / d[a];
            tDelta[a] = Math.abs(1 / d[a]);
        }
    }

    let t = tEnter;
    const cap = dim[0] + dim[1] + dim[2] + 3; // every step advances one cell within bounds → bounded
    for (let i = 0; i < cap; i++) {
        if (get(data, c[0], c[1], c[2]) >= ISO) {
            return {
                cell: [c[0], c[1], c[2]],
                place: [c[0] + n[0], c[1] + n[1], c[2] + n[2]],
                distance: t,
            };
        }
        // advance the axis whose next cell boundary is nearest; the crossed face becomes the entry normal.
        let a = 0;
        if (tMax[1] < tMax[a]) a = 1;
        if (tMax[2] < tMax[a]) a = 2;
        t = tMax[a];
        if (t > maxDist) return null;
        c[a] += step[a];
        if (c[a] < 0 || c[a] >= dim[a]) return null; // left the grid
        tMax[a] += tDelta[a];
        n[0] = 0;
        n[1] = 0;
        n[2] = 0;
        n[a] = -step[a];
    }
    return null;
}

/**
 * blend a smooth radial weight into the density field over `radius` cells around `(cx, cy, cz)` — `delta > 0`
 * adds, `delta < 0` carves — clamped to [0, {@link DENSITY}] and to the grid. The falloff is a smoothstep
 * peaking at the centre, so repeated small dabs grow the isosurface outward across {@link ISO} continuously
 * (the add-weight sculpt, not a hard stamp). Returns the chunk slots whose OCCUPANCY (solid = density ≥ ISO)
 * flipped — the cells that change the mesh, so only those re-upload + re-mesh; sub-threshold weight just
 * accumulates on the CPU mirror.
 */
export function brush(
    data: Float32Array,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
    delta: number,
): Set<number> {
    const touched = new Set<number>();
    const r = Math.max(1, Math.round(radius));
    for (let z = Math.max(0, cz - r); z <= Math.min(DIM.z - 1, cz + r); z++) {
        const dz = z - cz;
        for (let y = Math.max(0, cy - r); y <= Math.min(DIM.y - 1, cy + r); y++) {
            const dy = y - cy;
            for (let x = Math.max(0, cx - r); x <= Math.min(DIM.x - 1, cx + r); x++) {
                const dx = x - cx;
                const tt = 1 - Math.sqrt(dx * dx + dy * dy + dz * dz) / radius;
                if (tt <= 0) continue; // outside the brush sphere
                const old = get(data, x, y, z);
                const next = Math.min(DENSITY, Math.max(0, old + delta * tt * tt * (3 - 2 * tt)));
                if (next === old) continue;
                set(data, x, y, z, next);
                if (old >= ISO !== next >= ISO) touched.add(chunkSlot(x, y, z));
            }
        }
    }
    return touched;
}
