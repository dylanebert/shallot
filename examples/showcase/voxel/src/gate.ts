import type { Mirror } from "@dylanebert/shallot";
import { type Check, settle } from "./harness";
import { brush } from "./voxel/edit";
import { generate, solidFractionBand } from "./voxel/generate";
import { DENSITY, DIM, faces, ISO, solid, TOTAL_CELLS } from "./voxel/grid";
import { commitEdit, readGrid, uploadVoxels } from "./voxel/mesher";
import { RELIEF } from "./voxel/noise";
import { checker, recenter, single, slab, solidChunk, sphere, tunnel } from "./voxel/patterns";

// The voxel mesher's correctness gate — the watertight-seam invariant, generation properties, and the carve
// round-trip, run on the real device. It's the showcase dogfooding its own testing: published-`@dylanebert/
// shallot` surface + this project's own lib + driver, no reach into any repo harness. `main.ts` exposes it on
// `window.__voxelGate`; the project's own Playwright (`test/voxel.spec.ts`) drives it on a GPU.
//
// The gate has two halves. (1) Mesher correctness: for each canonical pattern the GPU's atomic face count
// must equal the analytic `faces()` oracle (exact equality = watertight, a doubled seam over-counts, a gap
// under-counts). (2) Generation: the generated grid is deterministic in its seed (two runs read back
// bit-identical), its solid fraction lands in the derived band, its surface relief is real (not a flat
// plane), the mesher stays watertight over it, and a carve → grid-write → remesh stays watertight too.

// the six canonical patterns — each a distinct mesher code path (isolated voxel, occluded interior,
// every-face-exposed, flat plane, inward faces, cross-chunk curve). `faces()` is the analytic expected count.
const PATTERNS: { name: string; author: (data: Float32Array) => void }[] = [
    { name: "single", author: (d) => single(d, 40, 50, 60) },
    { name: "solid", author: (d) => solidChunk(d, 1, 1, 1) },
    { name: "checker", author: (d) => checker(d, 0, 0, 0, 16, 16, 16) },
    { name: "slab", author: (d) => slab(d, 3) },
    { name: "tunnel", author: (d) => tunnel(d, 10, 10, 10, 16) },
    { name: "sphere", author: (d) => sphere(d, 128, 128, 128, 40) },
];

// a fixed seed for the determinism gate, independent of the live terrain seed.
const GATE_SEED = 0x0072_7374;

function authorGrid(name: string): Float32Array {
    const data = new Float32Array(TOTAL_CELLS);
    const pat = PATTERNS.find((p) => p.name === name) ?? PATTERNS[PATTERNS.length - 1];
    pat.author(data);
    return recenter(data);
}

function countSolid(grid: Float32Array): number {
    let n = 0;
    for (let i = 0; i < grid.length; i++) if (grid[i] >= ISO) n++;
    return n;
}

// the standard deviation of the per-column surface height (the highest solid cell in each x,z column). A
// flat generator yields one constant height (std 0); the layered-perlin heightmap yields rolling relief.
// Floor 0.05·RELIEF: a single 2D perlin octave realizes std ≈ 0.2, fbm-normalization (÷ the ~1.9 amplitude
// sum) leaves std(fbm2) ≳ 0.1, so the surface std is ≳ 0.1·RELIEF; halving it for seed-to-seed variation
// gives a not-flat floor with ~2× margin, not a tuned threshold.
function reliefStd(grid: Float32Array): number {
    let sum = 0;
    let sumSq = 0;
    let n = 0;
    for (let z = 0; z < DIM.z; z++) {
        for (let x = 0; x < DIM.x; x++) {
            let top = 0;
            for (let y = DIM.y - 1; y >= 0; y--) {
                if (solid(grid, x, y, z)) {
                    top = y;
                    break;
                }
            }
            sum += top;
            sumSq += top * top;
            n++;
        }
    }
    const mean = sum / n;
    return Math.sqrt(Math.max(0, sumSq / n - mean * mean));
}

function equalGrid(a: Float32Array, b: Float32Array): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

// the meshed face count, read back from the indirect record's first word (the index count = faces × 6).
function liveFaces(indirect: Mirror): number | null {
    if (!indirect.snapshot) return null;
    return new Uint32Array(indirect.snapshot.bytes)[0] / 6;
}

/** run the full mesher gate against the live device, restoring the generated terrain when done. `indirect`
 *  is a {@link Mirror} of the voxel draw record (its first word = the index count). */
export async function gate(indirect: Mirror): Promise<Check[]> {
    const checks: Check[] = [];

    // (1) mesher correctness: swap in each canonical grid, re-mesh, assert the GPU's atomic face count
    // equals the analytic oracle. Exact equality across the sphere (spanning chunks) is the watertight-seam
    // check — no doubled border face, no gap.
    for (const pat of PATTERNS) {
        const grid = authorGrid(pat.name);
        uploadVoxels(grid);
        await settle(indirect);
        const got = liveFaces(indirect);
        const want = faces(grid);
        const seam = pat.name === "sphere" ? " — watertight across chunk seams" : "";
        checks.push({
            name: pat.name,
            pass: got === want,
            detail: `${got} faces (expected ${want})${seam}`,
        });
    }

    // (2) generation: generate twice at one seed and read both grids back. The first drives the
    // density/relief/watertight checks; the pair drives determinism.
    await generate(GATE_SEED);
    await settle(indirect);
    const grid = await readGrid();
    const gpuFaces = liveFaces(indirect);
    await generate(GATE_SEED);
    await settle(indirect);
    const grid2 = await readGrid();

    checks.push({
        name: "deterministic",
        pass: equalGrid(grid, grid2),
        detail: "seed → bit-identical grid across two runs",
    });

    const frac = countSolid(grid) / TOTAL_CELLS;
    const [lo, hi] = solidFractionBand();
    checks.push({
        name: "density",
        pass: frac >= lo && frac <= hi,
        detail: `solid ${frac.toFixed(3)} ∈ [${lo.toFixed(3)}, ${hi.toFixed(3)}] (heightmap band)`,
    });

    const std = reliefStd(grid);
    const reliefFloor = RELIEF * 0.05;
    checks.push({
        name: "relief",
        pass: std > reliefFloor,
        detail: `surface-height std ${std.toFixed(1)} > ${reliefFloor.toFixed(1)} (rolling, not flat)`,
    });

    checks.push({
        name: "generated-watertight",
        pass: gpuFaces === faces(grid),
        detail: `${gpuFaces} faces (expected ${faces(grid)})`,
    });

    // (3) carve: blend a large negative weight into the solid sphere's core (a falloff dab carves a hollow),
    // commit the touched chunks, re-mesh, and assert the GPU's atomic face count still equals the analytic
    // oracle on the edited density grid — the carve → grid-write → remesh → watertight gate (opening interior
    // faces is the inverse of the tunnel pattern). The interactive pick is march-unit-tested + hand-driven;
    // this gates the GPU edit→remesh on the real device.
    const edited = authorGrid("sphere");
    uploadVoxels(edited);
    const touched = brush(edited, DIM.x / 2, DIM.y / 2, DIM.z / 2, 12, -DENSITY);
    commitEdit(touched);
    await settle(indirect);
    checks.push({
        name: "carve-watertight",
        pass: liveFaces(indirect) === faces(edited),
        detail: `${liveFaces(indirect)} faces (expected ${faces(edited)}) after carving the core`,
    });

    // restore the generated terrain for the live view.
    await generate(GATE_SEED);
    await settle(indirect);
    return checks;
}
