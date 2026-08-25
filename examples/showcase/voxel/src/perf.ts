import type { Mirror } from "@dylanebert/shallot";
import { settle } from "./harness";
import { brush } from "./voxel/edit";
import { TOTAL_CELLS } from "./voxel/grid";
import { commitEdit, EmitTelemetry, uploadVoxels } from "./voxel/mesher";
import { single } from "./voxel/patterns";

// The structural perf probe (`showcase-frame-floor` S2): drives one deterministic, single-chunk carve and
// reports the emit dispatch scope it triggers — `test/perf.spec.ts` gates the reported workgroup count
// against touched-chunk volume, never against wall-clock (Locked decision: gate form (b), not (a)). This
// module only produces the sample; `window.__voxelPerf` (boot.ts) is the only thing Playwright drives.

// deep inside chunk (0,0,0) — CHUNK=32, so a radius-2 brush stays 14+ cells from every chunk seam and
// exactly one chunk's occupancy flips.
const PROBE_X = 16;
const PROBE_Y = 16;
const PROBE_Z = 16;
const PROBE_RADIUS = 2;

export interface EmitDispatchSample {
    /** chunk slots whose occupancy flipped this edit — the touched-chunk volume the structural gate
     *  reasons about. */
    touchedChunks: number;
    /** the workgroup count {@link EmitTelemetry} recorded for the fire this edit triggered. */
    workgroups: number;
}

/** author a lone solid voxel, carve it away, and read the dispatch scope the resulting re-mesh fires — a
 *  one-chunk edit, structurally the smallest non-trivial touch. Two dirty fires land (the author upload,
 *  then the carve); `cursor` (a {@link Mirror} of `Voxels.cursor`) settles after each so the read trails
 *  the fire it measures, never a stale one. */
export async function emitDispatchSample(cursor: Mirror): Promise<EmitDispatchSample> {
    const grid = new Float32Array(TOTAL_CELLS);
    single(grid, PROBE_X, PROBE_Y, PROBE_Z);
    uploadVoxels(grid);
    await settle(cursor);

    const touched = brush(grid, PROBE_X, PROBE_Y, PROBE_Z, PROBE_RADIUS, -1);
    commitEdit(touched);
    await settle(cursor);

    return { touchedChunks: touched.size, workgroups: EmitTelemetry.lastWorkgroups };
}
