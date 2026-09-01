// The cell grid's compute-pass contract: the bind group layout + kernel a fill pass writes cells
// through, and the headless producer (`createCellGrid` / `fillCellGrid`) S1 ships to prove the shape —
// allocate, dispatch, read back through `standard/mirror`'s existing buffer-level readback, no new
// readback machinery of its own. S3's web sink and S4's terminal encoder both consume the same
// `CellGrid.buffer`; only the kernel body that decides *what* a cell holds is expected to change once a
// real scene exists to sample — this fill pass writes a deterministic test pattern standing in for that.

import tgpu, { type StorageFlag, type TgpuBuffer, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import { Compute } from "../../engine";
import { Cell, packCell } from "./cell";

const WG = 8; // 2D workgroup — matches the grid's own 2D shape rather than a flattened 1D index

/** the fill pass's dims: grid width/height + the glyph ramp length the test pattern wraps against.
 *  @internal */
export const GridParams = d.struct({ cols: d.u32, rows: d.u32, glyphCount: d.u32 });

/** the fill pass's one bind group: the dims uniform in, the cell buffer out.
 *  @internal */
export const gridLayout = tgpu.bindGroupLayout({
    params: { uniform: GridParams },
    cells: { storage: (n: number) => d.arrayOf(Cell, n), access: "mutable" },
});

// A diagonal glyph-ramp sweep with a fg/bg gradient over cell coordinates — deterministic, no scene
// input, standing in for a real render-target sample until a sink supplies one (S3). The schema, bind
// group, and dispatch shape are what a real producer reuses; only this body's content source is expected
// to change.
const fillKernel = tgpu.computeFn({
    workgroupSize: [WG, WG],
    in: { gid: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const x = input.gid.x;
    const y = input.gid.y;
    const cols = gridLayout.$.params.cols;
    const rows = gridLayout.$.params.rows;
    if (x >= cols || y >= rows) return;
    const i = y * cols + x;
    const glyph = (x + y) % gridLayout.$.params.glyphCount;
    const one = d.f32(1);
    const u = d.f32(x) / d.f32(cols);
    const v = d.f32(y) / d.f32(rows);
    // fg/bg alpha (the linear lane packLdrColor passes straight into packUnorm4x8, no sRGB transfer)
    // sweeps column/row across the unorm8 lattice's own midpoints — packed inverted (`one - e`, below),
    // `(x + 0.5) / 255` and `(y + 0.5) / 255` are the *mathematically* exact tie `k + 0.5` for
    // k = 254 - x (resp. 254 - y) — `cells.ts`'s own local reference states the same k. Whether the
    // emitted WGSL division actually lands each sample's f32 value exactly on that tie is device
    // arithmetic the `cells` gym scenario measures rather than this comment asserting — WGSL division
    // carries a spec'd ULP tolerance, not a correctly-rounded guarantee, so some samples construct the
    // seam by luck rather than by guarantee (the fixture's own module doc has the measured derivation).
    // Packed inverted (`one - e`) so the visible byte lands near-opaque (~245..254 of 255) rather than
    // the near-transparent 1..10 the un-inverted small numerator alone would pack to — a consumer sees a
    // near-solid grid, not a blank one. The inversion is where the tie becomes *predictable*, not where
    // it stays exact: `e` (~0.002..0.998) carries mantissa bits down to roughly exponent -9 (ulp ~2^-32),
    // but `one - e` lands at exponent -1 (ulp ~2^-24) and so discards `e`'s eight lowest mantissa bits —
    // exactly the width WGSL's permitted division-ULP slack lives in. The subtraction re-rounds `e` at
    // that coarser ulp and so *absorbs* the division's slack rather than adding to it; that re-rounding is
    // the only new rounding the inversion introduces, and it's the reason the packed tie is predictable
    // across the division's ULP tolerance rather than merely mathematically exact on paper. `cells.ts`'s
    // own local half-to-even reference predicts the byte this produces on a real device, tie or not.
    const alphaFg = one - (d.f32(x) + d.f32(0.5)) / d.f32(255);
    const alphaBg = one - (d.f32(y) + d.f32(0.5)) / d.f32(255);
    const packed = packCell(
        glyph,
        d.vec4f(u, v, one - u, alphaFg),
        d.vec4f(one - u, one - v, v, alphaBg),
    );
    gridLayout.$.cells[i].glyph = packed.x;
    gridLayout.$.cells[i].fg = packed.y;
    gridLayout.$.cells[i].bg = packed.z;
});

let _pipeline: TgpuComputePipeline | null = null;

function pipeline(): TgpuComputePipeline {
    if (_pipeline) return _pipeline;
    _pipeline = Compute.root.createComputePipeline({ compute: fillKernel }).$name("cells-fill");
    return _pipeline;
}

/** drop the memoized fill pipeline. Pipelines bind to the root that created them, so a re-adopted device
 *  needs a fresh one — mirrors `extras/text/sdf.ts`'s `resetPipelines`.
 *  @internal */
export function resetPipeline(): void {
    _pipeline = null;
}

/**
 * a headless cell grid: the GPU-owned buffer plus the dims a fill pass needs, sized `cols * rows *`
 * {@link CELL_BYTES}. Sibling of `extras/text`'s glyph buffer — `buffer` is a plain `TgpuBuffer`, so a
 * caller reads it back through `mirror(grid.buffer)` (`standard/mirror`), with no cells-owned readback
 * path.
 */
export interface CellGrid {
    readonly cols: number;
    readonly rows: number;
    readonly glyphCount: number;
    readonly buffer: TgpuBuffer<d.WgslArray<typeof Cell>> & StorageFlag;
}

/**
 * allocate a headless cell grid of `cols * rows` cells against the adopted device (`Compute.root`).
 * Empty until {@link fillCellGrid} dispatches the compute pass. `glyphCount` must be at least 1 — the
 * fill kernel wraps the test-pattern glyph index against it.
 *
 * @example const grid = createCellGrid(80, 24, CELL_GLYPH_COUNT); // the printable-ASCII ramp, ramp.ts
 */
export function createCellGrid(cols: number, rows: number, glyphCount: number): CellGrid {
    if (glyphCount < 1)
        throw new Error(`[cells] createCellGrid: glyphCount must be >= 1, got ${glyphCount}`);
    const buffer = Compute.root
        .createBuffer(d.arrayOf(Cell, cols * rows))
        .$usage("storage")
        .$name("cells-grid");
    return { cols, rows, glyphCount, buffer };
}

/**
 * dispatch the fill compute pass over `grid`, writing every cell in place — the headless producer this
 * stage ships. Encodes, submits, and returns; read the result back with `mirror(grid.buffer)`.
 *
 * @example
 * const grid = createCellGrid(80, 24, CELL_GLYPH_COUNT);
 * fillCellGrid(grid);
 * const m = mirror(grid.buffer);
 * // MirrorSystem (or a manual Mirror.flush) populates m.snapshot on a later frame
 */
export function fillCellGrid(grid: CellGrid): void {
    const device = Compute.device;
    const params = Compute.root
        .createBuffer(GridParams, { cols: grid.cols, rows: grid.rows, glyphCount: grid.glyphCount })
        .$usage("uniform");
    const group = Compute.root.createBindGroup(gridLayout, { params, cells: grid.buffer });
    const encoder = device.createCommandEncoder({ label: "cells-fill" });
    const pass = encoder.beginComputePass({
        label: "cells-fill",
        timestampWrites: Compute.span?.("cells:fill"),
    });
    pipeline()
        .with(group)
        .with(pass)
        .dispatchWorkgroups(Math.ceil(grid.cols / WG), Math.ceil(grid.rows / WG));
    pass.end();
    device.queue.submit([encoder.finish()]);
    // destroyed once the submit is in flight — mirrors extras/text/sdf.ts's SDFGenerator.flush temp
    // buffers, which the same reasoning covers: the driver keeps a destroyed buffer alive for work
    // already submitted against it.
    params.destroy();
}

/** the emitted fill-pass WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function gridWgsl(): string {
    return tgpu.resolve([fillKernel], { names: "strict" });
}
