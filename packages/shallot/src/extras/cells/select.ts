// The real content producer `grid.ts`'s module doc names as the thing S3 ships: two compute passes that
// fill a cell grid from a rendered scene, structure-first (Locked decision, `specs/shallot-tui.md`'s
// glyph-selection addendum) rather than `grid.ts`'s deterministic test pattern, which stays untouched —
// the shape-proving headless producer the `cells` gym scenario and its own tests pin, not the content
// path a real scene drives. `CellsPlugin` (`./index.ts`) dispatches both passes each frame, in order.
//
// Pass 1 (`avgKernel`) block-samples `source` (the camera's offscreen scene color) into one tonemapped,
// Reinhard-compressed linear color per cell — a fixed 3×3 sub-sample grid per cell regardless of the
// cell's actual pixel footprint, so cost is `9 * cols * rows` texel reads independent of render
// resolution, not `cols * rows * cellPixels`. This block average is also this design's low-pass filter:
// rather than a true Difference-of-Gaussians (two device-side blur passes at different radii, which a
// cell-grid producer has no cheap way to run at cell granularity), pass 2's Sobel runs directly on
// *neighboring cells'* own block averages — each already a spatial low-pass at the cell's own scale, so
// the two-pass split plays DoG's role at effectively one octave rather than two. Simpler, one dispatch
// cheaper, and correct for what this unit needs: an edge that survives averaging to cell granularity is
// exactly the edge worth drawing as a directional glyph at that granularity; finer structure the average
// erases wouldn't survive being drawn as one glyph per cell either.
//
// Pass 2 (`selectKernel`) reads pass 1's per-cell averages (a 3×3 neighborhood, clamped to the grid edge)
// and picks a glyph: Sobel magnitude over the neighborhood's luma decides edge-vs-fill, gated by
// {@link EDGE_MAGNITUDE_THRESHOLD}; over threshold picks a directional glyph from the gradient angle
// (converted to its perpendicular tangent bucket, `ramp.ts`'s locked contract); under it picks a
// coverage-ordered fill glyph from the cell's own luma. `fg`/`bg` are a fixed high-contrast pair (white
// glyph ink, the cell's own average as background) — deliberately not itself part of the structure-first
// question the Locked decision settles, since criterion 8 strips color entirely; tuning fg/bg for the
// colored web preview is cosmetic and left open for the taste read that criterion asks for.

import tgpu, { type StorageFlag, type TgpuBuffer, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";
import { Cell, packCell } from "./cell";
import { CELL_FILL_GLYPHS } from "./ramp";

type CellBuffer = TgpuBuffer<d.WgslArray<typeof Cell>> & StorageFlag;

const WG = 8;
// a fixed 3×3 sub-sample grid per cell (module doc above) — bounds pass 1's cost by grid size alone,
// never by render resolution.
const CELL_SAMPLES = 3;
const PI = Math.PI;

/**
 * the Sobel-magnitude gate between a directional glyph and the coverage-ordered fill ramp
 * (module doc above). Derived, not fitted: the 3×3 Sobel kernel's weights sum to 4 per axis, so a hard
 * black-to-white step edge (a post-tonemap luma jump of 1.0) between two adjacent cells reads magnitude
 * ≈4 when axis-aligned. The diagonal case is **not** ≈5.66 (`gx = gy = 4` is unreachable — the two
 * kernels share taps, e.g. `l02`/`l20` carry opposite sign between `gx` and `gy`, so driving `gx` to its
 * axis maximum caps what `gy` can reach at the same time): the true maximum over the 8 free luma taps in
 * `[0, 1]` a linear objective over a box, so the max sits at a vertex, brute-forced over all 256 in
 * `select.test.ts` — is `4·cos(t) + 2·sin(t)` maximized over `t`, `sqrt(20) ≈ 4.47`. 0.4 requires roughly
 * a 0.1 average per-tap luma delta across the neighborhood to fire (the axis-aligned case, `0.4 / 4`) —
 * well above sensor/render noise on a lit 3D scene, well below what a silhouette or a hard shadow
 * boundary produces. This is the constant criterion 8's taste read is expected to move — record a new
 * derivation here if it does, not just the number.
 */
export const EDGE_MAGNITUDE_THRESHOLD = 0.4;

/** pass 1 + pass 2's shared dims uniform: the cell grid shape plus the source texture's own pixel size
 *  (block-sample bounds derive from both). @internal */
export const SelectParams = d.struct({ cols: d.u32, rows: d.u32, srcW: d.u32, srcH: d.u32 });

/** pass 1's bind group: the dims uniform, the source color texture, the per-cell average out.
 *  @internal */
export const avgLayout = tgpu.bindGroupLayout({
    params: { uniform: SelectParams, visibility: ["compute"] },
    // unfilterable-float: this pass only ever `textureLoad`s (an exact texel fetch, no interpolation), so
    // it never needs the `float32-filterable` feature a `Float` sample type would require for a 32-bit-
    // per-channel source (`rgba32float`, the differential arm's synthetic fixture format) — `rg11b10ufloat`
    // (the production offscreen's format) is core-filterable regardless, but declaring the tighter,
    // textureLoad-sufficient sample type here is correct for both and portable to a source this pass
    // never asks to interpolate.
    source: {
        texture: d.texture2d(d.f32),
        sampleType: "unfilterable-float",
        visibility: ["compute"],
    },
    avg: {
        storage: (n: number) => d.arrayOf(d.vec4f, n),
        access: "mutable",
        visibility: ["compute"],
    },
});

/** pass 2's bind group: the dims uniform, pass 1's averages in, the cell buffer out. @internal */
export const selectLayout = tgpu.bindGroupLayout({
    params: { uniform: SelectParams, visibility: ["compute"] },
    avg: {
        storage: (n: number) => d.arrayOf(d.vec4f, n),
        access: "readonly",
        visibility: ["compute"],
    },
    cells: {
        storage: (n: number) => d.arrayOf(Cell, n),
        access: "mutable",
        visibility: ["compute"],
    },
});

/** Reinhard tonemap (`c / (c + 1)`), per channel — the same cheap operator `avgKernel` compresses each
 *  sample through before averaging, so a >1 HDR sample doesn't dominate its cell's mean. Not the engine's
 *  full `Glaze` chain (grade, saturation, OkLab posterize/dither, vignette): this value only ever feeds a
 *  luma/edge read and a background swatch, never the pixel the viewer directly judges tone against.
 *  @example const ldr = reinhard(hdrColor); */
export const reinhard = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    return std.div(c, std.add(c, d.vec3f(1)));
});

/** rec709 luma — the scalar both the fill-ramp index and the Sobel edge read derive from.
 *  @example const l = luma(color); */
export const luma = tgpu.fn(
    [d.vec3f],
    d.f32,
)((c) => {
    "use gpu";
    return std.dot(c, d.vec3f(0.2126, 0.7152, 0.0722));
});

/**
 * the gradient `(gx, gy)` → a directional glyph index (`ramp.ts`'s `CELL_DIRECTIONAL_GLYPHS`, appended
 * after the fill ramp). `gx`/`gy` are read in `selectKernel`'s own grid frame — row index increases
 * **downward** (`l02`/`l12`/`l22` sit at `cy2 = yi + 1`, the frame `gy = bottom row − top row` is
 * measured in), while `ramp.ts`'s four bucket labels (`0°=-, 45°=/, 90°=|, 135°=\`) are stated as the
 * angle the glyph visually runs, i.e. standard math convention (y **up**). `atan2(-gy, gx)` converts the
 * grid's y-down gradient into that y-up frame before bucketing — a grid-frame positive `gy` (brighter
 * below) is a math-frame negative `gy` (brighter toward −y) — folds to `[0, π)` (a line has no polarity —
 * gradient 10° and gradient 190° are the same edge), buckets into the four standard 45°-spaced Canny
 * non-max-suppression buckets, then **rotates by two buckets (90°) to the perpendicular tangent bucket**
 * (`ramp.ts`'s locked contract): a gradient is perpendicular to the edge it belongs to, so indexing the
 * directional set by a raw gradient bucket draws every glyph turned 90° from the edge it's meant to
 * represent. Dropping the y-down→y-up conversion swaps the two diagonal glyphs and leaves the two
 * axis-aligned buckets untouched — negating `gy` doesn't change `atan2`'s bucket for `gy = 0` (horizontal)
 * or `gx = 0` (vertical), only for a true diagonal, which is exactly the shape the first version of this
 * function shipped with (`specs/shallot-tui.md`'s s3r item 1). A pure function of `(gx, gy)` — no binding
 * access — so it's callable directly from `bun test` (typegpu's dual CPU/GPU form, `packCell`'s own shape)
 * with no device.
 * @example const glyph = directionalGlyphIndex(gx, gy); // CELL_FILL_GLYPHS.length + a tangent bucket 0..3
 */
export const directionalGlyphIndex = tgpu.fn(
    [d.f32, d.f32],
    d.u32,
)((gx, gy) => {
    "use gpu";
    let angle = std.atan2(-gy, gx);
    if (angle < 0) angle = angle + PI;
    const gradientBucket = d.u32(std.round(angle / (PI / 4))) % 4;
    const tangentBucket = (gradientBucket + 2) % 4;
    return d.u32(CELL_FILL_GLYPHS.length) + tangentBucket;
});

const avgKernel = tgpu.computeFn({
    workgroupSize: [WG, WG],
    in: { gid: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const x = input.gid.x;
    const y = input.gid.y;
    const cols = avgLayout.$.params.cols;
    const rows = avgLayout.$.params.rows;
    if (x >= cols || y >= rows) return;
    const srcW = avgLayout.$.params.srcW;
    const srcH = avgLayout.$.params.srcH;
    const blockW = d.f32(srcW) / d.f32(cols);
    const blockH = d.f32(srcH) / d.f32(rows);

    let sum = d.vec3f(0, 0, 0);
    for (let sy = d.u32(0); sy < d.u32(CELL_SAMPLES); sy = sy + 1) {
        for (let sx = d.u32(0); sx < d.u32(CELL_SAMPLES); sx = sx + 1) {
            const u = d.f32(x) + (d.f32(sx) + 0.5) / d.f32(CELL_SAMPLES);
            const v = d.f32(y) + (d.f32(sy) + 0.5) / d.f32(CELL_SAMPLES);
            const px = std.min(d.u32(u * blockW), srcW - 1);
            const py = std.min(d.u32(v * blockH), srcH - 1);
            const sample = std.textureLoad(avgLayout.$.source, d.vec2u(px, py), 0).xyz;
            sum = std.add(sum, reinhard(sample));
        }
    }
    const avg = std.mul(sum, 1 / (CELL_SAMPLES * CELL_SAMPLES));
    const i = y * cols + x;
    avgLayout.$.avg[i] = d.vec4f(avg, 1);
});

// A diagonal-then-axis-aligned 3×3 Sobel, unrolled over the 8 neighbors clamped to the grid edge (never
// wraps): weights sum to 4 per axis, matching EDGE_MAGNITUDE_THRESHOLD's derivation above.
const selectKernel = tgpu.computeFn({
    workgroupSize: [WG, WG],
    in: { gid: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const x = input.gid.x;
    const y = input.gid.y;
    const cols = selectLayout.$.params.cols;
    const rows = selectLayout.$.params.rows;
    if (x >= cols || y >= rows) return;

    const colsI = d.i32(cols);
    const rowsI = d.i32(rows);
    const xi = d.i32(x);
    const yi = d.i32(y);
    const cx0 = std.clamp(xi - 1, 0, colsI - 1);
    const cx2 = std.clamp(xi + 1, 0, colsI - 1);
    const cy0 = std.clamp(yi - 1, 0, rowsI - 1);
    const cy2 = std.clamp(yi + 1, 0, rowsI - 1);

    const l00 = luma(selectLayout.$.avg[cy0 * colsI + cx0].xyz);
    const l10 = luma(selectLayout.$.avg[cy0 * colsI + xi].xyz);
    const l20 = luma(selectLayout.$.avg[cy0 * colsI + cx2].xyz);
    const l01 = luma(selectLayout.$.avg[yi * colsI + cx0].xyz);
    const l21 = luma(selectLayout.$.avg[yi * colsI + cx2].xyz);
    const l02 = luma(selectLayout.$.avg[cy2 * colsI + cx0].xyz);
    const l12 = luma(selectLayout.$.avg[cy2 * colsI + xi].xyz);
    const l22 = luma(selectLayout.$.avg[cy2 * colsI + cx2].xyz);

    const gx = l20 + 2 * l21 + l22 - (l00 + 2 * l01 + l02);
    const gy = l02 + 2 * l12 + l22 - (l00 + 2 * l10 + l20);
    const magnitude = std.sqrt(gx * gx + gy * gy);

    const own = selectLayout.$.avg[yi * colsI + xi].xyz;
    const ownLuma = luma(own);

    let glyph = d.u32(0);
    if (magnitude > EDGE_MAGNITUDE_THRESHOLD) {
        glyph = directionalGlyphIndex(gx, gy);
    } else {
        const fillCount = d.u32(CELL_FILL_GLYPHS.length);
        const idx = d.u32(std.round(std.saturate(ownLuma) * d.f32(fillCount - 1)));
        glyph = std.min(idx, fillCount - 1);
    }

    const fg = std.select(d.vec3f(0, 0, 0), d.vec3f(1, 1, 1), ownLuma < 0.5);
    const packed = packCell(glyph, d.vec4f(fg, 1), d.vec4f(own, 1));
    const i = y * cols + x;
    selectLayout.$.cells[i].glyph = packed.x;
    selectLayout.$.cells[i].fg = packed.y;
    selectLayout.$.cells[i].bg = packed.z;
});

let _avgBuffer: ReturnType<typeof allocAvg> | null = null;
let _avgCells = 0;
let _avgPipeline: TgpuComputePipeline | null = null;
let _selectPipeline: TgpuComputePipeline | null = null;
let _paramsBuffer: ReturnType<typeof allocParams> | null = null;

function allocParams() {
    return Compute.root.createBuffer(SelectParams).$usage("uniform").$name("cells-select-params");
}

// one persistent params uniform, written (never recreated) per call. `recordSelect` records into a
// caller-owned encoder it never submits itself (`CellsPlugin` shares one per-frame `Render.encoder` with
// the draw pass), so a fresh per-call buffer destroyed on any local timeline — synchronously, or via
// `onSubmittedWorkDone()` read against whatever happened to be in flight at call time — can be destroyed
// before the *this* recording's own submit ever executes (measured: "Buffer used in submit while
// destroyed" GPUValidationError flooding every frame after the first). Writing into one long-lived buffer
// sidesteps the lifecycle question entirely — mirrors `standard/glaze/index.ts`'s per-slot `configBuffer`.
function paramsBuffer() {
    if (!_paramsBuffer) _paramsBuffer = allocParams();
    return _paramsBuffer;
}

function allocAvg(cells: number) {
    return Compute.root
        .createBuffer(d.arrayOf(d.vec4f, cells))
        .$usage("storage")
        .$name("cells-select-avg");
}

// the intermediate per-cell average buffer, (re)allocated to `cols * rows`. Not caller-visible — pass 1
// writes it, pass 2 reads it, in the same dispatch; nothing outside this module needs it. Shared across
// calls within one frame (like `paramsBuffer`): safe as long as every caller in that frame requests the
// same `cols * rows` (one camera, `CellsPlugin`'s fixed grid) — a size change mid-frame would `.destroy()`
// this buffer while an earlier call's still-unsubmitted recording references the old instance, the same
// hazard class `paramsBuffer`'s docblock names. Per-camera keying is the fix once a real multi-camera
// caller exists; out of this module's scope today.

function avgBuffer(cells: number) {
    if (_avgBuffer && _avgCells === cells) return _avgBuffer;
    _avgBuffer?.destroy();
    _avgBuffer = allocAvg(cells);
    _avgCells = cells;
    return _avgBuffer;
}

function pipelines(): { avg: TgpuComputePipeline; select: TgpuComputePipeline } {
    if (!_avgPipeline) {
        _avgPipeline = Compute.root
            .createComputePipeline({ compute: avgKernel })
            .$name("cells-avg");
    }
    if (!_selectPipeline) {
        _selectPipeline = Compute.root
            .createComputePipeline({ compute: selectKernel })
            .$name("cells-select");
    }
    return { avg: _avgPipeline, select: _selectPipeline };
}

/** drop the memoized select pipelines + the intermediate average and params buffers — a re-adopted device
 *  needs fresh ones (`grid.ts`'s `resetPipeline` shape). @internal */
export function resetSelectPipelines(): void {
    _avgBuffer?.destroy();
    _avgBuffer = null;
    _avgCells = 0;
    _avgPipeline = null;
    _selectPipeline = null;
    _paramsBuffer?.destroy();
    _paramsBuffer = null;
}

/**
 * record both passes into `encoder`, over `source` (a float-sampleable 2D texture view — the camera's
 * offscreen scene color in production, `view.framebuffer`; any RGBA/RG11B10 float texture in a test or a
 * differential arm), writing every cell of `cells` (a `Cell` storage buffer sized `cols * rows`,
 * `grid.ts`'s `CellGrid.buffer` shape). Allocates/reuses its own intermediate average buffer, sized to
 * `cols * rows`. Records only — the caller submits (`CellsPlugin` shares one per-frame `Render.encoder`
 * with the draw pass that reads this dispatch's output; {@link dispatchSelect} is the standalone,
 * submit-its-own-encoder convenience wrapper for a test or gym scenario with no frame encoder of its own).
 *
 * @example recordSelect(encoder, cellsBuffer, cols, rows, framebufferView, srcW, srcH);
 */
export function recordSelect(
    encoder: GPUCommandEncoder,
    cells: CellBuffer,
    cols: number,
    rows: number,
    source: GPUTextureView,
    srcW: number,
    srcH: number,
): void {
    const params = paramsBuffer();
    params.write({ cols, rows, srcW, srcH });
    const avg = avgBuffer(cols * rows);
    const { avg: avgPipeline, select: selectPipeline } = pipelines();

    const avgGroup = Compute.root.createBindGroup(avgLayout, { params, source, avg });
    const selectGroup = Compute.root.createBindGroup(selectLayout, { params, avg, cells });

    const dispatchDims: [number, number] = [Math.ceil(cols / WG), Math.ceil(rows / WG)];

    const avgPass = encoder.beginComputePass({
        label: "cells-avg",
        timestampWrites: Compute.span?.("cells:avg"),
    });
    avgPipeline
        .with(avgGroup)
        .with(avgPass)
        .dispatchWorkgroups(...dispatchDims);
    avgPass.end();

    const selectPass = encoder.beginComputePass({
        label: "cells-select",
        timestampWrites: Compute.span?.("cells:select"),
    });
    selectPipeline
        .with(selectGroup)
        .with(selectPass)
        .dispatchWorkgroups(...dispatchDims);
    selectPass.end();
}

/**
 * standalone convenience: record + submit both passes in their own command buffer. For a test or gym
 * scenario with no per-frame `Render.encoder` to share (`grid.ts`'s `fillCellGrid` shape) — production
 * (`CellsPlugin`) calls {@link recordSelect} directly against the shared frame encoder instead.
 *
 * @example dispatchSelect(cellsBuffer, cols, rows, framebufferView, srcW, srcH);
 */
export function dispatchSelect(
    cells: CellBuffer,
    cols: number,
    rows: number,
    source: GPUTextureView,
    srcW: number,
    srcH: number,
): void {
    const device = Compute.device;
    const encoder = device.createCommandEncoder({ label: "cells-select" });
    recordSelect(encoder, cells, cols, rows, source, srcW, srcH);
    device.queue.submit([encoder.finish()]);
}

/** the emitted select-pass WGSL (both kernels) — the device-free structural seam its test resolves.
 *  @internal */
export function selectWgsl(): string {
    return tgpu.resolve([avgKernel, selectKernel], { names: "strict" });
}
