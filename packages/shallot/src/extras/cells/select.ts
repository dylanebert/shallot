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
// coverage-ordered fill glyph from the cell's own luma. The s3r fill-treatment amendment's "a shared face
// boundary carries its own rule" (`specs/shallot-tui.md`) is a **second, independent** gate,
// {@link FACE_BOUNDARY_MAGNITUDE_THRESHOLD} — not a lower setting of the same one: the full 3×3 Sobel
// kernel's diagonal taps carry a real face boundary's influence *past* its own row/column into an
// adjacent, genuinely flat cell (measured directly, `FACE_BOUNDARY_MAGNITUDE_THRESHOLD`'s own docblock has
// the reproduction), so a single lowered Sobel threshold sensitive enough to catch a dim interior boundary
// also mis-fires one cell into every flat neighbor beside one, turning a small box's whole visible surface
// into edge glyphs. The boundary gate instead reads a **4-connected central difference** (own-row
// left/right, own-column up/down only, no diagonals) — a metric that cannot see past its immediate
// neighbor, so it reads exactly 0 for the flat-neighbor case the Sobel-based approach mis-fired on. Either
// gate firing selects a directional glyph, oriented by the *full* Sobel `(gx, gy)` (the diagonal taps are
// fine for angle, since a slightly bled angle is a far smaller defect than a mis-classified cell); `fg` is
// forced to full-bright white whenever either fires — a solid, continuous, brighter-than-either-fill
// stroke, per the reference's own contrast hierarchy (edges continuous and bright, fill dim and broken) —
// and `bg` stays the cell's own tonemapped average (unchanged, cosmetic for the colored web preview;
// criterion 8 strips color entirely so nothing here bears on it).

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
 * derivation here if it does, not just the number. Unchanged by the s3r fill-treatment amendment: the
 * shared-face-boundary rule it adds is {@link FACE_BOUNDARY_MAGNITUDE_THRESHOLD}, a separate gate on a
 * separate, more localized metric — own docblock has the reason a single lowered Sobel threshold doesn't
 * serve both roles.
 */
export const EDGE_MAGNITUDE_THRESHOLD = 0.4;

/**
 * the shared-face-boundary gate (the s3r fill-treatment amendment, `specs/shallot-tui.md`: "a shared face
 * boundary carries its own rule") — independent of {@link EDGE_MAGNITUDE_THRESHOLD}, on an independent
 * metric: a **4-connected central difference**, `dx = l21 - l01` (own row, right minus left) and
 * `dy = l12 - l10` (own column, bottom minus top), `magnitude = sqrt(dx² + dy²)`, using none of the full
 * Sobel kernel's diagonal taps (`l00`, `l20`, `l02`, `l22`). Tried first as a second, lower threshold on
 * the *same* Sobel magnitude `selectKernel` already computes, and measured wrong: the Sobel kernel's
 * diagonal taps carry a real boundary's influence *past* its own row/column into an adjacent, genuinely
 * flat cell — reproduced directly against `examples/recipes/render-to-a-terminal` at a yaw showing three
 * faces (2.4 rad): a cell sitting one row below a real horizontal face boundary, with identical luma to
 * both its immediate left and right neighbors *and* to the same cell one row above and one row below (a
 * textbook-flat 4-connected neighborhood), still read a nonzero Sobel magnitude solely from its two
 * diagonal corner taps landing in the *next* face over — turning a small on-screen cube's entire visible
 * surface into directional glyphs, the opposite of rule 3's "facade ink is low." The central-difference
 * metric reads exactly 0 there (its four inputs are the flat 4-neighbors only), while still reading the
 * *target* defect correctly: at the two-face boundary the amendment names, the adjacent cell differs by
 * roughly a 0.012 luma step, giving `magnitude ≈ 0.012` directly (weights are ±1, not Sobel's ±4, so no 4×
 * amplification). The threshold is derived the same way {@link BG_MATCH_EPSILON} derives its own budget:
 * half the fill ramp's own per-index luma step (`0.5 / (CELL_FILL_GLYPHS.length - 1)`, ≈0.0058 at 87 fill
 * glyphs) — comfortably above the quantization-noise floor (`BG_MATCH_EPSILON`'s own ~0.0007 derivation,
 * several times below it) and comfortably below the measured real boundary (≈2x margin), re-deriving
 * automatically if the ramp's own length moves. When this gate fires, the emitted glyph's *angle* still
 * comes from the full Sobel `(gx, gy)` (`directionalGlyphIndex`) — a slightly bled angle at the boundary's
 * own edge is a far smaller defect than the mis-classified-flat-cell one this gate exists to avoid.
 */
export const FACE_BOUNDARY_MAGNITUDE_THRESHOLD = 0.5 / (CELL_FILL_GLYPHS.length - 1);

/**
 * the background-match gate: how close a cell's own tonemapped block average must sit to the
 * caller-supplied background reference (`SelectParams.bg`, tonemapped the same way inside the kernel)
 * before the cell is forced to the blank fill glyph regardless of luma or edge (the s3r item-8 repair —
 * `specs/shallot-tui.md`'s residue log names the mechanism: a non-black clear color's tonemapped luma
 * never reaches the ramp's zero-coverage entry, so an unlit background read as a uniform field of `'`
 * instead of blank). Derived from `rg11b10ufloat`'s own quantization, not fitted: the format packs R/G
 * as 11-bit unsigned floats (5 exponent + 6 mantissa bits) and B as 10-bit (5 exponent + 5 mantissa),
 * so the worst-case per-channel relative step is `2^-5` (B) — for the default clear color's raw linear
 * B channel (~0.0212), that's a ~0.00066 absolute step, propagated through reinhard (slope ≈1 near 0)
 * to a similar post-tonemap delta; a pass-1 block average of 9 bit-identical background samples divided
 * by 9 adds only float-rounding noise on top, several orders below that. 0.004 clears that quantization
 * budget by roughly 4-6x while staying well under the fill ramp's own per-index luma step (`1 /
 * (CELL_FILL_GLYPHS.length - 1)` ≈ 0.0116 at 87 fill glyphs, post rule-2's curved-glyph exclusion) — the
 * gap a legitimately-one-step-darker real surface sits at, so this gate should not misfire on it.
 */
export const BG_MATCH_EPSILON = 0.004;

/** the sentinel background reference {@link recordSelect} / {@link dispatchSelect} default to when the
 *  caller has no known background color (e.g. a synthetic test texture with no camera behind it):
 *  tonemapped by the same `reinhard()` the kernel applies to a real reference, `reinhard(-1e6)` lands at
 *  `≈1.000001`, outside every luma a real scene sample's own reinhard-compressed average can reach
 *  (strictly `< 1`), so {@link BG_MATCH_EPSILON}'s distance check can never fire against it. @internal */
export const NO_BACKGROUND: readonly [number, number, number] = [-1e6, -1e6, -1e6];

/** pass 1 + pass 2's shared dims uniform: the cell grid shape, the source texture's own pixel size
 *  (block-sample bounds derive from both), and pass 2's background reference (`bg`, raw linear, un-
 *  tonemapped — {@link NO_BACKGROUND} when the caller has none). @internal */
export const SelectParams = d.struct({
    cols: d.u32,
    rows: d.u32,
    srcW: d.u32,
    srcH: d.u32,
    bg: d.vec4f,
});

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

/**
 * the fill role's own luma→index map: round `luma` linearly across the coverage-ordered ramp, clamped to
 * the last fill index. Extracted from `selectKernel`'s own inline arithmetic (unchanged) so the exact same
 * mapping is callable with no device (`directionalGlyphIndex`'s own dual CPU/GPU shape) — the facade-ink
 * measurement below (rule 3, `specs/shallot-tui.md`'s fill-treatment amendment) has to use the *real*
 * mapping the kernel selects with, not a re-derived copy that could silently drift from it
 * (`checks.md`: "an oracle that shares an assumption with the thing it checks proves nothing" runs the
 * other way here — the risk is a *second*, drifting implementation, not a shared blind spot, so sharing one
 * function is the fix).
 * @example const idx = fillIndexForLuma(0.55); // an index into CELL_FILL_GLYPHS
 */
export const fillIndexForLuma = tgpu.fn(
    [d.f32],
    d.u32,
)((luma) => {
    "use gpu";
    const fillCount = d.u32(CELL_FILL_GLYPHS.length);
    const idx = d.u32(std.round(std.saturate(luma) * d.f32(fillCount - 1)));
    return std.min(idx, fillCount - 1);
});

/**
 * the shared-face-boundary gate's own metric ({@link FACE_BOUNDARY_MAGNITUDE_THRESHOLD}'s own docblock has
 * the derivation and the reproduction that motivated it): a 4-connected central difference over the
 * *immediate* left/right/top/bottom neighbors only — no diagonal taps — so it cannot see past its own row
 * or column the way the full Sobel kernel (`selectKernel`'s `gx`/`gy`) does. Extracted to a pure,
 * dual-mode function (`directionalGlyphIndex`'s own shape) so `select.test.ts` can reproduce the exact
 * bled-Sobel-but-flat-4-neighbor scenario this gate exists to fix, with no device.
 * @example const m = localBoundaryMagnitude(0.5, 0.5, 0.5, 0.5); // 0 — a flat 4-neighborhood
 */
export const localBoundaryMagnitude = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32],
    d.f32,
)((left, right, top, bottom) => {
    "use gpu";
    const dx = right - left;
    const dy = bottom - top;
    return std.sqrt(dx * dx + dy * dy);
});

/**
 * the representative facade-luma population the facade-ink measurement (rule 3, `specs/shallot-tui.md`'s
 * fill-treatment amendment) averages over: 101 evenly spaced samples across the full `[0, 1]` luma domain
 * `fillIndexForLuma` maps from. Deliberately the *whole* domain rather than this one recipe's own narrow
 * observed band (`EDGE_MAGNITUDE_THRESHOLD`'s own docblock cites that band, roughly 0.517-0.603, at one
 * orbit angle) — the mapping this measures is a property of the selector, not of one camera pose, and a
 * population scoped to a single frame's own readings would be self-graded (`checks.md`: "an item gated by
 * a file its own diff creates is self-graded"). Exported so the two-sided arms (`select.test.ts`) and the
 * real-device measurement (`examples/gym/src/scenarios/cells.ts`'s own `assertFacadeInk`) read one shared
 * population rather than two that could quietly diverge.
 */
export const FACADE_LUMA_SWEEP: readonly number[] = Array.from({ length: 101 }, (_, i) => i / 100);

/** the facade-ink band's floor and ceiling (rule 3, `specs/shallot-tui.md`: "the target is the number,"
 *  ~10-35% off the reference's own measured facade crops). Real, achievable numbers, not aspirational
 *  ones: a single ASCII glyph's own outline coverage in this brand font tops out under 10% even fully
 *  saturated to a 1×1-em footprint (`glyphs.ts`'s `FILL_GLYPH_INK_SCALE`, `draw.ts`'s `INK_DILATE_PX` —
 *  both docblocks carry the measurement that made the floor reachable at all), so these bounds are read
 *  off {@link FACADE_LUMA_SWEEP} against the real `cells` gym scenario's own per-glyph device readback, not
 *  copied from the reference's crops unread. */
export const FACADE_INK_FLOOR = 0.1;
export const FACADE_INK_CEILING = 0.35;

/**
 * average rendered ink over a population of facade lumas, each mapped to its fill index by
 * {@link fillIndexForLuma} (the real selector mapping, not a re-derived copy) and looked up in
 * `inkByIndex` — the real per-glyph device readback in production (`examples/gym/src/scenarios/cells.ts`'s
 * `assertFacadeInk`), or a synthetic all-high / all-zero array in the two-sided vacuity arms
 * (`select.test.ts`), which is what proves this function — not just its production caller — actually
 * discriminates a dense population from a blank one.
 * @example facadeInkFraction(FACADE_LUMA_SWEEP, realInkReadback); // ~0.1-0.35 in production
 */
export function facadeInkFraction(lumas: readonly number[], inkByIndex: readonly number[]): number {
    if (lumas.length === 0) return 0;
    let sum = 0;
    for (const luma of lumas) {
        const idx = fillIndexForLuma(luma);
        sum += inkByIndex[idx] ?? 0;
    }
    return sum / lumas.length;
}

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

    // the shared-face-boundary gate's own localized metric (FACE_BOUNDARY_MAGNITUDE_THRESHOLD's own
    // docblock has the derivation): a 4-connected central difference, no diagonal taps, so it can't bleed
    // a real boundary's influence into an adjacent flat cell the way the full Sobel kernel above does.
    const magnitudeLocal = localBoundaryMagnitude(l01, l21, l10, l12);

    const own = selectLayout.$.avg[yi * colsI + xi].xyz;
    const ownLuma = luma(own);
    const bgTone = reinhard(selectLayout.$.params.bg.xyz);

    let glyph = d.u32(0);
    // two independent gates: the silhouette/hard-shadow Sobel gate (EDGE_MAGNITUDE_THRESHOLD) and the
    // shared-face-boundary localized gate (FACE_BOUNDARY_MAGNITUDE_THRESHOLD) — either firing selects a
    // directional glyph, oriented by the full Sobel (gx, gy) regardless of which gate fired, and gets the
    // same bright, continuous-stroke emphasis below, per the reference's own contrast hierarchy (edges
    // continuous and bright, fill dim and broken).
    const isEdge =
        magnitude > EDGE_MAGNITUDE_THRESHOLD || magnitudeLocal > FACE_BOUNDARY_MAGNITUDE_THRESHOLD;
    if (std.distance(own, bgTone) < BG_MATCH_EPSILON) {
        // untouched background: force the ramp's own zero-coverage entry (glyph index 0, guaranteed
        // to be the blank glyph — `ramp.ts`'s `candidateChars` starts at code point 0x20 and ties at
        // zero coverage break by ascending code point, so nothing sorts ahead of it) rather than
        // whatever index the background's own non-zero clear luma would otherwise round to.
        glyph = d.u32(0);
    } else if (isEdge) {
        glyph = directionalGlyphIndex(gx, gy);
    } else {
        glyph = fillIndexForLuma(ownLuma);
    }

    // fill ink stays luma-derived (dim and broken, per the reference); an edge glyph always gets
    // full-bright fg regardless of its own luma, so the stroke reads brighter than either adjacent fill
    // rather than sometimes matching or undershooting a bright fill's own ink (`ownLuma < 0.5`'s old split
    // would otherwise hand a bright-face edge black ink, the opposite of "brighter than either fill").
    const fillFg = std.select(d.vec3f(0, 0, 0), d.vec3f(1, 1, 1), ownLuma < 0.5);
    const fg = std.select(fillFg, d.vec3f(1, 1, 1), isEdge);
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
 * `bg` is the raw linear (un-tonemapped) color of the source's own empty background — a camera's
 * unpacked-to-linear `Camera.clearColor` in production — so a cell whose source region is untouched
 * background selects the blank fill glyph instead of whatever index its clear luma would otherwise round
 * to ({@link BG_MATCH_EPSILON}'s module doc names the mechanism). Omit it (or pass {@link NO_BACKGROUND})
 * when the source has no real background — a synthetic test texture with no camera behind it — and the
 * gate never fires.
 *
 * @example recordSelect(encoder, cellsBuffer, cols, rows, framebufferView, srcW, srcH, [0.02, 0.02, 0.02]);
 */
export function recordSelect(
    encoder: GPUCommandEncoder,
    cells: CellBuffer,
    cols: number,
    rows: number,
    source: GPUTextureView,
    srcW: number,
    srcH: number,
    bg: readonly [number, number, number] = NO_BACKGROUND,
): void {
    const params = paramsBuffer();
    params.write({ cols, rows, srcW, srcH, bg: [bg[0], bg[1], bg[2], 0] });
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
 * (`CellsPlugin`) calls {@link recordSelect} directly against the shared frame encoder instead. `bg` is
 * {@link recordSelect}'s own background-reference parameter, same default.
 *
 * @example dispatchSelect(cellsBuffer, cols, rows, framebufferView, srcW, srcH, [0.02, 0.02, 0.02]);
 */
export function dispatchSelect(
    cells: CellBuffer,
    cols: number,
    rows: number,
    source: GPUTextureView,
    srcW: number,
    srcH: number,
    bg: readonly [number, number, number] = NO_BACKGROUND,
): void {
    const device = Compute.device;
    const encoder = device.createCommandEncoder({ label: "cells-select" });
    recordSelect(encoder, cells, cols, rows, source, srcW, srcH, bg);
    device.queue.submit([encoder.finish()]);
}

/** the emitted select-pass WGSL (both kernels) — the device-free structural seam its test resolves.
 *  @internal */
export function selectWgsl(): string {
    return tgpu.resolve([avgKernel, selectKernel], { names: "strict" });
}
