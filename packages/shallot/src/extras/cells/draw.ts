// The web sink: an instanced draw of `cols * rows` monospace quads against the shared SDF glyph atlas
// (`text/core`), no readback — the Locked decision's "same pipeline with the expensive tail deleted"
// (`specs/shallot-tui.md`). A simplification of `extras/text`'s own instanced glyph draw: no world
// transform, no per-entity eid lookup, no per-string layout pass — every cell is a fixed monospace box
// whose position derives purely from its instance index. The quad's *geometry* always covers the whole
// box (`checks.md`'s "invariant enforced on the quantity it constrains": the Goal's "the grid is the
// frame" property lives in every pixel of every cell getting a real write, so the rasterized polygon can
// never be smaller than the cell — `specs/shallot-tui.md`'s s3r item 9), and the glyph's own ink is drawn
// *within* that full-cell quad, shrunk + centered to the glyph's own measured em-normalized footprint
// (`glyphFootprintT`/`cellFootprintPx` below, against `glyphs.ts`'s size table) — size-proportional
// placement, the same principle `extras/text`'s own `layoutText` applies per glyph (`ramp.ts`'s module
// doc names the contract this reads). No vertex/index buffer either: the quad's six corners come from
// `@builtin(vertex_index)` through a small lookup table (`fullscreenVs`'s shape, `extras/outline`), and
// `@builtin(instance_index)` is the cell — `col = iid % cols`, `row = iid / cols`.
//
// Targets `view.framebuffer` directly (the camera's offscreen scene color, RENDER_ATTACHMENT-usable —
// `standard/render/view.ts`'s `offscreen()`), drawn *after* the scene renders and *before* glaze
// composites it to the swapchain — the same post-color-seam ordering `extras/outline` uses
// (`after: [ColorSystem, OverlaySystem], before: [GlazeSystem]`). That ordering is what makes the color
// convention correct: `Cell.fg`/`Cell.bg` are sRGB-packed (`cell.ts`'s own contract), and this fragment
// stage decodes them to linear before writing — the framebuffer is linear HDR, and glaze's own tonemap +
// linear→sRGB encode is what turns that into the swapchain's bytes, exactly the path every other
// packed-color slab in this engine takes (`extras/text`'s own glyph color decode is the same shape). A
// direct-to-swapchain write skipping that decode would be visibly wrong in the opposite direction from
// what it looks like it fixes (`cell.ts`'s module doc states the general case).

import tgpu, { type StorageFlag, type TgpuBuffer, type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";
import { Render } from "../../standard/render/core";
import { sdfToSignedDistance, textSrgbToLinear } from "../text/core";
import { Cell } from "./cell";
import type { GlyphSizeBuffer, GlyphUvBuffer } from "./glyphs";

type CellBuffer = TgpuBuffer<d.WgslArray<typeof Cell>> & StorageFlag;

/** the draw's one bind group: dims uniform, the cell buffer + glyph uv/size tables (vertex-read), the
 *  shared atlas texture + sampler (fragment-read). @internal */
export const DrawParams = d.struct({ cols: d.u32, rows: d.u32, viewW: d.f32, viewH: d.f32 });

/** @internal */
export const drawLayout = tgpu.bindGroupLayout({
    params: { uniform: DrawParams, visibility: ["vertex", "fragment"] },
    cells: {
        storage: (n: number) => d.arrayOf(Cell, n),
        access: "readonly",
        visibility: ["vertex"],
    },
    glyphUv: {
        storage: (n: number) => d.arrayOf(d.vec4f, n),
        access: "readonly",
        visibility: ["vertex"],
    },
    glyphSize: {
        storage: (n: number) => d.arrayOf(d.vec2f, n),
        access: "readonly",
        visibility: ["vertex"],
    },
    atlasSamp: { sampler: "filtering", visibility: ["fragment"] },
    atlasTex: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
});

// floor under a glyph's own em-normalized width/height (`glyphs.ts`'s `glyphSizeTable`, each clamped to
// at most 1) before it feeds a division — a real font glyph's measured footprint is never exactly zero on
// either axis, but nothing enforces that at the type level, and `MISSING_GLYPH_SIZE`'s `(1, 1)` sentinel
// is already clear of it; this is a guard against a degenerate metric, not a documented case.
const MIN_GLYPH_EM = 0.001;

/**
 * a glyph's on-screen footprint in device px, isotropically scaled from its own em-normalized `size`
 * (`glyphs.ts`'s `glyphSizeTable`): both axes move by the *same* factor, `min(cellW, cellH)` — the
 * largest pixels-per-em scale that still keeps any glyph (`size` clamped to at most `(1, 1)`) inside the
 * *narrower* of the cell's two dimensions on both axes, so a footprint never overflows into a neighboring
 * cell. A per-axis factor (`cellW` on x, `cellH` on y, independently) was the s3r item 9 defect: it
 * re-stretches every glyph by the cell's own aspect ratio on top of the glyph's true shape, invisible on
 * a square test fixture (`cellW === cellH`, every `examples/gym/src/scenarios/cells.ts` cell — the two
 * factors collapse to the same number there, `draw.test.ts`'s own regression pins that collapse) and
 * highly visible on the real 80×24 grid's narrow-by-tall cells, where `min(cellW, cellH)` and
 * `max(cellW, cellH)` differ by roughly 2×. One shared scalar keeps a glyph's own aspect (both axes move
 * together, so a round mark stays round) while still rendering size-proportionally to its measured
 * coverage (`ramp.ts`'s ordering — the s3r item 2 property this must not re-break, `draw.test.ts`'s
 * "collapses to the old per-axis formula on a square cell" case proves the two coincide exactly there).
 *
 * @example cellFootprintPx(vec2f(0.5, 0.5), 16, 30); // vec2f(8, 8) — isotropic, capped by the narrower axis
 */
export const cellFootprintPx = tgpu.fn(
    [d.vec2f, d.f32, d.f32],
    d.vec2f,
)((size, cellW, cellH) => {
    "use gpu";
    const clamped = std.max(size, d.vec2f(MIN_GLYPH_EM, MIN_GLYPH_EM));
    const scale = std.min(cellW, cellH);
    return std.mul(clamped, scale);
});

/**
 * the inverse of a centered shrink: map a **full-cell** quad-local corner (0..1, spanning the whole cell)
 * to its position within the glyph's footprint (0..1 *inside* the footprint; outside it, this
 * extrapolates past `[0, 1]`). `sizeNorm` is the footprint's own fraction of the full cell on each axis
 * ({@link cellFootprintPx}'s result divided by `(cellW, cellH)` — never the raw, unscaled per-axis em
 * fraction, which reproduces the anisotropic-stretch defect {@link cellFootprintPx}'s docblock names).
 * The caller clamps the result to `[0, 1]` before using it as a uv-mix factor (so a fragment outside the
 * footprint samples the glyph tile's own edge texel rather than a neighboring glyph's) and gates ink on
 * whether it was in range before clamping (`cellFragment`'s `inside` test) — this function only computes
 * the raw, unclamped fraction, so both directions stay separately testable.
 *
 * @example glyphFootprintT(vec2f(0.5, 0.5), vec2f(0.5, 0.5)); // vec2f(0.5, 0.5) — cell center is always
 * the footprint center, at any footprint size
 */
export const glyphFootprintT = tgpu.fn(
    [d.vec2f, d.vec2f],
    d.vec2f,
)((corner, sizeNorm) => {
    "use gpu";
    const margin = std.mul(std.sub(d.vec2f(1, 1), sizeNorm), 0.5);
    return std.div(std.sub(corner, margin), sizeNorm);
});

// the unit quad's six corners (two triangles), indexed by @builtin(vertex_index) — no vertex buffer,
// mirroring `extras/outline`'s no-vertex-buffer fullscreen triangle one step up in vertex count
const QUAD_CORNERS = tgpu
    .const(d.arrayOf(d.vec2f, 6), [
        d.vec2f(0, 0),
        d.vec2f(1, 0),
        d.vec2f(1, 1),
        d.vec2f(0, 0),
        d.vec2f(1, 1),
        d.vec2f(0, 1),
    ])
    .$name("cellsQuadCorners");

const cellVaryings = {
    t: d.vec2f,
    rect: d.interpolate("flat", d.vec4f),
    gsize: d.interpolate("flat", d.vec2f),
    fg: d.vec4f,
    bg: d.vec4f,
    has: d.interpolate("flat", d.u32),
};

// the vertex stage's full output shape, `pos` included — declared once and reused verbatim for the
// fragment stage's `in` below (`cellFragment`) rather than the varyings alone. A fragment `in` that
// *omits* `pos` (the shape every field-less-of-pos struct in this pattern takes) is still legal WGSL, but
// it is a structurally different type from the vertex `out`, so `createRenderPipeline`'s stage-linking
// step converts one into the other at every resolve — the tgpu `[implicit-conversion]` console warning
// this fixes, verified by removing it (`draw.test.ts`'s own resolve-time proof, plus the real-browser
// reproduction the fix was validated against). Declaring both stages against the identical shape avoids
// the conversion at the source rather than suppressing the warning.
const cellVsOut = { pos: d.builtin.position, ...cellVaryings };

/** @internal */
export const cellVertex = tgpu
    .vertexFn({
        in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
        out: cellVsOut,
    })((input) => {
        "use gpu";
        const corner = QUAD_CORNERS.$[input.vidx];
        const cols = drawLayout.$.params.cols;
        const col = input.iid % cols;
        const row = input.iid / cols;
        const cellW = drawLayout.$.params.viewW / d.f32(cols);
        const cellH = drawLayout.$.params.viewH / d.f32(drawLayout.$.params.rows);

        const cell = Cell(drawLayout.$.cells[input.iid]);
        const rect = drawLayout.$.glyphUv[cell.glyph];
        const size = drawLayout.$.glyphSize[cell.glyph];
        const has = std.select(d.u32(0), d.u32(1), rect.z > rect.x);

        // full-cell geometry, unconditionally — every fragment of every cell gets a real write, so
        // nothing the scene rendered before this pass can show through (`specs/shallot-tui.md`'s s3r
        // item 9: "the grid is the frame", not an overlay on one).
        const px = (d.f32(col) + corner.x) * cellW;
        const py = (d.f32(row) + corner.y) * cellH;
        const ndcX = px / (drawLayout.$.params.viewW * 0.5) - 1;
        const ndcY = 1 - py / (drawLayout.$.params.viewH * 0.5);

        const gsize = cellFootprintPx(size, cellW, cellH);
        const sizeNorm = d.vec2f(gsize.x / cellW, gsize.y / cellH);
        const t = glyphFootprintT(corner, sizeNorm);

        const fg = std.unpack4x8unorm(cell.fg);
        const bg = std.unpack4x8unorm(cell.bg);

        return {
            pos: d.vec4f(ndcX, ndcY, 0, 1),
            t,
            rect,
            gsize,
            fg,
            bg,
            has,
        };
    })
    .$name("cellsVs");

/** @internal */
export const cellFragment = tgpu
    .fragmentFn({
        // `cellVsOut` (`pos` included), not `cellVaryings` alone — matches `cellVertex`'s `out` exactly
        // (module doc above). `input.pos` is otherwise unused here.
        in: cellVsOut,
        out: d.vec4f,
    })((input) => {
        "use gpu";
        // `fwidth` (inside sdfToSignedDistance's AA term) must run in uniform control flow — a branch
        // around it is invalid WGSL ("must only be called from uniform control flow") the instant
        // neighboring fragments disagree on whether they're inside the glyph's footprint, which two
        // adjacent fragments routinely do at the footprint's own edge. So every fragment samples +
        // computes AA unconditionally, over `t` clamped into the glyph's own uv rect (never
        // extrapolated past it — an unclamped `t` would sample a neighboring glyph's atlas tile), and
        // `has`/`inside` only zero the *result* via `select`, never the control flow the derivative runs
        // inside.
        const bgLinear = textSrgbToLinear(input.bg.xyz);
        const tc = std.clamp(input.t, d.vec2f(0, 0), d.vec2f(1, 1));
        const uv = std.mix(input.rect.xy, input.rect.zw, tc);
        const sdf = std.textureSample(drawLayout.$.atlasTex, drawLayout.$.atlasSamp, uv).x;
        const maxDim = std.max(input.gsize.x, input.gsize.y);
        const signedDist = sdfToSignedDistance(sdf, maxDim);
        const aa = std.length(std.fwidth(std.mul(input.t, input.gsize))) * 0.5;
        const rawAlpha = std.smoothstep(aa, -aa, signedDist);
        const inside =
            std.all(std.ge(input.t, d.vec2f(0, 0))) && std.all(std.le(input.t, d.vec2f(1, 1)));
        const alpha = std.select(d.f32(0), rawAlpha, input.has !== 0 && inside);
        const fgLinear = textSrgbToLinear(input.fg.xyz);
        return d.vec4f(std.mix(bgLinear, fgLinear, alpha), 1);
    })
    .$name("cellsFs");

let _pipeline: TgpuRenderPipeline<d.Vec4f> | null = null;
let _paramsBuffer: ReturnType<typeof allocParams> | null = null;

function allocParams() {
    return Compute.root.createBuffer(DrawParams).$usage("uniform").$name("cells-draw-params");
}

// one persistent params uniform, written (never recreated) per call — `drawCells` records into a
// caller-owned encoder it never submits itself (`CellsPlugin` shares one per-frame `Render.encoder`), so
// a fresh per-call buffer destroyed synchronously right after recording is destroyed before *this*
// recording's own submit ever executes (the exact "Buffer used in submit while destroyed"
// GPUValidationError `select.ts`'s `paramsBuffer` docblock names — measured here too). Shares that
// function's single-camera/stable-dims scope limit.
function paramsBuffer() {
    if (!_paramsBuffer) _paramsBuffer = allocParams();
    return _paramsBuffer;
}

/** the draw pipeline, over `Render.format` (the offscreen's HDR format — `view.framebuffer` targets
 *  match it exactly, `standard/render/view.ts`'s `offscreen()`). Memoized; a re-adopted device needs a
 *  fresh one ({@link resetDrawPipeline}). @internal */
export function drawPipeline(): TgpuRenderPipeline<d.Vec4f> {
    if (_pipeline) return _pipeline;
    _pipeline = Compute.root
        .createRenderPipeline({
            vertex: cellVertex,
            fragment: cellFragment,
            targets: { format: Render.format },
            primitive: { topology: "triangle-list", cullMode: "none" },
        })
        .$name("cells-draw");
    return _pipeline;
}

/** drop the memoized draw pipeline + params buffer (`grid.ts`'s `resetPipeline` shape). @internal */
export function resetDrawPipeline(): void {
    _pipeline = null;
    _paramsBuffer?.destroy();
    _paramsBuffer = null;
}

/**
 * draw every cell of `cellsBuffer` into `target` (a RENDER_ATTACHMENT-usable view matching
 * `Render.format` — `view.framebuffer` in production) as one instanced draw. `loadOp: "clear"`: the
 * geometry now tiles the view exactly (every cell's quad always covers its whole box, `cellVertex`'s own
 * docblock), so every pixel the pass writes is a real cell-authored value and nothing before this pass
 * needs to survive it — `"load"` was the s3r item 9 defect (`specs/shallot-tui.md`): it composited the
 * grid *over* the rendered scene, so a cols/rows pair that doesn't evenly divide the view (or any other
 * rasterizer-level gap) left the scene showing through. `"clear"` makes that impossible by construction
 * rather than by exact tiling alone — a gap now clears to black instead of leaking whatever rendered
 * before this pass.
 *
 * @example drawCells(encoder, view.framebuffer, cellsBuffer, glyphUv, glyphSize, atlas, sampler, 80, 24, view.width, view.height);
 */
export function drawCells(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    cellsBuffer: CellBuffer,
    glyphUv: GlyphUvBuffer,
    glyphSize: GlyphSizeBuffer,
    atlasTex: GPUTextureView,
    atlasSamp: GPUSampler,
    cols: number,
    rows: number,
    viewW: number,
    viewH: number,
): void {
    const params = paramsBuffer();
    params.write({ cols, rows, viewW, viewH });
    const group = Compute.root.createBindGroup(drawLayout, {
        params,
        cells: cellsBuffer,
        glyphUv,
        glyphSize,
        atlasSamp,
        atlasTex,
    });
    const pass = encoder.beginRenderPass({
        label: "cells-draw",
        timestampWrites: Compute.span?.("cells:draw"),
        colorAttachments: [
            {
                view: target,
                loadOp: "clear",
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                storeOp: "store",
            },
        ],
    });
    drawPipeline()
        .with(group)
        .with(pass)
        .draw(6, cols * rows);
    pass.end();
}

/** the emitted draw-pass WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function drawWgsl(): string {
    return tgpu.resolve([cellVertex, cellFragment], { names: "strict" });
}
