// The web sink: an instanced draw of `cols * rows` monospace quads against the shared SDF glyph atlas
// (`text/core`), no readback — the Locked decision's "same pipeline with the expensive tail deleted"
// (`specs/shallot-tui.md`). A simplification of `extras/text`'s own instanced glyph draw: no world
// transform, no per-entity eid lookup, no per-string layout pass — every cell is a fixed monospace box
// whose position derives purely from its instance index, and its glyph quad is shrunk + centered within
// that box to the glyph's own measured em-normalized footprint (`glyphLocalCorner` below, against
// `glyphs.ts`'s size table) rather than stretched to fill it — size-proportional placement, the same
// principle `extras/text`'s own `layoutText` applies per glyph (`ramp.ts`'s module doc names the
// contract this reads). No vertex/index buffer either: the quad's six corners come from
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

/**
 * map a quad-local corner (0..1, spanning the full glyph tile) to its footprint position within the
 * cell: shrunk and centered to `size` (the glyph's own em-normalized width/height, `glyphs.ts`'s
 * `glyphSizeTable`) rather than filling the whole cell regardless of the glyph's true extent. This is
 * the point of use the Locked decision's coverage-ordered ramp (`ramp.ts`) needs preserved — without it,
 * every glyph's own tightly-cropped SDF tile stretches across the same cell footprint, and the ramp's
 * monotone-coverage ordering renders indistinguishable (`specs/shallot-tui.md`'s s3r item 8). One TGSL
 * source, dual-mode like `cell.ts`'s `packCell`: `bun test` calls it directly on the CPU, `cellVertex`
 * resolves the identical body. `size = (1, 1)` (`glyphs.ts`'s `MISSING_GLYPH_SIZE`, a font-absent
 * glyph's sentinel) leaves the corner unchanged — the full-cell footprint — which is inert there since
 * the fragment stage never samples ink for a missing glyph (`has`'s zero-area-uv gate).
 *
 * @example glyphLocalCorner(vec2f(0, 0), vec2f(0.5, 0.5)); // vec2f(0.25, 0.25) — centered, half-size
 */
export const glyphLocalCorner = tgpu.fn(
    [d.vec2f, d.vec2f],
    d.vec2f,
)((corner, size) => {
    "use gpu";
    const margin = std.mul(std.sub(d.vec2f(1, 1), size), 0.5);
    return std.add(margin, std.mul(corner, size));
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
    corner: d.vec2f,
    uv: d.vec2f,
    gsize: d.vec2f,
    fg: d.vec4f,
    bg: d.vec4f,
    has: d.interpolate("flat", d.u32),
};

/** @internal */
export const cellVertex = tgpu
    .vertexFn({
        in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
        out: { pos: d.builtin.position, ...cellVaryings },
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
        const local = glyphLocalCorner(corner, size);
        const px = (d.f32(col) + local.x) * cellW;
        const py = (d.f32(row) + local.y) * cellH;
        const ndcX = px / (drawLayout.$.params.viewW * 0.5) - 1;
        const ndcY = 1 - py / (drawLayout.$.params.viewH * 0.5);

        const uv = std.mix(rect.xy, rect.zw, corner);
        const fg = std.unpack4x8unorm(cell.fg);
        const bg = std.unpack4x8unorm(cell.bg);

        return {
            pos: d.vec4f(ndcX, ndcY, 0, 1),
            corner,
            uv,
            gsize: d.vec2f(cellW * size.x, cellH * size.y),
            fg,
            bg,
            has,
        };
    })
    .$name("cellsVs");

/** @internal */
export const cellFragment = tgpu
    .fragmentFn({
        in: cellVaryings,
        out: d.vec4f,
    })((input) => {
        "use gpu";
        // `fwidth` (inside sdfToSignedDistance's AA term) must run in uniform control flow — a branch on
        // `has` around it is invalid WGSL ("must only be called from uniform control flow") the instant
        // neighboring fragments in the same quad disagree on whether their cell has a glyph, which two
        // adjacent cells routinely do. So every fragment samples + computes AA unconditionally, and the
        // zero-area sentinel (`glyphs.ts`'s `MISSING_GLYPH_UV`) only zeroes the *result* via `select`,
        // never the control flow the derivative runs inside.
        const bgLinear = textSrgbToLinear(input.bg.xyz);
        const sdf = std.textureSample(drawLayout.$.atlasTex, drawLayout.$.atlasSamp, input.uv).x;
        const maxDim = std.max(input.gsize.x, input.gsize.y);
        const signedDist = sdfToSignedDistance(sdf, maxDim);
        const aa = std.length(std.fwidth(std.mul(input.corner, input.gsize))) * 0.5;
        const rawAlpha = std.smoothstep(aa, -aa, signedDist);
        const alpha = std.select(d.f32(0), rawAlpha, input.has !== 0);
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
 * `Render.format` — `view.framebuffer` in production) as one instanced draw, `loadOp: "load"` so a
 * cols/rows pair that doesn't evenly divide the view leaves its uncovered edge strip showing the
 * rendered scene rather than clearing to black.
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
        colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
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
