// Cells — the ASCII-native render target's author-facing surface (`shallot-tui` spec). Add `CellsPlugin`
// to a project alongside a renderer (`RenderPlugin`, e.g. via `SearPlugin`) and every camera's scene
// composites through a character-cell grid instead of raw pixels: after the scene renders, a compute
// pass samples the camera's offscreen scene color into a structure-first-selected cell grid (`select.ts`
// — edge-tangent directional glyphs where a strong edge exists, a coverage-ordered fill glyph elsewhere,
// `ramp.ts`'s Locked-decision contract), and an instanced draw (`draw.ts`) renders that grid back into
// the same offscreen target — same pipeline, one pass earlier than glaze's tonemap + present. No author
// component: every camera composites through cells the same way, the way every camera composites through
// glaze's postfx chain by default (`standard/glaze`'s own zero-config shape). Each sink sizes the grid
// from its output surface; the web path uses canvas CSS dimensions while the framebuffer stays in
// device pixels.

import type { Plugin, State, System } from "../../engine";
import { Compute, unpackColor } from "../../engine";
import { GlazeSystem } from "../../standard/glaze";
import { Camera, RenderPlugin } from "../../standard/render";
import { OverlaySystem, Render, Views } from "../../standard/render/core";
import { ColorSystem } from "../../standard/sear/core";
import {
    createGlyphAtlas,
    disposeAtlases,
    ensureString,
    type GlyphAtlas,
    loadFont,
} from "../text/core";
import { drawCells, resetDrawPipeline } from "./draw";
import {
    buildGlyphSizeTable,
    buildGlyphUvTable,
    type GlyphSizeBuffer,
    type GlyphUvBuffer,
} from "./glyphs";
import { type CellGrid, createCellGrid, deriveCellGridSize } from "./grid";
import { CELL_GLYPH_COUNT, cellGlyphString } from "./ramp";
import { recordSelect, resetSelectPipelines } from "./select";

/** legacy fixture dimensions for headless producer examples; the live sink derives its own shape. @internal */
export const COLS = 80;
/** @internal */
export const ROWS = 24;

/** the default cells face: the house monospace with a uniform advance for every cell. */
export const DEFAULT_FONT =
    "https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKxjPQ.ttf";

let _atlas: GlyphAtlas | null = null;
let _glyphUv: GlyphUvBuffer | null = null;
let _glyphSize: GlyphSizeBuffer | null = null;
let _sampler: GPUSampler | null = null;
const _grids = new Map<number, CellGrid>();

function gridFor(eid: number, cols: number, rows: number): CellGrid {
    const cached = _grids.get(eid);
    if (cached?.cols === cols && cached.rows === rows) return cached;
    cached?.buffer.destroy();
    const grid = createCellGrid(cols, rows, CELL_GLYPH_COUNT);
    _grids.set(eid, grid);
    return grid;
}

/** the live cell grid `CellsSystem` selected and drew for camera `eid` this frame, or `undefined` before
 *  its first frame — a diagnostic/tooling seam (`scripts/dump-cells-ascii.ts`'s tier-0 text dump, which
 *  needs the real plugin-driven grid rather than reimplementing `recordSelect`'s own call against a
 *  camera it drives by hand) rather than a game-author API; not re-exported on the main `extras` barrel,
 *  same as everything else module-private to this directory (`core.ts`'s own docblock). */
export function cellsGridFor(eid: number): CellGrid | undefined {
    return _grids.get(eid);
}

// warn-once guard for the single-camera scope below — module state, not per-frame, so a scene that
// carries a second camera doesn't spam the console every frame.
let _warnedMultiCamera = false;

/**
 * for every camera with a rendered scene, select this frame's cell grid from its offscreen color, then
 * draw that grid back into the same target — one compute recording, one render recording, both against
 * the shared `Render.encoder` (`extras/outline`'s post-color-seam shape). A canvas-less view (no
 * `framebuffer`) is skipped, same as glaze.
 *
 * Single-camera per frame, enforced here rather than only disclosed: `select.ts`'s `avgBuffer`/
 * `paramsBuffer` and `draw.ts`'s `paramsBuffer` are each one persistent GPU resource, written (never
 * reallocated) per call and shared across every caller in the frame — correct only when every caller
 * that frame requests the same `cols`/`rows`/`viewW`/`viewH`. A second camera with a different view size
 * would silently corrupt both the first camera's still-unsubmitted recording and its own (`select.ts`'s
 * own docblock names the hazard; this is the demonstrated instance of it, `specs/shallot-tui.md`'s s3r
 * item 5). Per-camera buffer keying is the eventual fix; until then, draw the first camera with a
 * rendered scene and warn once rather than composite garbage for every camera after it.
 */
const CellsSystem: System = {
    name: "cells",
    group: "draw",
    after: [ColorSystem, OverlaySystem],
    before: [GlazeSystem],
    update(state: State) {
        const encoder = Render.encoder;
        if (!encoder || !Compute.device || !_atlas || !_glyphUv || !_glyphSize || !_sampler) return;
        let drawnEid: number | null = null;
        for (const eid of state.query([Camera])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            if (drawnEid !== null) {
                if (!_warnedMultiCamera) {
                    _warnedMultiCamera = true;
                    console.warn(
                        `shallot: Cells drew camera ${drawnEid} this frame; camera ${eid} shares its ` +
                            "one persistent select/draw buffer set and would silently corrupt both — " +
                            "skipping every camera after the first with a rendered scene until Cells " +
                            "supports per-camera buffers.",
                    );
                }
                break;
            }
            drawnEid = eid;
            const cellGeometry = view.canvas as
                | (HTMLCanvasElement & {
                      cellWidth?: number;
                      cellHeight?: number;
                      cellCols?: number;
                      cellRows?: number;
                  })
                | null;
            const terminalGeometry =
                cellGeometry?.cellWidth !== undefined && cellGeometry.cellHeight !== undefined;
            const rect = terminalGeometry ? null : cellGeometry?.getBoundingClientRect();
            const { cols, rows } = deriveCellGridSize(
                rect?.width ?? view.width,
                rect?.height ?? view.height,
                cellGeometry?.cellWidth,
                cellGeometry?.cellHeight,
            );
            if (cellGeometry) {
                cellGeometry.cellCols = cols;
                cellGeometry.cellRows = rows;
            }
            const grid = gridFor(eid, cols, rows);
            // the camera's own empty-background reference, raw linear — recordSelect tonemaps it the
            // same way it tonemaps every scene sample before comparing, so a cell whose source region
            // is untouched clear color selects the blank fill glyph instead of whatever non-zero index
            // its clear luma would otherwise round to (`select.ts`'s BG_MATCH_EPSILON module doc).
            const bg = unpackColor(Camera.clearColor.get(eid));
            recordSelect(
                encoder,
                grid.buffer,
                cols,
                rows,
                view.framebuffer,
                view.width,
                view.height,
                [bg.r, bg.g, bg.b],
            );
            drawCells(
                encoder,
                view.framebuffer,
                grid.buffer,
                _glyphUv,
                _glyphSize,
                _atlas.textureView,
                _sampler,
                cols,
                rows,
                view.width,
                view.height,
            );
        }
    },
};

/**
 * create the ASCII-native render target with an optional author-supplied monospace face. A process supports
 * one cells face; use one `fontUrl` for the process. After
 * `RenderPlugin` renders each camera's scene, the returned plugin selects + draws a character-cell grid
 * over it, before glaze composites the result to the swapchain.
 *
 * @example
 * ```
 * cells("/fonts/my-monospace.ttf");
 * ```
 */
export function cells(fontUrl = DEFAULT_FONT): Plugin {
    return {
        name: "Cells",
        systems: [CellsSystem],
        dependencies: [RenderPlugin],

        async warm() {
            if (!Compute.device) return;
            const device = Compute.device;
            const font = await loadFont(fontUrl);
            _atlas = createGlyphAtlas(device, font);
            ensureString(_atlas, cellGlyphString());
            _glyphUv = buildGlyphUvTable(_atlas);
            _glyphSize = buildGlyphSizeTable(_atlas);
            _sampler = device.createSampler({
                label: "cells",
                magFilter: "linear",
                minFilter: "linear",
            });
        },

        dispose() {
            if (_atlas) disposeAtlases([_atlas]);
            _atlas = null;
            _glyphUv?.destroy();
            _glyphUv = null;
            _glyphSize?.destroy();
            _glyphSize = null;
            _sampler = null;
            for (const grid of _grids.values()) grid.buffer.destroy();
            _grids.clear();
            resetSelectPipelines();
            resetDrawPipeline();
        },
    };
}

/** the ASCII-native render target using the default JetBrains Mono face. */
export const CellsPlugin: Plugin = cells();
