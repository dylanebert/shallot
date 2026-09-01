// Cells — the ASCII-native render target's author-facing surface (`shallot-tui` spec). Add `CellsPlugin`
// to a project alongside a renderer (`RenderPlugin`, e.g. via `SearPlugin`) and every camera's scene
// composites through a character-cell grid instead of raw pixels: after the scene renders, a compute
// pass samples the camera's offscreen scene color into a structure-first-selected cell grid (`select.ts`
// — edge-tangent directional glyphs where a strong edge exists, a coverage-ordered fill glyph elsewhere,
// `ramp.ts`'s Locked-decision contract), and an instanced draw (`draw.ts`) renders that grid back into
// the same offscreen target — same pipeline, one pass earlier than glaze's tonemap + present. No author
// component: every camera composites through cells the same way, the way every camera composites through
// glaze's postfx chain by default (`standard/glaze`'s own zero-config shape). A fixed 80×24 grid — the
// terminal default the spec's own two-seat measurement table centers on — no per-camera resize surface
// yet; that's an author-facing knob for a later unit, not this one's contract.

import type { Plugin, State, System } from "../../engine";
import { Compute } from "../../engine";
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
import { buildGlyphUvTable, type GlyphUvBuffer } from "./glyphs";
import { type CellGrid, createCellGrid } from "./grid";
import { CELL_GLYPH_COUNT, cellGlyphString } from "./ramp";
import { recordSelect, resetSelectPipelines } from "./select";

/** the fixed cell-grid shape (module doc above). @internal */
export const COLS = 80;
/** @internal */
export const ROWS = 24;

// matches `extras/text`'s own zero-config default face (Inter) — a cells scene needs no font
// registration of its own; an author-facing override is a later unit's surface, not this one's
const DEFAULT_FONT =
    "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf";

let _atlas: GlyphAtlas | null = null;
let _glyphUv: GlyphUvBuffer | null = null;
let _sampler: GPUSampler | null = null;
const _grids = new Map<number, CellGrid>();

function gridFor(eid: number): CellGrid {
    const cached = _grids.get(eid);
    if (cached) return cached;
    const grid = createCellGrid(COLS, ROWS, CELL_GLYPH_COUNT);
    _grids.set(eid, grid);
    return grid;
}

/**
 * for every camera with a rendered scene, select this frame's cell grid from its offscreen color, then
 * draw that grid back into the same target — one compute recording, one render recording, both against
 * the shared `Render.encoder` (`extras/outline`'s post-color-seam shape). A canvas-less view (no
 * `framebuffer`) is skipped, same as glaze.
 */
const CellsSystem: System = {
    name: "cells",
    group: "draw",
    after: [ColorSystem, OverlaySystem],
    before: [GlazeSystem],
    update(state: State) {
        const encoder = Render.encoder;
        if (!encoder || !Compute.device || !_atlas || !_glyphUv || !_sampler) return;
        for (const eid of state.query([Camera])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            const grid = gridFor(eid);
            recordSelect(
                encoder,
                grid.buffer,
                COLS,
                ROWS,
                view.framebuffer,
                view.width,
                view.height,
            );
            drawCells(
                encoder,
                view.framebuffer,
                grid.buffer,
                _glyphUv,
                _atlas.textureView,
                _sampler,
                COLS,
                ROWS,
                view.width,
                view.height,
            );
        }
    },
};

/**
 * the ASCII-native render target: after `RenderPlugin` renders each camera's scene, `CellsSystem`
 * selects + draws a character-cell grid over it, before glaze composites the result to the swapchain.
 */
export const CellsPlugin: Plugin = {
    name: "Cells",
    systems: [CellsSystem],
    dependencies: [RenderPlugin],

    async warm() {
        if (!Compute.device) return;
        const device = Compute.device;
        const font = await loadFont(DEFAULT_FONT);
        _atlas = createGlyphAtlas(device, font);
        ensureString(_atlas, cellGlyphString());
        _glyphUv = buildGlyphUvTable(_atlas);
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
        _sampler = null;
        for (const grid of _grids.values()) grid.buffer.destroy();
        _grids.clear();
        resetSelectPipelines();
        resetDrawPipeline();
    },
};
