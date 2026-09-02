// Cells' `*/core` extension surface — for a custom pipeline / tooling / diagnostic consumer
// (`exports.md`'s Barrel rules), the way `gltf/core` / `skin/core` / `tween/core` publish their own
// substrate ahead of (or instead of) an author-facing plugin. `CellsPlugin` (`./index.ts`) rides the main
// `extras` barrel; this subpath is the GPU cell layout contract underneath it: the packing (`Cell` + its
// byte/lane constants), the codec (`packCell` / `unpackCell`), the headless fill producer, the real
// content producer (`recordSelect` / `dispatchSelect`, structure-first glyph selection over a rendered
// scene), the glyph uv-rect + size table builders, the web sink's instanced draw, and the glyph ramp. Not on the
// main `extras` barrel itself: none of this is a component, singleton, or registration function a game
// author calls (`exports.md`'s barrel-vs-`*/core` split), and `Cell` / `createCellGrid` are exactly the
// kind of generic name that subpath exists to avoid colliding on the bare barrel (`exports.md`'s Naming
// section). The `examples/gym` `cells` scenario is this subpath's consumer — it drives every producer here
// through a `Mirror` readback, with no canvas of its own.
//
// Deliberately NOT re-exported here — every one of these is `@internal` in its defining file, meaning no
// external consumer needs it, only a sibling file within this same directory does, imported directly the
// way `extras/text`'s `Glyph` / `GLYPH_AT` / `resetPipelines` are (no subpath at all): `GridParams` and
// `gridLayout` (the fill pass's own uniform schema and bind group layout — a custom pipeline reuses the
// producer functions below, never the fill kernel's private wiring), `resetPipeline` /
// `resetSelectPipelines` / `resetDrawPipeline` (reachability proven device-free in each module's own
// test, alongside the hazard its docblock names — `CellsPlugin.dispose` is the only lifecycle owner), the
// bind-group layouts + kernel/shader factories each module's own `*Wgsl()` device-free test already
// imports directly (mirroring `standard/render/cluster.ts`'s own internal `gridWgsl`, which stays off
// `render/core` for the same reason), and the pure sub-pieces (`reinhard`, `luma`) a custom pipeline
// composing its own selection kernel would want individually, past `directionalGlyphIndex` below.
export {
    CELL_AT,
    CELL_BYTES,
    CELL_U32S,
    Cell,
    type DecodedCell,
    packCell,
    unpackCell,
} from "./cell";
export {
    cellFootprintPx,
    DrawParams as CellsDrawParams,
    drawCells,
    drawPipeline,
    glyphFootprintT,
    resetDrawPipeline,
} from "./draw";
export {
    buildGlyphSizeTable,
    buildGlyphUvTable,
    type GlyphSizeBuffer,
    type GlyphUvBuffer,
    glyphSizeRect,
    glyphSizeTable,
    glyphUvRect,
    glyphUvTable,
    MISSING_GLYPH_SIZE,
    MISSING_GLYPH_UV,
} from "./glyphs";
export { type CellGrid, createCellGrid, fillCellGrid } from "./grid";
export {
    CELL_DIRECTIONAL_GLYPHS,
    CELL_FILL_GLYPHS,
    CELL_GLYPH_COUNT,
    cellGlyphChar,
    cellGlyphString,
} from "./ramp";
export {
    BG_MATCH_EPSILON,
    directionalGlyphIndex,
    dispatchSelect,
    EDGE_MAGNITUDE_THRESHOLD,
    FACADE_BAND_LUMAS,
    FACADE_INK_CEILING,
    FACADE_INK_FLOOR,
    FACADE_PIXEL_LUMA_THRESHOLD,
    fillIndexForLuma,
    NO_BACKGROUND,
    recordSelect,
    resetSelectPipelines,
} from "./select";
