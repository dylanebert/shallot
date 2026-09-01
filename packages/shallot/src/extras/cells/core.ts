// Cells' `*/core` extension surface — for a custom pipeline / tooling / diagnostic consumer
// (`exports.md`'s Barrel rules), the way `gltf/core` / `skin/core` / `tween/core` publish their own
// substrate ahead of (or instead of) an author-facing plugin. No author-facing component or plugin
// exists yet — that lands once a real scene sampler exists (S3) — so today's public surface for this
// module *is* the substrate: the GPU cell layout contract (`Cell` + its packing) and the headless fill
// producer S1 ships. Not on the main `extras` barrel: none of this is a component, singleton, or
// registration function a game author calls (`exports.md`'s barrel-vs-`*/core` split), and `Cell` /
// `createCellGrid` are exactly the kind of generic name that subpath exists to avoid colliding on the
// bare barrel (`exports.md`'s Naming section). The `examples/gym` `cells` scenario is this subpath's
// first real consumer — it drives `createCellGrid` / `fillCellGrid` through a `Mirror` readback.
//
// Deliberately NOT re-exported here — every one of these is `@internal` in its defining file
// (`grid.ts`), meaning no external consumer needs it, only a sibling file within this same directory
// does, imported directly the way `extras/text`'s `Glyph` / `GLYPH_AT` / `resetPipelines` are (no
// subpath at all): `GridParams` and `gridLayout` (the fill pass's own uniform schema and bind group
// layout — a custom pipeline reuses the producer functions below, never the fill kernel's private
// wiring), `resetPipeline` (reachability proven device-free in `grid.test.ts`, alongside the hazard its
// own docblock names — see that file; no plugin owns a dispose lifecycle to call it from yet), and
// `gridWgsl` (the device-free structural test seam `grid.test.ts` already imports directly, mirroring
// `standard/render/cluster.ts`'s own internal `gridWgsl`, which stays off `render/core` for the same
// reason).
export {
    CELL_AT,
    CELL_BYTES,
    CELL_U32S,
    Cell,
    type DecodedCell,
    packCell,
    unpackCell,
} from "./cell";
export { type CellGrid, createCellGrid, fillCellGrid } from "./grid";
export {
    CELL_DIRECTIONAL_GLYPHS,
    CELL_FILL_GLYPHS,
    CELL_GLYPH_COUNT,
    cellGlyphChar,
    cellGlyphString,
} from "./ramp";
