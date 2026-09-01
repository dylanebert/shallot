// Cells — the ASCII-native render target's GPU cell-grid contract (`shallot-tui` spec, S1): the layout a
// compute pass writes and both sinks read (S3's web instanced draw against the shared glyph atlas, S4's
// terminal ANSI encoder), plus the headless fill producer this stage ships to prove the round trip. A
// simplification of `extras/text`'s SDF glyph pipeline reusing its atlas — monospace, no layout pass, no
// per-string rebuild signature, no anchor math. No author-facing component or plugin yet: that lands once
// a real scene sampler exists (S3), so this module's public surface today is the contract itself.

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
    type CellGrid,
    createCellGrid,
    fillCellGrid,
    GridParams,
    gridLayout,
    resetPipeline,
} from "./grid";
