// Text's `*/core` extension surface — the shared SDF glyph atlas substrate for a producer that needs it
// without the retained `Text` component + its layout/anchor machinery: atlas creation/warming, plus the
// SDF decode (`sdfToSignedDistance`) and packed-color decode (`textSrgbToLinear`) a consuming fragment
// stage evaluates the same way this module's own `fs` does. First consumer: `extras/cells`, the ASCII
// cell renderer's glyph atlas (`shallot-tui` spec's Locked decision: "renders directly on the GPU
// through the existing instanced SDF glyph atlas" — a simplification of this system, monospace and
// anchor-free, reusing the same atlas rather than building a second one). `check-imports.ts` forbids a deep
// cross-module import into `./atlas` / `./font` / `./glyph` directly, so this subpath is the sanctioned
// reuse seam — same shape as `skin/core` existing for glTF's PBR trio to compose against.
export {
    computeGlyphMetrics,
    createGlyphAtlas,
    disposeAtlases,
    ensureString,
    type GlyphAtlas,
    type GlyphMetrics,
} from "./atlas";
export { type Font, loadFont } from "./font";
export { sdfToSignedDistance, textSrgbToLinear } from "./glyph";
