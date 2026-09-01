// Cell — the GPU cell layout the terminal grid producer writes and both sinks (S3's web instanced draw,
// S4's terminal ANSI encoder) read: one glyph atlas index plus two packed LDR colors (fg, bg) — sRGB-
// encoded rgb + linear alpha, `pack4x8unorm`-packed exactly like every other packed color slab in this
// engine (`packLdrColor`, `engine/utils/encode.ts`). Locked: ANSI SGR truecolor bytes are sRGB by
// definition, so a cell forwarded byte-for-byte to S4's terminal encoder is only correct if these bytes
// already are sRGB — a linear pack would forward linear values as if they were sRGB and read visibly
// wrong on every non-mid-grey color. The CPU↔GPU boundary a compute pass and a CPU readback share, a
// sibling of the glyph instance layout (`extras/text/glyph.ts`) simplified to what a monospace cell
// needs — no world position, no atlas UV, no entity id, those ride the sink's own draw (`grid.ts`'s
// compute pass + S3's instanced quad). This file owns the layout alone, so a device-free test can pin it
// with no GPU.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import { packLdrColor } from "../../engine/utils/core";

/**
 * one terminal cell: a glyph atlas index plus two packed LDR colors (foreground, background) — sRGB rgb,
 * linear alpha (see the module doc for why sRGB is the locked space). The CPU↔GPU layout of the cell
 * grid buffer — every reader ({@link CELL_AT}, the web instanced draw, the terminal ANSI encoder)
 * derives its stride and offsets from this schema, never a duplicated one.
 */
export const Cell = d.struct({
    glyph: d.u32,
    fg: d.u32,
    bg: d.u32,
});

/** one cell's byte size, from {@link Cell} — the grid buffer's stride. */
export const CELL_BYTES = d.sizeOf(Cell);
/** the same stride in 4-byte lanes, the readback decoder's flat typed-array step. */
export const CELL_U32S = CELL_BYTES / 4;

// the u32-lane offset of one Cell field, from the schema — mirrors extras/text/glyph.ts's `GLYPH_AT`
const at = <T extends d.BaseData>(schema: T, field: (p: d.Infer<T>) => unknown) =>
    d.memoryLayoutOf(schema, field).offset / 4;

/** each {@link Cell} field's u32-lane index within one cell — {@link unpackCell}'s offsets. */
export const CELL_AT = {
    glyph: at(Cell, (c) => c.glyph),
    fg: at(Cell, (c) => c.fg),
    bg: at(Cell, (c) => c.bg),
} as const;

/**
 * the compute-pass pack: a glyph index plus two **linear-space** rgba colors (0..1, the shader's own
 * lighting math space) into the three raw words a {@link Cell} stores, `(glyph, fg, bg)`. Colors pack
 * through {@link packLdrColor} — the engine's own LDR color codec (sRGB-encode rgb, pass alpha through
 * linear, `pack4x8unorm`-pack) — the same codec every other packed-color slab in this engine uses, so a
 * cell producer shares its color space and its rounding/clamping behavior with the rest of the renderer
 * instead of inventing a second one, and the stored bytes are already sRGB — the space ANSI SGR
 * truecolor needs, with no conversion at S4's encoder. One TGSL source: `bun test` calls it directly on
 * the CPU, a compute kernel resolves the identical body (`grid.ts`'s fill pass). `packLdrColor`'s CPU arm
 * rounds half-up (`Math.round`) where the real WGSL `pack4x8unorm` intrinsic rounds half-to-even at an
 * exact lattice midpoint — a residual CPU↔GPU seam this device-free tier cannot observe (`cell.test.ts`
 * proves the CPU-side wiring only); the real-device dispatch differential lives in the `cells` gym
 * scenario (`bun bench --scenario cells`), the only tier allowed to depend on device execution
 * (`testing.md`).
 *
 * @example const cell = packCell(glyphIndex, vec4f(1, 0, 0, 1), vec4f(0, 0, 0, 1)); // red on black
 */
export const packCell = tgpu.fn(
    [d.u32, d.vec4f, d.vec4f],
    d.vec3u,
)((glyph, fg, bg) => {
    "use gpu";
    return d.vec3u(glyph, packLdrColor(fg.xyz, fg.w), packLdrColor(bg.xyz, bg.w));
});

/** one decoded {@link Cell}: the glyph index plus fg/bg as 0..255 rgba byte quads — what
 *  {@link unpackCell} decodes into, so a caller compares against {@link packCell}'s inputs at byte
 *  precision rather than reconstructing floats and eating a second rounding step. */
export interface DecodedCell {
    glyph: number;
    fg: [number, number, number, number];
    bg: [number, number, number, number];
}

function unpackBytes(word: number): [number, number, number, number] {
    return [word & 0xff, (word >>> 8) & 0xff, (word >>> 16) & 0xff, (word >>> 24) & 0xff];
}

/**
 * the CPU-side counterpart of {@link packCell}: read one cell out of a raw readback buffer (the
 * `Mirror` snapshot shape — opaque bytes, `standard/mirror`) at `index`, decoding the packed fg/bg
 * words to 0..255 rgba byte quads — already sRGB-encoded rgb + linear alpha (the module doc), so an
 * ANSI SGR truecolor encoder reads these bytes directly. Written independently of {@link packCell} on
 * purpose — their agreement is what a differential test proves, rather than assuming a shared
 * implementation can't drift ("Lattice drift between a CPU packer and a GPU unpacker",
 * `engine/utils/encode.ts`'s own codec-boundary comment).
 *
 * @example const cell = unpackCell(mirror.snapshot!.bytes, 0);
 */
export function unpackCell(bytes: ArrayBuffer, index: number): DecodedCell {
    const view = new Uint32Array(bytes, index * CELL_BYTES, CELL_U32S);
    return {
        glyph: view[CELL_AT.glyph],
        fg: unpackBytes(view[CELL_AT.fg]),
        bg: unpackBytes(view[CELL_AT.bg]),
    };
}
