import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import { vec4f } from "typegpu/data";
import { linearToSrgb1 } from "../../engine/utils/core";
import { CELL_AT, CELL_BYTES, CELL_U32S, Cell, packCell, unpackCell } from "./cell";

describe("Cell layout", () => {
    test("byte size + lane offsets match field declaration order", () => {
        expect(CELL_BYTES).toBe(12);
        expect(CELL_U32S).toBe(3);
        expect(CELL_AT).toEqual({ glyph: 0, fg: 1, bg: 2 });
    });

    test("resolves to a WGSL struct with the three declared u32 fields, plus packCell's fn", () => {
        const wgsl = tgpu.resolve([Cell, packCell], { names: "strict" });
        expect(wgsl).toContain("struct Cell");
        expect(wgsl).toContain("glyph: u32");
        expect(wgsl).toContain("fg: u32");
        expect(wgsl).toContain("bg: u32");
        expect(wgsl).toContain("fn packCell(glyph: u32, fg: vec4f, bg: vec4f) -> vec3u {");
    });
});

// Criterion 3 (shallot-tui S1): "cell packing round-trips GPU to CPU" — a differential between the
// GPU-side pack (`packCell`, a TGSL function `bun test` calls directly on the CPU — the same source a
// compute kernel resolves, `grid.ts`'s fill pass) and the CPU-side unpack (`unpackCell`, written
// independently against the raw readback bytes), bit-identical. "Lattice drift between a CPU packer and
// a GPU unpacker" (`engine/utils/encode.ts`'s codec-boundary comment) is the failure class this proves
// absent for this module's own codec — the addressing/wiring half of it. `packUnorm4x8`'s CPU arm rounds
// half-up (`Math.round`) where the real WGSL `pack4x8unorm` intrinsic rounds half-to-even at an exact
// lattice midpoint (`engine/utils/tgsl.ts`); this device-free tier calls the CPU arm on both sides of the
// pack/unpack seam and so cannot observe that residual CPU↔GPU divergence — the `cells` gym scenario
// (`bun bench --scenario cells`) is where the real device dispatch is differentiated against this same
// CPU reference (`testing.md`'s tier split: a default-suite verdict must not depend on device execution).
describe("packCell / unpackCell round-trip (bit-identical)", () => {
    // `packUnorm4x8`'s CPU arm rounds its f64 input to f32 first (`engine/utils/tgsl.ts`'s own `unorm8`
    // — a vec4f lane stores f32, so quantizing an f64 puts a rare halfway case on the other lattice point
    // from the shader); this mirrors only that float32 truncation, never the sRGB transfer itself.
    const clamp01 = (v: number) => Math.max(0, Math.min(1, Math.fround(v)));
    // The byte `unpackCell` should read back for a given linear rgb input — `linearToSrgb1` imported
    // (not reimplemented) from the same module `packCell` calls through `packLdrColor`, so this fixture
    // proves *wiring* (arg order into `packLdrColor`, word order into the three raw words, `unpackCell`'s
    // stride/offset), not the transfer function's own arithmetic (`encode.test.ts`'s `packLdrColor`
    // describe block owns that).
    const srgbByte = (linear: number) => Math.round(clamp01(linearToSrgb1(linear)) * 255);
    // alpha bypasses the sRGB transfer (`packLdrColor`, `engine/utils/encode.ts`) — passed straight into
    // `packUnorm4x8`, so it stays byte-exact.
    const alphaByte = (linear: number) => Math.round(clamp01(linear) * 255);
    const expectedRgba = (
        c: [number, number, number, number],
    ): [number, number, number, number] => [
        srgbByte(c[0]),
        srgbByte(c[1]),
        srgbByte(c[2]),
        alphaByte(c[3]),
    ];

    function packAndDecode(
        glyph: number,
        fg: [number, number, number, number],
        bg: [number, number, number, number],
    ) {
        const packed = packCell(
            glyph,
            vec4f(fg[0], fg[1], fg[2], fg[3]),
            vec4f(bg[0], bg[1], bg[2], bg[3]),
        );
        const buf = new ArrayBuffer(CELL_BYTES);
        new Uint32Array(buf).set([packed.x, packed.y, packed.z]);
        return unpackCell(buf, 0);
    }

    test("round-trips glyph + both colors at representative linear inputs", () => {
        const fg: [number, number, number, number] = [1, 0, 0.5, 1];
        const bg: [number, number, number, number] = [0, 1, 0.25, 0.8];
        const decoded = packAndDecode(42, fg, bg);
        expect(decoded.glyph).toBe(42);
        expect(decoded.fg).toEqual(expectedRgba(fg));
        expect(decoded.bg).toEqual(expectedRgba(bg));
    });

    // N2: the prior fixture used only k/255 multiples, landing exactly on the unorm8 lattice with no
    // rounding ambiguity — proving nothing about a fraction that actually needs rounding. `grid.ts`'s own
    // fill kernel produces exactly the class widened to here: `x / cols`, `y / rows`, arbitrary fractions.
    // None of these land on an exact lattice midpoint once the rgb channels go through `linearToSrgb1`
    // first (1/3 → ~156.2, 2/7 → ~145.6, 5/9 → ~196.6, 11/13 → ~236.9, 0.5/255 → ~6.5, 254.5/255 → ~254.8
    // — none within 0.1 of `k + 0.5`), so this arm proves wiring (arg order into `packLdrColor`, word
    // order into the three raw words, `unpackCell`'s stride/offset) on arbitrary fractions, not a
    // rounding-divergence check — and it couldn't be one regardless: `expectedRgba`'s own `Math.round`
    // (above) and `packCell`'s CPU arm round the same way, so a half-up/half-to-even divergence can't red
    // here at any input. The real-device rounding differential lives in the `cells` gym scenario
    // (`bun bench --scenario cells`), which sweeps fg/bg alpha across genuine lattice midpoints
    // (`grid.ts`'s fill kernel; `cell.ts`'s own module doc names the seam).
    test("round-trips arbitrary non-1/255-aligned fractions through the shared CPU rounding path", () => {
        for (const v of [1 / 3, 2 / 7, 5 / 9, 11 / 13, 0.5 / 255, 254.5 / 255]) {
            const decoded = packAndDecode(0, [v, v, v, v], [0, 0, 0, 1]);
            expect(decoded.fg).toEqual(expectedRgba([v, v, v, v]));
        }
    });

    test("alpha bypasses the sRGB transfer — stays byte-exact for every 0..255 value", () => {
        for (let k = 0; k <= 255; k++) {
            const decoded = packAndDecode(0, [0, 0, 0, k / 255], [0, 0, 0, 1]);
            expect(decoded.fg[3]).toBe(k);
        }
    });

    test("out-of-range channels clamp to [0, 255] on pack, and unpack reads the clamp", () => {
        const fg: [number, number, number, number] = [2, -1, 0, 0];
        const decoded = packAndDecode(0, fg, [0, 0, 0, 0]);
        expect(decoded.fg).toEqual(expectedRgba(fg));
    });

    test("decodes the right cell at a nonzero index — stride discipline", () => {
        const buf = new ArrayBuffer(CELL_BYTES * 2);
        const words = new Uint32Array(buf);
        const cell0 = packCell(1, vec4f(10 / 255, 0, 0, 1), vec4f(0, 0, 0, 1));
        const cell1 = packCell(2, vec4f(20 / 255, 0, 0, 1), vec4f(0, 0, 0, 1));
        words.set([cell0.x, cell0.y, cell0.z], 0);
        words.set([cell1.x, cell1.y, cell1.z], CELL_U32S);

        expect(unpackCell(buf, 0).glyph).toBe(1);
        expect(unpackCell(buf, 0).fg[0]).toBe(srgbByte(10 / 255));
        expect(unpackCell(buf, 1).glyph).toBe(2);
        expect(unpackCell(buf, 1).fg[0]).toBe(srgbByte(20 / 255));
    });

    test("a swapped fg/bg word order would be caught — the two channels don't decode identically", () => {
        const fg: [number, number, number, number] = [10 / 255, 20 / 255, 30 / 255, 1];
        const bg: [number, number, number, number] = [40 / 255, 50 / 255, 60 / 255, 1];
        const decoded = packAndDecode(0, fg, bg);
        expect(decoded.fg).not.toEqual(decoded.bg);
        expect(decoded.fg).toEqual(expectedRgba(fg));
        expect(decoded.bg).toEqual(expectedRgba(bg));
    });
});
