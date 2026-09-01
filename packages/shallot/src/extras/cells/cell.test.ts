import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import { vec4f } from "typegpu/data";
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
// independently against the raw readback bytes), bit-identical. gpu.md rule 6's "lattice drift between a
// CPU packer and a GPU unpacker" is the failure class this proves absent for this module's own codec.
describe("packCell / unpackCell round-trip (bit-identical)", () => {
    // byte-exact fractions (k/255) so packUnorm4x8's rounding introduces no error to round-trip through
    const byte = (k: number) => k / 255;

    function packAndDecode(
        glyph: number,
        fg: [number, number, number, number],
        bg: [number, number, number, number],
    ) {
        const packed = packCell(
            glyph,
            vec4f(byte(fg[0]), byte(fg[1]), byte(fg[2]), byte(fg[3])),
            vec4f(byte(bg[0]), byte(bg[1]), byte(bg[2]), byte(bg[3])),
        );
        const buf = new ArrayBuffer(CELL_BYTES);
        new Uint32Array(buf).set([packed.x, packed.y, packed.z]);
        return unpackCell(buf, 0);
    }

    test("round-trips glyph + both colors at representative byte values", () => {
        const decoded = packAndDecode(42, [255, 0, 128, 255], [0, 255, 64, 200]);
        expect(decoded.glyph).toBe(42);
        expect(decoded.fg).toEqual([255, 0, 128, 255]);
        expect(decoded.bg).toEqual([0, 255, 64, 200]);
    });

    test("round-trips every byte value 0..255 on a single channel", () => {
        for (let k = 0; k <= 255; k++) {
            const decoded = packAndDecode(0, [k, 0, 0, 255], [0, 0, 0, 255]);
            expect(decoded.fg[0]).toBe(k);
        }
    });

    test("out-of-range channels clamp to [0, 255] on pack, and unpack reads the clamp", () => {
        const packed = packCell(0, vec4f(2, -1, 0, 0), vec4f(0, 0, 0, 0));
        const buf = new ArrayBuffer(CELL_BYTES);
        new Uint32Array(buf).set([packed.x, packed.y, packed.z]);
        expect(unpackCell(buf, 0).fg).toEqual([255, 0, 0, 0]);
    });

    test("decodes the right cell at a nonzero index — stride discipline", () => {
        const buf = new ArrayBuffer(CELL_BYTES * 2);
        const words = new Uint32Array(buf);
        const cell0 = packCell(1, vec4f(byte(10), 0, 0, 1), vec4f(0, 0, 0, 1));
        const cell1 = packCell(2, vec4f(byte(20), 0, 0, 1), vec4f(0, 0, 0, 1));
        words.set([cell0.x, cell0.y, cell0.z], 0);
        words.set([cell1.x, cell1.y, cell1.z], CELL_U32S);

        expect(unpackCell(buf, 0).glyph).toBe(1);
        expect(unpackCell(buf, 0).fg[0]).toBe(10);
        expect(unpackCell(buf, 1).glyph).toBe(2);
        expect(unpackCell(buf, 1).fg[0]).toBe(20);
    });

    test("a swapped fg/bg word order would be caught — the two channels don't decode identically", () => {
        const decoded = packAndDecode(0, [10, 20, 30, 255], [40, 50, 60, 255]);
        expect(decoded.fg).not.toEqual(decoded.bg);
        expect(decoded.fg).toEqual([10, 20, 30, 255]);
        expect(decoded.bg).toEqual([40, 50, 60, 255]);
    });
});
