import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { vec2f, vec4f } from "typegpu/data";
import * as std from "typegpu/std";
import { idiv } from "./tgsl";

// Two properties of the escape leaves are testable without a device and invisible to a CPU round-trip:
// a dual leaf's ternary folds, so only its raw arm reaches the WGSL, and its emitted name is the
// authored one, since a raw splice site calls it by name. That the raw arm's WGSL computes what it
// says is the GPU tier's to show. The `typegpu/std` constructs imported in place of a leaf are held
// to the same two sides: their WGSL is the builtin, and their CPU arm keeps the lattice the codecs
// state.

const wgsl = (item: unknown) => tgpu.resolve([item] as never, { names: "strict" });

describe("escape leaves — emitted WGSL", () => {
    test("idiv folds to its raw arm, leaving no CPU branch behind", () => {
        const code = wgsl(idiv);
        expect(code).toContain("return idivWgsl(");
        expect(code).not.toContain("Math.");
    });

    test("authored names survive strict resolution, so a splice site can call them by name", () => {
        expect(wgsl(idiv)).toContain("fn idiv(");
    });

    test("the std constructs imported in place of a leaf emit their WGSL builtin inline", () => {
        const packs = tgpu.fn(
            [d.vec4f, d.u32],
            d.u32,
        )((v, e) => {
            "use gpu";
            return (
                std.pack2x16snorm(std.unpack2x16snorm(e)) +
                std.pack2x16unorm(std.unpack2x16unorm(e)) +
                std.pack4x8unorm(v) +
                std.bitcastF32toU32(v.x)
            );
        });
        const code = wgsl(packs);
        for (const builtin of [
            "pack2x16snorm(unpack2x16snorm(",
            "pack2x16unorm(unpack2x16unorm(",
            "pack4x8unorm(",
            "bitcast<u32>(",
        ]) {
            expect(code).toContain(builtin);
        }
        expect(code.match(/\bfn /g)).toHaveLength(1);
    });
});

describe("escape leaves — CPU arm", () => {
    test("snorm16x2 lattice: rails exact, lane x in the low half", () => {
        expect(std.pack2x16snorm(vec2f(0, -1))).toBe(0x80010000);
        expect(std.pack2x16snorm(vec2f(1, 0))).toBe(0x00007fff);
        const v = std.unpack2x16snorm(0x80010000);
        expect(v.x).toBe(0);
        expect(v.y).toBe(-1);
    });

    test("snorm16x2 clamps out-of-range lanes to [-1, 1]", () => {
        expect(std.pack2x16snorm(vec2f(2, -3))).toBe(std.pack2x16snorm(vec2f(1, -1)));
    });

    test("unorm16x2 lattice: rails exact, lane x in the low half", () => {
        expect(std.pack2x16unorm(vec2f(1, 0))).toBe(0x0000ffff);
        const v = std.unpack2x16unorm(0x0000ffff);
        expect(v.x).toBe(1);
        expect(v.y).toBe(0);
    });

    test("unorm16x2 clamps out-of-range lanes to [0, 1]", () => {
        expect(std.pack2x16unorm(vec2f(2, -1))).toBe(0x0000ffff);
    });

    test("pack4x8unorm rounds to nearest and clamps", () => {
        expect(std.pack4x8unorm(vec4f(0.5, 0.5, 0.5, 1))).toBe(0xff808080);
        expect(std.pack4x8unorm(vec4f(2, -1, 0, 0))).toBe(0x000000ff);
    });

    test("bitcastF32toU32 reinterprets, never converts", () => {
        expect(std.bitcastF32toU32(1)).toBe(0x3f800000);
        expect(std.bitcastF32toU32(-0)).toBe(0x80000000);
        expect(std.bitcastU32toF32(std.bitcastF32toU32(1.5))).toBe(1.5);
    });

    test("idiv truncates an inexact quotient", () => {
        expect(idiv(7, 2)).toBe(3);
    });

    test("idiv stays exact past 2²⁴, where a float divide has already lost the operand", () => {
        // TGSL emits `/` as a float division, and f32 cannot hold an integer past 2²⁴: 4294967295/3
        // comes back 43 too high through f32.
        expect(idiv(4294967295, 3)).toBe(1431655765);
        expect(Math.trunc(Math.fround(Math.fround(4294967295) / 3))).toBe(1431655808);
        expect(idiv(16777217, 1)).toBe(16777217);
        expect(Math.fround(16777217)).toBe(16777216);
    });
});
