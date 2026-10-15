import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import { vec2f, vec4f } from "typegpu/data";
import * as std from "typegpu/std";
import {
    bitcastF32toU32,
    idiv,
    packSnorm2x16,
    packUnorm2x16,
    packUnorm4x8,
    unpackSnorm2x16,
    unpackUnorm2x16,
} from "./tgsl";

// Two properties of the escape leaves are testable without a device and invisible to a CPU round-trip:
// a dual leaf's ternary folds, so only its raw arm reaches the WGSL, and its emitted name is the
// authored one, since a raw splice site calls it by name. That the raw arm's WGSL computes what it
// says is the GPU tier's to show.

const wgsl = (item: unknown) => tgpu.resolve([item] as never, { names: "strict" });

describe("escape leaves — emitted WGSL", () => {
    const duals: [string, unknown][] = [
        ["packSnorm2x16", packSnorm2x16],
        ["unpackSnorm2x16", unpackSnorm2x16],
        ["packUnorm2x16", packUnorm2x16],
        ["unpackUnorm2x16", unpackUnorm2x16],
        ["packUnorm4x8", packUnorm4x8],
        ["bitcastF32toU32", bitcastF32toU32],
        ["idiv", idiv],
    ];
    for (const [name, item] of duals) {
        test(`${name} folds to its raw arm, leaving no CPU branch behind`, () => {
            const code = wgsl(item);
            expect(code).toContain(`return ${name}Wgsl(`);
            for (const cpuOnly of ["32767", "65535", "255", "Math.", ">>>"]) {
                expect(code).not.toContain(cpuOnly);
            }
        });
    }

    test("authored names survive strict resolution, so a splice site can call them by name", () => {
        // A name that collides with a WGSL builtin is suffixed even under strict naming: an 8-bit pack
        // named pack4x8unorm would resolve as pack4x8unorm_1.
        for (const [name, item] of duals) {
            expect(wgsl(item)).toContain(`fn ${name}(`);
        }
        expect(wgsl(packUnorm4x8)).not.toContain("pack4x8unorm_1");
    });
});

describe("escape leaves — CPU arm", () => {
    test("snorm16x2 lattice: rails exact, lane x in the low half", () => {
        expect(packSnorm2x16(vec2f(0, -1))).toBe(0x80010000);
        expect(packSnorm2x16(vec2f(1, 0))).toBe(0x00007fff);
        const v = unpackSnorm2x16(0x80010000);
        expect(v.x).toBe(0);
        expect(v.y).toBe(-1);
    });

    test("snorm16x2 clamps out-of-range lanes to [-1, 1]", () => {
        expect(packSnorm2x16(vec2f(2, -3))).toBe(packSnorm2x16(vec2f(1, -1)));
    });

    test("unorm16x2 lattice: rails exact, lane x in the low half", () => {
        expect(packUnorm2x16(vec2f(1, 0))).toBe(0x0000ffff);
        const v = unpackUnorm2x16(0x0000ffff);
        expect(v.x).toBe(1);
        expect(v.y).toBe(0);
    });

    test("unorm16x2 clamps out-of-range lanes to [0, 1]", () => {
        expect(packUnorm2x16(vec2f(2, -1))).toBe(0x0000ffff);
    });

    test("packUnorm4x8 rounds to nearest and clamps, as WGSL's pack4x8unorm does", () => {
        expect(packUnorm4x8(vec4f(0.5, 0.5, 0.5, 1))).toBe(0xff808080);
        expect(packUnorm4x8(vec4f(2, -1, 0, 0))).toBe(0x000000ff);
    });

    test("bitcastF32toU32 reinterprets, never converts", () => {
        expect(bitcastF32toU32(1)).toBe(0x3f800000);
        expect(bitcastF32toU32(-0)).toBe(0x80000000);
        expect(std.bitcastU32toF32(bitcastF32toU32(1.5))).toBe(1.5);
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
