import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import { vec2f, vec4f } from "typegpu/data";
import * as std from "typegpu/std";
import {
    bitcastF32toU32,
    compareExchange,
    idiv,
    packSnorm2x16,
    packUnorm2x16,
    packUnorm4x8,
    subgroupUniformityOff,
    uniformLoad,
    unpackSnorm2x16,
    unpackUnorm2x16,
} from "./tgsl";

// The escape leaves are the only place a WGSL construct enters without a TGSL binding. Two properties
// are testable without a device and neither is visible from a CPU round-trip: a dual leaf's ternary
// must *fold* (only the GPU arm reaches the WGSL — an `if` would leak the CPU arm's arithmetic in
// after the return), and the emitted name must be the authored one, since a raw splice site calls it
// by name. That the raw arm's one-line body says what it says is not testable here — it is WGSL text
// copied straight through, and `bun bench` is what proves it compiles and computes.

const wgsl = (item: unknown) => tgpu.resolve([item] as never, { names: "strict" });

describe("escape leaves — emitted WGSL", () => {
    // every dual leaf: its wrapper delegates to the raw arm, and none of the CPU arm's arithmetic
    // (the lattice constants, `Math.*`, the JS-only operators) survives into the shader.
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

    test("authored names survive strict resolution — a splice site calls them by name", () => {
        // The reason the 8-bit pack is `packUnorm4x8` and not `pack4x8unorm`: a name that collides with
        // a WGSL builtin gets suffixed even under strict naming, and no splice site could reach it.
        for (const [name, item] of duals) {
            expect(wgsl(item)).toContain(`fn ${name}(`);
        }
        expect(wgsl(packUnorm4x8)).not.toContain("pack4x8unorm_1");
    });

    test("subgroupUniformityOff emits the module-scope directive", () => {
        const code = tgpu.resolve([subgroupUniformityOff, idiv] as never, { names: "strict" });
        expect(code).toContain("diagnostic(off, subgroup_uniformity);");
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

    test("packUnorm4x8 rounds and clamps, where typegpu/std truncates and wraps", () => {
        // The whole reason this leaf exists rather than a re-export. WGSL's intrinsic is
        // round-to-nearest on a clamped [0,1] lane; std's CPU implementation writes the raw byte.
        expect(packUnorm4x8(vec4f(0.5, 0.5, 0.5, 1))).toBe(0xff808080);
        expect(std.pack4x8unorm(vec4f(0.5, 0.5, 0.5, 1))).not.toBe(0xff808080);

        expect(packUnorm4x8(vec4f(2, -1, 0, 0))).toBe(0x000000ff);
        expect(std.pack4x8unorm(vec4f(2, -1, 0, 0))).not.toBe(0x000000ff);
    });

    test("bitcastF32toU32 reinterprets, never converts", () => {
        expect(bitcastF32toU32(1)).toBe(0x3f800000);
        expect(bitcastF32toU32(-0)).toBe(0x80000000);
        expect(std.bitcastU32toF32(bitcastF32toU32(1.5))).toBe(1.5);
    });

    test("idiv stays exact past 2²⁴, where a float divide has already lost the operand", () => {
        // The silently-wrong case this leaf exists for. TGSL emits `/` as a float division, and f32
        // can't hold an integer past 2²⁴ — 4294967295/3 comes back 43 too high through f32, and the
        // BVH bounds path divides in exactly this range.
        expect(idiv(4294967295, 3)).toBe(1431655765);
        expect(Math.trunc(Math.fround(Math.fround(4294967295) / 3))).toBe(1431655808);
        expect(idiv(16777217, 1)).toBe(16777217);
        expect(Math.fround(16777217)).toBe(16777216);
    });

    test("a GPU-only leaf refuses to run on the CPU rather than answering wrong", () => {
        // a workgroup pointer and an atomic have no CPU meaning, so these two have no dual arm
        expect(() => (uniformLoad as unknown as () => void)()).toThrow();
        expect(() => (compareExchange as unknown as () => void)()).toThrow();
    });
});
