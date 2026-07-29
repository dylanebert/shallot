import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../tests/wgsl";
import { capacity } from "../../engine";
import { packWgsl } from "./pack";

// The pack kernels' device-free structural seam. Real-GPU truth — survivor counts, survivor identity,
// per-view slot offsets — is the gym `render` scenario (`bun bench --scenario render`); what's assertable
// without a device is the emitted WGSL's shape, and that's where the TGSL port's two silent-wrong classes
// live: a literal seeded as `i32` (which flips the arithmetic signed and litters conversions), and a bare
// `/` on integers (which transpiles to *float* division).

const MEMBERSHIP_BASE = 65536;
const MEMBERSHIP_MASK = 4;
const SURFACES = 3;
const wgsl = packWgsl(MEMBERSHIP_BASE, MEMBERSHIP_MASK, SURFACES);
const all = [wgsl.count, wgsl.scan, wgsl.scatter];

describe("integer discipline", () => {
    test("the pack divides nothing — it has no arithmetic TGSL's float `/` could corrupt", () => {
        // the pair index is a multiply-add and the scan walks by addition, so a `/` appearing here at
        // all is the signal, whether it went through `idiv` or (worse) didn't
        for (const src of all) expect(flat(src)).not.toMatch(/[)\w] ?\/ ?/);
    });

    test("every integer local stays u32 — no signed literals or conversions leak in", () => {
        for (const src of all) integerDiscipline(src);
    });
});

describe("shared cull group", () => {
    test("count and scatter emit byte-identical cull declarations and `visible`", () => {
        // the shared-layout convention: one `cullLayout` both kernels reference, so the prefix is not
        // re-declared per kernel and the two gates cannot drift
        expect(body(wgsl.scatter, "fn visible(")).toBe(body(wgsl.count, "fn visible("));
        expect(body(wgsl.scatter, "fn partPair(")).toBe(body(wgsl.count, "fn partPair("));
        for (const name of ["surfaceField", "meshField", "membership", "transforms"]) {
            const decl = new RegExp(`@group\\(0\\) @binding\\(\\d+\\) var<storage, read> ${name}:`);
            expect(wgsl.count).toMatch(decl);
            expect(wgsl.scatter).toMatch(decl);
        }
    });

    test("each kernel's own I/O lands outside the shared cull group", () => {
        expect(wgsl.count).toMatch(/@group\(1\) @binding\(\d+\) var<storage, read_write> counts:/);
        expect(wgsl.scatter).toMatch(
            /@group\(1\) @binding\(\d+\) var<storage, read_write> packedEids:/,
        );
        // the scan shares no cull input, so it declares exactly one group
        expect(wgsl.scan).not.toContain("@group(1)");
        expect(wgsl.scan).not.toContain("cullVolumes");
    });
});

describe("folded constants", () => {
    test("the membership gate, surface count and capacity resolve to literals — no uniform for them", () => {
        for (const src of [wgsl.count, wgsl.scatter]) {
            expect(src).toContain(`membership[(${MEMBERSHIP_BASE}u + eid)] & ${MEMBERSHIP_MASK}u`);
            expect(src).toContain(`eid >= ${capacity}u`);
            expect(src).toContain(`sid >= ${SURFACES}u`);
            expect(src).toContain(`(mid * ${SURFACES}u)`);
        }
        // the pair count is the one late-arriving dimension, so it stays a uniform read
        expect(wgsl.count).toContain("params.pairCount");
    });

    test("the scan writes instanceCount and firstInstance, never the static lanes", () => {
        const main = body(wgsl.scan, "@compute");
        expect(main).toContain("drawArgs[((idx * 5u) + 1u)]");
        expect(main).toContain("drawArgs[((idx * 5u) + 4u)]");
        expect(main).not.toContain("(idx * 5u))]"); // lane 0 — indexCount
        expect(main).not.toContain("+ 2u)]"); // lane 2 — firstIndex
        expect(main).toContain(`slot * ${capacity}u`); // the slot's packedEids region base
    });
});
