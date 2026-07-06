import { describe, expect, test } from "bun:test";
import { entity, f16, f32, i32, slab, sparse, u8, u32, vec2, vec4 } from "../..";
import { lanes, refs } from "./core";

describe("sparse scalar", () => {
    test("set / get roundtrip across f32 / i32 / u32 / u8", () => {
        const cases = [
            { type: f32, eid: 7, value: -1.5 },
            { type: i32, eid: 7, value: -42 },
            { type: u32, eid: 7, value: 0xdead_beef },
            { type: u8, eid: 7, value: 200 },
        ];
        for (const { type, eid, value } of cases) {
            const field = sparse(type);
            field.set(eid, value);
            expect(field.get(eid)).toBe(value);
            expect(field.get(eid + 1)).toBe(0);
        }
    });

    test("f16 roundtrip is exact on representable values, within 1 ULP otherwise", () => {
        const field = sparse(f16);
        // powers of two and their sums are exact in f16
        for (const v of [0, 0.5, 1, -1, 1.5, 2, -2, 1024]) {
            field.set(5, v);
            expect(field.get(5)).toBe(v);
        }
        // 0.1 has no exact f16 form; round-to-nearest keeps it within a relative
        // half-ULP (10 stored mantissa bits + implicit leading 1 → 2⁻¹¹)
        field.set(5, 0.1);
        expect(Math.abs(field.get(5) - 0.1)).toBeLessThanOrEqual(0.1 * 2 ** -11);
    });

    test("unset eids return the type's zero", () => {
        const field = sparse(f32);
        expect(field.get(99999)).toBe(0);
        field.set(3, 1);
        expect(field.get(99999)).toBe(0);
    });
});

describe("sparse vector", () => {
    test("vec2 lanes share per-entity storage", () => {
        const v = sparse(vec2);
        v.x.set(3, 1.5);
        v.y.set(3, 2.5);
        expect(v.x.get(3)).toBe(1.5);
        expect(v.y.get(3)).toBe(2.5);
        expect(v.x.get(4)).toBe(0);
        expect(v.y.get(4)).toBe(0);
    });

    test("vec2 bulk set + read roundtrip", () => {
        const v = sparse(vec2);
        v.set(8, 1.5, -2.5);
        const dst = new Float32Array(2);
        v.read(8, dst);
        expect(Array.from(dst)).toEqual([1.5, -2.5]);
    });

    test("vec4 bulk set populates all lanes", () => {
        const v = sparse(vec4);
        v.set(2, 10, 20, 30, 40);
        expect(v.x.get(2)).toBe(10);
        expect(v.y.get(2)).toBe(20);
        expect(v.z.get(2)).toBe(30);
        expect(v.w.get(2)).toBe(40);
    });

    test("lane write materializes the entry, other lanes default to zero", () => {
        const v = sparse(vec4);
        v.z.set(5, 7);
        expect(v.x.get(5)).toBe(0);
        expect(v.y.get(5)).toBe(0);
        expect(v.z.get(5)).toBe(7);
        expect(v.w.get(5)).toBe(0);
    });

    test("read writes zeros into dst for unset eids", () => {
        const v = sparse(vec4);
        const dst = new Float32Array([9, 9, 9, 9]);
        v.read(42, dst);
        expect(Array.from(dst)).toEqual([0, 0, 0, 0]);
    });
});

describe("lanes()", () => {
    test("Single from sparse(f32)", () => {
        expect(lanes(sparse(f32))).toBe(1);
    });

    test("Pair from sparse(vec2)", () => {
        expect(lanes(sparse(vec2))).toBe(2);
    });

    test("Quad from sparse(vec4)", () => {
        expect(lanes(sparse(vec4))).toBe(4);
    });

    test("Quad from slab(vec4)", () => {
        expect(lanes(slab(vec4))).toBe(4);
    });

    test("lane Singles of a Pair/Quad report as Single", () => {
        const p = sparse(vec2);
        const q = sparse(vec4);
        expect(lanes(p.x)).toBe(1);
        expect(lanes(p.y)).toBe(1);
        expect(lanes(q.x)).toBe(1);
        expect(lanes(q.w)).toBe(1);
    });

    test("non-storage values report 0", () => {
        expect(lanes(new Float32Array(8))).toBe(0);
        expect(lanes([1, 2, 3])).toBe(0);
        expect(lanes(null)).toBe(0);
        expect(lanes(undefined)).toBe(0);
        expect(lanes({})).toBe(0);
    });
});

describe("refs()", () => {
    test("returns the sparse(entity) fields, not a plain u32 on the same component", () => {
        // the crux of the type-based approach: an entity ref and a u32 path-intern are byte-identical
        // storage — only the descriptor tells them apart, so no separate ref list can drift
        const Comp = { target: sparse(entity), field: sparse(u32), pos: sparse(vec4) };
        expect(refs(Comp)).toEqual(["target"]);
    });

    test("no entity fields → empty", () => {
        expect(refs({ x: sparse(f32), v: sparse(vec2) })).toEqual([]);
    });
});
