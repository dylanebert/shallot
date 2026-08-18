import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../../../packages/shallot/tests/wgsl";
import { FALLOFF, flattenHeight, flattenHeightGpu, flattenWgsl } from "./flatten";
import { mulberry32 } from "./noise";

// The flatten formula's own gate — the spec's Validation criterion, "Flattening — oracle: centerline
// height + monotone falloff". Pure cosine-ease math (`flatten.ts`'s module header), device-free.

describe("flattenHeight — the CPU reference", () => {
    test("at or inside the core boundary (coreDist <= 0), the target height wins outright — the centerline case", () => {
        expect(flattenHeight(100, 10, 0, FALLOFF)).toBe(10);
        expect(flattenHeight(100, 10, -0.001, FALLOFF)).toBe(10);
        expect(flattenHeight(100, 10, -50, FALLOFF)).toBe(10); // deep inside a wide road's core
    });

    test("at or past the falloff distance, fully natural — unmodified terrain", () => {
        expect(flattenHeight(100, 10, FALLOFF, FALLOFF)).toBe(100);
        expect(flattenHeight(100, 10, FALLOFF + 0.001, FALLOFF)).toBe(100);
        expect(flattenHeight(100, 10, 1000, FALLOFF)).toBe(100);
    });

    test("monotone across the falloff band, moving from target toward natural with no overshoot", () => {
        const natural = 40;
        const target = -10;
        const steps = 50;
        let prev = flattenHeight(natural, target, 0, FALLOFF);
        expect(prev).toBe(target);
        for (let i = 1; i <= steps; i++) {
            const coreDist = (i / steps) * FALLOFF;
            const h = flattenHeight(natural, target, coreDist, FALLOFF);
            expect(h).toBeGreaterThanOrEqual(prev - 1e-9); // monotone: never steps back toward target
            expect(h).toBeLessThanOrEqual(Math.max(natural, target) + 1e-9); // never overshoots either endpoint
            expect(h).toBeGreaterThanOrEqual(Math.min(natural, target) - 1e-9);
            prev = h;
        }
        expect(prev).toBeCloseTo(natural, 9);
    });

    test("monotonicity holds with natural below target too (a cutting, not a fill)", () => {
        const natural = -20;
        const target = 5;
        let prev = flattenHeight(natural, target, 0, FALLOFF);
        for (let i = 1; i <= 20; i++) {
            const coreDist = (i / 20) * FALLOFF;
            const h = flattenHeight(natural, target, coreDist, FALLOFF);
            expect(h).toBeLessThanOrEqual(prev + 1e-9); // moving downward toward natural, monotonically
            prev = h;
        }
    });

    test("natural === target collapses the whole band to one constant height", () => {
        for (const coreDist of [-5, 0, FALLOFF / 2, FALLOFF, FALLOFF * 3]) {
            expect(flattenHeight(7, 7, coreDist, FALLOFF)).toBe(7);
        }
    });
});

describe("differential oracle — flattenHeightGpu vs flattenHeight", () => {
    test("two independently written cosine-ease derivations agree over random inputs", () => {
        const rng = mulberry32(11);
        const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
        for (let i = 0; i < 200; i++) {
            const natural = rand(-40, 40);
            const target = rand(-40, 40);
            const falloff = rand(1, 20);
            const coreDist = rand(-falloff, falloff * 2);
            const cpu = flattenHeight(natural, target, coreDist, falloff);
            const gpu = flattenHeightGpu(natural, target, coreDist, falloff);
            expect(Math.abs(cpu - gpu)).toBeLessThan(1e-4); // f32 roundoff only, same geometric quantity
        }
    });
});

describe("flattenedHeightAt — WGSL structural resolve", () => {
    const wgsl = flattenWgsl();

    test("resolves the natural-height, network-geometry, and cosine-ease graph", () => {
        for (const helper of ["flattenedHeightAt", "flattenHeightGpu", "networkCore", "heightAt"]) {
            expect(wgsl).toContain(`fn ${helper}(`);
        }
        expect(wgsl.match(/var<storage/g)?.length).toBeGreaterThanOrEqual(3); // segments + polygons + polyVerts
    });

    test("integer discipline holds over the network-geometry loops", () => {
        // scoped to this module's own function (never the whole resolved graph): `heightAt`'s own
        // dependency chain (noise.ts's `perlin2`) legitimately uses i32 for its lattice hash, which
        // `generate.test.ts`'s equivalent check also scopes past by reading only the kernel body.
        integerDiscipline(body(flat(wgsl), "fn networkCore("));
    });
});

describe("flattenedHeightAt — degenerate empty network", () => {
    test("with no network buffers bound this is a JS-level concern only — networkCore's own sentinel\n     (f32-max coreDist) is what degrades an empty document to plain heightAt, pinned structurally: an\n     always-natural coreDist never satisfies coreDist <= 0 nor moves the blend off 'natural'", () => {
        // flattenHeight itself, at the sentinel coreDist networkCore returns when no primitive exists,
        // returns exactly the natural height regardless of target — the degradation this module's header
        // comment promises.
        const sentinel = 3.402823e38;
        expect(flattenHeight(12.5, 0, sentinel, FALLOFF)).toBe(12.5);
    });
});
