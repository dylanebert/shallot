import { describe, expect, test } from "bun:test";
import {
    body,
    flat,
    integerDiscipline,
    noIntegerDivision,
} from "../../../../../packages/shallot/tests/wgsl";
import { heightKernelWgsl } from "./generate";
import { HALF, SPACING, VERTS } from "./grid";

// The height kernel's structural gate — the device-free seam this stage's `bun test` relies on for the
// GPU dispatch (`testing.md`'s ladder rung (a)/(b): CPU-callable TGSL + exact resolved WGSL, never a
// bound device). The kernel's own seed-determinism *readback* runs on the real device instead
// (`gate.ts`, driven by `test/roads.spec.ts`) — the split voxel's `generate.test.ts`/`gate.ts` also make.

describe("height kernel reference", () => {
    const wgsl = heightKernelWgsl();

    test("resolves the complete noise, encode, and storage graph", () => {
        for (const helper of [
            "grad2",
            "perlin2",
            "fbm2",
            "heightAt",
            "encodePos",
            "encodeUv",
            "octEncodeNormal",
        ]) {
            expect(wgsl).toContain(`fn ${helper}(`);
        }
        expect(flat(wgsl)).toContain(
            "@group(0) @binding(0) var<storage, read> perm_1: array<u32, 512>;",
        );
        expect(wgsl.match(/var<storage/g)?.length).toBeGreaterThanOrEqual(3); // perm + vertices + position
    });

    test("dispatches one thread per vertex, with a bounds guard at the fence-post edge", () => {
        const kernel = flat(body(wgsl, "@compute"));
        expect(kernel).toContain("@workgroup_size(8, 8, 1)");
        expect(kernel).toContain(`gid_1.x >= ${VERTS}u`);
        expect(kernel).toContain(`gid_1.y >= ${VERTS}u`);
        integerDiscipline(kernel);
        noIntegerDivision(kernel);
    });

    test("samples the surface at its own column and its four neighbours (the finite-difference normal)", () => {
        const kernel = flat(body(wgsl, "@compute"));
        expect(kernel.match(/heightAt\(/g)?.length).toBe(5);
    });

    test("addresses the fixed grid's world position from the documented HALF/SPACING constants", () => {
        const kernel = flat(body(wgsl, "@compute"));
        expect(kernel).toContain(`(f32(ix) - ${HALF}f) * ${SPACING}f`);
        expect(kernel).toContain(`(f32(iz) - ${HALF}f) * ${SPACING}f`);
    });
});
