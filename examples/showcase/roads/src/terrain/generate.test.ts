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
// GPU dispatch (CPU-callable TGSL + exact resolved WGSL, never a bound device). The kernel's own
// seed-determinism *readback* runs on the real device instead
// (`gate.ts`, driven by `test/roads.spec.ts`) — the split voxel's `generate.test.ts`/`gate.ts` also make.

describe("height kernel reference", () => {
    const wgsl = heightKernelWgsl();

    test("resolves the complete noise, flatten, encode, and storage graph", () => {
        for (const helper of [
            "grad2",
            "perlin2",
            "fbm2",
            "heightAt",
            "flattenedHeightAt",
            "flattenHeightGpu",
            "encodePos",
            "encodeUv",
            "octEncodeNormal",
        ]) {
            expect(wgsl).toContain(`fn ${helper}(`);
        }
        expect(flat(wgsl)).toContain(
            "@group(0) @binding(0) var<storage, read> perm_1: array<u32, 512>;",
        );
        // perm + vertices + position + the flatten network's segments (`roads-interactive.md` stage
        // 1 deleted the polygon-stamp path's own polygons/polyVerts storage bindings)
        expect(wgsl.match(/var<storage/g)?.length).toBeGreaterThanOrEqual(4);
    });

    test("dispatches one thread per vertex, with a bounds guard at the fence-post edge", () => {
        const kernel = flat(body(wgsl, "@compute"));
        expect(kernel).toContain("@workgroup_size(8, 8, 1)");
        expect(kernel).toContain(`gid_1.x >= ${VERTS}u`);
        expect(kernel).toContain(`gid_1.y >= ${VERTS}u`);
        integerDiscipline(kernel);
        noIntegerDivision(kernel);
    });

    test("samples the flattened surface at its own column and its four neighbours (the finite-difference normal)", () => {
        const kernel = flat(body(wgsl, "@compute"));
        // every sample goes through flattenedHeightAt (flatten.ts), never a bare heightAt — the emitted
        // normal has to reflect the flattened surface too, or a flat road would still light like a slope.
        expect(kernel.match(/flattenedHeightAt\(/g)?.length).toBe(5);
        expect(kernel).not.toContain("heightAt(");
        // pin the four neighbour offsets themselves, not just the call count — a call count alone can't
        // tell "±eps on x" from "the same point sampled twice" (caught red-first: dropping the z+eps
        // offset in favour of a duplicate z-sample left the count at 5 and this suite green).
        expect(kernel).toContain("let y = flattenedHeightAt(x, z);");
        expect(kernel).toContain("let yx0 = flattenedHeightAt((x - eps), z);");
        expect(kernel).toContain("let yx1 = flattenedHeightAt((x + eps), z);");
        expect(kernel).toContain("let yz0 = flattenedHeightAt(x, (z - eps));");
        expect(kernel).toContain("let yz1 = flattenedHeightAt(x, (z + eps));");
    });

    test("addresses the fixed grid's world position from the documented HALF/SPACING constants", () => {
        const kernel = flat(body(wgsl, "@compute"));
        expect(kernel).toContain(`(f32(ix) - ${HALF}f) * ${SPACING}f`);
        expect(kernel).toContain(`(f32(iz) - ${HALF}f) * ${SPACING}f`);
    });
});
