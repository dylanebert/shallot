/**
 * The WGSL-1.0 portability gate over the resolved production shaders whose kernels reach
 * `workgroupUniformLoad`. Chrome's Tint admits a `ptr<workgroup, …>` function parameter under its
 * `unrestricted_pointer_parameters` extension; naga — Firefox's front end — rejects the module, so the
 * pre-fix `uniformLoad` leaf made the light-cull, AVBD solver/broadphase and BVH binning shaders
 * uncompilable on Firefox (RUM `e669fa3c`, which reached `shallot-light-cull` pipeline creation).
 *
 * The population is the whole emitted text of every module that owned such a call site, not just the
 * six kernels: a leaf reintroduced anywhere in one of these modules rides into their shaders too. Both
 * the size floor and the `workgroupUniformLoad` subset are pinned, so a resolver that silently returned
 * nothing cannot pass this by scanning an empty corpus.
 *
 * Workgroup-space parameters only, which is the population this gate cleared. naga rejects a *storage*
 * -space pointer parameter for the same reason, and the BVH binning shader still carries one in the
 * `compareExchange` leaf (`engine/utils/tgsl.ts`) — so passing this test is not yet a claim that every
 * shader here compiles on Firefox.
 */
import { describe, expect, test } from "bun:test";
import { stepWgsl } from "../src/standard/avbd/step";
import { radixWgsl } from "../src/standard/bvh/sort";
import { gridWgsl, lightCullWgsl } from "../src/standard/render/cluster";
import { portablePointers } from "./wgsl";

const membership = { base: 0, mask: 1 };

function shaders(): Record<string, string> {
    const { compact, cull } = lightCullWgsl(membership, { base: 0, mask: 2 }, { base: 0, mask: 4 });
    const radix = radixWgsl();
    const out: Record<string, string> = { compact, cull, grid: gridWgsl() };
    for (const [name, resolve] of Object.entries(stepWgsl)) out[`step.${name}`] = resolve();
    for (const [name, src] of Object.entries(radix)) out[`radix.${name}`] = src;
    return out;
}

describe("WGSL 1.0 portability", () => {
    test("no production shader declares a workgroup-address-space pointer parameter", () => {
        const population = shaders();
        // the three cluster passes, the five Onesweep passes and every AVBD pass
        expect(Object.keys(population).length).toBe(3 + 5 + Object.keys(stepWgsl).length);
        expect(Object.keys(population).length).toBeGreaterThan(20);
        for (const [name, wgsl] of Object.entries(population)) {
            expect(wgsl.length).toBeGreaterThan(0);
            try {
                portablePointers(wgsl);
            } catch (e) {
                throw new Error(`${name}: ${e instanceof Error ? e.message : String(e)}`);
            }
        }
    });

    test("the uniform-load sites survive the repair, inlined against their variable", () => {
        const population = shaders();
        const uniform = Object.entries(population)
            .filter(([, wgsl]) => wgsl.includes("workgroupUniformLoad("))
            .map(([name]) => name);
        // the exact six call sites: light cull's batch bound, AVBD's small broadphase, LDS solve
        // (count + color count) and small CSR coloring, and the BVH lookback's early-exit gate
        expect(uniform.toSorted()).toEqual([
            "cull",
            "radix.binning",
            "step.broadphaseSmall",
            "step.csrColorSmall",
            "step.solveLds",
        ]);
        for (const name of uniform)
            expect(population[name]).toMatch(/workgroupUniformLoad\(&?\(?&\w+\)?\)/);
    });
});
