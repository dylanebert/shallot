import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline, noDivision } from "../../../tests/wgsl";
import { radixLdsWgsl } from "./sort-lds";

// The subgroup-free arm's real gate is the `accel` gym scenario's `lds sort` rows on the device (sorted
// + stable over eleven distributions, including all-equal keys and the block boundaries). Device-free,
// what's checkable is that the emitted WGSL still has the properties the algorithm rests on: no
// subgroup op anywhere (the whole point of this arm), the digit-major stride the flat scan depends on,
// and the barrier structure of the two-phase Blelloch scan.

const wgsl = radixLdsWgsl();
const all = [wgsl.hist, wgsl.scan, wgsl.add, wgsl.reorder];
const THREADS = 256;
const RADIX = 16;
const SCAN_ITEMS = 512;

describe("integer discipline", () => {
    test("the arm divides nothing — every index is a shift, mask or multiply-add", () => {
        for (const src of all) {
            noDivision(src);
            integerDiscipline(src);
        }
    });
});

describe("subgroup-free by construction", () => {
    test("no kernel names a subgroup op or builtin — this arm is the WebKit tier", () => {
        for (const src of all) {
            expect(src).not.toContain("subgroup");
            expect(src).not.toContain("enable ");
        }
    });

    test("every cooperative step is a workgroup barrier over LDS", () => {
        expect(wgsl.hist).toContain("var<workgroup>");
        expect(wgsl.scan).toMatch(new RegExp(`var<workgroup> \\w+: array<u32, ${SCAN_ITEMS}>`));
        expect(wgsl.reorder).toMatch(
            new RegExp(`var<workgroup> \\w+: array<atomic<u32>, ${RADIX * 8}>`),
        );
        // the add-back pass is embarrassingly parallel, so it needs none
        expect(wgsl.add).not.toContain("workgroupBarrier");
    });
});

describe("histogram", () => {
    test("writes digit-major, which is what makes a flat scan the global digit base", () => {
        expect(flat(wgsl.hist)).toContain("blockSums[((lane * params.workgroupCount) + wid.x)]");
    });

    test("out-of-count keys are skipped, not counted as digit 0", () => {
        const main = flat(body(wgsl.hist, "@compute"));
        expect(main).toContain("countBuf[0i]");
        expect(main).toContain("if ((gid < count))");
    });

    test("only the first RADIX lanes clear and publish the shared histogram", () => {
        const main = flat(body(wgsl.hist, "@compute"));
        expect(main.match(new RegExp(`lane < ${RADIX}u`, "g"))).toHaveLength(2);
    });
});

describe("Blelloch scan", () => {
    test("both tree phases barrier at the top of the loop, before any lane reads a peer slot", () => {
        const main = flat(body(wgsl.scan, "@compute"));
        // up-sweep: barrier, then the gated combine
        expect(main).toContain("var up = 256u;");
        expect(main).toContain("while ((up > 0u)) { workgroupBarrier();");
        // down-sweep: the same shape, with the total zeroed at the root first
        expect(main).toContain(`temp[${SCAN_ITEMS - 1}i] = 0u`);
        expect(main).toContain(
            "while ((down < 512u)) { offset = (offset >> 1u); workgroupBarrier();",
        );
    });

    test("the chunk total is published before the root is cleared — the exclusive-scan seed", () => {
        const main = flat(body(wgsl.scan, "@compute"));
        const publish = main.indexOf(`chunkSums[wid.x] = temp[${SCAN_ITEMS - 1}i]`);
        const clear = main.indexOf(`temp[${SCAN_ITEMS - 1}i] = 0u`);
        expect(publish).toBeGreaterThanOrEqual(0);
        expect(clear).toBeGreaterThan(publish);
    });

    test("the two-per-thread load and store both bounds-check against the live element count", () => {
        const main = flat(body(wgsl.scan, "@compute"));
        expect(main).toContain("select(0u, items[g0], (g0 < n))");
        expect(main).toContain("if ((g0 < n)) { items[g0] = temp[e0]; }");
    });
});

describe("ranked scatter", () => {
    test("an invalid lane parks on digit RADIX so it flags no real digit's mask", () => {
        expect(flat(wgsl.reorder)).toContain(
            `select(${RADIX}u, ((k >> params.shift) & ${RADIX - 1}u), valid)`,
        );
    });

    test("a lane's rank counts only earlier lanes — the stability argument", () => {
        const main = flat(body(wgsl.reorder, "@compute"));
        // whole words below this lane's, then the partial word masked to bits under `bit`
        expect(main).toContain("while ((w < word))");
        expect(main).toContain("((1u << bit) - 1u)");
        // both shifts seed u32: a bare `1` transpiles i32, which drags the atomic through i32
        // conversions and overflows at lane 31
        expect(main).toContain("atomicOr(&masks[((digit * 8u) + word)], (1u << bit))");
        expect(main).toContain(`prefix[((digit * params.workgroupCount) + wid.x)]`);
    });

    test("the per-round fold clears the masks for the next round, all RADIX digits", () => {
        const main = flat(body(wgsl.reorder, "@compute"));
        expect(main).toContain("if ((r < 13u))"); // EPT - 1
        expect(main).toContain("atomicStore(&masks[idx], 0u)");
        expect(main).toContain("offsets[lane] = (offsets[lane] + c)");
    });

    test("both the key and its payload land at the same computed position", () => {
        const main = flat(body(wgsl.reorder, "@compute"));
        expect(main).toContain("outKeys[pos] = k;");
        expect(main).toContain("outVals[pos] = v;");
    });

    test("the workgroup size is one lane per mask bit — the popcount rank depends on it", () => {
        for (const src of all) expect(src).toContain(`@workgroup_size(${THREADS})`);
    });
});
