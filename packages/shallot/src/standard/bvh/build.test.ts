import { describe, expect, test } from "bun:test";
import { body, flat, IDIV_LEAF, integerDiscipline, noDivision } from "../../../tests/wgsl";
import { buildWgsl, sweepCount } from "./build";

// The build's real gate is the `accel` gym scenario on the device: per fixture it reads the nodes back
// and checks the oracle's structural invariants + SAH + ray-vs-brute-force agreement, plus the
// coherence probe that the old single-kernel climb failed. What's assertable with no device is the
// emitted shape — and the topology pass is where the port's risk concentrates, because it is the one
// kernel whose arithmetic is genuinely signed (`delta` returns -1 past the ends) sitting right next to
// the two constructs TGSL gets silently wrong: the integer `/` and an i32-seeded literal.

const wgsl = buildWgsl();
const all = [wgsl.prepare, wgsl.leaf, wgsl.topo, wgsl.sweep];
const LEVELS = 3;
const NODE_WORDS = 8;

describe("integer discipline", () => {
    test("no kernel divides anything but through `idiv`", () => {
        for (const src of all) noDivision(src);
    });

    test("both dispatch-size divides in prepare go through idiv", () => {
        expect(flat(wgsl.prepare)).toContain(IDIV_LEAF);
        expect(flat(wgsl.prepare)).toContain("idiv((total + 255u), 256u)");
        expect(flat(wgsl.prepare)).toContain("idiv((nInternal + 127u), 128u)");
    });

    test("i32 appears only in the topology pass, where the -1 sentinel needs it", () => {
        for (const src of [wgsl.prepare, wgsl.leaf, wgsl.sweep]) integerDiscipline(src);
        expect(wgsl.topo).toContain("-> i32");
        expect(body(wgsl.topo, "fn delta(")).toContain("return -1");
    });
});

describe("shared node prefix", () => {
    test("leaf, topo and sweep emit byte-identical node accessors", () => {
        expect(body(wgsl.topo, "fn nodeWord(")).toBe(body(wgsl.leaf, "fn nodeWord("));
        expect(body(wgsl.sweep, "fn nodeWord(")).toBe(body(wgsl.leaf, "fn nodeWord("));
        // and only the accessors a kernel actually calls are emitted — the raw-WGSL splice put all
        // seven in every shader, which DXC does not dead-strip (gpu.md "Dead code isn't free")
        expect(wgsl.leaf).not.toContain("fn nodeLeft(");
        expect(wgsl.topo).not.toContain("fn nodeMin(");
        expect(wgsl.sweep).not.toContain("fn writeLeaf(");
        for (const src of [wgsl.leaf, wgsl.topo, wgsl.sweep]) {
            expect(src).toMatch(/var<storage, read_write> nodes: array<u32>/);
            expect(src).toMatch(/var<storage, read> countBuf: array<u32>/);
        }
        // prepare derives dispatch sizes only, so it must not carry the node binding at all
        expect(wgsl.prepare).not.toContain("nodes");
    });

    test("the node base is read once per thread into a private var, not per accessor call", () => {
        // WGSL zero-initializes a private var, so typegpu emits the declaration without the `= 0u`
        expect(flat(wgsl.leaf)).toContain("var<private> nodeBase: u32;");
        for (const src of [wgsl.leaf, wgsl.topo, wgsl.sweep])
            expect(flat(src)).toContain("nodeBase = countBuf[1i]");
        expect(body(wgsl.leaf, "fn nodeWord(")).toContain(`(nodeBase + n) * ${NODE_WORDS}u`);
    });

    test("writeBounds touches the six bound words and never a child pointer", () => {
        const w = body(wgsl.sweep, "fn writeBounds(");
        for (const off of ["o]", "o + 1u)]", "o + 2u)]", "o + 4u)]", "o + 5u)]", "o + 6u)]"])
            expect(flat(w)).toContain(off);
        expect(flat(w)).not.toContain("+ 3u)]");
        expect(flat(w)).not.toContain("+ 7u)]");
        // writeLeaf is the one that owns those two: INVALID in leftChild, the prim index in rightChild
        const l = flat(body(wgsl.leaf, "fn writeLeaf("));
        expect(l).toContain("+ 3u)] = 4294967295u");
        expect(l).toContain("+ 7u)] = j");
    });
});

describe("relaxation sweep", () => {
    test("the resolve chain unrolls to one function per level, deepest terminating on validIn", () => {
        for (let k = 0; k < LEVELS; k++) expect(wgsl.sweep).toContain(`fn resolve${k}(c: u32)`);
        expect(wgsl.sweep).not.toContain(`fn resolve${LEVELS}(`);
        // level 0 is the base case: it reads no children, so it must not recurse
        const base = body(wgsl.sweep, "fn resolve0(c: u32)");
        expect(base).not.toContain("nodeLeft(");
        expect(base).toContain("validIn[c] == 1u");
        // each deeper level calls exactly the level below it, twice
        for (let k = 1; k < LEVELS; k++) {
            const lvl = body(wgsl.sweep, `fn resolve${k}(c: u32)`);
            expect(lvl).toContain(`resolve${k - 1}(nodeLeft(c))`);
            expect(lvl).toContain(`resolve${k - 1}(nodeRight(c))`);
            // one definition, no self-recursion (WGSL has none)
            expect(lvl.match(new RegExp(`resolve${k}\\(`, "g"))).toHaveLength(1);
        }
        // the entry point climbs from the top of the chain
        const main = body(wgsl.sweep, "@compute");
        expect(main).toContain(`resolve${LEVELS - 1}(nodeLeft(n))`);
        expect(main).toContain(`resolve${LEVELS - 1}(nodeRight(n))`);
    });

    test("a descendant's bounds are read only behind its prior-sweep valid flag", () => {
        // the coherence argument: every nodeMin/nodeMax in the chain sits inside the validIn gate
        for (let k = 0; k < LEVELS; k++) {
            const lvl = body(wgsl.sweep, `fn resolve${k}(c: u32)`);
            const gate = lvl.indexOf("validIn[c] == 1u");
            expect(gate).toBeGreaterThanOrEqual(0);
            expect(lvl.indexOf("nodeMin(c)")).toBeGreaterThan(gate);
        }
        // and validOut is written on every path out of the entry point, never left stale
        expect(body(wgsl.sweep, "@compute").match(/validOut\[n\] = /g)).toHaveLength(4);
    });

    test("the sweep covers internal slots only, and single-prim trees early-out", () => {
        const main = flat(body(wgsl.sweep, "@compute"));
        expect(main).toContain("primCount <= 1u");
        expect(main).toContain("primCount + gid.x");
        expect(main).toContain("(2u * primCount) - 2u");
    });
});

describe("topology", () => {
    test("the two searches stay dynamic loops — a constant bound would let DXC unroll them", () => {
        const range = body(wgsl.topo, "fn determineRange(");
        expect(range).toContain("while");
        expect(range).not.toMatch(/for \(/);
        // the doubling search's condition is the delta comparison itself, not a counter
        expect(flat(range)).toContain("while ((delta(ci, i, (i + (dir * lMax)), n) > deltaMin))");
        const split = flat(body(wgsl.topo, "fn findSplit("));
        expect(split).toContain("while (more)");
        expect(split).toContain("more = (stride > 1i)");
    });

    test("clz32 handles the zero input firstLeadingBit can't", () => {
        expect(flat(body(wgsl.topo, "fn clz32("))).toContain(
            "select((31u - firstLeadingBit(x)), 32u, (x == 0u))",
        );
    });

    test("the engine node layout remap survives: internal i → 2N−2−i, leaf → payload slot", () => {
        const main = flat(body(wgsl.topo, "@compute"));
        expect(main).toContain("(2u * primCount) - 2u");
        expect(main).toContain("select((last - split), payload[split], (split == range.x))");
        expect(main).toContain("writeChildren((last - i)");
    });
});

describe("sweep count", () => {
    test("always covers the worst-case tree height — an under-count is silently wrong", () => {
        // the relaxation runs a fixed number of sweeps; too few leaves internal nodes with unfinished
        // bounds and the traverser returns wrong hits, with nothing thrown. No fixture is deep enough
        // to catch it, so the derivation is pinned here instead
        for (const cap of [1, 2, 3, 4, 255, 256, 257, 1024, 65536, 1 << 20, 1 << 24, 1 << 30]) {
            const height = 30 + Math.ceil(Math.log2(Math.max(2, cap)));
            expect(sweepCount(cap) * LEVELS, `cap ${cap}`).toBeGreaterThanOrEqual(
                Math.min(64, height),
            );
        }
    });

    test("is capped, so a huge builder does not dispatch unboundedly", () => {
        // 30 Morton bits + the index tiebreak bound the radix-tree height at 62, so MAX_BOUNDS_STEPS
        // (64) caps the sweeps at 22 for any N a 30-bit code allows
        const cap = Math.ceil(64 / LEVELS);
        for (const n of [1, 1 << 24, 1 << 30, Number.MAX_SAFE_INTEGER])
            expect(sweepCount(n), `cap ${n}`).toBeLessThanOrEqual(cap);
        expect(sweepCount(Number.MAX_SAFE_INTEGER)).toBe(cap);
        // a one-prim build still runs a sweep budget (its threads all early-out)
        expect(sweepCount(1)).toBe(sweepCount(2));
    });
});
