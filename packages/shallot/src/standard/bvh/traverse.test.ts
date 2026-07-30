import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline, noDivision } from "../../../tests/wgsl";
import { BVH_INVALID, BVH_TRAIL_LEVELS, bvhRoot, bvhRootWgsl, bvhTraverseWgsl } from "./traverse";

// The traverser is not CPU-callable — every entry point reads either the consumer's `nodes` global or a
// `private` var — so the correctness truth stays the real-GPU `accel` gym gate against tests/bvh/oracle.ts
// (`nearestHitBvh` / `anyHitBvh`, every fixture × 128 rays, both device arms). What lives here is what the
// gym can't see: `bvhRoot`'s CPU arm (the one part that IS pure), and the emitted-WGSL structure the port
// must not drift on — every index a shift/mask (no division at all), unsigned throughout, and the
// relocatable contract (`nodes` read by name, the authored function names preserved so a raw splice site
// can still call them).

const wgsl = bvhTraverseWgsl();

describe("bvhRoot", () => {
    test("is the last-allocated node for N ≥ 2, node 0 below", () => {
        expect(bvhRoot(0)).toBe(0);
        expect(bvhRoot(1)).toBe(0);
        expect(bvhRoot(2)).toBe(2);
        expect(bvhRoot(3)).toBe(4);
        expect(bvhRoot(1024)).toBe(2046);
        expect(bvhRoot(1 << 20)).toBe(2 * (1 << 20) - 2);
    });

    test("emits WGSL a raw consumer can call by name", () => {
        expect(bvhRootWgsl()).toContain("fn bvhRoot(");
        noDivision(bvhRootWgsl());
    });
});

describe("emitted traversal WGSL", () => {
    test("exposes the relocatable contract by its authored names", () => {
        for (const name of ["fn bvhClosestHit(", "fn bvhAnyHit(", "struct BvhHit"])
            expect(wgsl).toContain(name);
    });

    test("reads the consumer's `nodes` global by name and declares none itself", () => {
        expect(wgsl).toContain("nodes[");
        // the chunk must NOT declare the binding — the consumer owns group/binding assignment
        expect(wgsl).not.toContain("var<storage");
        expect(wgsl).not.toContain("@group");
    });

    test("divides nowhere — every index is a shift, mask or multiply-add", () => {
        noDivision(wgsl);
    });

    test("stays unsigned: no i32 arithmetic, the signed-overflow trap the port removed", () => {
        integerDiscipline(wgsl);
    });

    test("keeps the per-invocation trail + stack in `private`, sized to the derived bounds", () => {
        // 2 bits per level, 16 levels per u32 word
        const words = Math.ceil((BVH_TRAIL_LEVELS * 2) / 32);
        expect(flat(wgsl)).toContain(`var<private> bvhTrail: array<u32, ${words}>`);
        expect(flat(wgsl)).toContain(`var<private> bvhStack: array<u32, ${BVH_TRAIL_LEVELS}>`);
    });

    test("closest-hit prunes against the live hit distance, not tMax", () => {
        const src = flat(body(wgsl, "fn bvhClosestHit("));
        // three prune sites (leaf, left child, right child) all read the tightened distance
        expect(src.match(/hitT/g)?.length).toBeGreaterThanOrEqual(5);
        expect(src).toContain("bvhStack[sp]");
    });

    test("any-hit zeroes its trail at entry — a stale trail would drop a hit", () => {
        const src = flat(body(wgsl, "fn bvhAnyHit("));
        expect(src).toMatch(/bvhTrail\[\w+\] = 0u/);
        expect(src).toContain("bvhSSCount = 0u");
        expect(src).toContain("bvhSSHead = 0u");
    });

    // The trail is the highest-risk arithmetic here and the `accel` gym gate is the only place it
    // executes, so the fast tier pins its layout verbatim: 2 bits per level, 16 levels per u32 word,
    // and the `sub != 15` guard that exists solely to avoid the undefined shift-by-32 at a word
    // boundary. An off-by-one in any of the three is the mistake this port could plausibly make.
    test("packs the restart trail 2 bits per level, 16 levels per word", () => {
        const get = flat(body(wgsl, "fn bvhTrailGet("));
        expect(get).toContain("bvhTrail[(level >> 4u)] >> ((level & 15u) * 2u)) & 3u");

        const set = flat(body(wgsl, "fn bvhTrailSet("));
        expect(set).toContain("let w = (level >> 4u);");
        expect(set).toContain("let s = ((level & 15u) * 2u);");
        // the mask literals are u32-seeded: a bare `3` materializes i32 and overflows at level 15
        expect(set).toContain("(bvhTrail[w] & ~(3u << s)) | (v << s)");
    });

    test("the reset keeps every level at or above pl and guards the word-boundary shift", () => {
        const reset = flat(body(wgsl, "fn bvhTrailResetBelow("));
        expect(reset).toContain("let w0 = (pl >> 4u);");
        expect(reset).toContain("let sub = (pl & 15u);");
        // sub == 15 means pl is the word's last level, so there is nothing to keep and `1u << 32` is
        // undefined — the guard is why the shift below is always in range
        expect(reset).toContain("if ((sub != 15u))");
        expect(reset).toContain("(1u << ((sub + 1u) * 2u)) - 1u");
    });

    test("the short-stack ring wraps by mask, sized to its power-of-two capacity", () => {
        const push = flat(body(wgsl, "fn bvhPushFar("));
        expect(push).toContain("bvhSS[((bvhSSHead + bvhSSCount) & 7u)] = far");
        expect(push).toContain("bvhSSHead = ((bvhSSHead + 1u) & 7u)");
    });

    test("the miss sentinel is the one a caller compares against, above any finite tMax", () => {
        // the slab test reports a miss as the sentinel and every caller gates on it — so the value
        // must exceed any tMax a consumer passes (both pass 1e30), or a real miss reads as a hit
        expect(3e38).toBeGreaterThan(1e30);
        expect(flat(body(wgsl, "fn bvhSlab("))).toContain("return 3e+38f;");
        for (const fn of ["fn bvhClosestHit(", "fn bvhAnyHit("])
            expect(flat(body(wgsl, fn)).match(/< 3e\+38f/g)?.length).toBe(3); // leaf + both children
        // a miss reports the leaf sentinel, not a stale primitive
        expect(flat(body(wgsl, "fn bvhClosestHit("))).toContain(`var hitPrim = ${BVH_INVALID}u`);
    });
});
