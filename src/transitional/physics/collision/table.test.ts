import { expect } from "bun:test";
import { check } from "../../../harness/check";
import { boundingPowerOf2, roundUpPowerOf2 } from "../common/bits";
import { addKey, containsKey, createSet, keyHash, pairKeyHi, pairKeyLo, removeKey } from "./table";

// Ports test_table.c: fill every i<j shape pair, remove the j==i+1 diagonal, verify membership
// (querying with reversed args to exercise the symmetric key), then drain. Span is smaller than
// the C's 317 — still drives the set through many grows and the full backward-shift delete path.

check(
    "the pair hash set's power-of-two sizing rounds a capacity up to the next power",
    {
        claim: "the pair hash set sizes its table with a wrong power of two, so probing wraps over a non-power-of-two mask",
    },
    () => {
        const power = boundingPowerOf2(3008);
        expect(power).toBe(12);
        expect(roundUpPowerOf2(3008)).toBe(1 << power);
    },
);

check(
    "the pair hash set fills, removes its diagonal, answers reversed queries and drains",
    {
        claim: "the pair hash set loses or resurrects a pair across grows and backward-shift deletes, so the broad phase reports stale membership",
    },
    () => {
        const N = 40;
        const itemCount = (N * N - N) / 2;
        const removed = new Array<boolean>(itemCount).fill(false);

        const set = createSet(16);

        for (let i = 0; i < N; ++i) {
            for (let j = i + 1; j < N; ++j) {
                addKey(set, i, j, 0);
            }
        }
        expect(set.count).toBe(itemCount);

        let k = 0;
        let removeCount = 0;
        for (let i = 0; i < N; ++i) {
            for (let j = i + 1; j < N; ++j) {
                if (j === i + 1) {
                    removeKey(set, i, j, 0);
                    removed[k++] = true;
                    removeCount += 1;
                } else {
                    removed[k++] = false;
                }
            }
        }
        expect(set.count).toBe(itemCount - removeCount);

        k = 0;
        for (let i = 0; i < N; ++i) {
            for (let j = i + 1; j < N; ++j) {
                // Reversed args — the key is symmetric, so still present unless removed.
                expect(containsKey(set, j, i, 0) || removed[k], `pair (${j}, ${i}, 0)`).toBe(true);
                k += 1;
            }
        }

        for (let i = 0; i < N; ++i) {
            for (let j = i + 1; j < N; ++j) {
                removeKey(set, i, j, 0);
            }
        }
        expect(set.count).toBe(0);
    },
);

check(
    "the pair hash set reports duplicates and separates pairs by child index",
    {
        claim: "the pair hash set treats a reversed insert as new or merges two child indices of one shape pair, so per-child manifolds collide",
    },
    () => {
        const set = createSet(16);
        expect(addKey(set, 3, 7, 0)).toBe(false);
        expect(addKey(set, 7, 3, 0)).toBe(true); // symmetric duplicate
        expect(addKey(set, 3, 7, 1)).toBe(false); // different child
        expect(set.count).toBe(2);
    },
);

const MaxShape = (1 << 22) - 1;
const MaxChild = (1 << 20) - 1;

check(
    "the pair key's split halves match the u64 oracle on packing edge cases",
    {
        claim: "the pair key packs a field across the word boundary wrongly at its extremes, so saturated shape or child indices alias",
    },
    () => {
        // [s1, s2, child, hi, lo, hash] from Box3D's u64 b3ShapePairKey + b3KeyHash. The larger shape
        // index is the only field straddling bit 32; the last four rows walk that boundary.
        for (const [s1, s2, c, hi, lo, hash] of [
            [0, 0, 0, 0x0, 0x0, 0x0],
            [0, 1, 0, 0x0, 0x100000, 0x7657ae14],
            [1, 0, 0, 0x0, 0x100000, 0x7657ae14],
            [MaxShape, MaxShape, MaxChild, 0xffffffff, 0xffffffff, 0x4b825f21],
            [0, MaxShape, MaxChild, 0x3ff, 0xffffffff, 0xdbe0fe82],
            [MaxShape, 0, 0, 0x3ff, 0xfff00000, 0x235cfe2e],
            [MaxShape - 1, MaxShape, MaxChild, 0xfffffbff, 0xffffffff, 0xe86c277b],
            [1, 2, MaxChild, 0x400, 0x2fffff, 0x15cd2890],
            [0, 0xfff, 0, 0x0, 0xfff00000, 0x670ea74e],
            [0, 0x1000, 0, 0x1, 0x0, 0xa5f1419],
            [0, 0xfff000, 0, 0x3ff, 0x0, 0xa0cfccd8],
            [7, 0b1010101010_101010101010, 0xabcde, 0x1eaa, 0xaaaabcde, 0x8f4f233b],
        ]) {
            const label = `(${s1}, ${s2}, ${c})`;
            expect(pairKeyHi(s1, s2), `${label} hi`).toBe(hi);
            expect(pairKeyLo(s1, s2, c), `${label} lo`).toBe(lo);
            expect(keyHash(pairKeyHi(s1, s2), pairKeyLo(s1, s2, c)), `${label} hash`).toBe(hash);
        }
    },
);

check(
    "the pair key is symmetric in its shapes and injective over distinct triples",
    {
        claim: "the pair key maps two distinct shape-child triples to one key or answers differently for a reversed pair, so the broad phase drops a pair",
    },
    () => {
        const seen = new Set<string>();
        for (let s1 = 0; s1 < 12; ++s1) {
            for (let s2 = s1 + 1; s2 < 12; ++s2) {
                for (let c = 0; c < 4; ++c) {
                    const label = `(${s1}, ${s2}, ${c})`;
                    expect(pairKeyHi(s2, s1), `${label} hi reversed`).toBe(pairKeyHi(s1, s2));
                    expect(pairKeyLo(s2, s1, c), `${label} lo reversed`).toBe(pairKeyLo(s1, s2, c));
                    seen.add(`${pairKeyHi(s1, s2)}:${pairKeyLo(s1, s2, c)}`);
                }
            }
        }
        expect(seen.size).toBe(((12 * 12 - 12) / 2) * 4);
    },
);
