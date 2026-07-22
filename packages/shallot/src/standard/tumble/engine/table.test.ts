import { describe, expect, test } from "bun:test";
import { boundingPowerOf2, roundUpPowerOf2 } from "./bits";
import { addKey, containsKey, createSet, keyHash, pairKeyHi, pairKeyLo, removeKey } from "./table";

// Ports test_table.c: fill every i<j shape pair, remove the j==i+1 diagonal, verify membership
// (querying with reversed args to exercise the symmetric key), then drain. Span is smaller than
// the C's 317 — still drives the set through many grows and the full backward-shift delete path.
describe("hash set", () => {
    test("power-of-two helpers", () => {
        const power = boundingPowerOf2(3008);
        expect(power).toBe(12);
        expect(roundUpPowerOf2(3008)).toBe(1 << power);
    });

    test("fill / remove-diagonal / contains / drain", () => {
        const N = 200;
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
                expect(containsKey(set, j, i, 0) || removed[k]).toBe(true);
                k += 1;
            }
        }

        for (let i = 0; i < N; ++i) {
            for (let j = i + 1; j < N; ++j) {
                removeKey(set, i, j, 0);
            }
        }
        expect(set.count).toBe(0);
    });

    test("addKey reports duplicates; child index distinguishes pairs", () => {
        const set = createSet(16);
        expect(addKey(set, 3, 7, 0)).toBe(false);
        expect(addKey(set, 7, 3, 0)).toBe(true); // symmetric duplicate
        expect(addKey(set, 3, 7, 1)).toBe(false); // different child
        expect(set.count).toBe(2);
    });
});

// The u32-halves key + fmix must be bit-identical to the u64 bigint form Box3D specifies. The
// oracle below is the direct transcription of b3ShapePairKey + b3KeyHash; the split implementation
// is what ships. A single moved bit changes which pairs the broad phase considers new.
const SHAPE_MASK = BigInt((1 << 22) - 1);
const CHILD_MASK = BigInt((1 << 20) - 1);
const U64 = 0xffffffffffffffffn;

function oracleKey(s1: number, s2: number, c: number): bigint {
    const lo = BigInt(s1 < s2 ? s1 : s2) & SHAPE_MASK;
    const hi = BigInt(s1 < s2 ? s2 : s1) & SHAPE_MASK;
    return ((lo << 42n) | (hi << 20n) | (BigInt(c) & CHILD_MASK)) & U64;
}

function oracleHash(key: bigint): number {
    let h = key & U64;
    h ^= h >> 33n;
    h = (h * 0xff51afd7ed558ccdn) & U64;
    h ^= h >> 33n;
    h = (h * 0xc4ceb9fe1a85ec53n) & U64;
    h ^= h >> 33n;
    return Number(h & 0xffffffffn);
}

function assertMatch(s1: number, s2: number, c: number): void {
    const key = oracleKey(s1, s2, c);
    const label = `(${s1}, ${s2}, ${c})`;
    expect(pairKeyHi(s1, s2), `${label} hi`).toBe(Number(key >> 32n));
    expect(pairKeyLo(s1, s2, c), `${label} lo`).toBe(Number(key & 0xffffffffn));
    expect(keyHash(pairKeyHi(s1, s2), pairKeyLo(s1, s2, c)), `${label} hash`).toBe(oracleHash(key));
}

describe("split key + fmix vs the u64 bigint oracle", () => {
    const MaxShape = (1 << 22) - 1;
    const MaxChild = (1 << 20) - 1;

    test("packing edge cases", () => {
        for (const [s1, s2, c] of [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [MaxShape, MaxShape, MaxChild],
            [0, MaxShape, MaxChild],
            [MaxShape, 0, 0],
            [MaxShape - 1, MaxShape, MaxChild],
            [1, 2, MaxChild],
            // The larger shape index is the only field straddling bit 32: its top 10 bits land in
            // the high word, its low 12 in the low word. These walk that boundary.
            [0, 0xfff, 0],
            [0, 0x1000, 0],
            [0, 0xfff000, 0],
            [7, 0b1010101010_101010101010, 0xabcde],
        ]) {
            assertMatch(s1, s2, c);
        }
    });

    test("random triples", () => {
        // Deterministic LCG — a failing seed is reproducible.
        let seed = 0x9e3779b9;
        const next = (bound: number): number => {
            seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
            return seed % bound;
        };
        for (let i = 0; i < 20000; ++i) {
            assertMatch(next(MaxShape + 1), next(MaxShape + 1), next(MaxChild + 1));
        }
    });

    test("the key is symmetric and injective over distinct triples", () => {
        const seen = new Set<string>();
        for (let s1 = 0; s1 < 12; ++s1) {
            for (let s2 = s1 + 1; s2 < 12; ++s2) {
                for (let c = 0; c < 4; ++c) {
                    expect(pairKeyHi(s2, s1)).toBe(pairKeyHi(s1, s2));
                    expect(pairKeyLo(s2, s1, c)).toBe(pairKeyLo(s1, s2, c));
                    seen.add(`${pairKeyHi(s1, s2)}:${pairKeyLo(s1, s2, c)}`);
                }
            }
        }
        expect(seen.size).toBe(((12 * 12 - 12) / 2) * 4);
    });
});
