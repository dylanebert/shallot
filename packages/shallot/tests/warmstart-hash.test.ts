import { describe, test, expect } from "bun:test";

const MAX_PROBE = 128;
const HASH_EMPTY = 0xffffffff;

function packKey(bodyA: number, bodyB: number, slot: number): number {
    const lo = Math.min(bodyA, bodyB);
    const hi = Math.max(bodyA, bodyB);
    let h = (Math.imul(lo, 0x9e3779b9) + hi) >>> 0;
    h = (h ^ Math.imul(slot, 0x517cc1b7)) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    h = Math.imul(h, 0x85ebca6b) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0;
    h = Math.imul(h, 0xc2b2ae35) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    if (h === HASH_EMPTY) h = (h ^ 1) >>> 0;
    return h;
}

function hashKey(k: number): number {
    let h = k >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    h = Math.imul(h, 0x85ebca6b) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0;
    h = Math.imul(h, 0xc2b2ae35) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    return h;
}

function hashInsert(table: Uint32Array, key: number): { slot: number; probes: number } {
    const mask = (table.length - 1) >>> 0;
    const start = hashKey(key) & mask;
    for (let p = 0; p < MAX_PROBE; p++) {
        const idx = (start + p) & mask;
        const stored = table[idx]!;
        if (stored === HASH_EMPTY || stored === key) {
            table[idx] = key;
            return { slot: idx, probes: p + 1 };
        }
    }
    return { slot: -1, probes: MAX_PROBE };
}

function hashLookup(table: Uint32Array, key: number): { slot: number; probes: number } {
    const mask = (table.length - 1) >>> 0;
    const start = hashKey(key) & mask;
    for (let p = 0; p < MAX_PROBE; p++) {
        const idx = (start + p) & mask;
        const stored = table[idx]!;
        if (stored === key) return { slot: idx, probes: p + 1 };
        if (stored === HASH_EMPTY) return { slot: -1, probes: p + 1 };
    }
    return { slot: -1, probes: MAX_PROBE };
}

function randomPairs(bodyCount: number, n: number, seed: number): Array<[number, number, number]> {
    let s = seed >>> 0;
    function next(): number {
        s = Math.imul(s ^ (s >>> 13), 0x5bd1e995) >>> 0;
        s = (s ^ (s >>> 15)) >>> 0;
        return s;
    }
    const pairs: Array<[number, number, number]> = [];
    for (let i = 0; i < n; i++) {
        const a = next() % bodyCount;
        let b = next() % bodyCount;
        if (b === a) b = (a + 1) % bodyCount;
        const slot = next() % 4;
        pairs.push([a, b, slot]);
    }
    return pairs;
}

function countKeyCollisions(pairs: Array<[number, number, number]>): number {
    const keyToTriple = new Map<number, string>();
    let collisions = 0;
    for (const [a, b, slot] of pairs) {
        const key = packKey(a, b, slot);
        const triple = `${Math.min(a, b)},${Math.max(a, b)},${slot}`;
        const existing = keyToTriple.get(key);
        if (existing !== undefined && existing !== triple) collisions++;
        keyToTriple.set(key, triple);
    }
    return collisions;
}

describe("warmstart hash table", () => {
    describe("packKey correctness", () => {
        test("no collisions at realistic load (cap=1024, 16K constraints)", () => {
            const pairs = randomPairs(1024, 16_384, 11111);
            expect(countKeyCollisions(pairs)).toBe(0);
        });

        test("near-zero collisions at realistic load (cap=32768, 500K constraints)", () => {
            const pairs = randomPairs(32768, 500_000, 22222);
            // birthday paradox: 500K^2 / (2*2^32) ≈ 29
            expect(countKeyCollisions(pairs)).toBeLessThan(100);
        });

        test("never produces HASH_EMPTY sentinel", () => {
            const pairs = randomPairs(65536, 1_000_000, 88888);
            for (const [a, b, slot] of pairs) {
                expect(packKey(a, b, slot)).not.toBe(HASH_EMPTY);
            }
        });

        test("slot differentiation", () => {
            for (let i = 0; i < 1000; i++) {
                const keys = new Set<number>();
                for (let slot = 0; slot < 4; slot++) {
                    keys.add(packKey(i, i + 1, slot));
                }
                expect(keys.size).toBe(4);
            }
        });

        test("order invariant", () => {
            for (let i = 0; i < 1000; i++) {
                for (let slot = 0; slot < 4; slot++) {
                    expect(packKey(i, i + 100, slot)).toBe(packKey(i + 100, i, slot));
                }
            }
        });
    });

    describe("end-to-end insert/lookup", () => {
        function runInsertLookup(bodyCount: number, constraintCount: number, seed: number) {
            const hashMul = 32;
            const tableSize = nextPow2(bodyCount) * hashMul;
            const table = new Uint32Array(tableSize).fill(HASH_EMPTY);
            const pairs = randomPairs(bodyCount, constraintCount, seed);

            let overflows = 0;
            let maxProbe = 0;
            const insertedKeys = new Set<number>();

            for (const [a, b, slot] of pairs) {
                const key = packKey(a, b, slot);
                const result = hashInsert(table, key);
                if (result.slot === -1) {
                    overflows++;
                } else {
                    insertedKeys.add(key);
                    maxProbe = Math.max(maxProbe, result.probes);
                }
            }

            let lookupFailures = 0;
            for (const key of insertedKeys) {
                const result = hashLookup(table, key);
                if (result.slot === -1) lookupFailures++;
            }

            return { overflows, maxProbe, lookupFailures };
        }

        test("cap=1024, 16K constraints", () => {
            const r = runInsertLookup(1024, 16_384, 11111);
            expect(r.overflows).toBe(0);
            expect(r.lookupFailures).toBe(0);
            expect(r.maxProbe).toBeLessThan(MAX_PROBE);
        });

        test("cap=16384, 100K constraints", () => {
            const r = runInsertLookup(16384, 100_000, 22222);
            expect(r.overflows).toBe(0);
            expect(r.lookupFailures).toBe(0);
            expect(r.maxProbe).toBeLessThan(MAX_PROBE);
        });

        test("cap=65536, 500K constraints", () => {
            const r = runInsertLookup(65536, 500_000, 33333);
            expect(r.overflows).toBe(0);
            expect(r.lookupFailures).toBe(0);
            expect(r.maxProbe).toBeLessThan(MAX_PROBE);
        });

        test("cap=65536, 1M constraints", () => {
            const r = runInsertLookup(65536, 1_000_000, 44444);
            expect(r.overflows).toBe(0);
            expect(r.lookupFailures).toBe(0);
            expect(r.maxProbe).toBeLessThan(MAX_PROBE);
        });
    });

    describe("probe chain statistics", () => {
        function measureProbeChains(loadFactor: number, tableSize: number, seed: number) {
            const table = new Uint32Array(tableSize).fill(HASH_EMPTY);
            const count = Math.floor(tableSize * loadFactor);
            let s = seed >>> 0;
            function next(): number {
                s = Math.imul(s ^ (s >>> 13), 0x5bd1e995) >>> 0;
                s = (s ^ (s >>> 15)) >>> 0;
                return s;
            }

            let maxProbe = 0;
            let overflows = 0;

            for (let i = 0; i < count; i++) {
                let key = next();
                if (key === HASH_EMPTY) key = (key ^ 1) >>> 0;
                const result = hashInsert(table, key);
                if (result.slot === -1) {
                    overflows++;
                } else {
                    maxProbe = Math.max(maxProbe, result.probes);
                }
            }

            return { maxProbe, overflows };
        }

        test("50% load, zero overflow", () => {
            const r = measureProbeChains(0.5, 1 << 17, 44444);
            expect(r.overflows).toBe(0);
            expect(r.maxProbe).toBeLessThan(MAX_PROBE);
        });

        test("75% load, zero overflow", () => {
            const r = measureProbeChains(0.75, 1 << 17, 55555);
            expect(r.overflows).toBe(0);
            expect(r.maxProbe).toBeLessThan(MAX_PROBE);
        });
    });
});

function nextPow2(n: number): number {
    let v = n;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
