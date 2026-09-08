import { describe, expect, test } from "bun:test";
import { clearBit, getBit } from "./bitset";
import {
    BodyType,
    type BroadPhase,
    bufferMove,
    createBroadPhase,
    createProxy,
    destroyProxy,
    enlargeProxy,
    moveProxy,
    proxyId,
    proxyKey,
    proxyType,
    testOverlap,
} from "./broadphase";
import type { AABB } from "./math";

const box = (c: number, h: number): AABB => ({
    lowerBound: { x: c - h, y: c - h, z: c - h },
    upperBound: { x: c + h, y: c + h, z: c + h },
});

const DEFAULT_HI = 0xffffffff;
const DEFAULT_LO = 0xffffffff;

function moveKeys(bp: BroadPhase): number[] {
    const out: number[] = [];
    for (let i = 0; i < bp.moveArray.count; ++i) out.push(bp.moveArray.get(i));
    return out;
}

// Invariant: a bit is set in movedProxies[type] iff its key is present in moveArray.
function assertMoveInvariant(bp: BroadPhase) {
    const keys = moveKeys(bp);
    for (const key of keys) {
        expect(getBit(bp.movedProxies[proxyType(key)], proxyId(key))).toBe(true);
    }
    // no duplicates
    expect(new Set(keys).size).toBe(keys.length);
}

describe("proxy key packing", () => {
    test("pack/unpack round-trips id and type", () => {
        for (const type of [BodyType.Static, BodyType.Kinematic, BodyType.Dynamic]) {
            for (const id of [0, 1, 7, 42, 1000, 1 << 20]) {
                const key = proxyKey(id, type);
                expect(proxyType(key)).toBe(type);
                expect(proxyId(key)).toBe(id);
            }
        }
    });
});

describe("broadphase move buffer", () => {
    test("dynamic create buffers; static create does not (unless forced)", () => {
        const bp = createBroadPhase({ staticShapeCount: 8, dynamicShapeCount: 8 });

        const dyn = createProxy(
            bp,
            BodyType.Dynamic,
            box(0, 0.5),
            DEFAULT_HI,
            DEFAULT_LO,
            0,
            false,
        );
        const stat = createProxy(
            bp,
            BodyType.Static,
            box(5, 0.5),
            DEFAULT_HI,
            DEFAULT_LO,
            1,
            false,
        );
        const statForced = createProxy(
            bp,
            BodyType.Static,
            box(9, 0.5),
            DEFAULT_HI,
            DEFAULT_LO,
            2,
            true,
        );

        expect(moveKeys(bp)).toEqual([dyn, statForced]);
        expect(getBit(bp.movedProxies[BodyType.Static], proxyId(stat))).toBe(false);
        assertMoveInvariant(bp);
    });

    test("bufferMove dedups repeated keys", () => {
        const bp = createBroadPhase({ staticShapeCount: 8, dynamicShapeCount: 8 });
        const key = createProxy(
            bp,
            BodyType.Dynamic,
            box(0, 0.5),
            DEFAULT_HI,
            DEFAULT_LO,
            0,
            false,
        );
        bufferMove(bp, key);
        bufferMove(bp, key);
        expect(moveKeys(bp)).toEqual([key]);
        assertMoveInvariant(bp);
    });

    test("insertion order is deterministic", () => {
        const bp = createBroadPhase({ staticShapeCount: 8, dynamicShapeCount: 8 });
        const keys: number[] = [];
        for (let i = 0; i < 6; ++i) {
            keys.push(
                createProxy(
                    bp,
                    BodyType.Dynamic,
                    box(i * 2, 0.5),
                    DEFAULT_HI,
                    DEFAULT_LO,
                    i,
                    false,
                ),
            );
        }
        expect(moveKeys(bp)).toEqual(keys);
        assertMoveInvariant(bp);
    });

    test("destroy un-buffers and preserves the invariant", () => {
        const bp = createBroadPhase({ staticShapeCount: 8, dynamicShapeCount: 8 });
        const a = createProxy(bp, BodyType.Dynamic, box(0, 0.5), DEFAULT_HI, DEFAULT_LO, 0, false);
        const b = createProxy(bp, BodyType.Dynamic, box(2, 0.5), DEFAULT_HI, DEFAULT_LO, 1, false);
        const c = createProxy(bp, BodyType.Dynamic, box(4, 0.5), DEFAULT_HI, DEFAULT_LO, 2, false);

        // Destroy the middle one: swap-removed from the move array, bit cleared.
        destroyProxy(bp, b);
        const keys = moveKeys(bp);
        expect(keys).not.toContain(b);
        expect(keys).toContain(a);
        expect(keys).toContain(c);
        expect(getBit(bp.movedProxies[proxyType(b)], proxyId(b))).toBe(false);
        assertMoveInvariant(bp);
    });

    test("moveProxy re-buffers a proxy", () => {
        const bp = createBroadPhase({ staticShapeCount: 8, dynamicShapeCount: 8 });
        const key = createProxy(
            bp,
            BodyType.Dynamic,
            box(0, 0.5),
            DEFAULT_HI,
            DEFAULT_LO,
            0,
            false,
        );
        // Clear the buffer by destroying then recreating would change ids; instead move an existing.
        bp.moveArray.clear();
        clearMovedBit(bp, key);
        expect(moveKeys(bp)).toEqual([]);

        moveProxy(bp, key, box(3, 0.5));
        expect(moveKeys(bp)).toEqual([key]);
        assertMoveInvariant(bp);
    });
});

describe("broadphase overlap + enlarge guard", () => {
    test("testOverlap reflects proxy AABB overlap", () => {
        const bp = createBroadPhase({ staticShapeCount: 8, dynamicShapeCount: 8 });
        const a = createProxy(bp, BodyType.Dynamic, box(0, 1), DEFAULT_HI, DEFAULT_LO, 0, false);
        const b = createProxy(bp, BodyType.Dynamic, box(1.5, 1), DEFAULT_HI, DEFAULT_LO, 1, false);
        const c = createProxy(bp, BodyType.Dynamic, box(10, 1), DEFAULT_HI, DEFAULT_LO, 2, false);
        expect(testOverlap(bp, a, b)).toBe(true);
        expect(testOverlap(bp, a, c)).toBe(false);
    });

    test("enlarging a static proxy throws", () => {
        const bp = createBroadPhase({ staticShapeCount: 8, dynamicShapeCount: 8 });
        const stat = createProxy(
            bp,
            BodyType.Static,
            box(0, 0.5),
            DEFAULT_HI,
            DEFAULT_LO,
            0,
            false,
        );
        expect(() => enlargeProxy(bp, stat, box(0, 2))).toThrow();
    });
});

// Helper: clear a proxy's moved bit directly (mirrors the internal reset done each step).
function clearMovedBit(bp: BroadPhase, key: number): void {
    clearBit(bp.movedProxies[proxyType(key)], proxyId(key));
}
