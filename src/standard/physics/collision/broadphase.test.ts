import { expect, test } from "bun:test";
import type { AABB } from "../common/records";
import { BodyType } from "../common/types";
import {
    type BroadPhase,
    bufferMove,
    clearMoved,
    clearMoves,
    createBroadPhase,
    createProxy,
    destroyProxy,
    enlargeProxy,
    isMoved,
    moveCount,
    moveKey,
    moveProxy,
    proxyId,
    proxyKey,
    proxyType,
    testOverlap,
} from "./broadphase";

// The move buffer is what turns a changed proxy into a new contact pair, so its ordering and its
// bit-set mirror are the defects these checks are aimed at.

const box = (c: number, h: number): AABB => ({
    lowerBound: { x: c - h, y: c - h, z: c - h },
    upperBound: { x: c + h, y: c + h, z: c + h },
});

const DEFAULT_HI = 0xffffffff;
const DEFAULT_LO = 0xffffffff;

const fresh = () => createBroadPhase(undefined, { staticShapeCount: 8, dynamicShapeCount: 8 });

function moveKeys(bp: BroadPhase): number[] {
    const out: number[] = [];
    for (let i = 0; i < moveCount(bp); ++i) out.push(moveKey(bp, i));
    return out;
}

/** A bit is set in movedProxies[type] iff its key sits in moveArray exactly once. */
function assertMoveInvariant(bp: BroadPhase, where: string) {
    const keys = moveKeys(bp);
    for (const key of keys) {
        expect(
            isMoved(bp, proxyType(key), proxyId(key)),
            `${where}: key ${key} is in moveArray with no moved bit`,
        ).toBe(true);
    }
    expect(new Set(keys).size, `${where}: moveArray holds a duplicate key`).toBe(keys.length);
}

test("the broad-phase proxy key round-trips its id and body type through pack and unpack", () => {
    const types: BodyType[] = [BodyType.Static, BodyType.Kinematic, BodyType.Dynamic];
    for (const type of types) {
        for (const id of [0, 1, 7, 42, 1000, 1 << 20]) {
            const vector = `id ${id}, type ${type}`;
            const key = proxyKey(id, type);
            expect(proxyType(key), `${vector}: type did not round-trip`).toBe(type);
            expect(proxyId(key), `${vector}: id did not round-trip`).toBe(id);
        }
    }
});

test("the broad-phase move buffer records dynamic and forced-static proxies on create, in order, and leaves a plain static proxy unmoved", () => {
    const bp = fresh();
    const dyn = createProxy(bp, BodyType.Dynamic, box(0, 0.5), DEFAULT_HI, DEFAULT_LO, 0, false);
    const stat = createProxy(bp, BodyType.Static, box(5, 0.5), DEFAULT_HI, DEFAULT_LO, 1, false);
    const forced = createProxy(bp, BodyType.Static, box(9, 0.5), DEFAULT_HI, DEFAULT_LO, 2, true);

    expect(moveKeys(bp), "dynamic and forced-static creates should buffer, in order").toEqual([
        dyn,
        forced,
    ]);
    expect(
        isMoved(bp, BodyType.Static, proxyId(stat)),
        "unforced static create left a moved bit",
    ).toBe(false);
    assertMoveInvariant(bp, "static/dynamic create");
});

test("the broad-phase move buffer holds a proxy moved twice once", () => {
    const bp = fresh();
    const key = createProxy(bp, BodyType.Dynamic, box(0, 0.5), DEFAULT_HI, DEFAULT_LO, 0, false);
    bufferMove(bp, key);
    bufferMove(bp, key);
    expect(moveKeys(bp), "repeated bufferMove should not append").toEqual([key]);
    assertMoveInvariant(bp, "repeated bufferMove");
});

test("the broad-phase move buffer keeps buffered keys in creation order", () => {
    const bp = fresh();
    const keys: number[] = [];
    for (let i = 0; i < 6; ++i) {
        keys.push(
            createProxy(bp, BodyType.Dynamic, box(i * 2, 0.5), DEFAULT_HI, DEFAULT_LO, i, false),
        );
    }
    expect(moveKeys(bp), "insertion order should survive six creates").toEqual(keys);
    assertMoveInvariant(bp, "six creates");
});

test("destroying a buffered proxy drops its key and moved bit and keeps the other keys", () => {
    const bp = fresh();
    const a = createProxy(bp, BodyType.Dynamic, box(0, 0.5), DEFAULT_HI, DEFAULT_LO, 0, false);
    const b = createProxy(bp, BodyType.Dynamic, box(2, 0.5), DEFAULT_HI, DEFAULT_LO, 1, false);
    const c = createProxy(bp, BodyType.Dynamic, box(4, 0.5), DEFAULT_HI, DEFAULT_LO, 2, false);

    // the middle one: swap-removed from the move array, bit cleared.
    destroyProxy(bp, b);
    const keys = moveKeys(bp);
    expect(keys, "destroyed key b still buffered").not.toContain(b);
    expect(keys, "surviving key a was dropped").toContain(a);
    expect(keys, "surviving key c was dropped").toContain(c);
    expect(isMoved(bp, proxyType(b), proxyId(b)), "destroyed key b kept its moved bit").toBe(false);
    assertMoveInvariant(bp, "destroy middle");
});

test("the broad-phase move buffer re-buffers a proxy moved after the per-step clear", () => {
    const bp = fresh();
    const key = createProxy(bp, BodyType.Dynamic, box(0, 0.5), DEFAULT_HI, DEFAULT_LO, 0, false);
    // mirror the reset done at the end of each step, then move it.
    clearMoves(bp);
    clearMoved(bp, proxyType(key), proxyId(key));
    expect(moveKeys(bp), "the step reset should empty the buffer").toEqual([]);

    moveProxy(bp, key, box(3, 0.5));
    expect(moveKeys(bp), "moveProxy did not re-buffer the key").toEqual([key]);
    assertMoveInvariant(bp, "moveProxy after reset");
});

test("the broad-phase overlap test reports touching proxies overlapping and distant ones separated", () => {
    const bp = fresh();
    const a = createProxy(bp, BodyType.Dynamic, box(0, 1), DEFAULT_HI, DEFAULT_LO, 0, false);
    const b = createProxy(bp, BodyType.Dynamic, box(1.5, 1), DEFAULT_HI, DEFAULT_LO, 1, false);
    const c = createProxy(bp, BodyType.Dynamic, box(10, 1), DEFAULT_HI, DEFAULT_LO, 2, false);
    expect(testOverlap(bp, a, b), "touching proxies reported separated").toBe(true);
    expect(testOverlap(bp, a, c), "distant proxies reported overlapping").toBe(false);
});

test("enlarging a static proxy throws", () => {
    const bp = fresh();
    const stat = createProxy(bp, BodyType.Static, box(0, 0.5), DEFAULT_HI, DEFAULT_LO, 0, false);
    expect(() => enlargeProxy(bp, stat, box(0, 2))).toThrow();
});
