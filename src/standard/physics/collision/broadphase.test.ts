import { afterEach, expect, test } from "bun:test";
import type { AABB } from "../common/records";
import { BodyType } from "../common/types";
import { rebuild } from "../kernel/treecolumns";
import { createTestWorld, destroyTestWorld } from "../testing";
import type { WorldState } from "../world/world";
import {
    bufferMove,
    createProxy,
    destroyProxy,
    enlargeProxy,
    isMoved,
    moveProxy,
    proxyId,
    proxyKey,
    proxyType,
    testOverlap,
} from "./broadphase";

const worlds: WorldState[] = [];
afterEach(() => {
    for (const world of worlds) destroyTestWorld(world);
    worlds.length = 0;
});
const fresh = () => {
    const world = createTestWorld();
    worlds.push(world);
    return world.broadPhase;
};
const box = (c: number, h: number): AABB => ({
    lowerBound: { x: c - h, y: c - h, z: c - h },
    upperBound: { x: c + h, y: c + h, z: c + h },
});
const HI = 0xffffffff,
    LO = 0xffffffff;
test("the broad-phase proxy key round-trips its id and body type through pack and unpack", () => {
    for (const type of [BodyType.Static, BodyType.Kinematic, BodyType.Dynamic])
        for (const id of [0, 1, 7, 42, 1000, 1 << 20]) {
            const key = proxyKey(id, type);
            expect(proxyType(key)).toBe(type);
            expect(proxyId(key)).toBe(id);
        }
});
test("dynamic and forced-static creates mark leaves, while a plain static create stays unmoved", () => {
    const bp = fresh();
    const dynamic = createProxy(bp, BodyType.Dynamic, box(0, 0.5), HI, LO, 0, false);
    const forced = createProxy(bp, BodyType.Static, box(4, 0.5), HI, LO, 1, true);
    const plain = createProxy(bp, BodyType.Static, box(8, 0.5), HI, LO, 2, false);
    expect(isMoved(bp, proxyType(dynamic), proxyId(dynamic))).toBe(true);
    expect(isMoved(bp, proxyType(forced), proxyId(forced))).toBe(true);
    expect(isMoved(bp, proxyType(plain), proxyId(plain))).toBe(false);
});
test("repeated marking preserves proxy identity and a rebuild clears moved leaves", () => {
    const bp = fresh();
    const key = createProxy(bp, BodyType.Dynamic, box(0, 0.5), HI, LO, 0, false);
    bufferMove(bp, key);
    bufferMove(bp, key);
    expect(isMoved(bp, BodyType.Dynamic, proxyId(key))).toBe(true);
    rebuild(bp.trees[BodyType.Dynamic], false);
    expect(isMoved(bp, BodyType.Dynamic, proxyId(key))).toBe(false);
    expect(bp.trees[BodyType.Dynamic].proxyCount).toBe(1);
});
test("sibling relocation keeps every moved proxy's identity", () => {
    const bp = fresh();
    const keys = Array.from({ length: 6 }, (_, i) =>
        createProxy(bp, BodyType.Dynamic, box(i * 2, 0.5), HI, LO, i, false),
    );
    expect(new Set(keys).size).toBe(6);
    for (const key of keys) expect(isMoved(bp, BodyType.Dynamic, proxyId(key))).toBe(true);
    expect(bp.trees[BodyType.Dynamic].proxyCount).toBe(6);
});
test("destroying a moved proxy leaves surviving markers intact and reuses its proxy slot", () => {
    const bp = fresh();
    const a = createProxy(bp, BodyType.Dynamic, box(0, 0.5), HI, LO, 0, false),
        b = createProxy(bp, BodyType.Dynamic, box(2, 0.5), HI, LO, 1, false),
        c = createProxy(bp, BodyType.Dynamic, box(4, 0.5), HI, LO, 2, false);
    destroyProxy(bp, b);
    expect(isMoved(bp, BodyType.Dynamic, proxyId(b))).toBe(false);
    for (const key of [a, c]) expect(isMoved(bp, BodyType.Dynamic, proxyId(key))).toBe(true);
    expect(createProxy(bp, BodyType.Dynamic, box(6, 0.5), HI, LO, 3, false)).toBe(b);
});
test("moving after a rebuild marks the leaf again without marking its sibling", () => {
    const bp = fresh();
    const a = createProxy(bp, BodyType.Dynamic, box(0, 0.5), HI, LO, 0, false),
        b = createProxy(bp, BodyType.Dynamic, box(8, 0.5), HI, LO, 1, false);
    rebuild(bp.trees[BodyType.Dynamic], false);
    expect(isMoved(bp, BodyType.Dynamic, proxyId(a))).toBe(false);
    moveProxy(bp, a, box(8, 0.5));
    expect(isMoved(bp, BodyType.Dynamic, proxyId(a))).toBe(true);
    expect(isMoved(bp, BodyType.Dynamic, proxyId(b))).toBe(false);
    expect(testOverlap(bp, a, b)).toBe(true);
});
test("the broad-phase overlap test reports touching proxies overlapping and distant ones separated", () => {
    const bp = fresh();
    const a = createProxy(bp, BodyType.Dynamic, box(0, 0.5), HI, LO, 0, false),
        b = createProxy(bp, BodyType.Dynamic, box(1, 0.5), HI, LO, 1, false),
        c = createProxy(bp, BodyType.Static, box(4, 0.5), HI, LO, 2, false);
    expect(testOverlap(bp, a, b)).toBe(true);
    expect(testOverlap(bp, a, c)).toBe(false);
});
test("enlarging a static proxy throws", () => {
    const bp = fresh();
    const key = createProxy(bp, BodyType.Static, box(0, 0.5), HI, LO, 0, false);
    expect(() => enlargeProxy(bp, key, box(0, 1))).toThrow();
});
