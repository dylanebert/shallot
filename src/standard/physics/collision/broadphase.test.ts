import { expect } from "bun:test";
import { check } from "../../../harness/check";
import { getBit } from "../common/bitset";
import type { AABB } from "../common/math";
import {
    BodyType,
    type BodyTypeValue,
    type BroadPhase,
    bufferMove,
    clearMoved,
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

// The move buffer is what turns a changed proxy into a new contact pair, so its ordering and its
// bit-set mirror are the defects these checks are aimed at.

const box = (c: number, h: number): AABB => ({
    lowerBound: { x: c - h, y: c - h, z: c - h },
    upperBound: { x: c + h, y: c + h, z: c + h },
});

const DEFAULT_HI = 0xffffffff;
const DEFAULT_LO = 0xffffffff;

const fresh = () => createBroadPhase({ staticShapeCount: 8, dynamicShapeCount: 8 });

function moveKeys(bp: BroadPhase): number[] {
    const out: number[] = [];
    for (let i = 0; i < bp.moveArray.count; ++i) out.push(bp.moveArray.get(i));
    return out;
}

/** A bit is set in movedProxies[type] iff its key sits in moveArray exactly once. */
function assertMoveInvariant(bp: BroadPhase, where: string) {
    const keys = moveKeys(bp);
    for (const key of keys) {
        expect(
            getBit(bp.movedProxies[proxyType(key)], proxyId(key)),
            `${where}: key ${key} is in moveArray with no moved bit`,
        ).toBe(true);
    }
    expect(new Set(keys).size, `${where}: moveArray holds a duplicate key`).toBe(keys.length);
}

check(
    "proxy key packs and unpacks id and body type",
    {
        claim: "the broad-phase proxy key loses the id or the body type across pack and unpack, so a moved proxy would be attributed to the wrong tree",
        tier: "step",
    },
    () => {
        const types: BodyTypeValue[] = [BodyType.Static, BodyType.Kinematic, BodyType.Dynamic];
        for (const type of types) {
            for (const id of [0, 1, 7, 42, 1000, 1 << 20]) {
                const vector = `id ${id}, type ${type}`;
                const key = proxyKey(id, type);
                expect(proxyType(key), `${vector}: type did not round-trip`).toBe(type);
                expect(proxyId(key), `${vector}: id did not round-trip`).toBe(id);
            }
        }
    },
);

check(
    "the broad-phase move buffer takes dynamic creates and skips unforced static ones",
    {
        claim: "the broad-phase move buffer records a plain static proxy on create, so every static shape would be queried for new pairs on the step it is added",
        tier: "step",
    },
    () => {
        const bp = fresh();
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
        const forced = createProxy(
            bp,
            BodyType.Static,
            box(9, 0.5),
            DEFAULT_HI,
            DEFAULT_LO,
            2,
            true,
        );

        expect(moveKeys(bp), "dynamic and forced-static creates should buffer, in order").toEqual([
            dyn,
            forced,
        ]);
        expect(
            getBit(bp.movedProxies[BodyType.Static], proxyId(stat)),
            "unforced static create left a moved bit",
        ).toBe(false);
        assertMoveInvariant(bp, "static/dynamic create");
    },
);

check(
    "the broad-phase move buffer dedups a key buffered twice",
    {
        claim: "the broad-phase move buffer appends a key it already holds, so one proxy moved twice in a step would be pair-queried twice",
        tier: "step",
    },
    () => {
        const bp = fresh();
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
        expect(moveKeys(bp), "repeated bufferMove should not append").toEqual([key]);
        assertMoveInvariant(bp, "repeated bufferMove");
    },
);

check(
    "the broad-phase move buffer keeps creates in insertion order",
    {
        claim: "the broad-phase move buffer reorders buffered keys against creation order, so pair finding would stop being deterministic across runs",
        tier: "step",
    },
    () => {
        const bp = fresh();
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
        expect(moveKeys(bp), "insertion order should survive six creates").toEqual(keys);
        assertMoveInvariant(bp, "six creates");
    },
);

check(
    "a destroyed proxy leaves the broad-phase move buffer and its bit",
    {
        claim: "the broad-phase move buffer keeps a destroyed proxy's key or its moved bit, so a freed tree slot would be pair-queried after destruction",
        tier: "step",
    },
    () => {
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
        expect(
            getBit(bp.movedProxies[proxyType(b)], proxyId(b)),
            "destroyed key b kept its moved bit",
        ).toBe(false);
        assertMoveInvariant(bp, "destroy middle");
    },
);

check(
    "moveProxy re-enters a cleared proxy into the broad-phase move buffer",
    {
        claim: "the broad-phase move buffer misses a proxy moved after the per-step clear, so a body that moved would never be re-queried for pairs",
        tier: "step",
    },
    () => {
        const bp = fresh();
        const key = createProxy(
            bp,
            BodyType.Dynamic,
            box(0, 0.5),
            DEFAULT_HI,
            DEFAULT_LO,
            0,
            false,
        );
        // mirror the reset done at the end of each step, then move it.
        bp.moveArray.clear();
        clearMoved(bp, proxyType(key), proxyId(key));
        expect(moveKeys(bp), "the step reset should empty the buffer").toEqual([]);

        moveProxy(bp, key, box(3, 0.5));
        expect(moveKeys(bp), "moveProxy did not re-buffer the key").toEqual([key]);
        assertMoveInvariant(bp, "moveProxy after reset");
    },
);

check(
    "testOverlap on two proxy keys reflects their tree AABBs",
    {
        claim: "the broad-phase proxy key overlap test reads the wrong tree slot, so overlapping proxies would report separated and drop the contact",
        tier: "step",
    },
    () => {
        const bp = fresh();
        const a = createProxy(bp, BodyType.Dynamic, box(0, 1), DEFAULT_HI, DEFAULT_LO, 0, false);
        const b = createProxy(bp, BodyType.Dynamic, box(1.5, 1), DEFAULT_HI, DEFAULT_LO, 1, false);
        const c = createProxy(bp, BodyType.Dynamic, box(10, 1), DEFAULT_HI, DEFAULT_LO, 2, false);
        expect(testOverlap(bp, a, b), "touching proxies reported separated").toBe(true);
        expect(testOverlap(bp, a, c), "distant proxies reported overlapping").toBe(false);
    },
);

check(
    "enlarging a static proxy key throws",
    {
        claim: "the broad-phase proxy key guard lets a static proxy be enlarged, so the static tree would silently need rebuilding mid-step",
        tier: "step",
    },
    () => {
        const bp = fresh();
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
    },
);
