// Filter collision logic (b3ShouldShapesCollide). The fixture scenes all use the default all-pass
// filter, so its three branches — same non-zero group, category/mask overlap, and no overlap — are
// pinned here instead. Ported from test_shape's filter cases.

import { expect, test } from "bun:test";
import { BodyType, makeBoxHull, World } from "./index";
import { shouldShapesCollide } from "./pairs";
import { defaultFilter, type FilterBits, toFilterBits } from "./types";

function filter(categoryBits: bigint, maskBits: bigint, groupIndex: number): FilterBits {
    return toFilterBits({ categoryBits, maskBits, groupIndex });
}

test("default filters collide", () => {
    expect(shouldShapesCollide(toFilterBits(defaultFilter()), toFilterBits(defaultFilter()))).toBe(
        true,
    );
});

test("category/mask must overlap in both directions", () => {
    // A sees B (B's category is in A's mask) but B does not see A (A's category not in B's mask).
    const a = filter(0b01n, 0b10n, 0);
    const b = filter(0b10n, 0b10n, 0);
    expect(shouldShapesCollide(a, b)).toBe(false);

    // Disjoint categories and masks: no overlap either way.
    expect(shouldShapesCollide(filter(0b01n, 0b01n, 0), filter(0b10n, 0b10n, 0))).toBe(false);

    // Overlapping both ways.
    expect(shouldShapesCollide(filter(0b01n, 0b10n, 0), filter(0b10n, 0b01n, 0))).toBe(true);
});

test("bits above 32 survive the u64 split", () => {
    // The filter is stored as two u32 halves, so a category living only in the high half must still
    // meet a mask that only covers the high half — and must not alias the low half's bits.
    const hiOnly = 1n << 40n;
    const loOnly = 1n << 8n;
    expect(shouldShapesCollide(filter(hiOnly, hiOnly, 0), filter(hiOnly, hiOnly, 0))).toBe(true);
    expect(shouldShapesCollide(filter(hiOnly, hiOnly, 0), filter(loOnly, loOnly, 0))).toBe(false);

    // The top bit of each half is the sign bit of a signed 32-bit AND — it must not read as "no match".
    const topBits = (1n << 63n) | (1n << 31n);
    expect(shouldShapesCollide(filter(topBits, topBits, 0), filter(topBits, topBits, 0))).toBe(
        true,
    );
});

test("a shared non-zero group overrides the mask", () => {
    // Same positive group forces collision despite disjoint masks.
    expect(shouldShapesCollide(filter(0b01n, 0b01n, 7), filter(0b10n, 0b10n, 7))).toBe(true);

    // Same negative group forbids collision despite overlapping masks.
    expect(shouldShapesCollide(filter(0xffn, 0xffn, -3), filter(0xffn, 0xffn, -3))).toBe(false);

    // Different groups fall through to the mask test.
    expect(shouldShapesCollide(filter(0b01n, 0b01n, 1), filter(0b10n, 0b10n, 2))).toBe(false);
});

// The moved-proxy dedup and pair-set-membership rejection live in pairwork.rs (wasm-only), otherwise
// covered only by end-to-end fixtures. Pin both through the public API. Two dynamic boxes overlap along
// x and both drift +y (zero gravity, sleep off) so both sit in the move buffer every step. Step 1: both
// moved, the pair is found from both sides — dedup must emit it exactly once (one begin event). Step 2:
// the pair persists in the pair-set, so the re-query must reject it — no duplicate contact, no new begin.
test("overlapping moved proxies create one contact and don't duplicate it across steps", () => {
    const world = new World({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    const drift = { x: 0, y: 0.5, z: 0 };

    const a = world.createBody({
        type: BodyType.Dynamic,
        position: { x: -0.25, y: 0, z: 0 },
        linearVelocity: drift,
    });
    a.createHull({ enableContactEvents: true }, makeBoxHull(0.5, 0.5, 0.5));
    const b = world.createBody({
        type: BodyType.Dynamic,
        position: { x: 0.25, y: 0, z: 0 },
        linearVelocity: drift,
    });
    b.createHull({ enableContactEvents: true }, makeBoxHull(0.5, 0.5, 0.5));

    let beginTotal = 0;
    let endTotal = 0;

    world.step(1 / 60);
    let ev = world.getContactEvents();
    beginTotal += ev.beginEvents.length;
    endTotal += ev.endEvents.length;
    // Dedup: the overlapping pair emitted exactly once, not once per moved side.
    expect(beginTotal).toBe(1);

    world.step(1 / 60);
    ev = world.getContactEvents();
    beginTotal += ev.beginEvents.length;
    endTotal += ev.endEvents.length;
    // Membership: the pair persisted, so step 2 creates no duplicate contact and fires no new begin;
    // the overlap is deep enough that they stay touching (no end event).
    expect(beginTotal).toBe(1);
    expect(endTotal).toBe(0);
});
