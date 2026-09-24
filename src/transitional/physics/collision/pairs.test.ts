// Collision filtering and contact pair creation. The fixture scenes all use the default all-pass
// filter, so `shouldShapesCollide`'s branches — category/mask overlap, the u64 split, and the shared
// non-zero group override — are pinned here directly. The last check drives the real World so the
// moved-proxy dedup and the pair-set membership rejection, which otherwise live only inside the wasm
// pair work, are observable through the public contact events.

import { afterAll, beforeAll, expect } from "bun:test";
import { check } from "../../../harness/check";
import { BodyType, init, makeBoxHull, shutdown, World } from "../api/index";
import { defaultFilter, type FilterBits, toFilterBits } from "../common/types";
import { shouldShapesCollide } from "./pairs";

function filter(categoryBits: bigint, maskBits: bigint, groupIndex: number): FilterBits {
    return toFilterBits({ categoryBits, maskBits, groupIndex });
}

check(
    "category and mask must overlap in both directions",
    {
        claim: "collision filtering by category and mask lets a one-sided match through, so a shape that sees another collides with it even when that other shape's mask excludes it — and the default all-pass filter stops colliding with itself",
    },
    () => {
        // the default filter is the all-pass case every fixture scene rides on.
        expect(
            shouldShapesCollide(toFilterBits(defaultFilter()), toFilterBits(defaultFilter())),
        ).toBe(true);

        // A sees B (B's category is in A's mask) but B does not see A (A's category not in B's mask).
        expect(shouldShapesCollide(filter(0b01n, 0b10n, 0), filter(0b10n, 0b10n, 0))).toBe(false);

        // Disjoint categories and masks: no overlap either way.
        expect(shouldShapesCollide(filter(0b01n, 0b01n, 0), filter(0b10n, 0b10n, 0))).toBe(false);

        // Overlapping both ways.
        expect(shouldShapesCollide(filter(0b01n, 0b10n, 0), filter(0b10n, 0b01n, 0))).toBe(true);
    },
);

check(
    "filter bits above 32 survive the u64 split",
    {
        claim: "collision filtering across the two u32 halves of the 64-bit filter drops or aliases the high half, so a category above bit 32 either misses its own mask or matches an unrelated low-half one, and a half's top bit reads as no match through a signed AND",
    },
    () => {
        const hiOnly = 1n << 40n;
        const loOnly = 1n << 8n;
        expect(shouldShapesCollide(filter(hiOnly, hiOnly, 0), filter(hiOnly, hiOnly, 0))).toBe(
            true,
        );
        expect(shouldShapesCollide(filter(hiOnly, hiOnly, 0), filter(loOnly, loOnly, 0))).toBe(
            false,
        );

        // The top bit of each half is the sign bit of a signed 32-bit AND — it must not read as "no match".
        const topBits = (1n << 63n) | (1n << 31n);
        expect(shouldShapesCollide(filter(topBits, topBits, 0), filter(topBits, topBits, 0))).toBe(
            true,
        );
    },
);

check(
    "a shared non-zero filter group overrides the mask",
    {
        claim: "collision filtering by group index stops taking precedence over the mask, so a shared positive group no longer forces the pair together, a shared negative group no longer keeps it apart, or two different groups skip the mask test instead of falling through to it",
    },
    () => {
        // Same positive group forces collision despite disjoint masks.
        expect(shouldShapesCollide(filter(0b01n, 0b01n, 7), filter(0b10n, 0b10n, 7))).toBe(true);

        // Same negative group forbids collision despite overlapping masks.
        expect(shouldShapesCollide(filter(0xffn, 0xffn, -3), filter(0xffn, 0xffn, -3))).toBe(false);

        // Different groups fall through to the mask test.
        expect(shouldShapesCollide(filter(0b01n, 0b01n, 1), filter(0b10n, 0b10n, 2))).toBe(false);
    },
);

beforeAll(async () => {
    await init();
});
afterAll(shutdown);

check(
    "overlapping moved proxies create one contact and don't duplicate it across steps",
    {
        claim: "contact pair creation from the broad phase emits an overlapping pair once per moved side instead of deduplicating it, or re-creates a pair already in the pair set on a later step, so a single touching pair fires two begin-touch events",
    },
    () => {
        // Two dynamic boxes overlap along x and both drift +y (zero gravity, sleep off) so both sit in
        // the move buffer every step. Step 1: both moved, the pair is found from both sides — dedup
        // must emit it exactly once. Step 2: the pair persists in the pair set, so the re-query must
        // reject it — no duplicate contact, no new begin.
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
        // Membership: the pair persisted, so step 2 creates no duplicate contact and fires no new
        // begin; the overlap is deep enough that they stay touching (no end event).
        expect(beginTotal).toBe(1);
        expect(endTotal).toBe(0);

        world.destroy();
    },
);
