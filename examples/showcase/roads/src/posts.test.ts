import { describe, expect, test } from "bun:test";
import { flat } from "../../../../packages/shallot/tests/wgsl";
import {
    isLiveSlot,
    liveSlotCount,
    POST_COUNT,
    POST_OFFSET,
    POST_SPACING,
    postLateralSign,
    postStation,
    postsWgsl,
} from "./posts";
import { FLAT_CORE_MARGIN } from "./terrain/flatten-math";
import { SPACING, WORLD_EXTENT } from "./terrain/grid";

// Stage 5's posts test — the WGSL structural seam plus the plain-TS behavioural witnesses for the
// station/lateral/slot-count/scale-0 derivations. The structural arm (the resolved WGSL contains
// `flattenedHeightAt`) is the named oracle Validation requires; the behavioural arms witness the
// real properties by calling the exported pure derivation functions and checking against
// independently-derived expectations — not by re-deriving from the subject. Each behavioural arm's
// docblock records the red seen when the production value was mutated to prove it can fail.

describe("posts emitted WGSL", () => {
    test("the resolved kernel contains flattenedHeightAt (the named oracle)", () => {
        const wgsl = flat(postsWgsl());
        // the named oracle: the kernel calls flatten.ts's own flattenedHeightAt TGSL fn, bound
        // through the network bind group — never re-deriving height on the CPU for this consumer.
        expect(wgsl).toContain("flattenedHeightAt");
    });

    test("the resolved kernel's station, lateral sign, and slot index are the correct literals", () => {
        // LITERALS, not values re-derived from POST_SPACING/POST_OFFSET — an arm that recomputes its
        // own rule goes green on a wrong constant (this unit's stage-3 precedent). The assertions below
        // pin the exact emitted WGSL text: the station `(f32(i) + 1f) * 20f` (not `f32(i) * 20f`),
        // the lateral `select(1f, -1f, ((i & 1u) == 1u))` (not a swapped or constant sign), and the slot
        // index `segments[0i]` (not `segments[1i]`). Each would fail under the mutation it names.
        const wgsl = flat(postsWgsl());

        // station: (i + 1) * POST_SPACING emits as `(f32(i) + 1f) * 20f` — the `+ 1f` is what
        // discriminates from `i * POST_SPACING` which emits `(f32(i) * 20f)` with no `+ 1f`.
        expect(wgsl).toContain("(f32(i) + 1f) * 20f");

        // lateral sign: even → +1, odd → −1, emitted as `select(1f, -1f, ((i & 1u) == 1u))` —
        // a swapped sign emits `select(-1f, 1f, ...)` and a constant sign emits no `select` at all.
        expect(wgsl).toContain("select(1f, -1f, ((i & 1u) == 1u))");

        // slot index: the kernel reads segment 0, emitted as `segments[0i]` — a wrong slot index
        // emits `segments[1i]` or similar.
        expect(wgsl).toContain("segments[0i]");
    });
});

describe("post station derivation", () => {
    test("postStation(i) = (i + 1) * POST_SPACING — first post at POST_SPACING, not station 0", () => {
        // independently derived: (i + 1) * POST_SPACING, not i * POST_SPACING
        // red: mutating postStation to return i * POST_SPACING makes postStation(0) = 0 (not 20)
        for (const i of [0, 1, 2, 5, 10, 72]) {
            expect(postStation(i)).toBe((i + 1) * POST_SPACING);
        }
    });

    test("postStation is strictly increasing — no two slots share a station", () => {
        for (let i = 0; i < 10; i++) {
            expect(postStation(i + 1)).toBeGreaterThan(postStation(i));
        }
    });
});

describe("post lateral derivation", () => {
    test("postLateralSign alternates: +1 for even, -1 for odd", () => {
        // independently derived: even → +1, odd → -1
        // red: mutating postLateralSign to always return 1 makes postLateralSign(1) = 1 (not -1)
        expect(postLateralSign(0)).toBe(1);
        expect(postLateralSign(1)).toBe(-1);
        expect(postLateralSign(2)).toBe(1);
        expect(postLateralSign(3)).toBe(-1);
    });
});

describe("live-slot count derivation", () => {
    test("liveSlotCount = floor(chordLength / POST_SPACING) — not ceil or floor + 1", () => {
        // independently derived: Math.floor(chordLength / POST_SPACING)
        // red: mutating liveSlotCount to Math.ceil makes liveSlotCount(201) = 11 (not 10)
        expect(liveSlotCount(200)).toBe(10);
        expect(liveSlotCount(201)).toBe(10);
        expect(liveSlotCount(199)).toBe(9);
        expect(liveSlotCount(0)).toBe(0);
        expect(liveSlotCount(POST_SPACING)).toBe(1);
        expect(liveSlotCount(POST_SPACING - 0.001)).toBe(0);
    });

    test("isLiveSlot agrees with liveSlotCount — the count of live slots matches", () => {
        // the arm and the production derivation come from the same expression:
        // isLiveSlot(i, L) = postStation(i) <= L, and liveSlotCount(L) = #{i : postStation(i) <= L}
        for (const L of [100, 200, 201, 500]) {
            let count = 0;
            for (let i = 0; i < POST_COUNT; i++) {
                if (isLiveSlot(i, L)) count++;
            }
            expect(count).toBe(liveSlotCount(L));
        }
    });

    test("isLiveSlot boundary: station == chordLength is live, station > chordLength is not", () => {
        // the boundary is inclusive: station <= chordLength is live
        // red: mutating isLiveSlot to use < instead of <= makes the station == chordLength case dead
        const i = 5;
        const station = postStation(i);
        expect(isLiveSlot(i, station)).toBe(true);
        expect(isLiveSlot(i, station - 0.001)).toBe(false);
        expect(isLiveSlot(i, station + 0.001)).toBe(true);
    });
});

describe("POST_COUNT sizing", () => {
    test("POST_COUNT = ceil(WORLD_DIAGONAL / POST_SPACING), fixed across edits", () => {
        // independently derived: ceil(WORLD_EXTENT * SQRT2 / POST_SPACING)
        const worldDiagonal = WORLD_EXTENT * Math.SQRT2;
        expect(POST_COUNT).toBe(Math.ceil(worldDiagonal / POST_SPACING));
    });

    test("POST_COUNT is large enough for the worst-case (corner-to-corner) chord", () => {
        // every live slot must fit: floor(worldDiagonal / POST_SPACING) <= POST_COUNT
        const worldDiagonal = WORLD_EXTENT * Math.SQRT2;
        expect(liveSlotCount(worldDiagonal)).toBeLessThanOrEqual(POST_COUNT);
    });
});

describe("POST_OFFSET flat-core band", () => {
    test("POST_OFFSET is strictly inside the flat core: 0 < POST_OFFSET < FLAT_CORE_MARGIN", () => {
        // independently derived from flatten-math.ts:
        // the flat core extends to halfWidth + FLAT_CORE_MARGIN from the centreline;
        // a post at halfWidth + POST_OFFSET is inside iff POST_OFFSET < FLAT_CORE_MARGIN;
        // outside the road iff POST_OFFSET > 0.
        // red: mutating POST_OFFSET to FLAT_CORE_MARGIN makes this fail (not strictly inside)
        // red: mutating POST_OFFSET to 0 makes this fail (on the road surface)
        expect(POST_OFFSET).toBeGreaterThan(0);
        expect(POST_OFFSET).toBeLessThan(FLAT_CORE_MARGIN);
    });

    test("POST_OFFSET = SPACING (one grid cell from the road edge)", () => {
        expect(POST_OFFSET).toBe(SPACING);
    });

    test("FLAT_CORE_MARGIN = √2 * SPACING (flatten-math.ts's own derivation)", () => {
        // the constant the band is derived against, re-checked here so the band test
        // doesn't silently pass if FLAT_CORE_MARGIN moves
        expect(FLAT_CORE_MARGIN).toBe(Math.SQRT2 * SPACING);
    });
});
