import { describe, expect, test } from "bun:test";
import { flat } from "../../../../packages/shallot/tests/wgsl";
import { mulberry32 } from "./dragCorpus";
import { clampToBound } from "./editPure";
import { ROAD_MIN_LENGTH } from "./overlay/network";
import {
    isLiveSlot,
    liveSlotCount,
    MAX_CHORD_GRADE,
    POST_BURIAL_DEPTH,
    POST_COLOR,
    POST_COUNT,
    POST_HEIGHT,
    POST_MESH,
    POST_OFFSET,
    POST_RADIUS,
    POST_SHAFT_LENGTH,
    POST_SPACING,
    postLateralSign,
    postStation,
    postsSurfaceWgsl,
    postsWgsl,
    postVertexOffset,
} from "./posts";
import { FLAT_CORE_MARGIN } from "./terrain/flatten";
import { SPACING, WORLD_EXTENT, WORLD_HALF } from "./terrain/grid";
import { makePermutation, RELIEF } from "./terrain/noise";
import { heightAtCpu } from "./terrain/profile";

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
        // pin the exact emitted WGSL text: the station `(f32(i) + 1f) * 2f` (not `f32(i) * 2f`),
        // the lateral `select(1f, -1f, ((i & 1u) == 1u))` (not a swapped or constant sign), and the slot
        // index `segments[0i]` (not `segments[1i]`). Each would fail under the mutation it names.
        const wgsl = flat(postsWgsl());

        // station: (i + 1) * POST_SPACING emits as `(f32(i) + 1f) * 2f` — the `+ 1f` is what
        // discriminates from `i * POST_SPACING` which emits `(f32(i) * 2f)` with no `+ 1f`.
        expect(wgsl).toContain("(f32(i) + 1f) * 2f");

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
        // red: mutating postStation to return i * POST_SPACING makes postStation(0) = 0 (not 2)
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
        // red: mutating liveSlotCount to Math.ceil makes liveSlotCount(201) = 101 (not 100)
        expect(liveSlotCount(200)).toBe(100);
        expect(liveSlotCount(201)).toBe(100);
        expect(liveSlotCount(199)).toBe(99);
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
    test("the whole footing is strictly inside the flat core: POST_RADIUS < POST_OFFSET < FLAT_CORE_MARGIN − POST_RADIUS", () => {
        // independently derived from flatten.ts: the flat core reaches halfWidth + FLAT_CORE_MARGIN
        // from the centreline, and the post occupies halfWidth + POST_OFFSET ± POST_RADIUS — so the whole
        // footing is strictly inside the affine region (where the mesh reproduces the field exactly, the
        // property the placement oracle reads) and strictly off the pavement iff both bounds below hold.
        // Stage 11 re-checked the LOWER end on purpose: 0.4 m approaches it, and the older
        // `0 < POST_OFFSET` form ignored the post's own radius.
        // red: POST_OFFSET = 0.1 (< POST_RADIUS) fails the lower bound — the footing overhangs the asphalt
        // red: POST_OFFSET = 5.6 (> FLAT_CORE_MARGIN − POST_RADIUS) fails the upper bound
        expect(POST_OFFSET).toBeGreaterThan(POST_RADIUS);
        expect(POST_OFFSET).toBeLessThan(FLAT_CORE_MARGIN - POST_RADIUS);
    });

    test("the measured margins at the kerb-line offset are the ones written beside the constant", () => {
        // the measurement the stage reports, asserted rather than printed: 0.28 m of clearance from the
        // pavement edge to the post's near face, 5.13685 m from its far face to the flat core's edge.
        expect(POST_OFFSET - POST_RADIUS).toBeCloseTo(0.28, 6);
        expect(FLAT_CORE_MARGIN - (POST_OFFSET + POST_RADIUS)).toBeCloseTo(5.136854, 5);
    });

    test("POST_OFFSET is the kerb line (~0.4 m), not stage 5's flat-core convenience (SPACING = 4 m)", () => {
        // the referent's own value, pinned as a literal: a kerbside bollard stands immediately off the
        // pavement edge. 4 m (13 ft) is what made the row read as posts standing in a field. The literal
        // is the whole assertion — a companion `not.toBe(SPACING)` was dropped at stage 11's repair as
        // vacuous, since `toBe(0.4)` already refutes every value `SPACING` could hold.
        expect(POST_OFFSET).toBe(0.4);
    });

    test("FLAT_CORE_MARGIN = √2 * SPACING (flatten.ts's own derivation)", () => {
        // the constant the band is derived against, re-checked here so the band test
        // doesn't silently pass if FLAT_CORE_MARGIN moves
        expect(FLAT_CORE_MARGIN).toBe(Math.SQRT2 * SPACING);
    });
});

describe("the referent's own dimensions", () => {
    test("POST_SPACING is held at the kerb row's 2.0 m upper bound", () => {
        expect(POST_SPACING).toBe(2);
    });

    test("POST_HEIGHT is the referent's ~1 m above-grade height and POST_RADIUS its pipe OD/2", () => {
        expect(POST_HEIGHT).toBe(1);
        expect(POST_RADIUS).toBe(0.12);
        // the pipe the referent is made of: 8 in Sch 40 OD 219 mm (r = 0.1095) → 10 in Sch 40 OD 273 mm
        // (r = 0.1365). The radius sits between the two.
        expect(POST_RADIUS).toBeGreaterThan(0.1095);
        expect(POST_RADIUS).toBeLessThan(0.1365);
    });

    test("POST_COLOR is RAL 1023 traffic yellow, and the emitted FS carries it", () => {
        // the finish standard, as a literal triple — stage 5's [0.5, 0.4, 0.3] had no referent at all.
        expect(POST_COLOR).toEqual([0.941, 0.792, 0.0]);
        // and the FS actually uses it: the resolved surface WGSL carries the f32 rounding of those
        // components. red: reverting POST_COLOR to [0.5, 0.4, 0.3] emits 0.5/0.4/0.30000001 instead.
        const wgsl = flat(postsSurfaceWgsl());
        expect(wgsl).toContain("0.9409999");
        expect(wgsl).toContain("0.7919999");
    });
});

describe("the footing depth's derivation", () => {
    test("MAX_CHORD_GRADE is the analytic ceiling 2 · RELIEF / ROAD_MIN_LENGTH", () => {
        // the chord's target height is linear between the endpoints' natural heights, so the along-chord
        // grade is |Δh| / chordLength; |Δh| <= 2 · RELIEF and chordLength >= ROAD_MIN_LENGTH.
        expect(MAX_CHORD_GRADE).toBe((2 * RELIEF) / ROAD_MIN_LENGTH);
        expect(MAX_CHORD_GRADE).toBe(1);
    });

    test("no admissible chord's grade exceeds the ceiling the footing is sized against", () => {
        // the measurement the stage reports, asserted: a 5-seed × 400-chord scan of admissible chords
        // (both endpoints clamped into the world by the live drag's own `clampToBound`, length >=
        // ROAD_MIN_LENGTH) against the analytic ceiling. Half the corpus is drawn as SHORT chords — length
        // in [ROAD_MIN_LENGTH, 2 · ROAD_MIN_LENGTH] — because grade is |Δh| / length and a uniform endpoint
        // pair is almost always long: measured at stage 11's repair, the corpus reads **0.151686** while
        // its uniform-endpoint half alone reads **0.123455**.
        //
        // **Why this scan and not `dragCorpus`** (stage 12's frozen fixture, the one derivation both
        // flatness tiers read): `dragCorpus` *filters by* `MAX_GRADE = 0.12` — it rejects exactly the steep
        // chords this arm exists to find, because the flatness oracle's longitudinal bound is that grade.
        // A corpus that discards its steepest members cannot measure the steepest member. What this arm
        // does take from the live path is the **bound**: `clampToBound` (`editPure.ts`), so "admissible"
        // here means what the drag means by it, rather than the `WORLD_HALF − SPACING` this arm used to
        // hand-roll (equal at 4 m today by coincidence — `BOUND_MARGIN = ROAD_HALF_WIDTH`, an unrelated
        // concept). The RNG is the repo's own `mulberry32`, so this file hand-rolls no generator either.
        let worst = 0;
        const rand = mulberry32(0x9e3779b9);
        for (const seed of [1337, 1, 2, 99, 4242]) {
            const perm = makePermutation(seed);
            for (let n = 0; n < 400; n++) {
                const [ax, az] = clampToBound(
                    (rand() * 2 - 1) * WORLD_HALF,
                    (rand() * 2 - 1) * WORLD_HALF,
                );
                let bx: number;
                let bz: number;
                if (n % 2 === 0) {
                    [bx, bz] = clampToBound(
                        (rand() * 2 - 1) * WORLD_HALF,
                        (rand() * 2 - 1) * WORLD_HALF,
                    );
                } else {
                    const theta = rand() * Math.PI * 2;
                    const L = ROAD_MIN_LENGTH * (1 + rand());
                    [bx, bz] = clampToBound(ax + Math.cos(theta) * L, az + Math.sin(theta) * L);
                }
                const len = Math.hypot(bx - ax, bz - az);
                if (len < ROAD_MIN_LENGTH) continue;
                const grade = Math.abs(heightAtCpu(bx, bz, perm) - heightAtCpu(ax, az, perm)) / len;
                worst = Math.max(worst, grade);
            }
        }
        expect(worst).toBeLessThanOrEqual(MAX_CHORD_GRADE);
        // a **corpus non-vacuity tripwire**, and nothing more — restated at stage 11's repair, because the
        // old wording ("grades steep enough that the footing matters") was false at this value: grade 0.1
        // needs 0.012 m of burial, 5 % of the shipped 0.24 m. **No arm in this file can witness that the
        // depth is needed**, because the domain it is sized for (grade → MAX_CHORD_GRADE = 1.0) is ~6.6×
        // outside anything this terrain reaches — the footing is sized against the admissible domain by
        // construction, which is the trade the derivation states. What this floor catches is a scan that
        // stopped scanning: a broken RNG, a clamp that collapses every chord, or a filter that rejects the
        // whole corpus would all read near 0 and are all silent against the ceiling assertion above. The
        // literal sits well under the reading (0.151686) rather than at it, so a seed change does not red
        // it — and it is under the uniform half's own 0.123455 too, so it survives losing the short-chord
        // half entirely.
        expect(worst).toBeGreaterThan(0.1);
    });

    test("POST_BURIAL_DEPTH buries the base ring at the worst grade, with the diameter form's 2× margin", () => {
        // required depth: the base ring's uphill side sits grade · POST_RADIUS above the field height
        // sampled at the post's centre (the field is exactly flat laterally inside the flat core, so the
        // along-chord grade is the only rise across the footprint).
        const required = MAX_CHORD_GRADE * POST_RADIUS;
        expect(POST_BURIAL_DEPTH).toBe(MAX_CHORD_GRADE * 2 * POST_RADIUS);
        expect(POST_BURIAL_DEPTH).toBeCloseTo(0.24, 12);
        expect(POST_BURIAL_DEPTH).toBeCloseTo(2 * required, 12);
    });

    test("POST_SHAFT_LENGTH is derived so the footing adds below the referent's height, never out of it", () => {
        expect(POST_SHAFT_LENGTH).toBeCloseTo(POST_HEIGHT + POST_BURIAL_DEPTH - POST_RADIUS, 12);
        expect(POST_SHAFT_LENGTH).toBeCloseTo(1.12, 12);
    });
});

describe("the capsule's core/cap decomposition (the VS's own arithmetic)", () => {
    test("the above-grade extent is exactly POST_HEIGHT — the footing never eats the referent's height", () => {
        // the dome's apex is the mesh's highest point (localY = 1).
        // red: taking the burial out of the height (shaftLength = POST_HEIGHT − POST_RADIUS) reads 0.76.
        expect(postVertexOffset(0, 1, 0)[1]).toBeCloseTo(POST_HEIGHT, 12);
    });

    test("the bottom cap's highest point is below the surface for the worst grade the chord allows", () => {
        // the bottom cap's highest point is the shaft's base ring (localY = −0.5), and the uphill side of
        // that ring sits MAX_CHORD_GRADE · POST_RADIUS above the sampled centre height — so the ring must
        // sit at least that far below the surface, and the cap's own lowest point below it again.
        // red-first, witnessed 2026-08-22 at `POST_BURIAL_DEPTH = 0`: the base ring sits flush at y = 0, so
        // the uphill side of the ring reads 0.12 m ABOVE the surface — `Expected: <= 0, Received: 0.12` —
        // and three more arms red with it (the depth, the derived shaft length, and the emitted VS
        // literals). **`burial = 0` is NOT the shape stage 5 shipped**, and calling it that would overclaim
        // the witness: stage 5 mapped the mesh's *lowest point* to the surface, putting the base ring
        // 0.25 m ABOVE grade with the whole bottom hemisphere on show, so no value of
        // `POST_BURIAL_DEPTH` reproduces it — `burial = 0` is already a strictly better shape (0.12 m of
        // exposure against 0.25 m). The pre-image witness is the companion assertion in the emitted-VS arm
        // below, `not.toContain("input.localPos.y * 0.5f")`, which is stage 5's actual y scale. This arm's
        // mutation witnesses that the *depth* is load-bearing, not that stage 5 is refuted.
        const baseRing = postVertexOffset(0, -0.5, 0)[1];
        expect(baseRing).toBeCloseTo(-POST_BURIAL_DEPTH, 12);
        expect(baseRing + MAX_CHORD_GRADE * POST_RADIUS).toBeLessThanOrEqual(0);
        expect(postVertexOffset(0, -1, 0)[1]).toBeLessThan(baseRing);
    });

    test("both caps are spheres of exactly POST_RADIUS, independent of the shaft length", () => {
        // the property an arm over total height alone cannot see — and the defect that shipped: stage 5
        // scaled y by POST_HEIGHT/2 and x/z by POST_RADIUS · 2, stretching each cap into an ellipsoid
        // 0.25 m tall and 0.12 m wide (a bullet nose). Here the cap term takes the SAME factor x/z take,
        // so every cap vertex sits at exactly POST_RADIUS from its cap centre at any radius/length ratio.
        // red: scaling the cap by shaftLength instead of the radius makes the distance vary with the
        // shaft length (and differ from POST_RADIUS at every angle but 0).
        for (const shaftLength of [0.3, POST_SHAFT_LENGTH, 5]) {
            const params = { ...POST_MESH, shaftLength };
            const topCentre = shaftLength - POST_MESH.burial;
            const bottomCentre = -POST_MESH.burial;
            for (const theta of [0, 0.3, Math.PI / 4, 1.2, Math.PI / 2]) {
                const lx = 0.5 * Math.cos(theta);
                const lyTop = 0.5 + 0.5 * Math.sin(theta);
                const top = postVertexOffset(lx, lyTop, 0, params);
                expect(Math.hypot(top[0], top[1] - topCentre, top[2])).toBeCloseTo(POST_RADIUS, 12);
                const bottom = postVertexOffset(lx, -lyTop, 0, params);
                expect(Math.hypot(bottom[0], bottom[1] - bottomCentre, bottom[2])).toBeCloseTo(
                    POST_RADIUS,
                    12,
                );
            }
        }
    });

    test("the shaft is a cylinder of POST_RADIUS — the core term carries the length, the radial term the width", () => {
        // the discriminating half is the RADIAL one: `side[0] ≈ POST_RADIUS` at every height, which is
        // what makes the shaft a cylinder rather than a taper. The y half is pinned as **literals** at the
        // shaft's two ends rather than as the function's own expression — restating
        // `ly * shaftLength + shaftLength/2 − burial` here would re-derive the subject's own rule and go
        // green on wrong constants (stage 3's precedent, and the defect this repair closes).
        for (const ly of [-0.5, -0.2, 0, 0.25, 0.5]) {
            const side = postVertexOffset(0.5, ly, 0);
            expect(side[0]).toBeCloseTo(POST_RADIUS, 12);
        }
        // the base ring sits POST_BURIAL_DEPTH below the surface: −0.24 m.
        expect(postVertexOffset(0.5, -0.5, 0)[1]).toBeCloseTo(-0.24, 12);
        // the dome's underside sits at shaftLength − burial = 1.12 − 0.24 = 0.88 m above it.
        expect(postVertexOffset(0.5, 0.5, 0)[1]).toBeCloseTo(0.88, 12);
        // and the z factor is the same radial one x takes — asserted at nonzero `lz`, which no other arm
        // does (every other mesh arm passes `lz = 0`, where a dropped or wrong z factor is invisible).
        expect(postVertexOffset(0, 0, 0.5)[2]).toBeCloseTo(POST_RADIUS, 12);
        expect(postVertexOffset(0, 0, -0.5)[2]).toBeCloseTo(-POST_RADIUS, 12);
        expect(postVertexOffset(0, 0, 0.25)[2]).toBeCloseTo(POST_RADIUS / 2, 12);
    });

    test("a scale-0 slot collapses to the record position (the fixed-instanceCount mechanism)", () => {
        // component-wise, because a collapsed −z vertex reads as −0 and `toEqual` distinguishes it
        const collapsed = postVertexOffset(0.5, 1, -0.5, POST_MESH, 0);
        expect(collapsed[0]).toBeCloseTo(0, 12);
        expect(collapsed[1]).toBeCloseTo(0, 12);
        expect(collapsed[2]).toBeCloseTo(0, 12);
    });

    test("the resolved VS emits the decomposition, with the cap and the radius sharing one factor", () => {
        // structural, over the emitted text, because the sphericity is a property of WHICH factor the cap
        // term takes: `cap * radial` and `localPos.x * radial` must be the same `radial`, and only the
        // core may carry the shaft length. Literals rather than values re-derived from the constants
        // (stage 3's precedent).
        const wgsl = flat(postsSurfaceWgsl());
        expect(wgsl).toContain("clamp(input.localPos.y, -0.5f, 0.5f)");
        expect(wgsl).toContain("(input.localPos.y - core)");
        expect(wgsl).toContain("(cap * radial)");
        expect(wgsl).toContain("(input.localPos.x * radial)");
        expect(wgsl).toContain("(input.localPos.z * radial)");
        // and `radial`'s own VALUE, not just its identifier — added at stage 11's repair, which found that
        // the three assertions above pin only that one shared factor exists and pass unchanged under any
        // value for it. Nothing else in the suite reads the VS's numbers: `postVertexOffset` carries its
        // own `params.radius * 2`, `checkPosts` reads records rather than vertices, and the fs probes are
        // on-road while the posts are now off it — so the post's rendered width and its cap radius, this
        // stage's headline property, were unpinned in production. POST_RADIUS * 2 = 0.24 in f32.
        // red witnessed 2026-08-22 — the VS's own `radial` mutated to `d.f32(POST_RADIUS)` (dropping the
        // mesh's 0.5-radius compensation, the plausible wrong factor), this file run, the production line
        // then restored byte-identical. Verbatim, with the emitted WGSL of `Received` elided after the
        // decomposition:
        //   error: expect(received).toMatch(expected)
        //   Expected substring or pattern: /radial = 0\.2399\d*f/
        //   Received: "... let core = clamp(input.localPos.y, -0.5f, 0.5f); let cap =
        //   (input.localPos.y - core); const radial = 0.11999999731779099f; let sx =
        //   ((input.localPos.x * radial) * scale); ..."
        //   at <anonymous> (.../examples/showcase/roads/src/posts.test.ts, the assertion below)
        // (26 pass / 1 fail — this arm alone, which is the point: nothing else in the file moved.)
        expect(wgsl).toMatch(/radial = 0\.2399\d*f/);
        // the shaft length on the core term, and the burial in the translation — f32 roundings of
        // 1.12 and 1.12/2 − 0.24 = 0.32.
        expect(wgsl).toMatch(/core \* 1\.12\d*f/);
        expect(wgsl).toMatch(/\+ 0\.3199\d*f/);
        // and the y scale stage 5 shipped (POST_HEIGHT / 2 = 0.5) is gone from the VS
        expect(wgsl).not.toContain("input.localPos.y * 0.5f");
    });
});
