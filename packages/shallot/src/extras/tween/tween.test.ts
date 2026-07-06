import { beforeEach, describe, expect, test } from "bun:test";
import { stepFor } from "../../../tests/helpers";
import { build, f32, load, type Plugin, parse, type State, sparse, vec4 } from "../..";
import { clear } from "../../engine/ecs/core";
import { Sequence, sequence, Tween, TweenPlugin, TweenState, tween } from "./";
import {
    Composite,
    EASING_FUNCTIONS,
    Fill,
    getEasing,
    getEasingIndex,
    getEasingName,
    owns,
    sample,
} from "./core";

describe("tween atom — owns", () => {
    const D = 1;

    test("forwards (default) owns the active interval and the after tail, not before", () => {
        expect(owns(-0.1, D, Fill.Forwards)).toBe(false);
        expect(owns(0, D, Fill.Forwards)).toBe(true);
        expect(owns(0.5, D, Fill.Forwards)).toBe(true);
        expect(owns(D, D, Fill.Forwards)).toBe(true);
        expect(owns(D + 0.1, D, Fill.Forwards)).toBe(true);
    });

    test("none owns only the active interval", () => {
        expect(owns(-0.1, D, Fill.None)).toBe(false);
        expect(owns(0, D, Fill.None)).toBe(true);
        expect(owns(0.5, D, Fill.None)).toBe(true);
        expect(owns(D, D, Fill.None)).toBe(false);
        expect(owns(D + 0.1, D, Fill.None)).toBe(false);
    });

    test("backwards owns the before tail and the active interval, not after", () => {
        expect(owns(-0.1, D, Fill.Backwards)).toBe(true);
        expect(owns(0.5, D, Fill.Backwards)).toBe(true);
        expect(owns(D + 0.1, D, Fill.Backwards)).toBe(false);
    });

    test("both owns every phase", () => {
        expect(owns(-0.1, D, Fill.Both)).toBe(true);
        expect(owns(0.5, D, Fill.Both)).toBe(true);
        expect(owns(D + 0.1, D, Fill.Both)).toBe(true);
    });

    test("instant (duration 0) is owned at and after zero under forwards, not before", () => {
        expect(owns(-0.1, 0, Fill.Forwards)).toBe(false);
        expect(owns(0, 0, Fill.Forwards)).toBe(true);
        expect(owns(0.1, 0, Fill.Forwards)).toBe(true);
    });
});

describe("tween atom — sample", () => {
    test("replace linear interpolates the midpoint", () => {
        expect(sample(0.5, 1, 0, 0, 100, Composite.Replace, 0)).toBe(50);
    });

    test("replace saturates: before samples the start, after samples the end", () => {
        expect(sample(-1, 1, 0, 10, 90, Composite.Replace, 0)).toBe(10);
        expect(sample(0, 1, 0, 10, 90, Composite.Replace, 0)).toBe(10);
        expect(sample(1, 1, 0, 10, 90, Composite.Replace, 0)).toBe(90);
        expect(sample(2, 1, 0, 10, 90, Composite.Replace, 0)).toBe(90);
    });

    test("instant (duration 0) samples the end value", () => {
        expect(sample(0, 0, 0, 10, 90, Composite.Replace, 0)).toBe(90);
    });

    test("replace applies the easing curve", () => {
        const i = getEasingIndex("ease-in-quad");
        // ease-in-quad(0.5) = 0.25, so the eased midpoint is 25 — below the linear 50
        expect(sample(0.5, 1, i, 0, 100, Composite.Replace, 0)).toBe(25);
    });

    test("add lays the sampled value on top of the base", () => {
        expect(sample(0.5, 1, 0, 0, 100, Composite.Add, 7)).toBe(57);
    });

    test("add with a delta that settles to zero leaves the base untouched (SSOT relaxation)", () => {
        // a recoil: from a kick of 5 back to 0. at completion the delta is 0,
        // so the base — whatever a sim wrote — is returned exactly.
        expect(sample(1, 1, 0, 5, 0, Composite.Add, 42.5)).toBe(42.5);
    });
});

describe("easing set", () => {
    test("ships all 31 curves; unknown name and out-of-range index fall back to linear", () => {
        expect(EASING_FUNCTIONS.length).toBe(31);
        expect(getEasingIndex("nope")).toBe(0);
        expect(getEasingName(999)).toBe("linear");
        expect(getEasing(999)(0.5)).toBe(0.5);
    });

    test("every curve is normalized at the endpoints and finite across the interval", () => {
        // the easing contract — a curve reparametrizes [0,1]→[0,1] with fixed
        // endpoints, only the path between differs. transcendental curves
        // (sine/expo/elastic) land within float round-off of the endpoint. the
        // interior samples drive the in-vs-out arms of the in-out curves.
        for (const ease of EASING_FUNCTIONS) {
            expect(ease(0)).toBeCloseTo(0, 10);
            expect(ease(1)).toBeCloseTo(1, 10);
            expect(Number.isFinite(ease(0.25))).toBe(true);
            expect(Number.isFinite(ease(0.75))).toBe(true);
        }
    });

    test("name ↔ index round-trips for every curve", () => {
        for (let i = 0; i < EASING_FUNCTIONS.length; i++) {
            expect(getEasingIndex(getEasingName(i))).toBe(i);
        }
    });

    test("the curve array and the name index stay aligned", () => {
        // two hand-maintained lists; spot-checking the first curves pins that
        // EASING_FUNCTIONS[getEasingIndex(name)] is actually that named curve
        expect(getEasing(getEasingIndex("linear"))(0.5)).toBe(0.5);
        expect(getEasing(getEasingIndex("ease-in-quad"))(0.5)).toBe(0.25);
        expect(getEasing(getEasingIndex("ease-out-quad"))(0.5)).toBe(0.75);
    });
});

// The Tween effect. The component + the sampling system + the parse-time-bound
// typed setter, driven programmatically via `tween()`. A Single `sparse(f32)`
// field is a plain f64 Map (no quantization). The values below are exact
// multiples in real arithmetic, but a multi-step run reads them through a clock
// summed from f64 step dts, so they carry float round-off (≤1e-13 here) and
// assert at the exact tier (1e-10, testing.md) — tight enough to catch any
// behavior change. A single `step(0)` or a directly set `elapsed` has no sum to
// accumulate, so those stay `toBe`.

const Probe = { value: sparse(f32), vec: sparse(vec4) };
const TestPlugin: Plugin = {
    name: "Test",
    components: { Probe },
    traits: { Probe: { defaults: () => ({ value: 0, vec: [0, 0, 0, 0] }) } },
};

describe("tween effect", () => {
    let state: State;
    let target: number;

    beforeEach(async () => {
        clear();
        ({ state } = await build({ plugins: [TweenPlugin, TestPlugin], defaults: false }));
        target = state.create();
        state.add(target, Probe);
    });

    test("interpolates a scalar field over time", () => {
        tween(state, target, "probe.value", { to: 100, duration: 1 });
        stepFor(state, 0.5);
        expect(Probe.value.get(target)).toBeCloseTo(50, 10);
        stepFor(state, 0.5);
        expect(Probe.value.get(target)).toBeCloseTo(100, 10);
    });

    test("captures the current value as the start", () => {
        Probe.value.set(target, 40);
        tween(state, target, "probe.value", { to: 140, duration: 1 });
        stepFor(state, 0.5);
        expect(Probe.value.get(target)).toBeCloseTo(90, 10); // 40 → 140 at 0.5
    });

    test("an explicit from animates fromTo", () => {
        Probe.value.set(target, 999); // ignored — from is explicit
        tween(state, target, "probe.value", { from: 20, to: 100, duration: 1 });
        stepFor(state, 0.5);
        expect(Probe.value.get(target)).toBeCloseTo(60, 10); // 20 → 100 at 0.5
    });

    test("applies the easing curve", () => {
        tween(state, target, "probe.value", { to: 100, duration: 1, easing: "ease-in-quad" });
        stepFor(state, 0.5);
        expect(Probe.value.get(target)).toBeCloseTo(25, 10); // ease-in-quad(0.5) = 0.25 → 25
    });

    test("marks Complete and holds the end (default forwards fill)", () => {
        const t = tween(state, target, "probe.value", { to: 100, duration: 0.5 })!;
        stepFor(state, 0.6);
        expect(Tween.state.get(t)).toBe(TweenState.Complete);
        expect(Probe.value.get(target)).toBeCloseTo(100, 10);
        Probe.value.set(target, 555);
        state.step(1 / 60);
        expect(Probe.value.get(target)).toBeCloseTo(100, 10); // forwards holds the end
    });

    test("Idle pauses; replaying resumes from the same elapsed", () => {
        const t = tween(state, target, "probe.value", { to: 100, duration: 1 })!;
        stepFor(state, 0.25);
        expect(Probe.value.get(target)).toBeCloseTo(25, 10);
        Tween.state.set(t, TweenState.Idle);
        stepFor(state, 0.25);
        expect(Probe.value.get(target)).toBeCloseTo(25, 10);
        Tween.state.set(t, TweenState.Playing);
        stepFor(state, 0.5);
        expect(Probe.value.get(target)).toBeCloseTo(75, 10); // resumed: 0.25 + 0.5
    });

    test("seeks via a writable elapsed", () => {
        const t = tween(state, target, "probe.value", { to: 100, duration: 1 })!;
        stepFor(state, 0.1); // captures from = 0
        Tween.elapsed.set(t, 0.8);
        state.step(0);
        expect(Probe.value.get(target)).toBe(80);
    });

    test("tweens a vector lane independently", () => {
        tween(state, target, "probe.vec.x", { to: 100, duration: 1 });
        tween(state, target, "probe.vec.y", { to: 200, duration: 1 });
        stepFor(state, 0.5);
        expect(Probe.vec.x.get(target)).toBeCloseTo(50, 10);
        expect(Probe.vec.y.get(target)).toBeCloseTo(100, 10);
        expect(Probe.vec.z.get(target)).toBe(0); // untouched — exact default
    });

    test("returns null for a path that doesn't name a scalar field", () => {
        expect(tween(state, target, "value", { to: 1 })).toBeNull(); // no component
        expect(tween(state, target, "probe.missing", { to: 1 })).toBeNull(); // no field
        expect(tween(state, target, "unknown.x", { to: 1 })).toBeNull(); // no component
        expect(tween(state, target, "probe.vec", { to: 1 })).toBeNull(); // vector, not scalar
    });

    test("instant (duration 0) jumps to the end", () => {
        Probe.value.set(target, 5);
        const t = tween(state, target, "probe.value", { to: 100, duration: 0 })!;
        state.step(0);
        expect(Probe.value.get(target)).toBe(100);
        expect(Tween.state.get(t)).toBe(TweenState.Complete);
    });

    test("loop rewinds on completion and never completes", () => {
        const t = tween(state, target, "probe.value", {
            from: 0,
            to: 100,
            duration: 0.1,
            loop: true,
        })!;
        let sawHigh = false;
        let sawLowAfterHigh = false;
        for (let i = 0; i < 20; i++) {
            state.step(0.03); // 30ms < the dt clamp, so each step lands cleanly
            const v = Probe.value.get(target);
            if (v > 80) sawHigh = true;
            else if (sawHigh && v < 20) sawLowAfterHigh = true;
        }
        expect(Tween.state.get(t)).toBe(TweenState.Playing); // loops, so never Complete
        expect(sawLowAfterHigh).toBe(true); // wrapped back toward the start
    });
});

describe("tween composite + fill", () => {
    let state: State;
    let target: number;

    beforeEach(async () => {
        clear();
        ({ state } = await build({ plugins: [TweenPlugin, TestPlugin], defaults: false }));
        target = state.create();
        state.add(target, Probe);
    });

    test("add lays the delta on top of the base, default start at zero", () => {
        Probe.value.set(target, 100);
        // no explicit from — add defaults the delta's start to 0 (not a captured value)
        tween(state, target, "probe.value", { to: 4, composite: "add", duration: 0 });
        state.step(0);
        expect(Probe.value.get(target)).toBe(104); // base 100 + delta (0 → 4)
    });

    test("add over a sim-written field lays on top, then restores the base on completion", () => {
        // the SSOT relaxation: a recoil delta (6 → 0) over a field a sim writes
        // authoritatively each frame. mid-tween the delta rides on top; on
        // completion it's zero and the base is untouched.
        tween(state, target, "probe.value", { from: 6, to: 0, composite: "add", duration: 0.1 });
        const dt = 1 / 60;
        Probe.value.set(target, 50);
        state.step(dt);
        expect(Probe.value.get(target)).toBeGreaterThan(50); // delta laid on top
        expect(Probe.value.get(target)).toBeLessThanOrEqual(56);
        for (let t = dt; t <= 0.25; t += dt) {
            Probe.value.set(target, 50); // sim writes the authoritative base each frame
            state.step(dt);
        }
        expect(Probe.value.get(target)).toBe(50); // delta gone, base intact
    });

    test("fill none releases the field after the active interval", () => {
        const t = tween(state, target, "probe.value", { to: 100, fill: "none", duration: 0.1 })!;
        stepFor(state, 0.2);
        expect(Tween.state.get(t)).toBe(TweenState.Complete);
        Probe.value.set(target, 555); // external write after the tween released
        state.step(1 / 60);
        expect(Probe.value.get(target)).toBe(555); // none doesn't hold — not overwritten
    });
});

describe("tween sequence", () => {
    let state: State;
    let target: number;

    beforeEach(async () => {
        clear();
        ({ state } = await build({ plugins: [TweenPlugin, TestPlugin], defaults: false }));
        target = state.create();
        state.add(target, Probe);
    });

    test("plays positioned tweens in order on one looping clock", () => {
        const seq = sequence(state, { loop: true });
        // up over [0, 0.1] then down over [0.1, 0.2] on the same field; fill none hands off
        tween(state, target, "probe.value", {
            from: 0,
            to: 100,
            at: 0,
            duration: 0.1,
            fill: "none",
            sequence: seq,
        });
        tween(state, target, "probe.value", {
            from: 100,
            to: 0,
            at: 0.1,
            duration: 0.1,
            fill: "none",
            sequence: seq,
        });

        expect(Sequence.duration.get(seq)).toBeCloseTo(0.2, 5); // period spans both children

        state.step(0.025); // up at 25% → 25 (down idle, before its window)
        expect(Probe.value.get(target)).toBeCloseTo(25, 10);
        state.step(0.05); // elapsed 0.075, up at 75% → 75
        expect(Probe.value.get(target)).toBeCloseTo(75, 10);
        state.step(0.05); // elapsed 0.125, up released, down at 25% → 75 (falling)
        expect(Probe.value.get(target)).toBeCloseTo(75, 10);
        state.step(0.05); // elapsed 0.175, down at 75% → 25
        expect(Probe.value.get(target)).toBeCloseTo(25, 10);
        state.step(0.05); // elapsed 0.225 → wraps to 0.025, up at 25% → 25
        expect(Sequence.state.get(seq)).toBe(TweenState.Playing); // loops, never completes
        expect(Probe.value.get(target)).toBeCloseTo(25, 10);
    });

    test("a non-looping sequence completes and holds the final pose", () => {
        const seq = sequence(state); // loop defaults off
        tween(state, target, "probe.value", {
            from: 0,
            to: 100,
            at: 0,
            duration: 0.1,
            sequence: seq,
        });
        stepFor(state, 0.3);
        expect(Sequence.state.get(seq)).toBe(TweenState.Complete);
        expect(Probe.value.get(target)).toBeCloseTo(100, 10); // forwards holds the end past the period
    });
});

describe("tween field + easing trait round-trip", () => {
    const traits = TweenPlugin.traits!.Tween;
    const parseField = traits.parse!.field as (raw: string) => number;
    const formatField = traits.format!.field as (id: number) => string;
    const parseEasing = traits.parse!.easing as (raw: string) => number;
    const formatEasing = traits.format!.easing as (id: number) => string;

    beforeEach(async () => {
        clear();
        // initialize resets the interning table + reserves id 0 = ""
        await build({ plugins: [TweenPlugin, TestPlugin], defaults: false });
    });

    test("a field path interns to an id and formats back to the path", () => {
        const id = parseField("probe.value");
        expect(formatField(id)).toBe("probe.value");
        expect(parseField("probe.value")).toBe(id); // identical paths dedupe
        expect(parseField("probe.vec.y")).not.toBe(id); // distinct paths don't
    });

    test("id 0 is the empty path — the Tween.field default", () => {
        expect(formatField(0)).toBe("");
    });

    test("an easing name parses to its index and formats back", () => {
        const id = parseEasing("ease-out-quad");
        expect(id).toBe(getEasingIndex("ease-out-quad"));
        expect(formatEasing(id)).toBe("ease-out-quad");
        expect(formatEasing(0)).toBe("linear");
    });
});

describe("tween scene wiring", () => {
    let state: State;

    beforeEach(async () => {
        clear();
        ({ state } = await build({ plugins: [TweenPlugin, TestPlugin], defaults: false }));
    });

    test("a scene tween animates its @name target and autoplays", () => {
        const nodes = parse(
            `<scene>
                <a id="p" probe />
                <a tween="target: @p; field: probe.value; to: 100; duration: 1" />
            </scene>`,
        );
        const map = load(nodes, state);
        const p = map.get(nodes[0])!;
        const t = map.get(nodes[1])!;

        expect(Tween.target.get(t)).toBe(p); // @name resolved
        expect(Tween.state.get(t)).toBe(TweenState.Playing); // autoplays, no explicit state

        stepFor(state, 0.5);
        expect(Probe.value.get(p)).toBeCloseTo(50, 10);
        stepFor(state, 0.5);
        expect(Probe.value.get(p)).toBeCloseTo(100, 10);
    });

    test("a scene tween writes a vector lane via the dotted path", () => {
        const nodes = parse(
            `<scene>
                <a id="p" probe />
                <a tween="target: @p; field: probe.vec.y; from: 0; to: 200; duration: 1" />
            </scene>`,
        );
        const map = load(nodes, state);
        const p = map.get(nodes[0])!;
        stepFor(state, 0.5);
        expect(Probe.vec.y.get(p)).toBeCloseTo(100, 10);
        expect(Probe.vec.x.get(p)).toBe(0); // other lanes untouched — exact default
    });

    test("@name places tweens on a sequence and its period spans them", () => {
        const nodes = parse(
            `<scene>
                <a id="p" probe />
                <a id="seq" sequence="loop: 1" />
                <a tween="target: @p; field: probe.value; from: 0; to: 100; at: 0; duration: 0.1; fill: none; sequence: @seq" />
                <a tween="target: @p; field: probe.value; from: 100; to: 0; at: 0.1; duration: 0.1; fill: none; sequence: @seq" />
            </scene>`,
        );
        const map = load(nodes, state);
        const p = map.get(nodes[0])!;
        const seq = map.get(nodes[1])!;
        const rise = map.get(nodes[2])!;

        expect(Tween.sequence.get(rise)).toBe(seq); // @seq resolved

        state.step(0.025); // setup grows the period to 0.2, then the clock advances
        expect(Sequence.duration.get(seq)).toBeCloseTo(0.2, 5);
        expect(Probe.value.get(p)).toBeCloseTo(25, 10); // rise at 25%
        state.step(0.05); // elapsed 0.075, rise at 75%
        expect(Probe.value.get(p)).toBeCloseTo(75, 10);
        state.step(0.05); // elapsed 0.125, rise released, fall at 25% → 75 (falling)
        expect(Probe.value.get(p)).toBeCloseTo(75, 10);
        state.step(0.05); // elapsed 0.175, fall at 75% → 25
        expect(Probe.value.get(p)).toBeCloseTo(25, 10);
        state.step(0.05); // elapsed 0.225 → wraps to 0.025, rise at 25%
        expect(Sequence.state.get(seq)).toBe(TweenState.Playing); // loops, never completes
        expect(Probe.value.get(p)).toBeCloseTo(25, 10);
    });
});
