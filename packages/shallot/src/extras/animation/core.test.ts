import { describe, expect, test } from "bun:test";
import {
    Composite,
    EASING_FUNCTIONS,
    Fill,
    getEasing,
    getEasingIndex,
    getEasingName,
    keyframes,
    mixer,
    owns,
    Pose,
    sample,
    script,
} from "./core";

describe("animation timing atom", () => {
    test("preserves every fill mode and the instant boundary", () => {
        expect([
            owns(-0.1, 1, Fill.Forwards),
            owns(0, 1, Fill.Forwards),
            owns(1, 1, Fill.Forwards),
        ]).toEqual([false, true, true]);
        expect([owns(-0.1, 1, Fill.None), owns(0, 1, Fill.None), owns(1, 1, Fill.None)]).toEqual([
            false,
            true,
            false,
        ]);
        expect([
            owns(-0.1, 1, Fill.Backwards),
            owns(0.5, 1, Fill.Backwards),
            owns(1.1, 1, Fill.Backwards),
        ]).toEqual([true, true, false]);
        expect([
            owns(-0.1, 1, Fill.Both),
            owns(0.5, 1, Fill.Both),
            owns(1.1, 1, Fill.Both),
        ]).toEqual([true, true, true]);
        expect([
            owns(-0.1, 0, Fill.Forwards),
            owns(0, 0, Fill.Forwards),
            owns(0.1, 0, Fill.Forwards),
        ]).toEqual([false, true, true]);
    });

    test("preserves replace, add, saturation, instant, and easing sampling", () => {
        expect(sample(0.5, 1, 0, 0, 100, Composite.Replace, 0)).toBe(50);
        expect(sample(-1, 1, 0, 10, 90, Composite.Replace, 0)).toBe(10);
        expect(sample(2, 1, 0, 10, 90, Composite.Replace, 0)).toBe(90);
        expect(sample(0, 0, 0, 10, 90, Composite.Replace, 0)).toBe(90);
        expect(sample(0.5, 1, getEasingIndex("ease-in-quad"), 0, 100, Composite.Replace, 0)).toBe(
            25,
        );
        expect(sample(0.5, 1, 0, 0, 100, Composite.Add, 7)).toBe(57);
        expect(sample(1, 1, 0, 5, 0, Composite.Add, 42.5)).toBe(42.5);
    });

    test("preserves the complete easing table", () => {
        expect(EASING_FUNCTIONS.length).toBe(31);
        expect(getEasingIndex("nope")).toBe(0);
        expect(getEasingName(999)).toBe("linear");
        expect(getEasing(999)(0.5)).toBe(0.5);
        for (let i = 0; i < EASING_FUNCTIONS.length; i++) {
            const ease = EASING_FUNCTIONS[i];
            expect(ease(0)).toBeCloseTo(0, 10);
            expect(ease(1)).toBeCloseTo(1, 10);
            expect(Number.isFinite(ease(0.25))).toBe(true);
            expect(Number.isFinite(ease(0.75))).toBe(true);
            expect(getEasingIndex(getEasingName(i))).toBe(i);
        }
        expect(getEasing(getEasingIndex("ease-out-quad"))(0.5)).toBe(0.75);
    });
});

describe("playables", () => {
    test("keyframes applies per-segment easing, fill, and composite", () => {
        const pose = new Pose();
        const clip = keyframes(
            {
                x: [
                    { offset: 0, value: 0, easing: "ease-in-quad" },
                    { offset: 0.5, value: 100 },
                    { offset: 1, value: 200 },
                ],
                delta: [
                    { offset: 0, value: 2, composite: "add" },
                    { offset: 1, value: 4 },
                ],
            },
            { duration: 2, fill: "none" },
        );
        clip.evaluate(0.5, pose);
        expect(pose.get("x")).toBe(25);
        expect(pose.get("delta")).toBe(2.5);
        clip.evaluate(3, pose);
        expect([...pose]).toEqual([]);
    });

    test("script evaluates arbitrary time into the reusable pose", () => {
        const pose = new Pose();
        const clip = script((t, output) => output.set("x", t * 3), 4);
        clip.evaluate(2, pose);
        expect(pose.get("x")).toBe(6);
        clip.evaluate(1, pose);
        expect([...pose]).toEqual([["x", 3]]);
    });

    test("mixer remaps strip start, offset, and scale", () => {
        const pose = new Pose();
        const clip = mixer([
            [
                {
                    playable: script((t, output) => output.set("x", t), 10),
                    start: 4,
                    offset: 2,
                    scale: 2,
                    blend: "replace",
                },
            ],
        ]);
        clip.evaluate(8, pose);
        expect(pose.get("x")).toBe(4);
        expect(clip.duration).toBe(20);
    });

    test("later tracks replace earlier tracks and add layering sums", () => {
        const pose = new Pose();
        const constant = (value: number) => script((_t, output) => output.set("x", value), 1);
        const clip = mixer([
            [{ playable: constant(10), start: 0, blend: "replace" }],
            [{ playable: constant(4), start: 0, blend: "add" }],
            [{ playable: constant(7), start: 0, blend: "replace" }],
            [{ playable: constant(3), start: 0, blend: "add" }],
        ]);
        clip.evaluate(0.5, pose);
        expect(pose.get("x")).toBe(10);
    });

    test("nested evaluation is idempotent over varied time order", () => {
        const pose = new Pose();
        const wave = keyframes({ x: [{ value: 0 }, { value: 20 }] }, { duration: 2, fill: "both" });
        const clip = mixer([
            [{ playable: wave, start: 1, scale: 0.5, blend: "replace" }],
            [
                {
                    playable: script((t, output) => output.set("y", t * t), 4),
                    start: 0,
                    blend: "add",
                },
            ],
        ]);
        const times = [1.25, 3, 0, 2.5, 0.75, 1.25, 4, 2.5];
        const readings = new Map<number, string>();
        for (const t of times) {
            clip.evaluate(t, pose);
            const reading = JSON.stringify([...pose].sort(([a], [b]) => a.localeCompare(b)));
            const prior = readings.get(t);
            if (prior) expect(reading).toBe(prior);
            else readings.set(t, reading);
        }
    });
});
