import { expect, test } from "bun:test";
import { createApp } from "../app";
import { Time } from "./scheduler";

test("preFixed discard reduces paced ticks to the remaining overstep, including zero, without stopping frames", async () => {
    let discard = Time.FIXED_DT * 2;
    let frames = 0;
    let draws = 0;
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "DiscardOverstep",
                recovery: "stateless",
                systems: [
                    {
                        group: "preFixed",
                        update(w) {
                            w.discardOverstep(discard);
                        },
                    },
                    {
                        group: "simulation",
                        update() {
                            frames++;
                        },
                    },
                    {
                        group: "draw",
                        update() {
                            draws++;
                        },
                    },
                ],
            },
        ],
    });
    const w = app.world;
    try {
        w.step(Time.FIXED_DT * 3.75);
        expect(w.time.fixedTick).toBe(1);
        expect(w.time.fixedSteps).toBe(1);
        expect(w.time.fixedAlpha).toBeCloseTo(0.75);
        w.step(Time.FIXED_DT * 2);
        expect(w.time.fixedTick).toBe(1);
        expect(w.time.fixedSteps).toBe(0);
        expect(w.time.fixedAlpha).toBeCloseTo(0.75);
        discard = 100;
        w.step(Time.FIXED_DT);
        expect(w.time.fixedTick).toBe(1);
        expect(w.time.fixedAlpha).toBe(0);
        discard = Time.FIXED_DT * 0.5;
        w.step(Time.FIXED_DT);
        expect(w.time.fixedTick).toBe(1);
        expect(w.time.fixedAlpha).toBeCloseTo(0.5);
        w.tick();
        expect(w.time.fixedTick).toBe(2);
        expect(w.time.fixedAlpha).toBeCloseTo(0.5);
        expect(frames).toBe(4);
        expect(draws).toBe(4);
    } finally {
        app.dispose();
    }
});

test("discard refuses outside preFixed and rejects negative or non-finite durations", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "DiscardRefusals",
                recovery: "stateless",
                systems: (
                    ["setup", "preFixed", "fixed", "postFixed", "simulation", "draw"] as const
                ).map((group) => ({
                    group,
                    update(w) {
                        if (group === "preFixed") {
                            for (const invalid of [-1, NaN, Infinity])
                                expect(() => w.discardOverstep(invalid)).toThrow(
                                    "finite, non-negative",
                                );
                            w.discardOverstep(0);
                        } else expect(() => w.discardOverstep(0)).toThrow("preFixed");
                    },
                })),
            },
        ],
    });
    try {
        expect(() => app.world.discardOverstep(0)).toThrow("preFixed");
        app.world.step();
        app.world.tick();
        expect(app.world.time.fixedTick).toBe(2);
    } finally {
        app.dispose();
    }
});
