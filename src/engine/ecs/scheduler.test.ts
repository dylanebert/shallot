import { expect, test } from "bun:test";
import { Scheduler } from "./scheduler";
import type { World } from "./state";

test("the scheduler consumes each updated duration from its lifetime frame input", () => {
    const scheduler = new Scheduler();
    const input = { deltaTime: 0.01 };
    scheduler.step({} as World, input);
    expect(scheduler.time.rawDeltaTime).toBe(0.01);
    input.deltaTime = 0.02;
    scheduler.step({} as World, input);
    expect(scheduler.time.rawDeltaTime).toBe(0.02);
    expect(scheduler.time.elapsed).toBeCloseTo(0.03);
});

test("step refuses a non-finite or negative delta before advancing its clock", () => {
    const scheduler = new Scheduler();
    const before = { ...scheduler.time };
    for (const delta of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
        expect(() => scheduler.step({} as World, { deltaTime: delta })).toThrow(
            "step deltaTime must be a finite, non-negative number",
        );
        expect(scheduler.time).toEqual(before);
    }
});
