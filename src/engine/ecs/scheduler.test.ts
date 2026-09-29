import { expect, test } from "bun:test";
import { Scheduler } from "./scheduler";
import type { State } from "./state";

test("step refuses a non-finite or negative delta before advancing its clock", () => {
    const scheduler = new Scheduler();
    const before = { ...scheduler.time };
    for (const delta of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
        expect(() => scheduler.step({} as State, delta)).toThrow(
            "step deltaTime must be a finite, non-negative number",
        );
        expect(scheduler.time).toEqual(before);
    }
}, 250);
