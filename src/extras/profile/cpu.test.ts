import { expect, test } from "bun:test";
import { cpuTotal } from "./cpu";

test("CPU totals count outermost timings, not slash-delimited parts", () => {
    expect(
        cpuTotal(
            new Map([
                ["Physics/step/solve", 3],
                ["Physics/step", 5],
                ["Physics/step/solve/constraints", 2],
                ["Physics/stepping", 7],
                ["draw", 11],
            ]),
        ),
    ).toBe(23);
    expect(cpuTotal(new Map([["orphan/part", 4]]))).toBe(4);
    expect(cpuTotal(new Map())).toBe(0);
});
