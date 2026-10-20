import { expect, test } from "bun:test";
import { jfaPassCount, jfaStep, MAX_JFA_PASSES } from "./passes";

for (const [width, levels] of [
    [0, 1],
    [1, 1],
    [1.5, 2],
    [2, 2],
    [2.5, 2],
    [3, 2],
    [3.5, 3],
    [4, 3],
    [5.5, 3],
    [7, 3],
    [7.5, 4],
    [8, 4],
    [9.5, 4],
    [14, 4],
    [15, 4],
    [15.5, 5],
    [16, 5],
    [31, 5],
    [31.5, 6],
    [32, 6],
    [63, 6],
    [63.5, 7],
    [64, 7],
    [65, 7],
]) {
    test(`width ${width} takes ${levels} Golus ladder levels`, () => {
        expect(jfaPassCount(width!)).toBe(levels!);
        const steps = Array.from({ length: levels! }, (_, i) =>
            jfaStep(MAX_JFA_PASSES - levels! + i),
        );
        expect(steps.at(-1)).toBe(1);
        const reach = steps.reduce((sum, step) => sum + step, 0);
        expect(reach).toBeGreaterThanOrEqual(Math.max(1, Math.min(64, width!)));
        if (levels! > 1) expect((reach - 1) / 2).toBeLessThan(Math.max(1, Math.min(64, width!)));
    });
}
