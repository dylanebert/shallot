import { describe, expect, test } from "bun:test";
import { assertMotion, frameDifference } from "./motion";

describe("assertMotion", () => {
    test("passes a moving pair", () => {
        expect(assertMotion([0, 0, 0], [0, 4, 8], 3)).toBe(4);
    });

    test("rejects a parked pair", () => {
        expect(() => assertMotion([2, 2], [2, 2], 0.1)).toThrow("samples are parked");
    });
});

describe("frameDifference", () => {
    test("reports the mean absolute difference", () => {
        expect(frameDifference([0, 0, 0], [0, 4, 8])).toBe(4);
    });

    // the whole reason this seam exists: a retrying caller (`expect.poll`) must be able to sample a
    // parked frame and sample again. A throw there ends the poll on its first reading.
    test("returns a parked reading instead of throwing", () => {
        expect(frameDifference([2, 2], [2, 2])).toBe(0);
        expect(frameDifference([2, 2], [2, 2.08])).toBeCloseTo(0.04, 10);
    });

    test("rejects mismatched or empty samples", () => {
        expect(() => frameDifference([1, 2], [1])).toThrow("same non-zero length");
        expect(() => frameDifference([], [])).toThrow("same non-zero length");
    });
});
