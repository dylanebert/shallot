import { describe, expect, test } from "bun:test";
import { assertMotion } from "./motion";

describe("assertMotion", () => {
    test("passes a moving pair", () => {
        expect(assertMotion([0, 0, 0], [0, 4, 8], 3)).toBe(4);
    });

    test("rejects a parked pair", () => {
        expect(() => assertMotion([2, 2], [2, 2], 0.1)).toThrow("samples are parked");
    });
});
