import { describe, expect, test } from "bun:test";
import { degrees, radians } from "./input";

describe("unit converters", () => {
    test("radians is the identity unit", () => {
        for (const x of [0, 1, -2.5, Math.PI]) {
            expect(radians.to(x)).toBe(x);
            expect(radians.from(x)).toBe(x);
        }
    });

    test("degrees maps half-turn ↔ 180 and round-trips", () => {
        expect(degrees.to(Math.PI)).toBeCloseTo(180, 10);
        expect(degrees.from(180)).toBeCloseTo(Math.PI, 10);
        // to/from are inverse — the editor relies on this when it stores an edited shown value
        for (const r of [0, 0.5, 1.2345, -Math.PI / 3]) {
            expect(degrees.from(degrees.to(r))).toBeCloseTo(r, 12);
        }
    });
});
