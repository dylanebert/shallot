import { expect, test } from "bun:test";
import { frameDelta } from "./frame-delta";

test("the first frame has no elapsed delta", () => {
    expect(frameDelta(16.7, -1)).toBe(0);
});

test("frameDelta converts elapsed milliseconds to seconds", () => {
    expect(frameDelta(33.4, 16.7)).toBeCloseTo(0.0167);
});

test("frameDelta floors a negative difference at zero", () => {
    expect(frameDelta(16, 17)).toBe(0);
});
