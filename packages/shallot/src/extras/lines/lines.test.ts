import { describe, expect, test } from "bun:test";
import { arrow, box, Lines, segment } from "./segments";

// The immediate API's box/arrow expansion is the arithmetic that decides how many segments draw. It's
// pure — the GPU draw itself is validated in `bun bench`; the color codec (packColor) in color.test.ts.
describe("kitchen lines immediate API", () => {
    test("segment appends one, box twelve edges, arrow shaft + four fins", () => {
        const a = Lines.count;
        segment([0, 0, 0], [1, 0, 0], 0xffffff);
        expect(Lines.count - a).toBe(1);
        const b = Lines.count;
        box([0, 0, 0], [1, 1, 1], 0xffffff);
        expect(Lines.count - b).toBe(12);
        const c = Lines.count;
        arrow([0, 0, 0], [0, 1, 0], 0xffffff);
        expect(Lines.count - c).toBe(5);
    });

    test("a zero-length arrow emits the shaft but no head", () => {
        const a = Lines.count;
        arrow([1, 1, 1], [1, 1, 1], 0xffffff);
        expect(Lines.count - a).toBe(1);
    });
});
