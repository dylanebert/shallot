import { expect, test } from "bun:test";
import { classifyRendered } from "./rendered";

const SIZE = 64;
const grid = (pixel: (x: number, y: number) => number): number[] => {
    const out: number[] = [];
    for (let y = 0; y < SIZE; y++)
        for (let x = 0; x < SIZE; x++) {
            const value = pixel(x, y);
            out.push(value, value, value);
        }
    return out;
};

test("the rendered law rejects a flat clear frame", () => {
    expect(classifyRendered(grid(() => 24)).rendered).toBe(false);
});

test("the rendered law rejects a faint clear-only gradient", () => {
    // This models the hardware blank-frame symptom: one extreme edge pixel differs, then the clear
    // gradient rapidly levels out. A top-left/whole-frame changed-pixel count admits it even though the
    // centre and corner regions both read the same clear field.
    const faintClear = grid((x, y) => Math.min(24, (x + y) * 6));
    expect(classifyRendered(faintClear).rendered).toBe(false);
});

test("the rendered law accepts centrally framed structure", () => {
    const scene = grid((x, y) => (x >= 16 && x < 48 && y >= 16 && y < 48 ? 90 : 10));
    expect(classifyRendered(scene).rendered).toBe(true);
});
