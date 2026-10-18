import tgpu from "typegpu";
import * as d from "typegpu/data";

// Seven-frame averaging limits trails while accumulating sub-cell samples.
export const HISTORY_WEIGHT = 6 / 7;
export const TEMPORAL_FRAMES = 16;

function radicalInverse(index: number, base: number): number {
    let value = 0;
    let fraction = 1 / base;
    while (index > 0) {
        value += (index % base) * fraction;
        index = Math.floor(index / base);
        fraction /= base;
    }
    return value;
}

const OFFSETS = Array.from(
    { length: TEMPORAL_FRAMES },
    (_, i) =>
        [radicalInverse(i + 1, 2), radicalInverse(i + 1, 3), radicalInverse(i + 1, 5)] as const,
);

export const HALTON = tgpu.const(
    d.arrayOf(d.vec3f, TEMPORAL_FRAMES),
    OFFSETS.map((offset) => d.vec3f(...offset)),
);
