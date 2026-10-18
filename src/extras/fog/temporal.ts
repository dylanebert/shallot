// Seven-frame averaging limits trails while accumulating sub-cell samples.
export const HISTORY_WEIGHT = 6 / 7;

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
    { length: 16 },
    (_, i) =>
        [radicalInverse(i + 1, 2), radicalInverse(i + 1, 3), radicalInverse(i + 1, 5)] as const,
);

export function haltonOffset(frame: number): readonly [number, number, number] {
    return OFFSETS[frame % OFFSETS.length];
}
