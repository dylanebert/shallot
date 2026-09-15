import { expect } from "bun:test";
import { check } from "../../harness/check";
import { median } from "./coalesce";

// The frame loop keeps the recent rAF intervals in a fixed ring and reads their median as the live present
// cadence, which is what `coalesce` measures a double-fire against. The ring is written cyclically and its
// count saturates, so the median must read the same value a sort of those samples would, in any order.

const WINDOW = 20;

// the sorted-copy median, the definition the ring insertion has to reproduce
function sortedMedian(samples: number[]): number {
    const sorted = [...samples].sort((a, b) => a - b);
    return sorted[sorted.length >> 1];
}

// fill a ring of `WINDOW` slots with `samples` in order, returning the ring, its live count, and the
// samples still resident (the last `WINDOW` of them)
function ring(samples: number[]): { intervals: Float64Array; count: number; live: number[] } {
    const intervals = new Float64Array(WINDOW);
    let next = 0;
    let count = 0;
    for (const sample of samples) {
        intervals[next] = sample;
        next = (next + 1) % WINDOW;
        if (count < WINDOW) count++;
    }
    return { intervals, count, live: samples.slice(-WINDOW) };
}

check(
    "the interval median matches a sort of the live samples",
    {
        claim: "the frame loop's median reads a different value from a sort of the same intervals, so the present cadence it coalesces double-fires against would be wrong",
        subject: ["src/engine/app/coalesce.ts"],
    },
    () => {
        const scratch = new Float64Array(WINDOW);
        // ascending, descending, unsorted, and a repeated-value window
        const windows: number[][] = [
            [4.1],
            [16.7, 4.2, 8.3],
            Array.from({ length: WINDOW }, (_, i) => i + 1),
            Array.from({ length: WINDOW }, (_, i) => WINDOW - i),
            [7, 3, 19, 3, 11, 2, 7, 15, 4, 4, 9, 1, 12, 6, 8, 5, 13, 10, 14, 0],
            Array.from({ length: WINDOW }, () => 4.16),
        ];
        for (const samples of windows) {
            const { intervals, count } = ring(samples);
            expect(median(intervals, count, scratch)).toBe(sortedMedian(samples));
        }
    },
);

check(
    "the interval median reads only the live samples of a partly filled ring",
    {
        claim: "the median reads past the live sample count into a partly filled ring's zeroed slots, so the first frames after a rebuild would report a near-zero cadence and coalesce real frames away",
        subject: ["src/engine/app/coalesce.ts"],
    },
    () => {
        const scratch = new Float64Array(WINDOW);
        const samples = [16.7, 16.6, 16.8, 16.7, 16.9, 16.6, 16.7];
        const { intervals, count } = ring(samples);
        expect(count).toBe(samples.length);
        expect(median(intervals, count, scratch)).toBe(sortedMedian(samples));
        // the untouched tail of the ring is zero, and a median that read it would fall to it
        expect(intervals[WINDOW - 1]).toBe(0);
        expect(median(intervals, count, scratch)).toBeGreaterThan(16);
        expect(median(intervals, 0, scratch)).toBe(0);
    },
);

check(
    "the interval median follows the ring past a wrap",
    {
        claim: "the median keeps reading intervals the ring has already overwritten, so a cadence change would take a whole window to show and the coalesce threshold would lag it",
        subject: ["src/engine/app/coalesce.ts"],
    },
    () => {
        const scratch = new Float64Array(WINDOW);
        // a 60Hz window fully overwritten by a 240Hz one, and a half-overwritten mix
        const wrapped = [
            ...Array.from({ length: WINDOW }, () => 16.7),
            ...Array.from({ length: WINDOW }, () => 4.16),
        ];
        const full = ring(wrapped);
        expect(full.count).toBe(WINDOW);
        expect(median(full.intervals, full.count, scratch)).toBe(4.16);

        const half = ring([...Array.from({ length: WINDOW }, () => 16.7), ...Array(10).fill(4.16)]);
        expect(median(half.intervals, half.count, scratch)).toBe(sortedMedian(half.live));
    },
);
