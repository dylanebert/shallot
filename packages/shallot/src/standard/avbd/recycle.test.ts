import { describe, expect, test } from "bun:test";
import { diffStamps } from "./recycle";

// diffStamps is the CPU pre-pack stamp diff the AVBD PackSystem runs to catch a same-update destroy+create
// realias. The seed path itself is GPU-only (the pack shader), so the gym `backend`
// recycle station is the behavioral gate; this pins the bookkeeping that decides WHICH eids the plugin
// reseeds — the part with the branches, testable with no device.

describe("diffStamps — the AVBD pre-pack stamp diff", () => {
    test("a first-seen eid is a fresh spawn — not recycled (the GPU pack seeds it via its own gate)", () => {
        const seen = new Map<number, number>();
        const stamps = new Map([[5, 1]]);
        // eid 5 seen for the first time: it must NOT reseed (a fresh spawn's GPU seed flag is already 0),
        // and its stamp must be recorded so a later realias is visible.
        expect(diffStamps([5], (e) => stamps.get(e) ?? 0, seen)).toEqual([]);
        expect(seen.get(5)).toBe(1);
    });

    test("an unchanged stamp is the settled body — not recycled", () => {
        const seen = new Map([[5, 1]]);
        const stamps = new Map([[5, 1]]);
        expect(diffStamps([5], (e) => stamps.get(e) ?? 0, seen)).toEqual([]);
    });

    test("a changed stamp is a realias — reseed the recycled eid", () => {
        const seen = new Map([[5, 1]]);
        // eid 5 destroyed + recreated in one update bumps its create-stamp 1 → 2 while staying a Body member
        const stamps = new Map([[5, 2]]);
        expect(diffStamps([5], (e) => stamps.get(e) ?? 0, seen)).toEqual([5]);
        expect(seen.get(5)).toBe(2); // the new stamp is recorded, so it fires exactly once
        expect(diffStamps([5], (e) => stamps.get(e) ?? 0, seen)).toEqual([]);
    });

    test("only the recycled eids of a mixed set reseed", () => {
        // eid 2 fresh (first-seen), eid 3 unchanged, eid 4 realiased — only 4 reseeds
        const seen = new Map([
            [3, 7],
            [4, 7],
        ]);
        const stamps = new Map([
            [2, 1],
            [3, 7],
            [4, 8],
        ]);
        expect(diffStamps([2, 3, 4], (e) => stamps.get(e) ?? 0, seen)).toEqual([4]);
        expect(seen.get(2)).toBe(1);
    });
});
