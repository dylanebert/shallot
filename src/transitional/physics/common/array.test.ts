import { expect } from "bun:test";
import { check } from "../../../harness/check";
import { intVec, NULL_INDEX, qsort } from "./array";

// Ports test_container.c's numeric-array behaviors. The C struct-array and type-genericity cases
// test the macro's C-preprocessor mechanics; GrowVec is genuinely generic via its factory, so
// those don't carry over.

check(
    "GrowVec push, get and count agree",
    {
        claim: "GrowVec.push writes past or before its own tail, so the value read back at index zero is not the value pushed and count does not advance",
    },
    () => {
        const a = intVec();
        a.push(42);
        expect(a.get(0)).toBe(42);
        expect(a.count).toBe(1);
        a.push(7);
        a.push(9);
        let sum = 0;
        for (let i = 0; i < a.count; ++i) sum += a.get(i);
        expect(sum).toBe(58);
    },
);

check(
    "GrowVec emplace returns writable indices across regrowth",
    {
        claim: "GrowVec.emplace hands back an index outside the backing store or loses earlier elements when it doubles capacity",
    },
    () => {
        const a = intVec();
        const n = 100;
        for (let i = 0; i < n; ++i) {
            a.set(a.emplace(), i);
        }
        expect(a.count).toBe(n);
        let sum = 0;
        for (let i = 0; i < a.count; ++i) {
            sum += a.get(i);
        }
        expect(sum).toBe((n * (n - 1)) / 2);
    },
);

check(
    "GrowVec reserve sizes capacity and removeSwap preserves the multiset",
    {
        claim: "GrowVec.reserve leaves count nonzero or removeSwap drops or duplicates an element while draining from the head",
    },
    () => {
        const a = intVec();
        const n = 100;
        a.reserve(n);
        expect(a.capacity).toBe(n);
        expect(a.count).toBe(0);

        for (let i = 0; i < n; ++i) {
            a.push(i);
        }
        let sum = 0;
        for (let i = 0; i < n; ++i) {
            sum += a.get(0);
            a.removeSwap(0);
        }
        expect(sum).toBe((n * (n - 1)) / 2);
        expect(a.count).toBe(0);
    },
);

check(
    "GrowVec removeSwap reports the moved index",
    {
        claim: "GrowVec.removeSwap returns the wrong moved index or fails to return NULL_INDEX for the tail, so callers fix up a back-index that never moved",
    },
    () => {
        const a = intVec();
        a.push(10);
        a.push(11);
        a.push(12);
        expect(a.removeSwap(0)).toBe(2); // last (index 2) moved into 0
        expect(a.get(0)).toBe(12);
        expect(a.removeSwap(1)).toBe(NULL_INDEX); // removing the tail moves nothing
    },
);

check(
    "GrowVec resize sets capacity and count, and pop drains",
    {
        claim: "GrowVec.resize sets capacity without count, or pop returns elements in an order other than last in first out",
    },
    () => {
        const a = intVec();
        const n = 10;
        a.resize(n);
        expect(a.capacity).toBe(n);
        expect(a.count).toBe(n);
        for (let i = 0; i < n; ++i) {
            a.set(i, i);
        }
        const drained: number[] = [];
        while (a.count > 0) {
            drained.push(a.pop());
        }
        expect(drained).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    },
);

// qsort is a faithful port of Box3D's QSORT macro (quicksort with an insertion-sort cutoff at 16).
// The mesh-sphere fixture only ever sorts a handful of triangles (the insertion path), so these
// cover the recursive partition + subfile-stack path directly, plus the tie-heavy duplicate case.

const sorted = (arr: number[]): number[] => {
    const a = arr.slice();
    qsort(
        a.length,
        (i, j) => a[i] < a[j],
        (i, j) => {
            const t = a[i];
            a[i] = a[j];
            a[j] = t;
        },
    );
    return a;
};

check(
    "qsort orders a large array through the partition and stack path",
    {
        claim: "the qsort port mishandles its subfile stack or the i>=j partition boundary above the insertion cutoff, leaving 200 duplicate-heavy keys out of order",
    },
    () => {
        // 200 elements (>> the cutoff of 16) with many duplicates (values 0..49) exercises the
        // recursive partitioning and the i>=j partition boundary.
        const input = Array.from({ length: 200 }, (_, i) => (i * 137 + 41) % 50);
        expect(sorted(input)).toEqual(input.slice().sort((x, y) => x - y));
    },
);

check(
    "qsort orders duplicate keys through the insertion path",
    {
        claim: "the qsort insertion-sort cutoff branch mis-sorts a short run containing repeated keys",
    },
    () => {
        expect(sorted([5, 1, 5, 3, 1, 2, 5, 0, 3, 2])).toEqual([0, 1, 1, 2, 2, 3, 3, 5, 5, 5]);
    },
);

check(
    "qsort handles empty, single and pair inputs",
    {
        claim: "the qsort port indexes out of bounds or swaps needlessly on inputs of length zero, one or two",
    },
    () => {
        expect(sorted([])).toEqual([]);
        expect(sorted([7])).toEqual([7]);
        expect(sorted([2, 1])).toEqual([1, 2]);
    },
);
