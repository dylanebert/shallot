import { describe, expect, test } from "bun:test";
import { intVec, NULL_INDEX, qsort } from "./array";

// Ports test_container.c's numeric-array behaviors. The C struct-array and type-genericity cases
// test the macro's C-preprocessor mechanics; GrowVec is genuinely generic via its factory, so
// those don't carry over.
describe("GrowVec", () => {
    test("push and get", () => {
        const a = intVec();
        a.push(42);
        expect(a.get(0)).toBe(42);
        expect(a.count).toBe(1);
    });

    test("iteration sums", () => {
        const a = intVec();
        a.push(1);
        a.push(2);
        a.push(3);
        let sum = 0;
        for (let i = 0; i < a.count; ++i) {
            sum += a.get(i);
        }
        expect(sum).toBe(6);
    });

    test("emplace returns writable indices", () => {
        const a = intVec();
        const n = 100;
        for (let i = 0; i < n; ++i) {
            a.set(a.emplace(), i);
        }
        let sum = 0;
        for (let i = 0; i < a.count; ++i) {
            sum += a.get(i);
        }
        expect(sum).toBe((n * (n - 1)) / 2);
    });

    test("reserve sizes capacity; removeSwap keeps the multiset", () => {
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
    });

    test("removeSwap returns the moved index, or NULL_INDEX for the tail", () => {
        const a = intVec();
        a.push(10);
        a.push(11);
        a.push(12);
        expect(a.removeSwap(0)).toBe(2); // last (index 2) moved into 0
        expect(a.get(0)).toBe(12);
        expect(a.removeSwap(1)).toBe(NULL_INDEX); // removing the tail moves nothing
    });

    test("resize sets capacity and count; pop drains", () => {
        const a = intVec();
        const n = 10;
        a.resize(n);
        expect(a.capacity).toBe(n);
        expect(a.count).toBe(n);
        for (let i = 0; i < n; ++i) {
            a.set(i, i);
        }
        let sum = 0;
        while (a.count > 0) {
            sum += a.pop();
        }
        expect(sum).toBe((n * (n - 1)) / 2);
    });
});

// qsort is a faithful port of Box3D's QSORT macro (quicksort with an insertion-sort cutoff at 16).
// The mesh-sphere fixture only ever sorts a handful of triangles (the insertion path), so these
// cover the recursive partition + subfile-stack path directly, plus the tie-heavy duplicate case.
describe("qsort", () => {
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

    test("sorts a large array through the partition + stack path", () => {
        // 200 elements (>> the cutoff of 16) with many duplicates (values 0..49) exercises the
        // recursive partitioning and the i>=j partition boundary.
        const input = Array.from({ length: 200 }, (_, i) => (i * 137 + 41) % 50);
        expect(sorted(input)).toEqual(input.slice().sort((x, y) => x - y));
    });

    test("sorts through the insertion path with duplicate keys", () => {
        expect(sorted([5, 1, 5, 3, 1, 2, 5, 0, 3, 2])).toEqual([0, 1, 1, 2, 2, 3, 3, 5, 5, 5]);
    });

    test("handles trivial sizes", () => {
        expect(sorted([])).toEqual([]);
        expect(sorted([7])).toEqual([7]);
        expect(sorted([2, 1])).toEqual([1, 2]);
    });
});
