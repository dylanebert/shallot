import { expect, test } from "bun:test";
import { init } from "./kernel";
import { createProxy, createTree, destroyTree, query } from "./treecolumns";

await init(undefined, { threads: 0 });

test("a larger nested independent tree query preserves the outer hits and visit stats", () => {
    const box = {
        lowerBound: { x: -1, y: -1, z: -1 },
        upperBound: { x: 1000, y: 1, z: 1 },
    };
    const build = (count: number, userOffset = 0) => {
        const tree = createTree(count);
        for (let i = 0; i < count; i++)
            createProxy(
                tree,
                {
                    lowerBound: { x: i * 2, y: 0, z: 0 },
                    upperBound: { x: i * 2 + 1, y: 1, z: 1 },
                },
                0xffffffff,
                0xffffffff,
                userOffset + i,
            );
        return tree;
    };
    const outer = build(3);
    const inner = build(128, 1000);
    try {
        const expected: number[] = [];
        const stats = query(outer, box, 0xffffffff, 0xffffffff, false, (_, user) => {
            expected.push(user);
            return true;
        });
        expect(expected.toSorted()).toEqual([0, 1, 2]);
        query(inner, box, 0xffffffff, 0xffffffff, false, () => true);
        const hits: number[] = [];
        let nested = false;
        const actual = query(outer, box, 0xffffffff, 0xffffffff, false, (_, user) => {
            hits.push(user);
            if (!nested) {
                nested = true;
                const innerHits: number[] = [];
                const innerStats = query(
                    inner,
                    box,
                    0xffffffff,
                    0xffffffff,
                    false,
                    (_, innerUser) => {
                        innerHits.push(innerUser);
                        return true;
                    },
                );
                expect(innerHits.toSorted((a, b) => a - b)).toEqual(
                    Array.from({ length: 128 }, (_, i) => 1000 + i),
                );
                expect(innerStats.leafVisits).toBe(128);
            }
            return true;
        });
        expect(hits).toEqual(expected);
        expect(actual).toEqual(stats);
    } finally {
        destroyTree(outer);
        destroyTree(inner);
    }
});
