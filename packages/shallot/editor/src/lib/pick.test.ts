import { describe, expect, test } from "bun:test";
import type { Node } from "@dylanebert/shallot/editor";
import { CLICK_SLOP, isClick, nextSelection, nodeForEid } from "./pick";

describe("isClick — a tap selects, an orbit drag doesn't", () => {
    test("zero movement is a click", () => {
        expect(isClick(100, 100, 100, 100)).toBe(true);
    });

    test("a jitter within the slop is still a click", () => {
        expect(isClick(100, 100, 102, 101)).toBe(true);
    });

    test("an orbit drag past the slop is not a click", () => {
        expect(isClick(100, 100, 140, 105)).toBe(false);
    });

    test("the slop boundary is inclusive (radial, not per-axis)", () => {
        expect(isClick(0, 0, CLICK_SLOP, 0)).toBe(true);
        expect(isClick(0, 0, CLICK_SLOP + 1, 0)).toBe(false);
        // a diagonal past the radius is a drag even though neither axis exceeds the slop
        expect(isClick(0, 0, CLICK_SLOP, CLICK_SLOP)).toBe(false);
    });
});

describe("nodeForEid — resolve a picked eid to its scene node", () => {
    const a: Node = { attrs: [], children: [] };
    const b: Node = { attrs: [], children: [] };
    const nodeMap = new Map<Node, number>([
        [a, 3],
        [b, 7],
    ]);

    test("returns the node whose eid matches", () => {
        expect(nodeForEid(3, nodeMap)).toBe(a);
        expect(nodeForEid(7, nodeMap)).toBe(b);
    });

    test("empty space (eid < 0) resolves to null — the caller deselects", () => {
        expect(nodeForEid(-1, nodeMap)).toBeNull();
    });

    test("an eid no node owns resolves to null", () => {
        expect(nodeForEid(99, nodeMap)).toBeNull();
    });
});

describe("nextSelection — toggle-everywhere multi-select", () => {
    const a: Node = { attrs: [], children: [] };
    const b: Node = { attrs: [], children: [] };
    const c: Node = { attrs: [], children: [] };

    test("a plain pick selects only the picked node", () => {
        expect(nextSelection([], a, false)).toEqual([a]);
        expect(nextSelection([b, c], a, false)).toEqual([a]);
    });

    test("a plain pick on empty space clears the selection", () => {
        expect(nextSelection([a, b], null, false)).toEqual([]);
    });

    test("a modifier pick appends a new node, preserving order", () => {
        expect(nextSelection([a, b], c, true)).toEqual([a, b, c]);
    });

    test("a modifier pick toggles an already-selected node out", () => {
        expect(nextSelection([a, b, c], b, true)).toEqual([a, c]);
    });

    test("a modifier pick on empty space keeps the selection", () => {
        expect(nextSelection([a, b], null, true)).toEqual([a, b]);
    });
});
