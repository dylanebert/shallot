import { describe, expect, test } from "bun:test";
import { getBit } from "./bitset";
import { DYNAMIC_COLOR_COUNT, OVERFLOW_INDEX } from "./core";
import { createGraph, greedyColor } from "./graph";
import { BodyType } from "./types";

// The greedy graph coloring (constraint_graph.c), tested independent of the FORCE_OVERFLOW flag the
// live path gates it behind. Assignment is the load-bearing, order-sensitive part; the full colored
// solve is bit-exact-verified against the default-config fixtures at the wide-solver wiring stage.
const { Static, Kinematic, Dynamic } = BodyType;

describe("greedyColor", () => {
    test("dynamic-dynamic pairs pack into the lowest non-conflicting color", () => {
        const graph = createGraph(8);
        // First pair takes color 0 and claims both bodies there.
        expect(greedyColor(graph, 0, 1, Dynamic, Dynamic)).toBe(0);
        expect(getBit(graph.colors[0].bodySet, 0)).toBe(true);
        expect(getBit(graph.colors[0].bodySet, 1)).toBe(true);
        // A disjoint pair fits the same color.
        expect(greedyColor(graph, 2, 3, Dynamic, Dynamic)).toBe(0);
        // Sharing body 0 conflicts in color 0, so it spills to color 1.
        expect(greedyColor(graph, 0, 4, Dynamic, Dynamic)).toBe(1);
        expect(getBit(graph.colors[1].bodySet, 0)).toBe(true);
        expect(getBit(graph.colors[1].bodySet, 4)).toBe(true);
    });

    test("a dynamic body saturating every dynamic color spills to overflow", () => {
        const graph = createGraph(8);
        for (let i = 0; i < DYNAMIC_COLOR_COUNT; ++i) {
            // Each pair shares body 0, so each lands in the next dynamic color.
            expect(greedyColor(graph, 0, 10 + i, Dynamic, Dynamic)).toBe(i);
        }
        // Body 0 now occupies all dynamic colors; the next shared pair overflows.
        expect(greedyColor(graph, 0, 99, Dynamic, Dynamic)).toBe(OVERFLOW_INDEX);
    });

    test("dynamic-static constraints build from the high end, tracking only the dynamic body", () => {
        const graph = createGraph(8);
        // Highest non-overflow color first (higher solver priority than dyn-dyn).
        expect(greedyColor(graph, 5, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 1);
        expect(getBit(graph.colors[OVERFLOW_INDEX - 1].bodySet, 5)).toBe(true);
        // The static side is never tracked, so a different dynamic body reuses the top color.
        expect(greedyColor(graph, 6, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 1);
        // The same dynamic body conflicts at the top and steps down one color.
        expect(greedyColor(graph, 5, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 2);
    });

    test("the static side is symmetric on body B", () => {
        const graph = createGraph(8);
        expect(greedyColor(graph, 0, 7, Static, Dynamic)).toBe(OVERFLOW_INDEX - 1);
        expect(getBit(graph.colors[OVERFLOW_INDEX - 1].bodySet, 7)).toBe(true);
        expect(getBit(graph.colors[OVERFLOW_INDEX - 1].bodySet, 0)).toBe(false);
    });

    test("kinematic bodies color like static ones (only the dynamic bit is tracked)", () => {
        const graph = createGraph(8);
        // Kinematic-dynamic is not the dyn-dyn branch: it builds from the high end like dyn-static.
        expect(greedyColor(graph, 3, 8, Kinematic, Dynamic)).toBe(OVERFLOW_INDEX - 1);
        expect(getBit(graph.colors[OVERFLOW_INDEX - 1].bodySet, 8)).toBe(true);
        // Dynamic-dynamic stays in the low colors, disjoint from the static-reserved high end.
        expect(greedyColor(graph, 3, 8, Dynamic, Dynamic)).toBe(0);
    });
});
