import { afterEach, expect, test } from "bun:test";
import { DYNAMIC_COLOR_COUNT, OVERFLOW_INDEX } from "../common/constants";
import { BodyType } from "../common/types";
import { type Kernel, kernel } from "../kernel/kernel";
import { createTestWorld, destroyTestWorld } from "../testing";

let world: ReturnType<typeof createTestWorld>;
function createGraph(_capacity: number): Kernel {
    world = createTestWorld();
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
afterEach(() => destroyTestWorld(world));
function assign(graph: Kernel, a: number, b: number, ta: number, tb: number): number {
    return graph.graphAssignColor(a, b, ta, tb);
}
function bodyBit(graph: Kernel, color: number, id: number): boolean {
    return !!graph.graphBodyBit(color, id);
}
const { Static, Kinematic, Dynamic } = BodyType;

test("the constraint graph puts a dynamic-dynamic pair sharing a body with a colored pair in the next color, and a disjoint pair in the same color", () => {
    const graph = createGraph(8);
    expect(assign(graph, 0, 1, Dynamic, Dynamic)).toBe(0);
    expect(bodyBit(graph, 0, 0)).toBe(true);
    expect(bodyBit(graph, 0, 1)).toBe(true);
    expect(assign(graph, 2, 3, Dynamic, Dynamic)).toBe(0);
    expect(assign(graph, 0, 4, Dynamic, Dynamic)).toBe(1);
    expect(bodyBit(graph, 1, 0)).toBe(true);
    expect(bodyBit(graph, 1, 4)).toBe(true);
});

test("a constraint on a dynamic body already in every dynamic graph color goes to the overflow batch", () => {
    const graph = createGraph(8);
    for (let i = 0; i < DYNAMIC_COLOR_COUNT; ++i) {
        expect(assign(graph, 0, 10 + i, Dynamic, Dynamic)).toBe(i);
    }
    expect(assign(graph, 0, 99, Dynamic, Dynamic)).toBe(OVERFLOW_INDEX);
});

test("dynamic-static constraints take colors from the high end and record only the dynamic body, so a shared static body never conflicts", () => {
    const graph = createGraph(8);
    expect(assign(graph, 5, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 1);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 5)).toBe(true);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 0)).toBe(false);
    expect(assign(graph, 6, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 1);
    expect(assign(graph, 5, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 2);
});

test("a static-A/dynamic-B constraint is colored from the high end and records only body B", () => {
    const graph = createGraph(8);
    expect(assign(graph, 0, 7, Static, Dynamic)).toBe(OVERFLOW_INDEX - 1);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 7)).toBe(true);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 0)).toBe(false);
});

test("a kinematic-dynamic constraint is colored like a static one, from the high end, leaving the low colors to dynamic pairs", () => {
    const graph = createGraph(8);
    expect(assign(graph, 3, 8, Kinematic, Dynamic)).toBe(OVERFLOW_INDEX - 1);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 8)).toBe(true);
    expect(assign(graph, 3, 8, Dynamic, Dynamic)).toBe(0);
});
