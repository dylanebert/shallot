import { afterEach, expect, test } from "bun:test";
import { PhysicsWorld } from "../api";
import { DYNAMIC_COLOR_COUNT, OVERFLOW_INDEX } from "../common/constants";
import { BodyType } from "../common/types";
import { type Kernel, kernel } from "../kernel/kernel";

let world: PhysicsWorld;
function createGraph(_capacity: number): Kernel {
    world = new PhysicsWorld();
    const k = kernel(world.state.ecsState);
    k.bodySetActiveWorld(world.state.worldId);
    return k;
}
afterEach(() => world.destroy());
function assign(graph: Kernel, a: number, b: number, ta: number, tb: number): number {
    return graph.graphAssignColor(a, b, ta, tb);
}
function bodyBit(graph: Kernel, color: number, id: number): boolean {
    return !!graph.graphBodyBit(color, id);
}
const { Static, Kinematic, Dynamic } = BodyType;

test("the constraint graph puts two dynamic-dynamic pairs sharing a body in one color, so the colored solver would write the same body from two constraints in the same batch", () => {
    const graph = createGraph(8);
    expect(assign(graph, 0, 1, Dynamic, Dynamic)).toBe(0);
    expect(bodyBit(graph, 0, 0)).toBe(true);
    expect(bodyBit(graph, 0, 1)).toBe(true);
    expect(assign(graph, 2, 3, Dynamic, Dynamic)).toBe(0);
    expect(assign(graph, 0, 4, Dynamic, Dynamic)).toBe(1);
    expect(bodyBit(graph, 1, 0)).toBe(true);
    expect(bodyBit(graph, 1, 4)).toBe(true);
});

test("a dynamic body already present in every dynamic graph color takes one more constraint into a color instead of the overflow batch, so that constraint would race its own body", () => {
    const graph = createGraph(8);
    for (let i = 0; i < DYNAMIC_COLOR_COUNT; ++i) {
        expect(assign(graph, 0, 10 + i, Dynamic, Dynamic)).toBe(i);
    }
    expect(assign(graph, 0, 99, Dynamic, Dynamic)).toBe(OVERFLOW_INDEX);
});

test("dynamic-static constraints are colored from the low end or record the static body in the color's body set, so static-anchored constraints would crowd the dynamic-dynamic colors and false-conflict on a shared static body", () => {
    const graph = createGraph(8);
    expect(assign(graph, 5, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 1);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 5)).toBe(true);
    expect(assign(graph, 6, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 1);
    expect(assign(graph, 5, 0, Dynamic, Static)).toBe(OVERFLOW_INDEX - 2);
});

test("the graph colorer handles a static body only in the A slot, so a static-A/dynamic-B constraint would be colored as if both sides were dynamic", () => {
    const graph = createGraph(8);
    expect(assign(graph, 0, 7, Static, Dynamic)).toBe(OVERFLOW_INDEX - 1);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 7)).toBe(true);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 0)).toBe(false);
});

test("a kinematic body is colored on the dynamic-dynamic branch, so kinematic-anchored constraints would occupy the low colors and conflict with real dynamic pairs", () => {
    const graph = createGraph(8);
    expect(assign(graph, 3, 8, Kinematic, Dynamic)).toBe(OVERFLOW_INDEX - 1);
    expect(bodyBit(graph, OVERFLOW_INDEX - 1, 8)).toBe(true);
    expect(assign(graph, 3, 8, Dynamic, Dynamic)).toBe(0);
});
