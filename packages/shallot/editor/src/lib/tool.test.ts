import { describe, expect, test } from "bun:test";
import { DEFAULT_TOOL, TOOLS, Tool, toolForKey } from "./tool";

// the toolbar renders from TOOLS while the gizmo switches on Tool; if the two drift, a mode button
// silently vanishes or doubles. Guard the one invariant that catches it: TOOLS lists every Tool once.
describe("tool", () => {
    test("TOOLS lists every Tool exactly once", () => {
        const listed = TOOLS.map((t) => t.id).sort();
        const all = Object.values(Tool).sort();
        expect(listed).toEqual(all);
    });

    test("DEFAULT_TOOL is one of the tools", () => {
        expect(TOOLS.some((t) => t.id === DEFAULT_TOOL)).toBe(true);
    });

    test("the number row selects tools in toolbar order", () => {
        expect(toolForKey("1")).toBe(Tool.Select);
        expect(toolForKey("2")).toBe(Tool.Move);
        expect(toolForKey("3")).toBe(Tool.Rotate);
        expect(toolForKey("4")).toBe(Tool.Scale);
    });

    test("keys outside the tool range map to nothing", () => {
        expect(toolForKey("0")).toBeNull();
        expect(toolForKey("5")).toBeNull();
        expect(toolForKey("w")).toBeNull();
        expect(toolForKey("")).toBeNull();
    });
});
