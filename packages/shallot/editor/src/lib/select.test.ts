import { describe, expect, test } from "bun:test";
import { opensMenu, step, typeahead } from "./select";

describe("step", () => {
    test("arrows move and wrap", () => {
        expect(step("ArrowDown", 0, 3)).toEqual({ active: 1, commit: false, close: false });
        expect(step("ArrowDown", 2, 3)).toEqual({ active: 0, commit: false, close: false });
        expect(step("ArrowUp", 0, 3)).toEqual({ active: 2, commit: false, close: false });
    });

    test("Home/End jump to the ends", () => {
        expect(step("Home", 2, 3)?.active).toBe(0);
        expect(step("End", 0, 3)?.active).toBe(2);
    });

    test("Enter and Space commit and close at the current highlight", () => {
        expect(step("Enter", 1, 3)).toEqual({ active: 1, commit: true, close: true });
        expect(step(" ", 2, 3)).toEqual({ active: 2, commit: true, close: true });
    });

    test("Escape and Tab close without committing", () => {
        expect(step("Escape", 1, 3)).toEqual({ active: 1, commit: false, close: true });
        expect(step("Tab", 1, 3)).toEqual({ active: 1, commit: false, close: true });
    });

    test("unhandled key passes through", () => {
        expect(step("a", 0, 3)).toBeNull();
    });

    test("empty option set handles every key as a no-op", () => {
        expect(step("ArrowDown", 0, 0)).toBeNull();
        expect(step("Enter", 0, 0)).toBeNull();
    });
});

describe("opensMenu", () => {
    test("opens on arrows and Enter/Space, ignores others", () => {
        for (const k of ["ArrowDown", "ArrowUp", "Enter", " "]) expect(opensMenu(k)).toBe(true);
        for (const k of ["Escape", "Tab", "a", "Home"]) expect(opensMenu(k)).toBe(false);
    });
});

describe("typeahead", () => {
    const labels = ["free", "locked", "loose"];

    test("matches the next label starting with the char, wrapping from `from`", () => {
        expect(typeahead(labels, "f", 0)).toBe(0);
        expect(typeahead(labels, "l", 0)).toBe(1);
    });

    test("repeated press cycles between matches", () => {
        expect(typeahead(labels, "l", 1)).toBe(2);
        expect(typeahead(labels, "l", 2)).toBe(1);
    });

    test("case-insensitive", () => {
        expect(typeahead(labels, "F", 2)).toBe(0);
    });

    test("no match returns -1", () => {
        expect(typeahead(labels, "z", 0)).toBe(-1);
    });
});
