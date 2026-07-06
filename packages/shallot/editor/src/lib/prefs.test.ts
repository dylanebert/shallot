import { describe, expect, test } from "bun:test";
import { clampWidth, PANEL_MAX, PANEL_MIN, parsePrefs, serializePrefs } from "./prefs";

describe("clampWidth", () => {
    test("holds a width inside the bounds", () => {
        expect(clampWidth(300)).toBe(300);
    });
    test("clamps below the minimum and above the maximum", () => {
        expect(clampWidth(PANEL_MIN - 100)).toBe(PANEL_MIN);
        expect(clampWidth(PANEL_MAX + 100)).toBe(PANEL_MAX);
    });
});

describe("parsePrefs", () => {
    test("absent storage is an empty prefs object", () => {
        expect(parsePrefs(null)).toEqual({});
    });
    test("corrupt storage degrades to empty rather than throwing", () => {
        expect(parsePrefs("{not json")).toEqual({});
    });
    test("round-trips through serializePrefs", () => {
        const prefs = {
            overlays: { edit: 3, play: 0 },
            outlinerWidth: 320,
            plugins: { proj: { a: false, b: true } },
        };
        expect(parsePrefs(serializePrefs(prefs))).toEqual(prefs);
    });
});
