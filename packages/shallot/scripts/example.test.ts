import { describe, expect, test } from "bun:test";
import { sliceRegion } from "./example";

describe("sliceRegion", () => {
    const src = [
        "import { run } from '@dylanebert/shallot';",
        "",
        "// #region camera",
        "const { state } = await run({});",
        "const cam = state.create();",
        "// #endregion",
        "",
        "// #region configure",
        "    Orbit.distance.set(cam, 20);",
        "    Orbit.sensitivity.set(cam, 0.005);",
        "// #endregion",
    ].join("\n");

    test("extracts a region without its markers", () => {
        expect(sliceRegion(src, "camera")).toBe(
            "const { state } = await run({});\nconst cam = state.create();",
        );
    });

    test("dedents to column zero", () => {
        expect(sliceRegion(src, "configure")).toBe(
            "Orbit.distance.set(cam, 20);\nOrbit.sensitivity.set(cam, 0.005);",
        );
    });

    test("returns null for an absent region", () => {
        expect(sliceRegion(src, "missing")).toBeNull();
    });

    test("returns null for an unterminated region", () => {
        expect(sliceRegion("// #region open\nconst x = 1;", "open")).toBeNull();
    });
});
