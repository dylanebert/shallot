import { describe, expect, test } from "bun:test";
import { detectTier } from "../src/color-support";

describe("detectTier", () => {
    test("non-tty always reads plain, regardless of color env", () => {
        expect(detectTier({ isTTY: false, env: {} })).toBe("plain");
        expect(
            detectTier({ isTTY: false, env: { COLORTERM: "truecolor", TERM: "xterm-256color" } }),
        ).toBe("plain");
    });

    test("NO_COLOR forces glyph on a tty even with a truecolor-capable TERM", () => {
        expect(detectTier({ isTTY: true, env: { NO_COLOR: "1", COLORTERM: "truecolor" } })).toBe(
            "glyph",
        );
    });

    test("an empty NO_COLOR does not force glyph (the spec's own example of an unset value)", () => {
        expect(detectTier({ isTTY: true, env: { NO_COLOR: "", COLORTERM: "truecolor" } })).toBe(
            "truecolor",
        );
    });

    test("COLORTERM=truecolor or 24bit selects truecolor over a lower TERM hint", () => {
        expect(detectTier({ isTTY: true, env: { COLORTERM: "truecolor", TERM: "xterm" } })).toBe(
            "truecolor",
        );
        expect(detectTier({ isTTY: true, env: { COLORTERM: "24bit" } })).toBe("truecolor");
    });

    test("a -256color TERM selects ansi256 absent COLORTERM", () => {
        expect(detectTier({ isTTY: true, env: { TERM: "xterm-256color" } })).toBe("ansi256");
        expect(detectTier({ isTTY: true, env: { TERM: "screen-256color" } })).toBe("ansi256");
    });

    test("TERM=dumb selects glyph even absent NO_COLOR", () => {
        expect(detectTier({ isTTY: true, env: { TERM: "dumb" } })).toBe("glyph");
    });

    test("an unrecognized or absent TERM with no COLORTERM defaults to glyph", () => {
        expect(detectTier({ isTTY: true, env: {} })).toBe("glyph");
        expect(detectTier({ isTTY: true, env: { TERM: "vt100" } })).toBe("glyph");
    });
});
