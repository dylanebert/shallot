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

    test("N6: TERM=dumb wins over COLORTERM=truecolor — supports-color treats a dumb terminal as a hard no-color regardless of COLORTERM's claim", () => {
        expect(detectTier({ isTTY: true, env: { COLORTERM: "truecolor", TERM: "dumb" } })).toBe(
            "glyph",
        );
    });

    test("an unrecognized or absent TERM with no COLORTERM defaults to glyph", () => {
        expect(detectTier({ isTTY: true, env: {} })).toBe("glyph");
        expect(detectTier({ isTTY: true, env: { TERM: "vt100" } })).toBe("glyph");
    });

    test("a -direct TERM (the terminfo direct-color convention) selects truecolor", () => {
        expect(detectTier({ isTTY: true, env: { TERM: "xterm-direct" } })).toBe("truecolor");
    });

    test("xterm-kitty and xterm-ghostty select truecolor by TERM alone — the common ssh case where COLORTERM isn't inherited", () => {
        expect(detectTier({ isTTY: true, env: { TERM: "xterm-kitty" } })).toBe("truecolor");
        expect(detectTier({ isTTY: true, env: { TERM: "xterm-ghostty" } })).toBe("truecolor");
    });
});

describe("FORCE_COLOR — the standard override for CI and wrapped runs", () => {
    test("FORCE_COLOR=0 (or false) forces glyph even with a truecolor-capable COLORTERM", () => {
        expect(detectTier({ isTTY: true, env: { FORCE_COLOR: "0", COLORTERM: "truecolor" } })).toBe(
            "glyph",
        );
        expect(
            detectTier({ isTTY: true, env: { FORCE_COLOR: "false", COLORTERM: "truecolor" } }),
        ).toBe("glyph");
    });

    test("FORCE_COLOR=1 or 2 force ansi256 — this ladder has no separate 16-color tier to map to", () => {
        expect(detectTier({ isTTY: true, env: { FORCE_COLOR: "1" } })).toBe("ansi256");
        expect(detectTier({ isTTY: true, env: { FORCE_COLOR: "2" } })).toBe("ansi256");
    });

    test("FORCE_COLOR=3 forces truecolor", () => {
        expect(detectTier({ isTTY: true, env: { FORCE_COLOR: "3" } })).toBe("truecolor");
    });

    test("a bare or non-numeric FORCE_COLOR (no explicit level) forces the lowest color tier on", () => {
        expect(detectTier({ isTTY: true, env: { FORCE_COLOR: "" } })).toBe("ansi256");
        expect(detectTier({ isTTY: true, env: { FORCE_COLOR: "true" } })).toBe("ansi256");
    });

    test("FORCE_COLOR wins over NO_COLOR — the standard's own precedence, since FORCE_COLOR is the more deliberate, explicit signal", () => {
        expect(detectTier({ isTTY: true, env: { FORCE_COLOR: "3", NO_COLOR: "1" } })).toBe(
            "truecolor",
        );
        expect(detectTier({ isTTY: true, env: { FORCE_COLOR: "0", NO_COLOR: "" } })).toBe("glyph");
    });

    test("N6: FORCE_COLOR overrides a non-TTY sink — the primary reason the variable exists (CI, a wrapped run, `| less -R`), matching supports-color's own short-circuit-only-when-undefined rule", () => {
        expect(detectTier({ isTTY: false, env: { FORCE_COLOR: "3" } })).toBe("truecolor");
        expect(detectTier({ isTTY: false, env: { FORCE_COLOR: "1" } })).toBe("ansi256");
        expect(detectTier({ isTTY: false, env: { FORCE_COLOR: "0" } })).toBe("glyph");
        // absent FORCE_COLOR, a non-TTY sink still always reads plain.
        expect(detectTier({ isTTY: false, env: { COLORTERM: "truecolor" } })).toBe("plain");
    });
});
