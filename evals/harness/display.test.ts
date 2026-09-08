// The eval gate's display decision: which platforms can launch a headed browser, and which must
// skip honestly. The linux branch is the one this seat runs; the others are only reachable through
// the pure seam, which is why it takes the platform rather than reading `process.platform`.

import { expect, test } from "bun:test";
import { detectDisplay, detectDisplayForPlatform } from "./display";

test("detectDisplay — win32 returns false, not true", () => {
    // an unsupported platform returns false. The defect this guards is the opposite default: every
    // non-linux platform returning true, so a headless box never skipped and the gate red for the
    // wrong reason.
    expect(detectDisplayForPlatform("win32")).toBe(false);
});

test("detectDisplay — any unrecognised platform returns false", () => {
    expect(detectDisplayForPlatform("freebsd")).toBe(false);
    expect(detectDisplayForPlatform("sunos")).toBe(false);
});

test("detectDisplay — linux reads the session's display variables, either one", () => {
    const { DISPLAY, WAYLAND_DISPLAY } = process.env;
    try {
        process.env.DISPLAY = ":0";
        WAYLAND_DISPLAY === undefined
            ? delete process.env.WAYLAND_DISPLAY
            : (process.env.WAYLAND_DISPLAY = WAYLAND_DISPLAY);
        expect(detectDisplayForPlatform("linux")).toBe(true);

        delete process.env.DISPLAY;
        process.env.WAYLAND_DISPLAY = "wayland-1";
        expect(detectDisplayForPlatform("linux")).toBe(true);

        delete process.env.WAYLAND_DISPLAY;
        expect(detectDisplayForPlatform("linux")).toBe(false);
    } finally {
        DISPLAY === undefined ? delete process.env.DISPLAY : (process.env.DISPLAY = DISPLAY);
        WAYLAND_DISPLAY === undefined
            ? delete process.env.WAYLAND_DISPLAY
            : (process.env.WAYLAND_DISPLAY = WAYLAND_DISPLAY);
    }
});

test("detectDisplay delegates to the seam with this process's own platform", () => {
    expect(detectDisplay()).toBe(detectDisplayForPlatform(process.platform));
});
