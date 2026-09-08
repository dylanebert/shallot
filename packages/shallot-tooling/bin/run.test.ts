import { describe, expect, test } from "bun:test";
import { linuxRunEnv, resolveRunTarget, windowsRunCommand } from "./run";

describe("resolveRunTarget", () => {
    test("defaults to web when no target is given", () => {
        expect(resolveRunTarget()).toEqual({ kind: "web" });
        expect(resolveRunTarget(undefined)).toEqual({ kind: "web" });
    });

    test("selects mac/linux/windows by name", () => {
        expect(resolveRunTarget("mac")).toEqual({ kind: "mac" });
        expect(resolveRunTarget("linux")).toEqual({ kind: "linux" });
        expect(resolveRunTarget("windows")).toEqual({ kind: "windows" });
    });

    test("an unrecognized target carries its own name for the error message, not a generic tag", () => {
        expect(resolveRunTarget("switch")).toEqual({ kind: "unknown", target: "switch" });
    });
});

describe("linuxRunEnv", () => {
    test("leaves the env untouched when not portable", () => {
        const base = { PATH: "/usr/bin" };
        expect(linuxRunEnv("/out", false, base)).toEqual(base);
    });

    test("portable prefixes LD_LIBRARY_PATH with the bundle's cef/ dir", () => {
        const env = linuxRunEnv("/out", true, { LD_LIBRARY_PATH: "/existing" });
        expect(env.LD_LIBRARY_PATH).toBe("/out/cef:/existing");
    });

    test("portable with no prior LD_LIBRARY_PATH doesn't leave a trailing 'undefined'", () => {
        const env = linuxRunEnv("/out", true, {});
        expect(env.LD_LIBRARY_PATH).toBe("/out/cef:");
    });
});

describe("windowsRunCommand", () => {
    test("cds into the WSL-resolved windows path and launches the project's .exe", () => {
        expect(windowsRunCommand("C:\\Users\\me\\out", "my-project")).toBe(
            "cd 'C:\\Users\\me\\out'; .\\my-project.exe",
        );
    });
});
