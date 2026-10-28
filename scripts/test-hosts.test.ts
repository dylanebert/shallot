import { expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { hostFiles, ONLY } from "./test-hosts";

const ROOT = resolve(import.meta.dir, "..");

test("every host-limited test file exists", () => {
    expect(Object.keys(ONLY).filter((path) => !existsSync(resolve(ROOT, path)))).toEqual([]);
});

const FILES = [
    "./src/extras/fog/temporal.node.ts",
    ".\\src\\extras\\fog\\fog.node.ts",
    "./scripts/publish-browser-session.node.ts",
    "examples/interact",
];

test("hosted, hostFiles keeps, in order, only the given paths a platform runs, under either separator", () => {
    expect(hostFiles(FILES, "darwin", true)).toEqual(FILES);
    expect(hostFiles(FILES, "linux", true)).toEqual([FILES[0], FILES[2]]);
    expect(hostFiles(FILES, "win32", true)).toEqual([FILES[0]]);
});

test("outside hosted CI, every path runs on every platform", () => {
    for (const platform of ["darwin", "linux", "win32"] as const)
        expect(hostFiles(FILES, platform, false)).toEqual(FILES);
});
