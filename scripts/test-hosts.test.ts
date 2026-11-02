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
    "examples/gltf",
    "src/core/rendering/tests/images",
    "src/extras/effect-stack/browser",
];

test("hosted, hostFiles keeps, in order, only the given paths a platform runs, under either separator", () => {
    expect(hostFiles(FILES, "darwin", true)).toEqual(FILES);
    expect(hostFiles(FILES, "linux", true)).toEqual([
        FILES[0],
        FILES[2],
        FILES[4],
        FILES[5],
        FILES[6],
    ]);
    expect(hostFiles(FILES, "win32", true)).toEqual([FILES[0]]);
});

test("physics hints follow panels on both hosted browser platforms", () => {
    const subjects = ["examples/physics/tests/panels", "examples/physics/tests/hints"];
    for (const platform of ["linux", "win32"] as const)
        expect(hostFiles(subjects, platform, true)).toEqual(subjects);
    expect(hostFiles(subjects, "darwin", true)).toEqual([]);
    for (const platform of ["darwin", "linux", "win32"] as const)
        expect(hostFiles(subjects, platform, false)).toEqual(subjects);
});

test("physics stepped subjects keep the same hosted browser platforms", () => {
    const subjects = ["examples/physics/tests/stepped", "examples/physics/tests/stepped-controls"];
    for (const platform of ["linux", "win32"] as const)
        expect(hostFiles(subjects, platform, true)).toEqual(subjects);
    expect(hostFiles(subjects, "darwin", true)).toEqual([]);
    for (const platform of ["darwin", "linux", "win32"] as const)
        expect(hostFiles(subjects, platform, false)).toEqual(subjects);
});

test("outside hosted CI, every path runs on every platform", () => {
    for (const platform of ["darwin", "linux", "win32"] as const)
        expect(hostFiles(FILES, platform, false)).toEqual(FILES);
});
