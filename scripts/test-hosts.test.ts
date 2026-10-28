import { expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { hostFiles, ONLY } from "./test-hosts";

const ROOT = resolve(import.meta.dir, "..");

test("every host-limited test file exists", () => {
    expect(Object.keys(ONLY).filter((path) => !existsSync(resolve(ROOT, path)))).toEqual([]);
});

test("hostFiles keeps, in order, only the given files a platform runs, under either separator", () => {
    const files = [
        "./src/extras/fog/temporal.node.ts",
        ".\\src\\extras\\fog\\fog.node.ts",
        "./scripts/publish-browser-session.node.ts",
    ];
    expect(hostFiles(files, "darwin")).toEqual(files);
    expect(hostFiles(files, "linux")).toEqual([files[0], files[2]]);
    expect(hostFiles(files, "win32")).toEqual([files[0]]);
});
