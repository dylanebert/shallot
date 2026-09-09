import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { main, scaffold, template } from "../../create-shallot/index";

describe("scaffold", () => {
    test("writes every templated file, creating nested parent dirs (public/, public/scenes/)", () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-scaffold-"));
        scaffold(dir, template("demo"));

        for (const rel of [
            "package.json",
            "shallot.json",
            "public/icon.svg",
            "public/scenes/scene.scene",
        ]) {
            expect(existsSync(join(dir, rel))).toBe(true);
        }
        expect(JSON.parse(readFileSync(join(dir, "package.json"), "utf-8")).name).toBe("demo");
    });
});

describe("main", () => {
    test("no project-name argument: usage error, exit 1, nothing written", () => {
        expect(main([])).toBe(1);
        expect(main(["--flag-only"])).toBe(1);
    });

    test("an existing directory is refused rather than overwritten", () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-create-"));
        expect(main([dir])).toBe(1);
    });

    test("scaffolds the project and returns 0", () => {
        // an absolute path stands in for the project name: `resolve(name)` is then the name itself,
        // so the test controls the target dir without touching process.cwd().
        const dir = join(mkdtempSync(join(tmpdir(), "shallot-create-")), "fresh-project");
        expect(main([dir])).toBe(0);
        expect(existsSync(join(dir, "shallot.json"))).toBe(true);
        expect(existsSync(join(dir, "public/icon.svg"))).toBe(true);
    });
});
