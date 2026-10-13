import { expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { relative, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

function exampleFiles(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const path = resolve(directory, entry.name);
        if (entry.isDirectory()) {
            if (["build", "dist", "node_modules"].includes(entry.name)) return [];
            return exampleFiles(path);
        }
        return entry.isFile() ? [path] : [];
    });
}

function examplesWithoutChecks(): string[] {
    return readdirSync(resolve(ROOT, "examples"), { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => {
            const directory = resolve(ROOT, "examples", entry.name);
            const hasCheck = exampleFiles(directory).some((path) =>
                /\.(?:test|node|e2e)\.ts$/.test(path),
            );
            return hasCheck ? undefined : relative(ROOT, directory).replaceAll("\\", "/");
        })
        .filter((path): path is string => path !== undefined)
        .sort();
}

test("every installed example has at least one check", () => {
    expect(examplesWithoutChecks()).toEqual([]);
});
