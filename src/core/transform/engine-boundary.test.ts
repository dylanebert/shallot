import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Glob } from "bun";

test("the engine names no placement code", () => {
    const root = resolve(import.meta.dir, "../../engine");
    const findings: string[] = [];
    for (const file of new Glob("**/*.ts").scanSync(root)) {
        if (/\.(?:test|node|oracle|fixture)\.ts$/.test(file)) continue;
        if (
            /\bTransform\b|GlobalTransform|globalTransform|\bteleport\b|global-transform/.test(
                readFileSync(resolve(root, file), "utf8"),
            )
        )
            findings.push(file);
    }
    expect(findings).toEqual([]);
});
