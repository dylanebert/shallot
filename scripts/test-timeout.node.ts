import { expect, setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { removeTestDirectory } from "./test-directory";

const ROOT = resolve(import.meta.dir, "..");

test("the test script rejects an over-budget cheap test after the first file", () => {
    const packageJson = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")) as {
        scripts: { test: string };
    };
    const [runtime, command, ...args] = packageJson.scripts.test.trim().split(/\s+/);
    expect([runtime, command]).toEqual(["bun", "test"]);

    const fixture = mkdtempSync(join(tmpdir(), "shallot-test-timeout-"));
    try {
        writeFileSync(
            join(fixture, "first.test.ts"),
            'import { test } from "bun:test";\ntest("first file", () => {});\n',
        );
        writeFileSync(
            join(fixture, "slow.test.ts"),
            'import { test } from "bun:test";\ntest("a slow cheap test", async () => { await new Promise((resolve) => setTimeout(resolve, 350)); });\n',
        );
        const result = Bun.spawnSync(
            [runtime, command, ...args, "./first.test.ts", "./slow.test.ts"],
            {
                cwd: fixture,
                env: { ...process.env, CI: "" },
                stdout: "pipe",
                stderr: "pipe",
            },
        );
        const output = `${result.stdout.toString()}${result.stderr.toString()}`;
        expect(output.indexOf("(pass) first file")).toBeGreaterThanOrEqual(0);
        expect(output.indexOf("(pass) first file")).toBeLessThan(
            output.indexOf("a slow cheap test"),
        );
        expect(result.exitCode, output).toBe(1);
        expect(output).toContain("timed out after 250ms");
    } finally {
        removeTestDirectory(fixture);
    }
});
