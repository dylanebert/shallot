import { expect, setDefaultTimeout, test } from "bun:test";

setDefaultTimeout(20_000);

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

test("the test script's default rejects a cheap test that runs longer than 250 ms", () => {
    const packageJson = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")) as {
        scripts: { test: string };
    };
    const [runtime, command, ...args] = packageJson.scripts.test.trim().split(/\s+/);
    expect([runtime, command]).toEqual(["bun", "test"]);

    const fixture = mkdtempSync(join(tmpdir(), "shallot-test-timeout-"));
    try {
        writeFileSync(
            join(fixture, "slow.test.ts"),
            'import { test } from "bun:test";\ntest("a slow cheap test", async () => { await new Promise((resolve) => setTimeout(resolve, 350)); });\n',
        );
        const result = Bun.spawnSync([runtime, command, ...args, "./slow.test.ts"], {
            cwd: fixture,
            stdout: "pipe",
            stderr: "pipe",
        });
        const output = `${result.stdout.toString()}${result.stderr.toString()}`;
        expect(result.exitCode, output).toBe(1);
        expect(output).toContain("timed out after 250ms");
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
}, 5_000);
