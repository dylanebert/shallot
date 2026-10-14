import { expect, setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

test("every CI job declares its timeout budget", async () => {
    for (const workflow of ["test.yml", "release.yml"]) {
        const source = await Bun.file(resolve(ROOT, ".github/workflows", workflow)).text();
        const { jobs } = Bun.YAML.parse(source) as {
            jobs: Record<string, { "timeout-minutes"?: number }>;
        };
        for (const [name, job] of Object.entries(jobs)) {
            expect(job["timeout-minutes"], `${workflow}: ${name}`).toBeGreaterThan(0);
        }
    }
});

test("src/project/policy.test.ts passes", () => {
    const result = Bun.spawnSync(["bun", "test", "src/project/policy.test.ts"], {
        cwd: ROOT,
        stdout: "pipe",
        stderr: "pipe",
    });
    if (!result.success) {
        throw new Error(
            [result.stdout.toString(), result.stderr.toString()].filter(Boolean).join("\n"),
        );
    }
});

test("browser subjects satisfy their repository policy", () => {
    const result = Bun.spawnSync(["sh", "scripts/browser.sh", "--check-only"], {
        cwd: ROOT,
        stdout: "pipe",
        stderr: "pipe",
    });
    if (!result.success) {
        throw new Error(
            [result.stdout.toString(), result.stderr.toString()].filter(Boolean).join("\n"),
        );
    }
});
