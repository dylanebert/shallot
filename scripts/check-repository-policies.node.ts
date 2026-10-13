import { setDefaultTimeout, test } from "bun:test";

import { CEILING } from "@dylanebert/shallot/testing";

setDefaultTimeout(CEILING.node);

import { resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

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
