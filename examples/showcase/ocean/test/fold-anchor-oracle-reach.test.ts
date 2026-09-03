import { expect, test } from "bun:test";
import { fileURLToPath } from "node:url";

// The by-path oracle command, both the documented root-package script and the exact command this
// sentinel spawns — one string, so an edit to either side can't drift from the other (pattern:
// `slope-oracle-reach.test.ts`).
const ROOT_ORACLE_COMMAND =
    "FOLD_ENSEMBLE_MODE=full bun test ./examples/showcase/ocean/test/fold-anchor.oracle.ts";
const REDUCED_ORACLE_COMMAND =
    "env FOLD_ENSEMBLE_MODE=reduced bun test ./examples/showcase/ocean/test/fold-anchor.oracle.ts";

const REPO_ROOT = fileURLToPath(new URL("../../../..", import.meta.url));

test("the choppiness fold-anchor oracle remains reachable through the root script", async () => {
    const packageJson = await Bun.file(new URL("../../../../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:ocean-fold"]).toBe(ROOT_ORACLE_COMMAND);
    expect(await Bun.file(new URL("./fold-anchor.oracle.ts", import.meta.url)).exists()).toBe(true);
});

/**
 * The reach sentinel's whole job: prove a default-suite `bun run test` run actually EXECUTES the
 * fold-anchor oracle's assertions, not merely that a `package.json` string names it —
 * `fold-anchor.oracle.ts` sits outside bun's default `.test.ts` glob (bun 1.4.0: a bare filter with
 * no `./` prefix and no `.test`/`.spec` in the name matches nothing), so nothing upstream of this
 * sentinel ever runs it. Spawns `ROOT_ORACLE_COMMAND` itself — argv split from the same string
 * asserted above, never a second hand-typed invocation — from the repo root, and reads its own
 * stdout/stderr for a nonzero, exit-code-independent pass count: an emptied oracle file (zero
 * `test()` calls) still exits 0 with "0 pass, 0 fail", so exit code alone is a vacuous witness.
 * Deleting `fold-anchor.oracle.ts` reds this sentinel too, on exit code (a `./<gone path>` filter
 * matches nothing and exits 1). Bun 1.4.0 writes its own pass/fail summary to STDERR, not STDOUT.
 */
test("a default-suite run actually executes the fold-anchor oracle's assertions", () => {
    const [cmd, ...args] = REDUCED_ORACLE_COMMAND.split(" ");
    const proc = Bun.spawnSync([cmd, ...args], {
        cwd: REPO_ROOT,
        stdout: "pipe",
        stderr: "pipe",
    });
    const stdout = proc.stdout.toString();
    const stderr = proc.stderr.toString();
    expect(proc.exitCode, `${stdout}\n${stderr}`).toBe(0);

    const passMatch = stderr.match(/(\d+) pass/);
    expect(passMatch, `${stdout}\n${stderr}`).not.toBeNull();
    expect(Number(passMatch?.[1])).toBeGreaterThan(0);
    expect(stderr).toContain("0 fail");
});
