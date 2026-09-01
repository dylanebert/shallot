import { expect, test } from "bun:test";
import { fileURLToPath } from "node:url";

// The by-path oracle command, both the documented root-package script and the exact command this
// sentinel spawns — one string, so an edit to either side can't drift from the other.
const ROOT_ORACLE_COMMAND =
    "bun test ./packages/shallot-ocean/tests/mesh-inversion-sweep.oracle.ts";

const REPO_ROOT = fileURLToPath(new URL("../../..", import.meta.url));

test("the mesh-inversion-sweep oracle file remains reachable through the root script", async () => {
    const packageJson = await Bun.file(new URL("../../../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:ocean-mesh-inversion"]).toBe(ROOT_ORACLE_COMMAND);
    expect(
        await Bun.file(new URL("./mesh-inversion-sweep.oracle.ts", import.meta.url)).exists(),
    ).toBe(true);
});

/**
 * The reach sentinel's whole job: prove a default-suite `bun run test` run actually EXECUTES the
 * mesh-inversion-sweep oracle's assertions, not merely that a `package.json` string names it —
 * `mesh-inversion-sweep.oracle.ts` sits outside bun's default `.test.ts` glob (see
 * `slope-oracle-reach.test.ts`'s docblock for the witnessed bun 1.4.0 behavior this relies on), so
 * nothing upstream of this sentinel ever runs it. Spawns `ROOT_ORACLE_COMMAND` itself — argv split
 * from the same string asserted above, never a second hand-typed invocation — from the repo root,
 * and reads its own stdout/stderr for a nonzero, exit-code-independent pass count: an emptied
 * oracle file (zero `test()` calls) still exits 0 with "0 pass, 0 fail", so exit code alone is a
 * vacuous witness. Deleting the oracle file reds this sentinel too, on exit code (a `./`-prefixed
 * filter matching nothing exits 1, "had no matches"). Bun 1.4.0 writes its own pass/fail summary to
 * STDERR, not STDOUT — read the summary from the stream bun actually writes it to.
 */
test("a default-suite run actually executes the mesh-inversion-sweep oracle's assertions", () => {
    const [cmd, ...args] = ROOT_ORACLE_COMMAND.split(" ");
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
    const failMatch = stderr.match(/(\d+) fail/);
    expect(failMatch, `${stdout}\n${stderr}`).not.toBeNull();
    expect(Number(failMatch?.[1])).toBe(0);
});
