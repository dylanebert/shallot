import { expect, test } from "bun:test";

const ORACLE_COMMAND = "bun test ./packages/shallot-ocean/tests/realization.oracle.ts";

/**
 * This sentinel is string-match-only, the exact defect class `slope-oracle-reach.test.ts` now
 * names in its own docblock and fixes for its sibling oracle (spawn `ORACLE_COMMAND`, read a
 * nonzero pass count off the child's own stdout/stderr): it asserts only that `package.json`'s
 * `test:ocean-realization` string matches and that the oracle file exists, never that a default-
 * suite run actually executes the realization oracle's assertions. The execution-proving form is
 * NOT copied here: `bun test ./packages/shallot-ocean/tests/realization.oracle.ts` measured at
 * 13.08s wall clock in this worktree (2026-09-01, `/usr/bin/time -v`, 5 pass / 0 fail), well over
 * shallot's 5000ms per-file test-duration cap (`checks.md`'s test-cap discipline; also cited at
 * `scripts/check-docs.test.ts:18`) — spawning it from inside this file would red on the cap alone,
 * not on any real defect in the oracle. Until `realization.oracle.ts` itself gets cheaper or the
 * cap gets a stated per-subprocess carve-out, this sentinel stays string-match-only, and the gap
 * it leaves stands named here: nothing upstream of a person running
 * `bun run test:ocean-realization` by hand proves the realization oracle's assertions execute in
 * the default `bun run test` suite.
 */
test("the real-space realization oracle remains reachable through its package script", async () => {
    const packageJson = await Bun.file(new URL("../../../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:ocean-realization"]).toBe(ORACLE_COMMAND);
    expect(await Bun.file(new URL("./realization.oracle.ts", import.meta.url)).exists()).toBe(true);
});
