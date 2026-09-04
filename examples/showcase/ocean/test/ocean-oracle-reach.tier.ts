// By-path tier for the ocean oracle reach property: each arm must spawn the exact `bun test`
// child for an oracle and observe a nonzero test count, because package-script and file-existence
// sentinels cannot prove that the oracle assertions execute. The subprocess is the property, and
// each child runs a full deterministic oracle, so these arms cannot live under the default suite's
// 5000 ms per-file cap or bun's 5000 ms per-test timeout.
//
// Trigger paths: this file; `package.json`'s `test:ocean-fold`, `test:ocean-slope`, and
// `test:ocean-mesh-inversion` scripts; and
// `fold-anchor.oracle.ts`, `slope.oracle.ts`, `mesh-inversion-sweep.oracle.ts` plus their transitive
// import cones. This header is the by-path registry: run from the Shallot root with
// `bun test ./examples/showcase/ocean/test/ocean-oracle-reach.tier.ts`.

import { expect, test } from "bun:test";
import { fileURLToPath } from "node:url";

const FOLD_REACH_COMMAND =
    "env FOLD_ENSEMBLE_MODE=reduced bun test ./examples/showcase/ocean/test/fold-anchor.oracle.ts";
const MESH_INVERSION_REACH_COMMAND =
    "bun test ./examples/showcase/ocean/test/mesh-inversion-sweep.oracle.ts";
const SLOPE_REACH_COMMAND = "bun test ./examples/showcase/ocean/test/slope.oracle.ts";

const REPO_ROOT = fileURLToPath(new URL("../../../..", import.meta.url));

function assertOracleExecutes(command: string): void {
    const [cmd, ...args] = command.split(" ");
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
}

test("the fold-anchor oracle executes assertions through its by-path command", () => {
    assertOracleExecutes(FOLD_REACH_COMMAND);
}, 30_000);

test("the slope oracle executes assertions through its by-path command", () => {
    assertOracleExecutes(SLOPE_REACH_COMMAND);
}, 30_000);

test("the mesh-inversion-sweep oracle executes assertions through its by-path command", () => {
    assertOracleExecutes(MESH_INVERSION_REACH_COMMAND);
}, 30_000);
