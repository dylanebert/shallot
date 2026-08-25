// S3 arm — scripts/local-test.ts failed scaffold stops before pack
//
// Invariant: a failed scaffold (bun create-shallot exits nonzero) stops the run before pack.
// Before the S1 fix, the scaffold spawnSync exit code was unchecked — a failed scaffold fell
// through to `bun pm pack`. The fix checks `scaffold.exitCode !== 0` and exits 1.
//
// The arm runs local-test.ts as a subprocess with a name whose parent directory doesn't exist,
// so `bun create-shallot <name>` fails (cannot create the dir). The script must exit 1 with
// "scaffold failed" in the output, and must NOT reach the pack step (no "pack failed" or "cd").

import { expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const SCRIPT = resolve(import.meta.dir, "local-test.ts");

test("local-test — a failed scaffold exits 1 before pack", async () => {
    // Create a file (not a directory) as a blocker: create-shallot tries to mkdirSync under the
    // project dir, and a file where a directory is expected causes ENOTDIR → scaffold exits nonzero.
    // The name is relative to cwd, so we set cwd to a temp dir to keep the test hermetic.
    const tmp = mkdtempSync(join(tmpdir(), "shallot-local-arm-"));
    writeFileSync(join(tmp, "blocker"), ""); // a file, not a directory
    const name = join("blocker", "test-project");

    const proc = Bun.spawn({
        cmd: ["bun", SCRIPT, name],
        stdout: "pipe",
        stderr: "pipe",
        cwd: tmp,
    });
    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    const out = stdout + stderr;

    // The scaffold must fail and the script must exit 1.
    expect(exitCode).toBe(1);
    expect(out).toContain("scaffold failed");

    // The script must NOT reach the pack step — no "pack failed" or "cd " in the output.
    expect(out).not.toContain("pack failed");
    expect(out).not.toMatch(/^cd /m);

    rmSync(tmp, { recursive: true, force: true });
});
