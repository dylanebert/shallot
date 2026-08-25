// S3 arm — scripts/check-versions.ts failed git run fails the gate
//
// Invariant: a failed `git tag --list` (the --release arm's only gate for an unbumped release)
// fails loud (exit 1), not passes by discarding the exit code. Before the S1 fix, the script read
// `git tag --list` stdout without checking exit — a failed git run passed the only gate that
// catches an unbumped release. The fix checks `found.exitCode !== 0` and calls fail().
//
// The arm stubs `git` on PATH with a script that exits 1, runs check-versions.ts --release as a
// subprocess, and asserts exit 1. The stub is a temp executable prepended to PATH.

import { expect, test } from "bun:test";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const SCRIPT = resolve(import.meta.dir, "check-versions.ts");

test("check-versions --release — a failed git tag --list fails loud (exit 1)", async () => {
    // Create a temp dir with a fake `git` that always exits 1.
    const tmp = mkdtempSync(join(tmpdir(), "shallot-versions-arm-"));
    const fakeGit = join(tmp, "git");
    writeFileSync(fakeGit, "#!/bin/sh\nexit 1\n");
    chmodSync(fakeGit, 0o755);

    // Prepend the temp dir to PATH so the fake git is found first.
    const env = { ...process.env, PATH: `${tmp}:${process.env.PATH}` };

    try {
        const proc = Bun.spawn({
            cmd: ["bun", SCRIPT, "--release"],
            stdout: "pipe",
            stderr: "pipe",
            env,
        });
        const [stdout, stderr, exitCode] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);
        const out = stdout + stderr;
        expect(exitCode).toBe(1);
        expect(out).toContain("git tag --list failed");
    } finally {
        rmSync(tmp, { recursive: true, force: true });
    }
});
