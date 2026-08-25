// S3 arm — scripts/flows.ts unknown-selector exit
//
// Invariant: an unknown --flow selector exits nonzero (exit 2), ahead of the skipReason() display
// skip. Before the S1 fix, --flow bogus matched no branch, allPass stayed true, and the script
// exited 0 "PASS: flows green" — a vacuous green. The fix moved the unknown-selector guard ahead
// of skipReason() so a headless seat can observe it.

import { expect, test } from "bun:test";
import { resolve } from "node:path";

const SCRIPT = resolve(import.meta.dir, "flows.ts");

test("flows — unknown --flow exits 2 (ahead of the display skip)", async () => {
    const proc = Bun.spawn({
        cmd: ["bun", SCRIPT, "--flow", "bogus"],
        stdout: "pipe",
        stderr: "pipe",
    });
    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    const out = stdout + stderr;
    expect(exitCode).toBe(2);
    expect(out).toContain("no flow");
});

test("flows --help exits 0 (the selector guard does not fire on help)", async () => {
    const proc = Bun.spawn({
        cmd: ["bun", SCRIPT, "--help"],
        stdout: "pipe",
        stderr: "pipe",
    });
    const exitCode = await proc.exited;
    expect(exitCode).toBe(0);
});
