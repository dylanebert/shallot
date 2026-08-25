// S3 arm — scripts/verify.ts discarded child exit code reddens the verdict
//
// Invariant: a discarded child exit code reddens the verdict. Before the S1 fix, spawnVerify
// awaited proc.exited and discarded the code — the verdict rested entirely on parsed stdout.
// The fix returns exitCode from spawnVerify, and verify() applies applyExitCodeGate() which
// checks `exitCode !== 0` and sets pass: false regardless of what the parsed stdout says.
//
// The guard exists for the opposite case of a missing dir: a driver exiting nonzero while its
// stdout plausibly says `pass: true`. The arm builds that fixture — a stub driver script that
// prints a passing JSON envelope and exits nonzero — runs it as a real subprocess, captures the
// real stdout and exit code, and passes them through the real extractResult + applyExitCodeGate.
// The verdict must be red (pass: false). This is behavioral: it exercises the actual decision
// logic against a real subprocess's output and exit code, not a grep over source text.

import { expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { applyExitCodeGate, extractResult, verify } from "./verify";

test("verify — a nonzero child exit with a passing envelope reddens the verdict", async () => {
    // Build a stub driver script that prints a passing JSON envelope and exits nonzero.
    // This is the case the guard exists for: stdout says pass:true but the process exited nonzero.
    const tmp = mkdtempSync(join(tmpdir(), "shallot-verify-arm-"));
    const stub = join(tmp, "stub-driver.ts");
    writeFileSync(
        stub,
        `// stub: prints a passing envelope, exits 1\n` +
            `console.log(JSON.stringify({ pass: true, verdict: { ok: true } }));\n` +
            `process.exit(1);\n`,
    );
    try {
        const proc = Bun.spawn({ cmd: ["bun", stub], stdout: "pipe", stderr: "pipe" });
        const [stdout, exitCode] = await Promise.all([
            new Response(proc.stdout).text(),
            proc.exited,
        ]);
        // The stub exited nonzero with a passing envelope on stdout.
        expect(exitCode).not.toBe(0);
        const result = extractResult(stdout);
        expect(result?.pass).toBe(true);
        // The gate must redden the verdict — pass: false regardless of stdout.
        const verdict = applyExitCodeGate(result, exitCode);
        expect(verdict).not.toBeNull();
        expect(verdict!.pass).toBe(false);
    } finally {
        rmSync(tmp, { recursive: true, force: true });
    }
});

test("verify — a zero exit with a passing envelope stays green (gate is not vacuously red)", () => {
    const stdout = JSON.stringify({ pass: true, verdict: { ok: true } });
    const result = extractResult(stdout);
    const verdict = applyExitCodeGate(result, 0);
    expect(verdict).not.toBeNull();
    expect(verdict!.pass).toBe(true);
});

test("verify — a nonexistent dir still reds (the original behavioral arm, kept)", async () => {
    // The original arm: a nonexistent dir causes the CLI subprocess to exit nonzero and report
    // pass: false. This still works and is kept as a second behavioral check.
    const result = await verify("/nonexistent-shallot-arm-dir-12345", [], true);
    expect(result).not.toBeNull();
    expect(result!.pass).toBe(false);
});
