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
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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

test("batch subprocess preserves page verdicts and refuses incomplete transport", async () => {
    const tmp = mkdtempSync(join(tmpdir(), "shallot-batch-"));
    const stub = join(tmp, "bun");
    writeFileSync(stub, '#!/bin/sh\nprintf "%s\\n" "$BATCH_STDOUT"\nexit "$BATCH_EXIT"\n');
    chmodSync(stub, 0o755);
    const page = (name: string, pass: boolean) => ({
        pass,
        url: `http://localhost:1234/?scenario=${name}`,
    });
    const cases = [
        { rows: [page("a", true), page("b", true)], exit: 0, pass: true, observed: [true, true] },
        {
            rows: [page("a", true), page("b", false)],
            exit: 1,
            pass: false,
            observed: [true, false],
        },
        { rows: [page("a", true), page("b", true)], exit: 1, pass: false, observed: [true, true] },
        {
            rows: [page("a", true), page("b", false)],
            exit: 0,
            pass: false,
            observed: [true, false],
        },
        { rows: [page("a", true)], exit: 0, pass: false, observed: [true, null] },
        { rows: [page("a", true), page("a", true)], exit: 0, pass: false, observed: [true, null] },
        { rows: [page("b", true), page("a", true)], exit: 0, pass: false, observed: [null, null] },
        { rows: [page("a", true), { pass: "yes" }], exit: 0, pass: false, observed: [true, null] },
        { rows: [], exit: 0, pass: false, observed: [null, null] },
        { rows: "truncated", exit: 1, pass: false, observed: [null, null] },
    ];
    try {
        for (const c of cases) {
            const proc = Bun.spawn(
                [
                    process.execPath,
                    "--eval",
                    `import { verifyBatch } from ${JSON.stringify(new URL("./verify.ts", import.meta.url).pathname)}; console.log(JSON.stringify(await verifyBatch("unused", ["scenario=a", "scenario=b"], [], true)));`,
                ],
                {
                    stdout: "pipe",
                    stderr: "pipe",
                    env: {
                        ...process.env,
                        PATH: `${tmp}:${process.env.PATH}`,
                        BATCH_STDOUT: JSON.stringify(c.rows),
                        BATCH_EXIT: String(c.exit),
                    },
                },
            );
            const output = await new Response(proc.stdout).text();
            expect(await proc.exited).toBe(0);
            const outcome = JSON.parse(output);
            expect(outcome.pass).toBe(c.pass);
            expect(outcome.exitCode).toBe(c.exit);
            expect(outcome.results.map((r: { pass: boolean } | null) => r?.pass ?? null)).toEqual(
                c.observed,
            );
        }
        for (const failed of [true, false]) {
            const proc = Bun.spawn(
                [
                    process.execPath,
                    new URL("./bench.ts", import.meta.url).pathname,
                    "--sweep",
                    "--for",
                    "packages/shallot-runtime/src/extras/outline/index.ts",
                    "packages/shallot-runtime/src/extras/cells/grid.ts",
                ],
                {
                    cwd: new URL("..", import.meta.url).pathname,
                    stdout: "pipe",
                    stderr: "pipe",
                    env: {
                        ...process.env,
                        DISPLAY: ":fixture",
                        SHALLOT_DISPLAY_REQUIRED: "1",
                        PATH: `${tmp}:${process.env.PATH}`,
                        BATCH_STDOUT: JSON.stringify([
                            page("outline", true),
                            page("cells", !failed),
                        ]),
                        BATCH_EXIT: "1",
                    },
                },
            );
            const output = await new Response(proc.stdout).text();
            const errors = await new Response(proc.stderr).text();
            expect(await proc.exited).toBe(1);
            expect(output).toContain("✓ outline");
            expect(output).toContain(failed ? "✗ cells" : "✓ cells");
            expect(output).toContain(
                `"selected":2,"executed":2,"pass":${failed ? 1 : 2},"fail":${failed ? 1 : 0},"unavailable":0`,
            );
            expect(errors).toContain("verify process exited 1");
        }
        const missing = join(tmp, "missing-assets.ts");
        writeFileSync(
            missing,
            `import { mock } from "bun:test"; import * as fs from "node:fs";
const exists = fs.existsSync;
mock.module("node:fs", () => ({ ...fs, existsSync: (path) => String(path).endsWith("sponza/Sponza-KTX-Draco.glb") ? false : exists(path) }));`,
        );
        for (const flags of [
            ["--scenario", "gltf"],
            ["--sweep", "--for", "packages/shallot-runtime/src/extras/gltf/index.ts"],
        ]) {
            const proc = Bun.spawn(
                [
                    process.execPath,
                    "--preload",
                    missing,
                    new URL("./bench.ts", import.meta.url).pathname,
                    ...flags,
                ],
                {
                    cwd: new URL("..", import.meta.url).pathname,
                    stdout: "pipe",
                    stderr: "pipe",
                    env: {
                        ...process.env,
                        DISPLAY: ":fixture",
                        SHALLOT_DISPLAY_REQUIRED: "1",
                        PATH: `${tmp}:${process.env.PATH}`,
                    },
                },
            );
            const output = await new Response(proc.stdout).text();
            expect(await proc.exited).toBe(1);
            expect(output).toContain('"selected":1,"executed":0,"pass":0,"fail":0,"unavailable":1');
        }
    } finally {
        rmSync(tmp, { recursive: true, force: true });
    }
});
