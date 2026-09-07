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
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { applyExitCodeGate, extractResult, verify } from "./verify";
import { bridgePrereq, headedAssignment } from "./wsl-bridge";

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

// The WSL bridge — not the verify CLI — is what launches the browser under `--connect`, so
// `SHALLOT_HEADED` has to cross the powershell hop to the host launcher or a "headed" attribution run
// is silently headless and every frame-timing number it prints was taken on a display-less frame
// clock. These arms pin the assignment both ways and pin that nothing from the environment is
// interpolated into a powershell command line.
test("batch subprocess preserves page verdicts and refuses incomplete transport", async () => {
    const tmp = mkdtempSync(join(tmpdir(), "shallot-batch-"));
    const stub = join(tmp, "bun");
    writeFileSync(stub, '#!/bin/sh\nprintf "%s\\n" "$BATCH_STDOUT"\nexit "$BATCH_EXIT"\n');
    chmodSync(stub, 0o755);
    // The arm's subject is the batch protocol, not the seat's transport. `scripts/verify.ts` picks its
    // child command from `isWSL` (an `existsSync` of the WSLInterop binfmt entry), so on a WSL seat the
    // spawn is `node <bundle> --connect <bridge>` and the stub on PATH is never reached — the arm would
    // stand up a real Windows-host browser bridge instead of deciding a protocol case. This preload makes
    // that one probe read false in every child, so both seats take the native `bun <cli>` shape the stub
    // intercepts. It is a no-op on a seat that has no WSLInterop entry.
    const nativeTransport = join(tmp, "native-transport.ts");
    writeFileSync(
        nativeTransport,
        `import { mock } from "bun:test"; import * as fs from "node:fs";
const exists = fs.existsSync;
mock.module("node:fs", () => ({ ...fs, existsSync: (path) => String(path) === "/proc/sys/fs/binfmt_misc/WSLInterop" ? false : exists(path) }));`,
    );
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
                    "--preload",
                    nativeTransport,
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
                    "--preload",
                    nativeTransport,
                    new URL("./bench.ts", import.meta.url).pathname,
                    "--sweep",
                    "--for",
                    "packages/shallot/src/extras/outline/index.ts",
                    "packages/shallot/src/extras/cells/grid.ts",
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
mock.module("node:fs", () => ({ ...fs, existsSync: (path) => String(path).endsWith("sponza/Sponza-KTX-Draco.glb") || String(path) === "/proc/sys/fs/binfmt_misc/WSLInterop" ? false : exists(path) }));`,
        );
        for (const flags of [
            ["--scenario", "gltf"],
            ["--sweep", "--for", "packages/shallot/src/extras/gltf/index.ts"],
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
test("headedAssignment emits the $env line only when SHALLOT_HEADED is set", () => {
    expect(headedAssignment({})).toBe("");
    expect(headedAssignment({ SHALLOT_HEADED: "1" })).toBe("$env:SHALLOT_HEADED='1'; ");
});

test("headedAssignment normalizes the value rather than interpolating the environment's", () => {
    expect(headedAssignment({ SHALLOT_HEADED: "'; calc.exe; $x='" })).toBe(
        "$env:SHALLOT_HEADED='1'; ",
    );
});

test("the host launcher gates headless on SHALLOT_HEADED rather than hardcoding true", () => {
    const source = readFileSync(new URL("./wsl-bridge.ts", import.meta.url), "utf8");
    expect(source).toContain("headless: !process.env.SHALLOT_HEADED");
    expect(source).not.toContain("headless: true");
});

// A seat without powershell interop is the prerequisite probe's own documented outcome — a reason it
// returns, so `skipReason` can refuse with it. `Bun.spawnSync` throws on an absent executable rather
// than reporting a nonzero exit, so the probe has to survive that: without the catch in `sh` the throw
// escapes here and the selector exits 2 with no attributable row instead of failing closed. Driven in a
// child because PATH is process-wide.
test("bridgePrereq reports a reason on a seat with no powershell interop rather than throwing", async () => {
    const path = (process.env.PATH ?? "")
        .split(":")
        .filter((entry) => !entry.startsWith("/mnt/"))
        .join(":");
    const proc = Bun.spawn(
        [
            process.execPath,
            "--eval",
            `import { bridgePrereq } from ${JSON.stringify(new URL("./wsl-bridge.ts", import.meta.url).pathname)}; console.log(JSON.stringify(bridgePrereq()));`,
        ],
        { stdout: "pipe", stderr: "pipe", env: { ...process.env, PATH: path } },
    );
    const [stdout, exitCode] = await Promise.all([new Response(proc.stdout).text(), proc.exited]);
    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout.trim())).toContain("powershell interop");
});

// The in-process direction: on this seat the probe answers without throwing either way, and its answer is
// a reason or null — never an exception the caller has to guess at.
test("bridgePrereq answers with a reason or null on the running seat", () => {
    const reason = bridgePrereq();
    expect(reason === null || typeof reason === "string").toBe(true);
});
