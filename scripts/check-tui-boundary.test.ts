// Criterion 2 — "The encoder imports nothing from the engine" (shallot-tui spec, Validation #2).
//
// Mirrors check-imports.test.ts's shape: seed a temporary real violation file inside
// packages/shallot-tui/src/, run the script as a subprocess, assert exit 1, clean up in finally.
// Two violation shapes, since the script guards two escape routes (a relative climb out of the
// package, and a bare @dylanebert/shallot import at any subpath) — an import-graph check that
// only caught one would leave the other silently open.

import { expect, test } from "bun:test";
import { existsSync, unlinkSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const SCRIPT = resolve(import.meta.dir, "check-tui-boundary.ts");
const SEED = resolve(import.meta.dir, "../packages/shallot-tui/src/_s2_arm_violation.ts");

async function runScript(): Promise<{ exitCode: number; out: string }> {
    const proc = Bun.spawn({ cmd: ["bun", SCRIPT], stdout: "pipe", stderr: "pipe" });
    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    return { exitCode, out: stdout + stderr };
}

test("check-tui-boundary — a bare @dylanebert/shallot import exits nonzero (exit 1)", async () => {
    const violation =
        "// S2 arm seed — temporary violation file, removed in finally\n" +
        'import { State } from "@dylanebert/shallot";\n' +
        "export const _seed = State;\n";
    writeFileSync(SEED, violation);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/engine-boundary violation/);
        expect(out).toMatch(/engine-agnostic/);
    } finally {
        if (existsSync(SEED)) unlinkSync(SEED);
    }
});

test("check-tui-boundary — a relative import escaping the package exits nonzero (exit 1)", async () => {
    const violation =
        "// S2 arm seed — temporary violation file, removed in finally\n" +
        'import { rgbEqual } from "../../shallot/src/engine/utils/encode";\n' +
        "export const _seed = rgbEqual;\n";
    writeFileSync(SEED, violation);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/escapes packages\/shallot-tui/);
    } finally {
        if (existsSync(SEED)) unlinkSync(SEED);
    }
});

test("check-tui-boundary — no violations exits 0 (the gate is not vacuously red)", async () => {
    const { exitCode } = await runScript();
    expect(exitCode).toBe(0);
});
