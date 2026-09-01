// "The encoder imports nothing from the engine" — packages/shallot-tui must stay engine-agnostic.
//
// Mirrors check-imports.test.ts's shape: seed a temporary real violation inside
// packages/shallot-tui, run the script as a subprocess, assert exit 1, clean up in finally.
// Seven violation shapes, since the script guards seven escape routes: a relative climb out of
// the package (static), a bare @dylanebert/shallot import at any subpath (static), the same two
// via a dynamic import, a call-style specifier (require/createRequire), a workspace sibling that
// itself depends on the engine, an unscanned tests/ directory, and a package.json dependency
// edge. A check that only caught the first two would leave the rest silently open — which is
// exactly what happened before this file's B3 repair.

import { expect, test } from "bun:test";
import { existsSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const SCRIPT = resolve(import.meta.dir, "check-tui-boundary.ts");
const PKG_ROOT = resolve(import.meta.dir, "../packages/shallot-tui");
const SEED_SRC = resolve(PKG_ROOT, "src/_s2_arm_violation.ts");
const SEED_TESTS = resolve(PKG_ROOT, "tests/_s2_arm_violation.test.ts");
const PKG_JSON = resolve(PKG_ROOT, "package.json");

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
    writeFileSync(SEED_SRC, violation);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/engine-boundary violation/);
        expect(out).toMatch(/engine-agnostic/);
    } finally {
        if (existsSync(SEED_SRC)) unlinkSync(SEED_SRC);
    }
});

test("check-tui-boundary — a relative import escaping the package exits nonzero (exit 1)", async () => {
    const violation =
        "// S2 arm seed — temporary violation file, removed in finally\n" +
        'import { rgbEqual } from "../../shallot/src/engine/utils/encode";\n' +
        "export const _seed = rgbEqual;\n";
    writeFileSync(SEED_SRC, violation);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/escapes packages\/shallot-tui/);
    } finally {
        if (existsSync(SEED_SRC)) unlinkSync(SEED_SRC);
    }
});

test("check-tui-boundary — B3 evasion 1: a dynamic import() of the engine exits nonzero (exit 1)", async () => {
    const violation =
        "// S2 arm seed — temporary violation file, removed in finally\n" +
        "export async function load() {\n" +
        '    return await import("@dylanebert/shallot");\n' +
        "}\n";
    writeFileSync(SEED_SRC, violation);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/engine-boundary violation/);
        expect(out).toMatch(/@dylanebert\/shallot/);
    } finally {
        if (existsSync(SEED_SRC)) unlinkSync(SEED_SRC);
    }
});

test("check-tui-boundary — B3 evasion 2: createRequire(...)(engine) exits nonzero (exit 1)", async () => {
    const violation =
        "// S2 arm seed — temporary violation file, removed in finally\n" +
        'import { createRequire } from "node:module";\n' +
        "const req = createRequire(import.meta.url);\n" +
        'export const _seed = req("@dylanebert/shallot");\n';
    writeFileSync(SEED_SRC, violation);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/engine-boundary violation/);
        expect(out).toMatch(/@dylanebert\/shallot/);
    } finally {
        if (existsSync(SEED_SRC)) unlinkSync(SEED_SRC);
    }
});

test("check-tui-boundary — B3 evasion 3: a workspace sibling that itself depends on the engine exits nonzero (exit 1)", async () => {
    // @dylanebert/shallot-ocean declares @dylanebert/shallot as a peer/dev dependency in its own
    // package.json (packages/shallot-ocean/package.json) — importing it is a one-hop route to the
    // engine that a bare specifier-prefix check misses, since "shallot-ocean" isn't a subpath of
    // "shallot".
    const violation =
        "// S2 arm seed — temporary violation file, removed in finally\n" +
        'import { Ocean } from "@dylanebert/shallot-ocean";\n' +
        "export const _seed = Ocean;\n";
    writeFileSync(SEED_SRC, violation);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/engine-boundary violation/);
        expect(out).toMatch(/shallot-ocean/);
        expect(out).toMatch(/transitive/);
    } finally {
        if (existsSync(SEED_SRC)) unlinkSync(SEED_SRC);
    }
});

test("check-tui-boundary — B3 evasion 4: tests/ is scanned too, not just src/ (exit 1)", async () => {
    const violation =
        "// S2 arm seed — temporary violation file, removed in finally\n" +
        'import { State } from "@dylanebert/shallot";\n' +
        "export const _seed = State;\n";
    writeFileSync(SEED_TESTS, violation);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/engine-boundary violation/);
        expect(out).toMatch(/tests\/_s2_arm_violation\.test\.ts/);
    } finally {
        if (existsSync(SEED_TESTS)) unlinkSync(SEED_TESTS);
    }
});

test("check-tui-boundary — B3 evasion 5: a package.json dependency on the engine exits nonzero (exit 1), unread by a source scan alone", async () => {
    const original = readFileSync(PKG_JSON, "utf8");
    const pkg = JSON.parse(original);
    pkg.dependencies = { "@dylanebert/shallot": "workspace:*" };
    writeFileSync(PKG_JSON, `${JSON.stringify(pkg, null, 4)}\n`);
    try {
        const { exitCode, out } = await runScript();
        expect(exitCode).toBe(1);
        expect(out).toMatch(/engine-boundary violation/);
        expect(out).toMatch(/package\.json dependency/);
    } finally {
        writeFileSync(PKG_JSON, original);
    }
});

test("check-tui-boundary — no violations exits 0 (the gate is not vacuously red)", async () => {
    const { exitCode } = await runScript();
    expect(exitCode).toBe(0);
});
