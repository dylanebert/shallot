// S3 arm — scripts/check-imports.ts violations cause nonzero exit
//
// Invariant: cross-module import violations cause a nonzero exit (exit 1), not just a console.warn.
// Before the S1 fix, violations only console.warn'd and the script always exited 0 — the bun check
// arm could never fail. The fix added process.exit(1) in the violations branch.
//
// The arm seeds a temporary violation file in packages/shallot/src/, runs the script as a
// subprocess, asserts exit 1, and cleans up in finally. The script resolves paths relative to
// import.meta.dir (hardcoded to packages/shallot/src), so the seed must land in the real tree.
// The temp file is a .ts file that imports across a module boundary through a non-barrel path.

import { expect, test } from "bun:test";
import { existsSync, unlinkSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const SCRIPT = resolve(import.meta.dir, "check-imports.ts");
const SEED = resolve(import.meta.dir, "../packages/shallot/src/engine/_s3_arm_violation.ts");

test("check-imports — violations exit nonzero (exit 1)", async () => {
    // Seed a cross-module import violation: a file in engine/ that imports from standard/render/
    // through a non-barrel path. check-imports.ts scans packages/shallot/src for .ts files and
    // flags cross-module imports that don't go through barrel exports or allowed subpaths.
    const violation =
        `// S3 arm seed — temporary violation file, removed in finally\n` +
        `import { something } from "../standard/render/camera";\n` +
        `export const _seed = something;\n`;
    writeFileSync(SEED, violation);
    try {
        const proc = Bun.spawn({
            cmd: ["bun", SCRIPT],
            stdout: "pipe",
            stderr: "pipe",
        });
        const [stdout, stderr, exitCode] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);
        const out = stdout + stderr;
        expect(exitCode).toBe(1);
        // The exit code is the discriminator; the fragment is a secondary stable check.
        expect(out).toMatch(/cross-module/);
    } finally {
        if (existsSync(SEED)) unlinkSync(SEED);
    }
});

test("check-imports — no violations exits 0 (the gate is not vacuously red)", async () => {
    // The gate should pass when there are no violations — the arm seeds nothing and the real tree
    // is clean. This is the green direction: the gate is not vacuously red.
    const proc = Bun.spawn({
        cmd: ["bun", SCRIPT],
        stdout: "pipe",
        stderr: "pipe",
    });
    const exitCode = await proc.exited;
    expect(exitCode).toBe(0);
});
