import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    realpathSync,
    renameSync,
    statSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { missingCrateDiagnosticPass } from "./install-test";

// V1 physical composition: run the packed public bin in an absent-only consumer, not a workspace link.
// Cost: bun pm pack + bun install + two public native-build commands (no browser, adapter or cargo build).
// Run by path after the focused V1 unit witnesses:
//
//     bun test --timeout 120000 ./scripts/install-test.probes.ts
//
// The original command freezes the wrong-boundary red: Linux's system-webview refusal wins before the
// missing-crate diagnostic. The second changes only this invocation to --portable, which admits the
// intended guard without compiling a native shell. The hidden crate is restored in a finally block and
// its complete path/byte digest is compared after restoration.

const ROOT = resolve(import.meta.dir, "..");
const ENGINE = resolve(ROOT);
const CLI = "node_modules/.bin/shallot";

type CommandResult = { ok: boolean; out: string; exit: number };

function digestTree(root: string): string {
    const digest = createHash("sha256");
    const files = [...new Bun.Glob("**/*").scanSync({ cwd: root })]
        .filter((file) => statSync(join(root, file)).isFile())
        .sort();
    for (const file of files) {
        digest.update(file);
        digest.update(readFileSync(join(root, file)));
    }
    return digest.digest("hex");
}

function run(evidence: string, name: string, argv: string[], cwd: string): CommandResult {
    const result = Bun.spawnSync(argv, { cwd, stdout: "pipe", stderr: "pipe" });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    writeFileSync(join(evidence, `${name}.stdout`), stdout);
    writeFileSync(join(evidence, `${name}.stderr`), stderr);
    writeFileSync(
        join(evidence, `${name}.json`),
        JSON.stringify({ argv, cwd, exit: result.exitCode, runtime: Bun.version }, null, 2),
    );
    return { ok: result.exitCode === 0, out: `${stdout}\n${stderr}`, exit: result.exitCode };
}

function packEngine(work: string): string {
    const packDir = join(work, "pack");
    mkdirSync(packDir);
    const packed = run(work, "pack", ["bun", "pm", "pack", "--destination", packDir], ENGINE);
    expect(packed.ok).toBe(true);
    const tarball = readdirSync(packDir).find((file) => file.endsWith(".tgz"));
    expect(tarball).toBeDefined();
    return join(packDir, tarball as string);
}

describe("packed missing-crate diagnostic admission", () => {
    test("the original boundary reds, portable reaches the diagnostic, and the crate is restored exactly", () => {
        const work = realpathSync(mkdtempSync(join(tmpdir(), "shallot-verification-default-v1-")));
        const app = join(work, "app");
        mkdirSync(app);
        const tarball = packEngine(work);
        writeFileSync(
            join(app, "package.json"),
            `${JSON.stringify(
                {
                    private: true,
                    type: "module",
                    dependencies: {
                        "@dylanebert/shallot": `file:${tarball}`,
                        typegpu: "~0.12.5",
                        vite: "^8.0.16",
                    },
                },
                null,
                2,
            )}\n`,
        );
        writeFileSync(join(app, "shallot.json"), "{}\n");
        const installed = run(work, "install", ["bun", "install"], app);
        expect(installed.ok).toBe(true);

        const shipped = join(app, "node_modules/@dylanebert/shallot");
        expect(existsSync(shipped)).toBe(true);
        expect(realpathSync(shipped)).toBe(shipped);
        const crate = join(shipped, "rust/native");
        const hidden = `${crate}.hidden`;
        expect(existsSync(join(crate, "Cargo.toml"))).toBe(true);
        expect(existsSync(hidden)).toBe(false);
        const before = digestTree(crate);

        let original: CommandResult | undefined;
        let portable: CommandResult | undefined;
        try {
            renameSync(crate, hidden);
            original = run(work, "original", ["bun", CLI, "build", ".", "--target", "linux"], app);
            portable = run(
                work,
                "portable",
                ["bun", CLI, "build", ".", "--target", "linux", "--portable"],
                app,
            );
        } finally {
            if (existsSync(hidden)) renameSync(hidden, crate);
        }

        expect(original).toBeDefined();
        expect(original?.ok).toBe(false);
        expect(original?.out).toContain("Cannot build linux");
        expect(missingCrateDiagnosticPass(original as CommandResult)).toBe(false);

        expect(portable).toBeDefined();
        expect(portable?.ok).toBe(false);
        expect(missingCrateDiagnosticPass(portable as CommandResult)).toBe(true);
        expect(portable?.out).not.toMatch(/ENOENT|No such file or directory/);
        expect(portable?.out).not.toMatch(/cargo (?:build|xwin)/);

        expect(existsSync(join(crate, "Cargo.toml"))).toBe(true);
        expect(existsSync(hidden)).toBe(false);
        expect(digestTree(crate)).toBe(before);
        console.log(`V1 physical evidence: ${work}`);
    }, 120_000);
});
