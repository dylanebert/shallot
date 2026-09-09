import { expect, test } from "bun:test";
import {
    cpSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

const root = resolve(import.meta.dir, "../../..");

test("projector refuses absent records before inputs, grants generated records, and refuses stale source without writing", () => {
    const fixture = mkdtempSync(join(tmpdir(), "shallot-project-"));
    const script = "packages/shallot-runtime/scripts/project.ts";
    const record = join(fixture, "packages/shallot/runtime-inputs.json");
    const run = (...args: string[]) => {
        const proc = Bun.spawnSync([process.execPath, script, ...args], { cwd: fixture });
        const output = proc.stdout.toString() + proc.stderr.toString();
        console.log(`[project ${args.join(" ")}] exit=${proc.exitCode}\n${output}`);
        return { code: proc.exitCode, output };
    };
    const copy = (file: string) => {
        mkdirSync(dirname(join(fixture, file)), { recursive: true });
        cpSync(join(root, file), join(fixture, file));
    };
    try {
        copy(script);
        copy("packages/shallot/scripts/projections.ts");
        const absent = run("--check");
        expect(absent.code).toBe(1);
        expect(absent.output).toContain("missing runtime-inputs.json");
        expect(absent.output).toContain("bun run build");
        expect(absent.output).not.toContain("ENOENT");
        expect(existsSync(record)).toBe(false);
        const files = Bun.spawnSync(
            [
                "git",
                "ls-files",
                "--",
                "packages/shallot-runtime",
                "packages/shallot-tumble",
                "packages/shallot/package.json",
                "packages/shallot/src/harness/index.ts",
            ],
            { cwd: root },
        );
        expect(files.exitCode).toBe(0);
        for (const file of files.stdout.toString().trim().split("\n")) copy(file);
        for (const name of ["shallot_audio.js", "shallot_audio.d.ts", "shallot_audio.wasm"])
            copy(`packages/shallot-runtime/rust/audio/pkg/${name}`);
        expect(Bun.spawnSync(["git", "init", "--quiet"], { cwd: fixture }).exitCode).toBe(0);
        expect(
            Bun.spawnSync(
                ["git", "add", "packages/shallot-runtime/src", "packages/shallot-tumble/src"],
                { cwd: fixture },
            ).exitCode,
        ).toBe(0);
        const maintained = join(fixture, "packages/shallot/src/harness/index.ts");
        const original = readFileSync(maintained, "utf8");
        expect(run().code).toBe(0);
        expect(run("--check").code).toBe(0);
        const source = join(fixture, "packages/shallot-runtime/src/index.ts");
        writeFileSync(source, `${readFileSync(source, "utf8")}\n// changed canonical source\n`);
        const before = readFileSync(record, "utf8");
        const stale = run("--check");
        expect(stale.code).toBe(1);
        expect(stale.output).toContain("stale canonical inputs");
        expect(readFileSync(record, "utf8")).toBe(before);
        expect(readFileSync(source, "utf8")).toContain("changed canonical source");
        expect(readFileSync(maintained, "utf8")).toBe(original);
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});
