import { expect } from "bun:test";
import { resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { parseCliArgs } from "./index";

const ROOT = resolve(import.meta.dir, "../..");
const BIN = resolve(ROOT, "bin/shallot.ts");

type Result = { code: number; stdout: string; stderr: string };

function cli(...args: string[]): Result {
    const result = Bun.spawnSync([process.execPath, BIN, ...args], {
        cwd: ROOT,
        stdout: "pipe",
        stderr: "pipe",
    });
    return {
        code: result.exitCode ?? -1,
        stdout: result.stdout.toString(),
        stderr: result.stderr.toString(),
    };
}

function position(text: string, section: string): number {
    const index = text.indexOf(section);
    expect(index).toBeGreaterThanOrEqual(0);
    return index;
}

check(
    "top-level help leads with common use",
    {
        claim: "shallot top-level help presents purpose, usage, commands, and common examples without operation-specific options",
    },
    () => {
        const result = cli();
        expect(result.code).toBe(0);
        expect(result.stderr).toBe("");
        const help = result.stdout;
        expect(position(help, "shallot — run and build a shallot project")).toBeLessThan(
            position(help, "Usage"),
        );
        expect(position(help, "Usage")).toBeLessThan(position(help, "Commands"));
        expect(position(help, "Commands")).toBeLessThan(position(help, "Common examples"));
        expect(help).toContain("shallot dev                  Run with hot reload");
        expect(help).toContain(
            "shallot add first-person     Copy the first-person recipe into ./first-person",
        );
        expect(help).toContain("shallot <command> --help");
        expect(help).not.toContain("Native release builds download");
        expect(help).not.toContain("--target <platform>");
    },
);

check(
    "create help is focused and successful",
    { claim: "shallot create --help explains the create redirect without attempting creation" },
    () => {
        const result = cli("create", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("bun create shallot <name>");
        expect(result.stdout).not.toContain("shallot — run and build a shallot project");
        expect(result.stderr).toBe("");
    },
);

check(
    "dev help is focused and successful",
    { claim: "shallot dev --help presents dev options without build-only options" },
    () => {
        const result = cli("dev", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("--strict-port");
        expect(result.stdout).not.toContain("--release");
        expect(result.stderr).toBe("");
    },
);

check(
    "build help is focused and successful",
    { claim: "shallot build --help presents build options without dev-only options" },
    () => {
        const result = cli("build", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("--release");
        expect(result.stdout).not.toContain("--port <n>");
        expect(result.stderr).toBe("");
    },
);

check(
    "run help is focused and successful",
    { claim: "shallot run --help presents run options without strict dev-only options" },
    () => {
        const result = cli("run", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("--port <n>");
        expect(result.stdout).not.toContain("--strict-port");
        expect(result.stderr).toBe("");
    },
);

check(
    "add help remains focused and successful",
    { claim: "shallot add --help presents add usage without writing a destination" },
    () => {
        const result = cli("add", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("shallot add [name] [dir]");
        expect(result.stderr).toBe("");
    },
);

check(
    "external verbs remain delegated",
    { claim: "an unknown verb remains an external shallot command candidate" },
    () => {
        expect(parseCliArgs(["lint", "--help"])).toEqual({
            kind: "external",
            verb: "lint",
            rest: ["--help"],
        });
    },
);

check(
    "invalid option is diagnosed without running a project",
    { claim: "an unknown option fails before project execution" },
    () => {
        const result = cli("dev", "--not-an-option");
        expect(result.code).toBe(1);
        expect(result.stdout).toBe("");
        expect(result.stderr).toContain("unknown option: --not-an-option");
    },
);

check(
    "invalid target is diagnosed without running a project",
    { claim: "an unknown build target reports its invocation error" },
    () => {
        const result = cli("build", "--target", "not-a-platform");
        expect(result.code).toBe(1);
        expect(result.stdout).toBe("");
        expect(result.stderr).toContain("unknown target: not-a-platform");
    },
);
