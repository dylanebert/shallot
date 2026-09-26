import { expect } from "bun:test";
import { resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { main, parseCliArgs } from "./index";

type Result = { code: number; stdout: string; stderr: string };

class ExitStatus extends Error {
    constructor(readonly code: number) {
        super(`exit ${code}`);
    }
}

async function cli(...args: string[]): Promise<Result> {
    const stdout: string[] = [];
    const stderr: string[] = [];
    const log = console.log;
    const error = console.error;
    console.log = (...values: unknown[]) => stdout.push(values.join(" "));
    console.error = (...values: unknown[]) => stderr.push(values.join(" "));
    let code = 0;
    try {
        await main(args, (status) => {
            throw new ExitStatus(status);
        });
    } catch (caught) {
        if (caught instanceof ExitStatus) code = caught.code;
        else throw caught;
    } finally {
        console.log = log;
        console.error = error;
    }
    return { code, stdout: stdout.join("\n"), stderr: stderr.join("\n") };
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
    async () => {
        const result = await cli();
        expect(result.code).toBe(0);
        expect(result.stderr).toBe("");
        const help = result.stdout;
        expect(position(help, "shallot — build and preview a Shallot project")).toBeLessThan(
            position(help, "Usage"),
        );
        expect(position(help, "Usage")).toBeLessThan(position(help, "Commands"));
        expect(position(help, "Commands")).toBeLessThan(position(help, "Common examples"));
        expect(help).toContain("shallot dev                  Run with hot reload");
        expect(help).toContain("preview   Launch an existing build");
        expect(help).toContain(
            "shallot add first-person     Copy the first-person recipe into ./first-person",
        );
        const commands = help
            .split("  Commands\n")[1]
            ?.split("\n\n")[0]
            .split("\n")
            .map((line) => line.trim().split(/\s+/)[0]);
        expect(commands).toEqual(["dev", "build", "preview", "add", "test"]);
        expect(help).toContain("shallot <command> --help");
        expect(help).toContain("shallot build && shallot preview");
        expect(help).not.toContain("Native release builds download");
        expect(help).not.toContain("--target <platform>");
        expect(help).not.toContain("shallot build --target");
    },
);

check(
    "creation remains discoverable through Bun",
    {
        claim: "shallot help points to bun create without accepting create as a Shallot command",
    },
    async () => {
        expect(parseCliArgs(["create", "my-game"])).toEqual({ kind: "unknown", verb: "create" });
        const result = await cli();
        expect(result.stdout).toContain("bun create shallot <name>");
        expect(result.stdout).not.toContain("shallot create");
    },
);

check(
    "dev help is focused and successful",
    { claim: "shallot dev --help presents dev options without build-only options" },
    async () => {
        const result = await cli("dev", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("Vite HMR");
        expect(result.stdout).toContain("debug app without HMR");
        expect(result.stdout).toContain("windows, mac, linux");
        expect(result.stdout).toContain("--portable");
        expect(result.stdout).toContain("--strict-port");
        expect(result.stdout).toContain("(web only)");
        expect(result.stdout).toContain("shallot build --help");
        expect(result.stdout).not.toContain("--release");
        expect(result.stderr).toBe("");
    },
);

check(
    "build help is focused and successful",
    { claim: "shallot build --help presents build options without dev-only options" },
    async () => {
        const result = await cli("build", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("--release");
        expect(result.stdout).toContain("Native requirements");
        expect(result.stdout).toContain("GitHub Releases");
        expect(result.stdout).toContain("source checkout");
        expect(result.stdout).toContain("Rust toolchain");
        expect(result.stdout).toContain("cargo-xwin");
        expect(result.stdout).toContain("CEF_PATH");
        expect(result.stdout).not.toContain("--port <n>");
        expect(result.stderr).toBe("");
    },
);

check(
    "preview help is focused and successful",
    {
        claim: "shallot preview --help presents launch options without rebuilding or dev-only options",
    },
    async () => {
        const result = await cli("preview", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("Launch an existing build without rebuilding it.");
        expect(result.stdout).toContain("--port <n>");
        expect(result.stdout).toContain("(web only)");
        expect(result.stdout).toContain("Native targets use the requirements");
        expect(result.stdout).toContain("shallot build --help");
        expect(result.stdout).not.toContain("shallot run");
        expect(result.stdout).toContain("--no-open");
        expect(result.stdout).not.toContain("--strict-port");
        expect(result.stderr).toBe("");
    },
);

check(
    "the installed bin handles test help before starting the runner",
    {
        claim: "shallot test --help prints its command help instead of starting the test runner",
        size: "integration",
        subject: "bin/shallot.ts",
    },
    () => {
        const root = resolve(import.meta.dir, "../..");
        const result = Bun.spawnSync(
            [process.execPath, resolve(root, "bin/shallot.ts"), "test", "--help"],
            { cwd: root, stdout: "pipe", stderr: "pipe" },
        );
        expect(result.exitCode).toBe(0);
        expect(result.stdout.toString()).toContain("shallot test [options]");
        expect(result.stdout.toString()).toContain("--oracle <claim>");
        expect(result.stdout.toString()).not.toContain("checks (parsed");
        expect(result.stderr.toString()).toBe("");
    },
);

check(
    "test help describes the accepted selectors",
    {
        claim: "shallot test --help lists the runner's population, integration, selector, and oracle options",
    },
    async () => {
        const result = await cli("test", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("shallot test [options]");
        expect(result.stdout).toContain("--list");
        expect(result.stdout).toContain("--integration");
        expect(result.stdout).toContain("--base <ref>");
        expect(result.stdout).toContain("--diff <ref>");
        expect(result.stdout).toContain("--all");
        expect(result.stdout).toContain("--requires <tag>");
        expect(result.stdout).toContain("--subject <prefix>");
        expect(result.stdout).toContain("--oracle <claim>");
        expect(result.stderr).toBe("");
    },
);

check(
    "add help remains focused and successful",
    { claim: "shallot add --help presents add usage without writing a destination" },
    async () => {
        const result = await cli("add", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("shallot add [name] [dir]");
        expect(result.stdout).toContain("destination defaults to the recipe name");
        expect(result.stdout).toContain("Common examples");
        expect(result.stdout).toContain("Options");
        expect(result.stderr).toBe("");
    },
);

check(
    "the retired run command refuses",
    { claim: "shallot run is no longer accepted and points users to the command list" },
    async () => {
        const result = await cli("run");
        expect(result.code).toBe(1);
        expect(result.stdout).toBe("");
        expect(result.stderr).toContain("unknown command: run");
        expect(result.stderr).toContain("See `shallot --help` for available commands.");
    },
);

check(
    "invalid option is diagnosed without running a project",
    { claim: "an unknown option fails before project execution" },
    async () => {
        const result = await cli("dev", "--not-an-option");
        expect(result.code).toBe(1);
        expect(result.stdout).toBe("");
        expect(result.stderr).toContain("unknown option: --not-an-option");
        expect(result.stderr).toContain("See `shallot dev --help`");
    },
);

check(
    "invalid target is diagnosed without running a project",
    { claim: "an unknown build target reports its invocation error" },
    async () => {
        const result = await cli("build", "--target", "not-a-platform");
        expect(result.code).toBe(1);
        expect(result.stdout).toBe("");
        expect(result.stderr).toContain("unknown target: not-a-platform");
        expect(result.stderr).toContain("See `shallot build --help`");
    },
);
