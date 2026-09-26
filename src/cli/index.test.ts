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

check(
    "top-level help leads with common use",
    {
        claim: "shallot top-level help presents purpose, usage, commands, and common examples without operation-specific options",
    },
    async () => {
        const result = await cli();
        expect(result.code).toBe(0);
        expect(result.stderr).toBe("");
        expect(result.stdout.trim()).toBe(
            `
  shallot — develop, build and test a Shallot project

  Usage
    shallot <command> [dir] [options]

  Commands
    dev       Run the project with hot reload
    build     Build for distribution
    preview   Run the last build without rebuilding
    add       Copy an example into your project; with no name, list them
    test      Run the project's checks; --list shows which would run

  Common examples
    bun create shallot <name>          Create a project
    shallot dev                        Run with hot reload
    shallot test                       Run the checks
    shallot add first-person           Copy the first-person example into ./first-person
    shallot build && shallot preview   Build, then run the build

  Help
    shallot <command> --help    Show options and examples for one command
    -h, --help                  Show this help`.trim(),
        );
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
        expect(result.stdout.trim()).toBe(
            `
  shallot dev [dir] [options]

  Run a project with hot reload. A native target builds and runs a debug app instead, without hot reload.
  The directory defaults to the current directory (.).

  Common examples
    shallot dev
    shallot dev --no-open
    shallot dev --target mac --portable

  Options
    --target <platform>   web (default), windows, mac, linux
    --portable            Bundle the Chromium runtime (CEF); see 'shallot build --help'
    --port <n>            Web server port (web only)
    --strict-port         Fail if the web port is in use instead of picking another (web only)
    --no-open             Don't open a browser tab (web only)
    -h, --help            Show this help`.trim(),
        );
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
        expect(result.stdout.trim()).toBe(
            `
  shallot test [options]

  Run the project's checks. Unit checks run by default; integration checks run only when selected.

  Common examples
    shallot test
    shallot test --list
    shallot test --integration --base origin/main --diff HEAD
    shallot test --integration --base origin/main --diff HEAD --requires '!gpu' --requires '!browser' --no-unit-fallback

  Options
    --list                  List the selected checks without running them
    --integration           Select integration checks; requires --base and --diff unless a selector is used
    --no-unit-fallback      Do not run the unit sweep when no integration checks are selected
    --base <ref>            Base commit for changed-subject integration checks (with --diff)
    --diff <ref>            Diff commit for changed-subject integration checks (with --base)
    --all                   Select all integration checks (with --integration)
    --requires <tag|!tag>   Filter by a requirement; repeat to combine filters
    --subject <prefix>      Select integration checks by subject path (with --integration)
    --oracle <claim>        Select one named oracle
    -h, --help              Show this help`.trim(),
        );
        expect(result.stderr).toBe("");
    },
);

check(
    "add help remains focused and successful",
    { claim: "shallot add --help presents add usage without writing a destination" },
    async () => {
        const result = await cli("add", "--help");
        expect(result.code).toBe(0);
        expect(result.stdout.trim()).toBe(
            `
  shallot add [name] [dir]

  Without a name, lists available examples.
  With a name, copies one example into a project.
  The destination defaults to the example name relative to the current directory.
  An occupied destination is refused.

  Common examples
    shallot add
    shallot add first-person
    shallot add first-person my-game

  Options
    -h, --help  Show this help`.trim(),
        );
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
