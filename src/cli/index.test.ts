import { expect, test } from "bun:test";
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

test("bare shallot presents the project map, loop, commands, targets and help contract", async () => {
    const result = await cli();
    expect(result.code).toBe(0);
    expect(result.stderr).toBe("");
    expect(result.stdout.trim()).toBe(
        `
  shallot — tools for a Shallot project

  Shallot is a library: Vite runs your project and Bun tests it.

  Your loop
    shallot dev               develop with hot reload (vite; --target for desktop)
    shallot build             build for distribution (vite build; --target for desktop)
    shallot preview           run the last build (vite preview; --target for desktop)
    bun test                  run the cheap tests
    bun test ./<file>         run a named Bun tier (GPU, Cargo, Node, oracle)
    playwright test           run the browser tests

  Commands
    add [example]             copy an example into your project; with no name, list them

  Targets
    --target <platform>       web (default), windows, mac, linux
    --release                 optimized native build
    --portable                bundle the Chromium runtime (CEF)

  Help
    shallot <command> --help`.trim(),
    );
});

test("the current shallot command map omits unimplemented create and skill commands", async () => {
    expect(parseCliArgs(["create", "my-game"])).toEqual({ kind: "unknown", verb: "create" });
    const result = await cli();
    expect(result.stdout).not.toContain("create");
    expect(result.stdout).not.toContain("--skill");
});

test("shallot dev --help presents dev options without build-only options", async () => {
    const result = await cli("dev", "--help");
    expect(result.code).toBe(0);
    expect(result.stdout.trim()).toBe(
        `
  shallot dev [dir] [Vite options]

  Develop with hot reload. A native target opens the Vite dev URL in a debug desktop shell.
  The directory defaults to the current directory (.). Other arguments go to Vite unchanged.

  Common examples
    shallot dev
    shallot dev --port 4000
    shallot dev --target mac --portable

  Shallot options
    --target <platform>   web (default), windows, mac, linux
    --portable            Bundle the Chromium runtime (CEF)
    -h, --help            Show this help`.trim(),
    );
    expect(result.stderr).toBe("");
});

test("shallot build --help presents build options without dev-only options", async () => {
    const result = await cli("build", "--help");
    expect(result.code).toBe(0);
    expect(result.stdout).toContain("--release");
    expect(result.stdout).toContain("Native requirements");
    expect(result.stdout).toContain("GitHub Releases");
    expect(result.stdout).toContain("source checkout");
    expect(result.stdout).toContain("Rust toolchain");
    expect(result.stdout).toContain("cargo-xwin");
    expect(result.stdout).toContain("CEF_PATH");
    expect(result.stderr).toBe("");
});

test("shallot preview --help presents launch options without rebuilding or dev-only options", async () => {
    const result = await cli("preview", "--help");
    expect(result.code).toBe(0);
    expect(result.stdout).toContain("Run the project's Vite preview server.");
    expect(result.stdout).toContain("For web, remaining arguments go to Vite unchanged.");
    expect(result.stdout).toContain("Native targets use the requirements");
    expect(result.stdout).toContain("shallot build --help");
    expect(result.stderr).toBe("");
});

test("shallot add --help presents add usage without writing a destination", async () => {
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
});

test("shallot run is no longer accepted and points users to the command list", async () => {
    const result = await cli("run");
    expect(result.code).toBe(1);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("unknown command: run");
    expect(result.stderr).toContain("See `shallot --help` for available commands.");
});

test("Vite-specific flags are passed through rather than interpreted as Shallot flags", () => {
    expect(parseCliArgs(["dev", "--host", "localhost", "--mode", "development"])).toEqual({
        kind: "run",
        subcmd: "dev",
        dir: ".",
        target: undefined,
        release: false,
        portable: false,
        viteArgs: ["--host", "localhost", "--mode", "development"],
    });
});

test("an unknown build target reports its invocation error", async () => {
    const result = await cli("build", "--target", "not-a-platform");
    expect(result.code).toBe(1);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("unknown target: not-a-platform");
    expect(result.stderr).toContain("See `shallot build --help`");
});
