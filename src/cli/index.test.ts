import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { expectError } from "../../scripts/test-console";
import { main, parseCliArgs } from "./index";

type Result = { code: number; stdout: string; stderr: string };

class ExitStatus extends Error {
    readonly code: number;

    constructor(code: number) {
        super(`exit ${code}`);
        this.code = code;
    }
}

async function cli(...args: string[]): Promise<Result> {
    const stdout: string[] = [];
    const stderr: string[] = [];
    const log = console.log;
    const error = console.error;
    console.log = (...values: unknown[]) => stdout.push(values.join(" "));
    console.error = (...values: unknown[]) => {
        stderr.push(values.join(" "));
        error(...values);
    };
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
    bun run dev                  run the Vite project
    bun test                     run cheap and GPU tests
    bun test ./<file>            run a Node or oracle file by path
    bunx playwright test         run browser tests

  Commands
    create <name>                create a project from minimal or an example
    migrate                      print the 0.9.5 to 0.10 migration guide
    dev --target <platform>      develop a desktop app
    build --target <platform>    build a desktop app
    preview --target <platform>  launch a desktop build

  Desktop options
    --target <platform>          windows, mac, linux
    --release                    optimized native build
    --portable                   bundle the Chromium runtime (CEF)

  Help
    -V, --version                print the installed version
    --skill                      print the Shallot project skill
    shallot <command> --help
    shallot help [command]`.trim(),
    );
});

test("version flags print the installed package version", async () => {
    const { version } = JSON.parse(
        readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
    );
    for (const flag of ["--version", "-V"]) {
        expect(await cli(flag)).toEqual({ code: 0, stdout: version, stderr: "" });
    }
});

test("version and skill flags refuse extra arguments without calling the flags unknown", async () => {
    for (const flag of ["--version", "-V", "--skill"]) {
        const message = `${flag} takes no arguments`;
        expectError(new RegExp(`^${message}$`));
        expectError(/^See `shallot --help`/);
        expect(await cli(flag, "extra")).toEqual({
            code: 1,
            stdout: "",
            stderr: `${message}\nSee \`shallot --help\` for available options.`,
        });
        expect(await cli(flag, "extra", "--help")).toEqual(await cli("--help"));
    }
});

test("shallot --skill prints the project skill and bare usage names it", async () => {
    const result = await cli("--skill");
    expect(result.code).toBe(0);
    expect(result.stderr).toBe("");
    expect(result.stdout).toBe(
        readFileSync(new URL("./skill.md", import.meta.url), "utf8").trimEnd(),
    );

    const usage = await cli();
    expect(usage.code).toBe(0);
    expect(usage.stdout).toContain("--skill");
});

test("shallot migrate prints the bundled migration guide and refuses arguments", async () => {
    const result = await cli("migrate");
    expect(result.code).toBe(0);
    expect(result.stderr).toBe("");
    expect(result.stdout).toBe(
        readFileSync(new URL("../../MIGRATION.md", import.meta.url), "utf8").trimEnd(),
    );

    const help = await cli("migrate", "--help");
    expect(help.code).toBe(0);
    expect(help.stdout).toContain("It changes no files.");

    expectError(/^migrate takes no arguments$/);
    expectError(/^See `shallot --help`/);
    const refused = await cli("migrate", "src");
    expect(refused.code).toBe(1);
    expect(refused.stdout).toBe("");
});

test("leading flags never become a project command and unknown top-level input fails", async () => {
    expect(parseCliArgs(["--help"])).toEqual({ kind: "usage", exitCode: 0 });
    expect(parseCliArgs(["-h"])).toEqual({ kind: "usage", exitCode: 0 });
    expect(parseCliArgs([])).toEqual({ kind: "usage", exitCode: 0 });
    expect(parseCliArgs(["dev", "--help"])).toEqual({ kind: "command-help", command: "dev" });
    for (const args of [["--foo"], ["--target", "web"], ["--release"], ["--portable", "dev"]]) {
        expect(() => parseCliArgs(args)).toThrow(`unknown option: ${args[0]}`);
        expectError(/^unknown option:/);
        expectError(/^See `shallot --help`/);
        const result = await cli(...args);
        expect(result.code).toBe(1);
        expect(result.stdout).toBe("");
    }

    for (const args of [
        ["--help"],
        ["-h"],
        ["--target", "--help"],
        ["--skill", "--help"],
        ["--foo", "-h"],
        ["typo", "--help"],
    ]) {
        const result = await cli(...args);
        expect(result.code).toBe(0);
        expect(result.stdout).toContain("shallot — tools for a Shallot project");
        expect(result.stderr).toBe("");
    }
});

test("shallot dev --help presents dev options without build-only options", async () => {
    const result = await cli("dev", "--help");
    expect(result.code).toBe(0);
    expect(result.stdout.trim()).toBe(
        `
  shallot dev --target <platform> [dir] [Vite options]

  Develop with hot reload in a native desktop shell. Web projects run \`bun run dev\`.
  The directory defaults to the current directory (.). Other arguments go to Vite unchanged.

  Common examples
    bun run dev
    shallot dev --target mac
    shallot dev --target mac --portable

  Shallot options
    --target <platform>   windows, mac, linux
    --portable            Bundle the Chromium runtime (CEF)
    -h, --help            Show this help`.trim(),
    );
    expect(result.stderr).toBe("");
});

test("shallot build --help presents build options without dev-only options", async () => {
    const result = await cli("build", "--help");
    expect(result.code).toBe(0);
    expect(result.stdout).toContain("shallot build --target <platform> [dir] [Vite options]");
    expect(result.stdout).toContain("--target <platform>   windows, mac, linux");
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
    expect(result.stdout).toContain("shallot preview --target <platform> [dir]");
    expect(result.stdout).toContain("--target <platform>   windows, mac, linux");
    expect(result.stdout).toContain("Launch an existing native desktop build.");
    expect(result.stdout).toContain("Web projects run `bun run preview`.");
    expect(result.stdout).toContain("Native targets use the requirements");
    expect(result.stdout).toContain("shallot build --help");
    expect(result.stderr).toBe("");
});

test("web commands refuse to shadow project scripts and name the desktop target path", async () => {
    const commands = [
        [
            "dev",
            "For web, run `bun run dev`; for desktop, rerun with `--target mac|windows|linux`.",
        ],
        [
            "build",
            "For web, run `bun run build`; for desktop, rerun with `--target mac|windows|linux`.",
        ],
        [
            "preview",
            "For web, run `bun run preview`; for desktop, rerun with `--target mac|windows|linux`.",
        ],
    ] as const;
    const expectedErrors = [
        /For web, run `bun run dev`;/,
        /For web, run `bun run build`;/,
        /For web, run `bun run preview`;/,
    ];
    for (const [index, [command, refusal]] of commands.entries()) {
        expectError(expectedErrors[index]!);
        const result = await cli(command);
        expect(result.code).toBe(1);
        expect(result.stdout).toBe("");
        expect(result.stderr).toBe(refusal);
    }

    expectError(/For web, run `bun run dev`;/);
    const webTarget = await cli("dev", "--target", "web");
    expect(webTarget.code).toBe(1);
    expect(webTarget.stdout).toBe("");
    expect(webTarget.stderr).toBe(commands[0][1]);
});

test("shallot create --help presents the outer-folder project command and templates", async () => {
    const result = await cli("create", "--help");
    expect(result.code).toBe(0);
    expect(result.stdout.trim()).toStartWith(
        `
  shallot create <name> [--template <example>]

  Create a project in the outer folder. The default template is minimal.

  Outer folder, without Shallot installed
    bun create shallot <name> [--template <example>]

  Inside a project where Shallot is installed
    bunx shallot create <name> [--template <example>]

  Options
    --template <example>  Copy an installed example
    -h, --help            Show this help`.trim(),
    );
    expect(result.stdout).toContain("\n  Templates\n    city —");
    expect(result.stdout).toContain("loading-screen — A game's own title card");
    expect(result.stderr).toBe("");
});

test("help aliases top-level and every command help", async () => {
    expect(await cli("help")).toEqual(await cli("--help"));
    for (const command of ["dev", "build", "preview", "create", "migrate"]) {
        expect(await cli("help", command)).toEqual(await cli(command, "--help"));
    }
    expectError(/^unknown command: missing$/);
    expectError(/^See `shallot --help`/);
    const result = await cli("help", "missing");
    expect(result.code).toBe(1);
    expect(result.stderr).toContain("unknown command: missing");
});

test("shallot run is no longer accepted and points users to the command list", async () => {
    expectError(/unknown command: run/);
    expectError(/See `shallot --help` for available commands\./);
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

test("dev refuses release and preview refuses arguments it cannot forward", async () => {
    for (const [command, rest, message] of [
        ["dev", ["--release"], "--release is not supported by dev"],
        ["preview", [".", "--foo", "extra"], "unexpected argument: --foo"],
        ["preview", ["--foo", "extra"], "unexpected argument: --foo"],
        ["preview", [".", "extra"], "unexpected argument: extra"],
    ] as const) {
        expect(() => parseCliArgs([command, "--target", "mac", ...rest])).toThrow(message);
        expectError(new RegExp(`^${message}$`));
        expectError(new RegExp("^See `shallot " + command + " --help`"));
        const result = await cli(command, "--target", "mac", ...rest);
        expect(result.code).toBe(1);
        expect(result.stdout).toBe("");
        expect(result.stderr).toContain(message);
    }
    for (const command of ["dev", "build"]) {
        const parsed = parseCliArgs([command, "--target", "mac", ".", "--foo", "extra"]);
        expect(parsed.kind === "run" && parsed.viteArgs).toEqual(["--foo", "extra"]);
    }
});

test("native commands name a missing project directory before looking for a build", async () => {
    const path = `/shallot-missing-project-${process.pid}`;
    for (const command of ["preview", "dev", "build"]) {
        expectError(new RegExp(`^no project directory at ${path}$`));
        const result = await cli(command, "--target", "mac", path);
        expect(result.code).toBe(1);
        expect(result.stderr).toBe(`no project directory at ${path}`);
    }
});

test("an unknown build target reports its invocation error", async () => {
    expectError(/unknown target: not-a-platform/);
    expectError(/See `shallot build --help`/);
    const result = await cli("build", "--target", "not-a-platform");
    expect(result.code).toBe(1);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("unknown target: not-a-platform");
    expect(result.stderr).toContain("See `shallot build --help`");
});
