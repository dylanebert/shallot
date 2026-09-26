import { resolve } from "node:path";
import { buildProject } from "./build";
import { startDev } from "./dev";
import { MissingBuildError, previewProject } from "./preview";

const usage = `
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
    -h, --help                  Show this help

`;

const commandUsage = {
    dev: `
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
    -h, --help            Show this help
`,
    build: `
  shallot build [dir] [options]

  Build a project for distribution.
  The directory defaults to the current directory (.).

  Common examples
    shallot build
    shallot build --target mac
    shallot build --target linux --portable

  Options
    --target <platform>   web (default), windows, mac, linux
    --release             Optimized build
    --portable            Bundle the Chromium runtime (CEF) instead of the system webview.
                          Larger, but self-contained and runs anywhere. Required on Linux
                          (WebKitGTK has no usable WebGPU) and for apps needing subgroups on macOS.
    -h, --help            Show this help

  Native requirements
    Native release builds use a prebuilt shell from GitHub Releases when available. A miss (404,
    offline, checksum mismatch, or source checkout) falls back to compiling the Rust native host from
    source. Debug builds always compile from source. Source builds need the Rust toolchain
    (https://rustup.rs) and, per target:
      mac       Xcode command line tools
      linux     --portable only (WebKitGTK has no usable WebGPU), with libx11-dev
      windows   cargo-xwin to cross-compile; --portable needs a Windows host with Visual Studio,
                its C++ workload and ATL
    Portable builds download CEF on first build, or use CEF_PATH when set.
`,
    preview: `
  shallot preview [dir] [options]

  Launch an existing build without rebuilding it.
  The directory defaults to the current directory (.). Run 'shallot build' first.

  Common examples
    shallot preview
    shallot preview --target mac
    shallot preview --target linux --portable

  Options
    --target <platform>   web (default), windows, mac, linux
    --release             Launch the optimized build
    --portable            Launch the build with its bundled Chromium runtime (CEF)
    --port <n>            Preview server port (web only)
    --no-open             Don't open a browser tab (web only)
    -h, --help            Show this help

  Native targets use the requirements documented by 'shallot build --help'.
`,
    test: `
  shallot test [options]

  Run the project's checks. Unit checks run by default; integration checks run only when selected.

  Common examples
    shallot test
    shallot test --list
    shallot test --integration --base origin/main --diff HEAD

  Options
    --list                  List the selected checks without running them
    --integration           Select integration checks; requires --base and --diff unless a selector is used
    --base <ref>            Base commit for changed-subject integration checks (with --diff)
    --diff <ref>            Diff commit for changed-subject integration checks (with --base)
    --all                   Select all integration checks (with --integration)
    --requires <tag>        Select integration checks by requirement (with --integration)
    --subject <prefix>      Select integration checks by subject path (with --integration)
    --oracle <claim>        Select one named oracle
    -h, --help              Show this help
`,
} as const;

export type CliArgs =
    | { kind: "add"; rest: string[] }
    | { kind: "command-help"; command: keyof typeof commandUsage }
    | { kind: "unknown"; verb: string }
    | { kind: "usage"; exitCode: 0 | 1 }
    | {
          kind: "run";
          subcmd: "dev" | "build" | "preview";
          dir: string;
          target?: string;
          release: boolean;
          portable: boolean;
          port?: number;
          strictPort: boolean;
          open: boolean;
      };

const PROJECT_VERBS = ["dev", "build", "preview"];
const TARGETS = ["web", "windows", "mac", "linux"];

/**
 * parse `shallot`'s top-level flags and pick which subcommand handles them. `add` owns its own flag
 * set, and unknown verbs refuse before the shared dev/build/preview parse. Throws on an unrecognized
 * `-`-prefixed option.
 */
export function parseCliArgs(raw: string[]): CliArgs {
    const verb = raw[0];
    if (verb === "add") return { kind: "add", rest: raw.slice(1) };
    if (verb === "test" && raw.slice(1).some((arg) => arg === "--help" || arg === "-h"))
        return { kind: "command-help", command: "test" };
    if (verb && !verb.startsWith("-") && !PROJECT_VERBS.includes(verb))
        return { kind: "unknown", verb };

    const positionalArgs: string[] = [];
    let target: string | undefined;
    let release = false;
    let portable = false;
    let port: number | undefined;
    let strictPort = false;
    let open = true;
    let help = false;

    const num = (flag: string, v: string): number => {
        if (v.trim() === "") {
            throw new Error(`invalid ${flag} value "${v}" — must not be empty`);
        }
        const n = Number(v);
        if (!Number.isFinite(n) || n <= 0) {
            throw new Error(`invalid ${flag} value "${v}" — expected a positive number`);
        }
        if (!Number.isInteger(n)) {
            throw new Error(`invalid ${flag} value "${v}" — expected an integer`);
        }
        return n;
    };

    for (let i = 0; i < raw.length; i++) {
        if (raw[i] === "--target" && raw[i + 1]) {
            target = raw[i + 1];
            i++;
        } else if (raw[i] === "--release") {
            release = true;
        } else if (raw[i] === "--portable") {
            portable = true;
        } else if (raw[i] === "--port" && raw[i + 1]) {
            port = num("--port", raw[i + 1]);
            i++;
        } else if (raw[i]?.startsWith("--port=")) {
            port = num("--port", raw[i].split("=")[1]);
        } else if (raw[i] === "--strict-port") {
            strictPort = true;
        } else if (raw[i] === "--no-open") {
            open = false;
        } else if (raw[i] === "--help" || raw[i] === "-h") {
            help = true;
        } else if (raw[i].startsWith("-")) {
            throw new Error(`unknown option: ${raw[i]}`);
        } else {
            positionalArgs.push(raw[i]);
        }
    }

    const subcmd = positionalArgs[0];
    if (help && subcmd !== undefined && Object.hasOwn(commandUsage, subcmd))
        return { kind: "command-help", command: subcmd as keyof typeof commandUsage };
    if (help) return { kind: "usage", exitCode: 0 };

    // bare `shallot [flags]` names no command — print usage rather than guess one.
    if (subcmd == null) return { kind: "usage", exitCode: 0 };

    return {
        kind: "run",
        subcmd: subcmd as "dev" | "build" | "preview",
        dir: positionalArgs[1] || ".",
        target,
        release,
        portable,
        port,
        strictPort,
        open,
    };
}

function helpHint(raw: string[]): string {
    const command = raw[0] && PROJECT_VERBS.includes(raw[0]) ? `shallot ${raw[0]}` : "shallot";
    return `See \`${command} --help\` for available options.`;
}

/** run the `shallot` CLI over argv (without the runtime and script) and exit with its status. */
export async function main(
    raw: string[],
    exit: (code: number) => never = process.exit,
): Promise<void> {
    let parsed: CliArgs;
    try {
        parsed = parseCliArgs(raw);
    } catch (e) {
        console.error(e instanceof Error ? e.message : String(e));
        console.error(helpHint(raw));
        exit(1);
    }

    if (parsed.kind === "usage") {
        console.log(usage);
        exit(parsed.exitCode);
    }
    if (parsed.kind === "command-help") {
        console.log(commandUsage[parsed.command]);
        exit(0);
    }
    if (parsed.kind === "add") {
        const { runAdd } = await import("./add");
        exit(await runAdd(parsed.rest));
    }
    if (parsed.kind === "unknown") {
        console.error(`unknown command: ${parsed.verb}`);
        console.error("See `shallot --help` for available commands.");
        exit(1);
    }

    if (parsed.target && !TARGETS.includes(parsed.target)) {
        console.error(`unknown target: ${parsed.target}`);
        console.error(`See \`shallot ${parsed.subcmd} --help\` for available targets and options.`);
        exit(1);
    }

    const projectDir = resolve(parsed.dir);
    if (parsed.subcmd === "dev") {
        // native webviews can't HMR — `dev --target <native>` is a debug build + run (run without --release)
        if (parsed.target && parsed.target !== "web") {
            await buildProject(projectDir, {
                target: parsed.target,
                release: false,
                portable: parsed.portable,
            });
            await previewProject(projectDir, {
                target: parsed.target,
                release: false,
                portable: parsed.portable,
            });
        } else {
            await startDev(projectDir, {
                port: parsed.port,
                strictPort: parsed.strictPort,
                open: parsed.open,
            });
        }
    } else if (parsed.subcmd === "build") {
        await buildProject(projectDir, {
            target: parsed.target,
            release: parsed.release,
            portable: parsed.portable,
        });
    } else if (parsed.subcmd === "preview") {
        try {
            await previewProject(projectDir, {
                target: parsed.target,
                port: parsed.port,
                release: parsed.release,
                portable: parsed.portable,
                open: parsed.open,
            });
        } catch (error) {
            if (!(error instanceof MissingBuildError)) throw error;
            console.error(error.message);
            exit(1);
        }
    }
}
