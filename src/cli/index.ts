import { resolve } from "node:path";
import { resolveViteCli, startViteDev } from "../project";
import { buildProject, type NativeTarget } from "./build";
import { previewProject } from "./preview";

const usage = `
  shallot — tools for a Shallot project

  Shallot is a library: Vite runs your project and Bun tests it.

  Your loop
    bun run dev               run the Vite project
    bun test                  run cheap and GPU tests
    bun test ./<file>         run a Node or oracle file by path
    bunx playwright test      run browser tests
    shallot dev --target <platform> (desktop)

  Commands
    add [example]             copy an example into your project; with no name, list them

  Targets
    --target <platform>       windows, mac, linux
    --release                 optimized native build
    --portable                bundle the Chromium runtime (CEF)

  Help
    shallot <command> --help
`;

const commandUsage = {
    dev: `
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
    -h, --help            Show this help
`,
    build: `
  shallot build --target <platform> [dir] [Vite options]

  Build a native desktop app with the project's Vite config. Web projects run \`bun run build\`.
  The directory defaults to the current directory (.). Other arguments go to Vite unchanged.

  Common examples
    bun run build
    shallot build --target mac
    shallot build --target linux --portable

  Shallot options
    --target <platform>   windows, mac, linux
    --release             Optimized native build
    --portable            Bundle the Chromium runtime (CEF) instead of the system webview.
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
  shallot preview --target <platform> [dir]

  Launch an existing native desktop build. Web projects run \`bun run preview\`.
  The directory defaults to the current directory (.).

  Common examples
    bun run preview
    shallot preview --target mac
    shallot preview --target linux --portable

  Shallot options
    --target <platform>   windows, mac, linux
    --release             Launch the optimized native build
    --portable            Launch the build with its bundled Chromium runtime (CEF)
    -h, --help            Show this help

  Native targets use the requirements documented by 'shallot build --help'.
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
          viteArgs: string[];
      };

const PROJECT_VERBS = ["dev", "build", "preview"] as const;
const TARGETS: readonly NativeTarget[] = ["windows", "mac", "linux"];

function isNativeTarget(target: string): target is NativeTarget {
    return (TARGETS as readonly string[]).includes(target);
}

/** Separate Shallot's target flags; preserve every Vite argument and its order. */
export function parseCliArgs(raw: string[]): CliArgs {
    const verb = raw[0];
    if (verb === "add") return { kind: "add", rest: raw.slice(1) };
    if (verb && !verb.startsWith("-") && !PROJECT_VERBS.some((command) => command === verb))
        return { kind: "unknown", verb };

    const subcmd = PROJECT_VERBS.find((command) => command === verb);
    let target: string | undefined;
    let release = false;
    let portable = false;
    let help = false;
    const viteArgs: string[] = [];

    for (let i = subcmd ? 1 : 0; i < raw.length; i++) {
        const arg = raw[i];
        if (arg === "--target") {
            const value = raw[i + 1];
            if (!value || value.startsWith("-")) throw new Error("--target requires a platform");
            target = value;
            i++;
        } else if (arg.startsWith("--target=")) {
            target = arg.slice("--target=".length);
            if (!target) throw new Error("--target requires a platform");
        } else if (arg === "--release") {
            release = true;
        } else if (arg === "--portable") {
            portable = true;
        } else if (arg === "--help" || arg === "-h") {
            help = true;
        } else {
            viteArgs.push(arg);
        }
    }

    if (help && subcmd) return { kind: "command-help", command: subcmd };
    if (help || !subcmd) return { kind: "usage", exitCode: 0 };

    // A leading positional is Vite's root, expressed as the command's working directory.
    const dir = viteArgs[0] && !viteArgs[0].startsWith("-") ? viteArgs.shift()! : ".";
    return { kind: "run", subcmd, dir, target, release, portable, viteArgs };
}

function helpHint(raw: string[]): string {
    const command =
        raw[0] && PROJECT_VERBS.some((verb) => verb === raw[0]) ? `shallot ${raw[0]}` : "shallot";
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

    const target = parsed.target;
    if (!target || target === "web") {
        console.error(
            `For web, run \`bun run ${parsed.subcmd}\`; for desktop, rerun with \`--target mac|windows|linux\`.`,
        );
        exit(1);
    }
    if (!isNativeTarget(target)) {
        console.error(`unknown target: ${target}`);
        console.error(`See \`shallot ${parsed.subcmd} --help\` for available targets and options.`);
        exit(1);
    }

    const projectDir = resolve(parsed.dir);
    if (parsed.subcmd === "dev") {
        let viteCli: string;
        try {
            viteCli = resolveViteCli(projectDir);
        } catch (error) {
            console.error(error instanceof Error ? error.message : String(error));
            exit(1);
        }
        const status = await buildProject(projectDir, {
            target,
            release: false,
            portable: parsed.portable,
            dev: true,
        });
        if (status !== 0) exit(status);
        const server = await startViteDev(projectDir, parsed.viteArgs, viteCli);
        let appStatus: number;
        try {
            appStatus = await previewProject(projectDir, {
                target,
                portable: parsed.portable,
                devUrl: server.url,
            });
        } finally {
            await server.close();
        }
        exit(appStatus);
    }

    if (parsed.subcmd === "build") {
        exit(
            await buildProject(projectDir, {
                target,
                release: parsed.release,
                portable: parsed.portable,
                args: parsed.viteArgs,
            }),
        );
    }

    try {
        exit(
            await previewProject(projectDir, {
                target,
                release: parsed.release,
                portable: parsed.portable,
            }),
        );
    } catch (error) {
        if (!(error instanceof Error)) throw error;
        console.error(error.message);
        exit(1);
    }
}
