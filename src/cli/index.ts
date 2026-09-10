import { spawnSync } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { delimiter, dirname, join, resolve } from "node:path";
import { buildProject } from "./build";
import { startDev } from "./dev";
import { runProject } from "./run";

const usage = `
  shallot — run and build a shallot project

  Usage
    shallot <command> [dir] [options]

  Commands
    create    Start a new project: bun create shallot <name>
    dev       Run the project standalone, with hot reload
    build     Build for distribution
    run       Build and run
    add       Copy an example recipe out of the package (bare: list them)
    check     Not yet available in this version

  Options
    --target <platform>   web (default), windows, mac, linux. Native release builds download a prebuilt
                          shell from GitHub Releases when available (no Rust toolchain needed); any miss
                          (404, offline, checksum mismatch, source checkout) silently falls back to
                          compiling the Rust native host from source, which requires the Rust toolchain
                          (+ per-target prerequisites; portable auto-downloads CEF on first build, or
                          set CEF_PATH). Debug builds always compile from source.
    --release             Optimized build (build, run)
    --portable            Bundle the Chromium runtime (CEF) instead of the system webview.
                          Larger, but self-contained and runs anywhere. Required on Linux
                          (WebKitGTK has no usable WebGPU) and for apps needing subgroups on macOS.
    --port <n>            Server port (dev, run)
    --strict-port         Fail if the port is in use instead of picking another
    --no-open             Don't open a browser tab (dev) — for a driver that brings its own
    -h, --help            Show this help

  Examples
    shallot dev                  Run with hot reload
    shallot build --target mac   Build a macOS app (system WKWebView)
    shallot build --target linux --portable   Build a self-contained Linux app
    shallot add first-person     Copy the first-person recipe into ./first-person

  Other verbs resolve to shallot-<verb> on your PATH.
`;

export type CliArgs =
    | { kind: "add"; rest: string[] }
    | { kind: "create" }
    | { kind: "check" }
    | { kind: "external"; verb: string; rest: string[] }
    | { kind: "usage"; exitCode: 0 | 1 }
    | {
          kind: "run";
          subcmd: "dev" | "build" | "run";
          dir: string;
          target?: string;
          release: boolean;
          portable: boolean;
          port?: number;
          strictPort: boolean;
          open: boolean;
      };

const PROJECT_VERBS = ["dev", "build", "run"];

/**
 * parse `shallot`'s top-level flags and pick which subcommand handles them. `add` owns its own flag
 * set and an unknown verb belongs to its external command, so both route before the shared
 * dev/build/run parse. Throws on an unrecognized `-`-prefixed option.
 */
export function parseCliArgs(raw: string[]): CliArgs {
    const verb = raw[0];
    if (verb === "add") return { kind: "add", rest: raw.slice(1) };
    if (verb === "create") return { kind: "create" };
    if (verb === "check") return { kind: "check" };
    if (verb && !verb.startsWith("-") && !PROJECT_VERBS.includes(verb))
        return { kind: "external", verb, rest: raw.slice(1) };

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

    if (help) return { kind: "usage", exitCode: 0 };

    const subcmd = positionalArgs[0];
    // bare `shallot [flags]` names no command — print usage rather than guess one.
    if (subcmd == null) return { kind: "usage", exitCode: 0 };

    return {
        kind: "run",
        subcmd: subcmd as "dev" | "build" | "run",
        dir: positionalArgs[1] || ".",
        target,
        release,
        portable,
        port,
        strictPort,
        open,
    };
}

function executable(path: string): boolean {
    try {
        return statSync(path).isFile();
    } catch {
        return false;
    }
}

/**
 * the executable an external verb resolves to, on the Cargo/git model: `shallot-<verb>` on `PATH`, then
 * a package bin of that name in the nearest `node_modules/.bin` from `cwd` upward. Null when neither
 * exists or the verb is not a plain command name.
 */
export function resolveExternal(
    verb: string,
    env: Record<string, string | undefined> = process.env,
    cwd = process.cwd(),
): string | null {
    if (!/^[a-z0-9][a-z0-9-]*$/i.test(verb)) return null;
    const name = `shallot-${verb}`;
    const exts = process.platform === "win32" ? (env.PATHEXT ?? ".EXE;.CMD;.BAT").split(";") : [""];
    const candidates = (dir: string) => exts.map((ext) => join(dir, name + ext));
    for (const dir of (env.PATH ?? "").split(delimiter).filter(Boolean)) {
        const hit = candidates(dir).find(executable);
        if (hit) return hit;
    }
    for (let dir = resolve(cwd); ; dir = dirname(dir)) {
        const bin = join(dir, "node_modules", ".bin");
        if (existsSync(bin)) {
            const hit = candidates(bin).find(executable);
            if (hit) return hit;
        }
        if (dirname(dir) === dir) return null;
    }
}

/** run the `shallot` CLI over argv (without the runtime and script) and exit with its status. */
export async function main(raw: string[]): Promise<void> {
    let parsed: CliArgs;
    try {
        parsed = parseCliArgs(raw);
    } catch (e) {
        console.error(e instanceof Error ? e.message : String(e));
        process.exit(1);
    }

    if (parsed.kind === "usage") {
        console.log(usage);
        process.exit(parsed.exitCode);
    }
    if (parsed.kind === "create") {
        console.error("Create a project with: bun create shallot <name>");
        process.exit(2);
    }
    if (parsed.kind === "check") {
        console.error("shallot check is not yet available in this version");
        process.exit(2);
    }
    if (parsed.kind === "add") {
        const { runAdd } = await import("./add");
        process.exit(await runAdd(parsed.rest));
    }
    if (parsed.kind === "external") {
        const bin = resolveExternal(parsed.verb);
        if (!bin) {
            console.error(`unknown command: ${parsed.verb}`);
            console.log(usage);
            process.exit(1);
        }
        const child = spawnSync(bin, parsed.rest, { stdio: "inherit" });
        if (child.error) throw child.error;
        process.exit(child.status ?? 1);
    }

    const projectDir = resolve(parsed.dir);
    if (parsed.subcmd === "dev") {
        // native webviews can't HMR — `dev --target <native>` is a debug build + run (run without --release)
        if (parsed.target && parsed.target !== "web") {
            await runProject(projectDir, {
                target: parsed.target,
                port: parsed.port,
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
    } else if (parsed.subcmd === "run") {
        await runProject(projectDir, {
            target: parsed.target,
            port: parsed.port,
            release: parsed.release,
            portable: parsed.portable,
        });
    }
}
