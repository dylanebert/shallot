#!/usr/bin/env bun
import { resolve } from "node:path";
import { buildProject } from "./build";
import { startDev } from "./dev";
import { runProject } from "./run";

const usage = `
  shallot — run and build a shallot project

  Usage
    shallot <command> [dir] [options]

  Commands
    dev       Run the project standalone, with hot reload
    build     Build for distribution
    run       Build and run
    verify    Boot the project in a headless browser and check it renders (shallot verify --help)
    recipe    Copy an example recipe out of the package (bare: list them)

  Options
    --target <platform>   web (default), windows, mac, linux. Native release builds download a prebuilt
                          shell from GitHub Releases when available (no Rust toolchain needed); any miss
                          (404, offline, checksum mismatch, source checkout) silently falls back to
                          compiling the Rust window host from source, which requires the Rust toolchain
                          (+ per-target prerequisites; portable auto-downloads CEF on first build, or
                          set CEF_PATH). Debug builds always compile from source.
    --release             Optimized build (build, run)
    --portable            Bundle the Chromium runtime (CEF) instead of the system webview.
                          Larger, but self-contained and runs anywhere. Required on Linux
                          (WebKitGTK has no usable WebGPU) and for apps needing subgroups on macOS.
    --port <n>            Server port (dev, run)
    --strict-port         Fail if the port is in use instead of picking another
    -h, --help            Show this help

  Examples
    shallot dev                  Run with hot reload
    shallot build --target mac   Build a macOS app (system WKWebView)
    shallot build --target linux --portable   Build a self-contained Linux app
    shallot recipe first-person  Copy the first-person recipe into ./first-person
`;

export type CliArgs =
    | { kind: "delegate"; cmd: "verify" | "recipe"; rest: string[] }
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
      };

/**
 * parse `shallot`'s top-level flags and pick which subcommand handles them. Pure — the CLI wiring and
 * the tests share it, the shape `parseVerifyArgs` (verify.ts) models. `verify`/`recipe` own their own
 * flag sets, so they're routed before the shared dev/build/run parse rather than teaching that loop
 * every one of their flags. Throws on an unrecognized `-`-prefixed option, same as `parseVerifyArgs`.
 */
export function parseCliArgs(raw: string[]): CliArgs {
    if (raw[0] === "verify") return { kind: "delegate", cmd: "verify", rest: raw.slice(1) };
    if (raw[0] === "recipe") return { kind: "delegate", cmd: "recipe", rest: raw.slice(1) };

    const positionalArgs: string[] = [];
    let target: string | undefined;
    let release = false;
    let portable = false;
    let port: number | undefined;
    let strictPort = false;
    let help = false;

    const num = (flag: string, v: string): number => {
        const n = Number(v);
        if (!Number.isFinite(n) || n <= 0) {
            throw new Error(`invalid ${flag} value "${v}" — expected a positive number`);
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
        } else if (raw[i] === "--help" || raw[i] === "-h") {
            help = true;
        } else if (raw[i].startsWith("-")) {
            throw new Error(`unknown option: ${raw[i]}`);
        } else {
            positionalArgs.push(raw[i]);
        }
    }

    if (help) return { kind: "usage", exitCode: 0 };

    const subcommands = ["dev", "build", "run"];
    const subcmd = positionalArgs[0];
    // bare `shallot [dir]` names no command — print usage rather than guess one.
    if (subcmd == null || !subcommands.includes(subcmd)) {
        return { kind: "usage", exitCode: subcmd == null ? 0 : 1 };
    }

    const dir = positionalArgs[1] || ".";
    return {
        kind: "run",
        subcmd: subcmd as "dev" | "build" | "run",
        dir,
        target,
        release,
        portable,
        port,
        strictPort,
    };
}

if (import.meta.main) {
    const raw = process.argv.slice(2);

    let parsed: CliArgs;
    try {
        parsed = parseCliArgs(raw);
    } catch (e) {
        console.error(e instanceof Error ? e.message : String(e));
        process.exit(1);
    }

    if (parsed.kind === "delegate") {
        if (parsed.cmd === "verify") {
            const { runVerify } = await import("./verify");
            process.exit(await runVerify(parsed.rest));
        } else {
            const { runRecipe } = await import("./recipe");
            process.exit(await runRecipe(parsed.rest));
        }
    } else if (parsed.kind === "usage") {
        console.log(usage);
        process.exit(parsed.exitCode);
    } else {
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
                await startDev(projectDir, { port: parsed.port, strictPort: parsed.strictPort });
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
}
