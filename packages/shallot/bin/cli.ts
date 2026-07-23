#!/usr/bin/env bun
import { resolve } from "node:path";
import { buildProject } from "./build";
import { startDev } from "./dev";
import { runProject } from "./run";

const raw = process.argv.slice(2);

// `verify` owns its own flag set (--dist, --screenshot, --query, --timeout, --json), so route it before
// the shared dev/build/run parse rather than teaching that loop every verify flag.
if (raw[0] === "verify") {
    const { runVerify } = await import("./verify");
    process.exit(await runVerify(raw.slice(1)));
}

// `recipe` copies a shipped example project out of the installed package — its own positional shape
// (name + dest dir), no shared dev/build/run flags, so route it before that parse.
if (raw[0] === "recipe") {
    const { runRecipe } = await import("./recipe");
    process.exit(await runRecipe(raw.slice(1)));
}

const positionalArgs: string[] = [];
let target: string | undefined;
let release = false;
let portable = false;
let port: number | undefined;
let strictPort = false;
let help = false;

for (let i = 0; i < raw.length; i++) {
    if (raw[i] === "--target" && raw[i + 1]) {
        target = raw[i + 1];
        i++;
    } else if (raw[i] === "--release") {
        release = true;
    } else if (raw[i] === "--portable") {
        portable = true;
    } else if (raw[i] === "--port" && raw[i + 1]) {
        port = parseInt(raw[i + 1]);
        i++;
    } else if (raw[i]?.startsWith("--port=")) {
        port = parseInt(raw[i].split("=")[1]);
    } else if (raw[i] === "--strict-port") {
        strictPort = true;
    } else if (raw[i] === "--help" || raw[i] === "-h") {
        help = true;
    } else if (raw[i].startsWith("-")) {
        console.error(`unknown option: ${raw[i]}`);
        process.exit(1);
    } else {
        positionalArgs.push(raw[i]);
    }
}

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
    --target <platform>   web (default), windows, mac, linux
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

if (help) {
    console.log(usage);
    process.exit(0);
}

const subcommands = ["dev", "build", "run"];
const subcmd = positionalArgs[0];
// bare `shallot [dir]` names no command — print usage rather than guess one.
if (subcmd == null || !subcommands.includes(subcmd)) {
    console.log(usage);
    process.exit(subcmd == null ? 0 : 1);
}
const dir = positionalArgs[1] || ".";
const projectDir = resolve(dir);

if (subcmd === "dev") {
    // native webviews can't HMR — `dev --target <native>` is a debug build + run (run without --release)
    if (target && target !== "web") {
        await runProject(projectDir, { target, port, release: false, portable });
    } else {
        await startDev(projectDir, { port, strictPort });
    }
} else if (subcmd === "build") {
    await buildProject(projectDir, { target, release, portable });
} else if (subcmd === "run") {
    await runProject(projectDir, { target, port, release, portable });
}
