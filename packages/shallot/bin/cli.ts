#!/usr/bin/env bun
import { resolve } from "node:path";
import { buildProject } from "./build";
import { startDev } from "./dev";
import { startEdit } from "./edit";
import { runProject } from "./run";

const raw = process.argv.slice(2);
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

if (help) {
    console.log(`
  shallot — run, edit, and build a shallot project

  Usage
    shallot [command] [dir] [options]

  Commands
    (none)    Open the project in the editor (default)
    edit      Open the project in the editor
    dev       Run the project standalone, with hot reload
    build     Build for distribution
    run       Build and run

  Options
    --target <platform>   web (default), windows, mac, linux
    --release             Optimized build (build, run)
    --portable            Bundle the Chromium runtime (CEF) instead of the system webview.
                          Larger, but self-contained and runs anywhere. Required on Linux
                          (WebKitGTK has no usable WebGPU) and for apps needing subgroups on macOS.
    --port <n>            Server port (dev, edit, run)
    --strict-port         Fail if the port is in use instead of picking another
    -h, --help            Show this help

  Examples
    shallot                      Edit the current project
    shallot dev                  Run with hot reload
    shallot build --target mac   Build a macOS app (system WKWebView)
    shallot build --target linux --portable   Build a self-contained Linux app
`);
    process.exit(0);
}

const subcommands = ["edit", "dev", "build", "run"];
const first = positionalArgs[0];
const isSubcmd = first != null && subcommands.includes(first);
// bare `shallot [dir]` opens the editor — the headline harness; `dev` runs the project standalone.
const subcmd = isSubcmd ? first : "edit";
const dir = isSubcmd ? positionalArgs[1] || "." : first || ".";
const projectDir = resolve(dir);

if (subcmd === "edit") {
    await startEdit(projectDir, { port, strictPort });
} else if (subcmd === "dev") {
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
