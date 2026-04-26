#!/usr/bin/env bun
import { resolve } from "node:path";
import { startDev } from "./dev";
import { buildProject } from "./build";
import { runProject } from "./run";

const raw = process.argv.slice(2);
const positionalArgs: string[] = [];
let target: string | undefined;
let release = false;
let port: number | undefined;
let help = false;

for (let i = 0; i < raw.length; i++) {
    if (raw[i] === "--target" && raw[i + 1]) {
        target = raw[i + 1];
        i++;
    } else if (raw[i] === "--release") {
        release = true;
    } else if (raw[i] === "--port" && raw[i + 1]) {
        port = parseInt(raw[i + 1]);
        i++;
    } else if (raw[i]?.startsWith("--port=")) {
        port = parseInt(raw[i].split("=")[1]);
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
  shallot [command] [dir] [options]

  Commands:
    dev        Open the editor (default)
    build      Build for distribution
    run        Build and run

  Options:
    --target <platform>   web (default), windows, mac, linux
    --release             Optimized release build (build, run)
    --port <n>            Server port (dev, run)
    -h, --help            Show help

  Examples:
    shallot .                                       Editor for current directory
    shallot build                                   Web build → dist/
    shallot build --target windows                  Windows debug → build/windows/debug/
    shallot build --target windows --release        Windows release → build/windows/release/
    shallot run                                     Web build + preview server
    shallot run --target windows                    Windows build + run
    shallot build --target mac                      macOS debug → build/mac/debug/
    shallot build --target mac --release            macOS release → build/mac/release/
    shallot run --target mac                        macOS build + run
    shallot build --target linux                    Linux debug → build/linux/debug/
    shallot build --target linux --release          Linux release → build/linux/release/
    shallot run --target linux                      Linux build + run
`);
    process.exit(0);
}

const subcommands = ["dev", "build", "run"];
const first = positionalArgs[0];
const isSubcmd = first != null && subcommands.includes(first);
const subcmd = isSubcmd ? first : "dev";
const dir = isSubcmd ? positionalArgs[1] || "." : first || ".";
const projectDir = resolve(dir);

if (subcmd === "dev") {
    await startDev(projectDir, { port });
} else if (subcmd === "build") {
    await buildProject(projectDir, { target, release });
} else if (subcmd === "run") {
    await runProject(projectDir, { target, port, release });
}
