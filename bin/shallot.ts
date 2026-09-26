#!/usr/bin/env bun
import { resolve } from "node:path";
import { main } from "../src/cli/index";

const args = process.argv.slice(2);
const command = args[0];
const carrier = resolve(import.meta.dir, "../scripts");
const root = process.cwd();

if (command === "workflow" || command === "test") {
    const script = command === "workflow" ? "surface.ts" : "test-runner.ts";
    const scriptArgs = command === "workflow" ? ["--workflow"] : args.slice(1);
    const proc = Bun.spawnSync([process.execPath, resolve(carrier, script), ...scriptArgs, "--root", root], { cwd: root, stdout: "inherit", stderr: "inherit" });
    if (proc.signalCode) {
        const signal = proc.signalCode as NodeJS.Signals;
        process.removeAllListeners(signal);
        process.kill(process.pid, signal);
    }
    process.exit(proc.exitCode ?? 1);
}

await main(args);
