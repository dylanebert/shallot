#!/usr/bin/env bun
import { resolve } from "node:path";
import { main } from "../src/cli/index";

const args = process.argv.slice(2);
const command = args[0];
const carrier = resolve(import.meta.dir, "../scripts");
const root = process.cwd();

if (command === "list" || command === "workflow" || command === "check" || command === "test") {
    const script = command === "list" || command === "workflow" ? "surface.ts" : command === "check" ? "check-surface.ts" : "test-runner.ts";
    const scriptArgs = command === "list" ? ["--list", ...args.slice(1)] : command === "workflow" ? ["--workflow"] : command === "test" && args.includes("--integration") ? ["--integration", ...args.slice(1).filter((arg) => arg !== "--integration")] : command === "test" ? ["--unit", ...args.slice(1)] : ["--list"];
    const proc = Bun.spawnSync([process.execPath, resolve(carrier, script), ...scriptArgs, "--root", root], { cwd: root, stdout: "inherit", stderr: "inherit" });
    process.exit(proc.exitCode ?? 1);
}

await main(args);
