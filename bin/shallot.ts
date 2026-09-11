#!/usr/bin/env bun
import { resolve } from "node:path";
import { main } from "../src/cli/index";

const args = process.argv.slice(2);
const command = args[0];
const carrier = resolve(import.meta.dir, "../scripts");
const root = process.cwd();

if (command === "list" || command === "workflow" || command === "check" || command === "test" || command === "test:integration") {
    const script = command === "list" || command === "workflow" ? "surface.ts" : command === "check" ? "check-surface.ts" : "test-runner.ts";
    const scriptArgs = command === "list" ? ["--list"] : command === "workflow" ? ["--workflow"] : command === "test:integration" ? ["--integration", ...args.slice(1)] : command === "test" ? ["--unit"] : ["--list"];
    const proc = Bun.spawnSync([process.execPath, resolve(carrier, script), ...scriptArgs, "--root", root], { cwd: root, stdout: "inherit", stderr: "inherit" });
    process.exit(proc.exitCode ?? 1);
}

await main(args);
