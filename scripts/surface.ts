#!/usr/bin/env bun
import { writeWorkflow } from "../src/harness/surface";

if (import.meta.main) {
    const args = Bun.argv.slice(2);
    const rootIndex = args.indexOf("--root");
    const root = rootIndex === -1 ? process.cwd() : (args[rootIndex + 1] ?? process.cwd());
    if (args.includes("--workflow")) {
        const result = writeWorkflow(root);
        console.log(
            result === "empty"
                ? "empty population; no workflow"
                : `${result} .github/workflows/test-surface.yml`,
        );
        process.exit(0);
    }
    console.error("usage: bun scripts/surface.ts --workflow [--root <dir>]");
    process.exit(1);
}
