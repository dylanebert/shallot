#!/usr/bin/env bun
import {
    collectPopulation,
    formatPopulation,
    readCheckDeclarations,
    writeWorkflow,
} from "../src/harness/surface";

export { readCheckDeclarations };

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
    if (!args.includes("--list")) {
        console.error("usage: bun scripts/surface.ts --list|--workflow [--root <dir>]");
        process.exit(1);
    }
    const population = collectPopulation(root);
    console.log(formatPopulation(population));
    if (population.invalid.length > 0 || population.undeclared.length > 0) process.exit(1);
}
