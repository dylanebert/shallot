#!/usr/bin/env bun
import {
    CHECK_REQUIREMENTS,
    collectPopulation,
    formatPopulation,
    readCheckDeclarations,
    selectIntegrationRows,
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
        console.error("usage: bun scripts/surface.ts --list|--workflow [selectors] [--root <dir>]");
        process.exit(1);
    }
    const valueAfter = (flag: string): string | undefined => {
        const index = args.indexOf(flag);
        return index === -1 ? undefined : args[index + 1];
    };
    const all = args.includes("--all");
    const requires = valueAfter("--requires");
    const subject = valueAfter("--subject");
    const base = valueAfter("--base");
    const diff = valueAfter("--diff");
    const selectorRequested = all || args.includes("--requires") || args.includes("--subject");
    const integration =
        args.includes("--integration") ||
        selectorRequested ||
        base !== undefined ||
        diff !== undefined;
    const refuse = (message: string): never => {
        console.error(`surface refused: ${message}`);
        process.exit(1);
    };
    if (args.includes("--requires") && (requires === undefined || requires.startsWith("--")))
        refuse("--requires needs a requirement tag");
    if (args.includes("--subject") && (subject === undefined || subject.startsWith("--")))
        refuse("--subject needs a path prefix");
    if (requires !== undefined && !CHECK_REQUIREMENTS.includes(requires as never))
        refuse(`unknown requirement tag: ${requires}`);
    if (subject !== undefined && subject.trim() === "") refuse("--subject needs a path prefix");
    if (selectorRequested && (base !== undefined || diff !== undefined))
        refuse("selectors cannot be combined with --base/--diff");
    if (integration && !selectorRequested && (base === undefined || diff === undefined))
        refuse("list integration requires --base <ref> and --diff <ref>");
    if (
        base === "" ||
        diff === "" ||
        (base !== undefined && diff === undefined) ||
        (base === undefined && diff !== undefined)
    )
        refuse("list integration requires --base <ref> and --diff <ref>");
    const isCommitObject = (ref: string): boolean => {
        const resolved = Bun.spawnSync(
            ["git", "rev-parse", "--verify", "--quiet", "--end-of-options", `${ref}^{commit}`],
            { cwd: root, stdout: "pipe", stderr: "pipe" },
        );
        return resolved.success && resolved.stdout.toString().trim() !== "";
    };
    if (
        base !== undefined &&
        diff !== undefined &&
        (!isCommitObject(base) || !isCommitObject(diff))
    )
        refuse(`integration refs must be existing commit objects: base=${base} diff=${diff}`);
    const population = collectPopulation(root);
    if (population.invalid.length > 0 || population.undeclared.length > 0) {
        console.log(formatPopulation(population));
        process.exit(1);
    }
    if (!integration) {
        console.log(formatPopulation(population));
        process.exit(0);
    }
    const rows = selectIntegrationRows(population, { all, requires, subject, base, diff });
    if (selectorRequested && rows.length === 0) refuse("selector matched no integration rows");
    console.log(formatPopulation(population, undefined, rows));
}
