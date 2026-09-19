#!/usr/bin/env bun
import {
    CHECK_REQUIREMENTS,
    collectPopulation,
    formatPopulation,
    selectIntegrationRows,
    selectOracleRows,
    writeWorkflow,
} from "../src/harness/surface";

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
    const oracle = valueAfter("--oracle");
    const oracleRequested = args.includes("--oracle");
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
    if (oracleRequested && args.filter((arg) => arg === "--oracle").length !== 1)
        refuse("--oracle accepts exactly one claim");
    if (
        oracleRequested &&
        (oracle === undefined || oracle.trim() === "" || oracle.startsWith("--"))
    )
        refuse("--oracle needs a claim");
    if (
        oracleRequested &&
        (integration ||
            selectorRequested ||
            base !== undefined ||
            diff !== undefined ||
            args.includes("--base") ||
            args.includes("--diff"))
    )
        refuse("--oracle cannot be combined with selectors or integration mode");
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
    if (oracle !== undefined) {
        const rows = selectOracleRows(population, oracle);
        if (rows.length !== 1) refuse(`named oracle not found: ${oracle}`);
        console.log(formatPopulation(population, undefined, rows));
        process.exit(0);
    }
    if (!integration) {
        console.log(formatPopulation(population));
        process.exit(0);
    }
    const rows = selectIntegrationRows(population, { all, requires, subject, base, diff });
    if (selectorRequested && rows.length === 0) refuse("selector matched no integration rows");
    console.log(formatPopulation(population, undefined, rows));
}
