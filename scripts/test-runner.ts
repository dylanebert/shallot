#!/usr/bin/env bun
import { resolve } from "node:path";
import {
    CHECK_REQUIREMENTS,
    collectPopulation,
    discoverTestFiles,
    selectIntegrationRows,
    selectOracleRows,
} from "../src/harness/surface";
import type { VerdictResult } from "../src/harness/verdict";

const args = Bun.argv.slice(2);
const rootIndex = args.indexOf("--root");
const root = resolve(
    rootIndex === -1 ? resolve(import.meta.dir, "..") : (args[rootIndex + 1] ?? process.cwd()),
);
const integration = args.includes("--integration");
const all = args.includes("--all");
const oracle = valueAfter("--oracle");
const oracleRequested = args.includes("--oracle");
const envBase = { ...process.env, SHALLOT_PROJECT_ROOT: root };

function valueAfter(flag: string): string | undefined {
    const index = args.indexOf(flag);
    return index === -1 ? undefined : args[index + 1];
}

function refuse(message: string): never {
    console.error(`surface refused: ${message}`);
    process.exit(1);
}

function run(files: string[], environment: NodeJS.ProcessEnv): number {
    if (files.length === 0) refuse("empty population; an empty run is never green");
    const proc = Bun.spawnSync(
        [
            process.execPath,
            "test",
            "--max-concurrency=1",
            "--pass-with-no-tests",
            ...files.map((file) => `./${file}`),
        ],
        { cwd: root, env: environment, stdout: "inherit", stderr: "inherit" },
    );
    return proc.exitCode ?? 1;
}

interface SelectedRun {
    result: VerdictResult;
    exitCode: number;
}

function selectedRun(row: (typeof population.rows)[number]): SelectedRun {
    const proc = Bun.spawnSync(
        [process.execPath, "test", "--max-concurrency=1", "--pass-with-no-tests", `./${row.file}`],
        { cwd: root, env: { ...envBase, KEX_S3_ROW: row.claim }, stdout: "pipe", stderr: "pipe" },
    );
    const stdout = proc.stdout.toString();
    const stderr = proc.stderr.toString();
    // Keep the child's streams intact: the runner owns the process boundary, not the check's
    // diagnostics. Parsing the structured verdict is only for the aggregate decision below.
    process.stdout.write(stdout);
    process.stderr.write(stderr);
    const verdict = `${stdout}\n${stderr}`
        .split("\n")
        .map((line) => line.match(/^shallot verdict (.+)$/)?.[1])
        .filter((line): line is string => line !== undefined)
        .map((line) => {
            try {
                return JSON.parse(line) as { claim?: unknown; result?: unknown };
            } catch {
                return undefined;
            }
        })
        .find(
            (value): value is { claim: string; result: VerdictResult } =>
                value?.claim === row.claim &&
                (value.result === "pass" ||
                    value.result === "fail" ||
                    value.result === "refused" ||
                    value.result === "unrun"),
        );
    if (verdict === undefined) {
        console.error(
            `selected integration: ${row.claim} (fail; no verdict; child exited ${proc.exitCode ?? 1})`,
        );
        return { result: "fail", exitCode: proc.exitCode ?? 1 };
    }
    return { result: verdict.result, exitCode: proc.exitCode ?? 1 };
}

const base = valueAfter("--base");
const diff = valueAfter("--diff");
const requires = valueAfter("--requires");
const subject = valueAfter("--subject");
const selectorRequested = all || args.includes("--requires") || args.includes("--subject");
if (oracleRequested && args.filter((arg) => arg === "--oracle").length !== 1)
    refuse("--oracle accepts exactly one claim");
if (oracleRequested && (oracle === undefined || oracle.trim() === "" || oracle.startsWith("--")))
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
if (selectorRequested && !integration) refuse("integration selectors require --integration");
if (selectorRequested && (base !== undefined || diff !== undefined))
    refuse("selectors cannot be combined with --base/--diff");
if (integration && !selectorRequested && (base === undefined || diff === undefined))
    refuse("integration test requires --base <ref> and --diff <ref>");

const population = collectPopulation(root);
if (population.invalid.length > 0 || population.undeclared.length > 0) {
    refuse(
        [
            ...population.invalid,
            ...population.undeclared.map((file) => `${file.file} ${file.reason}`),
        ].join("; "),
    );
}
if (oracle !== undefined) {
    const selected = selectOracleRows(population, oracle);
    if (selected.length !== 1) refuse(`named oracle not found: ${oracle}`);
    process.exit(run([selected[0].file], { ...envBase, KEX_S3_ROW: oracle }));
}
const files = discoverTestFiles(root);
if (!integration) {
    const environment = { ...envBase, SHALLOT_UNIT_ONLY: "1" };
    process.exit(run(files, environment));
}
function isCommitObject(ref: string): boolean {
    const resolved = Bun.spawnSync(
        ["git", "rev-parse", "--verify", "--quiet", "--end-of-options", `${ref}^{commit}`],
        { cwd: root, stdout: "pipe", stderr: "pipe" },
    );
    return resolved.success && resolved.stdout.toString().trim() !== "";
}

if (
    base === "" ||
    diff === "" ||
    (base !== undefined && diff === undefined) ||
    (base === undefined && diff !== undefined)
)
    refuse("integration test requires --base <ref> and --diff <ref>");
if (base !== undefined && diff !== undefined && (!isCommitObject(base) || !isCommitObject(diff)))
    refuse(`integration refs must be existing commit objects: base=${base} diff=${diff}`);
const selected = selectIntegrationRows(population, { all, requires, subject, base, diff });
if (selectorRequested && selected.length === 0) refuse("selector matched no integration rows");
if (selected.length === 0) {
    // Unit rows still get their normal hermetic proof, but no integration/no-op command is claimed.
    process.exit(run(files, { ...envBase, SHALLOT_UNIT_ONLY: "1" }));
}
let failed = false;
let ran = 0;
let refused = 0;
let unrun = 0;
for (const row of selected) {
    const outcome = selectedRun(row);
    console.log(`selected integration: ${row.claim} (${outcome.result}) exit=${outcome.exitCode}`);
    if (outcome.result === "pass" || outcome.result === "fail") ran += 1;
    if (outcome.result === "refused") refused += 1;
    if (outcome.result === "unrun") unrun += 1;
    if (outcome.result === "fail" || outcome.result === "refused" || outcome.exitCode !== 0)
        failed = true;
}
if (ran === 0) {
    console.error(
        `no integration rows ran (pass=${ran}, refused=${refused}, unrun=${unrun}); an all-unrun selection is never green`,
    );
    failed = true;
}
process.exit(failed ? 1 : 0);
