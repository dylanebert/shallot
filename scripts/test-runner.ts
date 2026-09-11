#!/usr/bin/env bun
import { resolve } from "node:path";
import {
    collectPopulation,
    discoverTestFiles,
    selectIntegrationRows,
} from "../src/harness/surface";

const args = Bun.argv.slice(2);
const rootIndex = args.indexOf("--root");
const root = resolve(
    rootIndex === -1 ? resolve(import.meta.dir, "..") : (args[rootIndex + 1] ?? process.cwd()),
);
const integration = args.includes("--integration");
const base = valueAfter("--base");
const diff = valueAfter("--diff");
const oracle = valueAfter("--oracle");
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
    if (files.length === 0) {
        console.log("empty population; no tests");
        return 0;
    }
    const proc = Bun.spawnSync(
        [process.execPath, "test", "--max-concurrency=1", "--pass-with-no-tests", ...files],
        { cwd: root, env: environment, stdout: "inherit", stderr: "inherit" },
    );
    return proc.exitCode ?? 1;
}

const population = collectPopulation(root);
if (population.invalid.length > 0 || population.undeclared.length > 0) {
    refuse(
        [
            ...population.invalid,
            ...population.undeclared.map((file) => `${file.file} ${file.reason}`),
        ].join("; "),
    );
}
const files = discoverTestFiles(root, oracle !== undefined);
if (!integration) {
    const environment = { ...envBase, SHALLOT_UNIT_ONLY: "1" };
    process.exit(run(files, environment));
}
if (base === undefined || diff === undefined || base === "" || diff === "") {
    refuse("test:integration requires --base <ref> and --diff <ref>");
}
function isCommitObject(ref: string): boolean {
    const resolved = Bun.spawnSync(
        ["git", "rev-parse", "--verify", "--quiet", "--end-of-options", `${ref}^{commit}`],
        { cwd: root, stdout: "pipe", stderr: "pipe" },
    );
    return resolved.success && resolved.stdout.toString().trim() !== "";
}

if (!isCommitObject(base) || !isCommitObject(diff))
    refuse(`integration refs must be existing commit objects: base=${base} diff=${diff}`);
const selected =
    oracle === undefined
        ? selectIntegrationRows(population, base, diff)
        : population.rows.filter((row) => row.claim === oracle);
if (oracle !== undefined && selected.length === 0) refuse(`named oracle not found: ${oracle}`);
if (selected.length === 0) {
    // Unit rows still get their normal hermetic proof, but no integration/no-op command is claimed.
    process.exit(run(files, { ...envBase, SHALLOT_UNIT_ONLY: "1" }));
}
for (const row of selected) {
    const code = run(files, { ...envBase, KEX_S3_ROW: row.claim });
    if (code !== 0) process.exit(code);
    console.log(`selected integration: ${row.claim}`);
}
