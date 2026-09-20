#!/usr/bin/env bun
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { relative, resolve } from "node:path";
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

interface RunArtifacts {
    directory: string;
    report: string;
    output: string;
}

function openArtifacts(): RunArtifacts | null {
    try {
        mkdirSync(resolve(root, ".artifacts"), { recursive: true });
        const directory = mkdtempSync(resolve(root, ".artifacts", "shallot-run-"));
        return {
            directory,
            report: resolve(directory, "junit.xml"),
            output: resolve(directory, "output.log"),
        };
    } catch (error) {
        console.error(
            `surface refused: report destination unavailable: ${(error as Error).message}`,
        );
        return null;
    }
}

function reportPath(artifacts: RunArtifacts): string {
    return relative(root, artifacts.report).split("\\").join("/");
}

function outputText(stdout: string, stderr: string): string {
    return `--- stdout ---\n${stdout}\n--- stderr ---\n${stderr}\n`;
}

function saveOutput(artifacts: RunArtifacts, output: string): boolean {
    try {
        writeFileSync(artifacts.output, output);
        return true;
    } catch (error) {
        console.error(`surface refused: run artifact unavailable: ${(error as Error).message}`);
        return false;
    }
}

interface JunitSummary {
    tests: number;
    failures: number;
    errors: number;
    skipped: number;
}

function junitAttribute(xml: string, name: string): number {
    const value = xml.match(new RegExp(`<testsuites\\b[^>]*\\b${name}="(\\d+)"`))?.[1];
    return value === undefined ? 0 : Number(value);
}

function readJunitSummary(path: string): JunitSummary | null {
    try {
        const xml = readFileSync(path, "utf8");
        return {
            tests: junitAttribute(xml, "tests"),
            failures: junitAttribute(xml, "failures"),
            errors: junitAttribute(xml, "errors"),
            skipped: junitAttribute(xml, "skipped"),
        };
    } catch {
        return null;
    }
}

function run(files: string[], environment: NodeJS.ProcessEnv): number {
    if (files.length === 0) refuse("empty population; an empty run is never green");
    const artifacts = openArtifacts();
    if (artifacts === null) return 1;
    const proc = Bun.spawnSync(
        [
            process.execPath,
            "test",
            "--max-concurrency=1",
            "--pass-with-no-tests",
            "--reporter=junit",
            `--reporter-outfile=${artifacts.report}`,
            ...files.map((file) => `./${file}`),
        ],
        { cwd: root, env: environment, stdout: "pipe", stderr: "pipe" },
    );
    const stdout = proc.stdout.toString();
    const stderr = proc.stderr.toString();
    let code = proc.exitCode ?? 1;
    if (!saveOutput(artifacts, outputText(stdout, stderr))) code = 1;
    const summary = readJunitSummary(artifacts.report);
    if (summary === null) {
        console.error(
            `surface refused: runner did not write a valid report: ${reportPath(artifacts)}`,
        );
        return 1;
    }
    if (code !== 0 || environment.KEX_S3_ROW !== undefined) {
        process.stdout.write(stdout);
        process.stderr.write(stderr);
    }
    const passed = Math.max(0, summary.tests - summary.failures - summary.errors - summary.skipped);
    console.log(
        `shallot test: ${passed} passed, ${summary.failures} failed, ${summary.errors} refused, ${summary.skipped} skipped; report: ${reportPath(artifacts)}`,
    );
    return code;
}

interface ChildVerdict {
    claim: string;
    result: VerdictResult;
    reason?: string;
    duration?: number;
}

interface SelectedRun {
    result: VerdictResult;
    exitCode: number;
    stdout: string;
    stderr: string;
    reason?: string;
    duration?: number;
    noVerdict?: boolean;
}

function normalizeChildOutcome(result: VerdictResult, exitCode: number): VerdictResult {
    if (exitCode === 0 || result === "fail" || result === "refused") return result;
    // An unrun verdict is honest only when the child completed normally. A module can register an
    // other-host row, emit its unrun verdict, then throw while loading; that load failure is red, not a
    // host mismatch. The same rule keeps a pass from hiding a nonzero child exit.
    return "fail";
}

function selectedRun(
    row: (typeof population.rows)[number],
    artifacts: RunArtifacts,
    index: number,
): SelectedRun {
    const nativeReport = resolve(artifacts.directory, `child-${index}.xml`);
    const proc = Bun.spawnSync(
        [
            process.execPath,
            "test",
            "--max-concurrency=1",
            "--pass-with-no-tests",
            "--reporter=junit",
            `--reporter-outfile=${nativeReport}`,
            `./${row.file}`,
        ],
        { cwd: root, env: { ...envBase, KEX_S3_ROW: row.claim }, stdout: "pipe", stderr: "pipe" },
    );
    const stdout = proc.stdout.toString();
    const stderr = proc.stderr.toString();
    const exitCode = proc.exitCode ?? 1;
    rmSync(nativeReport, { force: true });
    const verdict = `${stdout}\n${stderr}`
        .split("\n")
        .map((line) => line.match(/^shallot verdict (.+)$/)?.[1])
        .filter((line): line is string => line !== undefined)
        .map((line): ChildVerdict | undefined => {
            try {
                const value = JSON.parse(line) as Partial<ChildVerdict>;
                return value.claim === row.claim &&
                    (value.result === "pass" ||
                        value.result === "fail" ||
                        value.result === "refused" ||
                        value.result === "unrun")
                    ? {
                          claim: value.claim,
                          result: value.result,
                          ...(typeof value.reason === "string" ? { reason: value.reason } : {}),
                          ...(typeof value.duration === "number"
                              ? { duration: value.duration }
                              : {}),
                      }
                    : undefined;
            } catch {
                return undefined;
            }
        })
        .find((value): value is ChildVerdict => value !== undefined);
    if (verdict === undefined) {
        return {
            result: "fail",
            exitCode,
            stdout,
            stderr,
            noVerdict: true,
        };
    }
    return {
        result: normalizeChildOutcome(verdict.result, exitCode),
        exitCode,
        stdout,
        stderr,
        ...(verdict.reason === undefined ? {} : { reason: verdict.reason }),
        ...(verdict.duration === undefined ? {} : { duration: verdict.duration }),
    };
}

function xml(value: string): string {
    const safe = [...value]
        .filter((character) => {
            const code = character.codePointAt(0) ?? 0;
            return (
                code === 0x09 ||
                code === 0x0a ||
                code === 0x0d ||
                (code >= 0x20 &&
                    code !== 0xfffe &&
                    code !== 0xffff &&
                    (code < 0xd800 || code > 0xdfff))
            );
        })
        .join("");
    return safe
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&apos;");
}

function selectedJunit(
    rows: readonly (typeof population.rows)[number][],
    outcomes: readonly SelectedRun[],
): string {
    const failures = outcomes.filter((outcome) => outcome.result === "fail").length;
    const errors = outcomes.filter((outcome) => outcome.result === "refused").length;
    const skipped = outcomes.filter((outcome) => outcome.result === "unrun").length;
    const seconds = outcomes.reduce((sum, outcome) => sum + (outcome.duration ?? 0), 0) / 1000;
    const cases = rows
        .map((row, index) => {
            const outcome = outcomes[index];
            const time = ((outcome.duration ?? 0) / 1000).toFixed(6);
            const start = `  <testcase name="${xml(row.claim)}" classname="${xml(row.file)}" file="${xml(row.file)}" time="${time}"`;
            if (outcome.result === "pass") return `${start} />`;
            const details = outputText(outcome.stdout, outcome.stderr);
            if (outcome.result === "unrun") {
                return `${start}>\n    <skipped message="${xml(outcome.reason ?? "host premise is unavailable")}" />\n  </testcase>`;
            }
            const type = outcome.result === "refused" ? "refused" : "failure";
            const message =
                outcome.reason ?? (outcome.result === "refused" ? "check refused" : "check failed");
            const element = outcome.result === "refused" ? "error" : "failure";
            return `${start}>\n    <${element} type="${type}" message="${xml(message)}">${xml(details)}</${element}>\n    <system-out>${xml(outcome.stdout)}</system-out>\n    <system-err>${xml(outcome.stderr)}</system-err>\n  </testcase>`;
        })
        .join("\n");
    return `<?xml version="1.0" encoding="UTF-8"?>\n<testsuites name="shallot test run" tests="${rows.length}" failures="${failures}" errors="${errors}" skipped="${skipped}" time="${seconds.toFixed(6)}">\n<testsuite name="selected integration" tests="${rows.length}" failures="${failures}" errors="${errors}" skipped="${skipped}" time="${seconds.toFixed(6)}">\n${cases}\n</testsuite>\n</testsuites>\n`;
}

function selectedSummary(
    outcomes: readonly SelectedRun[],
    artifacts: RunArtifacts,
    failed: boolean,
): number {
    const passed = outcomes.filter((outcome) => outcome.result === "pass").length;
    const failures = outcomes.filter((outcome) => outcome.result === "fail").length;
    const refused = outcomes.filter((outcome) => outcome.result === "refused").length;
    const unrun = outcomes.filter((outcome) => outcome.result === "unrun").length;
    console.log(
        `shallot test: ${passed} passed, ${failures} failed, ${refused} refused, ${unrun} unrun; report: ${reportPath(artifacts)}`,
    );
    return failed ? 1 : 0;
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
const artifacts = openArtifacts();
if (artifacts === null) process.exit(1);
const outcomes: SelectedRun[] = [];
const evidence: string[] = [];
let failed = false;
let ran = 0;
let refused = 0;
let unrun = 0;
for (const [index, row] of selected.entries()) {
    const outcome = selectedRun(row, artifacts, index);
    outcomes.push(outcome);
    evidence.push(
        `=== ${row.claim} (${outcome.result}) ===\n${outputText(outcome.stdout, outcome.stderr)}`,
    );
    if (outcome.result === "pass" || outcome.result === "fail") ran += 1;
    if (outcome.result === "refused") refused += 1;
    if (outcome.result === "unrun") unrun += 1;
    if (outcome.result === "fail" || outcome.result === "refused" || outcome.exitCode !== 0)
        failed = true;
    if (outcome.result === "fail" || outcome.result === "refused") {
        console.error(
            `selected integration: ${row.claim} (${outcome.result}${outcome.noVerdict ? `; no verdict; child exited ${outcome.exitCode}` : ""})`,
        );
        process.stdout.write(outcome.stdout);
        process.stderr.write(outcome.stderr);
    }
}
if (ran === 0) {
    console.error(
        `no integration rows ran (pass=${ran}, refused=${refused}, unrun=${unrun}); an all-unrun selection is never green`,
    );
    failed = true;
}
try {
    writeFileSync(artifacts.report, selectedJunit(selected, outcomes));
} catch (error) {
    console.error(`surface refused: report destination unavailable: ${(error as Error).message}`);
    failed = true;
}
if (!saveOutput(artifacts, evidence.join("\n"))) failed = true;
process.exit(selectedSummary(outcomes, artifacts, failed));
