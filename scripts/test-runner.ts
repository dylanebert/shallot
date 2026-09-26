#!/usr/bin/env bun
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import {
    CHECK_REQUIREMENTS,
    collectPopulation,
    discoverTestFiles,
    formatPopulation,
    readSurface,
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
const list = args.includes("--list");
const all = args.includes("--all");
const oracle = valueAfter("--oracle");
const oracleRequested = args.includes("--oracle");
const envBase = { ...process.env, SHALLOT_PROJECT_ROOT: root };
const HARNESS_PRELOAD = resolve(import.meta.dir, "../src/harness/preload.ts");

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
        const artifactsRoot = resolve(root, ".artifacts");
        mkdirSync(artifactsRoot, { recursive: true });
        for (const entry of readdirSync(artifactsRoot, { withFileTypes: true })) {
            if (entry.name.startsWith("shallot-run-"))
                rmSync(resolve(artifactsRoot, entry.name), { recursive: true, force: true });
        }
        const directory = mkdtempSync(resolve(artifactsRoot, "shallot-run-"));
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

function childEnvironment(
    environment: NodeJS.ProcessEnv,
    temporaryDirectory?: string,
): NodeJS.ProcessEnv {
    return {
        ...Object.fromEntries(
            Object.entries(environment).filter(
                ([key]) => !key.startsWith("GIT_") && key !== "SHALLOT_TEST_RUNNER_TRIPWIRE_MS",
            ),
        ),
        GIT_CONFIG_GLOBAL: "/dev/null",
        GIT_CONFIG_NOSYSTEM: "1",
        GIT_CEILING_DIRECTORIES: resolve(root, ".artifacts"),
        ...(temporaryDirectory === undefined ? {} : { TMPDIR: temporaryDirectory }),
    };
}

type ChildKind = "unit sweep" | "integration row" | "oracle";

interface ChildExecution {
    exitCode: number;
    stdout: string;
    stderr: string;
    redReason?: string;
    signal?: NodeJS.Signals;
}

const TRIPWIRES: Record<ChildKind, number> = {
    "unit sweep": 60_000,
    "integration row": 300_000,
    oracle: 300_000,
};
const FIXTURE_TRIPWIRE_LIMIT_MS = 1_000;
const TEARDOWN_GRACE_MS = 10_000;

function childTripwire(environment: NodeJS.ProcessEnv, kind: ChildKind): number {
    const override = environment.SHALLOT_TEST_RUNNER_TRIPWIRE_MS;
    if (override === undefined || !/^\d+$/.test(override)) return TRIPWIRES[kind];
    const milliseconds = Number(override);
    return milliseconds > 0 && milliseconds <= FIXTURE_TRIPWIRE_LIMIT_MS
        ? Math.min(TRIPWIRES[kind], milliseconds)
        : TRIPWIRES[kind];
}

function groupAlive(group: number): boolean {
    try {
        process.kill(-group, 0);
        return true;
    } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === "ESRCH") return false;
        if (code === "EPERM") return true;
        throw error;
    }
}

function processNames(group: number): string {
    try {
        const listing = Bun.spawnSync(["ps", "-axo", "pid=,pgid=,command="], {
            cwd: root,
            stdout: "pipe",
            stderr: "pipe",
        });
        if (listing.exitCode === 0) {
            const processes = listing.stdout
                .toString()
                .split("\n")
                .flatMap((line) => {
                    const match = line.match(/^\s*(\d+)\s+(\d+)\s+(.*)$/);
                    return match === null || Number(match[2]) !== group
                        ? []
                        : [{ pid: Number(match[1]), command: match[3] }];
                });
            if (processes.length > 0)
                return processes.map(({ pid, command }) => `pid ${pid} (${command})`).join(", ");
        }
    } catch {}
    return `process group ${group}`;
}

function signalGroup(group: number, signal: NodeJS.Signals): void {
    try {
        process.kill(-group, signal);
    } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code !== "ESRCH" && code !== "EPERM") throw error;
    }
}

async function waitForGroupExit(group: number, milliseconds: number): Promise<boolean> {
    const deadline = Date.now() + milliseconds;
    let alive = groupAlive(group);
    while (alive && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 50));
        alive = groupAlive(group);
    }
    return alive;
}

async function teardownGroup(group: number): Promise<boolean> {
    signalGroup(group, "SIGTERM");
    let alive = await waitForGroupExit(group, TEARDOWN_GRACE_MS);
    if (alive) {
        signalGroup(group, "SIGKILL");
        alive = await waitForGroupExit(group, TEARDOWN_GRACE_MS);
    }
    return alive;
}

async function spawnTest(
    artifacts: RunArtifacts,
    environment: NodeJS.ProcessEnv,
    command: string[],
    kind: ChildKind,
): Promise<ChildExecution> {
    const temporaryDirectory = mkdtempSync(resolve(artifacts.directory, "tmp-"));
    let proc: Bun.Subprocess<"ignore", "pipe", "pipe">;
    try {
        proc = Bun.spawn(command, {
            cwd: root,
            env: childEnvironment(environment, temporaryDirectory),
            stdout: "pipe",
            stderr: "pipe",
            detached: true,
        });
    } catch (error) {
        rmSync(temporaryDirectory, { recursive: true, force: true });
        throw error;
    }
    const stdoutPromise = new Response(proc.stdout).text().catch(() => "");
    const stderrPromise = new Response(proc.stderr).text().catch(() => "");
    const tripwire = childTripwire(environment, kind);
    let receivedSignal: NodeJS.Signals | undefined;
    let resolveSignal: ((signal: NodeJS.Signals) => void) | undefined;
    const signalPromise = new Promise<NodeJS.Signals>((resolve) => {
        resolveSignal = resolve;
    });
    const handlers = new Map<NodeJS.Signals, () => void>();
    for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
        const handler = () => {
            if (!groupAlive(proc.pid)) {
                process.removeListener(signal, handler);
                handlers.delete(signal);
                process.kill(process.pid, signal);
                return;
            }
            receivedSignal ??= signal;
            signalGroup(proc.pid, signal);
            resolveSignal?.(signal);
        };
        handlers.set(signal, handler);
        process.on(signal, handler);
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    const redReasons: string[] = [];
    let exitCode = 1;
    try {
        const stop = await Promise.race([
            proc.exited.then((code) => ({ kind: "exit" as const, code })),
            signalPromise.then((signal) => ({ kind: "signal" as const, signal })),
            new Promise<{ kind: "tripwire" }>((resolve) => {
                timer = setTimeout(() => resolve({ kind: "tripwire" }), tripwire);
            }),
        ]);
        if (receivedSignal !== undefined) {
            redReasons.push(`runner received ${receivedSignal} during ${kind}`);
        } else if (stop.kind === "tripwire" && proc.exitCode === null) {
            redReasons.push(`${kind} exceeded its ${tripwire} ms tripwire`);
        }

        if (stop.kind === "exit" || proc.exitCode !== null) {
            try {
                if (groupAlive(proc.pid))
                    redReasons.push(`process remained after child exit: ${processNames(proc.pid)}`);
            } catch (error) {
                redReasons.push((error as Error).message);
            }
        }
        const remaining = await teardownGroup(proc.pid);
        if (remaining) redReasons.push(`process remained after SIGKILL: ${processNames(proc.pid)}`);
        if (
            receivedSignal !== undefined &&
            !redReasons.some((reason) => reason.includes(receivedSignal!))
        )
            redReasons.push(`runner received ${receivedSignal} during ${kind}`);
        exitCode = redReasons.length > 0 ? 1 : (proc.exitCode ?? 1);
        for (const [signal, handler] of handlers) process.removeListener(signal, handler);
        handlers.clear();
        await proc.exited;
        const [stdout, stderr] = await Promise.all([stdoutPromise, stderrPromise]);
        return {
            exitCode,
            stdout,
            stderr,
            ...(redReasons.length === 0 ? {} : { redReason: redReasons.join("; ") }),
            ...(receivedSignal === undefined ? {} : { signal: receivedSignal }),
        };
    } finally {
        if (timer !== undefined) clearTimeout(timer);
        for (const [signal, handler] of handlers) process.removeListener(signal, handler);
        rmSync(temporaryDirectory, { recursive: true, force: true });
    }
}

function failedJunit(name: string, reason: string, stdout: string, stderr: string): string {
    const details = outputText(stdout, stderr);
    return `<?xml version="1.0" encoding="UTF-8"?>\n<testsuites name="shallot test run" tests="1" failures="1" errors="0" skipped="0" time="0">\n<testsuite name="runner child" tests="1" failures="1" errors="0" skipped="0" time="0">\n  <testcase name="${xml(name)}" time="0">\n    <failure type="failure" message="${xml(reason)}">${xml(details)}</failure>\n    <system-out>${xml(stdout)}</system-out>\n    <system-err>${xml(stderr)}</system-err>\n  </testcase>\n</testsuite>\n</testsuites>\n`;
}

async function run(
    files: string[],
    environment: NodeJS.ProcessEnv,
    kind: "unit sweep" | "oracle",
    name: string,
): Promise<number> {
    if (files.length === 0) refuse("empty population; an empty run is never green");
    const artifacts = openArtifacts();
    if (artifacts === null) return 1;
    const child = await spawnTest(
        artifacts,
        environment,
        [
            process.execPath,
            "test",
            "--preload",
            HARNESS_PRELOAD,
            "--max-concurrency=1",
            "--pass-with-no-tests",
            "--reporter=junit",
            `--reporter-outfile=${artifacts.report}`,
            ...files.map((file) => `./${file}`),
        ],
        kind,
    );
    const output =
        outputText(child.stdout, child.stderr) +
        (child.redReason === undefined ? "" : `--- runner ---\n${child.redReason}\n`);
    let code = child.exitCode;
    if (!saveOutput(artifacts, output)) code = 1;
    let summary: JunitSummary | null;
    if (child.redReason !== undefined) {
        try {
            writeFileSync(
                artifacts.report,
                failedJunit(name, child.redReason, child.stdout, child.stderr),
            );
            summary = { tests: 1, failures: 1, errors: 0, skipped: 0 };
        } catch {
            summary = null;
        }
    } else {
        summary = readJunitSummary(artifacts.report);
    }
    if (summary === null) {
        console.error(
            `surface refused: runner did not write a valid report: ${reportPath(artifacts)}`,
        );
        if (child.signal !== undefined) process.kill(process.pid, child.signal);
        return 1;
    }
    if (child.redReason !== undefined) console.error(`${kind}: ${child.redReason}`);
    if (code !== 0) {
        process.stdout.write(child.stdout);
        process.stderr.write(child.stderr);
    }
    const passed = Math.max(0, summary.tests - summary.failures - summary.errors - summary.skipped);
    console.log(
        `shallot test: ${passed} passed, ${summary.failures} failed, ${summary.errors} refused, ${summary.skipped} skipped; report: ${reportPath(artifacts)}`,
    );
    if (child.signal !== undefined) process.kill(process.pid, child.signal);
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
    signal?: NodeJS.Signals;
    reason?: string;
    duration?: number;
    noVerdict?: boolean;
}

function normalizeChildOutcome(verdict: ChildVerdict, exitCode: number): ChildVerdict {
    if (exitCode === 0 || verdict.result === "fail" || verdict.result === "refused") return verdict;
    // An unrun verdict is honest only when the child completed normally. A module can emit an unrun
    // verdict, then throw while loading; that load failure is red, not an unrun outcome. Clear the
    // unrun reason with the result so the normalized failure is attributed to the actual child failure
    // while stdout/stderr retain its diagnostics. The same rule keeps a pass from hiding a nonzero child exit.
    return { ...verdict, result: "fail", reason: undefined };
}

async function selectedRun(
    row: (typeof population.rows)[number],
    artifacts: RunArtifacts,
    index: number,
): Promise<SelectedRun> {
    const nativeReport = resolve(artifacts.directory, `child-${index}.xml`);
    const child = await spawnTest(
        artifacts,
        { ...envBase, KEX_S3_ROW: row.claim },
        [
            process.execPath,
            "test",
            "--preload",
            HARNESS_PRELOAD,
            "--max-concurrency=1",
            "--pass-with-no-tests",
            "--reporter=junit",
            `--reporter-outfile=${nativeReport}`,
            `./${row.file}`,
        ],
        "integration row",
    );
    const { stdout, stderr } = child;
    const exitCode = child.exitCode;
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
        const runtimeUndeclared = exitCode === 0;
        return {
            result: runtimeUndeclared ? "refused" : "fail",
            exitCode,
            stdout,
            stderr,
            ...(child.signal === undefined ? {} : { signal: child.signal }),
            ...(runtimeUndeclared
                ? {
                      reason: `undeclared check at runtime: ${row.file} declares a check statically but registered none when it ran`,
                  }
                : child.redReason === undefined
                  ? {}
                  : { reason: child.redReason }),
            noVerdict: true,
        };
    }
    const normalized = normalizeChildOutcome(verdict, exitCode);
    return {
        result: child.redReason === undefined ? normalized.result : "fail",
        exitCode,
        stdout,
        stderr,
        ...(child.signal === undefined ? {} : { signal: child.signal }),
        ...(normalized.reason === undefined && child.redReason === undefined
            ? {}
            : {
                  reason: [normalized.reason, child.redReason]
                      .filter((reason): reason is string => reason !== undefined)
                      .join("; "),
              }),
        ...(normalized.duration === undefined ? {} : { duration: normalized.duration }),
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
    runnerFailure?: string,
): string {
    const failures =
        outcomes.filter((outcome) => outcome.result === "fail").length +
        (runnerFailure === undefined ? 0 : 1);
    const errors = outcomes.filter((outcome) => outcome.result === "refused").length;
    const skipped = outcomes.filter((outcome) => outcome.result === "unrun").length;
    const tests = rows.length + (runnerFailure === undefined ? 0 : 1);
    const seconds = outcomes.reduce((sum, outcome) => sum + (outcome.duration ?? 0), 0) / 1000;
    const cases = rows
        .map((row, index) => {
            const outcome = outcomes[index];
            const time = ((outcome.duration ?? 0) / 1000).toFixed(6);
            const start = `  <testcase name="${xml(row.claim)}" classname="${xml(row.file)}" file="${xml(row.file)}" time="${time}"`;
            if (outcome.result === "pass") return `${start} />`;
            const details = outputText(outcome.stdout, outcome.stderr);
            if (outcome.result === "unrun") {
                return `${start}>\n    <skipped message="${xml(outcome.reason ?? "premise is unavailable")}" />\n  </testcase>`;
            }
            const type = outcome.result === "refused" ? "refused" : "failure";
            const message =
                outcome.reason ?? (outcome.result === "refused" ? "check refused" : "check failed");
            const element = outcome.result === "refused" ? "error" : "failure";
            return `${start}>\n    <${element} type="${type}" message="${xml(message)}">${xml(details)}</${element}>\n    <system-out>${xml(outcome.stdout)}</system-out>\n    <system-err>${xml(outcome.stderr)}</system-err>\n  </testcase>`;
        })
        .join("\n");
    const runnerCase =
        runnerFailure === undefined
            ? ""
            : `\n  <testcase name="runner interruption" classname="scripts/test-runner.ts" file="scripts/test-runner.ts" time="0">\n    <failure type="failure" message="${xml(runnerFailure)}" />\n  </testcase>`;
    return `<?xml version="1.0" encoding="UTF-8"?>\n<testsuites name="shallot test run" tests="${tests}" failures="${failures}" errors="${errors}" skipped="${skipped}" time="${seconds.toFixed(6)}">\n<testsuite name="selected integration" tests="${tests}" failures="${failures}" errors="${errors}" skipped="${skipped}" time="${seconds.toFixed(6)}">\n${cases}${runnerCase}\n</testsuite>\n</testsuites>\n`;
}

function selectedSummary(
    outcomes: readonly SelectedRun[],
    artifacts: RunArtifacts,
    failed: boolean,
    runnerFailure?: string,
): number {
    const passed = outcomes.filter((outcome) => outcome.result === "pass").length;
    const failures =
        outcomes.filter((outcome) => outcome.result === "fail").length +
        (runnerFailure === undefined ? 0 : 1);
    const refused = outcomes.filter((outcome) => outcome.result === "refused").length;
    const unrun = outcomes.filter((outcome) => outcome.result === "unrun").length;
    console.log(
        `shallot test: ${passed} passed, ${failures} failed, ${refused} refused, ${unrun} unrun; report: ${reportPath(artifacts)}`,
    );
    return failed ? 1 : 0;
}

const base = valueAfter("--base");
const diff = valueAfter("--diff");
const requireFlags = args.flatMap((arg, index) => (arg === "--requires" ? [index] : []));
const subject = valueAfter("--subject");
const selectorRequested = all || requireFlags.length > 0 || args.includes("--subject");
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
if (
    requireFlags.some((index) => args[index + 1] === undefined || args[index + 1]?.startsWith("--"))
)
    refuse("--requires needs a requirement tag");
const requires = requireFlags.map((index) => args[index + 1] as string);
if (args.includes("--subject") && (subject === undefined || subject.startsWith("--")))
    refuse("--subject needs a path prefix");
for (const filter of requires) {
    const requirement = filter.startsWith("!") ? filter.slice(1) : filter;
    if (requirement === "" || !CHECK_REQUIREMENTS.includes(requirement as never))
        refuse(`unknown requirement tag: ${requirement}`);
}
if (subject !== undefined && subject.trim() === "") refuse("--subject needs a path prefix");
if (selectorRequested && !integration) refuse("integration selectors require --integration");
if (
    (all || args.includes("--subject")) &&
    (base !== undefined || diff !== undefined || args.includes("--base") || args.includes("--diff"))
)
    refuse("--all/--subject cannot be combined with --base/--diff");
if (integration && !selectorRequested && (base === undefined || diff === undefined))
    refuse("integration test requires --base <ref> and --diff <ref>");

const population = collectPopulation(root);
const violations = readSurface(root, population);
if (violations.length > 0) refuse(violations.join("; "));
if (oracle !== undefined) {
    const selected = selectOracleRows(population, oracle);
    if (selected.length !== 1) refuse(`named oracle not found: ${oracle}`);
    if (list) {
        console.log(formatPopulation(population, undefined, selected));
        process.exit(0);
    }
    process.exit(
        await run([selected[0].file], { ...envBase, KEX_S3_ROW: oracle }, "oracle", oracle),
    );
}
const files = discoverTestFiles(root);
if (!integration) {
    if (list) {
        console.log(formatPopulation(population));
        process.exit(0);
    }
    const environment = { ...envBase, SHALLOT_UNIT_ONLY: "1" };
    process.exit(await run(files, environment, "unit sweep", "unit sweep"));
}
function isCommitObject(ref: string): boolean {
    const resolved = Bun.spawnSync(
        ["git", "rev-parse", "--verify", "--quiet", "--end-of-options", `${ref}^{commit}`],
        { cwd: root, env: childEnvironment(envBase), stdout: "pipe", stderr: "pipe" },
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
const selected = selectIntegrationRows(population, {
    all,
    requires: requires.length === 0 ? undefined : requires,
    subject,
    base,
    diff,
});
if (selectorRequested && selected.length === 0 && base === undefined && diff === undefined)
    refuse("selector matched no integration rows");
if (list) {
    console.log(formatPopulation(population, undefined, selected));
    process.exit(0);
}
if (selected.length === 0) {
    // Unit rows still get their normal hermetic proof, but no integration/no-op command is claimed.
    process.exit(
        await run(files, { ...envBase, SHALLOT_UNIT_ONLY: "1" }, "unit sweep", "unit sweep"),
    );
}
const artifacts = openArtifacts();
if (artifacts === null) process.exit(1);
const outcomes: SelectedRun[] = [];
const evidence: string[] = [];
let receivedSignal: NodeJS.Signals | undefined;
let failed = false;
let ran = 0;
let refused = 0;
let unrun = 0;
for (const [index, row] of selected.entries()) {
    if (receivedSignal !== undefined) {
        const reason = `runner received ${receivedSignal} before child start`;
        failed = true;
        for (const pending of selected.slice(index)) {
            const outcome: SelectedRun = {
                result: "unrun",
                exitCode: 0,
                stdout: "",
                stderr: "",
                reason,
            };
            outcomes.push(outcome);
            evidence.push(
                `=== ${pending.claim} (unrun) ===\nreason: ${reason}\n${outputText("", "")}`,
            );
            unrun += 1;
        }
        break;
    }
    const outcome = await selectedRun(row, artifacts, index);
    outcomes.push(outcome);
    receivedSignal ??= outcome.signal;
    evidence.push(
        `=== ${row.claim} (${outcome.result}) ===\n${outcome.reason === undefined ? "" : `reason: ${outcome.reason}\n`}${outputText(outcome.stdout, outcome.stderr)}`,
    );
    if (outcome.result === "pass" || outcome.result === "fail") ran += 1;
    if (outcome.result === "refused") refused += 1;
    if (outcome.result === "unrun") unrun += 1;
    if (outcome.result === "fail" || outcome.result === "refused" || outcome.exitCode !== 0)
        failed = true;
    if (outcome.result === "fail" || outcome.result === "refused") {
        console.error(
            `selected integration: ${row.claim} (${outcome.result}${outcome.reason === undefined ? (outcome.noVerdict ? `; no verdict; child exited ${outcome.exitCode}` : "") : `; ${outcome.reason}`})`,
        );
        process.stdout.write(outcome.stdout);
        process.stderr.write(outcome.stderr);
    }
}
const signalFailure =
    receivedSignal !== undefined &&
    !outcomes.some(
        (outcome) =>
            outcome.result === "fail" &&
            outcome.reason?.includes(`runner received ${receivedSignal}`),
    )
        ? `runner received ${receivedSignal} during integration run`
        : undefined;
if (receivedSignal !== undefined) failed = true;
if (signalFailure !== undefined) {
    evidence.push(`=== runner interruption (fail) ===\nreason: ${signalFailure}\n`);
    console.error(`selected integration: runner interruption (fail; ${signalFailure})`);
}
if (ran === 0) {
    console.error(
        `no integration rows ran (pass=${ran}, refused=${refused}, unrun=${unrun}); an all-unrun selection is never green`,
    );
    failed = true;
}
try {
    writeFileSync(artifacts.report, selectedJunit(selected, outcomes, signalFailure));
} catch (error) {
    console.error(`surface refused: report destination unavailable: ${(error as Error).message}`);
    failed = true;
}
if (!saveOutput(artifacts, evidence.join("\n"))) failed = true;
const exitCode = selectedSummary(outcomes, artifacts, failed, signalFailure);
if (receivedSignal !== undefined) process.kill(process.pid, receivedSignal);
process.exit(exitCode);
