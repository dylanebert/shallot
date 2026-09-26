import { expect } from "bun:test";
import {
    chmodSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";

const ROOT = resolve(import.meta.dir, "..");
const CHECK_MODULE = resolve(ROOT, "src/harness/check");
const PRELOAD = resolve(ROOT, "src/harness/preload.ts");

function runRunnerWithEnv(tree: string, environment: NodeJS.ProcessEnv, ...args: string[]) {
    return Bun.spawnSync(
        ["bun", resolve(ROOT, "scripts/test-runner.ts"), "--root", tree, ...args],
        {
            cwd: ROOT,
            env: {
                ...process.env,
                ...environment,
                SHALLOT_HOST: "invented-seat",
                HYPRLAND_INSTANCE_SIGNATURE: "invented-compositor",
                SHALLOT_DISPLAY_SEAT: "",
            },
            stdout: "pipe",
            stderr: "pipe",
        },
    );
}

function runRunner(tree: string, ...args: string[]) {
    return runRunnerWithEnv(tree, {}, ...args);
}

function spawnRunnerWithEnv(
    tree: string,
    environment: NodeJS.ProcessEnv,
    ...args: string[]
): Bun.Subprocess<"ignore", "pipe", "pipe"> {
    return Bun.spawn(["bun", resolve(ROOT, "scripts/test-runner.ts"), "--root", tree, ...args], {
        cwd: ROOT,
        env: {
            ...process.env,
            ...environment,
            SHALLOT_HOST: "invented-seat",
            HYPRLAND_INSTANCE_SIGNATURE: "invented-compositor",
            SHALLOT_DISPLAY_SEAT: "",
        },
        stdout: "pipe",
        stderr: "pipe",
        detached: true,
    });
}

function groupMembers(group: number): number[] {
    const listing = Bun.spawnSync(["ps", "-axo", "pid=,pgid="], {
        stdout: "pipe",
        stderr: "pipe",
    });
    if (listing.exitCode !== 0) throw new Error(listing.stderr.toString());
    return listing.stdout
        .toString()
        .split("\n")
        .flatMap((line) => {
            const match = line.match(/^\s*(\d+)\s+(\d+)\s*$/);
            return match !== null && Number(match[2]) === group ? [Number(match[1])] : [];
        });
}

async function waitForFile(path: string, timeout = 10_000): Promise<void> {
    const deadline = Date.now() + timeout;
    while (!existsSync(path) && Date.now() < deadline) await Bun.sleep(20);
    if (!existsSync(path)) throw new Error(`fixture did not create ${path}`);
}

async function expectGroupGone(group: number): Promise<void> {
    const deadline = Date.now() + 15_000;
    let members = groupMembers(group);
    while (members.length > 0 && Date.now() < deadline) {
        await Bun.sleep(50);
        members = groupMembers(group);
    }
    expect(members).toEqual([]);
}

function killGroup(group: number): void {
    try {
        process.kill(-group, "SIGKILL");
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
    }
}

function processAlive(pid: number): boolean {
    try {
        process.kill(pid, 0);
        return true;
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ESRCH") return false;
        throw error;
    }
}

function outputOf(run: ReturnType<typeof runRunner>): string {
    return `${run.stdout.toString()}\n${run.stderr.toString()}`;
}

function reportPathOf(tree: string, output: string): string {
    const line = output.split("\n").find((entry) => entry.includes("report: "));
    const path = line?.match(/report: (.+)$/)?.[1];
    if (path === undefined) throw new Error(`report path missing from output: ${output}`);
    return resolve(tree, path.trim());
}

function reportOf(tree: string, output: string): string {
    return readFileSync(reportPathOf(tree, output), "utf8");
}

function lineOf(source: string, text: string): number {
    const line = source.split("\n").findIndex((entry) => entry.includes(text));
    if (line === -1) throw new Error(`fixture marker not found: ${text}`);
    return line + 1;
}

check(
    "runner source diagnostics",
    {
        claim: "the check runner preserves original assertion and exception source locations and values through redirected child output while undeclared files still refuse",
        size: "integration",
        subject: ["src/harness/preload.ts", "scripts/test-runner.ts"],
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-diagnostics-"));
        const tests = join(tree, "tests");
        const fixture = join(tests, "diagnostics.oracle.ts");
        const firstObserved = "observed-first";
        const firstExpected = "expected-first";
        const secondObserved = 17;
        const secondExpected = 23;
        const exceptionMessage = "ordinary fixture exception";
        const firstAssertion = `expect(${JSON.stringify(firstObserved)}).toBe(${JSON.stringify(firstExpected)})`;
        const secondAssertion = `expect(${secondObserved}).toBe(${secondExpected})`;
        const exceptionSource = `throw new Error(${JSON.stringify(exceptionMessage)})`;
        const source = `import { expect } from "bun:test";
import { check } from ${JSON.stringify(CHECK_MODULE)};

check("first assertion", { claim: "fixture assertion first", size: "integration" }, () => {
    console.error("redirected assertion diagnostic");
    ${firstAssertion};
});

check("second assertion", { claim: "fixture assertion second", size: "integration" }, () => {
    ${secondAssertion};
});

check("ordinary exception", { claim: "fixture ordinary exception", size: "integration" }, () => {
    ${exceptionSource};
});
`;
        try {
            mkdirSync(tests, { recursive: true });
            writeFileSync(
                join(tree, "bunfig.toml"),
                `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
            );
            writeFileSync(fixture, source);

            const first = runRunner(tree, "--oracle", "fixture assertion first");
            const firstOutput = `${first.stdout.toString()}\n${first.stderr.toString()}`;
            expect(first.exitCode).not.toBe(0);
            expect(firstOutput).toContain("redirected assertion diagnostic");
            expect(firstOutput).toContain(`Expected: ${JSON.stringify(firstExpected)}`);
            expect(firstOutput).toContain(`Received: ${JSON.stringify(firstObserved)}`);
            expect(firstOutput).toContain(`${fixture}:${lineOf(source, firstAssertion)}:`);

            const second = runRunner(tree, "--oracle", "fixture assertion second");
            const secondOutput = `${second.stdout.toString()}\n${second.stderr.toString()}`;
            expect(second.exitCode).not.toBe(0);
            expect(secondOutput).toContain(`Expected: ${secondExpected}`);
            expect(secondOutput).toContain(`Received: ${secondObserved}`);
            expect(secondOutput).toContain(`${fixture}:${lineOf(source, secondAssertion)}:`);
            expect(lineOf(source, firstAssertion)).not.toBe(lineOf(source, secondAssertion));

            const exceptionRun = runRunner(tree, "--oracle", "fixture ordinary exception");
            const exceptionOutput = `${exceptionRun.stdout.toString()}\n${exceptionRun.stderr.toString()}`;
            expect(exceptionRun.exitCode).not.toBe(0);
            expect(exceptionOutput).toContain(exceptionMessage);
            expect(exceptionOutput).toContain(`${fixture}:${lineOf(source, exceptionSource)}:`);

            const refusedTree = mkdtempSync(join(tmpdir(), "shallot-runner-undeclared-"));
            try {
                writeFileSync(
                    join(refusedTree, "bunfig.toml"),
                    `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
                );
                const refused = join(refusedTree, "naked.test.ts");
                writeFileSync(
                    refused,
                    `import { check } from ${JSON.stringify(CHECK_MODULE)};\n\nif (false) check("must refuse", { claim: "must refuse", size: "integration" }, () => {});\n`,
                );
                const refusal = runRunner(refusedTree, "--integration", "--all");
                expect(refusal.exitCode).not.toBe(0);
                expect(refusal.stderr.toString()).toContain(
                    "declares a check statically but registered none when it ran",
                );
            } finally {
                rmSync(refusedTree, { recursive: true, force: true });
            }
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "runner report agreement",
    {
        claim: "the check runner writes one retrievable JUnit report and owned child evidence whose totals and diagnostics agree with console outcomes",
        size: "integration",
        subject: ["scripts/test-runner.ts", "bunfig.toml", "CONTRIBUTING.md"],
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-report-agreement-"));
        const tests = join(tree, "tests");
        const head = `import { check } from ${JSON.stringify(CHECK_MODULE)};\n`;
        const writeCheck = (
            file: string,
            name: string,
            claim: string,
            subject: string,
            body: string,
            options = `size: "integration", subject: ${JSON.stringify(subject)}`,
        ) =>
            writeFileSync(
                join(tests, file),
                `${head}check(${JSON.stringify(name)}, { claim: ${JSON.stringify(claim)}, ${options} }, () => {\n${body}\n});\n`,
            );
        try {
            mkdirSync(tests, { recursive: true });
            writeFileSync(
                join(tree, "bunfig.toml"),
                `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
            );
            writeCheck(
                "a-report-pass.test.ts",
                "report pass one",
                "a report pass one",
                "src/report.ts",
                `    return { ok: true };`,
            );
            writeCheck(
                "b-report-pass.test.ts",
                "report pass two",
                "b report pass two",
                "src/report.ts",
                `    return { ok: true };`,
            );
            const green = runRunner(tree, "--integration", "--subject", "src/report");
            const greenOutput = outputOf(green);
            expect(green.exitCode).toBe(0);
            expect(greenOutput.trim()).toMatch(
                /^shallot test: 2 passed, 0 failed, 0 refused, 0 unrun; report: \.artifacts\/shallot-run-[^/]+\/junit\.xml$/,
            );
            expect(greenOutput).not.toContain("shallot verdict");
            expect(greenOutput).not.toContain("<testsuites");
            const greenReport = reportOf(tree, greenOutput);
            expect((greenReport.match(/<testsuites /g) ?? []).length).toBe(1);
            expect(greenReport).toContain('tests="2"');
            expect(greenReport).toContain('failures="0"');
            expect(greenReport).toContain('errors="0"');
            expect(greenReport).toContain('skipped="0"');
            expect(greenReport).toMatch(/<testcase name="a report pass one"[^>]*\/>/);
            expect(greenReport).toMatch(/<testcase name="b report pass two"[^>]*\/>/);
            expect((greenReport.match(/<testcase /g) ?? []).length).toBe(2);
            const greenEvidence = readFileSync(
                reportPathOf(tree, greenOutput).replace("junit.xml", "output.log"),
                "utf8",
            );
            expect(greenEvidence).toContain("=== a report pass one (pass) ===");
            expect(greenEvidence).toContain("=== b report pass two (pass) ===");

            writeCheck(
                "f-report-oracle.oracle.ts",
                "report oracle",
                "f report oracle",
                "src/report.ts",
                `    return { ok: true };`,
            );
            const oracle = runRunner(tree, "--oracle", "f report oracle");
            const oracleOutput = outputOf(oracle);
            expect(oracle.exitCode).toBe(0);
            expect(oracleOutput.trim()).toMatch(
                /^shallot test: 1 passed, 0 failed, 0 refused, 0 skipped; report: \.artifacts\/shallot-run-[^/]+\/junit\.xml$/,
            );
            expect(oracleOutput).not.toContain("shallot verdict");
            expect(oracleOutput).not.toContain("(pass)");
            expect(oracleOutput).not.toContain("Ran 1 test");

            writeCheck(
                "c-report-failure.test.ts",
                "report failure",
                "c report failure",
                "src/report-failure.ts",
                `    console.log("redirected report stdout");\n    console.error("redirected report stderr");\n    throw new Error("report failure detail");`,
            );
            const failure = runRunner(tree, "--integration", "--subject", "src/report-failure");
            const failureOutput = outputOf(failure);
            expect(failure.exitCode).not.toBe(0);
            expect(failureOutput).toContain("redirected report stdout");
            expect(failureOutput).toContain("redirected report stderr");
            expect(failureOutput).toContain("report:");
            expect(failureOutput).not.toContain("<testsuites");
            const failureReport = reportOf(tree, failureOutput);
            expect(failureReport).toContain('tests="1"');
            expect(failureReport).toContain('failures="1"');
            expect(failureReport).toContain("redirected report stdout");
            expect(failureReport).toContain("redirected report stderr");
            expect(failureReport).toContain("report failure detail");

            writeCheck(
                "d-report-refusal.test.ts",
                "report refusal",
                "d report refusal",
                "src/report-refusal.ts",
                `    return { ok: true };`,
                `size: "integration", subject: "src/report-refusal.ts", requires: ["display"]`,
            );
            const refusal = runRunner(tree, "--integration", "--subject", "src/report-refusal");
            const refusalOutput = outputOf(refusal);
            expect(refusal.exitCode).not.toBe(0);
            expect(refusalOutput).toContain("display seat unavailable");
            const refusalReport = reportOf(tree, refusalOutput);
            expect(refusalReport).toContain('tests="1"');
            expect(refusalReport).toContain('errors="1"');
            expect(refusalReport).toContain("display seat unavailable");

            writeCheck(
                "e-report-unit.test.ts",
                "report unit",
                "e report unit",
                "src/report-unit.ts",
                `    return { ok: true };`,
                `size: "unit"`,
            );
            const unit = runRunner(tree);
            const unitOutput = outputOf(unit);
            expect(unit.exitCode).toBe(0);
            expect(unitOutput).toContain("report:");
            expect(unitOutput).not.toContain("<testsuites");
            expect(reportOf(tree, unitOutput)).toContain("<testsuites");

            const unwritableTree = mkdtempSync(join(tmpdir(), "shallot-runner-report-unwritable-"));
            try {
                const unwritableTests = join(unwritableTree, "tests");
                mkdirSync(unwritableTests, { recursive: true });
                writeFileSync(
                    join(unwritableTree, "bunfig.toml"),
                    `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
                );
                writeFileSync(
                    join(unwritableTests, "one.test.ts"),
                    `${head}check("one", { claim: "unwritable one", size: "unit" }, () => {});\n`,
                );
                writeFileSync(join(unwritableTree, ".artifacts"), "not a directory");
                const unwritable = runRunner(unwritableTree);
                expect(unwritable.exitCode).not.toBe(0);
                expect(outputOf(unwritable)).toContain("report destination unavailable");
            } finally {
                rmSync(unwritableTree, { recursive: true, force: true });
            }
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "runner selection outcomes",
    {
        claim: "the check runner attempts every selected integration row, reports failure and refusal before later passes, preserves generic unrun report handling, and never runs an unselected failure",
        size: "integration",
        subject: ["scripts/test-runner.ts", "src/harness/check.ts"],
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-selection-outcomes-"));
        const tests = join(tree, "tests");
        const marker = join(tree, "unselected-body-ran");
        const head =
            `import { writeFileSync } from "node:fs";\n` +
            `import { check } from ${JSON.stringify(CHECK_MODULE)};\n`;
        const writeCheck = (
            file: string,
            name: string,
            claim: string,
            subject: string,
            body: string,
            requires = "",
        ) =>
            writeFileSync(
                join(tests, file),
                `${head}check(${JSON.stringify(name)}, { claim: ${JSON.stringify(claim)}, size: "integration", subject: ${JSON.stringify(subject)}${requires} }, () => {\n${body}\n});\n`,
            );
        try {
            mkdirSync(tests, { recursive: true });
            writeFileSync(
                join(tree, "bunfig.toml"),
                `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
            );
            writeCheck(
                "a-first-failure.test.ts",
                "first selected failure",
                "a first selected failure",
                "src/selected.ts",
                `    throw new Error("first selected failure");`,
            );
            writeCheck(
                "b-later-pass.test.ts",
                "later selected pass",
                "b later selected pass",
                "src/selected.ts",
                `    return { ok: true };`,
            );
            writeCheck(
                "c-registration-refusal.test.ts",
                "registration refusal",
                "c registration refusal",
                "src/registration.ts",
                `    return { ok: true };`,
                `, requires: ["display"]`,
            );
            writeCheck(
                "d-registration-pass.test.ts",
                "registration pass",
                "d registration pass",
                "src/registration.ts",
                `    return { ok: true };`,
            );
            writeCheck(
                "e-display-refusal.test.ts",
                "display refusal",
                "e display refusal",
                "src/display.ts",
                `    return { ok: true };`,
                `, requires: ["display"]`,
            );
            writeCheck(
                "f-unrun.test.ts",
                "fixture unrun",
                "f all unrun",
                "src/other.ts",
                `    console.log("shallot verdict {\\"claim\\":\\"f all unrun\\",\\"size\\":\\"integration\\",\\"result\\":\\"unrun\\",\\"reason\\":\\"fixture was not executed\\"}");\n    process.exit(0);`,
            );
            writeCheck(
                "g-mixed-pass.test.ts",
                "mixed pass",
                "g mixed pass",
                "src/mixed.ts",
                `    return { ok: true };`,
            );
            writeCheck(
                "h-unselected-failure.test.ts",
                "unselected failure",
                "h unselected failure",
                "src/unselected.ts",
                `    writeFileSync(${JSON.stringify(marker)}, "ran");\n    throw new Error("unselected body must not run");`,
            );
            // The first selected red used to terminate the carrier before it attempted its passing
            // sibling. The fixture's [test].preload is the production declaration path, not a stand-in.
            const firstRed = runRunner(tree, "--integration", "--subject", "src/selected");
            const firstRedOutput = outputOf(firstRed);
            expect(firstRed.exitCode).not.toBe(0);
            const firstRedReport = reportOf(tree, firstRedOutput);
            expect(firstRedReport).toContain('name="a first selected failure"');
            expect(firstRedReport).toContain('<failure type="failure"');
            expect(firstRedReport).toContain('name="b later selected pass"');
            expect(firstRedReport).toMatch(/<testcase name="b later selected pass"[^>]*\/>/);

            // Registration throws after emitting its refusal verdict. The later selected row must still
            // be attempted rather than being hidden by the process exit.
            const registration = runRunner(tree, "--integration", "--subject", "src/registration");
            const registrationOutput = outputOf(registration);
            expect(registration.exitCode).not.toBe(0);
            const registrationReport = reportOf(tree, registrationOutput);
            expect(registrationReport).toContain('name="c registration refusal"');
            expect(registrationReport).toContain('<error type="refused"');
            expect(registrationReport).toMatch(/<testcase name="d registration pass"[^>]*\/>/);

            const display = runRunner(tree, "--integration", "--subject", "src/display");
            const displayOutput = outputOf(display);
            expect(display.exitCode).not.toBe(0);
            const displayReport = reportOf(tree, displayOutput);
            expect(displayReport).toContain('name="e display refusal"');
            expect(displayReport).toContain('<error type="refused"');

            // A selected file can register a row, then fail while loading the rest of the module. That
            // is a failed selected row, not an unrun outcome; its child diagnostic must be visible, and a
            // later selected row must still be attempted.
            writeFileSync(
                join(tests, "j-registration-then-load-failure.test.ts"),
                `${head}check("load failure", { claim: "j registration then load failure", size: "integration", subject: "src/load.ts" }, () => {});\n` +
                    `console.log("shallot verdict {\\"claim\\":\\"j registration then load failure\\",\\"size\\":\\"integration\\",\\"result\\":\\"unrun\\",\\"reason\\":\\"fixture was not executed\\"}");\n` +
                    `await import("./missing-fixture-module");\n`,
            );
            writeCheck(
                "k-load-failure-continuation.test.ts",
                "load failure continuation",
                "k load failure continuation",
                "src/load.ts",
                `    return { ok: true };`,
            );
            const loadFailure = runRunner(tree, "--integration", "--subject", "src/load");
            const loadFailureOutput = outputOf(loadFailure);
            expect(loadFailure.exitCode).not.toBe(0);
            expect(loadFailureOutput).toContain("Cannot find module");
            expect(loadFailureOutput).toContain(
                'shallot verdict {"claim":"j registration then load failure","size":"integration","result":"unrun","reason":"fixture was not executed"}',
            );
            expect(loadFailureOutput).toContain(
                "selected integration: j registration then load failure (fail",
            );
            const loadFailureReport = reportOf(tree, loadFailureOutput);
            expect(loadFailureReport).toContain('tests="2"');
            expect(loadFailureReport).toContain('failures="1"');
            expect(loadFailureReport).toContain('skipped="0"');
            expect(loadFailureReport).toContain("Cannot find module");
            const loadFailureMessage = loadFailureReport.match(
                /<testcase name="j registration then load failure"[\s\S]*?<failure\b[^>]*\bmessage="([^"]*)"/,
            )?.[1];
            expect(loadFailureMessage).toBe("check failed");
            expect(loadFailureMessage).not.toContain("host premise is unavailable");
            expect(loadFailureReport).toMatch(
                /<testcase name="k load failure continuation"[^>]*\/>/,
            );

            // An all-unrun selection used to inherit Bun's successful skipped-test exit code. It is an
            // explicit no-row-ran failure, and the fixture supplies no product verdict.
            const allUnrun = runRunner(tree, "--integration", "--subject", "src/other");
            const allUnrunOutput = outputOf(allUnrun);
            expect(allUnrun.exitCode).not.toBe(0);
            const allUnrunReport = reportOf(tree, allUnrunOutput);
            expect(allUnrunReport).toContain('name="f all unrun"');
            expect(allUnrunReport).toContain("<skipped message=");
            expect(allUnrunOutput).toContain("no integration rows ran");
            expect(allUnrunReport).toContain("fixture was not executed");

            // The mixed selection includes the passing row and a generic unrun outcome by selecting the
            // shared subject prefix; one executed row still makes the selection attributable.
            writeCheck(
                "i-mixed-unrun.test.ts",
                "mixed unrun",
                "g mixed unrun",
                "src/mixed.ts",
                `    console.log("shallot verdict {\\"claim\\":\\"g mixed unrun\\",\\"size\\":\\"integration\\",\\"result\\":\\"unrun\\",\\"reason\\":\\"fixture was not executed\\"}");\n    process.exit(0);`,
            );
            const mixedWithUnrun = runRunner(tree, "--integration", "--subject", "src/mixed");
            const mixedWithUnrunOutput = outputOf(mixedWithUnrun);
            expect(mixedWithUnrun.exitCode).toBe(0);
            const mixedReport = reportOf(tree, mixedWithUnrunOutput);
            expect(mixedReport).toMatch(/<testcase name="g mixed pass"[^>]*\/>/);
            expect(mixedReport).toContain('name="g mixed unrun"');
            expect(mixedReport).toContain("<skipped message=");
            expect(mixedReport).toContain("fixture was not executed");
            expect(existsSync(marker)).toBe(false);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "runner child Git and temp isolation",
    {
        claim: "runner children isolate Git and temporary state, and each run replaces its prior report directory",
        size: "integration",
        subject: "scripts/test-runner.ts",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-git-isolation-"));
        const tests = join(tree, "tests");
        const artifacts = join(tree, ".artifacts");
        const globalConfig = join(tree, "host.gitconfig");
        const hostGit = join(tree, "host-git");
        const hostGitDirectory = join(hostGit, ".git");
        const hostIndex = join(hostGitDirectory, "index");
        const hostTemporary = mkdtempSync(join(tmpdir(), "shallot-host-tmp-"));
        try {
            mkdirSync(tests, { recursive: true });
            writeFileSync(join(tree, "shallot.json"), "{}\n");
            writeFileSync(globalConfig, "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n");
            writeFileSync(
                join(tests, "isolation.oracle.ts"),
                `import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { check } from ${JSON.stringify(CHECK_MODULE)};

check("isolated child", { claim: "isolated child environment", size: "integration" }, () => {
    const artifacts = resolve(import.meta.dir, "../.artifacts");
    expect(process.env.GIT_DIR).toBeUndefined();
    expect(process.env.GIT_INDEX_FILE).toBeUndefined();
    expect(process.env.GIT_CONFIG_GLOBAL).toBe("/dev/null");
    expect(process.env.GIT_CONFIG_NOSYSTEM).toBe("1");
    expect(process.env.GIT_CEILING_DIRECTORIES).toBe(artifacts);
    expect(process.env.TMPDIR).toBeDefined();
    expect(tmpdir()).toBe(process.env.TMPDIR);
    expect(process.env.TMPDIR?.startsWith(join(artifacts, "shallot-run-"))).toBe(true);

    const beforeInit = mkdtempSync(join(tmpdir(), "shallot-before-init-"));
    try {
        const discovery = Bun.spawnSync(["git", "rev-parse", "--show-toplevel"], {
            cwd: beforeInit,
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(discovery.exitCode).not.toBe(0);
    } finally {
        rmSync(beforeInit, { recursive: true, force: true });
    }

    const repository = mkdtempSync(join(tmpdir(), "shallot-no-signing-"));
    try {
        const git = (...args: string[]) => {
            const proc = Bun.spawnSync(["git", ...args], {
                cwd: repository,
                stdout: "pipe",
                stderr: "pipe",
            });
            expect(proc.exitCode).toBe(0);
        };
        git("init", "-q");
        git("config", "user.email", "surface@example.test");
        git("config", "user.name", "surface");
        writeFileSync(join(repository, "tracked"), "isolated\\n");
        git("add", ".");
        git("commit", "-qm", "unsigned fixture commit");
    } finally {
        rmSync(repository, { recursive: true, force: true });
    }
});
`,
            );
            mkdirSync(hostGit, { recursive: true });
            const init = Bun.spawnSync(["git", "-C", hostGit, "init", "-q"], {
                cwd: ROOT,
                stdout: "pipe",
                stderr: "pipe",
            });
            expect(init.exitCode).toBe(0);
            writeFileSync(join(hostGit, "tracked"), "host index\n");
            const add = Bun.spawnSync(["git", "-C", hostGit, "add", "tracked"], {
                cwd: ROOT,
                stdout: "pipe",
                stderr: "pipe",
            });
            expect(add.exitCode).toBe(0);
            const configBefore = readFileSync(join(hostGitDirectory, "config"));
            const indexBefore = readFileSync(hostIndex);
            const priorDirectory = join(artifacts, "shallot-run-stale");
            mkdirSync(priorDirectory, { recursive: true });
            writeFileSync(join(priorDirectory, "old.log"), "stale\n");

            const environment = {
                GIT_CONFIG_GLOBAL: globalConfig,
                GIT_DIR: hostGitDirectory,
                GIT_INDEX_FILE: hostIndex,
                TMPDIR: hostTemporary,
            };
            const first = runRunnerWithEnv(
                tree,
                environment,
                "--oracle",
                "isolated child environment",
            );
            if (first.exitCode !== 0) throw new Error(outputOf(first));
            const firstReport = reportPathOf(tree, outputOf(first));
            expect(readFileSync(join(hostGitDirectory, "config"))).toEqual(configBefore);
            expect(readFileSync(hostIndex)).toEqual(indexBefore);
            expect(readdirSync(hostTemporary)).toEqual([]);
            expect(readdirSync(artifacts)).toEqual([basename(dirname(firstReport))]);
            expect(readdirSync(dirname(firstReport)).sort()).toEqual(["junit.xml", "output.log"]);

            const second = runRunnerWithEnv(
                tree,
                environment,
                "--oracle",
                "isolated child environment",
            );
            expect(second.exitCode).toBe(0);
            const secondReport = reportPathOf(tree, outputOf(second));
            expect(secondReport).not.toBe(firstReport);
            expect(readdirSync(hostTemporary)).toEqual([]);
            expect(readdirSync(artifacts)).toEqual([basename(dirname(secondReport))]);
            expect(readdirSync(dirname(secondReport)).sort()).toEqual(["junit.xml", "output.log"]);
        } finally {
            rmSync(hostTemporary, { recursive: true, force: true });
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "runner child lifetime",
    {
        claim: "the check runner bounds each child, forwards interruption, and reds and removes leftover processes without losing output",
        size: "integration",
        subject: "scripts/test-runner.ts",
        budget: 20_000,
    },
    async () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-lifetime-"));
        const tests = join(tree, "tests");
        const unitGroup = join(tree, "unit-group.pid");
        const rowGroup = join(tree, "row-group.pid");
        const oracleGroup = join(tree, "oracle-group.pid");
        const leakGroup = join(tree, "leak-group.pid");
        const leakPidFile = join(tree, "leak.pid");
        const interruptGroup = join(tree, "interrupt-group.pid");
        const laterRowStarted = join(tree, "later-row-started");
        const grandchildReady = join(tree, "grandchild-ready");
        const groupIds: number[] = [];
        let interruptedRunner: Bun.Subprocess<"ignore", "pipe", "pipe"> | undefined;
        const head = `import { check } from ${JSON.stringify(CHECK_MODULE)};\nimport { existsSync, writeFileSync } from "node:fs";\n`;
        try {
            mkdirSync(tests, { recursive: true });
            writeFileSync(
                join(tree, "bunfig.toml"),
                `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
            );
            writeFileSync(
                join(tests, "a-unit-hang.test.ts"),
                `${head}check("unit fixture", { claim: "fixture unit sweep tripwire", size: "unit" }, () => {});\n` +
                    `console.log("unit fixture retained output");\nwriteFileSync(${JSON.stringify(unitGroup)}, String(process.pid));\nawait new Promise(() => {});\n`,
            );
            writeFileSync(
                join(tests, "row-tripwire.test.ts"),
                `${head}check("integration fixture", { claim: "fixture integration row tripwire", size: "integration", subject: "src/lifetime-tripwire.ts" }, () => ({ ok: true }));\n` +
                    `console.log("integration fixture retained output");\nwriteFileSync(${JSON.stringify(rowGroup)}, String(process.pid));\nawait new Promise(() => {});\n`,
            );
            writeFileSync(
                join(tests, "oracle-hang.oracle.ts"),
                `${head}check("oracle fixture", { claim: "fixture oracle tripwire", size: "integration" }, () => ({ ok: true }));\n` +
                    `console.log("oracle fixture retained output");\nwriteFileSync(${JSON.stringify(oracleGroup)}, String(process.pid));\nawait new Promise(() => {});\n`,
            );
            const grandchild = `process.on("SIGTERM", () => {}); require("node:fs").writeFileSync(${JSON.stringify(grandchildReady)}, "ready"); setInterval(() => {}, 1000);`;
            writeFileSync(
                join(tests, "grandchild.test.ts"),
                `${head}check("grandchild fixture", { claim: "fixture lingering grandchild", size: "integration", subject: "src/lifetime-grandchild.ts" }, async () => {\n` +
                    `    console.log("grandchild fixture retained output");\n` +
                    `    const child = Bun.spawn([process.execPath, "-e", ${JSON.stringify(grandchild)}], { stdout: "ignore", stderr: "ignore" });\n` +
                    `    const deadline = Date.now() + 5000;\n    while (!existsSync(${JSON.stringify(grandchildReady)}) && Date.now() < deadline) await Bun.sleep(10);\n` +
                    `    if (!existsSync(${JSON.stringify(grandchildReady)})) throw new Error("grandchild did not start");\n` +
                    `    writeFileSync(${JSON.stringify(leakGroup)}, String(process.pid));\n    writeFileSync(${JSON.stringify(leakPidFile)}, String(child.pid));\n});\n`,
            );
            writeFileSync(
                join(tests, "interrupted.test.ts"),
                `${head}check("interrupted fixture", { claim: "fixture interrupted first row", size: "integration", subject: "src/lifetime-interrupted.ts" }, async () => {\n` +
                    `    console.log("interrupted fixture retained output");\n    writeFileSync(${JSON.stringify(interruptGroup)}, String(process.pid));\n` +
                    `    await new Promise(() => {});\n});\n`,
            );
            writeFileSync(
                join(tests, "later-row.test.ts"),
                `${head}check("later fixture", { claim: "fixture later row not started after signal", size: "integration", subject: "src/lifetime-interrupted.ts" }, () => {\n` +
                    `    writeFileSync(${JSON.stringify(laterRowStarted)}, "started");\n    return { ok: true };\n});\n`,
            );

            const tripwireEnvironment = { SHALLOT_TEST_RUNNER_TRIPWIRE_MS: "1000" };
            const unit = runRunnerWithEnv(tree, tripwireEnvironment);
            const unitOutput = outputOf(unit);
            expect(unit.exitCode).not.toBe(0);
            expect(unitOutput).toContain("unit sweep exceeded its 1000 ms tripwire");
            expect(unitOutput).toContain("unit fixture retained output");
            const unitPid = Number(readFileSync(unitGroup, "utf8"));
            groupIds.push(unitPid);
            await expectGroupGone(unitPid);
            expect(reportOf(tree, unitOutput)).toContain(
                "unit sweep exceeded its 1000 ms tripwire",
            );
            expect(
                readFileSync(
                    reportPathOf(tree, unitOutput).replace("junit.xml", "output.log"),
                    "utf8",
                ),
            ).toContain("unit fixture retained output");

            const row = runRunnerWithEnv(
                tree,
                tripwireEnvironment,
                "--integration",
                "--subject",
                "src/lifetime-tripwire",
            );
            const rowOutput = outputOf(row);
            expect(row.exitCode).not.toBe(0);
            expect(rowOutput).toContain("integration row exceeded its 1000 ms tripwire");
            expect(rowOutput).toContain("integration fixture retained output");
            const rowPid = Number(readFileSync(rowGroup, "utf8"));
            groupIds.push(rowPid);
            await expectGroupGone(rowPid);
            const rowReport = reportOf(tree, rowOutput);
            expect(rowReport).toContain("integration row exceeded its 1000 ms tripwire");

            const oracle = runRunnerWithEnv(
                tree,
                tripwireEnvironment,
                "--oracle",
                "fixture oracle tripwire",
            );
            const oracleOutput = outputOf(oracle);
            expect(oracle.exitCode).not.toBe(0);
            expect(oracleOutput).toContain("oracle exceeded its 1000 ms tripwire");
            expect(oracleOutput).toContain("oracle fixture retained output");
            const oraclePid = Number(readFileSync(oracleGroup, "utf8"));
            groupIds.push(oraclePid);
            await expectGroupGone(oraclePid);
            expect(reportOf(tree, oracleOutput)).toContain("oracle exceeded its 1000 ms tripwire");
            expect(
                readFileSync(
                    reportPathOf(tree, oracleOutput).replace("junit.xml", "output.log"),
                    "utf8",
                ),
            ).toContain("oracle fixture retained output");

            const leak = runRunnerWithEnv(
                tree,
                {},
                "--integration",
                "--subject",
                "src/lifetime-grandchild",
            );
            const leakOutput = outputOf(leak);
            expect(leak.exitCode).not.toBe(0);
            const leakGroupPid = Number(readFileSync(leakGroup, "utf8"));
            const grandchildPid = Number(readFileSync(leakPidFile, "utf8"));
            groupIds.push(leakGroupPid);
            await expectGroupGone(leakGroupPid);
            const leakReport = reportOf(tree, leakOutput);
            expect(leakReport).toContain(`pid ${grandchildPid} (`);
            expect(leakReport).toContain("grandchild fixture retained output");
            expect(leakReport).toContain("&quot;result&quot;:&quot;pass&quot;");

            interruptedRunner = spawnRunnerWithEnv(
                tree,
                {},
                "--integration",
                "--subject",
                "src/lifetime-interrupted",
            );
            const interruptedStdout = new Response(interruptedRunner.stdout).text();
            const interruptedStderr = new Response(interruptedRunner.stderr).text();
            await waitForFile(interruptGroup);
            const interruptPid = Number(readFileSync(interruptGroup, "utf8"));
            groupIds.push(interruptPid);
            interruptedRunner.kill("SIGTERM");
            await interruptedRunner.exited;
            const interruptedOutput = (await interruptedStdout) + (await interruptedStderr);
            expect(interruptedRunner.signalCode).toBe("SIGTERM");
            expect(interruptedOutput).toContain("runner received SIGTERM during integration row");
            expect(interruptedOutput).toContain("interrupted fixture retained output");
            expect(existsSync(laterRowStarted)).toBe(false);
            await expectGroupGone(interruptPid);
            const interruptedReport = reportOf(tree, interruptedOutput);
            expect(interruptedReport).toContain("runner received SIGTERM during integration row");
            expect(interruptedReport).toContain(
                'name="fixture later row not started after signal"',
            );
            expect(interruptedReport).toContain("runner received SIGTERM before child start");
            expect(interruptedOutput).toContain("0 passed, 1 failed, 0 refused, 1 unrun");
            expect(
                readFileSync(
                    reportPathOf(tree, interruptedOutput).replace("junit.xml", "output.log"),
                    "utf8",
                ),
            ).toContain("interrupted fixture retained output");
        } finally {
            if (interruptedRunner !== undefined && interruptedRunner.exitCode === null)
                interruptedRunner.kill("SIGKILL");
            for (const group of groupIds) {
                if (groupMembers(group).length > 0) killGroup(group);
            }
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "runner tripwire writes a red JUnit report",
    {
        claim: "a runner tripwire replaces partial child JUnit with a well-formed report whose totals count the timeout and preserve output",
        size: "integration",
        subject: "scripts/test-runner.ts",
    },
    async () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-partial-junit-"));
        const tests = join(tree, "tests");
        const childPidFile = join(tree, "child.pid");
        let childPid: number | undefined;
        let runner: Bun.Subprocess<"ignore", "pipe", "pipe"> | undefined;
        try {
            mkdirSync(tests, { recursive: true });
            writeFileSync(
                join(tree, "bunfig.toml"),
                `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
            );
            const partialReport = `<?xml version="1.0"?><testsuites tests="2"><testsuite tests="2">`;
            writeFileSync(
                join(tests, "partial.test.ts"),
                `import { check } from ${JSON.stringify(CHECK_MODULE)};\n` +
                    `import { readdirSync, writeFileSync } from "node:fs";\n` +
                    `import { join } from "node:path";\n` +
                    `check("fixture", { claim: "fixture partial JUnit tripwire", size: "unit" }, () => {});\n` +
                    `console.log("partial JUnit fixture retained output");\n` +
                    `const artifacts = ${JSON.stringify(join(tree, ".artifacts"))};\n` +
                    `const run = readdirSync(artifacts).find((entry) => entry.startsWith("shallot-run-"));\n` +
                    `if (run === undefined) throw new Error("runner artifacts missing");\n` +
                    `writeFileSync(join(artifacts, run, "junit.xml"), ${JSON.stringify(partialReport)});\n` +
                    `writeFileSync(${JSON.stringify(childPidFile)}, String(process.pid));\n` +
                    `await new Promise(() => {});\n`,
            );
            runner = spawnRunnerWithEnv(tree, {
                SHALLOT_TEST_RUNNER_TRIPWIRE_MS: "1000",
            });
            const stdout = new Response(runner.stdout).text();
            const stderr = new Response(runner.stderr).text();
            await waitForFile(childPidFile);
            childPid = Number(readFileSync(childPidFile, "utf8"));
            const completed = await Promise.race([
                runner.exited.then(() => true),
                Bun.sleep(3000).then(() => false),
            ]);
            if (!completed) {
                runner.kill("SIGTERM");
                await runner.exited;
                killGroup(childPid);
            }
            const output = (await stdout) + (await stderr);
            expect(completed).toBe(true);
            expect(runner.exitCode).not.toBe(0);
            expect(output).toContain("unit sweep exceeded its 1000 ms tripwire");
            expect(output).toContain("partial JUnit fixture retained output");
            expect(output).toContain("shallot test: 0 passed, 1 failed, 0 refused, 0 skipped");
            const report = reportOf(tree, output);
            expect(report).toMatch(/<testsuites\b[^>]*\btests="1"[^>]*\bfailures="1"/);
            expect(report).toMatch(/<testsuite\b[^>]*\btests="1"[^>]*\bfailures="1"/);
            expect(report).toContain('<failure type="failure"');
            expect(report.trimEnd()).toEndWith("</testsuites>");
            expect(report).not.toContain('<testsuites tests="2">');
            expect(
                readFileSync(reportPathOf(tree, output).replace("junit.xml", "output.log"), "utf8"),
            ).toContain("partial JUnit fixture retained output");
        } finally {
            if (runner !== undefined && runner.exitCode === null) runner.kill("SIGKILL");
            if (childPid !== undefined && groupMembers(childPid).length > 0) killGroup(childPid);
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "runner teardown survives process-name lookup failure",
    {
        claim: "a process-name lookup failure cannot prevent the runner from escalating teardown to SIGKILL",
        size: "integration",
        subject: "scripts/test-runner.ts",
    },
    async () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-no-ps-"));
        const tests = join(tree, "tests");
        const bin = join(tree, "bin");
        const ps = join(bin, "ps");
        const ready = join(tree, "grandchild-ready");
        const groupFile = join(tree, "group.pid");
        const childFile = join(tree, "child.pid");
        const grandchild = `process.on("SIGTERM", () => {}); require("node:fs").writeFileSync(${JSON.stringify(ready)}, "ready"); setInterval(() => {}, 1000);`;
        let group: number | undefined;
        try {
            mkdirSync(tests, { recursive: true });
            mkdirSync(bin, { recursive: true });
            writeFileSync(
                join(tree, "bunfig.toml"),
                `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
            );
            writeFileSync(
                ps,
                "#!/bin/sh\nprintf 'ps: unsupported fixture lookup\\n' >&2\nexit 1\n",
            );
            chmodSync(ps, 0o755);
            writeFileSync(
                join(tests, "leak.test.ts"),
                `import { check } from ${JSON.stringify(CHECK_MODULE)};\n` +
                    `import { existsSync, writeFileSync } from "node:fs";\n` +
                    `check("leak", { claim: "fixture ps lookup failure", size: "integration", subject: "src/no-ps.ts" }, async () => {\n` +
                    `    console.log("ps failure fixture retained output");\n` +
                    `    const child = Bun.spawn([process.execPath, "-e", ${JSON.stringify(grandchild)}], { stdout: "ignore", stderr: "ignore" });\n` +
                    `    const deadline = Date.now() + 5000;\n    while (!existsSync(${JSON.stringify(ready)}) && Date.now() < deadline) await Bun.sleep(10);\n` +
                    `    if (!existsSync(${JSON.stringify(ready)})) throw new Error("grandchild did not start");\n` +
                    `    writeFileSync(${JSON.stringify(groupFile)}, String(process.pid));\n    writeFileSync(${JSON.stringify(childFile)}, String(child.pid));\n` +
                    `});\n`,
            );

            const result = runRunnerWithEnv(
                tree,
                { PATH: `${bin}:${process.env.PATH ?? ""}` },
                "--integration",
                "--subject",
                "src/no-ps",
            );
            const output = outputOf(result);
            group = Number(readFileSync(groupFile, "utf8"));
            const grandchildPid = Number(readFileSync(childFile, "utf8"));
            expect(result.exitCode).not.toBe(0);
            expect(output).toContain(`process remained after child exit: process group ${group}`);
            expect(reportOf(tree, output)).toContain(`process group ${group}`);
            expect(reportOf(tree, output)).toContain("ps failure fixture retained output");
            expect(groupMembers(group)).not.toContain(grandchildPid);
            await expectGroupGone(group);
        } finally {
            if (group !== undefined && groupMembers(group).length > 0) killGroup(group);
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "runner signal during final output keeps the default disposition",
    {
        claim: "a signal during final output terminates the runner by that signal instead of returning green",
        size: "integration",
        subject: "scripts/test-runner.ts",
        budget: 20_000,
    },
    async () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-final-signal-"));
        const tests = join(tree, "tests");
        const artifacts = join(tree, ".artifacts");
        const childPidFile = join(tree, "child.pid");
        let runner: Bun.Subprocess<"ignore", "pipe", "pipe"> | undefined;
        let childPid: number | undefined;
        try {
            mkdirSync(tests, { recursive: true });
            writeFileSync(
                join(tree, "bunfig.toml"),
                `[test]\npreload = [${JSON.stringify(PRELOAD)}]\n`,
            );
            writeFileSync(
                join(tests, "large-output.test.ts"),
                `import { check } from ${JSON.stringify(CHECK_MODULE)};\n` +
                    `import { writeFileSync } from "node:fs";\n` +
                    `check("large output", { claim: "fixture finalization signal row", size: "integration", subject: "src/final-signal.ts" }, async () => {\n` +
                    `    writeFileSync(${JSON.stringify(childPidFile)}, String(process.pid));\n` +
                    `    console.log("finalization output fixture");\n` +
                    `    await new Promise((resolve) => process.stdout.write("x".repeat(80_000_000) + "\\n", resolve));\n` +
                    `});\n`,
            );
            runner = spawnRunnerWithEnv(tree, {}, "--integration", "--subject", "src/final-signal");
            const stdout = new Response(runner.stdout).text();
            const stderr = new Response(runner.stderr).text();
            await waitForFile(childPidFile);
            childPid = Number(readFileSync(childPidFile, "utf8"));
            const deadline = Date.now() + 15_000;
            let interruptedDuringOutput = false;
            while (Date.now() < deadline && runner.exitCode === null) {
                const runDirectory = existsSync(artifacts)
                    ? readdirSync(artifacts).find((entry) => entry.startsWith("shallot-run-"))
                    : undefined;
                const reportFile =
                    runDirectory === undefined
                        ? undefined
                        : join(artifacts, runDirectory, "junit.xml");
                const outputPath =
                    runDirectory === undefined
                        ? undefined
                        : join(artifacts, runDirectory, "output.log");
                const rowFinished =
                    reportFile !== undefined &&
                    existsSync(reportFile) &&
                    readFileSync(reportFile, "utf8").includes(
                        'name="fixture finalization signal row"',
                    );
                if (
                    rowFinished &&
                    !processAlive(childPid) &&
                    outputPath !== undefined &&
                    existsSync(outputPath)
                ) {
                    runner.kill("SIGTERM");
                    interruptedDuringOutput = true;
                    break;
                }
                await Bun.sleep(1);
            }
            await runner.exited;
            await Promise.all([stdout, stderr]);
            expect(interruptedDuringOutput).toBe(true);
            expect(runner.signalCode).toBe("SIGTERM");
            const runDirectory = readdirSync(artifacts).find((entry) =>
                entry.startsWith("shallot-run-"),
            );
            expect(runDirectory).toBeDefined();
            const report = readFileSync(join(artifacts, runDirectory!, "junit.xml"), "utf8");
            expect(report).toMatch(/<testsuites\b[^>]*\btests="1"[^>]*\bfailures="0"/);
            expect(report).toContain('name="fixture finalization signal row"');
            await expectGroupGone(childPid);
        } finally {
            if (runner !== undefined && runner.exitCode === null) {
                runner.kill("SIGKILL");
                await runner.exited;
            }
            if (childPid !== undefined && groupMembers(childPid).length > 0) killGroup(childPid);
            rmSync(tree, { recursive: true, force: true });
        }
    },
);
