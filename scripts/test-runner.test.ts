import { expect } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";

const ROOT = resolve(import.meta.dir, "..");
const CHECK_MODULE = resolve(ROOT, "src/harness/check");
const PRELOAD = resolve(ROOT, "src/harness/preload.ts");

function runRunner(tree: string, ...args: string[]) {
    return Bun.spawnSync(
        ["bun", resolve(ROOT, "scripts/test-runner.ts"), "--root", tree, ...args],
        {
            cwd: ROOT,
            env: {
                ...process.env,
                SHALLOT_HOST: "invented-seat",
                HYPRLAND_INSTANCE_SIGNATURE: "invented-compositor",
                SHALLOT_DISPLAY_SEAT: "",
            },
            stdout: "pipe",
            stderr: "pipe",
        },
    );
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
                expect(refusal.stderr.toString()).toContain("registers no check()");
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
