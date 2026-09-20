import { expect } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";

const ROOT = resolve(import.meta.dir, "..");
const CHECK_MODULE = resolve(ROOT, "src/harness/check");
const PRELOAD = resolve(ROOT, "src/harness/preload.ts");

function runRunner(tree: string, ...args: string[]) {
    return Bun.spawnSync(
        ["bun", resolve(ROOT, "scripts/test-runner.ts"), "--root", tree, ...args],
        { cwd: ROOT, stdout: "pipe", stderr: "pipe" },
    );
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
