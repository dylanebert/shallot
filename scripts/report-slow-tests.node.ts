import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "./bun-child";

setDefaultTimeout(TIMEOUT.node);

test("a missing or malformed browser JUnit report fails instead of losing the warning", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-browser-report-"));
    const report = join(scratch, "report.xml");
    const json = join(scratch, "report.json");
    const output = join(scratch, "output");
    writeFileSync(json, '{"suites":[]}');
    try {
        for (const content of [
            undefined,
            "",
            "<testsuites></testsuites>",
            '<testsuites time="31"><testsuite></testsuites>',
            '<testsuites time="31">',
        ]) {
            if (content !== undefined) writeFileSync(report, content);
            const child = Bun.spawnSync({
                ...bunChild([
                    join(import.meta.dir, "report-slow-tests.ts"),
                    report,
                    "",
                    "examples/embedded-scene",
                    json,
                ]),
                stdout: "pipe",
                stderr: "pipe",
            });
            expect(child.exitCode, child.stderr.toString()).toBe(1);
        }
        writeFileSync(
            report,
            '<testsuites time="31"><testsuite time="1"><testcase name="page" time="1" /></testsuite></testsuites>',
        );
        const child = Bun.spawnSync({
            ...bunChild([
                join(import.meta.dir, "report-slow-tests.ts"),
                report,
                "",
                "examples/embedded-scene",
                json,
            ]),
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(child.exitCode, child.stderr.toString()).toBe(0);
        expect(child.stdout.toString()).toContain(
            'subject "examples/embedded-scene" took 31000.00 ms',
        );
        writeFileSync(output, "existing=value\n");
        const flaky = Bun.spawnSync({
            ...bunChild([
                join(import.meta.dir, "report-slow-tests.ts"),
                report,
                "",
                "examples/minimal",
                join(import.meta.dir, "browser-flaky.json"),
            ]),
            env: { ...process.env, GITHUB_OUTPUT: output },
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(flaky.exitCode, flaky.stderr.toString()).toBe(0);
        expect(flaky.stdout.toString()).toContain("failed then passed on retry");
        expect(readFileSync(output, "utf8")).toBe("existing=value\nflaky=true\n");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
});
