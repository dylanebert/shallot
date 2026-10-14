import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "./bun-child";

setDefaultTimeout(TIMEOUT.node);

test("a missing or malformed browser JUnit report fails instead of losing the warning", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-browser-report-"));
    const report = join(scratch, "report.xml");
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
            ]),
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(child.exitCode, child.stderr.toString()).toBe(0);
        expect(child.stdout.toString()).toContain(
            'subject "examples/embedded-scene" took 31000.00 ms',
        );
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
});
