import { expect, test } from "bun:test";
import { BUDGET, CEILING } from "@dylanebert/shallot/testing";
import {
    flakyTests,
    flakyWarning,
    slowBrowserTests,
    slowTests,
    warningAnnotation,
} from "./report-slow-tests";

test("recorded Playwright JSON warns for a flake with trace off and XML-like stdout", async () => {
    const json = await Bun.file(new URL("./browser-flaky.json", import.meta.url)).text();
    expect(json).toContain("<error>diagnostic text</error>");
    expect(json).toContain('"attachments": []');
    expect(flakyTests(json)).toEqual(["retry-spike.e2e.ts › nested › retry spike"]);
    expect(flakyWarning(flakyTests(json)[0]!, "examples/minimal", "Windows")).toBe(
        '::warning title=Flaky test%3A retry-spike.e2e.ts › nested › retry spike::subject "examples/minimal"; test "retry-spike.e2e.ts › nested › retry spike" failed then passed on retry; host Windows',
    );
    for (const status of ["expected", "unexpected", "skipped"])
        expect(flakyTests(json.replace('"status": "flaky"', `"status": "${status}"`))).toEqual([]);
});

test("recorded Playwright durations warn only above half the matching project's timeout", async () => {
    const recorded = await Bun.file(new URL("./browser-flaky.json", import.meta.url)).text();
    // The recorded retry's longest attempt is 229 ms; its test timeout stays 30 s.
    const report = recorded.replace('"timeout": 30000', '"timeout": 456');
    const slow = slowBrowserTests(report, "examples/minimal");
    expect(slow).toEqual([
        {
            name: "retry-spike.e2e.ts › nested › retry spike",
            file: "retry-spike.e2e.ts",
            tier: "browser",
            timeMs: 229,
            budget: 456,
            subject: "examples/minimal",
        },
    ]);
    expect(warningAnnotation(slow[0]!, "Linux")).toBe(
        '::warning title=Slow test%3A retry-spike.e2e.ts › nested › retry spike::subject "examples/minimal"; test "retry-spike.e2e.ts › nested › retry spike" took 229.00 ms; over half its Playwright project timeout 456 ms; host Linux; approaching the CI hang ceiling: make it cheaper, split it or move it to a heavier tier',
    );
    expect(
        slowBrowserTests(report.replace('"timeout": 456', '"timeout": 458'), "examples/minimal"),
    ).toEqual([]);
});

const report = `<?xml version="1.0"?>
<testsuites>
  <testsuite name="cheap" time="99">
    <testcase name="cheap &amp; slow" file="src/cheap.test.ts" time="1.251" />
    <testcase name="cheap boundary" file="src/boundary.test.ts" time="1.25" />
  </testsuite>
  <testsuite name="GPU">
    <testcase name="GPU boundary" file="src/render.gpu.test.ts" time="5" />
    <testcase name="GPU slow" file="src/render.gpu.test.ts" time="5.001" />
  </testsuite>
  <testsuite name="Node">
    <testcase name="Node boundary" file="src/run.node.ts" time="60" />
    <testcase name="Node slow" file="src/run.node.ts" time="60.001" />
  </testsuite>
</testsuites>`;

test("JUnit testcase time warns only above half the CI ceiling in each tier", () => {
    for (const [tier, file] of [
        ["cheap", "src/cheap.test.ts"],
        ["gpu", "src/render.gpu.test.ts"],
        ["node", "src/run.node.ts"],
    ] as const) {
        const report = (ms: number) =>
            `<testsuites><testcase name="test" file="${file}" time="${ms / 1000}" /></testsuites>`;
        expect(slowTests(report(BUDGET[tier] * 0.6))).toEqual([]);
        expect(slowTests(report(CEILING[tier] / 2))).toEqual([]);
        expect(slowTests(report(CEILING[tier] / 2 + 1))).toHaveLength(1);
    }
    expect(
        slowTests(report).map(({ name, tier, timeMs, budget }) => ({ name, tier, timeMs, budget })),
    ).toEqual([
        { name: "cheap & slow", tier: "cheap", timeMs: 1251, budget: 2500 },
        { name: "GPU slow", tier: "gpu", timeMs: 5001, budget: 10_000 },
        { name: "Node slow", tier: "node", timeMs: 60_001, budget: 120_000 },
    ]);
});

test("Playwright reports the whole browser subject, including work outside testcases", () => {
    const report = (seconds: number) =>
        `<testsuites time="${seconds}"><testsuite time="1"><testcase name="page" time="1" /></testsuite></testsuites>`;
    expect(slowTests(report((BUDGET.browser * 0.6) / 1000), "", "examples/embedded-scene")).toEqual(
        [],
    );
    expect(slowTests(report(CEILING.browser / 2000), "", "examples/embedded-scene")).toEqual([]);
    const slow = slowTests(report((CEILING.browser / 2 + 1) / 1000), "", "examples/embedded-scene");
    expect(slow).toEqual([
        {
            name: "examples/embedded-scene",
            file: "examples/embedded-scene",
            tier: "browser",
            timeMs: 150_001,
            budget: CEILING.browser,
        },
    ]);
    expect(warningAnnotation(slow[0]!, "Linux")).toContain(
        'subject "examples/embedded-scene" took 150001.00 ms; over half its browser CI ceiling 300000 ms',
    );
});

test("the warning names the test, host, CI ceiling, adapter printed for its file and the remedy", () => {
    const log = `./src/render.gpu.test.ts:
frame-encoder adapter: {
  class: "fallback",
  identity: "llvmpipe",
}
(pass) GPU slow [5001ms]`;
    const [slow] = slowTests(report, log).filter(({ name }) => name === "GPU slow");
    expect(slow).toBeDefined();
    expect(warningAnnotation(slow!, "Linux")).toBe(
        '::warning title=Slow test%3A GPU slow::test "GPU slow" took 5001.00 ms; over half its gpu CI ceiling 10000 ms; host Linux; adapter frame-encoder adapter: { class: "fallback", identity: "llvmpipe", }; approaching the CI hang ceiling: make it cheaper, split it or move it to a heavier tier',
    );
});
