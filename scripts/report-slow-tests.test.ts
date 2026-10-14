import { expect, test } from "bun:test";
import { flakyTests, flakyWarning, slowTests, warningAnnotation } from "./report-slow-tests";

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

const report = `<?xml version="1.0"?>
<testsuites>
  <testsuite name="cheap" time="99">
    <testcase name="cheap &amp; slow" file="src/cheap.test.ts" time="0.126" />
    <testcase name="cheap boundary" file="src/boundary.test.ts" time="0.125" />
  </testsuite>
  <testsuite name="GPU">
    <testcase name="GPU boundary" file="src/render.gpu.test.ts" time="0.5" />
    <testcase name="GPU slow" file="src/render.gpu.test.ts" time="0.501" />
  </testsuite>
  <testsuite name="Node">
    <testcase name="Node boundary" file="src/run.node.ts" time="10" />
    <testcase name="Node slow" file="src/run.node.ts" time="10.001" />
  </testsuite>
</testsuites>`;

test("JUnit testcase time warns only above half the speed budget, not the hang ceiling", () => {
    expect(
        slowTests(report).map(({ name, tier, timeMs, budget }) => ({ name, tier, timeMs, budget })),
    ).toEqual([
        { name: "cheap & slow", tier: "cheap", timeMs: 126, budget: 250 },
        { name: "GPU slow", tier: "gpu", timeMs: 501, budget: 1000 },
        { name: "Node slow", tier: "node", timeMs: 10_001, budget: 20_000 },
    ]);
});

test("Playwright reports the whole browser subject, including work outside testcases", () => {
    const report = (seconds: number) =>
        `<testsuites time="${seconds}"><testsuite time="1"><testcase name="page" time="1" /></testsuite></testsuites>`;
    expect(slowTests(report(30), "", "examples/embedded-scene")).toEqual([]);
    const slow = slowTests(report(30.001), "", "examples/embedded-scene");
    expect(slow).toEqual([
        {
            name: "examples/embedded-scene",
            file: "examples/embedded-scene",
            tier: "browser",
            timeMs: 30_001,
            budget: 60_000,
        },
    ]);
    expect(warningAnnotation(slow[0]!, "Linux")).toContain(
        'subject "examples/embedded-scene" took 30001.00 ms; over half its browser speed budget 60000 ms',
    );
});

test("the warning names the test, host, speed budget, adapter printed for its file and the remedy", () => {
    const log = `./src/render.gpu.test.ts:
frame-encoder adapter: {
  class: "fallback",
  identity: "llvmpipe",
}
(pass) GPU slow [501ms]`;
    const [slow] = slowTests(report, log).filter(({ name }) => name === "GPU slow");
    expect(slow).toBeDefined();
    expect(warningAnnotation(slow!, "Linux")).toBe(
        '::warning title=Slow test%3A GPU slow::test "GPU slow" took 501.00 ms; over half its gpu speed budget 1000 ms; host Linux; adapter frame-encoder adapter: { class: "fallback", identity: "llvmpipe", }; if it recurs on other runs, make it cheaper, split it or move it to a heavier tier; one slow runner needs nothing',
    );
});
