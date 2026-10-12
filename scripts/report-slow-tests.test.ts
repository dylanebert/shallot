import { expect, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { slowTests, warningAnnotation } from "./report-slow-tests";

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

test("JUnit testcase time uses its filename tier and warns only above half the ceiling", () => {
    const slow = slowTests(report);
    expect(
        slow.map(({ name, tier, timeMs, ceiling }) => ({ name, tier, timeMs, ceiling })),
    ).toEqual([
        { name: "cheap & slow", tier: "cheap", timeMs: 126, ceiling: CEILING.cheap },
        { name: "GPU slow", tier: "gpu", timeMs: 501, ceiling: CEILING.gpu },
        { name: "Node slow", tier: "node", timeMs: 10_001, ceiling: CEILING.node },
    ]);
});

test("the warning names the test, host, ceiling and adapter printed for its file", () => {
    const log = `./src/render.gpu.test.ts:
frame-encoder adapter: {
  class: "fallback",
  identity: "llvmpipe",
}
(pass) GPU slow [501ms]`;
    const [slow] = slowTests(report, log).filter(({ name }) => name === "GPU slow");
    expect(slow).toBeDefined();
    expect(warningAnnotation(slow!, "Linux")).toBe(
        '::warning title=Slow test%3A GPU slow::test "GPU slow" took 501.00 ms; gpu ceiling 1000 ms; host Linux; adapter frame-encoder adapter: { class: "fallback", identity: "llvmpipe", }',
    );
});
