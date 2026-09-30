import { expect, test } from "bun:test";
import { timeoutErrors } from "./check-timeouts";

const cheap = 'import { test } from "bun:test";';
const gpu = 'import { test, setDefaultTimeout } from "bun:test"; setDefaultTimeout(1000);';
const node = 'import { test, setDefaultTimeout } from "bun:test"; setDefaultTimeout(20_000);';

test("timeout ceilings reject oversized cases, options and drifted defaults in every tier", () => {
    expect(timeoutErrors("subject.test.ts", `${cheap} test("cheap", () => {}, 251);`)).toHaveLength(
        1,
    );
    expect(
        timeoutErrors("subject.gpu.test.ts", `${gpu} test("GPU", () => {}, { timeout: 2000 });`),
    ).toHaveLength(1);
    expect(
        timeoutErrors("subject.node.ts", `${node} test("Node", () => {}, 90_000_000);`),
    ).toHaveLength(1);
    expect(timeoutErrors("subject.gpu.test.ts", gpu.replace("1000", "120_000"))).toHaveLength(2);
    expect(timeoutErrors("subject.gpu.test.ts", cheap)).toHaveLength(1);
    expect(
        timeoutErrors(
            "subject.node.ts",
            `${node} const budget = 20_001; test("Node", () => {}, budget);`,
        ),
    ).toHaveLength(1);
    expect(
        timeoutErrors("subject.test.ts", `${cheap} test("unknown", () => {}, getBudget());`),
    ).toHaveLength(1);
});

test("browser subjects use one categorical global backstop rather than measured tailoring", () => {
    const config = (properties: string) => `export default { ${properties} };`;
    expect(timeoutErrors("playwright.config.ts", config("globalTimeout: 60_000"))).toEqual([]);
    for (const properties of [
        "",
        "globalTimeout: 6_000",
        "globalTimeout: 60_001",
        "globalTimeout: measuredRun()",
        "globalTimeout: 60_000, globalTimeout: 60_000",
    ]) {
        expect(timeoutErrors("playwright.config.ts", config(properties)).length).toBeGreaterThan(0);
    }
});

test("timeout ceilings accept lower constant budgets and leave manual oracles unbounded", () => {
    expect(timeoutErrors("subject.test.ts", `${cheap} test("cheap", () => {}, 250);`)).toEqual([]);
    expect(timeoutErrors("subject.gpu.test.ts", `${gpu} test("GPU", () => {}, 100);`)).toEqual([]);
    expect(
        timeoutErrors("subject.node.ts", `${node} test("Node", () => {}, { timeout: 20_000 });`),
    ).toEqual([]);
    expect(
        timeoutErrors("subject.oracle.ts", `${cheap} test("manual", () => {}, 90_000_000);`),
    ).toEqual([]);
});
