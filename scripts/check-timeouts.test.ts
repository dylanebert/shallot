import { expect, test } from "bun:test";
import { packageTimeoutErrors, timeoutErrors } from "./check-timeouts";
import { CEILING } from "./test-tiers";

const cheap = 'import { test } from "bun:test"; import { CEILING } from "./test-tiers";';
const header = (tier: "gpu" | "node") =>
    `import { test, setDefaultTimeout } from "bun:test"; import { CEILING } from "./test-tiers"; setDefaultTimeout(CEILING.${tier});`;
const browser = 'import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "./chromium";';

test("tier headers derive from the shared home and reject today's literal declarations", () => {
    for (const tier of ["gpu", "node"] as const) {
        const path = tier === "gpu" ? "subject.gpu.test.ts" : "subject.node.ts";
        expect(timeoutErrors(path, header(tier))).toEqual([]);
        expect(
            timeoutErrors(path, header(tier).replace(`CEILING.${tier}`, String(CEILING[tier]))),
        ).toHaveLength(2);
        expect(timeoutErrors(path, cheap)).toHaveLength(1);
        expect(
            timeoutErrors(path, `${header(tier)} setDefaultTimeout(CEILING.${tier});`),
        ).toHaveLength(1);
    }
});

test("per-test literal timeouts and restated file defaults are rejected", () => {
    for (const [path, source, tier] of [
        ["subject.test.ts", cheap, "cheap"],
        ["subject.gpu.test.ts", header("gpu"), "gpu"],
        ["subject.node.ts", header("node"), "node"],
    ] as const) {
        expect(
            timeoutErrors(path, `${source} test("literal", () => {}, ${CEILING[tier]});`),
        ).toHaveLength(1);
        expect(
            timeoutErrors(path, `${source} test("redundant", () => {}, CEILING.${tier});`),
        ).toHaveLength(1);
        expect(
            timeoutErrors(
                path,
                `${source} test("options", () => {}, { timeout: CEILING.${tier} });`,
            ),
        ).toHaveLength(1);
        expect(
            timeoutErrors(
                path,
                `${source} const ownBudget = ${CEILING[tier]}; test("alias", () => {}, ownBudget);`,
            ),
        ).toHaveLength(1);
    }
    expect(
        timeoutErrors(
            "subject.gpu.test.ts",
            `${header("gpu")} test("oversized", () => {}, CEILING.node);`,
        ),
    ).toHaveLength(1);
});

test("browser subjects inherit categorical values from their one shared home", () => {
    const config = (properties: string) =>
        `${browser} export default { ...BROWSER_CONFIG, ${properties} };`;
    expect(timeoutErrors("playwright.config.ts", config(""))).toEqual([]);
    expect(
        timeoutErrors(
            "playwright.config.ts",
            config("webServer: { ...WEB_SERVER_CONFIG, command: 'vite' }"),
        ),
    ).toEqual([]);
    for (const properties of [
        `globalTimeout: ${CEILING.browser}`,
        `timeout: ${CEILING.node}`,
        `timeout: ${CEILING.startup}`,
        `webServer: { timeout: ${CEILING.startup} }`,
        "workers: 2",
        "...BROWSER_CONFIG",
    ]) {
        expect(timeoutErrors("playwright.config.ts", config(properties)).length).toBeGreaterThan(0);
    }
    expect(timeoutErrors("playwright.config.ts", "export default {};")).toHaveLength(1);
    expect(
        timeoutErrors(
            "scripts/chromium.ts",
            'import { CEILING } from "./test-tiers"; export const config = { globalTimeout: CEILING.browser, webServer: { timeout: CEILING.startup } };',
        ),
    ).toEqual([]);
});

test("browser tests leave Playwright's test timeout at its default", () => {
    const source = 'import { test } from "playwright/test";';
    expect(timeoutErrors("subject.e2e.ts", `${source} test("browser", () => {});`)).toEqual([]);
    expect(
        timeoutErrors("subject.e2e.ts", `${source} test.setTimeout(${CEILING.startup});`),
    ).toHaveLength(1);
});

test("package test timeout follows the cheap tier and manual oracles remain unbounded", () => {
    const manifest = (timeout: number) =>
        JSON.stringify({ scripts: { test: `bun test --timeout=${timeout}` } });
    expect(packageTimeoutErrors(manifest(CEILING.cheap))).toEqual([]);
    expect(packageTimeoutErrors(manifest(CEILING.gpu))).toHaveLength(1);
    expect(packageTimeoutErrors(JSON.stringify({ scripts: { test: "bun test" } }))).toHaveLength(1);
    expect(
        timeoutErrors("subject.oracle.ts", `${cheap} test("manual", () => {}, CEILING.startup);`),
    ).toEqual([]);
});
