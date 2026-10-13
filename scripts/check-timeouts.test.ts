import { expect, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { browserCommandErrors, packageTimeoutErrors, timeoutErrors } from "./timeout-policy";

const cheap =
    'import { test } from "bun:test"; import { CEILING } from "@dylanebert/shallot/testing";';
const header = (tier: "gpu" | "node") =>
    `import { test, setDefaultTimeout } from "bun:test"; import { CEILING } from "@dylanebert/shallot/testing"; setDefaultTimeout(CEILING.${tier});`;
const browser = 'import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";';

test("cheap test fixtures that start a process require the Node tier", () => {
    const revertedPinScan = `${cheap} Bun.spawnSync(["git", "ls-files"]);`;
    expect(timeoutErrors("scripts/check-pins.test.ts", revertedPinScan)).toEqual([
        "scripts/check-pins.test.ts:1: cheap tests must not start processes; move this file to the Node tier",
    ]);
    expect(timeoutErrors("fixture.test.ts", `${cheap} Bun.$\`git status\`;`)).toHaveLength(1);
    expect(
        timeoutErrors(
            "fixture.test.ts",
            'import { spawn as start } from "node:child_process"; start("git");',
        ),
    ).toHaveLength(1);
    expect(
        timeoutErrors(
            "fixture.test.ts",
            'import * as childProcess from "node:child_process"; childProcess.execSync("git status");',
        ),
    ).toHaveLength(1);
    expect(
        timeoutErrors(
            "fixture.test.ts",
            'const { spawnSync: start } = require("node:child_process"); start("git", ["status"]);',
        ),
    ).toHaveLength(1);
    expect(
        timeoutErrors("fixture.gpu.test.ts", `${header("gpu")} Bun.spawnSync(["git"]);`),
    ).toEqual([]);
    expect(timeoutErrors("fixture.node.ts", `${header("node")} Bun.spawnSync(["git"]);`)).toEqual(
        [],
    );
});

test("tier headers derive from the published tooling subpath and reject literal declarations", () => {
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
    expect(
        timeoutErrors(
            "playwright.config.ts",
            config(
                "webServer: [{ ...WEB_SERVER_CONFIG, command: 'vite preview' }, { ...WEB_SERVER_CONFIG, command: 'vite' }]",
            ),
        ),
    ).toEqual([]);
    for (const properties of [
        `globalTimeout: ${CEILING.browser}`,
        `timeout: ${CEILING.node}`,
        `timeout: ${CEILING.startup}`,
        `webServer: { timeout: ${CEILING.startup} }`,
        `webServer: [{ ...WEB_SERVER_CONFIG, command: "vite" }, { timeout: ${CEILING.startup} }]`,
        "workers: 2",
        "...BROWSER_CONFIG",
    ]) {
        expect(timeoutErrors("playwright.config.ts", config(properties)).length).toBeGreaterThan(0);
    }
    expect(timeoutErrors("playwright.config.ts", "export default {};")).toHaveLength(1);
    expect(
        timeoutErrors(
            "playwright.config.ts",
            'import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "./chromium"; export default { ...BROWSER_CONFIG, webServer: { ...WEB_SERVER_CONFIG, command: "vite" } };',
        ),
    ).not.toEqual([]);
    expect(
        timeoutErrors(
            "src/project/testing.js",
            'const CEILING = { browser: 60_000, startup: 120_000 }; export const BROWSER_CONFIG = { testMatch: "**/*.e2e.ts" }; export const WEB_SERVER_CONFIG = { timeout: CEILING.startup };',
        ),
    ).toEqual([]);
    expect(
        timeoutErrors(
            "src/project/testing.js",
            "const CEILING = { browser: 60_000, startup: 120_000 }; export const BROWSER_CONFIG = { globalTimeout: CEILING.browser }; export const WEB_SERVER_CONFIG = { timeout: CEILING.startup };",
        ),
    ).toContain(
        "src/project/testing.js:1: shared globalTimeout belongs to scripts/browser.sh or Playwright's defaults",
    );
});

test("browser tests leave Playwright's test timeout at its default", () => {
    const source = 'import { test } from "playwright/test";';
    expect(timeoutErrors("subject.e2e.ts", `${source} test("browser", () => {});`)).toEqual([]);
    expect(
        timeoutErrors("subject.e2e.ts", `${source} test.setTimeout(${CEILING.startup});`),
    ).toHaveLength(1);
});

test("the browser runner pins the global and worker budgets", () => {
    const command = `bunx playwright test -c "$subject" --global-timeout=${CEILING.browser} --workers=1`;
    expect(browserCommandErrors(command)).toEqual([]);
    expect(browserCommandErrors(command.replace(String(CEILING.browser), "30000"))).toContain(
        "scripts/browser.sh: --global-timeout must equal CEILING.browser",
    );
    expect(browserCommandErrors(command.replace("--workers=1", "--workers=2"))).toContain(
        "scripts/browser.sh: each subject must run with --workers=1",
    );
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
