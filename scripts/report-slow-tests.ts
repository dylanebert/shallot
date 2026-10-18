import { appendFileSync } from "node:fs";
import { BUDGET } from "@dylanebert/shallot/testing";
import type { JSONReport, JSONReportSuite } from "playwright/types/testReporter";
import { normalizePath } from "../src/project/paths";

export type Tier = "cheap" | "gpu" | "node" | "browser";

export interface SlowTest {
    name: string;
    file: string;
    tier: Tier;
    timeMs: number;
    budget: number;
    adapter?: string;
    subject?: string;
    /** Time in a file outside its tests: module load and hooks such as `beforeAll`. */
    setup?: true;
}

function decodeXml(value: string): string {
    return value.replace(/&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos);/gi, (entity, code: string) => {
        if (code[0] === "#") {
            const point =
                code[1]?.toLowerCase() === "x"
                    ? Number.parseInt(code.slice(2), 16)
                    : Number(code.slice(1));
            return String.fromCodePoint(point);
        }
        return { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" }[code.toLowerCase()] ?? entity;
    });
}

function attributes(tag: string): Map<string, string> {
    return new Map(
        [...tag.matchAll(/([\w:.-]+)\s*=\s*"([^"]*)"/g)].map((match) => [
            match[1]!,
            decodeXml(match[2]!),
        ]),
    );
}

function tierForFile(file: string): Tier {
    return file.endsWith(".gpu.test.ts") ? "gpu" : file.endsWith(".node.ts") ? "node" : "cheap";
}

function normalized(path: string): string {
    return normalizePath(path)
        .replace(/^\.\//, "")
        .replace(/\/{2,}/g, "/");
}

function sameFile(left: string, right: string): boolean {
    const a = normalized(left);
    const b = normalized(right);
    return a === b || a.endsWith(`/${b}`) || b.endsWith(`/${a}`);
}

function adapterForFile(log: string, file: string): string | undefined {
    const lines = log.split(/\r?\n/);
    const adapters: string[] = [];
    let sectionFile: string | undefined;
    let section: string[] = [];
    const inspect = (): void => {
        if (!sectionFile || !sameFile(sectionFile, file)) return;
        for (let index = 0; index < section.length; index++) {
            if (!/\badapter\b/i.test(section[index]!)) continue;
            const detail: string[] = [];
            for (const line of section.slice(index, index + 8)) {
                if (/^\((?:pass|fail|skip)\)/.test(line.trim())) break;
                detail.push(line.trim());
            }
            const text = detail.join(" ").replace(/\s+/g, " ").trim();
            if (text) adapters.push(text.slice(0, 240));
        }
    };

    for (const line of lines) {
        const header = line.match(/^(.+\.(?:gpu\.test\.ts|node\.ts|test\.ts)):\s*$/);
        if (header) {
            inspect();
            sectionFile = header[1];
            section = [];
        } else if (sectionFile) {
            section.push(line);
        }
    }
    inspect();
    return [...new Set(adapters)].join("; ") || undefined;
}

function completeXml(report: string): boolean {
    const stack: string[] = [];
    let roots = 0;
    const tags =
        /<!\[CDATA\[[\s\S]*?\]\]>|<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<\/?([\w:.-]+)(?:\s+[\w:.-]+\s*=\s*(?:"[^"]*"|'[^']*'))*\s*\/?>/g;
    let end = 0;
    for (const match of report.matchAll(tags)) {
        const text = report.slice(end, match.index);
        if (text.includes("<") || (stack.length === 0 && text.trim())) return false;
        end = match.index + match[0].length;
        const name = match[1];
        if (!name) continue;
        if (match[0].startsWith("</")) {
            if (stack.pop() !== name) return false;
        } else {
            if (stack.length === 0 && (++roots !== 1 || name !== "testsuites")) return false;
            if (!match[0].endsWith("/>")) stack.push(name);
        }
    }
    return roots === 1 && stack.length === 0 && !report.slice(end).trim();
}

export function slowTests(report: string, log = "", subject?: string): SlowTest[] {
    const slow: SlowTest[] = [];
    if (subject !== undefined) {
        const root = report.match(/<testsuites\b[^>]*>/);
        const duration = root ? attributes(root[0]).get("time") : undefined;
        const seconds = duration?.trim() ? Number(duration) : NaN;
        if (!completeXml(report) || !Number.isFinite(seconds) || seconds < 0)
            throw new Error("invalid JUnit browser subject duration");
        const timeMs = seconds * 1000;
        if (timeMs > BUDGET.browser / 2)
            slow.push({
                name: subject,
                file: subject,
                tier: "browser",
                timeMs,
                budget: BUDGET.browser,
            });
        return slow;
    }
    const testTime = new Map<string, number>();
    for (const match of report.matchAll(/<testcase\b[^>]*>/g)) {
        const attrs = attributes(match[0]);
        const name = attrs.get("name");
        const file = attrs.get("file");
        const seconds = Number(attrs.get("time"));
        if (name === undefined || file === undefined || !Number.isFinite(seconds) || seconds < 0)
            throw new Error(`invalid JUnit testcase: ${match[0]}`);

        const tier = tierForFile(file);
        const budget = BUDGET[tier];
        const timeMs = seconds * 1000;
        testTime.set(file, (testTime.get(file) ?? 0) + timeMs);
        if (timeMs <= budget / 2) continue;
        slow.push({ name, file, tier, timeMs, budget, adapter: adapterForFile(log, file) });
    }
    // A file's suite, the outermost and so the longest of its suites, also times what runs outside its tests.
    // Only the Node tier warns on it: that time includes module load, and the first file to load the engine pays it
    // for the rest, which alone passes half the cheap and GPU budgets.
    const suiteTime = new Map<string, number>();
    for (const match of report.matchAll(/<testsuite\b[^>]*>/g)) {
        const attrs = attributes(match[0]);
        const file = attrs.get("file");
        if (file === undefined || tierForFile(file) !== "node") continue;
        const seconds = Number(attrs.get("time"));
        if (!Number.isFinite(seconds) || seconds < 0)
            throw new Error(`invalid JUnit testsuite: ${match[0]}`);
        suiteTime.set(file, Math.max(suiteTime.get(file) ?? 0, seconds * 1000));
    }
    for (const [file, suiteMs] of suiteTime) {
        const timeMs = suiteMs - (testTime.get(file) ?? 0);
        if (timeMs <= BUDGET.node / 2) continue;
        slow.push({ name: file, file, tier: "node", timeMs, budget: BUDGET.node, setup: true });
    }
    return slow;
}

export function slowBrowserTests(report: string, subject: string): SlowTest[] {
    const json: JSONReport = JSON.parse(report);
    const slow: SlowTest[] = [];
    const visit = (suite: JSONReportSuite, parents: string[]): void => {
        const titles = [...parents, suite.title];
        for (const spec of suite.specs) {
            for (const test of spec.tests) {
                const project = json.config.projects.find(({ id }) => id === test.projectId);
                if (!project || !Number.isFinite(project.timeout) || project.timeout < 0)
                    throw new Error(`invalid Playwright project timeout: ${test.projectId}`);
                const timeMs = Math.max(0, ...test.results.map(({ duration }) => duration));
                if (timeMs <= project.timeout / 2) continue;
                slow.push({
                    name: [
                        ...titles,
                        spec.title,
                        ...(test.projectName ? [test.projectName] : []),
                    ].join(" › "),
                    file: spec.file,
                    tier: "browser",
                    timeMs,
                    budget: project.timeout,
                    subject,
                });
            }
        }
        for (const child of suite.suites ?? []) visit(child, titles);
    };
    for (const suite of json.suites) visit(suite, []);
    return slow;
}

export function flakyTests(report: string): string[] {
    const json: JSONReport = JSON.parse(report);
    const flaky: string[] = [];
    const visit = (suite: JSONReportSuite, parents: string[]): void => {
        const titles = [...parents, suite.title];
        for (const spec of suite.specs) {
            for (const test of spec.tests) {
                if (test.status !== "flaky") continue;
                flaky.push(
                    [...titles, spec.title, ...(test.projectName ? [test.projectName] : [])].join(
                        " › ",
                    ),
                );
            }
        }
        for (const child of suite.suites ?? []) visit(child, titles);
    };
    for (const suite of json.suites) visit(suite, []);
    return flaky;
}

export function flakyWarning(test: string, subject: string, host: string): string {
    return `::warning title=${escapeProperty(`Flaky test: ${test}`)}::${escapeData(`subject ${JSON.stringify(subject)}; test ${JSON.stringify(test)} failed then passed on retry; host ${host}`)}`;
}

function escapeData(value: string): string {
    return value.replace(/%/g, "%25").replace(/\r/g, "%0D").replace(/\n/g, "%0A");
}

function escapeProperty(value: string): string {
    return escapeData(value).replace(/:/g, "%3A").replace(/,/g, "%2C");
}

/** CONTRIBUTING, Verification: what a warning asks of whoever reads it. */
const REMEDY =
    "if it recurs on other runs, make it cheaper, split it or move it to a heavier tier; one slow runner needs nothing";

export function warningAnnotation(test: SlowTest, host: string): string {
    const duration = `${test.timeMs.toFixed(2)} ms`;
    const message = [
        ...(test.subject ? [`subject ${JSON.stringify(test.subject)}`] : []),
        test.setup
            ? `setup of ${JSON.stringify(test.name)} took ${duration} outside its tests`
            : `${test.tier === "browser" && !test.subject ? "subject" : "test"} ${JSON.stringify(test.name)} took ${duration}`,
        test.subject
            ? `over half its Playwright project timeout ${test.budget} ms`
            : `over half its ${test.tier} speed budget ${test.budget} ms`,
        `host ${host}`,
        ...(test.adapter ? [`adapter ${test.adapter}`] : []),
        REMEDY,
    ].join("; ");
    return `::warning title=${escapeProperty(`Slow ${test.setup ? "setup" : "test"}: ${test.name}`)}::${escapeData(message)}`;
}

if (import.meta.main) {
    const [reportPath, logPath, subject, jsonPath] = process.argv.slice(2);
    if (!reportPath) {
        console.error(
            "usage: bun scripts/report-slow-tests.ts <junit-report> [console-log] [browser-subject] [playwright-json-report]",
        );
    } else {
        try {
            const report = Bun.file(reportPath);
            if (!(await report.exists())) {
                throw new Error(`no JUnit report at ${reportPath}`);
            } else {
                const log =
                    logPath && (await Bun.file(logPath).exists())
                        ? await Bun.file(logPath).text()
                        : "";
                const xml = await report.text();
                const host = process.env.RUNNER_OS ?? "local";
                for (const test of slowTests(xml, log, subject))
                    console.log(warningAnnotation(test, host));
                if (subject !== undefined) {
                    if (!jsonPath) throw new Error("missing Playwright JSON report");
                    const json = await Bun.file(jsonPath).text();
                    for (const test of slowBrowserTests(json, subject))
                        console.log(warningAnnotation(test, host));
                    const flaky = flakyTests(json);
                    for (const test of flaky) console.log(flakyWarning(test, subject, host));
                    if (flaky.length && process.env.GITHUB_OUTPUT)
                        appendFileSync(process.env.GITHUB_OUTPUT, "flaky=true\n");
                }
            }
        } catch (error) {
            console.error(`[slow-test-report] could not read report: ${String(error)}`);
            if (subject !== undefined) process.exitCode = 1;
        }
    }
}
