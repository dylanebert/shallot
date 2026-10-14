import { BUDGET } from "@dylanebert/shallot/testing";
import { normalizePath } from "../src/project/paths";

export type Tier = "cheap" | "gpu" | "node" | "browser";

export interface SlowTest {
    name: string;
    file: string;
    tier: Tier;
    timeMs: number;
    budget: number;
    adapter?: string;
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
        if (timeMs <= budget / 2) continue;
        slow.push({ name, file, tier, timeMs, budget, adapter: adapterForFile(log, file) });
    }
    return slow;
}

function escapeData(value: string): string {
    return value.replace(/%/g, "%25").replace(/\r/g, "%0D").replace(/\n/g, "%0A");
}

function escapeProperty(value: string): string {
    return escapeData(value).replace(/:/g, "%3A").replace(/,/g, "%2C");
}

export function warningAnnotation(test: SlowTest, host: string): string {
    const duration = `${test.timeMs.toFixed(2)} ms`;
    const message = [
        `${test.tier === "browser" ? "subject" : "test"} ${JSON.stringify(test.name)} took ${duration}`,
        `${test.tier} speed budget ${test.budget} ms`,
        `host ${host}`,
        ...(test.adapter ? [`adapter ${test.adapter}`] : []),
    ].join("; ");
    return `::warning title=${escapeProperty(`Slow test: ${test.name}`)}::${escapeData(message)}`;
}

if (import.meta.main) {
    const [reportPath, logPath, subject] = process.argv.slice(2);
    if (!reportPath) {
        console.error(
            "usage: bun scripts/report-slow-tests.ts <junit-report> [console-log] [browser-subject]",
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
                for (const test of slowTests(await report.text(), log, subject))
                    console.log(warningAnnotation(test, process.env.RUNNER_OS ?? "local"));
            }
        } catch (error) {
            console.error(`[slow-test-report] could not read report: ${String(error)}`);
            if (subject !== undefined) process.exitCode = 1;
        }
    }
}
