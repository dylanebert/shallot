import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";

const CREATION =
    /\.\s*(createShaderModule|createComputePipeline(?:Async)?|createRenderPipeline(?:Async)?|createGuardedComputePipeline)\s*\(/g;

function maskTrivia(source: string): string {
    const chars = [...source];
    let i = 0;
    while (i < chars.length) {
        const quote = chars[i];
        if (quote === "/" && chars[i + 1] === "/") {
            chars[i++] = " ";
            chars[i++] = " ";
            while (i < chars.length && chars[i] !== "\n") chars[i++] = " ";
            continue;
        }
        if (quote === "/" && chars[i + 1] === "*") {
            chars[i++] = " ";
            chars[i++] = " ";
            while (i < chars.length) {
                if (chars[i] === "*" && chars[i + 1] === "/") {
                    chars[i++] = " ";
                    chars[i++] = " ";
                    break;
                }
                if (chars[i] !== "\n") chars[i] = " ";
                i++;
            }
            continue;
        }
        // regex literal: /pattern/flags — must be masked before the string
        // check, because a quote or slash inside a regex would otherwise start
        // a string or comment and blind the rest of the file
        if (quote === "/" && chars[i + 1] !== "/" && chars[i + 1] !== "*") {
            // a / is a regex (not division) when it follows an operator or
            // punctuation, not an operand
            let j = i - 1;
            while (j >= 0 && /\s/.test(chars[j])) j--;
            const prev = j >= 0 ? chars[j] : "";
            if (prev === "" || /[=(,:[!&|^~+\-*%<>?{;]/.test(prev)) {
                chars[i++] = " ";
                let inClass = false;
                while (i < chars.length) {
                    if (chars[i] === "\\") {
                        chars[i++] = " ";
                        if (i < chars.length && chars[i] !== "\n") chars[i] = " ";
                        i++;
                        continue;
                    }
                    if (chars[i] === "[" && !inClass) {
                        inClass = true;
                        chars[i++] = " ";
                        continue;
                    }
                    if (chars[i] === "]" && inClass) {
                        inClass = false;
                        chars[i++] = " ";
                        continue;
                    }
                    if (chars[i] === "/" && !inClass) {
                        chars[i++] = " ";
                        while (i < chars.length && /[gimsuy]/.test(chars[i])) chars[i++] = " ";
                        break;
                    }
                    if (chars[i] === "\n") break;
                    chars[i++] = " ";
                }
                continue;
            }
        }
        if (quote === '"' || quote === "'" || quote === "`") {
            const end = quote;
            chars[i++] = " ";
            while (i < chars.length) {
                if (chars[i] === "\\") {
                    chars[i++] = " ";
                    if (i < chars.length && chars[i] !== "\n") chars[i] = " ";
                    i++;
                    continue;
                }
                if (chars[i] === end) {
                    chars[i++] = " ";
                    break;
                }
                if (chars[i] !== "\n") chars[i] = " ";
                i++;
            }
            continue;
        }
        i++;
    }
    return chars.join("");
}

function closingParen(source: string, open: number): number {
    let depth = 0;
    for (let i = open; i < source.length; i++) {
        if (source[i] === "(") depth++;
        if (source[i] === ")" && --depth === 0) return i;
    }
    return -1;
}

function hasTopLevelLabel(argument: string): boolean {
    const start = argument.search(/\S/);
    if (start < 0 || argument[start] !== "{") return false;
    let depth = 0;
    for (let i = start; i < argument.length; i++) {
        if (argument[i] === "{") depth++;
        else if (argument[i] === "}") depth--;
        else if (depth === 1 && argument.startsWith("label", i)) {
            const before = argument[i - 1];
            const after = argument[i + 5];
            if (!/[\w$]/.test(before ?? "") && !/[\w$]/.test(after ?? "")) {
                let colon = i + 5;
                while (/\s/.test(argument[colon] ?? "")) colon++;
                if ([":", ",", "}"].includes(argument[colon])) return true;
            }
        }
    }
    return false;
}

function unnamedCreations(source: string): { line: number; method: string }[] {
    const masked = maskTrivia(source);
    const failures: { line: number; method: string }[] = [];
    for (const match of masked.matchAll(CREATION)) {
        const open = match.index + match[0].lastIndexOf("(");
        const close = closingParen(masked, open);
        if (close < 0) continue;
        const argument = masked.slice(open + 1, close);
        const namedTypeGpu = /^\s*\)*\s*\.\s*\$name\s*\(/.test(masked.slice(close + 1));
        if (!hasTopLevelLabel(argument) && !namedTypeGpu) {
            failures.push({
                line: source.slice(0, match.index).split("\n").length,
                method: match[1],
            });
        }
    }
    return failures;
}

describe("GPU diagnostic labels", () => {
    test("distinguishes descriptor labels from nested labels and TypeGPU names", () => {
        expect(unnamedCreations(`device.createShaderModule({ label, code })`)).toEqual([]);
        expect(unnamedCreations(`root.createComputePipeline({ compute }).$name("named")`)).toEqual(
            [],
        );
        expect(
            unnamedCreations(
                `device.createComputePipeline({ compute: { module: device.createShaderModule({ label: "module", code }) } })`,
            ),
        ).toEqual([{ line: 1, method: "createComputePipeline" }]);
    });

    test("maskTrivia masks regex literals so a quote inside one does not blind the rest of the file", () => {
        // a regex literal containing a " would start a string under the old
        // maskTrivia, blinding every creation after it; the regex fix masks the
        // literal first so the unnamed createShaderModule below is still detected
        expect(
            unnamedCreations(`const re = /foo"bar/g;\n` + `device.createShaderModule({ code });`),
        ).toEqual([{ line: 2, method: "createShaderModule" }]);
    });

    test("every shader and pipeline creation in the instrumented engine path is named", async () => {
        const root = resolve(import.meta.dir, "../../..");
        const failures: string[] = [];
        let scannedCount = 0;
        let excludedTestCount = 0;
        for await (const path of new Bun.Glob("src/**/*.ts").scan({ cwd: root })) {
            if (path.endsWith(".test.ts")) {
                excludedTestCount++;
                continue;
            }
            scannedCount++;
            const source = await Bun.file(resolve(root, path)).text();
            for (const failure of unnamedCreations(source)) {
                failures.push(`${path}:${failure.line} ${failure.method}`);
            }
        }
        // non-vacuity floor: a wrong root or matchless glob scans zero files
        // and still reads green. derive the minimum from the tree's tracked
        // source declarations — independent of the scan root so a broken root
        // reds the floor rather than emptying both sides
        const declaredRoot = resolve(import.meta.dir, "../../..");
        const proc = Bun.spawnSync({
            cmd: ["git", "ls-files", "--", "src/"],
            cwd: declaredRoot,
            stdout: "pipe",
        });
        const trackedFiles = proc.stdout.toString().trim().split("\n").filter(Boolean);
        const trackedNonTestTs = trackedFiles.filter(
            (p) => p.endsWith(".ts") && !p.endsWith(".test.ts"),
        );
        const trackedTestTs = trackedFiles.filter((p) => p.endsWith(".test.ts"));
        expect(scannedCount).toBeGreaterThanOrEqual(trackedNonTestTs.length);
        // exclusion extent: the .test.ts skip drops this many tracked test files
        expect(excludedTestCount).toBeGreaterThanOrEqual(trackedTestTs.length);
        expect(failures).toEqual([]);
    });
});
