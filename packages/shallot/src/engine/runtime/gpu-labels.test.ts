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

    test("every shader and pipeline creation in the instrumented engine path is named", async () => {
        const root = resolve(import.meta.dir, "../../..");
        const failures: string[] = [];
        for await (const path of new Bun.Glob("src/**/*.ts").scan({ cwd: root })) {
            if (path.endsWith(".test.ts")) continue;
            const source = await Bun.file(resolve(root, path)).text();
            for (const failure of unnamedCreations(source)) {
                failures.push(`${path}:${failure.line} ${failure.method}`);
            }
        }
        expect(failures).toEqual([]);
    });
});
