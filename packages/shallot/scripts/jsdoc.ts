import { existsSync, readFileSync } from "node:fs";

/**
 * Source + JSDoc reading, shared by the docs build (`docs.ts`) and the field-doc parser (consumed by the
 * FIELDS table + the editor's annotation-sourced inspector). Pure over a file path — `docs.ts` runs the
 * whole build at import (top-level await), so anything a test or `fields.ts` needs lives here, importable
 * without triggering it.
 */

const fileCache = new Map<string, string[]>();

export function getLines(filePath: string): string[] {
    if (!fileCache.has(filePath)) {
        fileCache.set(filePath, readFileSync(filePath, "utf-8").split("\n"));
    }
    return fileCache.get(filePath)!;
}

export function findDefinitionLine(filePath: string, name: string): number {
    if (!existsSync(filePath)) return 1;
    const lines = getLines(filePath);
    const exportRe = new RegExp(
        `\\bexport\\s+(const|let|function|async\\s+function|class|enum|interface|type)\\s+${name}\\b`,
    );
    const localRe = new RegExp(
        `\\b(const|let|function|async\\s+function|class|enum|interface|type)\\s+${name}\\b`,
    );
    const reExportRe = new RegExp(`export\\s*(type\\s+)?\\{[^}]*\\b${name}\\b`);

    // prefer exported definitions over local ones
    let localMatch = -1;
    for (let i = 0; i < lines.length; i++) {
        const l = lines[i];
        if (exportRe.test(l)) return i + 1;
        if (reExportRe.test(l)) return i + 1;
        if (localMatch === -1 && localRe.test(l)) localMatch = i + 1;
    }
    return localMatch !== -1 ? localMatch : 1;
}

export interface JSDocResult {
    description: string | null;
    tags: string[];
    example: string | null;
    params: string | null;
}

export function parseJSDoc(filePath: string, defLine: number): JSDocResult {
    const lines = getLines(filePath);
    if (defLine <= 1) return { description: null, tags: [], example: null, params: null };

    let end = -1;
    for (let i = defLine - 2; i >= 0; i--) {
        const trimmed = lines[i].trim();
        if (trimmed === "") continue;
        if (trimmed.endsWith("*/")) {
            end = i;
            break;
        }
        break;
    }
    if (end === -1) return { description: null, tags: [], example: null, params: null };

    let start = -1;
    for (let i = end; i >= 0; i--) {
        if (lines[i].trimStart().startsWith("/**")) {
            start = i;
            break;
        }
    }
    if (start === -1) return { description: null, tags: [], example: null, params: null };

    // collect cleaned lines
    const cleaned: string[] = [];
    for (let i = start; i <= end; i++) {
        let line = lines[i].trim();
        line = line
            .replace(/^\/\*\*\s*/, "")
            .replace(/\s*\*\/\s*$/, "")
            .replace(/^\*\s?/, "");
        cleaned.push(line);
    }

    const tags: string[] = [];
    let example: string | null = null;
    let params: string | null = null;
    let inExample = false;
    const exampleLines: string[] = [];
    // the description is the leading paragraph — every line before the first blank line or first tag,
    // joined into one string. Taking only the first physical line truncated multi-line summaries
    // mid-clause ("override via" then nothing).
    const descLines: string[] = [];
    let descDone = false;

    for (const line of cleaned) {
        if (line.startsWith("@example")) {
            inExample = true;
            descDone = true;
            continue;
        }
        if (inExample) {
            if (line.startsWith("@")) {
                inExample = false;
                // fall through to tag handling below
            } else {
                exampleLines.push(line);
                continue;
            }
        }
        if (line.startsWith("@params ")) {
            params = line.slice(8).trim();
            continue;
        }
        if (line.startsWith("@")) {
            tags.push(line);
            descDone = true;
            continue;
        }
        if (descDone) continue;
        if (line === "") {
            if (descLines.length) descDone = true;
            continue;
        }
        descLines.push(line);
    }

    const description: string | null = descLines.length ? descLines.join(" ") : null;

    if (exampleLines.length > 0) {
        while (exampleLines.length > 0 && exampleLines[0] === "") exampleLines.shift();
        while (exampleLines.length > 0 && exampleLines[exampleLines.length - 1] === "")
            exampleLines.pop();
        if (exampleLines.length > 0) example = exampleLines.join("\n");
    }

    return { description, tags, example, params };
}

/**
 * a callable's parameter names as a compact `a, b, c` string, read from its signature at `lineNum`.
 * Splits on top-level commas only, so a param whose type holds commas (an inline object type, a tuple
 * `[number, number, number]`, a generic) stays one param; strips each param's type / default, keeps
 * `...rest`. Empty string for a no-arg call, null when the signature can't be read.
 *
 * @example
 * paramNames("src/extras/gltf/assets.ts", 416, "placeGltf") // "state, handle, opts"
 */
export function paramNames(filePath: string, lineNum: number, name: string): string | null {
    if (!existsSync(filePath)) return null;
    const lines = getLines(filePath);
    if (lineNum <= 0 || lineNum > lines.length) return null;
    // join lines until closing paren (handles multi-line signatures)
    let sig = "";
    for (let i = lineNum - 1; i < Math.min(lines.length, lineNum + 10); i++) {
        sig += ` ${lines[i]}`;
        if (sig.includes(")")) break;
    }
    const m = sig.match(new RegExp(`\\b${name}\\??\\s*(?:<[^>]*>)?\\s*\\(([^)]*)\\)`));
    if (!m) return null;
    const raw = m[1].trim();
    if (!raw) return "";
    const parts: string[] = [];
    let depth = 0;
    let cur = "";
    for (const ch of raw) {
        if (ch === "{" || ch === "[" || ch === "(" || ch === "<") depth++;
        else if (ch === "}" || ch === "]" || ch === ")" || ch === ">") depth--;
        if (ch === "," && depth === 0) {
            parts.push(cur);
            cur = "";
        } else cur += ch;
    }
    parts.push(cur);
    return parts
        .map((p) =>
            p
                .trim()
                .replace(/\s*[:=].*$/s, "")
                .replace(/^\.\.\.\s*/, "..."),
        )
        .filter(Boolean)
        .join(", ");
}

/**
 * per-field descriptions for a component, parsed from the doc comment above each field in its
 * `export const Name = { field: ... }` declaration. Keyed by the raw declared field name (camelCase) —
 * the consumer kebab-cases to match the inspector / FIELDS labels. A field with no doc comment is absent.
 * This is the annotation-sourced half of the UI reference: meaning lives once, beside the field, and feeds
 * the inspector hover, the docs reference, and IDE hover from the one comment.
 *
 * @example
 * parseComponentFields("src/extras/orbit/index.ts", "Orbit") // { yaw: "horizontal orbit angle…", … }
 */
export function parseComponentFields(filePath: string, name: string): Record<string, string> {
    if (!existsSync(filePath)) return {};
    const lines = getLines(filePath);
    const defLine = findDefinitionLine(filePath, name);
    // 1 is findDefinitionLine's not-found sentinel; a real component sits below its imports, never line 1
    if (defLine <= 1) return {};

    const out: Record<string, string> = {};
    let braceDepth = 0;
    let started = false;

    for (let i = defLine - 1; i < lines.length; i++) {
        const line = lines[i];
        for (const ch of line) {
            if (ch === "{") {
                braceDepth++;
                started = true;
            }
            if (ch === "}") braceDepth--;
        }
        if (started && braceDepth === 0) break;
        if (braceDepth !== 1) continue;

        const trimmed = line.trim();
        if (trimmed.startsWith("/") || trimmed.startsWith("*") || trimmed === "") continue;

        // a field line: `name: sparse(...)` / `name: slab(...)` — the storage factory follows the colon
        const fieldMatch = trimmed.match(/^(\w+)\s*:/);
        if (fieldMatch) {
            const doc = parseJSDoc(filePath, i + 1);
            if (doc.description) out[fieldMatch[1]] = doc.description;
        }
    }

    return out;
}

/** the balanced `{ … }` object literal of a top-level `export const name = { … }`, or "" if not found. */
export function objectLiteral(filePath: string, name: string): string {
    const lines = getLines(filePath);
    const defLine = findDefinitionLine(filePath, name);
    if (defLine <= 0) return "";
    let text = "";
    let depth = 0;
    let started = false;
    for (let i = defLine - 1; i < lines.length; i++) {
        for (const ch of lines[i]) {
            if (ch === "{") {
                depth++;
                started = true;
            }
            if (started) text += ch;
            if (ch === "}") {
                depth--;
                if (depth === 0) return text;
            }
        }
        if (started) text += "\n";
    }
    return text;
}

/**
 * whether the `export const <name>` declaration ends in `as const` — the enum marker. It sits on the
 * definition line for a single-line enum (`export const M = { A: 0 } as const;`) or on the object
 * literal's closing line for a multi-line one (`} as const;`), so inspecting only the def line misreads
 * a multi-line enum as a plain value.
 */
export function isAsConst(filePath: string, name: string): boolean {
    const defLine = findDefinitionLine(filePath, name);
    if (defLine <= 1) return false; // 1 is the not-found / missing-file sentinel; a real definition sits below imports
    const lines = getLines(filePath);
    if (/\bas const\b/.test(lines[defLine - 1] ?? "")) return true;
    let depth = 0;
    let started = false;
    for (let i = defLine - 1; i < lines.length; i++) {
        const line = lines[i] ?? "";
        for (const ch of line) {
            if (ch === "{") {
                depth++;
                started = true;
            } else if (ch === "}") depth--;
        }
        if (!started) return false; // not an object-literal export — the def line was the whole declaration
        if (depth === 0) return /\bas const\b/.test(line); // the closing `}` line carries the marker
    }
    return false;
}

/** an `as const` enum's `[option, value]` pairs, parsed from its object-literal text. */
export function enumOptions(text: string): [string, string][] {
    const out: [string, string][] = [];
    for (const m of text.matchAll(/(\w+)\s*:\s*(-?\d+|"[^"]*"|'[^']*')/g)) out.push([m[1], m[2]]);
    return out;
}

/** a plugin's bundled component / system / dependency identifiers, parsed from its object-literal text. */
export function pluginRefs(text: string): {
    components: string[];
    systems: string[];
    dependencies: string[];
} {
    const ids = (re: RegExp): string[] => {
        const m = text.match(re);
        return m
            ? m[1]
                  .split(",")
                  .map((s) => s.trim().split(":")[0].trim())
                  .filter(Boolean)
            : [];
    };
    return {
        components: ids(/\bcomponents\s*:\s*\{([^}]*)\}/),
        systems: ids(/\bsystems\s*:\s*\[([^\]]*)\]/),
        dependencies: ids(/\bdependencies\s*:\s*\[([^\]]*)\]/),
    };
}

export interface MemberExport {
    name: string;
    kind: "method" | "property" | "field";
    line: number;
    description: string | null;
    returnType?: string;
    example?: string | null;
    params?: string;
}

/**
 * documented members of a `@expand` class — getters as properties, methods with their param names.
 * interfaces are parsed by {@link parseInterfaceFields} instead (a class parse skips their `readonly`
 * property rows), so bail on an interface definition of the same name.
 */
export function parseClassMembers(filePath: string, className: string): MemberExport[] {
    if (!existsSync(filePath)) return [];
    const lines = getLines(filePath);
    const defLine = findDefinitionLine(filePath, className);
    if (defLine <= 0) return [];
    if (/\binterface\s/.test(lines[defLine - 1] ?? "")) return [];

    const members: MemberExport[] = [];
    let braceDepth = 0;
    let inClass = false;

    for (let i = defLine - 1; i < lines.length; i++) {
        const line = lines[i];
        const depthBefore = braceDepth;

        for (const ch of line) {
            if (ch === "{") {
                braceDepth++;
                inClass = true;
            }
            if (ch === "}") braceDepth--;
        }
        if (inClass && braceDepth === 0) break;
        if (depthBefore !== 1) continue;

        const trimmed = line.trim();
        if (trimmed.startsWith("private") || trimmed.startsWith("readonly")) continue;
        if (trimmed.startsWith("constructor")) continue;
        if (trimmed.startsWith("/") || trimmed.startsWith("*") || trimmed === "") continue;

        // getter → property, extract return type
        const getterMatch = trimmed.match(/^get\s+(\w+)\s*\(\s*\)\s*:\s*(.+?)\s*\{/);
        if (getterMatch) {
            const lineNum = i + 1;
            const doc = parseJSDoc(filePath, lineNum);
            const rawType = getterMatch[2];
            const bareType = rawType.replace(/^Readonly<(.+)>$/, "$1").replace(/\[\]$/, "");
            members.push({
                name: getterMatch[1],
                kind: "property",
                line: lineNum,
                description: doc.description,
                returnType: bareType,
                example: doc.example,
            });
            continue;
        }

        // skip setters (paired with getter)
        if (trimmed.startsWith("set ")) continue;

        // method
        const methodMatch = trimmed.match(/^(\w+)\s*[(<]/);
        if (methodMatch) {
            const lineNum = i + 1;
            const doc = parseJSDoc(filePath, lineNum);
            const params = paramNames(filePath, lineNum, methodMatch[1]);
            members.push({
                name: methodMatch[1],
                kind: "method",
                line: lineNum,
                description: doc.description,
                example: doc.example,
                params: params ?? "",
            });
        }
    }

    return members;
}

/**
 * documented members of a `@expand` interface — its property fields and its method signatures. A method
 * (`name(params): type`) reads as a call in the reference; a plain field as a property.
 */
export function parseInterfaceFields(filePath: string, name: string): MemberExport[] {
    if (!existsSync(filePath)) return [];
    const lines = getLines(filePath);
    // find the interface specifically (not a const/class with the same name)
    const interfaceRe = new RegExp(`\\binterface\\s+${name}\\b`);
    let defLine = -1;
    for (let i = 0; i < lines.length; i++) {
        if (interfaceRe.test(lines[i])) {
            defLine = i + 1;
            break;
        }
    }
    if (defLine <= 0) return [];

    const fields: MemberExport[] = [];
    let braceDepth = 0;
    let started = false;

    for (let i = defLine - 1; i < lines.length; i++) {
        const line = lines[i];
        for (const ch of line) {
            if (ch === "{") {
                braceDepth++;
                started = true;
            }
            if (ch === "}") braceDepth--;
        }
        if (started && braceDepth === 0) break;
        if (braceDepth !== 1) continue;

        const trimmed = line.trim();
        if (trimmed.startsWith("/") || trimmed.startsWith("*") || trimmed === "") continue;
        if (trimmed === "{") continue;

        // method signature: name(params): type — `?` marks an optional method (`error?(...)`)
        const methodMatch = trimmed.match(/^(\w+)\??\s*(?:<[^>]*>)?\s*\(/);
        if (methodMatch) {
            const lineNum = i + 1;
            const doc = parseJSDoc(filePath, lineNum);
            fields.push({
                name: methodMatch[1],
                kind: "method",
                line: lineNum,
                description: doc.description,
                example: doc.example,
                params: paramNames(filePath, lineNum, methodMatch[1]) ?? "",
            });
            continue;
        }

        // property field: name: type or readonly name: type
        const fieldMatch = trimmed.match(/^(?:readonly\s+)?(\w+)\s*[?:]?\s*:/);
        if (fieldMatch) {
            const lineNum = i + 1;
            const doc = parseJSDoc(filePath, lineNum);
            fields.push({
                name: fieldMatch[1],
                kind: "field",
                line: lineNum,
                description: doc.description,
                example: doc.example,
            });
        }
    }

    return fields;
}
