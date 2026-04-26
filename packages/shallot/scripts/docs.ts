import { resolve, relative } from "node:path";
import { readFileSync, readdirSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";

const PKG = resolve(import.meta.dir, "../src");
const DOCS = resolve(import.meta.dir, "../../../docs");
const DIST = resolve(DOCS, "dist");
// TODO: link to tagged release (e.g. /blob/v1.0.0/) instead of /blob/main/ once releases are standardized
const GITHUB = "https://github.com/dylanebert/shallot/blob/main/packages/shallot/src";

// Build core export map from package.json: subsystem name → core file path
const pkg = JSON.parse(readFileSync(resolve(import.meta.dir, "../package.json"), "utf-8"));
const coreExports = new Map<string, string>();
for (const [subpath, file] of Object.entries(pkg.exports as Record<string, string>)) {
    const match = subpath.match(/^\.\/(\w+)\/core$/);
    if (match) coreExports.set(match[1], resolve(import.meta.dir, "..", file));
}

interface Export {
    name: string;
    kind: "function" | "const" | "type";
    source: string;
    subsystem: string;
    line: number;
    description: string | null;
    example: string | null;
    params: string | null;
}

// cache source file contents for line lookups
const fileCache = new Map<string, string[]>();

function getLines(filePath: string): string[] {
    if (!fileCache.has(filePath)) {
        fileCache.set(filePath, readFileSync(filePath, "utf-8").split("\n"));
    }
    return fileCache.get(filePath)!;
}

function findDefinitionLine(filePath: string, name: string): number {
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

interface JSDocResult {
    description: string | null;
    tags: string[];
    example: string | null;
    params: string | null;
}

function parseJSDoc(filePath: string, defLine: number): JSDocResult {
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

    let description: string | null = null;
    const tags: string[] = [];
    let example: string | null = null;
    let params: string | null = null;
    let inExample = false;
    const exampleLines: string[] = [];

    for (const line of cleaned) {
        if (line.startsWith("@example")) {
            inExample = true;
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
        if (line === "") continue;
        if (line.startsWith("@params ")) {
            params = line.slice(8).trim();
        } else if (line.startsWith("@")) {
            tags.push(line);
        } else if (description === null) {
            description = line;
        }
    }

    if (exampleLines.length > 0) {
        while (exampleLines.length > 0 && exampleLines[0] === "") exampleLines.shift();
        while (exampleLines.length > 0 && exampleLines[exampleLines.length - 1] === "")
            exampleLines.pop();
        if (exampleLines.length > 0) example = exampleLines.join("\n");
    }

    return { description, tags, example, params };
}

function extractParamsFromLine(filePath: string, lineNum: number, name: string): string | null {
    if (!existsSync(filePath)) return null;
    const lines = getLines(filePath);
    if (lineNum <= 0 || lineNum > lines.length) return null;
    // join lines until closing paren (handles multi-line signatures)
    let sig = "";
    for (let i = lineNum - 1; i < Math.min(lines.length, lineNum + 10); i++) {
        sig += ` ${lines[i]}`;
        if (sig.includes(")")) break;
    }
    const m = sig.match(new RegExp(`\\b${name}\\s*(?:<[^>]*>)?\\s*\\(([^)]*)\\)`));
    if (!m) return null;
    const raw = m[1].trim();
    if (!raw) return "";
    return raw
        .split(",")
        .map((p) =>
            p
                .trim()
                .replace(/\s*[:=].*$/, "")
                .replace(/^\.\.\.\s*/, "..."),
        )
        .filter(Boolean)
        .join(", ");
}

function findDoc(
    filePath: string,
    name: string,
): { description: string | null; example: string | null; params: string | null } {
    if (!existsSync(filePath)) return { description: null, example: null, params: null };
    const defLine = findDefinitionLine(filePath, name);
    const doc = parseJSDoc(filePath, defLine);
    const params = doc.params ?? extractParamsFromLine(filePath, defLine, name);
    return { description: doc.description, example: doc.example, params };
}

function hasExpandTag(filePath: string, name: string): boolean {
    if (!existsSync(filePath)) return false;
    const defLine = findDefinitionLine(filePath, name);
    const { tags } = parseJSDoc(filePath, defLine);
    return tags.some((t) => t.startsWith("@expand"));
}

interface MemberExport {
    name: string;
    kind: "method" | "property" | "field";
    line: number;
    description: string | null;
    returnType?: string;
    example?: string | null;
    params?: string;
}

function parseClassMembers(filePath: string, className: string): MemberExport[] {
    if (!existsSync(filePath)) return [];
    const lines = getLines(filePath);
    const defLine = findDefinitionLine(filePath, className);
    if (defLine <= 0) return [];

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
            const params = extractParamsFromLine(filePath, lineNum, methodMatch[1]);
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

function parseInterfaceFields(filePath: string, name: string): MemberExport[] {
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

        // interface field: name: type or readonly name: type
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

function parseExports(indexPath: string, subsystem: string): Export[] {
    const content = readFileSync(indexPath, "utf-8");
    const dir = resolve(indexPath, "..");
    const exports: Export[] = [];

    // collapse multi-line export blocks into single lines
    const collapsed = content.replace(/export\s+(type\s+)?\{[^}]*\}/gs, (m) =>
        m.replace(/\n/g, " "),
    );

    for (const line of collapsed.split("\n")) {
        const trimmed = line.trim();

        // export type { A, B } from "./foo"
        const reExportType = trimmed.match(
            /^export\s+type\s*\{([^}]+)\}\s*from\s*["']\.\/([^"']+)["']/,
        );
        if (reExportType) {
            const names = reExportType[1].split(",").map((n) => n.trim());
            const source = reExportType[2].replace(/\.ts$/, "") + ".ts";
            const srcPath = resolve(dir, source);
            for (const n of names) {
                if (n)
                    exports.push({
                        name: n,
                        kind: "type",
                        source,
                        subsystem,
                        line: findDefinitionLine(srcPath, n),
                        ...findDoc(srcPath, n),
                    });
            }
            continue;
        }

        // export { A, B, C } from "./foo"
        const reExportFrom = trimmed.match(/^export\s*\{([^}]+)\}\s*from\s*["']\.\/([^"']+)["']/);
        if (reExportFrom) {
            const names = reExportFrom[1]
                .split(",")
                .map((n) => n.trim())
                .filter(Boolean);
            const source = reExportFrom[2].replace(/\.ts$/, "") + ".ts";
            const srcPath = resolve(dir, source);
            for (const n of names) {
                if (n.startsWith("type ")) {
                    const name = n.slice(5);
                    exports.push({
                        name,
                        kind: "type",
                        source,
                        subsystem,
                        line: findDefinitionLine(srcPath, name),
                        ...findDoc(srcPath, name),
                    });
                } else {
                    exports.push({
                        name: n,
                        kind: "const",
                        source,
                        subsystem,
                        line: findDefinitionLine(srcPath, n),
                        ...findDoc(srcPath, n),
                    });
                }
            }
            continue;
        }

        // export { A, B } (local re-export, no from)
        const reExportLocal = trimmed.match(/^export\s*\{([^}]+)\}\s*;?\s*$/);
        if (reExportLocal) {
            const names = reExportLocal[1]
                .split(",")
                .map((n) => n.trim())
                .filter(Boolean);
            for (const n of names) {
                if (n.startsWith("type ")) {
                    const name = n.slice(5);
                    exports.push({
                        name,
                        kind: "type",
                        source: "index.ts",
                        subsystem,
                        line: findDefinitionLine(indexPath, name),
                        ...findDoc(indexPath, name),
                    });
                } else {
                    exports.push({
                        name: n,
                        kind: "const",
                        source: "index.ts",
                        subsystem,
                        line: findDefinitionLine(indexPath, n),
                        ...findDoc(indexPath, n),
                    });
                }
            }
            continue;
        }

        // export function / export async function
        const funcExport = trimmed.match(/^export\s+(?:async\s+)?function\s+(\w+)/);
        if (funcExport) {
            exports.push({
                name: funcExport[1],
                kind: "function",
                source: "index.ts",
                subsystem,
                line: findDefinitionLine(indexPath, funcExport[1]),
                ...findDoc(indexPath, funcExport[1]),
            });
            continue;
        }

        // export const/let/class/enum
        const constExport = trimmed.match(/^export\s+(const|let|class|enum)\s+(\w+)/);
        if (constExport) {
            exports.push({
                name: constExport[2],
                kind: "const",
                source: "index.ts",
                subsystem,
                line: findDefinitionLine(indexPath, constExport[2]),
                ...findDoc(indexPath, constExport[2]),
            });
            continue;
        }

        // export interface/type
        const typeExport = trimmed.match(/^export\s+(interface|type)\s+(\w+)/);
        if (typeExport) {
            exports.push({
                name: typeExport[2],
                kind: "type",
                source: "index.ts",
                subsystem,
                line: findDefinitionLine(indexPath, typeExport[2]),
                ...findDoc(indexPath, typeExport[2]),
            });
            continue;
        }

        // export * from "./foo" — recursively expand
        const starExport = trimmed.match(/^export\s+\*\s+from\s*["']\.\/([^"']+)["']/);
        if (starExport) {
            const subPath = resolve(dir, starExport[1]);
            const candidates = [subPath + ".ts", subPath + "/index.ts"];
            for (const candidate of candidates) {
                if (existsSync(candidate)) {
                    const subExports = parseExports(candidate, subsystem);
                    const source = relative(dir, candidate).replace(/\/index\.ts$/, "/");
                    for (const e of subExports) {
                        exports.push({
                            ...e,
                            source: e.source === "index.ts" ? source : e.source,
                        });
                    }
                    break;
                }
            }
        }
    }

    return exports;
}

function escapeHtml(s: string): string {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const LINK_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>`;

function sourceTag(file: string, url: string): string {
    return `<div class="ref-source"><span class="ref-source-path">${file}</span><a href="${url}" target="_blank" rel="noopener" class="ref-source-link" aria-label="View source on GitHub">${LINK_ICON}</a></div>`;
}

function sourceLink(subsystem: string, source: string, line: number): string {
    return `${GITHUB}/${subsystem}/${source}#L${line}`;
}

function generateTable(subsystem: string, filePath?: string): string {
    let indexPath = filePath ?? resolve(PKG, subsystem, "index.ts");
    let singleFile = false;
    if (!filePath && !existsSync(indexPath)) {
        const fp = resolve(PKG, subsystem + ".ts");
        if (existsSync(fp)) {
            indexPath = fp;
            singleFile = true;
        } else {
            console.warn(`  warning: ${indexPath} not found`);
            return `*No exports found.*\n`;
        }
    }

    const all = parseExports(
        indexPath,
        subsystem.includes("/") ? subsystem.split("/")[0] : subsystem,
    );
    if (singleFile) {
        const fileName = subsystem.split("/").pop() + ".ts";
        for (const e of all) {
            if (e.source === "index.ts") e.source = fileName;
        }
    }
    if (all.length === 0) {
        return `*No exports found.*\n`;
    }

    // deduplicate
    const seen = new Set<string>();
    const deduped = all.filter((e) => {
        if (seen.has(e.name)) return false;
        seen.add(e.name);
        return true;
    });

    // @expand classes first, then alphabetical within each group
    const expandSet = new Set<string>();
    for (const e of deduped) {
        if (e.kind === "const") {
            const srcPath = resolve(
                singleFile
                    ? resolve(PKG, subsystem.split("/").slice(0, -1).join("/"))
                    : resolve(PKG, subsystem),
                e.source,
            );
            if (hasExpandTag(srcPath, e.name)) expandSet.add(e.name);
        }
    }
    deduped.sort((a, b) => a.name.localeCompare(b.name));

    const linkBase = singleFile ? subsystem.split("/").slice(0, -1).join("/") : subsystem;
    const lines: string[] = [];
    lines.push(`<div class="ref-list">`);

    for (const e of deduped) {
        const url = sourceLink(linkBase, e.source, e.line);
        const desc = e.description ?? "";

        // check for @expand (pre-computed in expandSet)
        let members: MemberExport[] = [];
        if (expandSet.has(e.name)) {
            const srcPath = resolve(
                singleFile
                    ? resolve(PKG, subsystem.split("/").slice(0, -1).join("/"))
                    : resolve(PKG, subsystem),
                e.source,
            );
            // class → methods + properties, interface/const → fields
            members = parseClassMembers(srcPath, e.name);
            if (members.length === 0) members = parseInterfaceFields(srcPath, e.name);
            // only show documented members — undocumented ones are internal
            members = members.filter((m) => m.description !== null);
            members.sort((a, b) => a.name.localeCompare(b.name));
        }

        if (members.length > 0) {
            lines.push(`<details class="ref-entry ref-group">`);
            lines.push(
                `<summary id="ref-${e.name}"><code>${e.name}</code><a href="${url}" target="_blank" rel="noopener" class="ref-src" aria-label="View source">${LINK_ICON}</a></summary>`,
            );
            if (desc) lines.push(`<p class="ref-desc">${desc}</p>`);
            if (e.example)
                lines.push(`<pre class="ref-example"><code>${escapeHtml(e.example)}</code></pre>`);
            lines.push(`<div class="ref-methods">`);
            for (const m of members) {
                const mUrl = sourceLink(linkBase, e.source, m.line);
                const mDesc = m.description ?? "";
                const label =
                    m.kind === "method"
                        ? `.${m.name}<span class="ref-params">(${m.params ?? ""})</span>`
                        : `.${m.name}`;
                const typeLink =
                    m.returnType && seen.has(m.returnType)
                        ? `<a href="#ref-${m.returnType}" class="ref-type">${m.returnType}</a>`
                        : "";
                const head = `<code>${label}</code>${typeLink}<a href="${mUrl}" target="_blank" rel="noopener" class="ref-src" aria-label="View source">${LINK_ICON}</a>`;
                const body = [
                    mDesc ? `<p class="ref-desc">${mDesc}</p>` : "",
                    m.example
                        ? `<pre class="ref-example"><code>${escapeHtml(m.example)}</code></pre>`
                        : "",
                ]
                    .filter(Boolean)
                    .join("\n");
                if (body) {
                    lines.push(`<details class="ref-entry ref-method">`);
                    lines.push(`<summary>${head}</summary>`);
                    lines.push(body);
                    lines.push(`</details>`);
                } else {
                    lines.push(`<div class="ref-item ref-method">${head}</div>`);
                }
            }
            lines.push(`</div>`);
            lines.push(`</details>`);
        } else {
            const label =
                e.params !== null
                    ? `${e.name}<span class="ref-params">(${e.params})</span>`
                    : e.name;
            const head = `<code>${label}</code><a href="${url}" target="_blank" rel="noopener" class="ref-src" aria-label="View source">${LINK_ICON}</a>`;
            const body = [
                desc ? `<p class="ref-desc">${desc}</p>` : "",
                e.example
                    ? `<pre class="ref-example"><code>${escapeHtml(e.example)}</code></pre>`
                    : "",
            ]
                .filter(Boolean)
                .join("\n");
            if (body) {
                lines.push(`<details class="ref-entry" id="ref-${e.name}">`);
                lines.push(`<summary>${head}</summary>`);
                lines.push(body);
                lines.push(`</details>`);
            } else {
                lines.push(`<div class="ref-item" id="ref-${e.name}">${head}</div>`);
            }
        }
    }

    lines.push(`</div>`);

    return lines.join("\n");
}

function generateCoreTable(subsystemName: string): string | null {
    const corePath = coreExports.get(subsystemName);
    if (!corePath || !existsSync(corePath)) return null;

    // Determine the subsystem prefix for source links
    // e.g. ecs/core.ts → engine/ecs, render/core.ts → standard/render
    const rel = relative(PKG, corePath);
    const subsystem = rel.split("/").slice(0, -1).join("/");

    return generateTable(subsystem, corePath);
}

// main
rmSync(DIST, { recursive: true, force: true });
mkdirSync(DIST, { recursive: true });

function walkDocs(dir: string): string[] {
    const results: string[] = [];
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.name === "dist") continue;
        const full = resolve(dir, entry.name);
        if (entry.isDirectory()) {
            results.push(...walkDocs(full));
        } else if (entry.name.endsWith(".md")) {
            results.push(full);
        }
    }
    return results;
}

const docs = walkDocs(DOCS);
let count = 0;

for (const src of docs) {
    const rel = relative(DOCS, src);
    const dst = resolve(DIST, rel);
    mkdirSync(resolve(dst, ".."), { recursive: true });
    let content = readFileSync(src, "utf-8");

    // Replace API markers with Reference heading + table
    content = content.replace(/<!-- API:(\S+) -->/g, (_, path) => {
        const srcFile = existsSync(resolve(PKG, path, "index.ts"))
            ? `${path}/index.ts`
            : `${path}.ts`;
        const srcLink = `${GITHUB}/${srcFile}`;
        return `## Reference\n\n${sourceTag(srcFile, srcLink)}\n\n${generateTable(path)}`;
    });

    content = content.replace(/<!-- CORE:(\S+) -->/g, (_, subsystem) => {
        const table = generateCoreTable(subsystem);
        if (!table) return "";
        const corePath = coreExports.get(subsystem);
        const rel = corePath ? relative(PKG, corePath) : "";
        const srcLink = rel ? `${GITHUB}/${rel}` : "";
        const srcLine = rel ? `${sourceTag(rel, srcLink)}\n\n` : "";
        return `### Core\n\n${srcLine}${table}`;
    });

    writeFileSync(dst, content);
    console.log(`  ${rel}`);
    count++;
}

console.log(`  → ${count} docs built to docs/dist/`);
