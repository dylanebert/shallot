import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import { Marked } from "marked";
import markedShiki from "marked-shiki";
import { createHighlighter } from "shiki";
import { kebab, schema } from "../src/engine/ecs/core";
import { exampleBlock } from "./example";
import { fieldRows, fieldTable, registerComponents } from "./fields";
import {
    enumOptions,
    findDefinitionLine,
    getLines,
    isAsConst,
    type MemberExport,
    objectLiteral,
    paramNames,
    parseClassMembers,
    parseComponentFields,
    parseInterfaceFields,
    parseJSDoc,
    pluginRefs,
} from "./jsdoc";
import { assemblePage, manifest } from "./literate";

const PKG = resolve(import.meta.dir, "../src");
const DOCS = resolve(import.meta.dir, "../../../docs");
const DIST = resolve(DOCS, "dist");
// pinned to the release tag so reference source links stay stable; retarget at each release
const GITHUB = "https://github.com/dylanebert/shallot/blob/v0.6.0/packages/shallot/src";

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

function findDoc(
    filePath: string,
    name: string,
): { description: string | null; example: string | null; params: string | null } {
    if (!existsSync(filePath)) return { description: null, example: null, params: null };
    const defLine = findDefinitionLine(filePath, name);
    const doc = parseJSDoc(filePath, defLine);
    const params = doc.params ?? paramNames(filePath, defLine, name);
    return { description: doc.description, example: doc.example, params };
}

function hasExpandTag(filePath: string, name: string): boolean {
    if (!existsSync(filePath)) return false;
    const defLine = findDefinitionLine(filePath, name);
    const { tags } = parseJSDoc(filePath, defLine);
    return tags.some((t) => t.startsWith("@expand"));
}

// a re-exported name keeps the kind of its definition, so a re-exported function reads as a function
// (its signature + badge), not a bare value — matching a directly-exported one. Without this a
// re-exported `arrow()` fell through `exportKind`'s function guard to the kebab-folded `schema()` check
// and mis-rendered as the same-named `Arrow` component
function reExportKind(srcPath: string, name: string): "function" | "const" {
    const defLine = findDefinitionLine(srcPath, name);
    const line = defLine > 1 ? (getLines(srcPath)[defLine - 1] ?? "") : "";
    return /\bexport\s+(?:async\s+)?function\s/.test(line) ? "function" : "const";
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
                        kind: reExportKind(srcPath, n),
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
                        kind: reExportKind(indexPath, n),
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

/** the export's kind for the reference badge, so a reader can tell a component from a plugin from an enum
 *  at a glance: a registered component, a `: Plugin`/`: System`-typed object, an `as const` enum, a class,
 *  a function, a type, or a plain value. */
function exportKind(e: Export, srcPath: string): string {
    if (e.kind === "type") return "type";
    if (e.kind === "function") return "function";
    if (schema(e.name)) return "component";
    const defLine = findDefinitionLine(srcPath, e.name);
    const line = defLine > 1 ? (getLines(srcPath)[defLine - 1] ?? "") : "";
    if (/\bexport\s+class\b/.test(line)) return "class";
    if (/\bexport\s+enum\b/.test(line) || isAsConst(srcPath, e.name)) return "enum";
    if (/:\s*Plugin\b/.test(line)) return "plugin";
    if (/:\s*System\b/.test(line)) return "system";
    return "value";
}

/** a component's fields as a reference table — field, type, default, and the field's doc — so the entry
 *  shows what the component holds, not just its one-line summary. The rows are the inspector's, reflected
 *  from `schema()` (`fields.ts`); the descriptions are the field doc comments. Empty when it has no fields. */
function fieldRefTable(name: string, srcPath: string): string {
    const rows = fieldRows(name) ?? [];
    if (rows.length === 0) return "";
    const raw = parseComponentFields(srcPath, name);
    const descs: Record<string, string> = {};
    for (const k of Object.keys(raw)) descs[kebab(k)] = raw[k];
    const body = rows
        .map(
            (r) =>
                `<tr><td><code>${r.field}</code></td><td class="ref-ftype">${escapeHtml(r.type)}</td>` +
                `<td class="ref-fdefault">${escapeHtml(r.default)}</td>` +
                `<td class="ref-fdesc">${escapeHtml(descs[r.field] ?? "")}</td></tr>`,
        )
        .join("\n");
    const head = `<thead><tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>`;
    return `<table class="ref-fields">${head}<tbody>\n${body}\n</tbody></table>`;
}

/** an `as const` enum's option → value rows as a reference table, so the entry shows what it can be. */
function enumTable(srcPath: string, name: string): string {
    const opts = enumOptions(objectLiteral(srcPath, name));
    if (opts.length === 0) return "";
    const rows = opts
        .map(
            ([opt, val]) =>
                `<tr><td><code>${opt}</code></td><td class="ref-fdefault">${escapeHtml(val)}</td></tr>`,
        )
        .join("\n");
    return `<table class="ref-fields"><thead><tr><th>Option</th><th>Value</th></tr></thead><tbody>\n${rows}\n</tbody></table>`;
}

/** a plugin's bundled components / systems / dependencies, linking the components also documented on the
 *  page, so the entry shows what enabling it brings in. */
function pluginTable(srcPath: string, name: string, seen: Set<string>): string {
    const { components, systems, dependencies } = pluginRefs(objectLiteral(srcPath, name));
    const link = (n: string) =>
        seen.has(n) ? `<a href="#ref-${n}" class="ref-type">${n}</a>` : escapeHtml(n);
    const row = (label: string, items: string[]) =>
        items.length ? `<tr><th>${label}</th><td>${items.map(link).join(", ")}</td></tr>` : "";
    const rows = [
        row("Components", components),
        row("Systems", systems),
        row("Dependencies", dependencies),
    ].filter(Boolean);
    if (rows.length === 0) return "";
    return `<table class="ref-parts"><tbody>\n${rows.join("\n")}\n</tbody></table>`;
}

/** JSDoc `{@link Name}` / `{@link Name|text}` → a same-page reference link when Name has an entry, else
 *  plain code. Same grammar the editor field-hover strips (`fielddocs.ts` `plain`). */
function linkTags(desc: string, seen: Set<string>): string {
    return desc.replace(/\{@link\s+([^}|\s]+)(?:[|\s]+([^}]+))?\}/g, (_, name, text) =>
        seen.has(name)
            ? `<a href="#ref-${name}" class="ref-type"><code>${text ?? name}</code></a>`
            : `<code>${text ?? name}</code>`,
    );
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

    // @expand classes first, then alphabetical within each group. checked for every kind, not just
    // const: an interface+const merge (a singleton `export const Profile: Profile` beside its
    // `export interface Profile`) dedups to the interface entry, whose JSDoc carries the @expand tag —
    // gating on const alone dropped its field table.
    const expandSet = new Set<string>();
    for (const e of deduped) {
        const srcPath = resolve(
            singleFile
                ? resolve(PKG, subsystem.split("/").slice(0, -1).join("/"))
                : resolve(PKG, subsystem),
            e.source,
        );
        if (hasExpandTag(srcPath, e.name)) expandSet.add(e.name);
    }
    deduped.sort((a, b) => a.name.localeCompare(b.name));

    const linkBase = singleFile ? subsystem.split("/").slice(0, -1).join("/") : subsystem;
    const lines: string[] = [];
    lines.push(`<div class="ref-list">`);

    for (const e of deduped) {
        const url = sourceLink(linkBase, e.source, e.line);
        const desc = linkTags(e.description ?? "", seen);
        const srcPath = resolve(
            singleFile
                ? resolve(PKG, subsystem.split("/").slice(0, -1).join("/"))
                : resolve(PKG, subsystem),
            e.source,
        );
        const kind = exportKind(e, srcPath);
        const tag = `<span class="ref-kind ref-kind-${kind}">${kind}</span>`;
        const srcIcon = `<a href="${url}" target="_blank" rel="noopener" class="ref-src" aria-label="View source">${LINK_ICON}</a>`;

        // a kind-specific body expands the entry: a component's fields, an enum's options, or the
        // components/systems a plugin bundles. the entry then shows what the export *holds*, not a bare name.
        const body =
            kind === "component"
                ? fieldRefTable(e.name, srcPath)
                : kind === "enum"
                  ? enumTable(srcPath, e.name)
                  : kind === "plugin"
                    ? pluginTable(srcPath, e.name, seen)
                    : "";
        if (body) {
            lines.push(`<details class="ref-entry ref-group">`);
            lines.push(
                `<summary id="ref-${e.name}">${tag}<code>${e.name}</code>${srcIcon}</summary>`,
            );
            if (desc) lines.push(`<p class="ref-desc">${desc}</p>`);
            lines.push(`<div class="ref-methods">${body}</div>`);
            lines.push(`</details>`);
            continue;
        }

        // @expand classes/interfaces list their documented members
        let members: MemberExport[] = [];
        if (expandSet.has(e.name)) {
            members = parseClassMembers(srcPath, e.name);
            if (members.length === 0) members = parseInterfaceFields(srcPath, e.name);
            members = members.filter((m) => m.description !== null);
            members.sort((a, b) => a.name.localeCompare(b.name));
        }

        if (members.length > 0) {
            lines.push(`<details class="ref-entry ref-group">`);
            lines.push(
                `<summary id="ref-${e.name}">${tag}<code>${e.name}</code>${srcIcon}</summary>`,
            );
            if (desc) lines.push(`<p class="ref-desc">${desc}</p>`);
            if (e.example)
                lines.push(`<pre class="ref-example"><code>${escapeHtml(e.example)}</code></pre>`);
            lines.push(`<div class="ref-methods">`);
            for (const m of members) {
                const mUrl = sourceLink(linkBase, e.source, m.line);
                const mDesc = linkTags(m.description ?? "", seen);
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
            const head = `${tag}<code>${label}</code>${srcIcon}`;
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

// markdown → HTML happens here at build time so docs/dist/ is a render-ready artifact:
// both the site and the editor are pure views over it, neither shipping a markdown or
// highlighter runtime. Tab markers (<!-- tabs -->, <!-- pick -->) are HTML comments that
// survive marked, so consumers split them into their own tab UI.
const highlighter = await createHighlighter({
    themes: ["vitesse-dark"],
    langs: ["typescript", "xml", "bash", "json"],
});

const HIGHLIGHT = { theme: "vitesse-dark", colorReplacements: { "#121212": "transparent" } };

const marked = new Marked(
    markedShiki({
        highlight: (code, lang) =>
            highlighter.codeToHtml(code, { lang: lang || "text", ...HIGHLIGHT }),
    }),
    {
        renderer: {
            // images stay as authored (e.g. /captures/x.webp); a consumer prefixes its base path
            image: ({ href, title, text }) =>
                `<img src="${href}" alt="${text}"${title ? ` title="${title}"` : ""} loading="lazy">`,
            link: ({ href, title, text }) => {
                const external = href.startsWith("http://") || href.startsWith("https://");
                const target = external ? ' target="_blank" rel="noopener"' : "";
                return `<a href="${href}"${title ? ` title="${title}"` : ""}${target}>${text}</a>`;
            },
        },
    },
);

// ref-example blocks reach marked as raw HTML (escaped code emitted by the API/CORE tables),
// so marked-shiki skips them; highlight them in a second pass the way fenced blocks get it
function highlightRefExamples(html: string): string {
    return html.replace(/<pre class="ref-example"><code>([\s\S]*?)<\/code><\/pre>/g, (_, code) => {
        const decoded = code.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
        return highlighter
            .codeToHtml(decoded, { lang: "typescript", ...HIGHLIGHT })
            .replace("<pre", '<pre class="ref-example"');
    });
}

async function renderBody(body: string): Promise<string> {
    // drop the leading H1 — the title renders from frontmatter
    const html = (await marked.parse(body)).replace(/^<h1>.*?<\/h1>\n?/, "");
    return highlightRefExamples(html);
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

// FIELDS markers read the component registry, so populate it before the page loop
await registerComponents();

// a page is rendered the same way whoever wrote it: hand-authored docs/**/*.md or projected from a
// manifest entry (assemblePage). both reach here as the same marker-laden markdown.
async function renderPage(rel: string, source: string): Promise<void> {
    const dst = resolve(DIST, rel);
    mkdirSync(resolve(dst, ".."), { recursive: true });
    let content = source;

    // Expand EXAMPLE markers into fenced snippets pulled from zoo specimens (marked highlights them
    // in renderBody, like any fenced block). A missing file/region logs a "warning:" docs:check gates.
    content = content.replace(/<!-- EXAMPLE:([^#\s]+)(?:#(\S+))? -->/g, (_, path, region) => {
        const { code, error } = exampleBlock(path, region);
        if (error || !code) {
            console.warn(`  warning: EXAMPLE ${path}${region ? `#${region}` : ""}: ${error}`);
            return "";
        }
        return code;
    });

    // Expand FIELDS markers into a component's reflection-generated field table (the inspector's rows at
    // its default pose). A missing/unregistered component logs a "warning:" docs:check gates, like EXAMPLE.
    content = content.replace(/<!-- FIELDS:(\S+) -->/g, (_, name) => {
        const { table, error } = fieldTable(name);
        if (error || !table) {
            console.warn(`  warning: FIELDS ${name}: ${error ?? "no table"}`);
            return "";
        }
        return table;
    });

    // Replace API markers with one Reference heading + a source-tagged table per comma-listed source
    // (a single source is the one-to-one page; a list draws a conceptual page from several modules).
    content = content.replace(/<!-- API:(\S+) -->/g, (_, paths) => {
        const tables = paths.split(",").map((path: string) => {
            const srcFile = existsSync(resolve(PKG, path, "index.ts"))
                ? `${path}/index.ts`
                : `${path}.ts`;
            const srcLink = `${GITHUB}/${srcFile}`;
            return `${sourceTag(srcFile, srcLink)}\n\n${generateTable(path)}`;
        });
        return `## Reference\n\n${tables.join("\n\n")}`;
    });

    content = content.replace(/<!-- CORE:(\S+) -->/g, (_, subsystems) => {
        const sections = subsystems
            .split(",")
            .map((subsystem: string) => {
                const table = generateCoreTable(subsystem);
                if (!table) return "";
                const corePath = coreExports.get(subsystem);
                const rel = corePath ? relative(PKG, corePath) : "";
                const srcLink = rel ? `${GITHUB}/${rel}` : "";
                const srcLine = rel ? `${sourceTag(rel, srcLink)}\n\n` : "";
                return `${srcLine}${table}`;
            })
            .filter(Boolean);
        if (sections.length === 0) return "";
        return `### Core\n\n${sections.join("\n\n")}`;
    });

    // frontmatter stays verbatim (consumers read title/source/icon from it); the body
    // renders to HTML so dist is ready to display
    const fm = content.match(/^---\n[\s\S]*?\n---\n/);
    const front = fm ? fm[0] : "";
    const rendered = await renderBody(content.slice(front.length));
    writeFileSync(dst, front ? `${front}\n${rendered}` : rendered);
    console.log(`  ${rel}`);
}

const docs = walkDocs(DOCS);
let count = 0;

for (const src of docs) {
    await renderPage(relative(DOCS, src), readFileSync(src, "utf-8"));
    count++;
}

// projected pages: assembled from a specimen + module source + the nav manifest, then rendered
// through the identical path above — no hand-authored markdown, so no drift surface.
for (const entry of manifest()) {
    await renderPage(`${entry.slug}.md`, assemblePage(entry));
    count++;
}

console.log(`  → ${count} docs built to docs/dist/`);
