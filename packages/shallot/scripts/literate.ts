import { existsSync, readdirSync, readFileSync } from "node:fs";
import { relative, resolve } from "node:path";

// the projection generator: a doc page is assembled from a literate specimen + module source +
// the nav manifest, never hand-authored. the specimen carries the page lead (`#doc:intro`) and the
// Code-tab walkthrough (`#doc:code`) as markdown prose in `// #doc:` comment blocks, each optionally
// paired with a `// #region` fold so prose and snippet stay colocated; the module source carries the
// Dev-tab internals (`#doc:dev`) the same way. assemblePage emits the same intermediate markdown a
// hand-authored page was — `<!-- EXAMPLE / API / CORE -->` markers and all — so the existing
// marker-expansion + render pipeline turns it into an identical docs/dist/ artifact. there is no
// second copy of any prose, so nothing can drift. (Reference: Godot's --doctool generated reference
// + Knuth's literate programming.)

const SCRIPTS = import.meta.dir;
const PKG = resolve(SCRIPTS, "../src");
const DOCS = resolve(SCRIPTS, "../../../docs");
const ZOO = resolve(SCRIPTS, "../../../examples/zoo");

export interface DocBlock {
    kind: "intro" | "code" | "dev";
    prose: string;
    /** a zoo-relative `path` or `path#region` for the paired snippet, or null for prose-only. */
    example: string | null;
    /** the page slug this block routes to when its specimen/source feeds several pages (`#doc:code
     *  page:<slug>`); null routes it to every page that lists this specimen/source (the one-to-one case). */
    page: string | null;
}

export interface PageEntry {
    slug: string;
    title: string;
    description: string;
    icon: string;
    order?: number;
    /** module source dir(s) under src/ — the author `<!-- API: -->` tables (Code tab), the GitHub source
     *  link, and (unless `core` overrides) the `#doc:dev` blocks + `<!-- CORE: -->` table. A single dir is
     *  the one-to-one page; a list draws a conceptual author page from several modules. Omit for an
     *  extension-only page (surfaces) that documents a `/core` surface with no author API of its own. */
    source?: string | string[];
    /** the extension source dir(s) whose `#doc:dev` blocks + `<!-- CORE: -->` table render (the Internals
     *  surface), decoupled from the author `source`. Defaults to `source`. Set it when a conceptual page's
     *  author API spans modules but its extension story is one (rendering: `source` = render+part+sear+glaze,
     *  `core` = render), or an extension-only page draws its CORE table from a module it lists no author API
     *  for (surfaces: `core` = sear, `source` omitted). */
    core?: string | string[];
    /** zoo specimen dir(s) under examples/zoo/ — the page lead `#doc:intro` + the Code `#doc:code` blocks.
     *  A list routes several specimens' blocks onto one page (`#doc:code page:<slug>` splits a shared one). */
    specimen: string | string[];
}

// a `#doc:` / `#region` / `#endregion` marker ends a block; a markdown heading (`// ### …`) does not.
const MARKER = /^\s*\/\/\s*#(doc:|region\b|endregion\b)/;
// the marker's tail carries optional space-separated `key:value` tags (`source:<path>`, `page:<slug>`).
const DOC = /^\s*\/\/\s*#doc:(intro|code|dev)\b(.*)$/;
const REGION = /^\s*\/\/\s*#region\s+(\S+)/;
const COMMENT = /^\s*\/\/ ?(.*)$/;

/**
 * parse `// #doc:<intro|code|dev> [source:<path>] [page:<slug>]` literate blocks from a source file. a
 * block's prose is the comment lines after the marker (// stripped, bare // → blank line) up to the first
 * non-comment line or marker. an unsourced block whose terminator is `// #region <name>` pairs with
 * that same-file region (zooRel is the file's path under examples/zoo/, or null for non-specimen files);
 * a `page:<slug>` tag routes the block to one page when its specimen/source feeds several.
 */
export function parseDocBlocks(source: string, zooRel: string | null): DocBlock[] {
    const lines = source.split("\n");
    const blocks: DocBlock[] = [];
    for (let i = 0; i < lines.length; i++) {
        const m = lines[i].match(DOC);
        if (!m) continue;
        const tail = m[2];
        const prose: string[] = [];
        let j = i + 1;
        for (; j < lines.length; j++) {
            if (MARKER.test(lines[j])) break;
            const c = lines[j].match(COMMENT);
            if (!c) break;
            prose.push(c[1]);
        }
        let example = tail.match(/\bsource:(\S+)/)?.[1] ?? null;
        if (!example && zooRel) {
            const region = lines[j]?.match(REGION);
            if (region) example = `${zooRel}#${region[1]}`;
        }
        blocks.push({
            kind: m[1] as DocBlock["kind"],
            prose: prose.join("\n").trim(),
            example,
            page: tail.match(/\bpage:(\S+)/)?.[1] ?? null,
        });
        i = j - 1;
    }
    return blocks;
}

// authored specimen/module sources only — never a dependency or build output, where a stray `#doc:`
// comment would leak into a page.
const SKIP = new Set(["node_modules", "dist", "build"]);

/** every non-test .ts file under a directory, sorted (so file order is deterministic page order). */
function tsFiles(dir: string): string[] {
    if (!existsSync(dir)) return [];
    const out: string[] = [];
    for (const e of readdirSync(dir, { withFileTypes: true })) {
        if (SKIP.has(e.name)) continue;
        const full = resolve(dir, e.name);
        if (e.isDirectory()) out.push(...tsFiles(full));
        else if (e.name.endsWith(".ts") && !e.name.endsWith(".test.ts")) out.push(full);
    }
    return out.sort();
}

function blocksFrom(dir: string, zoo: boolean): DocBlock[] {
    const out: DocBlock[] = [];
    for (const f of tsFiles(dir)) {
        out.push(...parseDocBlocks(readFileSync(f, "utf-8"), zoo ? relative(ZOO, f) : null));
    }
    return out;
}

/** normalize the one-to-one degenerate (a bare string) and the many-to-many list to an array. */
function list(v: string | string[]): string[] {
    return Array.isArray(v) ? v : [v];
}

/** the `<!-- CORE:… -->` key for a source dir. coreExports (docs.ts) and the hand-authored pages key the
 *  Internals table by the leaf segment (`standard/render` → `render`), so the API marker takes the full
 *  path but CORE takes the leaf. */
function coreKey(source: string): string {
    return source.split("/").pop() ?? source;
}

/** the extension source dir(s) whose `#doc:dev` blocks + CORE table render — `core` when set, else the
 *  author `source` (the one-to-one case: a module documents its own `/core`). Empty for an author-only leaf. */
function coreSources(entry: PageEntry): string[] {
    return entry.core !== undefined
        ? list(entry.core)
        : entry.source !== undefined
          ? list(entry.source)
          : [];
}

/**
 * the intermediate markdown for a manifest entry, from already-gathered blocks (pure — the fs read is
 * assemblePage): frontmatter + page lead + the specimen walkthrough + generated reference. Three page
 * shapes, by audience: an **author-only** page (`source`, no `#doc:dev`) is single-tab — walkthrough + the
 * `<!-- API: -->` table inline (orbit, the leaf); an **extender-only** page (`core`/`source` with `#doc:dev`
 * but no author `source`) is single-tab too — walkthrough + the `#doc:dev` internals + the `<!-- CORE: -->`
 * table inline (surfaces); a **dual-audience** page (an author `source` *and* `#doc:dev`) splits into a Code
 * tab (walkthrough + API) and an Internals tab (dev + CORE). CORE is decoupled from API via `core`, so a
 * conceptual page lists many author modules but one extension story (rendering: API render+part+sear+glaze,
 * CORE render). A block is dropped unless it routes here — its `page:<slug>` matches, or it has none (the
 * one-to-one case). Multiple sources emit one comma-joined marker each (docs.ts renders one heading, a
 * table per source); a multi-source author page omits the single `source:` frontmatter (it maps to no one
 * module, docs.md). Fed through docs.ts's marker-expansion + render, exactly as a hand-authored page was.
 */
export function composePage(
    entry: PageEntry,
    specimenBlocks: DocBlock[],
    sourceBlocks: DocBlock[],
): string {
    const api = entry.source !== undefined ? list(entry.source) : [];
    const core = coreSources(entry);
    const routed = (b: DocBlock) => b.page === null || b.page === entry.slug;
    const intro = specimenBlocks.filter((b) => b.kind === "intro" && routed(b));
    const code = specimenBlocks.filter((b) => b.kind === "code" && routed(b));
    const dev = sourceBlocks.filter((b) => b.kind === "dev" && routed(b));

    const render = (b: DocBlock) =>
        b.example ? `${b.prose}\n\n<!-- EXAMPLE:${b.example} -->` : b.prose;

    const out: string[] = [
        "---",
        `title: ${entry.title}`,
        `description: ${entry.description}`,
        ...(api.length === 1 ? [`source: ${api[0]}`] : []),
        `icon: ${entry.icon}`,
        ...(entry.order !== undefined ? [`order: ${entry.order}`] : []),
        "---",
        "",
        `# ${entry.title}`,
        "",
    ];
    if (intro.length) out.push(intro.map((b) => b.prose).join("\n\n"), "");

    const codeBlocks = code.map(render).join("\n\n");
    const apiMarker = `<!-- API:${api.join(",")} -->`;
    const coreMarker = `<!-- CORE:${core.map(coreKey).join(",")} -->`;
    const devBlocks = dev.map(render).join("\n\n");
    if (dev.length === 0) {
        // author-only leaf: walkthrough + API inline, no tab chrome. A specimen-only page (a guide — no
        // author `source`, no extension `core`) is walkthrough alone, no reference table.
        out.push(codeBlocks, "");
        if (api.length) out.push(apiMarker, "");
    } else if (api.length === 0) {
        // extender-only page: walkthrough + internals + CORE inline, no tab chrome (its whole audience is
        // the extender, so a Code/Internals split would be one empty tab).
        out.push(codeBlocks, "", devBlocks, "", coreMarker, "");
    } else {
        // dual-audience: Code tab (walkthrough + author API), Internals tab (internals + CORE).
        out.push("<!-- tabs -->", "", "<!-- tab: Code -->", "");
        out.push(codeBlocks, "", apiMarker, "");
        out.push("<!-- tab: Internals -->", "");
        out.push(devBlocks, "", coreMarker, "");
        out.push("<!-- /tabs -->", "");
    }
    return out.join("\n");
}

/** the raw `#doc:` blocks a page assembles from: specimen blocks (intro/code) and extension-source blocks
 *  (dev). Unfiltered by routing — the caller applies the `page:<slug>` filter (composePage does, and
 *  docs-check reads them for the per-block budget check). */
export function pageBlocks(entry: PageEntry): { specimen: DocBlock[]; source: DocBlock[] } {
    return {
        specimen: list(entry.specimen).flatMap((s) => blocksFrom(resolve(ZOO, s), true)),
        source: coreSources(entry).flatMap((s) => blocksFrom(resolve(PKG, s), false)),
    };
}

/** compose a manifest entry's page, reading its specimen(s) and source(s) from disk. `#doc:dev` blocks
 *  come from the extension sources (`core`, else `source`), never the author-only API sources. */
export function assemblePage(entry: PageEntry): string {
    const { specimen, source } = pageBlocks(entry);
    return composePage(entry, specimen, source);
}

/** the nav manifest: one entry per projected page, carrying the nav chrome + the specimen/source it assembles from. */
export function manifest(): PageEntry[] {
    return JSON.parse(readFileSync(resolve(DOCS, "manifest.json"), "utf-8"));
}
