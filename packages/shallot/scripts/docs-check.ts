#!/usr/bin/env bun
import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { Glob } from "bun";
import { assemblePage, manifest, pageBlocks } from "./literate";

/**
 * docs:check — the docs drift gate (part of `bun check`).
 *
 * The Reference tab is generated from source JSDoc by scripts/docs.ts, and the
 * site regenerates docs/dist/ on every build, so a stale committed artifact can't
 * drift. What CAN drift is the contract between code and prose. This gate runs the
 * build fresh and catches six classes:
 *
 * Objective (always fail on sight — the tree passes these today):
 *  - a build warning (a marker whose module is gone),
 *  - a dead API/CORE marker that rendered no exports,
 *  - a `source:` front-matter that doesn't resolve under src/,
 *  - a `doc:` cross-page link whose target is no known page slug (link).
 *
 * Ratchet (baseline-diff against docs-baseline.txt, the API-Extractor shape):
 *  - bare:   a release-public export with no JSDoc (generates a bare ref-item).
 *  - hole:   a tier subsystem (a package.json core or named subpath) with no page.
 *  - keep:   a prose mention of a generated reference entry (a leak candidate).
 *  - prose:  an em dash, banned word, or editorial opener in docs prose (authored
 *            markdown, `#doc:` blocks, or a rendered JSDoc summary).
 *  - budget: a `#doc:code` walkthrough block over the ≤3-sentence budget.
 *
 * `prose`/`budget` carry the writing debt the live site shipped; they ride the
 * ratchet while the docs pillar burns them to zero, then the empty baseline makes
 * them hard (any new violation is an added line that fails). The banned-word list
 * and the whole bar live here, not in a rule — a prose rule didn't survive
 * generation pressure; a `bun check` failure does.
 *
 * The ratchet classes have a real backlog the docs pillar burns down over many
 * passes, so a naive hard fail would wedge `bun check`. The baseline is that
 * backlog: the gate fails on any drift from it in either direction (a new bare
 * export, a filled one not yet removed, a new prose mention). `--update` rewrites
 * the baseline from the current tree; "green with an empty baseline" = complete.
 */

const dir = import.meta.dir;
const PKG = resolve(dir, "../src");
const DOCS = resolve(dir, "../../../docs");
const DIST = resolve(DOCS, "dist");
const BUILD = resolve(dir, "docs.ts");
const BASELINE = resolve(dir, "docs-baseline.txt");

// --- pure logic -------------------------------------------------------------

/** top-level exports that rendered with no JSDoc description. A `ref-desc` paragraph is the rendered
 * summary; an entry without one is bare whether it's a plain `ref-item` or expands to a field/option/parts
 * table (a component's fields or a plugin's parts don't substitute for its summary). Splitting on the
 * `id="ref-Name"` anchor scopes each chunk to one entry; members carry no id, so they're excluded. */
export function bareLines(html: string, page: string): string[] {
    const out: string[] = [];
    for (const part of html.split(/(?=id="ref-)/)) {
        const m = part.match(/^id="ref-([A-Za-z0-9_]+)"/);
        if (m && !part.includes('<p class="ref-desc">')) out.push(`bare ${page} ${m[1]}`);
    }
    return out;
}

/** every generated reference anchor name on a page. */
export function anchorNames(html: string): Set<string> {
    const names = new Set<string>();
    for (const m of html.matchAll(/id="ref-([A-Za-z0-9_]+)"/g)) names.add(m[1]);
    return names;
}

/** page prose for leak detection: source markdown minus the marker lines (which become the generated Reference) and minus fenced code (examples name exports legitimately). */
export function prose(src: string): string {
    const noFences = src.replace(/```[\s\S]*?```/g, "");
    return noFences
        .split("\n")
        .filter((l) => !/<!--\s*(API|CORE):/.test(l))
        .join("\n");
}

/** reference names mentioned in inline code in the prose — leak candidates a human resolves keep-or-cut. */
export function leakLines(proseText: string, names: Set<string>, page: string): string[] {
    const out: string[] = [];
    for (const name of names) {
        if (new RegExp("`[^`]*\\b" + name + "\\b[^`]*`").test(proseText)) {
            out.push(`keep ${page} ${name}`);
        }
    }
    return out;
}

/** subsystem key a doc marker covers: drop the `core` segment, take the last (`standard/bvh/core` → bvh, `CORE:render` → render). */
export function markerSubsystem(value: string): string {
    const parts = value.split("/").filter((p) => p !== "core");
    return parts[parts.length - 1];
}

/** tier subsystems that must map to a page: the core and named subpaths in package.json exports (the machine-readable tier list), minus the multi-page barrels. */
export function requiredSubsystems(exports: Record<string, string>): Set<string> {
    const out = new Set<string>();
    for (const subpath of Object.keys(exports)) {
        if (subpath === "." || subpath === "./extras" || subpath === "./editor") continue;
        // `./vite` is build-config tooling (the project plugin a standalone project's vite.config uses),
        // not a runtime subsystem that maps to its own page.
        if (subpath === "./vite") continue;
        if (subpath.includes("*")) continue;
        const key = subpath
            .replace(/^\.\//, "")
            .split("/")
            .filter((p) => p !== "core")[0];
        if (key) out.add(key);
    }
    return out;
}

/** subsystems required by a tier but with no doc page. */
export function holeLines(required: Set<string>, covered: Set<string>): string[] {
    const out: string[] = [];
    for (const sub of required) if (!covered.has(sub)) out.push(`hole ${sub}`);
    return out;
}

// --- prose gate (link / prose / budget) -------------------------------------
// The writing bar in docs.md doesn't survive generation pressure on its own: the live site shipped ~80 em
// dashes past a rule that said "keep them near zero". So the bar is mechanized here, and this file is the
// banned-list home too — docs.md points writers at these lists, editable in one place.

/** AI-tell words banned in docs prose. Curated for engine vocabulary: `harness`, `navigate`, and
 *  `landscape` are dropped, since shallot's test `harness/`, menu navigation, and terrain landscapes are
 *  legitimate literal uses the metaphorical ban would false-positive on. Matched case-insensitively on
 *  word boundaries. */
export const BANNED = [
    "delve",
    "leverage",
    "unlock",
    "foster",
    "elevate",
    "showcase",
    "journey",
    "tapestry",
    "seamless",
    "groundbreaking",
    "transformative",
    "pivotal",
    "comprehensive",
    "crucial",
    "compelling",
    "nuanced",
    "multifaceted",
    "cutting-edge",
    "utilize",
    "facilitate",
    "endeavor",
    "aforementioned",
    "underpin",
    "underscore",
    "noteworthy",
    "intricate",
    "commendable",
    "meticulous",
    "synergy",
    "embark",
    "esteemed",
    "holistic",
    "paradigm",
    "realm",
    "robust",
];

/** editorial openers / throat-clearing. Distinctive phrases, matched as case-insensitive substrings. */
export const OPENERS = [
    "in today's",
    "it's worth noting",
    "it is worth noting",
    "it's important to note",
    "it is important to note",
    "it is important to understand",
    "in order to",
    "in conclusion",
    "in summary",
    "in essence",
    "by the end of this",
    "a wide range of",
    "a variety of",
    "this is where",
];

/** page prose with everything that isn't prose stripped: frontmatter, fenced code (snippets name banned
 *  tokens legitimately), HTML/marker comments, and inline code (identifiers carry dots + banned words). */
export function proseText(src: string): string {
    return src
        .replace(/^---\n[\s\S]*?\n---\n/, "")
        .replace(/```[\s\S]*?```/g, "")
        .replace(/<!--[\s\S]*?-->/g, "")
        .replace(/`[^`]*`/g, "");
}

/** the rendered JSDoc summaries on a page (the `ref-desc` paragraphs), tags and inline code stripped —
 *  the prose gate covers JSDoc summaries as they actually ship, not only authored markdown. */
export function descText(html: string): string {
    const out: string[] = [];
    for (const m of html.matchAll(/<p class="ref-desc">([\s\S]*?)<\/p>/g)) {
        out.push(m[1].replace(/<code>[\s\S]*?<\/code>/g, "").replace(/<[^>]+>/g, ""));
    }
    return out.join("\n");
}

/** `doc:` cross-page link targets in a page (slug only, anchor dropped), read from prose (fences stripped,
 *  so a snippet demonstrating the syntax isn't a target). */
export function docTargets(src: string): string[] {
    const out: string[] = [];
    for (const m of proseText(src).matchAll(/\(doc:([^)\s#]+)/g)) out.push(m[1]);
    return out;
}

/** `doc:` targets that resolve to no known page slug — a broken cross-page link. Objective (hard fail). */
export function linkLines(src: string, valid: Set<string>, page: string): string[] {
    const out: string[] = [];
    for (const t of docTargets(src))
        if (!valid.has(t)) out.push(`docs/${page}: broken doc: link → ${t}`);
    return out;
}

/** prose violations on a page: em dash, banned word, editorial opener. One ratchet line each; the Set in
 *  gather() dedups a hit that appears in both authored prose and a rendered summary. */
export function proseFindings(text: string, page: string): string[] {
    const out: string[] = [];
    const lower = text.toLowerCase();
    if (text.includes("—")) out.push(`prose ${page} emdash`);
    for (const w of BANNED)
        if (new RegExp(`\\b${w}\\b`, "i").test(text)) out.push(`prose ${page} banned:${w}`);
    for (const p of OPENERS)
        if (lower.includes(p)) out.push(`prose ${page} opener:${p.replace(/[^a-z0-9]+/g, "-")}`);
    return out;
}

/** sentence count of a `#doc:code` block's prose, for the ≤3-sentence budget (docs.md / hardening.md).
 *  Inline code and the `e.g.`/`i.e.`/`etc.` abbreviations are neutralized so their dots don't inflate it;
 *  a terminator counts only before whitespace/end, so `Orbit.sensitivity` and `0.6.0` don't. */
export function sentenceCount(prose: string): number {
    const text = prose
        .replace(/`[^`]*`/g, "")
        .replace(/\be\.g\./gi, "eg")
        .replace(/\bi\.e\./gi, "ie")
        .replace(/\betc\./gi, "etc");
    return text.match(/[.!?]+(?=\s|$)/g)?.length ?? 0;
}

/** baseline file → the set of canonical lines (comments and blanks dropped). */
export function parseBaseline(text: string): Set<string> {
    const set = new Set<string>();
    for (const raw of text.split("\n")) {
        const line = raw.trim();
        if (line && !line.startsWith("#")) set.add(line);
    }
    return set;
}

const BASELINE_HEADER = `# docs:check baseline — regenerate with \`bun run docs:check --update\`.
# Each class trends to empty as the docs pillar fills it; keep is acknowledged how-to-use prose.
#   bare <page> <Export>     release-public export with no JSDoc (the reference-fill worklist)
#   hole <subsystem>         a tier subsystem (package.json subpath) with no doc page
#   keep <page> <Export>     acknowledged prose mention of a generated reference entry
#   prose <page> <kind>      em dash / banned:<word> / opener:<phrase> in docs prose — burn to zero
#   budget <page> <block>    a #doc:code block over the 3-sentence walkthrough budget — burn to zero
`;

/** canonical baseline text: header + sorted lines. */
export function formatBaseline(lines: Iterable<string>): string {
    const sorted = [...new Set(lines)].sort();
    return `${BASELINE_HEADER}${sorted.join("\n")}\n`;
}

// --- integration ------------------------------------------------------------

function resolvesUnderSrc(s: string): boolean {
    return (
        existsSync(resolve(PKG, s, "index.ts")) ||
        existsSync(resolve(PKG, `${s}.ts`)) ||
        (existsSync(resolve(PKG, s)) && statSync(resolve(PKG, s)).isDirectory())
    );
}

async function gather(): Promise<{ objective: string[]; ratchet: Set<string> }> {
    const objective: string[] = [];
    const ratchet = new Set<string>();

    // 1. rebuild docs fresh; a marker whose source path is missing logs "warning:"
    //    and emits an empty table rather than throwing, so capture both streams.
    const build = Bun.spawnSync(["bun", BUILD]);
    const out = build.stdout.toString() + build.stderr.toString();
    if (build.exitCode !== 0) objective.push(`docs build exited ${build.exitCode}:\n${out}`);
    for (const line of out.split("\n")) {
        if (/warning:/i.test(line)) objective.push(`docs build ${line.trim()}`);
    }

    // 2. dead marker: an unresolved API/CORE marker renders "No exports found".
    for await (const rel of new Glob("**/*.md").scan({ cwd: DIST })) {
        if ((await Bun.file(resolve(DIST, rel)).text()).includes("No exports found")) {
            objective.push(`docs/dist/${rel}: a reference marker resolved to no exports`);
        }
    }

    // 3. coverage holes need the markers across every page, so collect them while
    //    walking sources for the source: check, bare entries, and leak candidates. a page is
    //    either hand-authored (docs/**/*.md) or projected from a manifest entry (assemblePage);
    //    both are the same marker-laden markdown, so the per-page checks below run over both.
    const pages: { rel: string; src: string }[] = [];
    for await (const rel of new Glob("**/*.md").scan({ cwd: DOCS })) {
        if (rel.startsWith("dist/")) continue;
        pages.push({ rel, src: await Bun.file(resolve(DOCS, rel)).text() });
    }
    for (const entry of manifest())
        pages.push({ rel: `${entry.slug}.md`, src: assemblePage(entry) });

    // every page slug a `doc:` link may target: a projected manifest page or a hand-authored one.
    const valid = new Set(pages.map((p) => p.rel.replace(/\.md$/, "")));

    const covered = new Set<string>();
    for (const { rel, src } of pages) {
        const source = src.match(/^source:\s*(.+)$/m);
        if (source && !resolvesUnderSrc(source[1].trim())) {
            objective.push(`docs/${rel}: source: ${source[1].trim()} does not resolve under src/`);
        }
        for (const m of src.matchAll(/<!--\s*(?:API|CORE):(\S+)\s*-->/g)) {
            for (const s of m[1].split(",")) covered.add(markerSubsystem(s));
        }
        for (const l of linkLines(src, valid, rel)) objective.push(l);
        for (const l of proseFindings(proseText(src), rel)) ratchet.add(l);

        const distFile = resolve(DIST, rel);
        if (!existsSync(distFile)) continue;
        const html = await Bun.file(distFile).text();
        for (const l of bareLines(html, rel)) ratchet.add(l);
        for (const l of leakLines(prose(src), anchorNames(html), rel)) ratchet.add(l);
        // JSDoc summaries ship as rendered `ref-desc`, so the prose gate covers them too.
        for (const l of proseFindings(descText(html), rel)) ratchet.add(l);
    }

    // budget: a `#doc:code` walkthrough block is one snippet plus ≤3 sentences (hardening.md). Read the
    // routed code blocks per page (a `page:<slug>` block feeds only its page; an untagged one, every page
    // listing its specimen).
    for (const entry of manifest()) {
        const code = pageBlocks(entry).specimen.filter(
            (b) => b.kind === "code" && (b.page === null || b.page === entry.slug),
        );
        code.forEach((b, i) => {
            if (sentenceCount(b.prose) > 3)
                ratchet.add(`budget ${entry.slug}.md ${b.example ?? `#${i}`}`);
        });
    }

    const pkg = await Bun.file(resolve(dir, "../package.json")).json();
    for (const l of holeLines(requiredSubsystems(pkg.exports), covered)) ratchet.add(l);

    return { objective, ratchet };
}

async function main(): Promise<void> {
    const update = process.argv.includes("--update");
    const { objective, ratchet } = await gather();

    if (objective.length) {
        console.error(`✗ docs integrity:\n${objective.map((f) => `  ${f}`).join("\n")}`);
        process.exit(1);
    }

    if (update) {
        await Bun.write(BASELINE, formatBaseline(ratchet));
        console.log(`✓ docs:check baseline written (${ratchet.size} entries)`);
        return;
    }

    const baseline = existsSync(BASELINE)
        ? parseBaseline(await Bun.file(BASELINE).text())
        : new Set<string>();
    const added = [...ratchet].filter((l) => !baseline.has(l)).sort();
    const removed = [...baseline].filter((l) => !ratchet.has(l)).sort();

    if (added.length || removed.length) {
        console.error("✗ docs:check drift (run `bun run docs:check --update` to accept):");
        for (const l of added) console.error(`  + ${l}`);
        for (const l of removed) console.error(`  - ${l}  (resolved — drop from baseline)`);
        console.error(
            "\n  bare: write the JSDoc.  hole: add the page.  keep: cut the restatement, or accept if it's how-to-use." +
                "\n  prose: cut the em dash / banned word / opener.  budget: trim the #doc:code block to ≤3 sentences.",
        );
        process.exit(1);
    }
}

if (import.meta.main) await main();
