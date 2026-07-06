#!/usr/bin/env bun
import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { Glob } from "bun";
import { assemblePage, manifest } from "./literate";

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
 *  - a `source:` front-matter that doesn't resolve under src/.
 *
 * Ratchet (baseline-diff against docs-baseline.txt, the API-Extractor shape):
 *  - bare:  a release-public export with no JSDoc (generates a bare ref-item).
 *  - hole:  a tier subsystem (a package.json core or named subpath) with no page.
 *  - keep:  a prose mention of a generated reference entry (a leak candidate).
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
#   bare <page> <Export>   release-public export with no JSDoc (the reference-fill worklist)
#   hole <subsystem>       a tier subsystem (package.json subpath) with no doc page
#   keep <page> <Export>   acknowledged prose mention of a generated reference entry
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

    const covered = new Set<string>();
    for (const { rel, src } of pages) {
        const source = src.match(/^source:\s*(.+)$/m);
        if (source && !resolvesUnderSrc(source[1].trim())) {
            objective.push(`docs/${rel}: source: ${source[1].trim()} does not resolve under src/`);
        }
        for (const m of src.matchAll(/<!--\s*(?:API|CORE):(\S+)\s*-->/g)) {
            for (const s of m[1].split(",")) covered.add(markerSubsystem(s));
        }

        const distFile = resolve(DIST, rel);
        if (!existsSync(distFile)) continue;
        const html = await Bun.file(distFile).text();
        for (const l of bareLines(html, rel)) ratchet.add(l);
        for (const l of leakLines(prose(src), anchorNames(html), rel)) ratchet.add(l);
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
            "\n  bare: write the JSDoc.  hole: add the page.  keep: cut the restatement, or accept if it's how-to-use.",
        );
        process.exit(1);
    }
}

if (import.meta.main) await main();
