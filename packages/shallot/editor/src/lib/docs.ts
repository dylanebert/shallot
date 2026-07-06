// In-editor docs: pure transforms over the docs/dist artifact built by
// packages/shallot/scripts/docs.ts. That artifact is render-ready HTML with frontmatter and
// tab markers (`<!-- tabs -->`, `<!-- tab: Name -->`, `<!-- os -->`/`<!-- pick -->`); these
// functions split it into the block model a panel renders, plus a search index and a
// component→reference map for context-sensitive help. No markdown or highlighter runtime lives
// here — the build already rendered everything. The glob that feeds buildIndex lives in
// ./docs.corpus so this module stays pure and bun-testable (no import.meta.glob).

export type ContentBlock = { type: "content"; html: string };
export type InlineGroup = { type: "inline"; tabs: { name: string; html: string }[] };
export type AudienceTab = { name: string; blocks: (ContentBlock | InlineGroup)[] };
export type TabGroup = { type: "tabs"; tabs: AudienceTab[] };
export type Block = ContentBlock | TabGroup | InlineGroup;

export interface DocPage {
    slug: string;
    title: string;
    description: string;
    source: string;
    icon: string;
    group: string;
    order: number;
    blocks: Block[];
    tabNames: string[];
    text: string;
}

export interface DocSymbol {
    name: string;
    slug: string;
    anchor: string;
}

export interface DocIndex {
    pages: DocPage[];
    symbols: DocSymbol[];
}

export interface SearchHit {
    kind: "page" | "symbol";
    title: string;
    slug: string;
    group: string;
    anchor?: string;
}

function frontmatter(raw: string): { meta: Record<string, string>; body: string } {
    const m = raw.match(/^---\n([\s\S]*?)\n---\n/);
    if (!m) return { meta: {}, body: raw };
    const meta: Record<string, string> = {};
    for (const line of m[1].split("\n")) {
        const i = line.indexOf(":");
        if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    }
    return { meta, body: raw.slice(m[0].length) };
}

function parseInlineTabs(html: string, tag: string): (string | InlineGroup)[] {
    const parts = html.split(new RegExp(`<!--\\s*${tag}\\s*-->`));
    const result: (string | InlineGroup)[] = [];

    for (let i = 0; i < parts.length; i++) {
        if (i === 0) {
            if (parts[0].trim()) result.push(parts[0]);
            continue;
        }
        const closeSplit = parts[i].split(new RegExp(`<!--\\s*\\/${tag}\\s*-->`));
        const groupHtml = closeSplit[0];
        const after = closeSplit.slice(1).join("");

        const tabMatches = [
            ...groupHtml.matchAll(new RegExp(`<!--\\s*${tag}:\\s*(.+?)\\s*-->`, "g")),
        ];
        const tabs: { name: string; html: string }[] = [];

        for (let j = 0; j < tabMatches.length; j++) {
            const name = tabMatches[j][1];
            const start = tabMatches[j].index! + tabMatches[j][0].length;
            const end = j + 1 < tabMatches.length ? tabMatches[j + 1].index! : groupHtml.length;
            tabs.push({ name, html: groupHtml.slice(start, end) });
        }

        if (tabs.length > 0) result.push({ type: "inline", tabs });
        if (after.trim()) result.push(after);
    }
    return result;
}

const inlineTags = ["os", "pick"];

function expandInline(html: string, blocks: Block[]): void {
    let segments: (string | InlineGroup)[] = [html];
    for (const tag of inlineTags) {
        const next: (string | InlineGroup)[] = [];
        for (const seg of segments) {
            if (typeof seg === "string") next.push(...parseInlineTabs(seg, tag));
            else next.push(seg);
        }
        segments = next;
    }
    for (const seg of segments) {
        if (typeof seg === "string") blocks.push({ type: "content", html: seg });
        else blocks.push(seg);
    }
}

/** split rendered doc HTML into the audience-tab / inline-tab block model.
 * Mirrors the website's reader so both views over docs/dist stay identical.
 * @example
 * const { blocks, tabNames } = parseBlocks(html);
 */
export function parseBlocks(html: string): { blocks: Block[]; tabNames: string[] } {
    const blocks: Block[] = [];
    const tabNames = new Set<string>();
    const parts = html.split(/<!--\s*tabs\s*-->/);

    for (let i = 0; i < parts.length; i++) {
        if (i === 0) {
            const trimmed = parts[0].trim();
            if (trimmed) expandInline(trimmed, blocks);
            continue;
        }

        const closeSplit = parts[i].split(/<!--\s*\/tabs\s*-->/);
        const groupHtml = closeSplit[0];
        const after = closeSplit.slice(1).join("").trim();

        const tabMatches = [...groupHtml.matchAll(/<!--\s*tab:\s*(.+?)\s*-->/g)];
        const tabs: AudienceTab[] = [];

        for (let j = 0; j < tabMatches.length; j++) {
            const name = tabMatches[j][1];
            tabNames.add(name);
            const start = tabMatches[j].index! + tabMatches[j][0].length;
            const end = j + 1 < tabMatches.length ? tabMatches[j + 1].index! : groupHtml.length;
            const inner: (ContentBlock | InlineGroup)[] = [];
            expandInline(groupHtml.slice(start, end), inner);
            tabs.push({ name, blocks: inner });
        }

        if (tabs.length > 0) blocks.push({ type: "tabs", tabs });
        if (after) expandInline(after, blocks);
    }

    return { blocks, tabNames: [...tabNames] };
}

// category order in the browse list; an unlisted group sorts after these
const GROUP_ORDER = ["guide", "engine", "standard", "extras", "editor"];
function groupRank(group: string): number {
    const i = GROUP_ORDER.indexOf(group);
    return i === -1 ? GROUP_ORDER.length : i;
}

/** build the page index + reference-symbol map from a globbed docs/dist corpus
 * (path → raw file). Pages sort by category, then frontmatter order, then slug; the index page is dropped. */
export function buildIndex(corpus: Record<string, string>): DocIndex {
    const pages: DocPage[] = [];
    const symbols: DocSymbol[] = [];

    for (const [key, raw] of Object.entries(corpus)) {
        const slug = key.replace(/^.*\/docs\/dist\//, "").replace(/\.md$/, "");
        if (slug === "index") continue;
        const { meta, body } = frontmatter(raw);
        const { blocks, tabNames } = parseBlocks(body);
        pages.push({
            slug,
            title: meta.title ?? slug,
            description: meta.description ?? "",
            source: meta.source ?? "",
            icon: meta.icon ?? "",
            group: slug.includes("/") ? slug.split("/")[0] : "",
            order: meta.order ? Number.parseInt(meta.order, 10) : 99,
            blocks,
            tabNames,
            text: body
                .replace(/<[^>]+>/g, " ")
                .replace(/\s+/g, " ")
                .trim(),
        });

        const seen = new Set<string>();
        for (const m of body.matchAll(/id="ref-([A-Za-z0-9_$]+)"/g)) {
            if (seen.has(m[1])) continue;
            seen.add(m[1]);
            symbols.push({ name: m[1], slug, anchor: `ref-${m[1]}` });
        }
    }

    pages.sort(
        (a, b) =>
            groupRank(a.group) - groupRank(b.group) ||
            a.order - b.order ||
            a.slug.localeCompare(b.slug),
    );
    return { pages, symbols };
}

function isSubsequence(text: string, q: string): boolean {
    let i = 0;
    for (const c of text) if (c === q[i] && ++i === q.length) return true;
    return q.length === 0;
}

function score(text: string, q: string): number {
    if (text === q) return 100;
    if (text.startsWith(q)) return 50;
    const i = text.indexOf(q);
    if (i >= 0) return 30 - i * 0.1;
    return isSubsequence(text, q) ? 10 : 0;
}

/** rank doc pages and reference symbols against a query. Symbol-name and page-title matches
 * win over body-text matches; exact > prefix > substring > subsequence. */
export function search(index: DocIndex, query: string, limit = 20): SearchHit[] {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const ranked: { hit: SearchHit; s: number }[] = [];

    for (const p of index.pages) {
        const title = score(p.title.toLowerCase(), q);
        const s = title || (p.text.toLowerCase().includes(q) ? 5 : 0);
        if (s)
            ranked.push({ hit: { kind: "page", title: p.title, slug: p.slug, group: p.group }, s });
    }
    for (const sym of index.symbols) {
        const s = score(sym.name.toLowerCase(), q);
        if (s) {
            const page = index.pages.find((p) => p.slug === sym.slug);
            ranked.push({
                hit: {
                    kind: "symbol",
                    title: sym.name,
                    slug: sym.slug,
                    anchor: sym.anchor,
                    group: page?.group ?? "",
                },
                s,
            });
        }
    }

    ranked.sort((a, b) => b.s - a.s || a.hit.title.localeCompare(b.hit.title));
    return ranked.slice(0, limit).map((r) => r.hit);
}

/** resolve a scene component name (`orbit`, `directional-light`) to the doc page + reference
 * anchor that documents it — the seam for context-sensitive help from the inspector. Prefers a
 * reference symbol (PascalCased name), falls back to the page whose slug matches. */
export function docFor(
    index: DocIndex,
    component: string,
): { slug: string; anchor?: string } | null {
    const pascal = component
        .split(/[-_]/)
        .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
        .join("");
    const sym = index.symbols.find((s) => s.name === pascal);
    if (sym) return { slug: sym.slug, anchor: sym.anchor };
    const page = index.pages.find((p) => p.slug === component || p.slug.endsWith(`/${component}`));
    return page ? { slug: page.slug } : null;
}
