import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

// zoo specimens are the docs' teaching anchor: minimal editor-openable projects (one dir per module
// under examples/zoo/, each a `src/tune.ts` config + a scene), compile-gated in `bun check`. A page's
// Code/Scene snippets are `<!-- EXAMPLE:path#region -->` extractions from a specimen (the educational
// twin of `<!-- API: -->`), so the shown code can't drift from code that compiles. `path` is relative
// to examples/zoo/ (`orbit/src/tune.ts`); `#region` (a `// #region name` … `// #endregion` fold block)
// is optional — omit it to extract a whole file (a `.scene`). Fence language follows the extension.

const ZOO = resolve(import.meta.dir, "../../../examples/zoo");

/** drop the deepest common leading whitespace so an extracted region reads at column zero. */
function dedent(lines: string[]): string[] {
    const indents = lines.filter((l) => l.trim()).map((l) => l.match(/^\s*/)![0].length);
    const min = indents.length ? Math.min(...indents) : 0;
    return lines.map((l) => l.slice(min));
}

/**
 * slice a `// #region <name>` … `// #endregion` block out of source, dedented, marker lines
 * dropped. Returns null when the region is absent or unterminated.
 */
export function sliceRegion(source: string, region: string): string | null {
    const lines = source.split("\n");
    const start = lines.findIndex((l) => l.trim() === `// #region ${region}`);
    if (start === -1) return null;
    const body: string[] = [];
    for (let i = start + 1; i < lines.length; i++) {
        const t = lines[i].trim();
        if (t === "// #endregion") return dedent(body).join("\n").trim();
        if (t.startsWith("// #region ")) continue; // tolerate a stray nested marker
        body.push(lines[i]);
    }
    return null;
}

/** fence language for a specimen file: `.scene` is the engine's XML scene format, `.json` a manifest, else TS. */
function lang(path: string): string {
    if (path.endsWith(".scene") || path.endsWith(".xml")) return "xml";
    if (path.endsWith(".json")) return "json";
    return "typescript";
}

/** the fenced code block a `<!-- EXAMPLE:path#region -->` marker expands to, or an error string. */
export function exampleBlock(path: string, region?: string): { code?: string; error?: string } {
    const file = resolve(ZOO, path);
    if (!existsSync(file)) return { error: `no specimen examples/zoo/${path}` };
    const text = readFileSync(file, "utf-8");
    const code = region ? sliceRegion(text, region) : text.trim();
    if (code === null) return { error: `no region "${region}" in ${path}` };
    return { code: `\`\`\`${lang(path)}\n${code}\n\`\`\`` };
}
