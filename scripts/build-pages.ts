import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
    DARK,
    fromBlocks,
    LIGHT,
    lockup,
    MARK,
    toCells,
    toSvg,
    toText,
    word,
} from "../site/brand/mark";
import { brandPage } from "../site/brand/page";
import { toPng } from "../site/brand/png";
import { llmsTxt, siteIndex } from "../site/home";
import { ROSTER } from "../site/roster";
import { datadogInitSnippet } from "../site/rum-config";

// `bun run site:pages` — the site's own pages without the demos: out/site/index.html, llms.txt and
// out/site/brand/ with its downloads. `build-site.ts` calls the same function after the demo
// loop, so this is also how to iterate on the pages locally without a demo build. Serve
// out/site with any static server to view. Run alone it builds in staging mode: every link and
// label names the local commit, never a release tag the tree may be ahead of. Production labels
// come from `bun run site`, which pins to the published version.

const root = resolve(import.meta.dir, "..");

export async function bundleClient(): Promise<string> {
    const result = await Bun.build({
        entrypoints: [resolve(root, "site/brand/client.ts")],
        target: "browser",
        minify: true,
    });
    if (!result.success) {
        for (const log of result.logs) console.error(log);
        throw new Error("failed to bundle site/brand/client.ts");
    }
    const output = result.outputs[0];
    if (!output) throw new Error("site/brand/client.ts bundle produced no output");
    return await output.text();
}

/** Centre a grid in a square frame; the favicon frame around the 12×14 mark. */
function framed(grid: ReturnType<typeof fromBlocks>, size: number): ReturnType<typeof fromBlocks> {
    const w = grid[0]?.length ?? 0;
    const h = grid.length;
    const ox = Math.floor((size - w) / 2);
    const oy = Math.floor((size - h) / 2);
    return Array.from({ length: size }, (_, y) =>
        Array.from({ length: size }, (_, x) => grid[y - oy]?.[x - ox] ?? null),
    );
}

/** Writes the brand page and every download into `out/brand/`. */
export async function buildBrand(
    outDir: string,
    clientScript?: string,
    rum: string = "",
): Promise<void> {
    const dir = resolve(outDir, "brand");
    mkdirSync(dir, { recursive: true });
    const mark = fromBlocks(MARK.m);
    const lock = lockup();
    const write = (name: string, data: string | Uint8Array) =>
        writeFileSync(resolve(dir, name), data);
    write("index.html", brandPage(clientScript ?? (await bundleClient()), rum));
    write("mark.svg", toSvg(mark, DARK, 1));
    write("mark.png", toPng(mark, DARK, 8));
    write("mark-16.png", toPng(framed(mark, 16), DARK, 1));
    write("mark-32.png", toPng(framed(mark, 16), DARK, 2));
    write("lockup-dark.svg", toSvg(lock, DARK, 1));
    write("lockup-light.svg", toSvg(lock, LIGHT, 1));
    write("lockup-dark.png", toPng(lock, DARK, 4, DARK.bg));
    write("lockup-light.png", toPng(lock, LIGHT, 4, LIGHT.bg));
    write("wordmark.svg", toSvg(word(), DARK, 1));
    write("wordmark.png", toPng(word(), DARK, 8));
    write("mark.txt", `${toText(toCells(mark))}\n`);
    write("mark.ts", readFileSync(resolve(root, "site/brand/mark.ts"), "utf8"));
}

/** Writes the home index, `llms.txt`, and the brand pages. `rumMode` picks the Datadog env
 * derivation the pages carry: the hostname-derived prod snippet tags a localhost preview
 * "local", so a standalone pages build uses it even while labelling itself staging. */
export async function buildPages(
    outDir: string,
    version: string,
    ref: string,
    mode: "prod" | "staging",
    rumMode: "prod" | "staging" = mode,
): Promise<void> {
    mkdirSync(outDir, { recursive: true });
    const client = await bundleClient();
    const rum = datadogInitSnippet(rumMode);
    writeFileSync(
        resolve(outDir, "index.html"),
        siteIndex(ROSTER, version, ref, mode, client, rum),
    );
    writeFileSync(resolve(outDir, "llms.txt"), llmsTxt(version, ref, mode));
    await buildBrand(outDir, client, rum);
}

if (import.meta.main) {
    const pkg = JSON.parse(
        readFileSync(resolve(root, "packages/shallot/package.json"), "utf8"),
    ) as {
        version: string;
    };
    const ref = Bun.spawnSync(["git", "rev-parse", "--short", "HEAD"], { cwd: root });
    const refShort = ref.stdout.toString().trim() || "unknown";
    const outDir = resolve(root, "out/site");
    await buildPages(outDir, pkg.version, refShort, "staging", "prod");
    console.log(
        `pages (staging · ${refShort}): ${outDir}/index.html, ${outDir}/llms.txt, ${outDir}/brand/`,
    );
}
