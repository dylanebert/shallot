// the pure readers behind the `./vite` plugin entry: a project's manifest, and the two functions its dev
// server's static-asset middleware is built from. Internal to `src/project/` — imported by `vite.ts` and
// by this file's own test, never re-exported from the published subpath (`exports.md` "Barrel rules": a
// module-internal export shared across sibling files lives in a sibling imported directly, the shape
// `sear/codegen.ts` already uses). `manifestPath` and `manifestWarnings` are the exception, re-exported by
// `vite.ts` because the CLI and consumers already resolve them through `@dylanebert/shallot/vite`.
//
// Nothing here touches vite, a server, or a hook — data in, data out — so the plugin entry keeps only the
// hooks, and these read as ordinary functions a test can call.

import { existsSync, readFileSync, statSync } from "fs";
import { join, sep } from "path";
import { KNOWN_ENGINE_PLUGINS } from "./engine";
import { type Manifest, normalize } from "./manifest";

/** a project's `shallot.json` manifest path — the project descriptor the toolchain reads. */
export function manifestPath(dir: string): string {
    return join(dir, "shallot.json");
}

/**
 * the toolchain-boundary warnings for a project's raw `shallot.json` text: an unparseable file (else
 * `normalize` silently swallows it to `{}`), and a bool key naming no engine plugin (else it surfaces
 * only as a cryptic esbuild "no export named ${name}Plugin" at bundle time). Pure over (raw, known) so
 * the project test pins both paths without touching disk or spying on the console. `readManifest` emits
 * each with the file path prefixed.
 */
export function manifestWarnings(raw: string, known: ReadonlySet<string>): string[] {
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        return ["not valid JSON, ignored (the project runs with default plugins)"];
    }
    const plugins = (parsed as { plugins?: unknown })?.plugins;
    if (typeof plugins !== "object" || plugins === null || Array.isArray(plugins)) return [];
    const warnings: string[] = [];
    for (const [name, value] of Object.entries(plugins)) {
        // a bool declares an engine plugin (a local uses a specifier); a bool outside the known set names
        // no engine plugin, so the generator's `${name}Plugin` import would miss
        if (typeof value === "boolean" && !known.has(name)) {
            warnings.push(
                `"${name}" is not a known engine plugin (use a module specifier for a local plugin)`,
            );
        }
    }
    return warnings;
}

/**
 * read + parse a project's manifest, tolerating its absence (a scene-only project → {}); warn loudly on
 * a corrupt file or an unknown-plugin key before `normalize` normalizes the mistake away.
 */
export function readManifest(absDir: string): Manifest {
    const path = manifestPath(absDir);
    let text: string;
    try {
        text = readFileSync(path, "utf-8");
    } catch {
        return {}; // no manifest — a scene-only project
    }
    for (const w of manifestWarnings(text, KNOWN_ENGINE_PLUGINS)) console.warn(`  ! ${path}: ${w}`);
    return normalize(text);
}

// MIME for project public assets. Without it `res.end(data)` sends no Content-Type, so an SVG served
// at `/icon.svg` (a project's icon) is rejected as a favicon and the tab falls back to a generic
// icon. Covers the asset types a project's public/ holds.
const MIME: Record<string, string> = {
    svg: "image/svg+xml",
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    webp: "image/webp",
    gif: "image/gif",
    ico: "image/x-icon",
    json: "application/json",
    scene: "text/plain; charset=utf-8",
    wasm: "application/wasm",
    glb: "model/gltf-binary",
    gltf: "model/gltf+json",
    bin: "application/octet-stream",
    ktx2: "image/ktx2",
};

/** the Content-Type for a served project asset, or undefined for a type the middleware doesn't name. */
export function contentType(path: string): string | undefined {
    return MIME[path.slice(path.lastIndexOf(".") + 1).toLowerCase()];
}

/**
 * the served file `pathname` maps to under `dir`, or `null` if there isn't one — joins, then rejects a
 * result that lands outside `dir` before ever touching disk. `configureServer`'s own `new URL(req.url,
 * "http://localhost")` call already strips every dot-segment payload upstream (WHATWG's path-normalize
 * step resolves "." / ".." — including percent-encoded forms — before this ever sees `pathname`, verified
 * against curl-style raw request lines), so the escape branch is unreachable through any real request;
 * it's tested directly here, on a raw `pathname` the URL layer never gets to normalize first.
 */
export function resolveAssetPath(dir: string, pathname: string): string | null {
    const filePath = join(dir, pathname);
    // segment boundary, not a string prefix: a plain `startsWith(dir)` also admits a *sibling* whose
    // name extends dir's basename (dir `/a/public` would accept `/a/public-secrets/x`), which is an
    // escape, not a descendant.
    if (filePath !== dir && !filePath.startsWith(dir + sep)) return null;
    return existsSync(filePath) && statSync(filePath).isFile() ? filePath : null;
}
