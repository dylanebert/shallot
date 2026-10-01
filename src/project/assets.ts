// Static-asset readers shared by project tooling.

import { existsSync, statSync } from "fs";
import { join, sep } from "path";
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
