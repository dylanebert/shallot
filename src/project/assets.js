// Static-asset readers shared by project tooling.

import { existsSync, statSync } from "node:fs";
import { join, sep } from "node:path";

// MIME for project public assets. Without it `res.end(data)` sends no Content-Type, so an SVG served
// at `/icon.svg` (a project's icon) is rejected as a favicon and the tab falls back to a generic
// icon. Covers the asset types a project's public/ holds.
/** @type {Record<string, string>} */
const MIME = {
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

/** The Content-Type for a served project asset, or undefined for an unnamed type. @param {string} path */
export function contentType(path) {
    return MIME[path.slice(path.lastIndexOf(".") + 1).toLowerCase()];
}

/**
 * Maps the served pathname under dir, or returns null if none exists. Joins, then rejects a result
 * outside dir before touching disk. The server's URL parsing strips dot segments upstream, but the
 * boundary check also protects direct callers.
 * @param {string} dir
 * @param {string} pathname
 * @returns {string | null}
 */
export function resolveAssetPath(dir, pathname) {
    const filePath = join(dir, pathname);
    // Segment boundary, not a string prefix: a plain `startsWith(dir)` also admits a sibling whose
    // name extends dir's basename.
    if (filePath !== dir && !filePath.startsWith(dir + sep)) return null;
    return existsSync(filePath) && statSync(filePath).isFile() ? filePath : null;
}
