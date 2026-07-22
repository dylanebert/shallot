import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { dirname, isAbsolute, join, relative, resolve } from "path";
import type { Plugin, Rollup, ViteDevServer } from "vite";
import { KNOWN_ENGINE_PLUGINS } from "./engine";
import { generateModule } from "./generate";
import { type Manifest, normalize } from "./manifest";

/**
 * cross-origin isolation headers, applied by every serve surface (`shallot dev`, `shallot run`'s preview,
 * `shallot verify`'s dev/ejected/dist boots). Tumble physics multithreads only when the page can hold a
 * shared `WebAssembly.Memory`, which a browser grants only to a cross-origin-isolated document — so the
 * dev/preview server sends COOP/COEP to enable the multithreaded kernel. A static host that can't set
 * headers (GitHub Pages) gets the single-thread kernel and one log, a documented fallback. The cost of
 * `require-corp`: every cross-origin subresource the page loads must be CORS-approved (a cors-mode fetch
 * against a CORS-enabled host, like extras/text's default gstatic font) or carry CORP — a plain no-cors
 * cross-origin load (`<img src="https://…">` from a host without CORP) is blocked in the isolated
 * document. Consumer-facing note: AGENTS.md "Build, run, verify".
 */
export const CROSS_ORIGIN_ISOLATION = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
};

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

// read + parse a project's manifest, tolerating its absence (a scene-only project → {}); warn loudly on a
// corrupt file or an unknown-plugin key before `normalize` normalizes the mistake away
function readManifest(absDir: string): Manifest {
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

export function discoverScenes(dir: string): string[] {
    const scenes: string[] = [];
    function walk(current: string) {
        for (const entry of readdirSync(current)) {
            if (entry === "node_modules" || entry === "dist") continue;
            const full = join(current, entry);
            if (statSync(full).isDirectory()) walk(full);
            else if (entry.endsWith(".scene")) scenes.push(relative(dir, full));
        }
    }
    try {
        walk(dir);
    } catch {}
    return scenes.sort();
}

export function findPublicDirs(projectDir: string): string[] {
    const dirs: string[] = [];
    const own = join(projectDir, "public");
    if (existsSync(own)) dirs.push(own);
    const parent = join(dirname(projectDir), "public");
    if (existsSync(parent) && parent !== own) dirs.push(parent);
    return dirs;
}

// the glTF importer's two container formats — the unit a live asset-swap watches. A changed `.glb`/`.gltf`
// maps directly to its cache `src`; its external sidecars (`.bin`, separate textures) re-decode through the
// container, which any re-export (Blender, glTF-Transform) rewrites — so watching the container covers the
// re-export workflow. A hand-edit of a sidecar alone (no container rewrite) is the one uncovered case: it
// needs the asset dependency graph the gltf module doesn't track, the deliberate boundary for this sub-stage.
const MODEL_EXT = /\.(glb|gltf)$/i;

/**
 * the glTF asset-cache `src` a changed project file maps to — its path relative to the public dir it sits
 * under (the path a scene's `part="mesh: …#i"` names and `readBinary` fetches, so the key `invalidate`
 * consumes), or `null` if it isn't a `.glb`/`.gltf` under a public dir. Always `/`-separated (a fetch path,
 * not an OS path), so it matches the cache key on Windows too. The watcher uses a match to full-reload
 * on a model change; an unmatched file falls through to the scene/manifest watch.
 */
export function assetSrc(file: string, publicDirs: string[]): string | null {
    if (!MODEL_EXT.test(file)) return null;
    for (const dir of publicDirs) {
        const rel = relative(dir, file);
        if (rel && !rel.startsWith("..") && !isAbsolute(rel)) return rel.replace(/\\/g, "/");
    }
    return null;
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

function contentType(path: string): string | undefined {
    return MIME[path.slice(path.lastIndexOf(".") + 1).toLowerCase()];
}

// signal a project file changing on disk: the dev server has no live edit session to weigh the change
// against, so a full page reload is the clean answer — the page re-imports `virtual:project` (already
// invalidated by the caller) and re-fetches assets.
function signalChange(server: ViteDevServer) {
    server.ws.send({ type: "full-reload" });
}

// serve project public/ assets (the project's own + a shared parent's) with correct MIME
function configureServer(server: ViteDevServer, projectDir?: string) {
    const publicDirs: string[] = projectDir ? findPublicDirs(resolve(projectDir)) : [];
    if (publicDirs.length === 0) return;

    server.middlewares.use((req, res, next) => {
        if (req.url) {
            const pathname = new URL(req.url, "http://localhost").pathname;
            for (const dir of publicDirs) {
                const filePath = join(dir, pathname);
                if (!filePath.startsWith(dir)) continue;
                if (existsSync(filePath) && statSync(filePath).isFile()) {
                    const data = readFileSync(filePath);
                    const mime = contentType(filePath);
                    if (mime) res.setHeader("Content-Type", mime);
                    // dev assets must never sit in the browser HTTP cache, or a live model edit re-fetches
                    // the stale bytes after `invalidate` (the worker decode reads the cached response) and the
                    // swap silently shows the old asset. no-store forces a fresh read every load.
                    res.setHeader("Cache-Control", "no-store");
                    res.end(data);
                    return;
                }
            }
        }
        next();
    });
}

// vite's asset scanner emits an output asset for every `new URL("…", import.meta.url)` at transform
// time — before tree-shaking. So a codec wasm ships even when its importing branch is shaken fully
// dead: orbit imports only `Orbit`, yet draco/basis/audio wasm (~830KB) land in dist/, referenced 0×.
// Walk the finished bundle and return every emitted asset no surviving file references. Conservative —
// an asset is kept the moment its hashed name appears in any reachable chunk or asset, so a codec a
// project actually uses (its `new URL` reference survives in a live chunk) is never dropped. The blind
// spot is an asset addressed by runtime string-building; the `new URL` codecs emit a literal name.
export function orphanedAssets(bundle: Rollup.OutputBundle): string[] {
    const files = Object.values(bundle);
    const text = (f: Rollup.OutputAsset | Rollup.OutputChunk) =>
        f.type === "chunk" ? f.code : typeof f.source === "string" ? f.source : "";
    // never prune a chunk (tree-shaking already pruned JS) or an html entry — seed them as kept roots
    const kept = new Set(files.filter((f) => f.type === "chunk" || f.fileName.endsWith(".html")));
    const assets = files.filter(
        (f): f is Rollup.OutputAsset => f.type === "asset" && !f.fileName.endsWith(".html"),
    );
    // references chain (html → js, css → font, asset → asset), so grow kept to a fixpoint
    let grew = true;
    while (grew) {
        grew = false;
        for (const a of assets) {
            if (kept.has(a)) continue;
            const name = a.fileName.slice(a.fileName.lastIndexOf("/") + 1);
            if ([...kept].some((k) => text(k).includes(name))) {
                kept.add(a);
                grew = true;
            }
        }
    }
    return assets.filter((a) => !kept.has(a)).map((a) => a.fileName);
}

export function projectPlugin(projectDir?: string): Plugin {
    const virtualId = "virtual:project";
    const resolvedId = "\0" + virtualId;
    let viteServer: ViteDevServer | undefined;

    return {
        name: "shallot-project",
        async resolveId(id, importer) {
            if (id === virtualId) return resolvedId;
            // virtual:project is a virtual module with no location, so vite resolves its imports against
            // the host root. A relative local is already absolutized by the generator, but a bare package
            // subpath (a project's installed/workspace plugin, e.g. `orrstead/core/grid`) would miss the
            // host's node_modules. Resolve those from the PROJECT dir, so a manifest can reference an
            // installed plugin by subpath (engine `@dylanebert/shallot` imports resolve here too, to the
            // project's copy).
            if (importer === resolvedId && projectDir) {
                const r = await this.resolve(id, join(resolve(projectDir), "__project__.js"), {
                    skipSelf: true,
                });
                if (r) return r;
            }
        },
        // generate the `virtual:project` module from the project's `shallot.json` — static imports for each
        // enabled plugin (engine via the barrel, locals via their specifier) + the scene + manifest.
        load(id) {
            if (id !== resolvedId) return;
            if (!projectDir) return generateModule({}, null, []);
            const absDir = resolve(projectDir);
            return generateModule(readManifest(absDir), absDir, discoverScenes(absDir));
        },
        configureServer(server) {
            viteServer = server;
            configureServer(server, projectDir);
            if (projectDir) {
                const absDir = resolve(projectDir);
                const publicDirs = findPublicDirs(absDir);
                server.watcher.add(absDir);
                // a shared parent public/ sits outside the project dir, so add it explicitly (the project's
                // own public/ is already covered by absDir) — a model there must still trigger the swap
                for (const pub of publicDirs) if (!pub.startsWith(absDir)) server.watcher.add(pub);
                // a `.scene` add/remove changes the scene list; a `shallot.json` edit changes the plugin
                // set — both re-generate `virtual:project`, so invalidate + reload. Local plugin `.ts`
                // edits ride HMR instead.
                const onProjectFile = (file: string) => {
                    // a model asset (.glb/.gltf) under a public dir → full-reload so the page re-fetches
                    // the model (no virtual:project change — the model isn't part of the generated
                    // module). Checked first, before the absDir guard, so a shared parent public/ is
                    // covered.
                    if (assetSrc(file, publicDirs)) {
                        signalChange(server);
                        return;
                    }
                    if (!file.startsWith(absDir)) return;
                    if (!file.endsWith(".scene") && file !== manifestPath(absDir)) return;
                    const mod = server.moduleGraph.getModuleById(resolvedId);
                    if (mod) server.moduleGraph.invalidateModule(mod);
                    signalChange(server);
                };
                server.watcher.on("change", onProjectFile);
                server.watcher.on("add", onProjectFile);
                server.watcher.on("unlink", onProjectFile);
            }
        },
        handleHotUpdate({ file }) {
            if (!projectDir || !viteServer) return;
            const absDir = resolve(projectDir);
            if (
                (file.endsWith(".scene") || file === manifestPath(absDir)) &&
                file.startsWith(absDir)
            ) {
                const mod = viteServer.moduleGraph.getModuleById(resolvedId);
                if (mod) viteServer.moduleGraph.invalidateModule(mod);
                signalChange(viteServer);
                return [];
            }
            // a project src/*.ts edit falls through to vite's default HMR: `virtual:project` imports the
            // local with no self-accept, so vite full-reloads the page — the clean rebuild path
        },
        // drop the assets vite's `new URL` scanner over-emitted (see orphanedAssets). Build-only (a
        // rollup output hook, never fires in dev), and homed here so every build path inherits it: the
        // synth build (bin/build.ts) and a standalone's own vite.config both run projectPlugin.
        generateBundle(_options, bundle) {
            const orphans = orphanedAssets(bundle);
            if (!orphans.length) return;
            let bytes = 0;
            for (const fileName of orphans) {
                const a = bundle[fileName];
                if (a?.type === "asset")
                    bytes += typeof a.source === "string" ? a.source.length : a.source.byteLength;
                delete bundle[fileName];
            }
            this.info(`pruned ${orphans.length} orphaned asset(s), ${(bytes / 1024) | 0}KB`);
        },
    };
}
