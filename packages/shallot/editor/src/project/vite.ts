import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "fs";
import type { IncomingMessage, ServerResponse } from "http";
import { dirname, isAbsolute, join, relative, resolve, sep } from "path";
import type { Plugin, Rollup, ViteDevServer } from "vite";
import { generateModule } from "./generate";
import { type Manifest, normalize } from "./manifest";

/** a project's `shallot.json` manifest path — the project descriptor the editor owns + dev mode reads. */
export function manifestPath(dir: string): string {
    return join(dir, "shallot.json");
}

// read + parse a project's manifest, tolerating its absence (a scene-only project → {})
function readManifest(absDir: string): Manifest {
    try {
        return normalize(readFileSync(manifestPath(absDir), "utf-8"));
    } catch {
        return {};
    }
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
 * not an OS path), so it matches the cache key on Windows too. The editor passes the src to `invalidate`
 * to re-decode just that asset; an unmatched file falls through to the scene/manifest watch.
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
// at `/icon.svg` (a project icon shadowing the editor's) is rejected as a favicon and the tab falls
// back to a generic icon. Covers the asset types a project's public/ holds.
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

function readBody(req: IncomingMessage): Promise<string> {
    return new Promise((resolve) => {
        let body = "";
        req.on("data", (chunk: Buffer) => {
            body += chunk.toString();
        });
        req.on("end", () => resolve(body));
    });
}

// the raw-bytes twin of readBody, for `/__api/asset` — a .glb through `chunk.toString()` is lossy
// (UTF-8 replacement characters), so binary bodies concatenate untouched
function readRawBody(req: IncomingMessage): Promise<Buffer> {
    return new Promise((resolve) => {
        const chunks: Buffer[] = [];
        req.on("data", (chunk: Buffer) => chunks.push(chunk));
        req.on("end", () => resolve(Buffer.concat(chunks)));
    });
}

function json(res: ServerResponse, data: unknown) {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(data));
}

function error(res: ServerResponse, status: number, message: string) {
    res.statusCode = status;
    json(res, { error: message });
}

const recentSaves = new Set<string>();

/** true while a just-written path sits in the editor's own-write window — the watcher skips signaling
 *  it, so an editor save (scene, manifest, or an `/__api/asset` upload) never bounces back as an
 *  external-change reload / asset invalidate. */
export function recentlySaved(file: string): boolean {
    return recentSaves.has(file);
}

function markSaved(file: string): void {
    recentSaves.add(file);
    setTimeout(() => recentSaves.delete(file), 2000);
}

/** result of an `/__api/asset` write: `src` is the final public-relative path (deduped on a
 *  byte-differing collision, reused verbatim on an identical one), `error` + non-200 `status` on a
 *  rejected write. */
export interface AssetWrite {
    status: number;
    src?: string;
    error?: string;
}

/**
 * write one uploaded asset under a project's `public/` dir — the binary write behind `POST /__api/asset`.
 * Traversal-guarded (the `/__api/file` guard, closed over the sibling-prefix hole with a separator-aware
 * compare). Collision policy mirrors `mintId`: identical bytes reuse the existing file (no write, no
 * watcher event); differing bytes land on the first free `name-2.ext` / `name-3.ext`, so an upload never
 * silently overwrites. A real write is marked as the editor's own (`recentlySaved`), so the public-dir
 * watcher doesn't bounce it back as a `shallot:asset` invalidate — the import flow loads the bytes live.
 */
export function writeAsset(publicDir: string, name: string, bytes: Buffer): AssetWrite {
    const absDir = resolve(publicDir);
    const absPath = resolve(absDir, name);
    if (absPath !== absDir && !absPath.startsWith(absDir + sep)) {
        return { status: 403, error: "Path traversal" };
    }
    const file = absPath.slice(absPath.lastIndexOf(sep) + 1);
    const ext = file.includes(".") ? file.slice(file.lastIndexOf(".")) : "";
    const stem = absPath.slice(0, absPath.length - ext.length);
    let target = absPath;
    for (let n = 2; existsSync(target); n++) {
        if (bytes.equals(readFileSync(target))) {
            return { status: 200, src: relative(absDir, target).replace(/\\/g, "/") };
        }
        target = `${stem}-${n}${ext}`;
    }
    mkdirSync(dirname(target), { recursive: true });
    markSaved(target);
    writeFileSync(target, bytes);
    return { status: 200, src: relative(absDir, target).replace(/\\/g, "/") };
}

// signal a project file changing on disk (after `recentSaves` filtered out the editor's own write). The
// editor must weigh the change against its unsaved state — reload when clean, a resolvable conflict when
// not — so it needs a typed event (`shallot:external`) it can intercept; a page reload would nuke unsaved
// edits + session state. A standalone dev server has no interceptor and no edit session, so it just reloads.
function signalChange(server: ViteDevServer, editor: boolean, dir: string, file: string) {
    if (!editor) {
        server.ws.send({ type: "full-reload" });
        return;
    }
    server.ws.send("shallot:external", {
        path: relative(dir, file),
        manifest: file === manifestPath(dir),
    });
}

// signal a model asset (.glb/.gltf) changing on disk → the editor drops it from the glTF asset cache and
// rebuilds (re-decodes off-thread, no page reload — the live-iteration loop). Unlike a scene/manifest
// change, the asset isn't the editor's document — its `<a gltf>` nodes re-spawn unchanged — so there's no
// unsaved-edit conflict to weigh and the editor reloads it unconditionally. A standalone dev server has no
// editor to do the in-place swap, so it full-reloads (re-fetches the model), mirroring `signalChange`.
function signalAsset(server: ViteDevServer, editor: boolean, src: string) {
    if (!editor) {
        server.ws.send({ type: "full-reload" });
        return;
    }
    server.ws.send("shallot:asset", { src });
}

function configureServer(server: ViteDevServer, projectDir?: string) {
    const publicDirs: string[] = projectDir ? findPublicDirs(resolve(projectDir)) : [];

    const logger = server.config.logger;
    const origInfo = logger.info.bind(logger);
    const origWarn = logger.warn.bind(logger);
    const origError = logger.error.bind(logger);
    // vite prefixes browser-forwarded console messages with `[console.*]`. The editor already captures the
    // browser console directly (App.svelte overrides), so echoing those back over the ws would re-emit them
    // in the browser, vite would re-forward them prefixed again, and one message becomes an unbounded
    // `[console.error] [console.error] …` flood. Only forward genuine server-side logs.
    const fromBrowser = (msg: string) => msg.includes("[console.");
    logger.info = (msg, opts) => {
        origInfo(msg, opts);
        if (!fromBrowser(msg)) server.ws.send("shallot:log", { level: "info", message: msg });
    };
    logger.warn = (msg, opts) => {
        origWarn(msg, opts);
        if (!fromBrowser(msg)) server.ws.send("shallot:log", { level: "warn", message: msg });
    };
    logger.error = (msg, opts) => {
        origError(msg, opts);
        if (!fromBrowser(msg)) server.ws.send("shallot:log", { level: "error", message: msg });
    };

    const pending = new Map<string, (result: unknown) => void>();
    let nextId = 0;
    server.ws.on("shallot:response", (data: { id: string; result: unknown }) => {
        const resolve = pending.get(data.id);
        if (resolve) {
            pending.delete(data.id);
            resolve(data.result);
        }
    });

    function request(type: string, payload?: unknown): Promise<unknown> {
        const id = String(++nextId);
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                pending.delete(id);
                reject(new Error("No editor connected"));
            }, 5000);
            pending.set(id, (result) => {
                clearTimeout(timeout);
                resolve(result);
            });
            server.ws.send("shallot:request", { id, type, payload });
        });
    }

    server.middlewares.use((req, res, next) => {
        const url = new URL(req.url || "", "http://localhost");

        if (url.pathname === "/__api/scene" && req.method === "GET") {
            const dir = url.searchParams.get("dir");
            const path = url.searchParams.get("path");
            if (!dir || !path) {
                res.statusCode = 400;
                res.end("Missing dir or path");
                return;
            }
            const absDir = resolve(dir);
            const absPath = resolve(absDir, path);
            if (!absPath.startsWith(absDir)) {
                res.statusCode = 403;
                res.end("Path traversal");
                return;
            }
            try {
                const content = readFileSync(absPath, "utf-8");
                res.setHeader("Content-Type", "text/plain");
                res.end(content);
            } catch {
                res.statusCode = 404;
                res.end("Scene not found");
            }
            return;
        }

        if (url.pathname === "/__api/scene" && req.method === "POST") {
            const dir = url.searchParams.get("dir");
            const path = url.searchParams.get("path");
            if (!dir || !path) {
                res.statusCode = 400;
                res.end("Missing dir or path");
                return;
            }
            const absDir = resolve(dir);
            const absPath = resolve(absDir, path);
            if (!absPath.startsWith(absDir)) {
                res.statusCode = 403;
                res.end("Path traversal");
                return;
            }
            readBody(req).then((body) => {
                try {
                    const { content } = JSON.parse(body);
                    markSaved(absPath);
                    writeFileSync(absPath, content, "utf-8");
                    res.setHeader("Content-Type", "application/json");
                    res.end(JSON.stringify({ ok: true }));
                } catch (e) {
                    res.statusCode = 500;
                    res.end(String(e));
                }
            });
            return;
        }

        // read / write the project's `shallot.json` manifest — the editor's plugin-enablement writes land
        // here. The fixed path (no user-supplied path) needs no traversal guard; the editor's own write is
        // deduped via `recentSaves` so the watcher doesn't reload the editor on top of its reactive rebuild.
        if (url.pathname === "/__api/manifest") {
            // the manifest is the served project's, so `dir` defaults to it — the editor passes its
            // active dir, a tool (capture) can omit it and get the project the server already knows.
            const dir = url.searchParams.get("dir") ?? projectDir;
            if (!dir) {
                error(res, 400, "Missing dir");
                return;
            }
            const absPath = manifestPath(resolve(dir));
            if (req.method === "GET") {
                try {
                    res.setHeader("Content-Type", "application/json");
                    res.end(readFileSync(absPath, "utf-8"));
                } catch {
                    res.setHeader("Content-Type", "application/json");
                    res.end("{}"); // no manifest yet — a scene-only project
                }
                return;
            }
            if (req.method === "POST") {
                readBody(req).then((body) => {
                    try {
                        const { content } = JSON.parse(body);
                        markSaved(absPath);
                        writeFileSync(absPath, content, "utf-8");
                        json(res, { ok: true });
                    } catch (e) {
                        error(res, 500, String(e));
                    }
                });
                return;
            }
        }

        // read/write a project source file, relative to the served project dir (path-traversal guarded).
        // The editor↔code write seam: the `hot-reload` capture flow edits a fixture plugin through this so
        // the dev server's own writeFileSync triggers the file watcher → virtual:project HMR accept → swap.
        // Routing the write through the server (not the test's fs) is what makes it work when the runner is
        // remote (WSL → Windows Chrome): the write lands on the filesystem the watcher actually watches.
        if (url.pathname === "/__api/file") {
            const path = url.searchParams.get("path");
            if (!projectDir || !path) {
                error(res, 400, "Missing project dir or path");
                return;
            }
            const absDir = resolve(projectDir);
            const absPath = resolve(absDir, path);
            if (!absPath.startsWith(absDir)) {
                error(res, 403, "Path traversal");
                return;
            }
            if (req.method === "GET") {
                try {
                    res.setHeader("Content-Type", "text/plain");
                    res.end(readFileSync(absPath, "utf-8"));
                } catch {
                    error(res, 404, "File not found");
                }
                return;
            }
            if (req.method === "POST") {
                readBody(req).then((body) => {
                    try {
                        writeFileSync(absPath, JSON.parse(body).content, "utf-8");
                        json(res, { ok: true });
                    } catch (e) {
                        error(res, 500, String(e));
                    }
                });
                return;
            }
        }

        // write one uploaded binary asset under the project's own public/ (created if absent) — the
        // editor's import-upload seam. Raw body, not JSON: `/__api/file` reads through `chunk.toString()`,
        // lossy for a .glb, so binary gets its own endpoint. `name` may carry subdirs (a .gltf's sidecar
        // layout); writeAsset guards traversal, dedupes collisions, and suppresses the watcher bounce.
        if (url.pathname === "/__api/asset" && req.method === "POST") {
            const name = url.searchParams.get("name");
            if (!projectDir || !name) {
                error(res, 400, "Missing project dir or name");
                return;
            }
            const publicDir = join(resolve(projectDir), "public");
            readRawBody(req).then((bytes) => {
                try {
                    const result = writeAsset(publicDir, name, bytes);
                    if (result.error) {
                        error(res, result.status, result.error);
                        return;
                    }
                    json(res, { src: result.src });
                } catch (e) {
                    error(res, 500, String(e));
                }
            });
            return;
        }

        if (url.pathname === "/__api/state" && req.method === "GET") {
            request("state")
                .then((r) => json(res, r))
                .catch((e) => error(res, 503, e.message));
            return;
        }

        if (url.pathname === "/__api/command" && req.method === "POST") {
            readBody(req)
                .then((body) => request("command", JSON.parse(body)))
                .then((r) => json(res, r))
                .catch((e) => error(res, 400, e.message));
            return;
        }

        if (url.pathname === "/__api/entities" && req.method === "GET") {
            const component = url.searchParams.get("component");
            const eid = url.searchParams.get("eid");
            request("entities", { component, eid: eid ? Number(eid) : null })
                .then((r) => json(res, r))
                .catch((e) => error(res, 503, e.message));
            return;
        }

        if (publicDirs.length > 0 && req.url && !req.url.startsWith("/__api")) {
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

export function projectPlugin(projectDir?: string, opts?: { editor?: boolean }): Plugin {
    const virtualId = "virtual:project";
    const resolvedId = "\0" + virtualId;
    let viteServer: ViteDevServer | undefined;
    // the per-local HMR block drives the EDITOR's in-place swap (emitProjectReload → /src/project/reload), so
    // it's editor-only — a standalone project's dev server has no editor reload module and a build has no HMR.
    // A standalone's `virtual:project` is static imports + the project object; vite full-reloads it on a plugin
    // edit (no self-accept), which the page reload cleans up.
    const hot = opts?.editor ?? false;

    return {
        name: "shallot-project",
        async resolveId(id, importer) {
            if (id === virtualId) return resolvedId;
            // virtual:project is a virtual module with no location, so vite resolves its imports against
            // the host root (the editor dir). A relative local is already absolutized by the generator, but
            // a bare package subpath (a project's installed/workspace plugin, e.g. `orrstead/core/grid`)
            // would miss the editor's node_modules. Resolve those from the PROJECT dir, so a manifest can
            // reference an installed plugin by subpath. (Engine `@dylanebert/shallot` imports resolve here
            // too — to the project's copy, the same physical files the editor's deduplicate maps to.)
            if (importer === resolvedId && projectDir) {
                const r = await this.resolve(id, join(resolve(projectDir), "__project__.js"), {
                    skipSelf: true,
                });
                if (r) return r;
            }
        },
        // generate the `virtual:project` module from the project's `shallot.json` — static imports for each
        // enabled plugin (engine via the barrel, locals via their specifier) + the scene + manifest. In the
        // editor (`hot`), the per-local `import.meta.hot.accept` the generator emits is the hot-reload seam:
        // a project code edit re-imports that plugin and hands it to the editor's in-place swap
        // (`emitProjectReload`), so the live State and its runtime state survive a code edit without a reload.
        load(id) {
            if (id !== resolvedId) return;
            if (!projectDir) return generateModule({}, null, [], hot);
            const absDir = resolve(projectDir);
            return generateModule(readManifest(absDir), absDir, discoverScenes(absDir), hot);
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
                // set — both re-generate `virtual:project`, so invalidate + reload (unless it's the
                // editor's own write, deduped by `recentSaves`). Local plugin `.ts` edits ride HMR instead.
                const onProjectFile = (file: string) => {
                    // a model asset (.glb/.gltf) under a public dir → invalidate that src + rebuild (no
                    // virtual:project change — the model isn't part of the generated module). Checked first,
                    // before the absDir guard, so a shared parent public/ is covered. An `/__api/asset`
                    // upload is the editor's own write (it loads the asset live), so it's deduped like a
                    // scene save; only a genuinely external write signals.
                    const src = assetSrc(file, publicDirs);
                    if (src) {
                        if (!recentSaves.has(file)) signalAsset(server, hot, src);
                        return;
                    }
                    if (!file.startsWith(absDir)) return;
                    if (!file.endsWith(".scene") && file !== manifestPath(absDir)) return;
                    const mod = server.moduleGraph.getModuleById(resolvedId);
                    if (mod) server.moduleGraph.invalidateModule(mod);
                    if (recentSaves.has(file)) return;
                    signalChange(server, hot, absDir, file);
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
                if (recentSaves.has(file)) return [];
                signalChange(viteServer, hot, absDir, file);
                return [];
            }
            // a project src/*.ts edit falls through to vite's default HMR, which reaches the
            // `virtual:project` accept above (re-import the local → swap), so the live State is never torn down
        },
        // drop the assets vite's `new URL` scanner over-emitted (see orphanedAssets). Build-only (a
        // rollup output hook, never fires in dev), and homed here so every build path inherits it: the
        // editor-first synth build (bin/build.ts) and a standalone's own vite.config both run projectPlugin.
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
