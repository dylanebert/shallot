import { existsSync, readFileSync } from "fs";
import { createRequire } from "node:module";
import { join, resolve } from "path";
import typegpu from "unplugin-typegpu/vite";
import type { Plugin, Rollup, ViteDevServer } from "vite";
import { contentType, resolveAssetPath } from "./assets";

/**
 * cross-origin isolation headers, applied by Vite's dev and preview servers. Physics multithreads only when the page can hold a
 * shared `WebAssembly.Memory`, which a browser grants only to a cross-origin-isolated document — so the
 * dev/preview server sends COOP/COEP to enable the multithreaded kernel. A static host that can't set
 * headers (GitHub Pages) gets the single-thread kernel and one log, a documented fallback. The cost of
 * `require-corp`: every cross-origin subresource the page loads must be CORS-approved (a cors-mode fetch
 * against a CORS-enabled host, like extras/text's default gstatic font) or carry CORP — a plain no-cors
 * cross-origin load (`<img src="https://…">` from a host without CORP) is blocked in the isolated
 * document. Consumer-facing note: AGENTS.md "Build, run, verify".
 */
const CROSS_ORIGIN_ISOLATION = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
};

/** Installed dependencies that consume Shallot must share the project's engine instance. */
function pluginPackages(projectDir: string): string[] {
    const projectManifest = join(projectDir, "package.json");
    if (!existsSync(projectManifest)) return [];
    const project = JSON.parse(readFileSync(projectManifest, "utf8"));
    const require = createRequire(projectManifest);
    return Object.keys({ ...project.dependencies, ...project.devDependencies }).filter((name) => {
        const path = require.resolve.paths(name)
            ?.map((directory) => join(directory, name, "package.json"))
            .find((path) => existsSync(path));
        if (!path) return false;
        const pkg = JSON.parse(readFileSync(path, "utf8"));
        return Boolean(pkg.dependencies?.["@dylanebert/shallot"] || pkg.peerDependencies?.["@dylanebert/shallot"]);
    });
}

function findPublicDirs(projectDir: string): string[] {
    const own = join(projectDir, "public");
    return existsSync(own) ? [own] : [];
}

// serve the project's public/ assets with correct MIME
function configureServer(server: ViteDevServer, projectDir: string) {
    const publicDirs = findPublicDirs(resolve(projectDir));
    if (publicDirs.length === 0) return;

    server.middlewares.use((req, res, next) => {
        if (req.url) {
            const pathname = new URL(req.url, "http://localhost").pathname;
            for (const dir of publicDirs) {
                const filePath = resolveAssetPath(dir, pathname);
                if (!filePath) continue;
                const data = readFileSync(filePath);
                const mime = contentType(filePath);
                if (mime) res.setHeader("Content-Type", mime);
                // Reloads must read changed asset bytes rather than a cached response.
                res.setHeader("Cache-Control", "no-store");
                res.end(data);
                return;
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
function orphanedAssets(bundle: Rollup.OutputBundle): string[] {
    const files = Object.values(bundle);
    const text = (f: Rollup.OutputAsset | Rollup.OutputChunk) =>
        f.type === "chunk" ? f.code : typeof f.source === "string" ? f.source : "";
    // never prune a chunk (tree-shaking already pruned JS) or an html entry — seed them as kept roots.
    // Vite records chunk→CSS edges in metadata because the CSS import is not part of chunk.code; read
    // that edge before the textual asset fixpoint so a manifest build can retain CSS before HTML exists.
    const kept = new Set(files.filter((f) => f.type === "chunk" || f.fileName.endsWith(".html")));
    for (const file of files) {
        if (file.type !== "chunk") continue;
        const metadata = file as Rollup.OutputChunk & {
            viteMetadata?: { importedCss?: ReadonlySet<string> };
        };
        for (const css of metadata.viteMetadata?.importedCss ?? []) {
            const imported = bundle[css];
            if (imported?.type === "asset") kept.add(imported);
        }
    }
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

/**
 * The Vite plugin set a Shallot project needs: `plugins: [shallot()]`. It carries TypeGPU's transform
 * alongside project support, and reaches engine source inside `node_modules`; a second pass corrupts its
 * metadata.
 */
export function shallot(projectDir?: string): Plugin[] {
    let absProjectDir = projectDir ? resolve(projectDir) : resolve(process.cwd());

    const projectPlugin: Plugin = {
        name: "shallot",
        config(config) {
            if (!projectDir) absProjectDir = resolve(config.root ?? process.cwd());
            const sharedDependencies = [
                "@dylanebert/shallot",
                "typegpu",
                ...pluginPackages(absProjectDir),
            ];
            return {
                resolve: { dedupe: sharedDependencies },
                optimizeDeps: { exclude: sharedDependencies },
                server: { headers: CROSS_ORIGIN_ISOLATION },
            };
        },
        configResolved(config) {
            if (config.plugins.filter((plugin) => plugin.name === "unplugin-typegpu").length > 1) {
                throw new Error(
                    "shallot() includes the TypeGPU transform; remove the separate typegpu() plugin",
                );
            }
        },
        configureServer(server) {
            configureServer(server, absProjectDir);
        },
        // drop the assets vite's `new URL` scanner over-emitted (see orphanedAssets). Build-only (a
        // rollup output hook, never fires in dev), and homed here so every project's Vite build inherits it.
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
    return [typegpu() as unknown as Plugin, projectPlugin];
}
