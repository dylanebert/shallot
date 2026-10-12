import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import typegpu from "unplugin-typegpu/vite";
import { contentType, resolveAssetPath } from "./assets.js";


/**
 * Cross-origin isolation headers, applied by Vite's dev and preview servers. Physics multithreads
 * only when the page can hold a shared `WebAssembly.Memory`, which a browser grants only to a
 * cross-origin-isolated document. A static host that cannot set headers gets the documented
 * single-thread fallback.
 */
const CROSS_ORIGIN_ISOLATION = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
};

/** Installed dependencies that consume Shallot must share the project's engine instance.
 * @param {string} projectDir
 * @returns {string[]}
 */
function pluginPackages(projectDir) {
    const projectManifest = join(projectDir, "package.json");
    if (!existsSync(projectManifest)) return [];
    const project = JSON.parse(readFileSync(projectManifest, "utf8"));
    const require = createRequire(projectManifest);
    return Object.keys({ ...project.dependencies, ...project.devDependencies }).filter((name) => {
        const path = require
            .resolve
            .paths(name)
            ?.map((directory) => join(directory, name, "package.json"))
            .find((candidate) => existsSync(candidate));
        if (!path) return false;
        const pkg = JSON.parse(readFileSync(path, "utf8"));
        return Boolean(pkg.dependencies?.["@dylanebert/shallot"] || pkg.peerDependencies?.["@dylanebert/shallot"]);
    });
}

/** @param {string} projectDir @returns {string[]} */
function findPublicDirs(projectDir) {
    const own = join(projectDir, "public");
    return existsSync(own) ? [own] : [];
}

/** @param {import("vite").ViteDevServer} server @param {string} projectDir */
function configureServer(server, projectDir) {
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

/**
 * Returns emitted assets without a surviving bundle reference.
 * @param {Record<string, any>} bundle
 * @returns {string[]}
 */
function orphanedAssets(bundle) {
    const files = Object.values(bundle);
    /** @param {any} file */
    const text = (file) =>
        file.type === "chunk" ? file.code : typeof file.source === "string" ? file.source : "";
    // Keep chunks and HTML roots. Vite records chunk-to-CSS edges in metadata rather than chunk.code.
    const kept = new Set(files.filter((file) => file.type === "chunk" || file.fileName.endsWith(".html")));
    for (const file of files) {
        if (file.type !== "chunk") continue;
        const metadata = /** @type {any} */ (file);
        for (const css of metadata.viteMetadata?.importedCss ?? []) {
            const imported = bundle[css];
            if (imported?.type === "asset") kept.add(imported);
        }
    }
    const assets = files.filter(
        (file) => file.type === "asset" && !file.fileName.endsWith(".html"),
    );
    let grew = true;
    while (grew) {
        grew = false;
        for (const asset of assets) {
            if (kept.has(asset)) continue;
            const name = asset.fileName.slice(asset.fileName.lastIndexOf("/") + 1);
            if ([...kept].some((file) => text(file).includes(name))) {
                kept.add(asset);
                grew = true;
            }
        }
    }
    return assets.filter((asset) => !kept.has(asset)).map((asset) => asset.fileName);
}

/**
 * The Vite plugin set a Shallot project needs: `plugins: [shallot()]`. It carries TypeGPU's transform
 * alongside project support, and reaches engine source inside `node_modules`; a second pass corrupts
 * its metadata.
 * @param {string} [projectDir]
 * @returns {import("vite").Plugin[]}
 */
export function shallot(projectDir) {
    let absProjectDir = projectDir ? resolve(projectDir) : resolve(process.cwd());

    /** @type {import("vite").Plugin} */
    const projectPlugin = {
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
        generateBundle(_options, bundle) {
            const orphans = orphanedAssets(/** @type {Record<string, any>} */ (bundle));
            if (!orphans.length) return;
            let bytes = 0;
            for (const fileName of orphans) {
                const asset = bundle[fileName];
                if (asset?.type === "asset")
                    bytes += typeof asset.source === "string" ? asset.source.length : asset.source.byteLength;
                delete bundle[fileName];
            }
            this.info(`pruned ${orphans.length} orphaned asset(s), ${(bytes / 1024) | 0}KB`);
        },
    };
    return [/** @type {import("vite").Plugin} */ (typegpu()), projectPlugin];
}
