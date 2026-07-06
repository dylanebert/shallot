import { basename, relative, resolve } from "node:path";
import { createServer, searchForWorkspaceRoot, type Plugin as VitePlugin } from "vite";
import { findPublicDirs, projectPlugin } from "../editor/src/project/vite";
import { synthIndex } from "./build";
import { composeViteConfig, loadProjectConfig, requireProject } from "./toolchain";

// serve the synthesized entry at `/` — a manifest project owns no index.html, so the CLI provides one.
// `transformIndexHtml` runs the page through vite's HTML pipeline (HMR client, inline-module extraction),
// so the inline `run(virtual:project)` script hot-reloads the same manifest the editor and a build resolve.
function synthIndexPlugin(name: string): VitePlugin {
    return {
        name: "shallot-synth-index",
        configureServer(server) {
            server.middlewares.use(async (req, res, next) => {
                const path = req.url?.split("?")[0];
                if (path !== "/" && path !== "/index.html") return next();
                const html = await server.transformIndexHtml(req.url ?? "/", synthIndex(name));
                res.statusCode = 200;
                res.setHeader("Content-Type", "text/html");
                res.end(html);
            });
        },
    };
}

/** the standalone (non-editor) vite dev config for a manifest project. `open` defaults true (the CLI). */
export function devConfig(
    absProjectDir: string,
    name: string,
    opts: { port?: number; strictPort?: boolean; open?: boolean },
) {
    return {
        root: absProjectDir,
        configFile: false as const,
        plugins: [projectPlugin(absProjectDir), synthIndexPlugin(name)],
        server: {
            port: opts.port,
            strictPort: opts.strictPort,
            open: opts.open ?? true,
            // searchForWorkspaceRoot restores vite's default fs.allow root (which an explicit `allow`
            // overrides). The engine package (`@dylanebert/shallot`, with its `rust/audio/pkg/*.wasm`
            // fetched over /@fs/) is covered by it when in-workspace; a cross-repo symlink (the orrstead
            // dev setup) lands outside, so the CLI's own engine dir is allowed explicitly too — same as the
            // editor's `packageDir`.
            fs: {
                allow: [
                    searchForWorkspaceRoot(absProjectDir),
                    absProjectDir,
                    resolve(import.meta.dir, ".."),
                    ...findPublicDirs(absProjectDir),
                ],
            },
        },
        build: { target: "esnext" as const },
    };
}

/**
 * `shallot dev` — run a project standalone (no editor): a vite HMR server over its `shallot.json`. The
 * project is pure data; the CLI supplies the entry + harness. `projectPlugin` (non-editor) resolves
 * `virtual:project` and full-reloads on a manifest / scene / plugin edit — the same resolver `shallot build`
 * uses, so dev and ship agree on the loaded plugins + scene + capacity.
 */
export async function startDev(projectDir: string, opts: { port?: number; strictPort?: boolean }) {
    const absProjectDir = resolve(projectDir);
    const name = basename(absProjectDir);

    requireProject(projectDir);

    console.log(`\n  🧅 shallot · ${name}\n`);

    // merge the project's own vite.config (svelte/react/aliases) the same way the editor + build do, so a
    // framework project runs identically across all three. No vite.config → the synthesized base unchanged.
    const project = await loadProjectConfig(absProjectDir, "serve", "development");
    if (project) console.log(`  · merged ${relative(absProjectDir, project.path)}\n`);

    // drop a project's own copy of the host plugins (a project may declare `projectPlugin` in its
    // vite.config for an ejected harness like a bench — the CLI host provides it here).
    const server = await createServer(
        composeViteConfig(
            devConfig(absProjectDir, name, opts),
            project,
            new Set(["shallot-project", "shallot-synth-index"]),
        ),
    );
    await server.listen();
    server.printUrls();
    console.log();
}
