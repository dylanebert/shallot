import { basename, relative, resolve } from "node:path";
import { createServer, searchForWorkspaceRoot, type Plugin as VitePlugin } from "vite";
import {
    CROSS_ORIGIN_ISOLATION,
    findPublicDirs,
    projectPlugin,
    typegpuPlugin,
} from "../src/project/vite";
import { synthIndex } from "./build";
import { composeViteConfig, loadProjectConfig, requireProject } from "./toolchain";

// serve the synthesized entry at `/` — a manifest project owns no index.html, so the CLI provides one.
// `transformIndexHtml` runs the page through vite's HTML pipeline (HMR client, inline-module extraction),
// so the inline `run(virtual:project)` script resolves the same manifest a build does.
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

/** the vite dev config for a manifest project. `open` defaults true (the CLI). */
export function devConfig(
    absProjectDir: string,
    name: string,
    opts: { port?: number; strictPort?: boolean; open?: boolean },
) {
    return {
        root: absProjectDir,
        configFile: false as const,
        // typegpu transpiles TGSL function bodies at build time — there is no runtime fallback, and
        // the engine's own kernels live in node_modules, so the transform must reach there too
        plugins: [typegpuPlugin(), projectPlugin(absProjectDir), synthIndexPlugin(name)],
        // a registry install resolves the engine inside node_modules (never a symlink, unlike a local
        // dev checkout), so Vite's dep scanner esbuild-prebundles it ahead of the typegpu transform on
        // first real page load — the bundled TGSL carries no build metadata, and TypeGPU's metadata-
        // free resolution fallback derives wrong WGSL for anything past a trivial body (a struct-output
        // cast is where it broke: `Cannot resolve struct cast from 'vertexVsOut' to 'vertexVs_Output'`
        // at pipeline warm; 5b-2f-5 reproduced this against a genuine `@dylanebert/shallot@0.9.0`
        // registry install). The bare specifier is enough for the engine's own subpaths: Vite's scanner
        // never descends into an excluded package to discover its own subpath imports, so every
        // `@dylanebert/shallot/*` subpath rides this one entry (verified empirically against the
        // registry-installed package; toolchain.test.ts pins that an ejected project's own
        // `optimizeDeps` still carries this exclusion through the merge). `typegpu` needs its own entry
        // though, not coverage-by-association: a consumer's own source commonly imports `typegpu/data`
        // directly (a consumer project importing a typegpu subpath on its own — install-test.ts's
        // fixture now does too), a second,
        // independent entry into the scanner distinct from the engine's. Verified empirically: without
        // this line, that shape produces a genuine same-version duplicate typegpu module (typegpu's own
        // "Found duplicate TypeGPU version. First was 0.11.9, this one is 0.11.9" warning fires) and
        // dies at pipeline warm with `Invalid property key 'type': Identifiers cannot start with
        // reserved keywords` inside typegpu's own `struct.js` — a different symptom than the missing-
        // metadata case above, same root cause.
        optimizeDeps: { exclude: ["@dylanebert/shallot", "typegpu"] },
        server: {
            port: opts.port,
            strictPort: opts.strictPort,
            open: opts.open ?? true,
            // cross-origin isolation so tumble physics multithreads (COOP/COEP → shared WebAssembly.Memory)
            headers: CROSS_ORIGIN_ISOLATION,
            // searchForWorkspaceRoot restores vite's default fs.allow root (which an explicit `allow`
            // overrides). The engine package (`@dylanebert/shallot`, with its `rust/audio/pkg/*.wasm`
            // fetched over /@fs/) is covered by it when in-workspace; a cross-repo symlink (a project
            // symlinked in from outside the workspace) lands outside, so the CLI's own engine dir is
            // allowed explicitly too.
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
 * `shallot dev` — run a project standalone: a vite HMR server over its `shallot.json`. The
 * project is pure data; the CLI supplies the entry + harness. `projectPlugin` resolves
 * `virtual:project` and full-reloads on a manifest / scene / plugin edit — the same resolver `shallot build`
 * uses, so dev and ship agree on the loaded plugins + scene + capacity.
 */
export async function startDev(projectDir: string, opts: { port?: number; strictPort?: boolean }) {
    const absProjectDir = resolve(projectDir);
    const name = basename(absProjectDir);

    requireProject(projectDir);

    console.log(`\n  🧅 shallot · ${name}\n`);

    // merge the project's own vite.config (svelte/react/aliases) the same way `shallot build` does, so a
    // framework project runs identically in both. No vite.config → the synthesized base unchanged.
    const project = await loadProjectConfig(absProjectDir, "serve", "development");
    if (project) console.log(`  · merged ${relative(absProjectDir, project.path)}\n`);

    // drop a project's own copy of the host plugins (a project may declare `projectPlugin` in its
    // vite.config for an ejected harness like a bench — the CLI host provides it here).
    const server = await createServer(
        composeViteConfig(
            devConfig(absProjectDir, name, opts),
            project,
            new Set(["shallot-project", "shallot-synth-index", "unplugin-typegpu"]),
        ),
    );
    await server.listen();
    server.printUrls();
    console.log();
}
