import { readFileSync } from "node:fs";
import { basename, dirname, join, relative, resolve } from "node:path";
import { svelte, vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import {
    createLogger,
    createServer,
    type PluginOption,
    searchForWorkspaceRoot,
    type Plugin as VitePlugin,
} from "vite";
import { projectPlugin } from "../editor/src/project/vite";
import { composeViteConfig, flattenPlugins, loadProjectConfig, requireProject } from "./toolchain";

function deduplicateShallot(packageDir: string, projectDir: string): VitePlugin {
    const pkg = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf-8"));
    const mapping = new Map<string, string>();
    for (const [key, value] of Object.entries(pkg.exports as Record<string, string>)) {
        if (key === "./src/*") continue;
        const specifier = key === "." ? pkg.name : `${pkg.name}/${key.slice(2)}`;
        mapping.set(specifier, resolve(packageDir, value));
    }

    const absProjectDir = resolve(projectDir);

    return {
        name: "shallot-deduplicate",
        enforce: "pre",
        resolveId(source, importer) {
            if (!importer || !mapping.has(source)) return;
            if (!importer.startsWith(absProjectDir)) return;
            return mapping.get(source);
        },
    };
}

export async function startEdit(projectDir: string, opts: { port?: number; strictPort?: boolean }) {
    const projectName = basename(projectDir);
    const packageDir = resolve(dirname(import.meta.dir));

    const absProjectDir = resolve(projectDir);
    requireProject(projectDir);

    const editorDir = resolve(packageDir, "editor");
    const inNodeModules = editorDir.split("/").includes("node_modules");

    console.log(`\n  🧅 shallot · ${projectName}\n`);

    const editorPlugins: PluginOption[] = [
        svelte({
            configFile: false,
            preprocess: vitePreprocess(),
            compilerOptions: {
                runes: true,
                css: inNodeModules ? "injected" : "external",
            },
        }),
        projectPlugin(projectDir, { editor: true }),
        deduplicateShallot(packageDir, projectDir),
    ];

    // The Storybook move: merge the project's OWN vite config so a standard vite+svelte/react app — its
    // deps, aliases, framework plugins — resolves through the editor host. `composeViteConfig` (shared with
    // `shallot dev` + build) overlays only resolve/define/optimizeDeps + the project's plugins; the editor
    // keeps its own root/entry/server/svelte (the project can't override the host), and drops a project
    // plugin colliding by name with an editor one (the editor wins → one runes-configured svelte instance).
    // `null` when the project has no vite.config (the editor-first layout). The editor core never learns
    // about React or a vite config — it all lives behind the virtual:project contract.
    const project = await loadProjectConfig(absProjectDir, "serve", "development");
    const editorNames = new Set((await flattenPlugins(editorPlugins)).map((p) => p.name));
    if (project) console.log(`  · merged ${relative(absProjectDir, project.path)}\n`);

    const base = {
        root: editorDir,
        plugins: editorPlugins,
        server: {
            port: opts.port,
            strictPort: opts.strictPort,
            open: true,
            // searchForWorkspaceRoot lets the editor read a project's workspace-sibling deps (a plugin
            // library like orrstead's `package/`, resolved from virtual:project) — mirrors `shallot dev`.
            fs: {
                allow: [editorDir, packageDir, projectDir, searchForWorkspaceRoot(absProjectDir)],
            },
        },
        customLogger: (() => {
            const l = createLogger();
            const w = l.warn.bind(l);
            const e = l.error.bind(l);
            l.warn = (msg, o) => {
                if (!msg.includes("[console.")) w(msg, o);
            };
            l.error = (msg, o) => {
                if (!msg.includes("[console.")) e(msg, o);
            };
            return l;
        })(),
        // editor deps stay explicitly included; dep discovery (scanning the project's own deps so they
        // pre-bundle) runs only for a real project — the editor-first layout has none, so noDiscovery
        // skips a needless scan there. A project's own optimizeDeps merges over this (include concatenated).
        optimizeDeps: {
            ...(project ? {} : { noDiscovery: true }),
            include: ["svelte", "lucide-static", "mediabunny"],
        },
        build: { target: "esnext" as const },
    };

    const server = await createServer(composeViteConfig(base, project, editorNames));
    await server.listen();
    server.printUrls();
    console.log();
}
