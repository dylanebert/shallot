import { existsSync } from "node:fs";
import { resolve } from "node:path";
import {
    loadConfigFromFile,
    mergeConfig,
    type PluginOption,
    type Plugin as VitePlugin,
} from "vite";
import { discoverScenes, manifestPath } from "../src/project/vite";

// One toolchain merge shared by `shallot dev` and `shallot build`. A manifest project is pure data, but a
// project that needs a framework (Svelte, React) declares it the standard vite way — its own
// `vite.config.ts` with `@sveltejs/vite-plugin-svelte` etc. Both commands load + merge that config
// identically here, so a framework project (orrstead) runs the same in dev and a build. No `vite.config` →
// the synthesized zero-config path (a manifest recipe is unaffected).

/** dir holds a shallot project — a shallot.json manifest or a .scene file. */
export function isProject(projectDir: string): boolean {
    const abs = resolve(projectDir);
    return existsSync(manifestPath(abs)) || discoverScenes(abs).length > 0;
}

/** exit with the scaffold hint when dir holds neither a shallot.json manifest nor a .scene file. */
export function requireProject(projectDir: string): void {
    if (isProject(projectDir)) return;
    console.error(`\n  ✗ No shallot project found at ${projectDir}`);
    console.error("    Expected a shallot.json manifest or a .scene file\n");
    console.error("    To create a project:");
    console.error("      bun create shallot my-game");
    console.error("      cd my-game && bun install");
    console.error("      bunx shallot dev\n");
    process.exit(1);
}

/**
 * flatten vite's nested + async plugin option arrays to named plugins. Plugin factories return arrays
 * (`@sveltejs/vite-plugin-svelte`, `@vitejs/plugin-react`) and a config entry may be a promise; vite
 * resolves both at config time, so we resolve them here to read each `.name` (e.g. for name-based dedup of
 * a project's plugins against the host's). Falsy entries (a conditional `cond && plugin()`) drop out.
 */
export async function flattenPlugins(
    plugins: PluginOption | PluginOption[] | undefined,
): Promise<VitePlugin[]> {
    const out: VitePlugin[] = [];
    const walk = async (p: PluginOption | Promise<PluginOption>) => {
        const v = await p;
        if (!v) return;
        if (Array.isArray(v)) {
            for (const x of v) await walk(x);
            return;
        }
        out.push(v);
    };
    await walk(plugins as PluginOption);
    return out;
}

/** a project's own resolved vite config, reduced to what a host may safely adopt. */
export interface ProjectConfig {
    /** the project's flattened plugins (its framework set: svelte/react/aliases) */
    plugins: VitePlugin[];
    /** resolve/define/optimizeDeps the host overlays onto its base (never root/server/build — the host owns those) */
    overlay: Record<string, unknown>;
    /** the resolved config file path, for logging */
    path: string;
}

/**
 * load a project's own `vite.config.{ts,js}` if it has one. Returns null for a zero-config manifest project
 * (the common case) — the caller then uses its synthesized base unchanged. `command` is `"serve"` for the
 * dev server, `"build"` for a build.
 */
export async function loadProjectConfig(
    absProjectDir: string,
    command: "serve" | "build",
    mode: string,
): Promise<ProjectConfig | null> {
    const loaded = await loadConfigFromFile({ command, mode }, undefined, absProjectDir);
    if (!loaded) return null;
    return {
        plugins: await flattenPlugins(loaded.config.plugins),
        overlay: {
            resolve: loaded.config.resolve,
            define: loaded.config.define,
            optimizeDeps: loaded.config.optimizeDeps,
        },
        path: loaded.path,
    };
}

/**
 * compose a host's base config (carrying its own `plugins`) with a project's loaded config. The project's
 * framework plugins go first (each claims only its own file types; vite re-sorts by `enforce`), then the
 * host's; the project's resolve/define/optimizeDeps overlay the base. `drop` removes project plugins that
 * collide by name with a host plugin (the host wins, so a project never doubles a host-provided
 * plugin). `project` null → the base unchanged.
 */
export function composeViteConfig(
    base: Record<string, unknown>,
    project: ProjectConfig | null,
    drop?: Set<string>,
): Record<string, unknown> {
    if (!project) return base;
    const hostPlugins = (base.plugins as PluginOption[] | undefined) ?? [];
    const projectPlugins = drop
        ? project.plugins.filter((p) => !drop.has(p.name))
        : project.plugins;
    return {
        ...mergeConfig(project.overlay, base),
        plugins: [...projectPlugins, ...hostPlugins],
    };
}
