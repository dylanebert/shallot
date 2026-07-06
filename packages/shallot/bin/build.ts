import { execSync } from "node:child_process";
import { cpSync, existsSync, rmSync, writeFileSync } from "node:fs";
import { basename, relative, resolve } from "node:path";
import { build as viteBuild } from "vite";
import {
    discoverScenes,
    findPublicDirs,
    manifestPath,
    projectPlugin,
} from "../editor/src/project/vite";
import { requiredFeatures, verdict } from "./features";
import { bundleNativeLinux, bundleNativeMac, bundleNativeWindows, nativeOutDir } from "./native";
import { composeViteConfig, loadProjectConfig } from "./toolchain";

// the entry a manifest project lacks: a page that runs the project's manifest. resolves the same
// `virtual:project` the editor reads (one resolver, no second manifest reader) — its `plugins` are the
// enabled set (engine via the barrel, tree-shaken; locals by specifier), `scene` the default scene.
// shared by the web build here and the standalone `shallot dev` server (bin/dev.ts).
export const synthIndex = (name: string) => `<!doctype html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" type="image/svg+xml" href="./icon.svg" />
        <title>${name}</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: #0c0a09; overflow: hidden; }
            canvas { display: block; width: 100vw; height: 100vh; }
        </style>
    </head>
    <body>
        <canvas id="canvas"></canvas>
        <script type="module">
            import { run } from "@dylanebert/shallot";
            import project from "virtual:project";
            // project.plugins is the manifest's complete resolved set (enabled defaults + extras + locals),
            // so defaults: false — re-adding DEFAULT_PLUGINS would resurrect a manifest-disabled default.
            // capacity is the manifest's (null → the engine default), so a build honors the same fixed
            // capacity the editor does — one source of truth across editor + shipped build.
            await run({ plugins: project.plugins, scene: project.scene ?? undefined, defaults: false, capacity: project.capacity ?? undefined });
        </script>
    </body>
</html>
`;

export async function buildWeb(projectDir: string): Promise<void> {
    // ejected shape: the project owns its index.html + vite.config (gym, showcase/visualization), so
    // build with its own vite.
    if (existsSync(resolve(projectDir, "index.html"))) {
        console.log(`\n  building ${basename(projectDir)} → dist/\n`);
        execSync("bunx vite build", { cwd: projectDir, stdio: "inherit" });
        console.log(`\n  done.\n`);
        return;
    }

    // manifest shape (the default): no vite boilerplate. ship from the project's manifest by synthesizing
    // a temp entry that resolves `virtual:project` (via the project plugin), building it with the engine's
    // own vite (the project needs no vite dep), then removing the entry. the manifest + scenes are the project.
    if (!existsSync(manifestPath(projectDir)) && discoverScenes(projectDir).length === 0) {
        console.error(`\n  No index.html, shallot.json, or scene found at ${projectDir}`);
        console.error("    A manifest project ships from its shallot.json; add one,");
        console.error("    or scaffold a project with `bun create shallot <name>`.\n");
        process.exit(1);
    }

    console.log(`\n  building ${basename(projectDir)} → dist/\n`);
    // merge the project's own vite.config (svelte/react/aliases) the same way dev + the editor do, so a
    // framework manifest project (e.g. a Svelte HUD) ships from `shallot build`. None → the base unchanged.
    const project = await loadProjectConfig(resolve(projectDir), "build", "production");
    if (project) console.log(`  · merged ${relative(projectDir, project.path)}\n`);
    const entry = resolve(projectDir, "index.html");
    writeFileSync(entry, synthIndex(basename(projectDir)));
    try {
        const buildConfig = {
            root: projectDir,
            base: "./",
            configFile: false as const,
            logLevel: "warn" as const,
            plugins: [projectPlugin(resolve(projectDir))],
            build: { target: "esnext", outDir: "dist", emptyOutDir: true },
        };
        // drop a project's own copy of the host plugin (a project may declare `projectPlugin` for an
        // ejected harness; the build host provides it).
        await viteBuild(composeViteConfig(buildConfig, project, new Set(["shallot-project"])));
    } finally {
        rmSync(entry, { force: true });
    }

    // vite copied the project's own public/; mirror the dev server (findPublicDirs) by also pulling a
    // shared parent public/ into the bundle, so the synthesized index's ./icon.svg (and any shared asset)
    // resolves the same as `shallot dev`. force: false keeps the project's own files + bundle output winning.
    const dist = resolve(projectDir, "dist");
    for (const dir of findPublicDirs(resolve(projectDir))) {
        cpSync(dir, dist, { recursive: true, force: false, errorOnExist: false });
    }

    console.log(`\n  done.\n`);
}

export async function buildProject(
    projectDir: string,
    opts: { target?: string; release?: boolean; portable?: boolean },
) {
    const target = opts.target;
    if (target === "windows" || target === "mac" || target === "linux") {
        const release = opts.release ?? false;
        const portable = opts.portable ?? false;

        // warn (don't block) when the chosen backend can't satisfy the app's required features — the
        // build still produces an artifact that reaches the engine's diagnostic tier at launch.
        for (const line of verdict(target, portable, await requiredFeatures(projectDir))) {
            console.warn(`  ! ${line}`);
        }

        const outputDir = nativeOutDir(projectDir, target, release, portable);
        const label = relative(projectDir, outputDir);
        console.log(`\n  building ${basename(projectDir)} → ${label}/\n`);
        const bundleOpts = { release, portable };
        if (target === "windows") await bundleNativeWindows(projectDir, outputDir, bundleOpts);
        else if (target === "mac") await bundleNativeMac(projectDir, outputDir, bundleOpts);
        else await bundleNativeLinux(projectDir, outputDir, bundleOpts);
        console.log(`\n  done. ${label}/\n`);
        return;
    }

    if (target && target !== "web") {
        console.error(`unknown target: ${target}`);
        process.exit(1);
    }

    await buildWeb(projectDir);
}
