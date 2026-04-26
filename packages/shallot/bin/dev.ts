import { resolve, dirname, basename, join } from "node:path";
import { existsSync, readFileSync } from "node:fs";
import { svelte, vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import { createServer, createLogger, type Plugin as VitePlugin } from "vite";
import { projectPlugin } from "../editor/src/project/vite";

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

export async function startDev(projectDir: string, opts: { port?: number }) {
    const projectName = basename(projectDir);
    const packageDir = resolve(dirname(import.meta.dir));

    if (
        !existsSync(resolve(projectDir, "src/lib.ts")) &&
        !existsSync(resolve(projectDir, "src/lib/index.ts"))
    ) {
        console.error(`\n  ✗ No shallot project found at ${projectDir}`);
        console.error("    Expected src/lib.ts or src/lib/index.ts\n");
        process.exit(1);
    }

    const editorDir = resolve(packageDir, "editor");
    const inNodeModules = editorDir.split("/").includes("node_modules");

    console.log(`\n  🧅 shallot · ${projectName}\n`);

    const server = await createServer({
        root: editorDir,
        plugins: [
            svelte({
                configFile: false,
                preprocess: vitePreprocess(),
                compilerOptions: {
                    runes: true,
                    css: inNodeModules ? "injected" : "external",
                },
            }),
            projectPlugin(projectDir),
            deduplicateShallot(packageDir, projectDir),
        ],
        server: {
            port: opts.port,
            open: true,
            fs: { allow: [editorDir, packageDir, projectDir] },
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
        optimizeDeps: {
            noDiscovery: true,
            include: ["svelte", "paneforge", "lucide-static", "mediabunny"],
        },
        build: { target: "esnext" },
    });
    await server.listen();
    server.printUrls();
    console.log();
}
