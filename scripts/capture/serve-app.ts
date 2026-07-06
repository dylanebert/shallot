import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { createServer } from "vite";

// Serves a standalone run() fixture — a plain vite dev server, no editor, no svelte. The fixture dir is
// argv[3] (default "app", the survive-reload fixture; "ui" is the ui-containment fixture). capture.ts boots
// these alongside the editor server, each on its own port. The `deduplicate` plugin below resolves
// `@dylanebert/shallot` from the
// fixture straight to engine source (the trick `shallot dev` uses) so the fixture needs no install. It
// duplicates `bin/edit.ts`'s deduplicateShallot on purpose: this script resolves `vite` from the repo
// root while `bin/` resolves it from `packages/shallot` (two installs), so a shared plugin object trips
// vite's type identity. The repo tooling keeps its own copy rather than pulling a test seam into the
// shipped CLI.
function deduplicate(packageDir: string, projectDir: string) {
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
        enforce: "pre" as const,
        resolveId(source: string, importer: string | undefined) {
            if (!importer || !mapping.has(source)) return;
            if (!importer.startsWith(absProjectDir)) return;
            return mapping.get(source);
        },
    };
}

const appDir = resolve(import.meta.dir, process.argv[3] ?? "app");
const packageDir = resolve(import.meta.dir, "..", "..", "packages", "shallot");
const port = Number(process.argv[2] ?? 3005);

const server = await createServer({
    root: appDir,
    plugins: [deduplicate(packageDir, appDir)],
    server: {
        port,
        strictPort: true,
        open: false,
        fs: { allow: [appDir, packageDir] },
    },
    optimizeDeps: { noDiscovery: true },
});

await server.listen();
server.printUrls();
