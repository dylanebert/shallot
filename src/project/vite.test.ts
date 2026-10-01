import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { shallot } from "./vite";

test("the Vite plugin only serves a project's own public assets", () => {
    const root = mkdtempSync(join(tmpdir(), "shallot-public-scope-"));
    const project = join(root, "example");
    mkdirSync(join(root, "public"), { recursive: true });
    mkdirSync(project);
    let middlewareCount = 0;
    const server = {
        middlewares: { use() { middlewareCount++; } },
        watcher: { add() {}, on() {} },
        ws: { send() {} },
        moduleGraph: { getModuleById() { return null; }, invalidateModule() {} },
    };
    try {
        const plugin = shallotProject(project);
        const configure = plugin.configureServer as unknown as (server: unknown) => void;
        configure(server);
        expect(middlewareCount).toBe(0);

        mkdirSync(join(project, "public"));
        configure(server);
        expect(middlewareCount).toBe(1);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});

function shallotProject(projectDir?: string) {
    const plugin = shallot(projectDir).find(({ name }) => name === "shallot");
    if (!plugin) throw new Error("shallot() did not return its project plugin");
    return plugin;
}

test("dependencies and devDependencies consuming Shallot are deduped and excluded, unlike unrelated packages", () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-project-plugin-"));
        try {
            const dependencies = { "peer-plugin": "1", "direct-plugin": "1", unrelated: "1" };
            const devDependencies = { "dev-plugin": "1" };
            writeFileSync(join(root, "package.json"), JSON.stringify({ dependencies, devDependencies }));
            for (const [name, engineSection] of [["peer-plugin", "peerDependencies"], ["direct-plugin", "dependencies"], ["dev-plugin", "peerDependencies"], ["unrelated", "devDependencies"]]) {
                const directory = join(root, "node_modules", name);
                mkdirSync(directory, { recursive: true });
                writeFileSync(join(directory, "package.json"), JSON.stringify({ name, main: "index.js", [engineSection]: { "@dylanebert/shallot": "*" } }));
                writeFileSync(join(directory, "index.js"), "export {};");
            }
            const plugins = shallot(root);
            expect(plugins.map(({ name }) => name)).toEqual(["unplugin-typegpu", "shallot"]);
            const plugin = plugins.find(({ name }) => name === "shallot");
            if (!plugin) throw new Error("shallot() did not return its project plugin");
            const configHook = plugin.config as unknown as (
                config: { root: string },
                env: unknown,
            ) => unknown;
            const config = configHook({ root }, {}) as {
                resolve?: { dedupe?: string[] };
                optimizeDeps?: { exclude?: string[] };
                server?: { headers?: Record<string, string> };
            };
            const expected = ["@dylanebert/shallot", "typegpu", "peer-plugin", "direct-plugin", "dev-plugin"];
            expect(config.resolve?.dedupe).toEqual(expected);
            expect(config.optimizeDeps?.exclude).toEqual(expected);
            expect(config.server?.headers).toEqual({
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Embedder-Policy": "require-corp",
            });
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

test("a project config cannot apply TypeGPU's metadata transform twice", () => {
        const plugins = shallot();
        const project = plugins.find(({ name }) => name === "shallot");
        if (!project) throw new Error("shallot() did not return its project plugin");
        const resolved = project.configResolved as unknown as (config: {
            plugins: { name: string }[];
        }) => void;
        const names = plugins.map(({ name }) => ({ name }));
        expect(() => resolved({ plugins: names })).not.toThrow();
        expect(() => resolved({ plugins: [...names, { name: "unplugin-typegpu" }] })).toThrow(
            "shallot() includes the TypeGPU transform",
        );
    });

test("a Vite build does not ship scanner-emitted assets without a surviving bundle reference", () => {
        const bundle = {
            "entry.js": { type: "chunk", fileName: "entry.js", code: 'load("used.png")' },
            "used.png": { type: "asset", fileName: "used.png", source: "used" },
            "orphan.png": { type: "asset", fileName: "orphan.png", source: "orphan" },
        };
        const messages: string[] = [];
        const generate = shallotProject().generateBundle as unknown as (
            this: { info(message: string): void },
            options: unknown,
            bundle: Record<string, unknown>,
        ) => void;
        generate.call({ info: (message) => messages.push(message) }, {}, bundle);
        expect(Object.hasOwn(bundle, "orphan.png")).toBe(false);
        expect(Object.hasOwn(bundle, "used.png")).toBe(true);
        expect(messages[0]).toContain("pruned 1 orphaned asset(s)");
    });
