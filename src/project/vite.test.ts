import { expect } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { shallot } from "./vite";

check(
    "shallot plugin: shares dependencies and configures both servers",
    {
        claim: "a project can load duplicate engine, typegpu or manifest plugin instances, and its dev or preview server lacks isolation headers",
    },
    () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-project-plugin-"));
        try {
            writeFileSync(
                join(root, "shallot.json"),
                JSON.stringify({
                    plugins: {
                        Grid: "@dylanebert/shallot-grid/core",
                        Local: "./src/local.ts",
                    },
                }),
            );
            const plugin = shallot(root);
            const configHook = plugin.config as unknown as (
                config: { root: string },
                env: unknown,
            ) => unknown;
            const config = configHook({ root }, {}) as {
                resolve?: { dedupe?: string[] };
                optimizeDeps?: { exclude?: string[] };
                server?: { headers?: Record<string, string> };
                preview?: { headers?: Record<string, string> };
            };
            const expected = ["@dylanebert/shallot", "typegpu", "@dylanebert/shallot-grid"];
            expect(config.resolve?.dedupe).toEqual(expected);
            expect(config.optimizeDeps?.exclude).toEqual(expected);
            expect(config.server?.headers).toEqual({
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Embedder-Policy": "require-corp",
            });
            expect(config.preview?.headers).toEqual(config.server?.headers);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);

check(
    "shallot plugin: provides virtual project data from shallot.json",
    {
        claim: "a project's virtual:project module reflects its shallot.json manifest and scene",
    },
    () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-virtual-project-"));
        try {
            writeFileSync(
                join(root, "shallot.json"),
                JSON.stringify({ scene: "scenes/arena.scene", plugins: { Physics: true } }),
            );
            const plugin = shallot(root);
            const load = plugin.load as unknown as (id: string) => string | undefined;
            const source = load("\0virtual:project");
            expect(source).toContain("PhysicsPlugin");
            expect(source).toContain('const scene = "scenes/arena.scene";');
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);

check(
    "shallot plugin: rejects a second TypeGPU transform",
    {
        claim: "a project config cannot apply TypeGPU's metadata transform twice",
    },
    () => {
        const plugin = shallot();
        const resolved = plugin.configResolved as unknown as (config: {
            plugins: { name: string }[];
        }) => void;
        expect(() => resolved({ plugins: [{ name: "shallot" }, { name: "unplugin-typegpu" }] })).toThrow(
            "shallot() includes the TypeGPU transform",
        );
    },
);

check(
    "shallot plugin: prunes only unreferenced build assets",
    {
        claim: "a Vite build does not ship scanner-emitted assets without a surviving bundle reference",
    },
    () => {
        const bundle = {
            "entry.js": { type: "chunk", fileName: "entry.js", code: 'load("used.png")' },
            "used.png": { type: "asset", fileName: "used.png", source: "used" },
            "orphan.png": { type: "asset", fileName: "orphan.png", source: "orphan" },
        };
        const messages: string[] = [];
        const generate = shallot().generateBundle as unknown as (
            this: { info(message: string): void },
            options: unknown,
            bundle: Record<string, unknown>,
        ) => void;
        generate.call({ info: (message) => messages.push(message) }, {}, bundle);
        expect(Object.hasOwn(bundle, "orphan.png")).toBe(false);
        expect(Object.hasOwn(bundle, "used.png")).toBe(true);
        expect(messages[0]).toContain("pruned 1 orphaned asset(s)");
    },
);

check(
    "shallot plugin: reloads on scene, manifest and model changes",
    {
        claim: "dev clients fully reload when a scene, shallot.json or public model changes",
    },
    () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-watch-"));
        try {
            const publicDir = join(root, "public");
            mkdirSync(publicDir);
            writeFileSync(join(root, "shallot.json"), "{}");
            const listeners = new Map<string, (file: string) => void>();
            const reloads: unknown[] = [];
            let invalidations = 0;
            const server = {
                middlewares: { use() {} },
                watcher: {
                    add() {},
                    on(event: string, listener: (file: string) => void) {
                        listeners.set(event, listener);
                    },
                },
                ws: { send(message: unknown) { reloads.push(message); } },
                moduleGraph: {
                    getModuleById() { return {}; },
                    invalidateModule() { invalidations++; },
                },
            };
            const plugin = shallot(root);
            const configure = plugin.configureServer as unknown as (server: unknown) => void;
            configure(server);
            const changed = listeners.get("change");
            if (!changed) throw new Error("shallot plugin did not install its file watcher");
            changed(join(root, "shallot.json"));
            changed(join(root, "arena.scene"));
            changed(join(publicDir, "ship.glb"));
            expect(reloads).toHaveLength(3);
            expect(invalidations).toBe(2);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);
