import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Plugin } from "vite";
import {
    composeViteConfig,
    flattenPlugins,
    isProject,
    loadProjectConfig,
    type ProjectConfig,
    requireProject,
} from "./toolchain";

const p = (name: string) => ({ name }) as Plugin;

describe("flattenPlugins", () => {
    test("flattens nested option arrays to named plugins in order", async () => {
        const out = await flattenPlugins([p("a"), [p("b"), [p("c")]]]);
        expect(out.map((x) => x.name)).toEqual(["a", "b", "c"]);
    });

    test("resolves promises and drops falsy entries (a conditional plugin)", async () => {
        const out = await flattenPlugins([
            Promise.resolve(p("a")),
            false,
            null,
            undefined,
            [Promise.resolve(p("b"))],
        ]);
        expect(out.map((x) => x.name)).toEqual(["a", "b"]);
    });

    test("undefined config plugins → empty", async () => {
        expect(await flattenPlugins(undefined)).toEqual([]);
    });
});

describe("composeViteConfig", () => {
    const base = { root: "/r", plugins: [p("host-a"), p("host-b")] };

    test("no project config → the base is returned unchanged", () => {
        expect(composeViteConfig(base, null)).toBe(base);
    });

    test("project plugins go first, then the host's; overlay merges over the base", () => {
        const project: ProjectConfig = {
            plugins: [p("svelte")],
            overlay: { define: { X: "1" } },
            path: "/proj/vite.config.ts",
        };
        const out = composeViteConfig(base, project);
        expect((out.plugins as Plugin[]).map((x) => x.name)).toEqual([
            "svelte",
            "host-a",
            "host-b",
        ]);
        expect((out as { define?: unknown }).define).toEqual({ X: "1" });
    });

    test("drop removes a project plugin colliding with a host plugin (the host wins)", () => {
        const project: ProjectConfig = {
            plugins: [p("svelte"), p("host-a")],
            overlay: {},
            path: "x",
        };
        const out = composeViteConfig(base, project, new Set(["host-a"]));
        // the project's own host-a is dropped; the host's host-a + host-b remain
        expect((out.plugins as Plugin[]).map((x) => x.name)).toEqual([
            "svelte",
            "host-a",
            "host-b",
        ]);
    });

    // the zero-config prebundle fix (5b-2f-5): the base declares `optimizeDeps.exclude` so a registry
    // install never gets esbuild-prebundled ahead of the typegpu transform. A project with its own
    // vite.config (an ejected consumer) merges its own `optimizeDeps` over the base — mergeConfig
    // concatenates array fields rather than replacing them, so the base's exclusion must survive
    // alongside the project's own, not get dropped by the overlay.
    test("the base's optimizeDeps.exclude survives a project's own optimizeDeps overlay", () => {
        const hostBase = {
            root: "/r",
            plugins: [],
            optimizeDeps: { exclude: ["@dylanebert/shallot"] },
        };
        const project: ProjectConfig = {
            plugins: [],
            overlay: { optimizeDeps: { exclude: ["some-other-pkg"] } },
            path: "/proj/vite.config.ts",
        };
        const out = composeViteConfig(hostBase, project) as {
            optimizeDeps?: { exclude?: string[] };
        };
        expect(out.optimizeDeps?.exclude).toContain("@dylanebert/shallot");
        expect(out.optimizeDeps?.exclude).toContain("some-other-pkg");
    });

    test("no project config → the base's optimizeDeps.exclude passes through unchanged", () => {
        const hostBase = {
            root: "/r",
            plugins: [],
            optimizeDeps: { exclude: ["@dylanebert/shallot"] },
        };
        const out = composeViteConfig(hostBase, null) as { optimizeDeps?: { exclude?: string[] } };
        expect(out.optimizeDeps?.exclude).toEqual(["@dylanebert/shallot"]);
    });
});

// isProject / requireProject / loadProjectConfig, by temp dir — the dev.test.ts technique
// (mkdtempSync + real fs). requireProject's failure branch (process.exit(1)) is deliberately not
// exercised here: it can't be called without killing the test runner, and the Locked decision forbids
// spying on it — that branch stays a named gap occupant in the registry.
describe("isProject", () => {
    test("true for a dir with a shallot.json manifest", () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-isproject-manifest-"));
        writeFileSync(join(dir, "shallot.json"), "{}\n");
        expect(isProject(dir)).toBe(true);
    });

    test("true for a dir with a nested .scene file and no manifest", () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-isproject-scene-"));
        mkdirSync(join(dir, "scenes"));
        writeFileSync(join(dir, "scenes", "demo.scene"), "");
        expect(isProject(dir)).toBe(true);
    });

    test("false for a dir with neither", () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-isproject-empty-"));
        expect(isProject(dir)).toBe(false);
    });
});

describe("requireProject", () => {
    test("a real project dir returns without throwing or exiting", () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-requireproject-"));
        writeFileSync(join(dir, "shallot.json"), "{}\n");
        expect(() => requireProject(dir)).not.toThrow();
    });
});

describe("loadProjectConfig", () => {
    test("a dir with no vite.config → null (the zero-config manifest path)", async () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-lpc-none-"));
        expect(await loadProjectConfig(dir, "serve", "development")).toBeNull();
    });

    test("a dir with its own vite.config → the flattened plugins + resolve/define overlay + path", async () => {
        // .mjs, not .ts: vite bundles a .ts config through a require() shim that bun's own module
        // loader rejects mid-test ("require() async module is unsupported") — a bun-test-specific
        // incompatibility with vite's config bundler, not something loadProjectConfig itself owns.
        // .mjs skips that bundling path (vite dynamic-imports it directly), which is what a project
        // authoring a config in plain ESM already gets.
        const dir = mkdtempSync(join(tmpdir(), "shallot-lpc-config-"));
        writeFileSync(
            join(dir, "vite.config.mjs"),
            [
                "export default {",
                '  plugins: [{ name: "svelte" }],',
                '  resolve: { alias: { "@": "/src" } },',
                "  define: { FOO: '\"bar\"' },",
                "};",
                "",
            ].join("\n"),
        );
        const config = await loadProjectConfig(dir, "serve", "development");
        if (!config) throw new Error("no config loaded for the vite.config.mjs written above");
        expect(config.plugins.map((x) => x.name)).toEqual(["svelte"]);
        // vite's `define` carries the raw injected-source string verbatim (a string constant needs its
        // own literal quotes, since the value is textually substituted, not evaluated)
        expect((config.overlay as { define?: unknown }).define).toEqual({ FOO: '"bar"' });
        expect(config.path).toBe(join(dir, "vite.config.mjs"));
    });
});
