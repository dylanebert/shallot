import { describe, expect, test } from "bun:test";
import type { Plugin } from "vite";
import { composeViteConfig, flattenPlugins, type ProjectConfig } from "./toolchain";

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
});
