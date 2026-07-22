import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { searchForWorkspaceRoot } from "vite";
import { synthIndex } from "./build";
import { devConfig } from "./dev";

describe("synthIndex", () => {
    test("synthesizes an entry that runs the manifest", () => {
        const html = synthIndex("demo");
        expect(html).toContain('id="canvas"');
        expect(html).toContain('from "@dylanebert/shallot"');
        expect(html).toContain('from "virtual:project"');
        // the build's resolved plugin set is authoritative — re-adding defaults would resurrect a disabled one
        expect(html).toContain("defaults: false");
    });
});

describe("devConfig", () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-dev-"));
    writeFileSync(join(dir, "shallot.json"), '{ "scene": null, "plugins": {} }\n');

    test("roots vite at the project with the project + synth-index plugins", () => {
        const config = devConfig(dir, "demo", { open: false });
        expect(config.root).toBe(dir);
        expect(config.plugins.map((p) => p.name)).toEqual([
            "shallot-project",
            "shallot-synth-index",
        ]);
        expect(config.server.fs.allow).toContain(dir);
        // the engine package (and its audio wasm, fetched over /@fs/) lives outside the project dir;
        // restoring vite's default workspace root keeps it servable
        expect(config.server.fs.allow).toContain(searchForWorkspaceRoot(dir));
    });

    test("open defaults true (the CLI) but is overridable", () => {
        expect(devConfig(dir, "demo", {}).server.open).toBe(true);
        expect(devConfig(dir, "demo", { open: false }).server.open).toBe(false);
    });

    test("sends the cross-origin isolation headers so tumble physics multithreads", () => {
        // the COOP/COEP the dev server (shallot dev + verify's project boot) needs for a shared
        // WebAssembly.Memory; a regression here silently degrades tumble to single-thread
        const headers = devConfig(dir, "demo", { open: false }).server.headers;
        expect(headers["Cross-Origin-Opener-Policy"]).toBe("same-origin");
        expect(headers["Cross-Origin-Embedder-Policy"]).toBe("require-corp");
    });
});
