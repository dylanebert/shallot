import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { searchForWorkspaceRoot } from "vite";
import { buildConfig, synthIndex } from "./build";
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

describe("buildConfig", () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-build-"));

    test("roots vite at the project with the TGSL transform + project plugins, no dev-only synth-index", () => {
        const config = buildConfig(dir);
        expect(config.root).toBe(dir);
        // buildWeb writes the synthesized entry itself and points vite straight at it — unlike
        // devConfig, there's no synthIndexPlugin middleware serving it on demand
        expect(config.plugins.map((p) => p.name)).toEqual(["unplugin-typegpu", "shallot-project"]);
    });

    test("outputs to dist/, cleared each build, with relative asset URLs (portable to a subpath / file://)", () => {
        const config = buildConfig(dir);
        expect(config.base).toBe("./");
        expect(config.build).toEqual({ target: "esnext", outDir: "dist", emptyOutDir: true });
    });
});

describe("devConfig", () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-dev-"));
    writeFileSync(join(dir, "shallot.json"), '{ "scene": null, "plugins": {} }\n');

    test("roots vite at the project with the TGSL transform + project + synth-index plugins", () => {
        const config = devConfig(dir, "demo", { open: false });
        expect(config.root).toBe(dir);
        // a manifest project ships no vite config of its own, so the CLI is the only place the TGSL
        // transform can come from — without it every engine shader resolves with no metadata
        expect(config.plugins.map((p) => p.name)).toEqual([
            "unplugin-typegpu",
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

// the invariant that motivated folding buildConfig out of dev.ts, which neither describe above asserts:
// one project previews through devConfig and ships through buildConfig, so the two must resolve the same
// project the same way. A root divergence loads a different manifest in dev than in the shipped bundle;
// a plugin-order divergence runs the TGSL transform after the project plugin, leaving `virtual:project`'s
// engine imports untransformed — and the CLI is the only place either config can come from.
describe("devConfig ∩ buildConfig — the shared prefix", () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-prefix-"));
    writeFileSync(join(dir, "shallot.json"), '{ "scene": null, "plugins": {} }\n');

    test("both root vite at the project and both run typegpu first", () => {
        const build = buildConfig(dir);
        const dev = devConfig(dir, "demo", { open: false });
        expect(build.root).toBe(dev.root);

        const buildPlugins = build.plugins.map((p) => p.name);
        const devPlugins = dev.plugins.map((p) => p.name);
        expect(buildPlugins[0]).toBe("unplugin-typegpu");
        // dev's list is build's plus its own synth-index middleware — asserted as a prefix, not as equal
        // membership, so a reorder on either side goes red while the two sets stay identical
        expect(devPlugins.slice(0, buildPlugins.length)).toEqual(buildPlugins);
    });
});
