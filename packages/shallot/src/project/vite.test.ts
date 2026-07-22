import { describe, expect, test } from "bun:test";
import type { Rollup } from "vite";
import { assetSrc, manifestWarnings, orphanedAssets } from "./vite";

// minimal bundle builders — the pure prune reads only type / fileName / code|source
const chunk = (fileName: string, code: string) =>
    ({ type: "chunk", fileName, code }) as Rollup.OutputChunk;
const asset = (fileName: string, source: string | Uint8Array) =>
    ({ type: "asset", fileName, source }) as Rollup.OutputAsset;
const bundle = (...files: (Rollup.OutputChunk | Rollup.OutputAsset)[]) =>
    Object.fromEntries(files.map((f) => [f.fileName, f])) as Rollup.OutputBundle;

describe("orphanedAssets", () => {
    test("drops a wasm no chunk references (the new-URL over-emit), keeps the live entry", () => {
        // orbit's shape: codec branch tree-shaken dead, yet vite emitted the wasm at transform time
        const b = bundle(
            asset("index.html", `<script src="./assets/index-AAA.js"></script>`),
            chunk("assets/index-AAA.js", `console.log("imports only Orbit")`),
            asset("assets/draco-BBB.wasm", new Uint8Array([0, 1, 2])),
        );
        expect(orphanedAssets(b)).toEqual(["assets/draco-BBB.wasm"]);
    });

    test("keeps a codec the project actually uses — its hashed name survives in a live chunk", () => {
        const b = bundle(
            chunk("assets/index-AAA.js", `new URL("./assets/draco-BBB.wasm", import.meta.url)`),
            asset("assets/draco-BBB.wasm", new Uint8Array([0, 1, 2])),
        );
        expect(orphanedAssets(b)).toEqual([]);
    });

    test("an asset referenced only by another asset survives via the fixpoint (css → font)", () => {
        // the case a single chunk-only scan gets wrong: the font is reached only through the css
        const b = bundle(
            asset("index.html", `<link rel="stylesheet" href="./assets/style-CCC.css">`),
            asset("assets/style-CCC.css", `@font-face{src:url(./font-DDD.woff2)}`),
            asset("assets/font-DDD.woff2", new Uint8Array([0])),
            asset("assets/orphan-EEE.wasm", new Uint8Array([9])),
        );
        expect(orphanedAssets(b)).toEqual(["assets/orphan-EEE.wasm"]);
    });
});

// the live asset-swap path → cache-key mapping. A changed model file's path maps to its public-relative
// cache src (the watcher's full-reload trigger); everything else falls through to the scene/manifest watch.
describe("assetSrc", () => {
    const pub = "/proj/public";

    test("maps a .glb under the public dir to its public-relative cache src", () => {
        expect(assetSrc("/proj/public/box.glb", [pub])).toBe("box.glb");
    });

    test("maps a nested .gltf to a /-joined src — the scene mesh-name + readBinary use", () => {
        expect(assetSrc("/proj/public/sponza/Sponza.gltf", [pub])).toBe("sponza/Sponza.gltf");
    });

    test("ignores a non-model file — the scene/manifest + sidecar boundary", () => {
        // a .scene / shallot.json ride their own watch; a sidecar (.bin) or separate texture re-decodes
        // through its container, which a re-export rewrites — the deliberate scope boundary for this stage
        expect(assetSrc("/proj/public/sponza/Sponza.bin", [pub])).toBeNull();
        expect(assetSrc("/proj/public/sponza/wall.png", [pub])).toBeNull();
        expect(assetSrc("/proj/scenes/a.scene", [pub])).toBeNull();
    });

    test("ignores a model outside every public dir (not a fetchable cache src)", () => {
        expect(assetSrc("/proj/src/box.glb", [pub])).toBeNull();
        expect(assetSrc("/other/box.glb", [pub])).toBeNull();
    });

    test("resolves against the matching dir when several public dirs are given", () => {
        const dirs = ["/proj/public", "/shared/public"];
        expect(assetSrc("/shared/public/tree.glb", dirs)).toBe("tree.glb");
        expect(assetSrc("/proj/public/box.glb", dirs)).toBe("box.glb");
    });

    test("matches the extension case-insensitively", () => {
        expect(assetSrc("/proj/public/Model.GLB", [pub])).toBe("Model.GLB");
    });
});

// the loud manifest boundary: a corrupt file and an unknown-plugin key each yield a warning line
describe("manifestWarnings", () => {
    const known = new Set(["Orbit", "Render"]);

    test("flags an unparseable manifest", () => {
        expect(manifestWarnings("{ not json", known)).toEqual([
            "not valid JSON, ignored (the project runs with default plugins)",
        ]);
    });

    test("flags a bool key naming no engine plugin, naming the key", () => {
        const w = manifestWarnings(`{ "plugins": { "Orbitt": true, "Render": false } }`, known);
        expect(w).toHaveLength(1);
        expect(w[0]).toContain("Orbitt");
    });

    test("is silent for a valid manifest — known engine bools + a local specifier", () => {
        expect(
            manifestWarnings(
                `{ "plugins": { "Orbit": true, "Render": false, "Demo": "./src/demo" } }`,
                known,
            ),
        ).toEqual([]);
    });
});
