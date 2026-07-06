import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import type { Rollup } from "vite";
import { assetSrc, orphanedAssets, recentlySaved, writeAsset } from "./vite";

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

// the live asset-swap path → cache-key mapping (roadmap glTF sub-stage 5). A changed model file's path
// maps to the `src` the editor hands `invalidate`; everything else falls through to the scene/manifest watch.
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

// the binary write behind POST /__api/asset — round-trip, guard, collision policy, watcher suppression.
// Real fs against a temp dir: the function IS fs orchestration, mocking it would test nothing.
describe("writeAsset", () => {
    let pub: string;
    // bytes a UTF-8 round-trip would corrupt (invalid sequences → replacement chars) — the reason the
    // endpoint exists at all, so the round-trip asserts on exactly these
    const Glb = Buffer.from([0x67, 0x6c, 0x54, 0x46, 0xff, 0xfe, 0x00, 0x80, 0xc3, 0x28]);
    const Other = Buffer.from([0x67, 0x6c, 0x54, 0x46, 0x01, 0x02, 0x03]);

    beforeEach(() => {
        pub = mkdtempSync(join(tmpdir(), "shallot-asset-"));
    });
    afterEach(() => {
        rmSync(pub, { recursive: true, force: true });
    });

    test("binary round-trip is byte-exact", () => {
        const r = writeAsset(pub, "model.glb", Glb);
        expect(r).toEqual({ status: 200, src: "model.glb" });
        expect(Buffer.from(readFileSync(join(pub, "model.glb"))).equals(Glb)).toBe(true);
    });

    test("a traversal name is rejected, nothing written", () => {
        const r = writeAsset(pub, "../evil.glb", Glb);
        expect(r.status).toBe(403);
        expect(existsSync(join(pub, "..", "evil.glb"))).toBe(false);
    });

    test("a sibling-prefix escape is rejected (separator-aware guard)", () => {
        const r = writeAsset(join(pub, "public"), "../public-evil/x.glb", Glb);
        expect(r.status).toBe(403);
    });

    test("differing bytes under an existing name dedupe to name-2.ext, original untouched", () => {
        writeAsset(pub, "model.glb", Glb);
        const r = writeAsset(pub, "model.glb", Other);
        expect(r.src).toBe("model-2.glb");
        expect(Buffer.from(readFileSync(join(pub, "model.glb"))).equals(Glb)).toBe(true);
        expect(Buffer.from(readFileSync(join(pub, "model-2.glb"))).equals(Other)).toBe(true);
    });

    test("identical bytes reuse the existing file without a write", () => {
        writeAsset(pub, "model.glb", Glb);
        const before = statSync(join(pub, "model.glb")).mtimeMs;
        writeFileSync(join(pub, "marker"), ""); // ensure the clock can move between writes
        const r = writeAsset(pub, "model.glb", Glb);
        expect(r.src).toBe("model.glb");
        expect(statSync(join(pub, "model.glb")).mtimeMs).toBe(before);
    });

    test("identical bytes reuse an earlier deduped copy", () => {
        writeAsset(pub, "model.glb", Glb);
        writeAsset(pub, "model.glb", Other); // → model-2.glb
        const r = writeAsset(pub, "model.glb", Other);
        expect(r.src).toBe("model-2.glb");
        expect(existsSync(join(pub, "model-3.glb"))).toBe(false);
    });

    test("a sidecar path creates its subdirectories (a .gltf's texture layout)", () => {
        const r = writeAsset(pub, "scan/textures/wall.png", Glb);
        expect(r.src).toBe("scan/textures/wall.png");
        expect(existsSync(join(pub, "scan", "textures", "wall.png"))).toBe(true);
    });

    test("a missing public dir is created (a fresh project's first import)", () => {
        const fresh = join(pub, "public");
        const r = writeAsset(fresh, "model.glb", Glb);
        expect(r.src).toBe("model.glb");
        expect(existsSync(join(fresh, "model.glb"))).toBe(true);
    });

    test("a write marks its path as the editor's own — the watcher suppression seam", () => {
        writeAsset(pub, "model.glb", Glb);
        expect(recentlySaved(join(pub, "model.glb"))).toBe(true);
        expect(recentlySaved(join(pub, "other.glb"))).toBe(false);
    });
});
