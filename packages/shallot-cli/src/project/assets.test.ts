import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { contentType, manifestWarnings, readManifest, resolveAssetPath } from "./assets";

// a fresh project dir per test — the shape every fs-reader test needs
function projectDir(): string {
    return mkdtempSync(join(tmpdir(), "shallot-assets-test-"));
}

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

// MIME for a project's public/ assets — without it res.end() sends no Content-Type and a browser
// misclassifies the response (an SVG icon rejected as a favicon).
describe("contentType", () => {
    test("maps the project-asset extensions the MIME table declares", () => {
        expect(contentType("icon.svg")).toBe("image/svg+xml");
        expect(contentType("model.glb")).toBe("model/gltf-binary");
        expect(contentType("data.wasm")).toBe("application/wasm");
    });

    test("matches the extension case-insensitively", () => {
        expect(contentType("ICON.SVG")).toBe("image/svg+xml");
    });

    test("is undefined for an unmapped or missing extension", () => {
        expect(contentType("readme.md")).toBeUndefined();
        expect(contentType("noext")).toBeUndefined();
    });
});

// resolveAssetPath is the extraction of configureServer's join-then-guard: given `new URL(req.url,
// "http://localhost")`'s own WHATWG dot-segment normalization already strips every crafted req.url this
// stage tried (encoded and raw ".." forms alike — see the extraction's own comment), the guard's
// `!filePath.startsWith(dir)` branch is unreachable through any real request. Tested directly on a raw
// pathname the URL layer never gets to normalize first, so the guard's own logic is still pinned.
describe("resolveAssetPath", () => {
    test("resolves an existing file under dir", () => {
        const dir = projectDir();
        writeFileSync(join(dir, "icon.svg"), "<svg/>");
        expect(resolveAssetPath(dir, "/icon.svg")).toBe(join(dir, "icon.svg"));
    });

    test("returns null for a file that doesn't exist under dir", () => {
        const dir = projectDir();
        expect(resolveAssetPath(dir, "/missing.svg")).toBeNull();
    });

    test("returns null for a directory — not a file", () => {
        const dir = projectDir();
        mkdirSync(join(dir, "sub"));
        expect(resolveAssetPath(dir, "/sub")).toBeNull();
    });

    test("rejects a pathname that joins outside dir — the traversal guard", () => {
        // a sibling file that WOULD resolve if the guard didn't reject the escape first
        const parent = projectDir();
        const dir = join(parent, "public");
        mkdirSync(dir);
        writeFileSync(join(parent, "secret.txt"), "top secret");
        expect(resolveAssetPath(dir, "/../secret.txt")).toBeNull();
    });

    test("rejects a sibling dir whose name merely extends dir's — a prefix is not a boundary", () => {
        // the case a string-prefix guard gets wrong: "/a/public-secrets".startsWith("/a/public") is
        // true, but the escape target is a sibling of dir, not a descendant. Only a separator-aware
        // boundary check rejects it.
        const parent = projectDir();
        const dir = join(parent, "public");
        const sibling = join(parent, "public-secrets");
        mkdirSync(dir);
        mkdirSync(sibling);
        writeFileSync(join(sibling, "secret.txt"), "top secret");
        expect(resolveAssetPath(dir, join("..", "public-secrets", "secret.txt"))).toBeNull();
    });
});

// readManifest: the toolchain boundary between a project's raw shallot.json and the normalized Manifest
// projectPlugin's `load` hook generates from.
describe("readManifest", () => {
    test("returns {} when the project has no manifest — a scene-only project", () => {
        expect(readManifest(projectDir())).toEqual({});
    });

    test("parses a valid manifest", () => {
        const dir = projectDir();
        writeFileSync(
            join(dir, "shallot.json"),
            JSON.stringify({ scene: "main.scene", plugins: { Render: false } }),
        );
        expect(readManifest(dir)).toEqual({ scene: "main.scene", plugins: { Render: false } });
    });

    test("tolerates a corrupt manifest — returns {} rather than throwing", () => {
        const dir = projectDir();
        writeFileSync(join(dir, "shallot.json"), "{ not json");
        expect(readManifest(dir)).toEqual({});
    });
});
