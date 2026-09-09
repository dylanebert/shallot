// The pack-time recipe projection, exercised on a fixture corpus and on the real one. The projection is
// what makes `shallot recipe gpu-particles` self-contained while one owner (`packages/shallot-gpu-particles`)
// keeps the only editable copy of the implementation: without it the copied-out project would name an
// unpublished workspace package and fail `bun install`. So the properties asserted here are the ones a
// second source copy would otherwise be maintained for.

import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
    dropDependency,
    inlineManifestEntry,
    projectRecipes,
    shippedIndex,
    VENDORED_PLUGINS,
} from "./prepack";

const REPO = resolve(import.meta.dir, "../../..");

/** a two-recipe corpus: one plain recipe with a smoke plugin, one whose plugin lives in a package. */
function fixture(): { root: string; src: string; dest: string } {
    const root = mkdtempSync(join(tmpdir(), "prepack-fixture-"));
    const src = join(root, "examples");
    mkdirSync(join(src, "recipes/plain/src"), { recursive: true });
    writeFileSync(join(src, "recipes/plain/src/smoke.ts"), "export default { name: 'Smoke' };\n");
    writeFileSync(join(src, "recipes/plain/src/build.ts"), "export default { name: 'Build' };\n");
    writeFileSync(join(src, "recipes/plain/tsconfig.json"), "{}\n");
    writeFileSync(
        join(src, "recipes/plain/shallot.json"),
        `${JSON.stringify({ plugins: { Build: "./src/build", Smoke: "./src/smoke" } }, null, 4)}\n`,
    );
    mkdirSync(join(src, "recipes/vendored/src"), { recursive: true });
    writeFileSync(
        join(src, "recipes/vendored/src/smoke.ts"),
        "export default { name: 'Smoke' };\n",
    );
    writeFileSync(
        join(src, "recipes/vendored/shallot.json"),
        `${JSON.stringify({ plugins: { Producer: "producer-pkg", Smoke: "./src/smoke" } }, null, 4)}\n`,
    );
    writeFileSync(
        join(src, "recipes/vendored/package.json"),
        `${JSON.stringify({ name: "vendored", dependencies: { engine: "workspace:*", "producer-pkg": "workspace:*" } }, null, 4)}\n`,
    );
    mkdirSync(join(root, "producer/src"), { recursive: true });
    writeFileSync(join(root, "producer/src/index.ts"), "export { default } from './impl';\n");
    writeFileSync(join(root, "producer/src/impl.ts"), "export default { name: 'Producer' };\n");
    return { root, src, dest: join(root, "projection") };
}

const TABLE = [
    { recipe: "vendored", from: "producer", dependency: "producer-pkg", into: "producer" },
] as const;

describe("recipe projection", () => {
    test("inlines a vendored plugin and drops its workspace dependency", () => {
        const { root, src, dest } = fixture();
        try {
            mkdirSync(dest, { recursive: true });
            projectRecipes(root, src, dest, TABLE);
            const recipe = join(dest, "recipes/vendored");
            expect(existsSync(join(recipe, "src/producer/index.ts"))).toBe(true);
            expect(existsSync(join(recipe, "src/producer/impl.ts"))).toBe(true);
            const manifest = JSON.parse(readFileSync(join(recipe, "shallot.json"), "utf8"));
            expect(manifest.plugins.Producer).toBe("./src/producer/index");
            // the smoke plugin (file + entry) still goes, and the vendored entry did not resurrect it
            expect(existsSync(join(recipe, "src/smoke.ts"))).toBe(false);
            expect(manifest.plugins.Smoke).toBeUndefined();
            const pkg = JSON.parse(readFileSync(join(recipe, "package.json"), "utf8"));
            expect(pkg.dependencies["producer-pkg"]).toBeUndefined();
            expect(pkg.dependencies.engine).toBe("workspace:*");
            // no projected byte anywhere in the recipe still names the unpublished package
            const named = [
                readFileSync(join(recipe, "shallot.json"), "utf8"),
                readFileSync(join(recipe, "package.json"), "utf8"),
                readFileSync(join(recipe, "src/producer/index.ts"), "utf8"),
            ].filter((text) => text.includes("producer-pkg"));
            expect(named).toEqual([]);
            // the untouched recipe keeps its own shape
            expect(existsSync(join(dest, "recipes/plain/src/build.ts"))).toBe(true);
            expect(existsSync(join(dest, "recipes/plain/tsconfig.json"))).toBe(false);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    test("refuses a table entry whose subject moved", () => {
        const { root, src, dest } = fixture();
        try {
            mkdirSync(dest, { recursive: true });
            expect(() => projectRecipes(root, src, dest, [{ ...TABLE[0], from: "gone" }])).toThrow(
                /vendored plugin source missing/,
            );
            rmSync(join(dest, "recipes"), { recursive: true, force: true });
            expect(() =>
                projectRecipes(root, src, dest, [{ ...TABLE[0], recipe: "plain" }]),
            ).toThrow(/no longer names producer-pkg/);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    test("both authored manifest forms move, and nothing else does", () => {
        const text = `${JSON.stringify(
            { plugins: { A: "pkg", B: ["pkg", false], C: "pkg-other", D: true } },
            null,
            4,
        )}\n`;
        const rewritten = JSON.parse(inlineManifestEntry(text, "pkg", "./src/p/index"));
        expect(rewritten.plugins).toEqual({
            A: "./src/p/index",
            B: ["./src/p/index", false],
            C: "pkg-other",
            D: true,
        });
        const pkg = JSON.parse(
            dropDependency(
                `${JSON.stringify({ dependencies: { pkg: "workspace:*", keep: "1.0.0" }, peerDependencies: { pkg: "*" } })}\n`,
                "pkg",
            ),
        );
        expect(pkg.dependencies).toEqual({ keep: "1.0.0" });
        expect(pkg.peerDependencies).toEqual({});
    });
});

describe("the real corpus", () => {
    test("every vendored-plugin row names a live package, recipe and manifest entry", () => {
        expect(VENDORED_PLUGINS.length).toBeGreaterThan(0);
        for (const entry of VENDORED_PLUGINS) {
            expect(existsSync(resolve(REPO, entry.from, "src/index.ts"))).toBe(true);
            const manifest = readFileSync(
                resolve(REPO, "examples/recipes", entry.recipe, "shallot.json"),
                "utf8",
            );
            expect(manifest).toContain(`"${entry.dependency}"`);
            const pkg = JSON.parse(
                readFileSync(
                    resolve(REPO, "examples/recipes", entry.recipe, "package.json"),
                    "utf8",
                ),
            ) as { dependencies?: Record<string, string> };
            expect(pkg.dependencies?.[entry.dependency]).toBe("workspace:*");
            // one implementation: the recipe keeps no source file of its own but its smoke plugin
            const own = new Bun.Glob("**/*.ts").scanSync({
                cwd: resolve(REPO, "examples/recipes", entry.recipe, "src"),
            });
            expect([...own].sort()).toEqual(["smoke.ts"]);
        }
    });

    test("the shipped index still derives from the corpus doc", () => {
        const index = shippedIndex(resolve(REPO, "examples"));
        expect(index).toContain("## Recipes");
        expect(index).not.toContain("## Bench");
        expect(index).toContain("shallot recipe");
    });
});
