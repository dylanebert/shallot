import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { template } from "../../create-shallot/index";
import { listRecipes, occupied, pinEngine, runRecipe } from "./recipe";
import { ENGINE_REFERENCE, recipeDoc } from "./scaffold";

// a fixture corpus: two runnable recipes + a stray non-recipe dir (no shallot.json)
function corpus(): { recipesDir: string; version: string } {
    const recipesDir = mkdtempSync(join(tmpdir(), "shallot-recipes-"));
    for (const name of ["build-a-scene", "orbit-camera"]) {
        const dir = join(recipesDir, name);
        mkdirSync(join(dir, "src"), { recursive: true });
        writeFileSync(join(dir, "shallot.json"), '{ "scene": null, "plugins": {} }\n');
        writeFileSync(
            join(dir, "package.json"),
            `${JSON.stringify(
                { name, private: true, dependencies: { "@dylanebert/shallot": "workspace:*" } },
                null,
                4,
            )}\n`,
        );
        writeFileSync(join(dir, "src", "main.ts"), "export {};\n");
    }
    mkdirSync(join(recipesDir, "not-a-recipe"), { recursive: true });
    return { recipesDir, version: "1.2.3" };
}

describe("pinEngine", () => {
    const dep = (marker: string) =>
        `{"dependencies":{"@dylanebert/shallot":"workspace:${marker}"}}`;
    const pinned = (marker: string) =>
        JSON.parse(pinEngine(dep(marker), "0.8.0")).dependencies["@dylanebert/shallot"];

    test("workspace:* and bare workspace: pin to the exact version", () => {
        expect(pinned("*")).toBe("0.8.0");
        expect(pinned("")).toBe("0.8.0");
    });

    test("workspace:^ / workspace:~ keep the range operator (bun publish semantics)", () => {
        expect(pinned("^")).toBe("^0.8.0");
        expect(pinned("~")).toBe("~0.8.0");
    });

    test("an explicit range keeps its version verbatim, only stripping workspace:", () => {
        expect(pinned("^1.2.3")).toBe("^1.2.3");
        expect(pinned("~1.2.3")).toBe("~1.2.3");
        expect(pinned("1.2.3")).toBe("1.2.3");
    });

    test("leaves a non-workspace dep untouched", () => {
        const out = pinEngine('{"dependencies":{"@dylanebert/shallot":"^0.7.0"}}', "0.8.0");
        expect(JSON.parse(out).dependencies["@dylanebert/shallot"]).toBe("^0.7.0");
    });

    test("rewrites a workspace dep in every dep field", () => {
        for (const field of ["devDependencies", "peerDependencies", "optionalDependencies"]) {
            const out = pinEngine(`{"${field}":{"@dylanebert/shallot":"workspace:*"}}`, "0.8.0");
            expect(JSON.parse(out)[field]["@dylanebert/shallot"]).toBe("0.8.0");
        }
    });
});

describe("occupied", () => {
    const base = mkdtempSync(join(tmpdir(), "shallot-occ-"));

    test("false for a missing path", () => {
        expect(occupied(join(base, "nope"))).toBe(false);
    });

    test("false for an empty dir, true once it holds an entry", () => {
        const dir = join(base, "d");
        mkdirSync(dir);
        expect(occupied(dir)).toBe(false);
        writeFileSync(join(dir, "x"), "");
        expect(occupied(dir)).toBe(true);
    });

    test("true for an existing regular file (no ENOTDIR)", () => {
        const file = join(base, "notes.txt");
        writeFileSync(file, "mine");
        expect(occupied(file)).toBe(true);
    });
});

describe("listRecipes", () => {
    test("names the shallot.json dirs, sorted, skipping non-recipes", () => {
        const { recipesDir } = corpus();
        expect(listRecipes(recipesDir)).toEqual(["build-a-scene", "orbit-camera"]);
    });

    test("empty for a missing corpus", () => {
        expect(listRecipes(join(tmpdir(), "shallot-absent-corpus"))).toEqual([]);
    });
});

describe("runRecipe", () => {
    test("errors when the corpus is absent", async () => {
        const env = { recipesDir: join(tmpdir(), "shallot-absent-corpus-2"), version: "1.0.0" };
        expect(await runRecipe([], env)).toBe(1);
    });

    test("bare invocation lists and exits 0", async () => {
        expect(await runRecipe([], corpus())).toBe(0);
    });

    test("unknown recipe name exits 1", async () => {
        expect(await runRecipe(["no-such-recipe"], corpus())).toBe(1);
    });

    test("copies a recipe out and pins its engine dep", async () => {
        const env = corpus();
        const dest = join(mkdtempSync(join(tmpdir(), "shallot-copyout-")), "orbit-camera");
        expect(await runRecipe(["orbit-camera", dest], env)).toBe(0);
        expect(existsSync(join(dest, "src", "main.ts"))).toBe(true);
        const pkg = JSON.parse(readFileSync(join(dest, "package.json"), "utf8"));
        expect(pkg.dependencies["@dylanebert/shallot"]).toBe("1.2.3");
    });

    test("emits the agent-surface pointer + a standalone tsconfig", async () => {
        const env = corpus();
        const dest = join(mkdtempSync(join(tmpdir(), "shallot-copyout-doc-")), "orbit-camera");
        expect(await runRecipe(["orbit-camera", dest], env)).toBe(0);
        for (const file of ["AGENTS.md", "CLAUDE.md"]) {
            const doc = readFileSync(join(dest, file), "utf8");
            expect(doc).toContain("node_modules/@dylanebert/shallot/AGENTS.md");
            expect(doc).toContain("node_modules/@dylanebert/shallot/examples/AGENTS.md");
        }
        const tsconfig = JSON.parse(readFileSync(join(dest, "tsconfig.json"), "utf8"));
        expect(tsconfig.compilerOptions.types).toContain("@webgpu/types");
    });

    test("refuses to copy into a non-empty dir", async () => {
        const env = corpus();
        const dest = mkdtempSync(join(tmpdir(), "shallot-nonempty-"));
        writeFileSync(join(dest, "keep.txt"), "mine");
        expect(await runRecipe(["build-a-scene", dest], env)).toBe(1);
        expect(existsSync(join(dest, "shallot.json"))).toBe(false);
    });
});

// The engine pointer is single-sourced in scaffold.ts (bin/ ships; create-shallot ships only its own
// index.ts and can't import it). This guards that create-shallot's own scaffold carries the identical
// stanza — the "stays consistent with" the two sources rely on, since neither can import the other.
describe("scaffold pointer is one source", () => {
    test("recipe copy-out doc embeds the ENGINE_REFERENCE stanza", () => {
        expect(recipeDoc("orbit-camera")).toContain(ENGINE_REFERENCE);
    });

    test("create-shallot's scaffold AGENTS.md / CLAUDE.md carry the same stanza", () => {
        const files = template("starter-app");
        expect(files["AGENTS.md"]).toContain(ENGINE_REFERENCE);
        expect(files["CLAUDE.md"]).toContain(ENGINE_REFERENCE);
    });
});
