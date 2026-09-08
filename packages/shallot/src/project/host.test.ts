import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    discoverScenes,
    headlessEngineNames,
    isProject,
    localModuleErrors,
    missingProjectMessage,
    plan,
    readProject,
} from "./host";

const DIR = "/proj";

/** an external project root: a temp dir outside this repo, the shape a real installed consumer has. */
function externalRoot(name: string): string {
    return mkdtempSync(join(tmpdir(), `shallot-host-${name}-`));
}

/** an installed bare plugin — a real `node_modules/<name>` package the project root resolves. */
function installBarePlugin(root: string, name: string): void {
    const dir = join(root, "node_modules", name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
        join(dir, "package.json"),
        JSON.stringify({ name, version: "1.0.0", main: "index.js" }),
    );
    writeFileSync(join(dir, "index.js"), `export default { name: ${JSON.stringify(name)} };\n`);
}

describe("plan", () => {
    test("an empty manifest enables every default, no locals, nothing disabled", () => {
        const { engine, locals, disabled } = plan({}, DIR);
        expect(engine).toEqual(["Slab", "Transforms", "Input", "Render", "Part", "Sear", "Glaze"]);
        expect(locals).toEqual([]);
        expect(disabled).toEqual([]);
    });

    test("a disabled default drops out and is recorded; an enabled extra joins the engine set", () => {
        const { engine, disabled } = plan({ plugins: { Glaze: false, Orbit: true } }, DIR);
        expect(engine).toContain("Orbit");
        expect(engine).not.toContain("Glaze");
        expect(disabled).toEqual(["Glaze"]);
    });

    test("a local specifier keeps its authored spec and resolves project-relative → absolute", () => {
        const { locals } = plan({ plugins: { Spin: "./src/spin" } }, DIR);
        expect(locals).toEqual([{ name: "Spin", spec: "./src/spin", path: "/proj/src/spin" }]);
    });

    test("a disabled local is not planned but is recorded; a bare specifier passes through", () => {
        const { locals, disabled } = plan(
            { plugins: { Off: ["./src/off", false], Pkg: "@scope/foo" } },
            DIR,
        );
        expect(locals).toEqual([{ name: "Pkg", spec: "@scope/foo", path: "@scope/foo" }]);
        expect(disabled).toEqual(["Off"]);
    });

    test("an arbitrary engine plugin (true, not a default) joins the engine set by name", () => {
        // the host trusts any `true` name as an engine plugin — no catalog gate, since it classifies
        // headless with no plugin objects to check against (`assets.ts`'s manifest warning is the
        // reader that names an unknown one).
        expect(plan({ plugins: { Foo: true } }, DIR).engine).toContain("Foo");
    });
});

describe("headlessEngineNames", () => {
    test("drops Glaze (no swapchain headless) and keeps every other enabled engine plugin", () => {
        const names = headlessEngineNames(plan({ plugins: { Cells: true } }, DIR));
        expect(names).not.toContain("Glaze");
        expect(names).toContain("Cells");
        expect(names).toContain("Render");
    });

    test("a manifest that already disabled Glaze gets the same set (the drop is unconditional)", () => {
        const names = headlessEngineNames(plan({ plugins: { Glaze: false, Cells: true } }, DIR));
        expect(names).toEqual(["Slab", "Transforms", "Input", "Render", "Part", "Sear", "Cells"]);
    });
});

describe("isProject / missingProjectMessage", () => {
    test("a manifest project, a scene-only project and neither", () => {
        const manifestOnly = externalRoot("manifest-only");
        writeFileSync(join(manifestOnly, "shallot.json"), "{}\n");
        expect(isProject(manifestOnly)).toBe(true);

        const sceneOnly = externalRoot("scene-only");
        mkdirSync(join(sceneOnly, "scenes"));
        writeFileSync(join(sceneOnly, "scenes", "main.scene"), "");
        expect(isProject(sceneOnly)).toBe(true);

        expect(isProject(externalRoot("empty"))).toBe(false);
    });

    test("the missing-project diagnostic names the dir and the scaffold command", () => {
        const lines = missingProjectMessage("/nope").join("\n");
        expect(lines).toContain("/nope");
        expect(lines).toContain("bun create shallot");
    });
});

describe("readProject", () => {
    test("a manifest-only external root: manifest, no scenes, resolved locals", () => {
        const root = externalRoot("read-manifest");
        installBarePlugin(root, "bare-plugin");
        mkdirSync(join(root, "src"));
        writeFileSync(join(root, "src", "spin.ts"), "export default { name: 'Spin' };\n");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { Cells: true, Spin: "./src/spin", Bare: "bare-plugin" } }),
        );
        const project = readProject(root);
        expect(project.dir).toBe(root);
        expect(project.scenes).toEqual([]);
        expect(project.engine).toContain("Cells");
        expect(project.locals.map((l) => l.name)).toEqual(["Spin", "Bare"]);
        expect(project.locals[0].path).toBe(join(root, "src/spin"));
        expect(project.locals[1].path).toBe("bare-plugin");
    });

    test("a scene-only external root: empty manifest, discovered scenes, default engine set", () => {
        const root = externalRoot("read-scene");
        mkdirSync(join(root, "public", "scenes"), { recursive: true });
        writeFileSync(join(root, "public", "scenes", "main.scene"), "");
        const project = readProject(root);
        expect(project.manifest).toEqual({});
        expect(project.scenes).toEqual([join("public", "scenes", "main.scene")]);
        expect(project.engine).toContain("Render");
        expect(project.locals).toEqual([]);
    });

    test("reads the project's own manifest only — never a planted private engine file", () => {
        // the seam takes a project root and nothing else: a private export target planted beside a
        // pretend installed engine must never enter its read set (the export-map read the TUI used to
        // do). The read set is pinned exactly, so a new read of any private path reds here.
        const root = externalRoot("read-set");
        writeFileSync(join(root, "shallot.json"), JSON.stringify({ plugins: { Cells: true } }));
        const engineRoot = externalRoot("planted-engine");
        mkdirSync(join(engineRoot, "src", "project"), { recursive: true });
        writeFileSync(join(engineRoot, "src", "project", "planted.ts"), "export const p = 1;\n");
        writeFileSync(
            join(engineRoot, "package.json"),
            JSON.stringify({ exports: { "./planted": "./src/project/planted.ts" } }),
        );

        const reads: string[] = [];
        const scanned: string[] = [];
        readProject(root, {
            readFile(path) {
                reads.push(path);
                return readFileSync(path, "utf8");
            },
            discoverScenes(dir) {
                scanned.push(dir);
                return discoverScenes(dir);
            },
        });
        expect(reads).toEqual([join(root, "shallot.json")]);
        expect(scanned).toEqual([root]);
        expect(reads.concat(scanned).some((p) => p.startsWith(engineRoot))).toBe(false);
    });
});

describe("localModuleErrors", () => {
    test("an installed bare plugin and a present relative plugin both resolve", () => {
        const root = externalRoot("deps-ok");
        installBarePlugin(root, "bare-plugin");
        mkdirSync(join(root, "src"));
        writeFileSync(join(root, "src", "spin.ts"), "export default { name: 'Spin' };\n");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { Spin: "./src/spin", Bare: "bare-plugin" } }),
        );
        expect(localModuleErrors(readProject(root))).toEqual([]);
    });

    test("a missing bare dependency and a missing relative file each name their manifest key", () => {
        const root = externalRoot("deps-missing");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { Gone: "not-installed", Spin: "./src/spin" } }),
        );
        const errors = localModuleErrors(readProject(root));
        expect(errors.length).toBe(2);
        expect(errors[0]).toContain('"Gone"');
        expect(errors[0]).toContain("not-installed");
        expect(errors[1]).toContain('"Spin"');
    });

    test("a disabled plugin's missing module is not a dependency error (it is never loaded)", () => {
        const root = externalRoot("deps-disabled");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { Gone: ["not-installed", false] } }),
        );
        expect(localModuleErrors(readProject(root))).toEqual([]);
    });
});
