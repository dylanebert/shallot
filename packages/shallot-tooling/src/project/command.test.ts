import { describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { EXIT_OK, EXIT_SETUP, loadLocalPlugins, planProject } from "./command";
import { generateModuleFromPlan } from "./generate";
import { readProject } from "./host";

// The command entry is the one seam `bin/tui.ts` reaches a project through, so its arms run from a real
// external project root (a temp dir outside this repo): a project root, its own `node_modules`, and
// nothing of this package's private layout.

function externalRoot(name: string): string {
    return mkdtempSync(join(tmpdir(), `shallot-command-${name}-`));
}

/** a local plugin module whose evaluation is observable: it writes `sentinel` at import time, so a
 *  plugin that must never load is proven not to have loaded rather than merely absent from a list. */
function writeLocalPlugin(root: string, file: string, sentinel: string, name: string): void {
    mkdirSync(join(root, "src"), { recursive: true });
    writeFileSync(
        join(root, "src", file),
        [
            `import { writeFileSync } from "node:fs";`,
            `writeFileSync(${JSON.stringify(sentinel)}, "evaluated");`,
            `export default { name: ${JSON.stringify(name)} };`,
            "",
        ].join("\n"),
    );
}

describe("planProject", () => {
    test("a dir holding neither a manifest nor a scene exits setup with the scaffold hint", () => {
        const root = externalRoot("no-project");
        const result = planProject(root);
        expect(result.code).toBe(EXIT_SETUP);
        expect(result.plan).toBeNull();
        expect(result.errors.join("\n")).toContain("bun create shallot");
    });

    test("a manifest-only external root plans clean, exit 0", () => {
        const root = externalRoot("manifest-only");
        writeFileSync(join(root, "shallot.json"), JSON.stringify({ plugins: { Cells: true } }));
        const result = planProject(root);
        expect(result.code).toBe(EXIT_OK);
        expect(result.errors).toEqual([]);
        expect(result.plan?.engine).toContain("Cells");
    });

    test("a scene-only external root plans clean, exit 0, with its scene discovered", () => {
        const root = externalRoot("scene-only");
        mkdirSync(join(root, "public"), { recursive: true });
        writeFileSync(join(root, "public", "main.scene"), "");
        const result = planProject(root);
        expect(result.code).toBe(EXIT_OK);
        expect(result.plan?.scenes).toEqual([join("public", "main.scene")]);
    });

    test("an installed bare plugin beside a local one plans clean from the project root", () => {
        const root = externalRoot("bare-plugin");
        const dir = join(root, "node_modules", "bare-plugin");
        mkdirSync(dir, { recursive: true });
        writeFileSync(
            join(dir, "package.json"),
            JSON.stringify({ name: "bare-plugin", version: "1.0.0", main: "index.js" }),
        );
        writeFileSync(join(dir, "index.js"), `export default { name: "Bare" };\n`);
        writeLocalPlugin(root, "spin.ts", join(root, "spin.txt"), "Spin");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { Spin: "./src/spin", Bare: "bare-plugin" } }),
        );
        const result = planProject(root);
        expect(result.code).toBe(EXIT_OK);
        expect(result.plan?.locals.map((l) => l.name)).toEqual(["Spin", "Bare"]);
    });

    test("a missing dependency exits setup before any plugin module is evaluated", () => {
        const root = externalRoot("missing-dep");
        const sentinel = join(root, "evaluated.txt");
        writeLocalPlugin(root, "spin.ts", sentinel, "Spin");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { Spin: "./src/spin", Gone: "not-installed" } }),
        );
        const result = planProject(root);
        expect(result.code).toBe(EXIT_SETUP);
        expect(result.plan).toBeNull();
        expect(result.errors.join("\n")).toContain("not-installed");
        // the sibling plugin resolves fine — validation runs to completion before any load, so nothing
        // was imported and there is no half-loaded project to clean up.
        expect(existsSync(sentinel)).toBe(false);
    });
});

describe("loadLocalPlugins", () => {
    test("an enabled local is evaluated; a disabled local is never evaluated", async () => {
        const root = externalRoot("disabled");
        const onSentinel = join(root, "on.txt");
        const offSentinel = join(root, "off.txt");
        writeLocalPlugin(root, "on.ts", onSentinel, "On");
        writeLocalPlugin(root, "off.ts", offSentinel, "Off");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { On: "./src/on", Off: ["./src/off", false] } }),
        );
        const result = planProject(root);
        expect(result.code).toBe(EXIT_OK);
        if (!result.plan) throw new Error("no plan");
        const plugins = await loadLocalPlugins(result.plan);
        expect(plugins.map((p) => p.name)).toEqual(["On"]);
        expect(existsSync(onSentinel)).toBe(true);
        expect(existsSync(offSentinel)).toBe(false);
    });

    test("a module that default-exports no Plugin fails loud, naming its manifest key", async () => {
        const root = externalRoot("bad-default");
        mkdirSync(join(root, "src"), { recursive: true });
        writeFileSync(join(root, "src", "nope.ts"), "export const notDefault = 1;\n");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { Nope: "./src/nope" } }),
        );
        const result = planProject(root);
        expect(result.code).toBe(EXIT_OK);
        if (!result.plan) throw new Error("no plan");
        await expect(loadLocalPlugins(result.plan)).rejects.toThrow('"Nope"');
    });
});

describe("one resolved plan, two consumers", () => {
    test("the browser generator emits exactly the local modules the command loads", async () => {
        const root = externalRoot("shared-plan");
        writeLocalPlugin(root, "on.ts", join(root, "on.txt"), "On");
        writeFileSync(
            join(root, "shallot.json"),
            JSON.stringify({ plugins: { Cells: true, On: "./src/on", Off: ["./src/off", false] } }),
        );
        // one plan object, handed to both consumers — the plan identity the A1 row names.
        const plan = readProject(root);
        const emitted = generateModuleFromPlan(plan);
        const loaded = await loadLocalPlugins(plan);

        expect(loaded.map((p) => p.name)).toEqual(["On"]);
        expect(plan.locals.length).toBe(1);
        for (const local of plan.locals) expect(emitted).toContain(JSON.stringify(local.path));
        // the disabled plugin reaches neither consumer: it is imported by neither (the manifest text
        // itself still rides the module as data, which is why the assertion reads the import lines)
        const imports = emitted.split("\n").filter((line) => line.startsWith("import "));
        expect(imports.filter((line) => line.includes("/src/off"))).toEqual([]);
        expect(imports.filter((line) => line.startsWith("import _l")).length).toBe(1);
        // and the engine half agrees: every planned engine plugin is imported by name. Glaze rides the
        // browser module (it composites the swapchain) and is dropped only for a headless run.
        for (const name of plan.engine) expect(emitted).toContain(`${name}Plugin`);
        expect(emitted).toContain("GlazePlugin");
    });
});

// The purity claim of the A1 row: the command entry loads in a bare `bun` process with no Vite in its
// loaded module graph and no browser/GPU global installed. `require.cache` is the observable — Bun
// records every loaded dependency there — and the control below imports `bin/toolchain.ts`, which really
// does import Vite, so a reader that could never see Vite fails that arm instead of passing this one
// vacuously.
const ENTRY = resolve(import.meta.dir, "command.ts");
const TOOLCHAIN = resolve(import.meta.dir, "../../bin/toolchain.ts");
const BROWSER_GLOBALS = [
    "document",
    "window",
    "ResizeObserver",
    "HTMLCanvasElement",
    "GPUBufferUsage",
    "GPUShaderStage",
    "GPUMapMode",
    "GPUTextureUsage",
];

function importInBareBun(entry: string): { vite: string[]; globals: string[]; modules: number } {
    const script = [
        `await import(${JSON.stringify(entry)});`,
        `const loaded = Object.keys(require.cache);`,
        `const vite = loaded.filter((m) => /(^|\\/)(vite|rollup|esbuild)([\\/@]|$)/.test(m));`,
        `const globals = ${JSON.stringify(BROWSER_GLOBALS)}.filter((k) => k in globalThis);`,
        `console.log(JSON.stringify({ vite, globals, modules: loaded.length }));`,
    ].join("\n");
    const run = spawnSync("bun", ["-e", script], { encoding: "utf8", timeout: 60_000 });
    if (run.status !== 0) throw new Error(`bare bun import failed: ${run.stderr}`);
    return JSON.parse(run.stdout.trim().split("\n").at(-1) ?? "{}");
}

describe("the pure entry in a bare bun process", () => {
    test("importing the command entry pulls in no Vite module and installs no browser/GPU global", () => {
        const subject = importInBareBun(ENTRY);
        expect(subject.vite).toEqual([]);
        expect(subject.globals).toEqual([]);
        expect(subject.modules).toBeGreaterThan(0);
    });

    test("control: the Vite-importing toolchain does load Vite through the same reader", () => {
        expect(importInBareBun(TOOLCHAIN).vite.length).toBeGreaterThan(0);
    });
});
