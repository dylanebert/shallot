import { afterAll, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "@dylanebert/shallot/testing";

setDefaultTimeout(CEILING.node);

import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { expectError } from "../../scripts/test-console";
import { runCreate } from "./create";

const ROOT = resolve(import.meta.dir, "../..");
const SCRATCH = mkdtempSync(join(tmpdir(), "shallot-create-"));
const ENGINE = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
const WRITTEN_FILES = [
    ".gitignore",
    ".github/workflows/test.yml",
    "AGENTS.md",
    "README.md",
    "bunfig.toml",
    "package.json",
    "tests/preload.ts",
];
const MINIMAL_FILES = [
    ".github/workflows/test.yml",
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "bunfig.toml",
    "index.html",
    "package.json",
    "playwright.config.ts",
    "public/icon.svg",
    "public/startup.js",
    "src/welcome/index.ts",
    "src/welcome/welcome.test.ts",
    "startup.e2e.ts",
    "tests/preload.ts",
    "tsconfig.json",
    "vite.config.ts",
];
const MINIMAL_TEMPLATE_FILES = [
    "index.html",
    "playwright.config.ts",
    "public/icon.svg",
    "public/startup.js",
    "src/welcome/index.ts",
    "src/welcome/welcome.test.ts",
    "startup.e2e.ts",
    "tsconfig.json",
    "vite.config.ts",
];
const FIRST_PERSON_TEMPLATE_FILES = [
    "browser.e2e.ts",
    "index.html",
    "playwright.config.ts",
    "public/icon.svg",
    "src/demo.node.ts",
    "src/demo.ts",
    "src/frame.performance.oracle.ts",
    "src/plugins.test.ts",
    "src/plugins.ts",
    "tests/ascent.node.ts",
    "tests/browser-install.ts",
    "tests/browser-setup.test.ts",
    "tests/browser-setup.ts",
    "tests/capture-page.ts",
    "tests/gpu-app.ts",
    "tsconfig.json",
    "vite.config.ts",
];
const FIRST_PERSON_FILES = [...FIRST_PERSON_TEMPLATE_FILES, ...WRITTEN_FILES];
const UNWANTED_FILES = ["CLAUDE.md", ".bun-version", "src/env.d.ts"];

afterAll(() => rmSync(SCRATCH, { recursive: true, force: true }));

function files(dir: string, prefix = ""): string[] {
    return readdirSync(join(dir, prefix), { withFileTypes: true }).flatMap((entry) => {
        const path = prefix ? `${prefix}/${entry.name}` : entry.name;
        return entry.isDirectory() ? files(dir, path) : [path];
    });
}

function expectedJobs(templateFiles: string[]): string[] {
    return [
        {
            name: "cheap",
            exists: templateFiles.some(
                (file) => /\.test\.ts$/.test(file) && !/\.gpu\.test\.ts$/.test(file),
            ),
        },
        { name: "gpu", exists: templateFiles.some((file) => /\.gpu\.test\.ts$/.test(file)) },
        { name: "node", exists: templateFiles.some((file) => /\.node\.ts$/.test(file)) },
        { name: "browser", exists: templateFiles.some((file) => /\.e2e\.ts$/.test(file)) },
    ]
        .filter((tier) => tier.exists)
        .map((tier) => tier.name);
}

function assertProject(destination: string, templateFiles: string[]): void {
    for (const path of UNWANTED_FILES) expect(existsSync(join(destination, path))).toBe(false);

    const workflow = readFileSync(join(destination, ".github/workflows/test.yml"), "utf8");
    const jobs = [...workflow.matchAll(/^ {2}(cheap|gpu|node|browser):$/gm)].map(
        (match) => match[1],
    );
    expect(jobs).toEqual(expectedJobs(templateFiles));

    const manifest = JSON.parse(readFileSync(join(destination, "package.json"), "utf8"));
    expect(manifest.name).toBe(basename(destination));
    expect(manifest.dependencies["@dylanebert/shallot"]).toBe(ENGINE.version);
    expect(manifest.devDependencies["@types/bun"]).toBe(ENGINE.devDependencies["@types/bun"]);
    expect(manifest.devDependencies.playwright).toBe(ENGINE.peerDependencies.playwright);
    expect(manifest.devDependencies.vite).toBe(ENGINE.devDependencies.vite);
    expect(manifest.scripts).toEqual({
        dev: "vite --host 127.0.0.1",
        build: "vite build",
        preview: "vite preview --host 127.0.0.1",
    });
    expect(readFileSync(join(destination, "bunfig.toml"), "utf8")).toBe(
        '[test]\npreload = ["./tests/preload.ts"]\n',
    );
    expect(readFileSync(join(destination, "tests/preload.ts"), "utf8")).toBe(
        'import { shallot } from "@dylanebert/shallot/bun";\nimport { plugin } from "bun";\n\nplugin(shallot({ root: import.meta.dir }));\n',
    );

    const readme = readFileSync(join(destination, "README.md"), "utf8");
    expect(readme).toContain(
        "`bun run dev`, `bun run preview` and the native shell serve the page with cross-origin isolation.",
    );
    expect(readme).toContain(
        "Elsewhere, Physics steps the single-threaded artifact after one log naming the missing headers.",
    );
    expect(readme).not.toContain("Create another project");
    expect(readme).not.toContain("publish targets");

    const source = files(destination)
        .filter((file) => /\.(?:[cm]?[jt]sx?|html)$/.test(file))
        .map((file) => readFileSync(join(destination, file), "utf8"))
        .join("\n");
    expect(manifest.devDependencies.webgpu).toBe(
        source.includes("@dylanebert/shallot/webgpu") ? ENGINE.peerDependencies.webgpu : undefined,
    );
    expect(manifest.devDependencies.typegpu).toBe(
        /(?:from\s*|import\s*(?:\(\s*)?|require\s*\(\s*)["']typegpu(?:\/[^"']*)?["']/.test(source)
            ? ENGINE.peerDependencies.typegpu
            : undefined,
    );
}

test("create writes exactly the 16-file minimal project", () => {
    const destination = join(SCRATCH, "minimal");
    expect(runCreate([destination])).toBe(0);
    expect(files(destination).sort()).toEqual([...MINIMAL_FILES].sort());
    assertProject(destination, MINIMAL_TEMPLATE_FILES);
});

test("create writes its literal first-person files and copies each template file unchanged", () => {
    const destination = join(SCRATCH, "first-person");
    const templateDir = join(ROOT, "examples/first-person");
    expect(runCreate([destination, "--template", "first-person"])).toBe(0);
    expect(files(destination).sort()).toEqual([...FIRST_PERSON_FILES].sort());
    for (const path of FIRST_PERSON_TEMPLATE_FILES) {
        expect(readFileSync(join(destination, path))).toEqual(
            readFileSync(join(templateDir, path)),
        );
    }
    assertProject(destination, FIRST_PERSON_TEMPLATE_FILES);
});

test("create refuses an occupied destination without changing it", () => {
    const destination = join(SCRATCH, "occupied");
    mkdirSync(destination);
    writeFileSync(join(destination, "keep.txt"), "untouched\n");
    expectError(/refusing to create in .*: directory is not empty/);
    expect(runCreate([destination, "--template", "minimal"])).toBe(1);
    expect(files(destination)).toEqual(["keep.txt"]);
    expect(readFileSync(join(destination, "keep.txt"), "utf8")).toBe("untouched\n");
});

test("create declares WebGPU and TypeGPU only when the template imports them", () => {
    const examplesDir = join(SCRATCH, "fixture-examples");
    const templateDir = join(examplesDir, "fixture");
    mkdirSync(join(templateDir, "src"), { recursive: true });
    writeFileSync(
        join(templateDir, "index.html"),
        '<meta name="description" content="Fixture template">\n',
    );
    writeFileSync(
        join(templateDir, "src/fixture.test.ts"),
        'import { setupGlobals } from "@dylanebert/shallot/webgpu";\nconst tgpu = require("typegpu");\nvoid setupGlobals;\nvoid tgpu;\n',
    );
    const destination = join(SCRATCH, "dependency-fixture");
    expect(
        runCreate([destination, "--template", "fixture"], { examplesDir, packageRoot: ROOT }),
    ).toBe(0);
    const manifest = JSON.parse(readFileSync(join(destination, "package.json"), "utf8"));
    expect(manifest.devDependencies.webgpu).toBe(ENGINE.peerDependencies.webgpu);
    expect(manifest.devDependencies.typegpu).toBe(ENGINE.peerDependencies.typegpu);
    const workflow = readFileSync(join(destination, ".github/workflows/test.yml"), "utf8");
    expect(workflow).toContain("  cheap:");
    expect(workflow).not.toContain("  gpu:");
    expect(workflow).not.toContain("  node:");
    expect(workflow).not.toContain("  browser:");
});

test("create declares each package a template imports, Svelte components included, at the engine's range, and checks the components with svelte-check", () => {
    const examplesDir = join(SCRATCH, "svelte-examples");
    const templateDir = join(examplesDir, "svelte");
    const packageRoot = join(SCRATCH, "svelte-engine");
    mkdirSync(join(templateDir, "src"), { recursive: true });
    mkdirSync(packageRoot);
    writeFileSync(
        join(packageRoot, "package.json"),
        JSON.stringify({
            version: ENGINE.version,
            peerDependencies: ENGINE.peerDependencies,
            devDependencies: {
                ...ENGINE.devDependencies,
                svelte: "^5.0.0",
                "@sveltejs/vite-plugin-svelte": "^7.0.0",
                "svelte-check": "^4.0.0",
                unimported: "^1.0.0",
            },
        }),
    );
    writeFileSync(join(templateDir, "index.html"), '<meta name="description" content="Svelte">\n');
    writeFileSync(
        join(templateDir, "vite.config.ts"),
        'import { svelte } from "@sveltejs/vite-plugin-svelte";\nexport default { plugins: [svelte()] };\n',
    );
    writeFileSync(
        join(templateDir, "src/Hud.svelte"),
        '<script lang="ts">\n    import { onMount } from "svelte";\n    onMount(() => {});\n</script>\n',
    );
    const destination = join(SCRATCH, "svelte-project");
    expect(runCreate([destination, "--template", "svelte"], { examplesDir, packageRoot })).toBe(0);
    const manifest = JSON.parse(readFileSync(join(destination, "package.json"), "utf8"));
    expect(manifest.devDependencies.svelte).toBe("^5.0.0");
    expect(manifest.devDependencies["@sveltejs/vite-plugin-svelte"]).toBe("^7.0.0");
    expect(manifest.devDependencies.unimported).toBeUndefined();
    expect(manifest.scripts.typecheck).toBe("svelte-check");
    expect(manifest.devDependencies["svelte-check"]).toBe("^4.0.0");
    const workflow = readFileSync(join(destination, ".github/workflows/test.yml"), "utf8");
    const cheap = workflow.slice(workflow.indexOf("  cheap:"));
    expect(cheap).toContain("      - name: typecheck\n        run: bun run typecheck\n");
    expect(cheap).not.toContain("bun test");
});

test("create detects conditional imports in inline module scripts", () => {
    const examplesDir = join(SCRATCH, "html-examples");
    const templateDir = join(examplesDir, "html-import");
    mkdirSync(templateDir, { recursive: true });
    writeFileSync(
        join(templateDir, "index.html"),
        '<meta name="description" content="HTML module import"><script type="module">import "typegpu";</script>',
    );
    const destination = join(SCRATCH, "html-import-project");
    expect(
        runCreate([destination, "--template", "html-import"], { examplesDir, packageRoot: ROOT }),
    ).toBe(0);
    const manifest = JSON.parse(readFileSync(join(destination, "package.json"), "utf8"));
    expect(manifest.devDependencies.typegpu).toBe(ENGINE.peerDependencies.typegpu);
    assertProject(destination, ["index.html"]);
});

test("an unknown template names installed examples and their descriptions", () => {
    const destination = join(SCRATCH, "unknown-template");
    expect(existsSync(destination)).toBe(false);
    const error = console.error;
    const output: string[] = [];
    console.error = (...values: unknown[]) => output.push(values.join(" "));
    try {
        expect(runCreate([destination, "--template", "missing"])).toBe(1);
    } finally {
        console.error = error;
    }
    expect(existsSync(destination)).toBe(false);
    expect(output.join("\n")).toContain("minimal");
    expect(output.join("\n")).toContain("first-person");
    expect(output.join("\n")).toContain(
        "custom-loading-screen — Override the default loading screen: the pre-import bootstrap and the runtime screen.",
    );
    expect(output.join("\n")).toContain(
        "embedded-scene — Show a Shallot scene inside a host page, its loading contained to the frame.",
    );
    expect(output.join("\n")).toMatch(/first-person.*—.*Climb/i);
});
