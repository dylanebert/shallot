import {
    cpSync,
    existsSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    statSync,
    writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

const PACKAGE_ROOT = resolve(import.meta.dir, "../..");
const ENGINE = "@dylanebert/shallot";
/** A project's install and build output: ignored by its Git, never copied from a template, never packed. */
export const PROJECT_OUTPUT = ["node_modules", "dist", "build", ".artifacts", "test-results"];
const PROJECT_GITIGNORE = PROJECT_OUTPUT.map((directory) => `${directory}/\n`).join("");
const PROJECT_PRELOAD = `import { shallot } from "@dylanebert/shallot/bun";
import { plugin } from "bun";

plugin(shallot({ root: import.meta.dir }));
`;
const CREATE_USAGE = `
  shallot create <name> [--template <example>]

  Create a project in the outer folder. The default template is minimal.

  Outer folder, without Shallot installed
    bun create shallot <name> [--template <example>]

  Inside a project where Shallot is installed
    bunx shallot create <name> [--template <example>]

  Options
    --template <example>  Copy an installed example
    -h, --help            Show this help
`;

interface Env {
    examplesDir: string;
    packageRoot: string;
}

interface Template {
    name: string;
    description?: string;
}

function env(): Env {
    return { examplesDir: resolve(PACKAGE_ROOT, "examples"), packageRoot: PACKAGE_ROOT };
}

function templateEntries(examplesDir: string): Template[] {
    if (!existsSync(examplesDir)) return [];
    return readdirSync(examplesDir, { withFileTypes: true })
        .filter(
            (entry) =>
                entry.isDirectory() && existsSync(resolve(examplesDir, entry.name, "index.html")),
        )
        .map((entry) => {
            const page = readFileSync(resolve(examplesDir, entry.name, "index.html"), "utf8");
            const meta = page.match(/<meta\b[^>]*\bname=["']description["'][^>]*>/i)?.[0];
            return {
                name: entry.name,
                description: meta?.match(/\bcontent=(["'])(.*?)\1/i)?.[2],
            };
        })
        .sort((a, b) => a.name.localeCompare(b.name));
}

function occupied(destination: string): boolean {
    if (!existsSync(destination)) return false;
    const stat = statSync(destination);
    return stat.isDirectory() ? readdirSync(destination).length > 0 : true;
}

/** The bare specifiers a template's sources import, `node:` and `bun:` builtins aside. */
function bareImports(root: string): Set<string> {
    const importPattern =
        /(?:\bfrom\s*|\bimport\s*(?:\(\s*)?|\brequire\s*\(\s*)["']([^"'./][^"']*)["']/g;
    const specifiers = new Set<string>();
    const pending = [root];
    while (pending.length > 0) {
        const dir = pending.pop()!;
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
            if (entry.name === "node_modules") continue;
            const path = join(dir, entry.name);
            if (entry.isDirectory()) pending.push(path);
            else if (entry.isFile() && /\.(?:[cm]?[jt]sx?|html|svelte)$/.test(entry.name)) {
                for (const match of readFileSync(path, "utf8").matchAll(importPattern))
                    if (!/^(?:node|bun):/.test(match[1]!)) specifiers.add(match[1]!);
            }
        }
    }
    return specifiers;
}

function packageName(specifier: string): string {
    const parts = specifier.split("/");
    return specifier.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0]!;
}

export function packageManifest(
    destination: string,
    templateDir: string,
    packageRoot: string,
): string {
    const engine = JSON.parse(readFileSync(resolve(packageRoot, "package.json"), "utf8"));
    const devDependencies: Record<string, string> = {
        "@types/bun": engine.devDependencies["@types/bun"],
        playwright: engine.peerDependencies.playwright,
        vite: engine.devDependencies.vite,
    };
    const imports = bareImports(templateDir);
    if (imports.has(`${ENGINE}/webgpu`)) devDependencies.webgpu = engine.peerDependencies.webgpu;
    // A template's own packages are those the engine is proved against, at the engine's range; the
    // import gate refuses a package the engine does not declare.
    for (const name of new Set([...imports].map(packageName))) {
        const range = engine.peerDependencies[name] ?? engine.devDependencies[name];
        if (name !== ENGINE && range) devDependencies[name] ??= range;
    }

    return `${JSON.stringify(
        {
            name: basename(destination),
            version: "0.0.0",
            private: true,
            type: "module",
            scripts: {
                dev: "vite --host 127.0.0.1",
                build: "vite build",
                preview: "vite preview --host 127.0.0.1",
            },
            dependencies: { [ENGINE]: engine.version },
            devDependencies,
        },
        null,
        4,
    )}\n`;
}

function listFiles(root: string, prefix = ""): string[] {
    return readdirSync(join(root, prefix), { withFileTypes: true }).flatMap((entry) => {
        const path = prefix ? join(prefix, entry.name) : entry.name;
        return entry.isDirectory() ? listFiles(root, path) : [path];
    });
}

function tierFiles(projectDir: string): Record<string, boolean> {
    const files = listFiles(projectDir);
    return {
        cheap: files.some((file) => /\.test\.ts$/.test(file) && !/\.gpu\.test\.ts$/.test(file)),
        gpu: files.some((file) => /\.gpu\.test\.ts$/.test(file)),
        node: files.some((file) => /\.node\.ts$/.test(file)),
        browser: files.some((file) => /\.e2e\.ts$/.test(file)),
    };
}

function setup(osList: string): string {
    const version = JSON.parse(
        readFileSync(resolve(PACKAGE_ROOT, "package.json"), "utf8"),
    ).packageManager.split("@")[1];
    return `    runs-on: \${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [${osList}]
    defaults:
      run:
        shell: bash
    steps:
      - uses: actions/checkout@v7
      - uses: oven-sh/setup-bun@v2
        with:
          bun-version: "${version}"
`;
}

function mesaSetup(): string {
    return `      - name: Install Mesa Vulkan driver
        if: matrix.os == 'ubuntu-latest'
        run: |
          sudo apt-get update
          sudo apt-get install -y mesa-vulkan-drivers
`;
}

function windowsGpuSetup(): string {
    return `      - name: Run Bun as node.exe
        if: matrix.os == 'windows-latest'
        shell: pwsh
        run: |
          $hostDir = Join-Path $env:RUNNER_TEMP 'bun-node'
          $realNodeDir = Split-Path (Get-Command node -ErrorAction Stop).Source -Parent
          Add-Content -Path $env:GITHUB_ENV -Value "SHALLOT_REAL_NODE_DIR=$realNodeDir"
          New-Item -ItemType Directory -Force -Path $hostDir | Out-Null
          Copy-Item (Get-Command bun).Source (Join-Path $hostDir 'node.exe')
          Add-Content -Path $env:GITHUB_PATH -Value $hostDir
      - name: Identify WARP adapter
        if: matrix.os == 'windows-latest'
        run: |
          node -e "import('webgpu').then(({ create, globals }) => { Object.assign(globalThis, globals); return create([]).requestAdapter(); }).then((adapter) => { if (!adapter) throw new Error('WARP adapter unavailable'); const info = adapter.info; const identity = [info.vendor, info.architecture, info.device, info.description].join(' '); if (!/warp|basic render/i.test(identity)) throw new Error('Expected WARP, got ' + identity); console.info('WARP adapter:', identity); })"
`;
}

function tierJob(tier: string): string {
    const osList = '"macos-15", "ubuntu-latest", "windows-latest"';
    let job = `  ${tier}:\n${setup(osList)}`;
    if (tier === "gpu" || tier === "node") {
        const nodeVersion = "26";
        job += mesaSetup();
        job += `      - uses: actions/setup-node@v6
        with:
          node-version: "${nodeVersion}"
      - run: bun install
`;
        job += windowsGpuSetup();
        const suffix = tier === "gpu" ? "gpu.test.ts" : "node.ts";
        const command = `bun test $(find ./src ./tests -name '*.${suffix}' -print | sort)`;
        job += `      - name: ${tier} tier
        if: matrix.os != 'windows-latest'
        run: ${command}
`;
        const argv = tier === "gpu" ? "'*.gpu.test.ts'" : "'*.node.ts'";
        job += `      - name: ${tier} tier on WARP
        if: matrix.os == 'windows-latest'
        run: node -e "const child = Bun.spawn([process.execPath, 'test', ...process.argv.slice(1)], { argv0: 'bun.exe', env: { ...process.env, PATH: process.env.SHALLOT_REAL_NODE_DIR + ';' + process.env.PATH }, stdin: 'inherit', stdout: 'inherit', stderr: 'inherit' }); child.exited.then((code) => process.exit(code));" -- $(find ./src ./tests -name ${argv} -print | sort)
`;
        return job;
    }

    if (tier === "browser") job += mesaSetup();
    job += `      - run: bun install
`;
    if (tier === "cheap")
        job += `      - name: cheap tier
        run: bun test --path-ignore-patterns '**/*.gpu.test.ts'
`;
    else if (tier === "browser")
        job += `      - name: Install Playwright Chromium
        run: bunx playwright install chromium
      - name: browser tier
        run: bunx playwright test
`;
    return job;
}

function testWorkflow(projectDir: string): string {
    const tiers = tierFiles(projectDir);
    const jobs = Object.keys(tiers)
        .filter((tier) => tiers[tier])
        .map((tier) => tierJob(tier))
        .join("\n");
    return `name: test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
${jobs}`;
}

function projectReadme(name: string): string {
    return `# ${name}

A Shallot game project.

## Run

\`bun install\`, then \`bun run dev\`. Build with \`bun run build\` and serve the build with \`bun run preview\`.

## Hosting

\`bun run dev\`, \`bun run preview\` and the native shell serve the page with cross-origin isolation. Elsewhere, Physics steps the single-threaded artifact after one log naming the missing headers.
`;
}

function projectAgents(): string {
    return `# Project loop

\`bun install\`, then \`bun run dev\`; \`bun test\`; \`bun test ./<file>\`; \`bunx playwright test\`; \`bun run build\`; \`bun run preview\`.

\`bunx shallot --skill\`
`;
}

/** Create a project from an installed example; project metadata has one producer here, not in templates. */
export function runCreate(args: string[], e: Env = env()): number {
    if (args.some((arg) => arg === "--help" || arg === "-h")) {
        console.log(CREATE_USAGE);
        return 0;
    }

    let name: string | undefined;
    let template = "minimal";
    for (let index = 0; index < args.length; index++) {
        const arg = args[index]!;
        if (arg === "--template") {
            const value = args[index + 1];
            if (!value || value.startsWith("-")) {
                console.error("--template requires an example name");
                return 1;
            }
            template = value;
            index++;
        } else if (arg.startsWith("--template=")) {
            template = arg.slice("--template=".length);
            if (!template) {
                console.error("--template requires an example name");
                return 1;
            }
        } else if (arg.startsWith("-")) {
            console.error(`unknown option: ${arg}`);
            console.error("See `shallot create --help` for available options.");
            return 1;
        } else if (name === undefined) name = arg;
        else {
            console.error(`unexpected argument: ${arg}`);
            console.error("See `shallot create --help` for usage.");
            return 1;
        }
    }
    if (!name) {
        console.error("create requires a project name");
        console.error("See `shallot create --help` for usage.");
        return 1;
    }

    const templates = templateEntries(e.examplesDir);
    if (!templates.length) {
        console.error(`no installed examples found in ${e.examplesDir}`);
        return 1;
    }
    const selected = templates.find((entry) => entry.name === template);
    if (!selected) {
        console.error(`unknown template: ${template}\n\nInstalled templates:`);
        for (const entry of templates)
            console.error(`  ${entry.name}${entry.description ? ` — ${entry.description}` : ""}`);
        return 1;
    }

    const destination = resolve(name);
    if (occupied(destination)) {
        console.error(`refusing to create in ${destination}: directory is not empty`);
        return 1;
    }

    const templateDir = resolve(e.examplesDir, selected.name);
    cpSync(templateDir, destination, {
        recursive: true,
        filter: (src) => !PROJECT_OUTPUT.includes(basename(src)),
    });
    const write = (path: string, contents: string) => {
        const absolute = resolve(destination, path);
        mkdirSync(dirname(absolute), { recursive: true });
        writeFileSync(absolute, contents);
    };
    write("package.json", packageManifest(destination, templateDir, e.packageRoot));
    write(".gitignore", PROJECT_GITIGNORE);
    write("bunfig.toml", '[test]\npreload = ["./tests/preload.ts"]\n');
    write("tests/preload.ts", PROJECT_PRELOAD);
    write("README.md", projectReadme(basename(destination)));
    write("AGENTS.md", projectAgents());
    write(".github/workflows/test.yml", testWorkflow(destination));

    console.log(`created ${destination} from ${selected.name}`);
    console.log(`  cd ${name} && bun install && bun run dev`);
    return 0;
}
