import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "@dylanebert/shallot/testing";

setDefaultTimeout(CEILING.node);

import {
    cpSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { normalizePath } from "../src/project/paths";

const ROOT = resolve(import.meta.dir, "..");
const ENGINE = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
const TYPEGPU_RANGE = ENGINE.peerDependencies.typegpu as string;

test("Dawn webgpu is an optional peer with the verified development pin", () => {
    expect(ENGINE.peerDependencies.webgpu).toBe("^0.6.2");
    expect(ENGINE.peerDependenciesMeta.webgpu).toEqual({ optional: true });
    expect(ENGINE.devDependencies.webgpu).toBe("^0.6.2");
    expect(ENGINE.exports["./webgpu"]).toBe("./src/engine/runtime/webgpu.ts");
});

test("Vite is a supported plugin peer and a Shallot development dependency", () => {
    expect(ENGINE.peerDependencies.vite).toBe("^8.0.0");
    expect(ENGINE.devDependencies.vite).toBe("^8.3.1");
    expect(ENGINE.dependencies.vite).toBeUndefined();
});

test("the script-only Babel parser is a development dependency", () => {
    expect(ENGINE.dependencies["@babel/parser"]).toBeUndefined();
    expect(ENGINE.devDependencies["@babel/parser"]).toBe("^8.0.6");
});

// c9caa355 dropped an unused Playwright peer; the published ./testing now imports Playwright's types,
// so it is an optional peer, as webgpu is for ./webgpu, and stays the engine's own dev dependency.
test("Playwright is an optional peer for ./testing and the engine's dev dependency", () => {
    expect(ENGINE.peerDependencies.playwright).toBe(ENGINE.devDependencies.playwright);
    expect(ENGINE.peerDependenciesMeta.playwright).toEqual({ optional: true });
    expect(ENGINE.devDependencies.playwright).toBe("^1.63.0");
});

test("Node tooling exports committed JavaScript with generated declaration types", () => {
    expect(ENGINE.exports["./testing"]).toEqual({
        types: "./dist/testing.d.ts",
        default: "./src/project/testing.js",
    });
    expect(ENGINE.exports["./vite"]).toEqual({
        types: "./dist/vite.d.ts",
        default: "./src/project/vite.js",
    });
});

test("prepare is the sole tooling pack hook", () => {
    expect(ENGINE.scripts.prepare).toBe("bun run ./scripts/tooling.ts");
    expect(ENGINE.scripts.prepack).toBeUndefined();
});

test("files excludes no paths under the unlisted scripts directory", () => {
    expect(ENGINE.files).not.toContain("scripts");
    expect(ENGINE.files.filter((entry: string) => entry.startsWith("!scripts/"))).toEqual([]);
});

function run(command: string[], cwd: string, label: string): string {
    const result = Bun.spawnSync(command, { cwd, stdout: "pipe", stderr: "pipe" });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

function installTypegpu(nodeModules: string): void {
    const typegpuSource = join(ROOT, "node_modules/typegpu");
    const typegpuManifest = JSON.parse(readFileSync(join(typegpuSource, "package.json"), "utf8"));
    mkdirSync(join(nodeModules, "typegpu"), { recursive: true });
    cpSync(typegpuSource, join(nodeModules, "typegpu"), { recursive: true });
    for (const name of Object.keys(typegpuManifest.dependencies ?? {})) {
        const target = resolve(ROOT, "node_modules", ...name.split("/"));
        if (!existsSync(target)) throw new Error(`missing TypeGPU dependency ${name}`);
        const link = resolve(nodeModules, ...name.split("/"));
        mkdirSync(dirname(link), { recursive: true });
        if (!existsSync(link)) symlinkSync(target, link, "dir");
    }
}

function writePreload(path: string): void {
    writeFileSync(
        path,
        'import { plugin } from "bun";\nimport { shallot } from "@dylanebert/shallot/bun";\nplugin(shallot({ root: import.meta.dir }));\n',
    );
}

function createLinkedFixture(scratch: string): { nodeModules: string; tests: string } {
    const nodeModules = join(scratch, "node_modules");
    const tests = join(scratch, "tests");

    mkdirSync(join(nodeModules, "@dylanebert"), { recursive: true });
    mkdirSync(tests, { recursive: true });
    // Match bun link's node_modules symlink without touching the global link registry.
    symlinkSync(ROOT, join(nodeModules, "@dylanebert/shallot"), "dir");
    installTypegpu(nodeModules);

    writeFileSync(
        join(scratch, "package.json"),
        `${JSON.stringify(
            {
                name: "shallot-package-shape-link-fixture",
                private: true,
                type: "module",
                dependencies: { "@dylanebert/shallot": "^0.9.5" },
                devDependencies: { typegpu: TYPEGPU_RANGE },
            },
            null,
            2,
        )}\n`,
    );
    writePreload(join(tests, "preload.ts"));
    return { nodeModules, tests };
}

function createWorkspaceFixture(scratch: string): { game: string } {
    const rootNodeModules = join(scratch, "node_modules");
    const game = join(scratch, "packages/game");
    const gameNodeModules = join(game, "node_modules");
    const tests = join(game, "tests");

    mkdirSync(join(rootNodeModules, "@dylanebert"), { recursive: true });
    mkdirSync(tests, { recursive: true });
    symlinkSync(ROOT, join(rootNodeModules, "@dylanebert/shallot"), "dir");
    installTypegpu(gameNodeModules);
    writeFileSync(
        join(scratch, "package.json"),
        `${JSON.stringify(
            {
                name: "shallot-package-shape-workspace",
                private: true,
                type: "module",
                workspaces: ["packages/*"],
            },
            null,
            2,
        )}\n`,
    );
    writeFileSync(
        join(game, "package.json"),
        `${JSON.stringify(
            {
                name: "workspace-game",
                private: true,
                type: "module",
                dependencies: { "@dylanebert/shallot": "^0.9.5" },
                devDependencies: { typegpu: TYPEGPU_RANGE },
            },
            null,
            2,
        )}\n`,
    );
    writeFileSync(
        join(scratch, "bunfig.toml"),
        '[test]\npreload = ["./packages/game/tests/preload.ts"]\n',
    );
    writePreload(join(tests, "preload.ts"));
    return { game };
}

test("the Bun preload dedupes linked-engine TypeGPU imports to the consumer peer", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-package-shape-link-"));
    try {
        const { tests } = createLinkedFixture(scratch);
        writeFileSync(
            join(tests, "linked.test.ts"),
            `import { expect, test } from "bun:test";\nimport tgpu from "typegpu";\nimport { checkTgsl } from "@dylanebert/shallot/runtime";\n\ntest("consumer and linked engine share one TypeGPU instance", () => {\n    expect(typeof tgpu.fn).toBe("function");\n    expect(() => checkTgsl()).not.toThrow();\n});\n`,
        );

        const consumerPath = resolve(scratch, "node_modules/typegpu/index.js");
        const producerPath = resolve(ROOT, "node_modules/typegpu/index.js");
        console.log(`TypeGPU resolution: consumer=${consumerPath}`);
        console.log(`TypeGPU natural linked-engine resolution=${producerPath}`);
        expect(consumerPath).not.toBe(producerPath);

        const output = run(
            ["bun", "test", "--preload=./preload.ts"],
            tests,
            "testing linked TypeGPU identity from a subdirectory",
        );
        expect(output).toContain("consumer and linked engine share one TypeGPU instance");
        expect(output).toContain("1 pass");
        expect(output).not.toContain("Found duplicate TypeGPU version");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
});

test("a Bun test from the workspace root resolves TypeGPU from the package that owns its preload", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-package-shape-workspace-"));
    try {
        const { game } = createWorkspaceFixture(scratch);
        const workspaceTypegpu = join(scratch, "node_modules/typegpu/package.json");
        expect(existsSync(workspaceTypegpu)).toBe(false);
        const gameTypegpu = join(game, "node_modules/typegpu/package.json");
        expect(existsSync(gameTypegpu)).toBe(true);
        expect(normalizePath(gameTypegpu)).toContain(
            "packages/game/node_modules/typegpu/package.json",
        );
        writeFileSync(
            join(game, "tests/mono.test.ts"),
            `import { expect, test } from "bun:test";\nimport tgpu from "typegpu";\nimport { checkTgsl } from "@dylanebert/shallot/runtime";\n\ntest("workspace consumer and linked engine share one TypeGPU instance", () => {\n    expect(typeof tgpu.fn).toBe("function");\n    expect(() => checkTgsl()).not.toThrow();\n});\n`,
        );

        const output = run(
            ["bun", "test", "--config=./bunfig.toml", "packages/game/tests/mono.test.ts"],
            scratch,
            "testing a workspace package from the workspace root",
        );
        expect(output).toContain("workspace consumer and linked engine share one TypeGPU instance");
        expect(output).toContain("1 pass");
        expect(output).not.toContain("Found duplicate TypeGPU version");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
});

test("an out-of-range project TypeGPU fails before either module evaluates", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-package-shape-peer-"));
    try {
        const { nodeModules, tests } = createLinkedFixture(scratch);
        const typegpuManifestPath = join(nodeModules, "typegpu/package.json");
        const typegpuManifest = JSON.parse(readFileSync(typegpuManifestPath, "utf8"));
        typegpuManifest.version = "0.13.0";
        writeFileSync(typegpuManifestPath, `${JSON.stringify(typegpuManifest, null, 2)}\n`);
        writeFileSync(
            join(nodeModules, "typegpu/index.js"),
            'console.log("TYPEGPU_PROJECT_COPY_EVALUATED"); export default {};\n',
        );
        writeFileSync(
            join(tests, "incompatible.test.ts"),
            'import { test } from "bun:test";\ntest("incompatible TypeGPU is not loaded", async () => {\n    console.log("TEST_MODULE_EVALUATED");\n    await import("typegpu");\n});\n',
        );

        const result = Bun.spawnSync(["bun", "test", "--preload=./preload.ts"], {
            cwd: tests,
            stdout: "pipe",
            stderr: "pipe",
        });
        const output = `${result.stdout.toString()}${result.stderr.toString()}`;
        expect(result.exitCode).not.toBe(0);
        expect(output).toContain("project has TypeGPU 0.13.0 at");
        expect(output).toContain(TYPEGPU_RANGE);
        expect(normalizePath(output)).toContain("node_modules/typegpu/package.json");
        expect(output).not.toContain("TYPEGPU_PROJECT_COPY_EVALUATED");
        expect(output).not.toContain("TEST_MODULE_EVALUATED");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
});

test("release creation passes --prerelease only for hyphenated tags", () => {
    const workflow = readFileSync(join(ROOT, ".github/workflows/release.yml"), "utf8");
    const argsBlock = workflow.match(
        /release_args=\(--verify-tag --generate-notes\)\n\s+if \[\[ "\$RELEASE_TAG" == \*-\* \]\]; then\n\s+release_args\+=\(--prerelease\)\n\s+fi/,
    )?.[0];
    if (!argsBlock) throw new Error("release.yml has no recognized prerelease argument logic");
    expect(workflow).toContain(`gh release create "$RELEASE_TAG" "\${release_args[@]}" \\\n`);

    for (const [tag, prerelease] of [
        ["v0.10.0-next.1", true],
        ["v0.10.0", false],
    ] as const) {
        const result = Bun.spawnSync(
            ["bash", "-c", `${argsBlock}\nprintf '%s\\n' "\${release_args[@]}"`],
            {
                env: { ...process.env, RELEASE_TAG: tag },
                stdout: "pipe",
                stderr: "pipe",
            },
        );
        const output = `${result.stdout.toString()}${result.stderr.toString()}`;
        if (result.exitCode !== 0) throw new Error(`release args for ${tag} failed:\n${output}`);
        const args = result.stdout.toString().trim().split("\n");
        expect(args).toEqual(
            prerelease
                ? ["--verify-tag", "--generate-notes", "--prerelease"]
                : ["--verify-tag", "--generate-notes"],
        );
        console.log(`${tag}: ${args.join(" ")}`);
    }
});
