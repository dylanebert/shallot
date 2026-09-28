import { expect, test } from "bun:test";
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

const ROOT = resolve(import.meta.dir, "..");
const ENGINE = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
const TYPEGPU_RANGE = ENGINE.peerDependencies.typegpu as string;

function run(command: string[], cwd: string, label: string): string {
    const result = Bun.spawnSync(command, { cwd, stdout: "pipe", stderr: "pipe" });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

function createLinkedFixture(scratch: string): { nodeModules: string; tests: string } {
    const nodeModules = join(scratch, "node_modules");
    const typegpuSource = join(ROOT, "node_modules/typegpu");
    const typegpuManifest = JSON.parse(readFileSync(join(typegpuSource, "package.json"), "utf8"));
    const tests = join(scratch, "tests");

    mkdirSync(join(nodeModules, "@dylanebert"), { recursive: true });
    mkdirSync(join(nodeModules, "typegpu"), { recursive: true });
    mkdirSync(tests, { recursive: true });
    // Match bun link's node_modules symlink without touching the global link registry.
    symlinkSync(ROOT, join(nodeModules, "@dylanebert/shallot"), "dir");
    cpSync(typegpuSource, join(nodeModules, "typegpu"), { recursive: true });
    for (const name of Object.keys(typegpuManifest.dependencies ?? {})) {
        const target = resolve(ROOT, "node_modules", ...name.split("/"));
        if (!existsSync(target)) throw new Error(`missing TypeGPU dependency ${name}`);
        const link = resolve(nodeModules, ...name.split("/"));
        mkdirSync(dirname(link), { recursive: true });
        if (!existsSync(link)) symlinkSync(target, link, "dir");
    }

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
    writeFileSync(
        join(tests, "preload.ts"),
        'import { plugin } from "bun";\nimport { shallot } from "@dylanebert/shallot/bun";\nplugin(shallot());\n',
    );
    return { nodeModules, tests };
}

test("the Bun preload dedupes linked-engine TypeGPU imports to the consumer peer", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-package-shape-link-"));
    try {
        const { tests } = createLinkedFixture(scratch);
        writeFileSync(
            join(tests, "linked.test.ts"),
            `import { expect, test } from "bun:test";\nimport tgpu from "typegpu";\nimport { checkTgsl } from "@dylanebert/shallot/runtime";\n\ntest("consumer and linked engine share one TypeGPU instance", () => {\n    expect(typeof tgpu.fn).toBe("function");\n    expect(() => checkTgsl()).not.toThrow();\n});\n`,
        );

        const consumerPath = Bun.resolveSync("typegpu", scratch);
        const producerPath = Bun.resolveSync("typegpu", ROOT);
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
        expect(output).toContain("node_modules/typegpu/package.json");
        expect(output).not.toContain("TYPEGPU_PROJECT_COPY_EVALUATED");
        expect(output).not.toContain("TEST_MODULE_EVALUATED");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
});

test("a staged overlay and bun install restore a published project pin", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-package-shape-stage-"));
    const project = join(scratch, "project");
    const rootManifest = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const tarball = join(scratch, `dylanebert-shallot-${rootManifest.version}.tgz`);

    mkdirSync(project);
    try {
        run(
            ["bun", "pm", "pack", "--ignore-scripts", "--destination", scratch, "--quiet"],
            ROOT,
            "packing staged Shallot",
        );
        writeFileSync(
            join(project, "package.json"),
            `${JSON.stringify(
                {
                    name: "shallot-package-shape-staged-fixture",
                    private: true,
                    type: "module",
                    dependencies: { "@dylanebert/shallot": "^0.9.5" },
                    devDependencies: { typegpu: TYPEGPU_RANGE },
                },
                null,
                2,
            )}\n`,
        );

        run(["bun", "install", "--no-progress"], project, "installing the published pin");
        const manifestBefore = readFileSync(join(project, "package.json"), "utf8");
        const lockBefore = readFileSync(join(project, "bun.lock"), "utf8");
        const installed = () =>
            JSON.parse(
                readFileSync(
                    join(project, "node_modules/@dylanebert/shallot/package.json"),
                    "utf8",
                ),
            ).version;
        const pinnedVersion = installed();
        expect(Bun.semver.satisfies(pinnedVersion, "^0.9.5")).toBe(true);
        console.log(`staged restore pin: @dylanebert/shallot@${pinnedVersion}`);

        run(["bun", "add", "--no-save", tarball], project, "overlaying the packed Shallot tarball");
        expect(installed()).toBe(rootManifest.version);
        expect(readFileSync(join(project, "package.json"), "utf8")).toBe(manifestBefore);
        expect(readFileSync(join(project, "bun.lock"), "utf8")).toBe(lockBefore);

        run(["bun", "install", "--no-progress"], project, "restoring the published pin");
        expect(installed()).toBe(pinnedVersion);
        expect(readFileSync(join(project, "package.json"), "utf8")).toBe(manifestBefore);
        expect(readFileSync(join(project, "bun.lock"), "utf8")).toBe(lockBefore);
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
