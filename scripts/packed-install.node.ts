import { afterAll, beforeAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { normalizePath } from "../src/project/paths";
import {
    assertCopiedNodeCase,
    assertCreatedExample,
    createExample,
    example,
    examples,
    packedInstall,
    projectFiles,
    ROOT,
    removeExample,
    run,
} from "./packed-install-fixture";

setDefaultTimeout(TIMEOUT.node);

// The package's exclusions, each also matching the contents of a directory it names.
const unpacked = (
    JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")) as { files: string[] }
).files
    .filter((pattern) => pattern.startsWith("!"))
    .flatMap((pattern) => [new Bun.Glob(pattern.slice(1)), new Bun.Glob(`${pattern.slice(1)}/**`)]);

// One copied example Node file proves a created project's Node tier: the default renderer, physics kernel and
// GPU opener run from the installed package under the copy's preload. Every example's Node files run in place.
const COPIED_NODE_SUBJECT = { name: "first-person", subject: "./tests/ascent.node.ts" };

// Each exported plugin's initialize and warm, which no example composition runs together, from the installed
// package under the same copy's preload.
const PACKED_PLUGINS_SUBJECT = "./tests/packed-plugins.node.ts";
const PACKED_PLUGINS_SOURCE = `import { expect, setDefaultTimeout, test } from "bun:test";
import * as Shallot from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";

setDefaultTimeout(TIMEOUT.node);

test("every exported plugin initializes, warms and steps from the installed package", async () => {
    const plugins = Object.entries(Shallot)
        .filter(([name, value]) => name.endsWith("Plugin") && typeof value === "object" && value !== null)
        .map(([, value]) => value as Shallot.Plugin);
    expect(plugins.map((plugin) => plugin.name)).toContain("Fog");
    const app = await openGpuApp({ plugins });
    try {
        app.world.step(Shallot.Time.FIXED_DT);
    } finally {
        app.dispose();
    }
});
`;

test("the packed CLI exposes the public package and its installed examples", () => {
    const { scratch, cliProject, rootPackage, installedPackage, cliInstall } = packedInstall();
    writeFileSync(
        join(cliProject, "fresh.test.ts"),
        `import { expect, test } from "bun:test";
import * as Rendering from "@dylanebert/shallot/rendering";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

test("the packed engine exposes no harness source or capture helpers", () => {
    expect(existsSync(resolve(import.meta.dir, "node_modules/@dylanebert/shallot/src/harness"))).toBe(false);
    expect("captureArtifact" in Rendering).toBe(false);
    expect("captureIdentityLabel" in Rendering).toBe(false);
    expect("captureIdentityMatches" in Rendering).toBe(false);
    expect("assertCaptureGeometry" in Rendering).toBe(false);
});

test("the packed TextPlugin font ships with its license and resolves from the package", async () => {
    const packageRoot = resolve(import.meta.dir, "node_modules/@dylanebert/shallot");
    const fontPath = resolve(packageRoot, "src/extras/text/Inter-Regular.ttf");
    const licensePath = resolve(packageRoot, "src/extras/text/Inter-OFL.txt");
    expect(existsSync(fontPath)).toBe(true);
    expect(readFileSync(licensePath, "utf8")).toContain("SIL Open Font License, Version 1.1");
    const { DEFAULT_FONT, loadFont } = await import("./node_modules/@dylanebert/shallot/src/extras/text/font.ts");
    expect(new URL(DEFAULT_FONT).protocol).toBe("file:");
    expect((await loadFont(DEFAULT_FONT)).advance("A")).toBeGreaterThan(0);
    const fontModule = resolve(packageRoot, "src/extras/text/font.ts");
    const node = Bun.spawnSync(
        [
            "node",
            "--input-type=module",
            "-e",
            'import { readFileSync } from "node:fs"; import { pathToFileURL } from "node:url"; const font = readFileSync(new URL("./Inter-Regular.ttf", pathToFileURL(' +
                JSON.stringify(fontModule) +
                '))); if (font.readUInt32BE(0) !== 0x00010000) throw new Error("packed font did not resolve in Node");',
        ],
        { cwd: import.meta.dir, stdout: "pipe", stderr: "pipe" },
    );
    expect(node.exitCode, node.stderr.toString()).toBe(0);
});

test("the packed Vite entry imports in Node and exposes only shallot", () => {
    const node = Bun.spawnSync(
        [
            "node",
            "--input-type=module",
            "-e",
            'const names = Object.keys(await import("@dylanebert/shallot/vite")).sort(); if (names.length !== 1 || names[0] !== "shallot") throw new Error("unexpected Vite exports: " + names.join(", "));',
        ],
        { cwd: import.meta.dir, stdout: "pipe", stderr: "pipe" },
    );
    expect(node.exitCode, node.stderr.toString()).toBe(0);
});
`,
    );

    const installedManifest = JSON.parse(
        readFileSync(join(installedPackage, "package.json"), "utf8"),
    );
    expect(installedManifest.exports["./brand"]).toBeUndefined();
    expect(existsSync(join(installedPackage, "assets/branding"))).toBe(false);
    expect(installedManifest.scripts.prepare).toBe(rootPackage.scripts.prepare);
    expect(cliInstall).not.toContain("build-tooling: generated dist/");

    const publicExports = Object.keys(installedManifest.exports).filter(
        (entry) => entry !== "./tsconfig.json",
    );
    expect(publicExports.length).toBeGreaterThan(0);
    const publicImportSource = publicExports
        .map((entry, index) => {
            const specifier =
                entry === "."
                    ? installedManifest.name
                    : `${installedManifest.name}/${entry.slice(2)}`;
            return `import * as publicExport${index} from ${JSON.stringify(specifier)};`;
        })
        .join("\n");
    writeFileSync(join(cliProject, "public-exports.ts"), `${publicImportSource}\n`);
    const typecheckConfig = join(cliProject, "tsconfig.public-exports.json");
    writeFileSync(
        typecheckConfig,
        JSON.stringify(
            {
                extends: "@dylanebert/shallot/tsconfig.json",
                include: ["public-exports.ts"],
            },
            null,
            2,
        ),
    );
    run(
        ["bun", "check", "-p", typecheckConfig],
        cliProject,
        "checking every public TypeScript export",
    );

    const packedProjectTests = run(["bun", "test"], cliProject, "testing packed project exports");
    expect(packedProjectTests).toContain(
        "the packed engine exposes no harness source or capture helpers",
    );
    expect(packedProjectTests).toContain(
        "the packed Vite entry imports in Node and exposes only shallot",
    );
    expect(packedProjectTests).toContain("3 pass");

    expect(existsSync(join(scratch, "package.json"))).toBe(false);
});

// Each example owns its setup so a selected subject never depends on an earlier test.
for (const name of examples) {
    describe(name, () => {
        beforeAll(() => createExample(name), TIMEOUT.startup);
        afterAll(() => removeExample(name), TIMEOUT.startup);

        test(`${name} creates and installs from the packed CLI`, () => {
            assertCreatedExample(name);
            if (name === "physics") {
                expect(
                    projectFiles(example(name).project).filter((path) =>
                        path.endsWith(".gold.txt"),
                    ),
                ).toEqual([]);
                expect(
                    projectFiles(example(name).project).filter((path) =>
                        path.endsWith(".parity.node.ts"),
                    ),
                ).toEqual([]);
            }
        });

        test(`${name} type-checks in its copy`, () => {
            const { project } = example(name);
            run(["bun", "check"], project, `bun check ${name}`);
        });

        test(`${name} builds in its copy`, () => {
            run(["bun", "run", "build"], example(name).project, `bun run build ${name}`);
        });

        test(`${name} passes its cheap tests in its copy`, () => {
            const testOutput = run(["bun", "test"], example(name).project, `bun test ${name}`);
            expect(testOutput).toContain(
                "(pass) the Bun preload transforms engine TGSL and keeps it callable on the CPU",
            );
            expect(testOutput).toContain("0 fail");
        });

        const nodeSubjects = projectFiles(join(ROOT, "examples", name))
            .map((path) => normalizePath(relative(ROOT, path)))
            .filter((path) => path.endsWith(".node.ts"))
            .filter((path) => !unpacked.some((glob) => glob.match(path)))
            .map((path) => `./${path.slice(`examples/${name}/`.length)}`)
            .sort();
        if (nodeSubjects.length)
            test(`${name} copies every shipped Node file`, () => {
                const { project } = example(name);
                expect(
                    projectFiles(project)
                        .filter((path) => path.endsWith(".node.ts"))
                        .map((path) => `./${normalizePath(relative(project, path))}`)
                        .sort(),
                ).toEqual(nodeSubjects);
            });
        if (name === COPIED_NODE_SUBJECT.name) {
            test(`${name} passes ${COPIED_NODE_SUBJECT.subject} in its copy`, () => {
                assertCopiedNodeCase(example(name).project, name, COPIED_NODE_SUBJECT.subject);
            });
            test(`${name}'s copy initializes and warms every exported plugin`, () => {
                const { project } = example(name);
                writeFileSync(join(project, PACKED_PLUGINS_SUBJECT), PACKED_PLUGINS_SOURCE);
                expect(assertCopiedNodeCase(project, name, PACKED_PLUGINS_SUBJECT)).toContain(
                    "1 pass",
                );
            });
        }

        test(`${name} refuses its tests without the TGSL preload`, () => {
            const noPreload = Bun.spawnSync(["bun", "--config=no-preload.toml", "test"], {
                cwd: example(name).project,
                stdout: "pipe",
                stderr: "pipe",
            });
            const noPreloadOutput = `${noPreload.stdout.toString()}${noPreload.stderr.toString()}`;
            expect(noPreload.exitCode).not.toBe(0);
            expect(noPreloadOutput).toContain("TGSL metadata is missing");
            expect(noPreloadOutput).toContain("unplugin-typegpu/bun");
        });
    });
}
