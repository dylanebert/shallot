import { beforeAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { parse } from "@babel/parser";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { normalizePath } from "../src/project/paths";
import {
    assertCopiedNodeCase,
    assertCreatedExample,
    createExample,
    example,
    examples,
    manualParticleCases,
    packedInstall,
    projectFiles,
    ROOT,
    run,
} from "./packed-install-fixture";

setDefaultTimeout(TIMEOUT.node);

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

function nodeTestNames(path: string): string[] {
    const ast = parse(readFileSync(path, "utf8"), {
        sourceType: "module",
        plugins: ["typescript"],
    });
    const topLevel = new Set(
        ast.program.body.flatMap((node) =>
            node.type === "ExpressionStatement" ? [node.expression] : [],
        ),
    );
    const names: string[] = [];
    const visit = (node: any) => {
        if (!node || typeof node !== "object") return;
        if (node.type === "CallExpression") {
            let callee = node.callee;
            while (callee.type === "MemberExpression" || callee.type === "CallExpression")
                callee = callee.object ?? callee.callee;
            if (callee.type === "Identifier" && ["test", "it", "describe"].includes(callee.name)) {
                if (
                    node.callee !== callee ||
                    callee.name !== "test" ||
                    !topLevel.has(node) ||
                    node.arguments[0]?.type !== "StringLiteral"
                )
                    throw new Error(
                        `${path}: per-case packed checks require top-level test calls with literal names`,
                    );
                names.push(node.arguments[0].value);
            }
        }
        for (const value of Object.values(node)) {
            if (Array.isArray(value)) for (const child of value) visit(child);
            else visit(value);
        }
    };
    visit(ast.program);
    if (!names.length || new Set(names).size !== names.length)
        throw new Error(`${path}: per-case packed checks require nonempty, unique test names`);
    return names;
}

// Each example owns its setup so a selected subject never depends on an earlier test.
for (const name of examples) {
    describe(name, () => {
        beforeAll(() => createExample(name), TIMEOUT.startup);

        test(`${name} creates and installs from the packed CLI`, () => {
            assertCreatedExample(name);
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
            .filter((path) => path.endsWith(".node.ts"))
            .map((path) => `./${normalizePath(relative(join(ROOT, "examples", name), path))}`)
            .sort();
        if (nodeSubjects.length)
            test(`${name} copies every Node file`, () => {
                const { project } = example(name);
                expect(
                    projectFiles(project)
                        .filter((path) => path.endsWith(".node.ts"))
                        .map((path) => `./${normalizePath(relative(project, path))}`)
                        .sort(),
                ).toEqual(nodeSubjects);
            });
        for (const subject of nodeSubjects) {
            const cases =
                name === "particles"
                    ? nodeTestNames(join(ROOT, "examples", name, subject))
                    : [undefined];
            for (const caseName of cases) {
                if (
                    name === "particles" &&
                    manualParticleCases.some(
                        ([file, claim]) => file === subject && claim === caseName,
                    )
                )
                    continue;
                test(`${name} passes ${subject}${caseName ? `: ${caseName}` : ""} in its copy`, () => {
                    const { project } = example(name);
                    assertCopiedNodeCase(project, name, subject, caseName);
                });
            }
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
