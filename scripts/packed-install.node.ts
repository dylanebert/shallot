import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";

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
import { basename, dirname, join, relative, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

type RootPackage = {
    version: string;
    devDependencies: Record<string, string>;
    peerDependencies: Record<string, string>;
    scripts: { prepare: string };
};

type PackedInstall = {
    scratch: string;
    cliProject: string;
    rootPackage: RootPackage;
    packageVersion: string;
    tarball: string;
    installedPackage: string;
    cliInstall: string;
};

let install: PackedInstall | undefined;

function run(command: string[], cwd: string, label: string, argv0?: string): string {
    const result = Bun.spawnSync(command, { cwd, stdout: "pipe", stderr: "pipe", argv0 });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

function packedInstall(): PackedInstall {
    if (!install) throw new Error("packed CLI setup did not complete");
    return install;
}

function projectFiles(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) {
            if (entry.name === "node_modules") return [];
            return projectFiles(path);
        }
        return entry.isFile() ? [path] : [];
    });
}

beforeAll(() => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-packed-install-"));
    try {
        const cliProject = join(scratch, "cli");
        const rootPackage = JSON.parse(
            readFileSync(join(ROOT, "package.json"), "utf8"),
        ) as RootPackage;
        const packageVersion = rootPackage.version;
        const tarballName = `dylanebert-shallot-${packageVersion}.tgz`;
        const tarball = join(scratch, tarballName);
        mkdirSync(cliProject);
        run(
            ["bun", "pm", "pack", "--ignore-scripts", "--destination", scratch, "--quiet"],
            ROOT,
            "packing Shallot",
        );
        if (!existsSync(tarball)) throw new Error(`packed tarball is missing: ${tarball}`);
        writeFileSync(
            join(cliProject, "package.json"),
            JSON.stringify(
                {
                    name: "packed-install-cli",
                    private: true,
                    type: "module",
                    dependencies: { "@dylanebert/shallot": `file:../${tarballName}` },
                    devDependencies: {
                        "@types/bun": rootPackage.devDependencies["@types/bun"],
                        playwright: rootPackage.devDependencies.playwright,
                        webgpu: rootPackage.devDependencies.webgpu,
                    },
                },
                null,
                2,
            ),
        );
        const cliInstall = run(
            ["bun", "install", "--linker=isolated", "--no-progress"],
            cliProject,
            "installing packed CLI project",
        );
        install = {
            scratch,
            cliProject,
            rootPackage,
            packageVersion,
            tarball,
            installedPackage: join(cliProject, "node_modules/@dylanebert/shallot"),
            cliInstall,
        };
    } catch (error) {
        rmSync(scratch, { recursive: true, force: true });
        throw error;
    }
}, CEILING.startup);

afterAll(() => {
    if (install) rmSync(install.scratch, { recursive: true, force: true });
});

test("the packed CLI exposes the public package and its installed examples", () => {
    const { scratch, cliProject, rootPackage, installedPackage, cliInstall, tarball } =
        packedInstall();
    writeFileSync(
        join(cliProject, "fresh.test.ts"),
        `import { expect, test } from "bun:test";
import * as Rendering from "@dylanebert/shallot/rendering";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const manifest = JSON.parse(readFileSync(resolve(import.meta.dir, "node_modules/@dylanebert/shallot/package.json"), "utf8"));

test("the packed engine exposes no test-support namespace or capture helpers", () => {
    // ./testing is the published test policy; only the retired harness and testing/* namespaces are refused.
    expect(Object.keys(manifest.exports).some((path) => /^\\.\\/(?:harness(?:\\/|$)|testing\\/)/.test(path))).toBe(false);
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
        "the packed engine exposes no test-support namespace or capture helpers",
    );
    expect(packedProjectTests).toContain(
        "the packed Vite entry imports in Node and exposes only shallot",
    );
    expect(packedProjectTests).toContain("3 pass");

    const recipes = readdirSync(join(ROOT, "examples"), { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name)
        .sort();
    expect(recipes).toEqual(["first-person", "loading-screen", "minimal"]);
    expect(existsSync(join(scratch, "package.json"))).toBe(false);

    // By file name from its directory: GNU tar, which Git Bash puts first on Windows, reads `C:` as a host.
    const packed = Bun.spawnSync(["tar", "-tzf", basename(tarball)], {
        cwd: dirname(tarball),
        stdout: "pipe",
        stderr: "pipe",
    });
    expect(packed.exitCode).toBe(0);
    expect(packed.stdout.toString()).toContain("tsconfig.base.json");
    expect(packed.stdout.toString()).toContain("examples/minimal/tsconfig.json");
    expect(packed.stdout.toString()).toContain("examples/minimal/src/welcome/welcome.test.ts");
    expect(packed.stdout.toString()).toContain("examples/first-person/tsconfig.json");
    expect(packed.stdout.toString()).toContain("examples/loading-screen/tsconfig.json");
    expect(packed.stdout.toString()).not.toContain("examples/minimal/package.json");
    expect(packed.stdout.toString()).not.toContain("examples/first-person/package.json");
    expect(packed.stdout.toString()).not.toContain("examples/loading-screen/package.json");
});

function installExample(name: "minimal" | "first-person" | "loading-screen"): void {
    const { scratch, cliProject, rootPackage, packageVersion, tarball, installedPackage } =
        packedInstall();
    const project = join(scratch, name);
    const args = ["bun", "run", join(installedPackage, "bin/shallot.ts"), "create", project];
    if (name !== "minimal") args.push("--template", name);
    run(args, cliProject, `shallot create ${name}`);

    const manifestPath = join(project, "package.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    expect(manifest.dependencies["@dylanebert/shallot"]).toBe(packageVersion);
    expect(manifest.devDependencies["@types/bun"]).toBe(rootPackage.devDependencies["@types/bun"]);
    expect(manifest.devDependencies.playwright).toBe(rootPackage.peerDependencies.playwright);
    expect(manifest.devDependencies.vite).toBe(rootPackage.devDependencies.vite);
    expect(manifest.devDependencies.typegpu).toBeUndefined();
    expect(manifest.devDependencies.webgpu).toBe(
        name === "first-person" || name === "loading-screen"
            ? rootPackage.peerDependencies.webgpu
            : undefined,
    );
    expect(existsSync(join(scratch, "package.json"))).toBe(false);

    const packedRange = `file:${relative(project, tarball).replaceAll("\\", "/")}`;
    manifest.dependencies["@dylanebert/shallot"] = packedRange;
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 4)}\n`);
    writeFileSync(join(project, "no-preload.toml"), "[test]\n");
    writeFileSync(
        join(project, "src/packed-tgsl.test.ts"),
        `import { expect, test } from "bun:test";
import { linearToSrgb3 } from "@dylanebert/shallot/rendering";
import { checkTgsl } from "@dylanebert/shallot/runtime";

test("the Bun preload transforms engine TGSL and keeps it callable on the CPU", () => {
    checkTgsl();
    const encoded = linearToSrgb3([0.5, 0.5, 0.5] as Parameters<typeof linearToSrgb3>[0]);
    expect(encoded.x).toBeCloseTo(0.7353569, 5);
    expect(encoded.y).toBeCloseTo(0.7353569, 5);
    expect(encoded.z).toBeCloseTo(0.7353569, 5);
});
`,
    );

    const installOutput = run(["bun", "install"], project, `bun install ${name}`);
    expect(installOutput).not.toContain("build-tooling: generated dist/");
    run(["bun", "check"], project, `bun check ${name}`);
    run(["bun", "run", "build"], project, `bun run build ${name}`);
    const testOutput = run(["bun", "test"], project, `bun test ${name}`);
    expect(testOutput).toContain(
        "(pass) the Bun preload transforms engine TGSL and keeps it callable on the CPU",
    );
    expect(testOutput).toContain("0 fail");
    const copiedFiles = projectFiles(project);
    for (const file of copiedFiles.filter((path) => path.endsWith(".node.ts")).sort()) {
        const subject = `./${relative(project, file).replaceAll("\\", "/")}`;
        // This process's own Bun: on Windows CI it is the copy named node.exe that Dawn requires, and
        // argv0 keeps that copy in Bun's mode rather than Node's, as CI's Node tier spawns it.
        const output = run(
            [process.execPath, "test", subject],
            project,
            `bun test ${name} ${subject}`,
            "bun",
        );
        expect(output).toContain("0 fail");
    }
    const noPreload = Bun.spawnSync(["bun", "--config=no-preload.toml", "test"], {
        cwd: project,
        stdout: "pipe",
        stderr: "pipe",
    });
    const noPreloadOutput = `${noPreload.stdout.toString()}${noPreload.stderr.toString()}`;
    expect(noPreload.exitCode).not.toBe(0);
    expect(noPreloadOutput).toContain("TGSL metadata is missing");
    expect(noPreloadOutput).toContain("unplugin-typegpu/bun");
}

test("minimal installs from the packed CLI and passes its project loop", () => {
    installExample("minimal");
});

test("first-person installs from the packed CLI and passes its project loop", () => {
    installExample("first-person");
});

test("loading-screen installs from the packed CLI and passes its project loop", () => {
    installExample("loading-screen");
});
