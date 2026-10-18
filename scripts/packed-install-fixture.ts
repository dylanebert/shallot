import { afterAll, beforeAll, expect } from "bun:test";
import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    realpathSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { normalizePath } from "../src/project/paths";
import { packedInstall as runPackedInstall } from "./packed-install-spawn";
import { removeTestDirectory } from "./test-directory";

export const ROOT = resolve(import.meta.dir, "..");
export const examples = readdirSync(join(ROOT, "examples"), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

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

export function run(command: string[], cwd: string, label: string, argv0?: string): string {
    if (command[0] === "bun" && command[1] === "install") {
        return runPackedInstall(command.slice(1), cwd, label);
    }
    const result = Bun.spawnSync(command, { cwd, stdout: "pipe", stderr: "pipe", argv0 });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

export function packedInstall(): PackedInstall {
    if (!install) throw new Error("packed CLI setup did not complete");
    return install;
}

export function projectFiles(directory: string): string[] {
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
    const scratch = realpathSync.native(mkdtempSync(join(tmpdir(), "shallot-packed-install-")));
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
        removeTestDirectory(scratch);
        throw error;
    }
}, TIMEOUT.startup);

afterAll(() => {
    if (install) removeTestDirectory(install.scratch);
}, TIMEOUT.startup);

const created = new Map<string, { manifest: Record<string, any>; installOutput: string }>();

/** The copy `create` made for `name`; each later step of its loop refuses when that step did not complete. */
export function example(name: string): { project: string; manifest: Record<string, any> } {
    if (!created.has(name)) throw new Error(`${name} was not created from the packed CLI`);
    const project = join(packedInstall().scratch, name);
    return { project, manifest: JSON.parse(readFileSync(join(project, "package.json"), "utf8")) };
}

export function createExample(name: string): void {
    const { scratch, cliProject, tarball, installedPackage } = packedInstall();
    const project = join(scratch, name);
    const args = ["bun", "run", join(installedPackage, "bin/shallot.ts"), "create", project];
    if (name !== "minimal") args.push("--template", name);
    run(args, cliProject, `shallot create ${name}`);

    const manifestPath = join(project, "package.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    const createdManifest = structuredClone(manifest);
    const packedRange = `file:${normalizePath(relative(project, tarball))}`;
    manifest.dependencies["@dylanebert/shallot"] = packedRange;
    // Svelte checking resolves its supported TypeScript online in the manual oracle.
    delete manifest.devDependencies["svelte-check"];
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
    created.set(name, { manifest: createdManifest, installOutput });
}

export function assertCreatedExample(name: string): void {
    example(name);
    const { manifest, installOutput } = created.get(name)!;
    const { scratch, rootPackage, packageVersion } = packedInstall();
    expect(manifest.dependencies["@dylanebert/shallot"]).toBe(packageVersion);
    expect(manifest.devDependencies["@types/bun"]).toBe(rootPackage.devDependencies["@types/bun"]);
    expect(manifest.devDependencies.playwright).toBe(rootPackage.peerDependencies.playwright);
    expect(manifest.devDependencies.vite).toBe(rootPackage.devDependencies.vite);
    for (const dependency of ["typegpu", "webgpu"])
        if (dependency in manifest.devDependencies)
            expect(manifest.devDependencies[dependency]).toBe(
                rootPackage.peerDependencies[dependency],
            );
    for (const dependency of ["svelte", "@sveltejs/vite-plugin-svelte", "svelte-check"])
        if (dependency in manifest.devDependencies)
            expect(manifest.devDependencies[dependency]).toBe(
                rootPackage.devDependencies[dependency],
            );
    expect(existsSync(join(scratch, "package.json"))).toBe(false);
    expect(installOutput).not.toContain("build-tooling: generated dist/");
}

export const manualParticleCases = [
    [
        "./src/fog.node.ts",
        "every converged shaft-edge profile agrees with the equal 16-phase linear average",
    ],
    ["./src/fog.node.ts", "a still shaft-edge patch stays calm across all 16 consecutive phases"],
    [
        "./src/motes.node.ts",
        "motes near a moving player's fixed path end farther from the path than with the player still",
    ],
] as const;

export function assertCopiedNodeCase(
    project: string,
    name: string,
    subject: string,
    caseName?: string,
): void {
    // This process's own Bun: on Windows CI it is the copy named node.exe that Dawn requires, and
    // argv0 keeps that copy in Bun's mode rather than Node's, as CI's Node tier spawns it.
    const output = run(
        [
            process.execPath,
            "test",
            subject,
            ...(caseName ? ["-t", `^${caseName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`] : []),
        ],
        project,
        `bun test ${name} ${subject}`,
        "bun",
    );
    expect(output).toContain("0 fail");
    if (caseName) expect(output).toContain("1 pass");
}
