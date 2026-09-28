import { expect, test } from "bun:test";
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
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

function run(command: string[], cwd: string, label: string): string {
    const result = Bun.spawnSync(command, { cwd, stdout: "pipe", stderr: "pipe" });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

test("packed examples install by copy-out and pass their TypeScript, Vite, and Bun checks", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-packed-install-"));
    const project = join(scratch, "project");
    const rootPackage = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const packageVersion = rootPackage.version as string;
    const tarballName = `dylanebert-shallot-${packageVersion}.tgz`;
    const tarball = join(scratch, tarballName);
    mkdirSync(project);
    try {
        run(
            ["bun", "pm", "pack", "--ignore-scripts", "--destination", scratch, "--quiet"],
            ROOT,
            "packing Shallot",
        );
        if (!existsSync(tarball)) throw new Error(`packed tarball is missing: ${tarball}`);

        writeFileSync(
            join(project, "package.json"),
            JSON.stringify(
                {
                    name: "packed-install-project",
                    private: true,
                    type: "module",
                    dependencies: {
                        "@dylanebert/shallot": `file:../${tarballName}`,
                    },
                    devDependencies: {
                        "@types/bun": rootPackage.devDependencies["@types/bun"],
                        playwright: rootPackage.devDependencies.playwright,
                        typegpu: rootPackage.peerDependencies.typegpu,
                        typescript: rootPackage.devDependencies.typescript,
                        vite: rootPackage.dependencies.vite,
                    },
                },
                null,
                2,
            ),
        );
        run(["bun", "install", "--no-progress"], project, "installing packed project");

        const installedPackage = join(project, "node_modules/@dylanebert/shallot");
        const manifest = JSON.parse(readFileSync(join(installedPackage, "package.json"), "utf8"));
        expect(
            Object.keys(manifest.exports).some((path: string) =>
                /^\.\/(?:harness|testing)(?:\/|$)/.test(path),
            ),
        ).toBe(false);
        expect(existsSync(join(installedPackage, "src/harness"))).toBe(false);
        const node = Bun.spawnSync(
            [
                "node",
                "--input-type=module",
                "-e",
                'const names = Object.keys(await import("@dylanebert/shallot/vite")).sort(); if (names.length !== 1 || names[0] !== "shallot") throw new Error("unexpected Vite exports: " + names.join(", "));',
            ],
            { cwd: project, stdout: "pipe", stderr: "pipe" },
        );
        expect(`${node.stdout.toString()}${node.stderr.toString()}`).toBe("");
        expect(node.exitCode).toBe(0);

        const recipes = readdirSync(join(ROOT, "examples"))
            .filter((name) => existsSync(join(ROOT, "examples", name, "shallot.json")))
            .filter(
                (name) =>
                    JSON.parse(readFileSync(join(ROOT, "examples", name, "shallot.json"), "utf8"))
                        .kind === "recipe",
            )
            .sort();
        expect(recipes).toEqual(["first-person", "loading-screen"]);

        for (const name of recipes) {
            const example = join(project, name);
            run(
                ["bun", "run", join(installedPackage, "bin/shallot.ts"), "add", name, name],
                project,
                `shallot add ${name}`,
            );

            const examplePackage = JSON.parse(readFileSync(join(example, "package.json"), "utf8"));
            expect(examplePackage.dependencies["@dylanebert/shallot"]).toBe(packageVersion);
            expect(readFileSync(join(example, "tsconfig.json"), "utf8")).toContain(
                '"@dylanebert/shallot/tsconfig.json"',
            );

            writeFileSync(
                join(example, "bunfig.toml"),
                '[test]\npreload = ["@dylanebert/shallot/bun"]\n',
            );
            writeFileSync(join(example, "no-preload.toml"), "[test]\n");
            writeFileSync(
                join(example, "src/packed-tgsl.test.ts"),
                `import { expect, test } from "bun:test";
import { linearToSrgb } from "@dylanebert/shallot/rendering";
import { checkTgsl } from "@dylanebert/shallot/runtime";
import tgpu from "typegpu";
import * as d from "typegpu/data";

test("the Bun preload transforms engine TGSL and keeps it callable on the CPU", () => {
    checkTgsl();
    expect(tgpu.resolve([linearToSrgb])).toContain("fn linearToSrgb");
    const encoded = linearToSrgb(d.vec3f(0.5, 0.5, 0.5));
    expect(encoded.x).toBeCloseTo(0.7353569, 5);
    expect(encoded.y).toBeCloseTo(0.7353569, 5);
    expect(encoded.z).toBeCloseTo(0.7353569, 5);
});
`,
            );

            run(
                [
                    "node",
                    join(project, "node_modules/typescript/bin/tsc"),
                    "--project",
                    "tsconfig.json",
                ],
                example,
                `tsc ${name}`,
            );
            run(
                ["bun", join(project, "node_modules/vite/bin/vite.js"), "build"],
                example,
                `vite build ${name}`,
            );
            const testOutput = run(["bun", "test"], example, `bun test ${name}`);
            expect(testOutput).toContain("1 pass");

            const noPreload = Bun.spawnSync(["bun", "--config=no-preload.toml", "test"], {
                cwd: example,
                stdout: "pipe",
                stderr: "pipe",
            });
            const noPreloadOutput = `${noPreload.stdout.toString()}${noPreload.stderr.toString()}`;
            expect(noPreload.exitCode).not.toBe(0);
            expect(noPreloadOutput).toContain("TGSL metadata is missing");
            expect(noPreloadOutput).toContain("unplugin-typegpu/bun");
        }

        const packed = Bun.spawnSync(["tar", "-tzf", tarball], {
            cwd: project,
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(packed.exitCode).toBe(0);
        expect(packed.stdout.toString()).toContain("examples/first-person/tsconfig.json");
        expect(packed.stdout.toString()).toContain("examples/loading-screen/tsconfig.json");
        expect(packed.stdout.toString()).not.toContain("examples/first-person/package.json");
        expect(packed.stdout.toString()).not.toContain("examples/loading-screen/package.json");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
}, 300_000);
