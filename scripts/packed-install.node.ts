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
import { join, relative, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

function run(command: string[], cwd: string, label: string): string {
    const result = Bun.spawnSync(command, { cwd, stdout: "pipe", stderr: "pipe" });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

test("packed examples install by copy-out as isolated standalone projects", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-packed-install-"));
    const cliProject = join(scratch, "cli");
    const rootPackage = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const packageVersion = rootPackage.version as string;
    const tarballName = `dylanebert-shallot-${packageVersion}.tgz`;
    const tarball = join(scratch, tarballName);
    mkdirSync(cliProject);
    try {
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
                },
                null,
                2,
            ),
        );
        writeFileSync(
            join(cliProject, "fresh.test.ts"),
            `import { expect, test } from "bun:test";
import { build, type Plugin } from "@dylanebert/shallot/app";
import { f32, sparse, Time } from "@dylanebert/shallot/ecs";
import * as Rendering from "@dylanebert/shallot/rendering";
import { drainLog, probeTexture } from "@dylanebert/shallot/runtime";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const manifest = JSON.parse(readFileSync(resolve(import.meta.dir, "node_modules/@dylanebert/shallot/package.json"), "utf8"));

test("the packed engine exposes no test-support namespace or capture helpers", () => {
    expect(Object.keys(manifest.exports).some((path) => /^\\.\\/(?:harness|testing)(?:\\/|$)/.test(path))).toBe(false);
    expect(existsSync(resolve(import.meta.dir, "node_modules/@dylanebert/shallot/src/harness"))).toBe(false);
    expect("captureArtifact" in Rendering).toBe(false);
    expect("captureIdentityLabel" in Rendering).toBe(false);
    expect("captureIdentityMatches" in Rendering).toBe(false);
    expect("assertCaptureGeometry" in Rendering).toBe(false);
});

test("a headless plugin set steps the world and exposes state through public engine subpaths", async () => {
    const Ticks = { value: sparse(f32) };
    let eid = -1;
    const Counter: Plugin = {
        name: "Counter",
        components: { counter: Ticks },
        initialize(state) {
            eid = state.create();
            state.add(eid, Ticks);
            Ticks.value.set(eid, 0);
        },
        systems: [{
            group: "fixed",
            update(state) {
                for (const entity of state.query([Ticks])) {
                    Ticks.value.set(entity, Ticks.value.get(entity) + 1);
                }
            },
        }],
    };
    const app = await build({ plugins: [Counter], defaults: false });
    try {
        app.state.step(Time.FIXED_DT);
        expect(app.state.time.fixedTick).toBe(1);
        expect(app.state.only([Ticks])).toBe(eid);
        expect(Ticks.value.get(eid)).toBe(1);
        expect(Rendering.CAPTURE_CONTRACT.width).toBe(1280);
        expect(typeof Rendering.captureFrame).toBe("function");
        expect(typeof probeTexture).toBe("function");
        expect(typeof drainLog).toBe("function");
    } finally {
        app.dispose();
    }
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
        const cliInstall = run(
            ["bun", "install", "--linker=isolated", "--no-progress"],
            cliProject,
            "installing packed CLI project",
        );
        const installedPackage = join(cliProject, "node_modules/@dylanebert/shallot");
        const installedManifest = JSON.parse(
            readFileSync(join(installedPackage, "package.json"), "utf8"),
        );
        expect(installedManifest.scripts.prepare).toBe(rootPackage.scripts.prepare);
        expect(cliInstall).not.toContain("build-tooling: compiled dist/vite.js");

        const packedProjectTests = run(
            ["bun", "test"],
            cliProject,
            "testing packed project exports",
        );
        expect(packedProjectTests).toContain(
            "the packed engine exposes no test-support namespace or capture helpers",
        );
        expect(packedProjectTests).toContain(
            "a headless plugin set steps the world and exposes state through public engine subpaths",
        );
        expect(packedProjectTests).toContain(
            "the packed Vite entry imports in Node and exposes only shallot",
        );
        expect(packedProjectTests).toContain("3 pass");

        const recipes = readdirSync(join(ROOT, "examples"))
            .filter((name) => existsSync(join(ROOT, "examples", name, "shallot.json")))
            .filter(
                (name) =>
                    JSON.parse(readFileSync(join(ROOT, "examples", name, "shallot.json"), "utf8"))
                        .kind === "recipe",
            )
            .sort();
        expect(recipes).toEqual(["first-person", "loading-screen"]);
        expect(existsSync(join(scratch, "package.json"))).toBe(false);

        for (const name of recipes) {
            const example = join(scratch, name);
            run(
                ["bun", "run", join(installedPackage, "bin/shallot.ts"), "add", name, example],
                cliProject,
                `shallot add ${name}`,
            );

            const examplePackagePath = join(example, "package.json");
            const examplePackage = JSON.parse(readFileSync(examplePackagePath, "utf8"));
            expect(examplePackage.dependencies["@dylanebert/shallot"]).toBe(packageVersion);
            expect(examplePackage.devDependencies["@types/bun"]).toBe(
                rootPackage.devDependencies["@types/bun"],
            );
            expect(examplePackage.devDependencies.typegpu).toBeUndefined();
            expect(existsSync(join(scratch, "package.json"))).toBe(false);
            expect(readFileSync(join(example, "tests/preload.ts"), "utf8")).toBe(
                'import { plugin } from "bun";\nimport { shallot } from "@dylanebert/shallot/bun";\nplugin(shallot());\n',
            );
            expect(readFileSync(join(example, "bunfig.toml"), "utf8")).toBe(
                '[test]\npreload = ["./tests/preload.ts"]\n',
            );

            const packedRange = `file:${relative(example, tarball).replaceAll("\\", "/")}`;
            examplePackage.dependencies["@dylanebert/shallot"] = packedRange;
            writeFileSync(examplePackagePath, `${JSON.stringify(examplePackage, null, 4)}\n`);
            writeFileSync(join(example, "no-preload.toml"), "[test]\n");
            writeFileSync(
                join(example, "src/packed-tgsl.test.ts"),
                `import { expect, test } from "bun:test";
import { linearToSrgb } from "@dylanebert/shallot/rendering";
import { checkTgsl } from "@dylanebert/shallot/runtime";

test("the Bun preload transforms engine TGSL and keeps it callable on the CPU", () => {
    checkTgsl();
    const encoded = linearToSrgb([0.5, 0.5, 0.5] as Parameters<typeof linearToSrgb>[0]);
    expect(encoded.x).toBeCloseTo(0.7353569, 5);
    expect(encoded.y).toBeCloseTo(0.7353569, 5);
    expect(encoded.z).toBeCloseTo(0.7353569, 5);
});
`,
            );

            const exampleInstall = run(
                ["bun", "install", "--linker=isolated", "--no-progress"],
                example,
                `installing ${name} from its own manifest`,
            );
            expect(exampleInstall).not.toContain("build-tooling: compiled dist/vite.js");
            expect(existsSync(join(example, "node_modules/@types/bun/package.json"))).toBe(true);
            run(
                [
                    "node",
                    join(example, "node_modules/typescript/bin/tsc"),
                    "--project",
                    "tsconfig.json",
                ],
                example,
                `tsc ${name}`,
            );
            run(
                ["bun", join(example, "node_modules/vite/bin/vite.js"), "build"],
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
            cwd: scratch,
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(packed.exitCode).toBe(0);
        expect(packed.stdout.toString()).toContain("tsconfig.base.json");
        expect(packed.stdout.toString()).toContain("examples/first-person/tsconfig.json");
        expect(packed.stdout.toString()).toContain("examples/loading-screen/tsconfig.json");
        expect(packed.stdout.toString()).not.toContain("examples/first-person/package.json");
        expect(packed.stdout.toString()).not.toContain("examples/loading-screen/package.json");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
}, 300_000);
