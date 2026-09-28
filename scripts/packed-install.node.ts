import { expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

test("a packed fresh Bun project steps and observes a headless world through engine exports", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-packed-install-"));
    const project = join(scratch, "project");
    const packageVersion = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")).version;
    const tarballName = `dylanebert-shallot-${packageVersion}.tgz`;
    mkdirSync(project);
    try {
        const packed = Bun.spawnSync(
            ["bun", "pm", "pack", "--ignore-scripts", "--destination", scratch, "--quiet"],
            { cwd: ROOT, stdout: "pipe", stderr: "pipe" },
        );
        if (packed.exitCode !== 0) {
            throw new Error(`packing Shallot failed:\n${packed.stderr.toString()}`);
        }

        writeFileSync(
            join(project, "package.json"),
            JSON.stringify({
                name: "packed-install-project",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:../${tarballName}`,
                    typegpu: "~0.12.5",
                },
            }),
        );
        writeFileSync(
            join(project, "fresh.test.ts"),
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
}, 5_000);
`,
        );

        const tarball = join(scratch, tarballName);
        if (!existsSync(tarball)) throw new Error(`packed tarball is missing: ${tarball}`);

        const install = Bun.spawnSync(["bun", "install", "--no-progress"], {
            cwd: project,
            stdout: "pipe",
            stderr: "pipe",
        });
        if (install.exitCode !== 0) {
            throw new Error(`fresh project install failed:\n${install.stderr.toString()}`);
        }

        const testRun = Bun.spawnSync(["bun", "test"], {
            cwd: project,
            stdout: "pipe",
            stderr: "pipe",
        });
        if (testRun.exitCode !== 0) {
            throw new Error(
                `fresh project plain bun test failed:\n${testRun.stdout.toString()}\n${testRun.stderr.toString()}`,
            );
        }
        expect(`${testRun.stdout.toString()}\n${testRun.stderr.toString()}`).toContain("3 pass");

        writeFileSync(
            join(project, "vite.config.ts"),
            'import { shallot } from "@dylanebert/shallot/vite";\nexport default { plugins: [shallot()] };\n',
        );
        writeFileSync(
            join(project, "tsconfig.node.json"),
            JSON.stringify({
                compilerOptions: {
                    target: "ESNext",
                    module: "NodeNext",
                    moduleResolution: "NodeNext",
                    strict: true,
                    noUnusedLocals: true,
                    noEmit: true,
                },
                include: ["vite.config.ts"],
            }),
        );
        const typecheck = Bun.spawnSync(
            ["node", resolve(ROOT, "node_modules/typescript/bin/tsc"), "-p", "tsconfig.node.json"],
            { cwd: project, stdout: "pipe", stderr: "pipe" },
        );
        expect(typecheck.exitCode, typecheck.stderr.toString()).toBe(0);

        const tar = execFileSync("tar", ["-tzf", tarball], {
            encoding: "utf8",
        });
        expect(tar.split("\n").some((path) => path.includes("/src/harness/"))).toBe(false);
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
}, 120_000);
