import { expect } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { check } from "./check";
import { collectPopulation } from "./surface";

const ROOT = resolve(import.meta.dir, "../..");
const CHECK_MODULE = JSON.stringify(resolve(import.meta.dir, "check.ts"));
const TYPEGPU_MODULE = JSON.stringify(import.meta.resolve("typegpu"));
const TYPEGPU_DATA_MODULE = JSON.stringify(import.meta.resolve("typegpu/data"));

check(
    "load refuses self subjects, import-time spawns and in-repository temp dirs",
    {
        claim: "discovery refuses a row naming its own file as subject, a check file spawning at module scope and a temp dir rooted outside tmpdir(), and admits each rule's legitimate form",
        subject: "src/harness/surface.ts",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-load-refusal-"));
        try {
            mkdirSync(join(tree, "src"));
            const write = (name: string, body: string) =>
                writeFileSync(
                    join(tree, "src", name),
                    `import { check } from ${CHECK_MODULE};\n${body}`,
                );
            write(
                "self.test.ts",
                'check("self", { claim: "self", subject: "src/self.test.ts" }, () => {});\n',
            );
            write(
                "spawn.test.ts",
                'const found = Bun.spawnSync(["true"]);\ncheck("spawn", { claim: "spawn" }, () => {});\n',
            );
            write(
                "repo-temp.test.ts",
                'import { mkdtempSync } from "node:fs";\ncheck("repo temp", { claim: "repo temp" }, () => { ' +
                    // Split so this file's own source does not read as an in-repository temp dir.
                    "mkdtemp" +
                    'Sync(join(import.meta.dir, ".x-")); });\n',
            );
            write(
                "admitted.test.ts",
                'import { mkdtempSync } from "node:fs";\nimport { tmpdir } from "node:os";\n' +
                    'const helper = () => Bun.spawnSync(["true"]);\n' +
                    'check("admitted", { claim: "admitted", subject: "src/admitted.ts" }, () => { mkdtempSync(join(tmpdir(), "x-")); helper(); });\n',
            );
            const population = collectPopulation(tree);
            expect(population.invalid).toEqual([
                "in-repository temp dir: src/repo-temp.test.ts calls mkdtempSync outside tmpdir(); root temp dirs at the host temp directory",
                'self subject: src/self.test.ts check("self") names its own file as subject; name the source it checks',
                "import-time spawn: src/spawn.test.ts spawns a process at module scope; spawn inside a check() body so discovery runs nothing",
            ]);
            expect(population.rows.map((row) => row.claim).sort()).toEqual([
                "admitted",
                "repo temp",
                "spawn",
            ]);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "a selected integration row loads only its own file and an empty population refuses",
    {
        claim: "the runner loads only a selected integration row's file, never every discovered file, and refuses an empty population instead of exiting green",
        size: "integration",
        subject: "scripts/test-runner.ts",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-runner-selection-"));
        try {
            const environment = { ...process.env };
            delete environment.KEX_S3_ROW;
            const runner = (...args: string[]) => {
                const proc = Bun.spawnSync(
                    ["bun", resolve(ROOT, "scripts/test-runner.ts"), "--root", tree, ...args],
                    { cwd: ROOT, env: environment, stdout: "pipe", stderr: "pipe" },
                );
                return { exitCode: proc.exitCode, stderr: proc.stderr.toString() };
            };

            const empty = runner();
            expect(empty.exitCode).toBe(1);
            expect(empty.stderr).toContain("empty population");

            mkdirSync(join(tree, "src"));
            const loaded = (name: string) => join(tree, `${name}-loaded`);
            for (const name of ["selected", "other"]) {
                writeFileSync(
                    join(tree, "src", `${name}.test.ts`),
                    `import { appendFileSync } from "node:fs";\nimport { check } from ${CHECK_MODULE};\n` +
                        `appendFileSync(${JSON.stringify(loaded(name))}, "loaded");\n` +
                        `check(${JSON.stringify(name)}, { claim: ${JSON.stringify(name)}, size: "integration", subject: "src/${name}.ts" }, () => {});\n`,
                );
            }
            const transformLoaded = join(tree, "transform-loaded");
            for (const [extension, name, value] of [
                ["js", "shader", 1],
                ["mjs", "moduleShader", 2],
            ] as const) {
                writeFileSync(
                    join(tree, "src", `selected-shader.${extension}`),
                    `import tgpu from ${TYPEGPU_MODULE};\n` +
                        `import * as d from ${TYPEGPU_DATA_MODULE};\n` +
                        `export const ${name} = tgpu.fn([d.f32], d.f32)((x) => { "use gpu"; return x + ${value}; });\n`,
                );
            }
            writeFileSync(
                join(tree, "src", "transform.test.ts"),
                `import { appendFileSync } from "node:fs";\n` +
                    `import tgpu from ${TYPEGPU_MODULE};\n` +
                    `import { shader } from "./selected-shader.js";\n` +
                    `import { moduleShader } from "./selected-shader.mjs";\n` +
                    `import { check } from ${CHECK_MODULE};\n` +
                    `appendFileSync(${JSON.stringify(transformLoaded)}, "loaded");\n` +
                    `check("transform", { claim: "user GPU rows transform imported JS and MJS shader modules", size: "integration", subject: "src/selected.ts" }, () => tgpu.resolve([shader, moduleShader]));\n`,
            );
            const selected = runner("--integration", "--subject", "src/selected");
            if (selected.exitCode !== 0) throw new Error(selected.stderr);
            expect(existsSync(loaded("selected"))).toBe(true);
            expect(existsSync(loaded("other"))).toBe(false);
            expect(existsSync(transformLoaded)).toBe(true);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);
