import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { Glob } from "bun";
import { TEST_TIER_SUFFIX_NAMES } from "../packages/shallot/tests/test-tiers";
import { EXAMPLE_GATES } from "./example-gates";
import { OCEAN_CPU_GATES } from "./ocean-oracle-gates";
import { main, selectCpuGates, selectExampleGates } from "./test-changed";

const dirs = (paths: string[]) => selectExampleGates(paths).map((row) => row.dir);
const cpus = (paths: string[]) => selectCpuGates(paths).map((row) => row.script);

describe("changed-path selector", () => {
    test("example selection preserves exact and whole-roster cones", () => {
        expect(dirs(["examples/recipes/moving-platform/src/plugin.ts"])).toEqual([
            "examples/recipes/moving-platform",
        ]);
        expect(dirs(["packages/shallot/src/standard/render/plugin.ts"])).toEqual(
            EXAMPLE_GATES.map((row) => row.dir),
        );
        expect(dirs(["bun.lock"])).toEqual(EXAMPLE_GATES.map((row) => row.dir));
        expect(dirs(["examples/showcase/visualization/package.json"])).toEqual(
            EXAMPLE_GATES.map((row) => row.dir),
        );
        expect(dirs(["docs/selector.md"])).toEqual([]);
    });

    test("every CPU oracle is selected by a real header-named path", () => {
        const witnesses: Record<string, string> = {
            "test:ocean-realization": "examples/showcase/ocean/src/ocean/fft.ts",
            "test:ocean-slope": "examples/showcase/ocean/src/ocean/slope.ts",
            "test:ocean-mesh-inversion": "examples/showcase/ocean/src/ocean/clipmap.ts",
            "test:ocean-fold": "examples/showcase/ocean/src/ocean/composed-fold.ts",
        };
        expect(Object.keys(witnesses).sort()).toEqual(
            OCEAN_CPU_GATES.map((row) => row.script).sort(),
        );
        for (const [script, path] of Object.entries(witnesses))
            expect(cpus([path])).toContain(script);
    });

    test("every CPU cover matches a tracked path and every command resolves to its root script", async () => {
        const tracked = Bun.spawnSync(["git", "ls-files"], { cwd: resolve(import.meta.dir, "..") });
        expect(tracked.success).toBe(true);
        const files = tracked.stdout.toString().split("\n").filter(Boolean);
        const pkg = await Bun.file(resolve(import.meta.dir, "../package.json")).json();
        for (const row of OCEAN_CPU_GATES) {
            expect(pkg.scripts[row.script]).toBeString();
            expect(pkg.scripts[row.script]).toContain(row.covers[0]);
            for (const cover of row.covers)
                expect(files.some((file) => new Glob(cover).match(file))).toBe(true);
        }
    });

    test("every explicit root test path resolves at least one default-tier test file", async () => {
        const root = resolve(import.meta.dir, "..");
        const pkg = await Bun.file(resolve(root, "package.json")).json();
        const tokens = pkg.scripts.test
            .match(/(?:^|\s)(?!bun$)([^\s]+)/g)
            .map((s: string) => s.trim())
            .slice(2);
        for (const token of tokens) {
            expect(existsSync(resolve(root, token))).toBe(true);
            const scan = new Glob("**/*.test.ts").scanSync({ cwd: resolve(root, token) });
            expect([...scan].length, token).toBeGreaterThan(0);
        }
    });

    test("the derived default cone launches no by-path tier", async () => {
        const root = resolve(import.meta.dir, "..");
        const pkg = await Bun.file(resolve(root, "package.json")).json();
        const dirs = pkg.scripts.test.split(/\s+/).slice(2);
        const byPathSuffixes = TEST_TIER_SUFFIX_NAMES.filter((name) => name !== "test");
        const tierCommand = new RegExp(
            `(?:\\.(?:${byPathSuffixes.join("|")})\\.ts|shallot\\s+verify|bun\\s+(?:bench|run\\s+(?:flows|recipes|test:install)))`,
        );
        const launch = new RegExp(
            `(?:const\\s+\\w*(?:COMMAND|CMD)\\s*=\\s*[\\s\\S]{0,300}${tierCommand.source}|Bun\\.spawn(?:Sync)?\\s*\\([\\s\\S]{0,300}${tierCommand.source})`,
        );
        for (const dir of dirs) {
            const files = new Glob("**/*.test.ts").scanSync({ cwd: resolve(root, dir) });
            for (const file of files) {
                const path = resolve(root, dir, file);
                expect(await Bun.file(path).text(), path).not.toMatch(launch);
            }
        }
    });
});

describe("changed-path execution tiers", () => {
    const args = ["--base", "base", "--diff", "head"];
    test("CPU runs before an unavailable display and reports that distinct verdict", async () => {
        const commands: string[] = [];
        const logs: string[] = [];
        const old = console.log;
        console.log = (...parts) => logs.push(parts.join(" "));
        try {
            const code = await main(args, {
                paths: async () => ["examples/showcase/ocean/src/ocean/fft.ts"],
                run: async (command) => {
                    commands.push(command);
                    return { ok: true, warnings: 0 };
                },
                displaySkip: () => "fixture seat",
            });
            expect(code).toBe(0);
            expect(commands).toContain("bun run test:ocean-realization");
            expect(
                logs.some((line) =>
                    line.includes("CPU rows passed; display rows were unavailable"),
                ),
            ).toBe(true);
        } finally {
            console.log = old;
        }
    });

    test("required display refusal and CPU failure are red", async () => {
        expect(
            await main(args, {
                paths: async () => ["examples/showcase/ocean/src/ocean/fft.ts"],
                run: async () => ({ ok: true, warnings: 0 }),
                displaySkip: () => "fixture",
                displayRequired: true,
            }),
        ).toBe(1);
        expect(
            await main(args, {
                paths: async () => ["examples/showcase/ocean/src/ocean/fft.ts"],
                run: async () => ({ ok: false, warnings: 0 }),
                displaySkip: () => "fixture",
            }),
        ).toBe(1);
    });

    test("zero selection is distinct and runs nothing", async () => {
        let ran = false;
        expect(
            await main(args, {
                paths: async () => ["README.md"],
                run: async () => {
                    ran = true;
                    return { ok: true, warnings: 0 };
                },
            }),
        ).toBe(0);
        expect(ran).toBe(false);
    });
});
