import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { Glob } from "bun";
import { TEST_TIER_SUFFIX_NAMES } from "../packages/shallot/tests/test-tiers";
import { EXAMPLE_GATES } from "./example-gates";
import { OCEAN_CPU_GATES } from "./ocean-oracle-gates";
import { changedPaths, main, runCommand, selectCpuGates, selectExampleGates } from "./test-changed";

const dirs = (paths: string[]) => selectExampleGates(paths).map((row) => row.dir);
const cpus = (paths: string[]) => selectCpuGates(paths).map((row) => row.script);

describe("changed-path selector", () => {
    test("example selection preserves assertion cones and whole-roster escalation", () => {
        expect(dirs(["examples/recipes/moving-platform/src/plugin.ts"])).toEqual([
            "examples/recipes/moving-platform",
        ]);
        expect(dirs(["packages/shallot-runtime/src/standard/render/plugin.ts"])).toEqual([
            "examples/recipes/day-night-sky",
            "examples/recipes/gpu-particles",
            "examples/flows/no-walls",
            "examples/showcase/collapse",
            "examples/showcase/ocean",
            "examples/showcase/roads",
            "examples/showcase/sandbox",
            "examples/showcase/visualization",
            "examples/showcase/voxel",
            "examples/gym",
        ]);
        expect(dirs(["packages/shallot-runtime/src/standard/fog/index.ts"])).toEqual([
            "examples/gym",
        ]);
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

    test("every CPU cover matches a tracked path and every command declares its recorded per-test ceiling", async () => {
        const tracked = Bun.spawnSync(["git", "ls-files"], { cwd: resolve(import.meta.dir, "..") });
        expect(tracked.success).toBe(true);
        const files = tracked.stdout.toString().split("\n").filter(Boolean);
        const pkg = await Bun.file(resolve(import.meta.dir, "../package.json")).json();
        for (const row of OCEAN_CPU_GATES) {
            const command = pkg.scripts[row.script];
            expect(command).toBeString();
            expect(command).toContain(row.covers[0]);
            expect(command).toContain(`--timeout ${row.timeoutMs}`);
            expect(row.recordedFraction).toBeGreaterThan(0);
            expect(row.recordedFraction).toBeLessThanOrEqual(0.5);
            for (const cover of row.covers)
                expect(files.some((file) => new Glob(cover).match(file))).toBe(true);
        }
    });

    test("a deleted example cover remains a changed path and selects its display row", async () => {
        const root = mkdtempSync(resolve(tmpdir(), "shallot-changed-example-delete-"));
        const run = (...args: string[]) => {
            const result = Bun.spawnSync(["git", ...args], { cwd: root });
            expect(result.success, result.stderr.toString()).toBe(true);
        };
        const deleted = "examples/recipes/annotate-the-world/src/smoke.ts";
        const path = resolve(root, deleted);
        mkdirSync(resolve(path, ".."), { recursive: true });
        try {
            run("init", "-q");
            run("config", "user.email", "gate@example.invalid");
            run("config", "user.name", "Gate Fixture");
            writeFileSync(path, "fixture", { flush: true });
            run("add", ".");
            run("commit", "-qm", "base");
            run("rm", "-q", deleted);
            run("commit", "-qm", "delete");
            const oldCwd = process.cwd();
            process.chdir(root);
            try {
                const paths = await changedPaths("HEAD^", "HEAD");
                expect(paths).toContain(deleted);
                expect(dirs(paths)).toEqual(["examples/recipes/annotate-the-world"]);
            } finally {
                process.chdir(oldCwd);
            }
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    test("a deleted oracle cover remains a changed path and selects its CPU row", async () => {
        const root = mkdtempSync(resolve(tmpdir(), "shallot-changed-delete-"));
        const run = (...args: string[]) => {
            const result = Bun.spawnSync(["git", ...args], { cwd: root });
            expect(result.success, result.stderr.toString()).toBe(true);
        };
        try {
            run("init", "-q");
            run("config", "user.email", "gate@example.invalid");
            run("config", "user.name", "Gate Fixture");
            const oracle = "examples/showcase/ocean/test/fold-anchor.oracle.ts";
            const path = resolve(root, oracle);
            mkdirSync(resolve(path, ".."), { recursive: true });
            writeFileSync(path, "fixture", { flush: true });
            run("add", ".");
            run("commit", "-qm", "base");
            run("rm", "-q", oracle);
            run("commit", "-qm", "delete");
            const oldCwd = process.cwd();
            process.chdir(root);
            try {
                const paths = await changedPaths("HEAD^", "HEAD");
                expect(paths).toContain(oracle);
                expect(cpus(paths)).toContain("test:ocean-fold");
            } finally {
                process.chdir(oldCwd);
            }
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    test("every explicit root test path resolves at least one default-tier test file", async () => {
        const root = resolve(import.meta.dir, "..");
        const pkg = await Bun.file(resolve(root, "package.json")).json();
        const command = /(?:^| && )bun test\s+([^&]+)$/.exec(pkg.scripts.test);
        expect(command).not.toBeNull();
        const tokens = command![1].trim().split(/\s+/);
        for (const token of tokens) {
            expect(existsSync(resolve(root, token))).toBe(true);
            const scan = new Glob("**/*.test.ts").scanSync({ cwd: resolve(root, token) });
            expect([...scan].length, token).toBeGreaterThan(0);
        }
    });

    test("the derived default cone launches no by-path tier", async () => {
        const root = resolve(import.meta.dir, "..");
        const pkg = await Bun.file(resolve(root, "package.json")).json();
        const command = /(?:^| && )bun test\s+([^&]+)$/.exec(pkg.scripts.test);
        expect(command).not.toBeNull();
        const dirs = command![1].trim().split(/\s+/);
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

describe("manifest-owned verify transport", () => {
    test("loads each manifest revision and executes its cwd/argv/exit through the local CLI", async () => {
        const project = realpathSync(mkdtempSync(resolve(tmpdir(), "shallot-manifest-verify-")));
        const cli = resolve(project, "driver.mjs");
        const observed = resolve(project, "observed.json");
        writeFileSync(
            cli,
            `import {writeFileSync} from 'node:fs'; writeFileSync(${JSON.stringify(observed)}, JSON.stringify({cwd:process.cwd(),argv:process.argv.slice(2),runtime:process.versions.bun ? 'bun' : 'node'})); process.exit(process.argv.includes('exit=7') ? 7 : 0);`,
        );
        const setGate = (gate: string) =>
            writeFileSync(resolve(project, "package.json"), JSON.stringify({ scripts: { gate } }));
        const command = `bun run --cwd ${project} gate`;
        try {
            setGate("bunx shallot verify . --screenshot native.png --query exit=7");
            expect(await runCommand(command, { cli })).toEqual({
                ok: false,
                warnings: 0,
            });
            expect(await Bun.file(observed).json()).toEqual({
                cwd: project,
                argv: ["verify", ".", "--screenshot", "native.png", "--query", "exit=7"],
                runtime: "bun",
            });
            setGate("bunx shallot verify . --port 34567 --query exit=0");
            expect(await runCommand(command, { cli })).toEqual({ ok: true, warnings: 0 });
            expect(await Bun.file(observed).json()).toEqual({
                cwd: project,
                argv: ["verify", ".", "--port", "34567", "--query", "exit=0"],
                runtime: "bun",
            });
            setGate("bunx shallot verify . && echo false-green");
            await expect(runCommand(command, { cli })).rejects.toThrow(
                "unsupported verify gate composition",
            );
            setGate("bun --eval 'process.exit(7)'");
            expect((await runCommand(command)).ok).toBe(false);
        } finally {
            rmSync(project, { recursive: true, force: true });
        }
    });

    test("warning headings survive the real selector and manifest transport composition", async () => {
        const project = realpathSync(mkdtempSync(resolve(tmpdir(), "shallot-warning-count-")));
        const cli = resolve(project, "driver.mjs");
        const reader = resolve(import.meta.dir, "test-changed.ts");
        // Match the CLI and wrapper emitters, including chatter that is not a heading.
        const cases = [
            { stdout: "  warnings (18):\n    deprecated\n", stderr: "", count: 18, exit: 0 },
            {
                stdout: "",
                stderr: "  ⚠ 2 console warning(s):\n    deprecated\n",
                count: 2,
                exit: 7,
            },
            {
                stdout: "  warnings (3):\n",
                stderr: "  ⚠ 2 console warning(s):\n",
                count: 5,
                exit: 0,
            },
            {
                stdout: '  warnings (0):\n    message mentions warnings (18):\n{"detail":"warnings (9):"}\n',
                stderr: "warnings (x):\nwarnings (-1):\nwarnings (2): trailing text\n",
                count: 0,
                exit: 0,
            },
        ];
        try {
            for (const row of cases) {
                writeFileSync(
                    cli,
                    `process.stdout.write(${JSON.stringify(row.stdout)}); process.stderr.write(${JSON.stringify(row.stderr)}); process.exit(${row.exit});`,
                );
                const child = Bun.spawn(
                    [
                        process.execPath,
                        "--eval",
                        `
                    import {main,runCommand} from ${JSON.stringify(reader)};
                    process.exit(await main(['--base','HEAD','--diff','HEAD'], {
                        paths: async () => ['examples/showcase/ocean/shallot.json'],
                        displaySkip: () => null,
                        displayRequired: true,
                        run: command => runCommand(command, {cli:${JSON.stringify(cli)}})
                    }));`,
                    ],
                    { stdout: "pipe", stderr: "pipe" },
                );
                const [stdout, stderr, code] = await Promise.all([
                    new Response(child.stdout).text(),
                    new Response(child.stderr).text(),
                    child.exited,
                ]);
                expect(code, stderr).toBe(row.exit === 0 ? 0 : 1);
                expect(stdout).toContain("manifest gate:");
                expect(stdout).toContain(
                    `argv=["bun",${JSON.stringify(cli)},"verify",".","--screenshot","ocean.png"]`,
                );
                expect(stdout).toContain(
                    `${row.exit === 0 ? "PASS" : "FAIL"}: display examples/showcase/ocean (${row.count} warnings)`,
                );
            }
        } finally {
            rmSync(project, { recursive: true, force: true });
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

    test("main routes the selected CPU row before its display row with exact commands", async () => {
        const commands: string[] = [];
        expect(
            await main(args, {
                paths: async () => ["examples/showcase/ocean/src/ocean/fft.ts"],
                run: async (command) => {
                    commands.push(command);
                    return { ok: true, warnings: 0 };
                },
                displaySkip: () => null,
            }),
        ).toBe(0);
        expect(commands).toEqual([
            "bun run test:ocean-realization",
            "bun run --cwd examples/showcase/ocean gate",
        ]);
    });

    test("display rows still accumulate a later failure after an earlier pass", async () => {
        const commands: string[] = [];
        expect(
            await main(args, {
                paths: async () => ["examples/showcase/ocean/src/ocean/fft.ts"],
                run: async (command) => {
                    commands.push(command);
                    return { ok: commands.length === 1, warnings: 0 };
                },
                displaySkip: () => null,
            }),
        ).toBe(1);
        expect(commands).toEqual([
            "bun run test:ocean-realization",
            "bun run --cwd examples/showcase/ocean gate",
        ]);
    });

    test("a compound gate propagates either child failure instead of swallowing it", async () => {
        expect(EXAMPLE_GATES.find((row) => row.tier === "gym")?.gate).toBe(
            "bun bench --sweep && bun run --cwd examples/gym gate",
        );
        const root = realpathSync(mkdtempSync(resolve(import.meta.dir, "..", ".tmp-gym-command-")));
        const marker = resolve(root, "marker");
        const command = `bun run --cwd ${root} gate`;
        const setGate = (gate: string) =>
            writeFileSync(resolve(root, "package.json"), JSON.stringify({ scripts: { gate } }));
        try {
            setGate(
                `bun -e 'process.exit(7)' && bun -e 'Bun.write(${JSON.stringify(marker)}, "ran")'`,
            );
            expect((await runCommand(command)).ok).toBe(false);
            expect(existsSync(marker)).toBe(false);

            setGate(
                `bun -e 'Bun.write(${JSON.stringify(marker)}, "ran")' && bun -e 'process.exit(7)'`,
            );
            expect((await runCommand(command)).ok).toBe(false);
            expect(existsSync(marker)).toBe(true);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
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
