import { beforeAll, expect, test } from "bun:test";
import {
    cpSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { EXAMPLE_GATES } from "./example-gates";
import { OCEAN_CPU_GATES } from "./ocean-oracle-gates";

const root = resolve(import.meta.dir, "..");
type Reading = { exitCode: number; output: string };
const readings = new Map<string, Reading>();

// A complete tracked-tree fixture keeps every preceding production guard live. Its index
// preserves the real population; only the named mutation differs from the working tree.
beforeAll(async () => {
    const fixture = mkdtempSync(join(root, "node_modules/.check-docs-"));
    const index = Bun.spawnSync(["git", "ls-files", "--stage"], { cwd: root });
    expect(index.exitCode).toBe(0);
    try {
        for (const entry of index.stdout.toString().trim().split("\n")) {
            const file = entry.slice(entry.indexOf("\t") + 1);
            mkdirSync(dirname(join(fixture, file)), { recursive: true });
            cpSync(join(root, file), join(fixture, file));
        }
        expect(Bun.spawnSync(["git", "init", "--quiet", fixture]).exitCode).toBe(0);
        expect(
            Bun.spawnSync(["git", "update-index", "--index-info"], {
                cwd: fixture,
                stdin: index.stdout,
            }).exitCode,
        ).toBe(0);
        symlinkSync(join(root, "node_modules"), join(fixture, "node_modules"));
        const run = async (name: string, edits: Record<string, (text: string) => string>) => {
            const originals = new Map<string, string>();
            try {
                for (const [file, mutate] of Object.entries(edits)) {
                    const path = join(fixture, file);
                    const original = readFileSync(path, "utf8");
                    originals.set(path, original);
                    const changed = mutate(original);
                    expect(changed).not.toBe(original);
                    writeFileSync(path, changed);
                }
                const proc = Bun.spawn(["bun", "scripts/check-docs.ts"], {
                    cwd: fixture,
                    stdout: "pipe",
                    stderr: "pipe",
                });
                const [stdout, stderr, exitCode] = await Promise.all([
                    new Response(proc.stdout).text(),
                    new Response(proc.stderr).text(),
                    proc.exited,
                ]);
                const reading = { exitCode, output: stdout + stderr };
                readings.set(name, reading);
                console.log(`[docs control: ${name}] exit=${exitCode}\n${reading.output}`);
            } finally {
                for (const [path, original] of originals) writeFileSync(path, original);
            }
        };
        await run("baseline", {});
        await run("pointers", {
            "scripts/rosters.ts": (text) =>
                text + "\n// see zzz-vacuity-dead-seed.md\n// see README.md\n// see checks.md\n",
        });
        await run("old release clause", {
            ".claude/rules/testing.md": (text) =>
                text.replace(
                    /then `SHALLOT_DISPLAY_REQUIRED=1 bun run test:changed --all`.*?then merge/,
                    "then `bun run test:changed --all` (the whole-roster release gate, subsuming `bun run demos`, recipes, flows, and every showcase gate; a skip is not green), then merge",
                ),
        });
        const oldCone =
            "bun run test       # unit tests over packages/shallot, scripts, evals, showcase/visualization/test (bun-webgpu)";
        await run("old README cone", {
            "README.md": (text) => text.replace(/^bun run test\s+#.*$/m, oldCone),
        });
        const pkg = JSON.parse(readFileSync(join(fixture, "package.json"), "utf8"));
        const paths = pkg.scripts.test.replace(/^bun test\s+/, "").split(/\s+/);
        const cone = `bun run test # unit tests over ${paths.join(", ")} (bun-webgpu)`;
        const commands = [
            ...OCEAN_CPU_GATES.map((row) => `bun run ${row.script}`),
            ...EXAMPLE_GATES.map((row) => row.gate),
        ];
        await run("all legitimate rows and exact cone", {
            "README.md": (text) =>
                text.replace(/^bun run test\s+#.*$/m, cone) +
                commands
                    .map(
                        (command) =>
                            `\n\n\`bun run test:changed -- --all\` subsumes \`${command}\`.`,
                    )
                    .join(""),
        });
        await run("extra cone member", {
            "README.md": (text) =>
                text.replace(
                    /^bun run test\s+#.*$/m,
                    cone.replace(" (bun-webgpu)", ", examples/gym/src (bun-webgpu)"),
                ),
        });
        await run("manifest moves independently", {
            "README.md": (text) => text.replace(/^bun run test\s+#.*$/m, cone),
            "package.json": (text) => {
                const changed = JSON.parse(text);
                changed.scripts.test += " examples/gym/src";
                return JSON.stringify(changed);
            },
        });
        await run("registry moves independently", {
            "README.md": (text) =>
                text + `\n\n\`bun run test:changed --all\` subsumes \`${EXAMPLE_GATES[0].gate}\`.`,
            "scripts/example-gates.ts": (text) =>
                text.replace(
                    `gate: ${JSON.stringify(EXAMPLE_GATES[0].gate)}`,
                    'gate: "bun run demos"',
                ),
        });
        await run("unclassified command claims", {
            "README.md": (text) =>
                text +
                "\n\n`bun run test:changed --all` subsumes `bun run test:ocean-slope` and demos.",
        });
        await run("unselected commands", {
            "README.md": (text) =>
                text +
                [
                    "bun run demos",
                    "bun run site",
                    "bun run rum-intake",
                    "bun run recipes --recipe nonexistent",
                ]
                    .map((command) => `\n\n\`bun run test:changed --all\` subsumes \`${command}\`.`)
                    .join(""),
        });
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
}, 60000);

test("production docs consumer grants the current tree", () => {
    expect(readings.get("baseline")?.exitCode).toBe(0);
});

test("pointer validity refuses dead and private-only basenames, grants a live basename", () => {
    const reading = readings.get("pointers")!;
    expect(reading.exitCode).toBe(1);
    expect(reading.output).toMatch(/rosters\.ts:\d+: zzz-vacuity-dead-seed\.md/);
    expect(reading.output).toMatch(/rosters\.ts:\d+: checks\.md/);
    expect(reading.output).not.toMatch(/rosters\.ts:\d+: README\.md/);
});

for (const [name, diagnostic] of [
    ["old release clause", "false subsumption: bun run demos"],
    ["old README cone", "stale root test cone"],
    ["extra cone member", "extra [examples/gym/src]"],
    ["manifest moves independently", "missing [examples/gym/src]"],
    ["unselected commands", "false subsumption: bun run site"],
    ["registry moves independently", `false subsumption: ${EXAMPLE_GATES[0].gate}`],
    ["unclassified command claims", "subsumption needs explicit row commands"],
]) {
    test(`command composition refuses ${name} after preceding guards`, () => {
        const reading = readings.get(name)!;
        expect(reading.exitCode).toBe(1);
        expect(reading.output).toContain("✗ command composition:");
        expect(reading.output).toContain(diagnostic);
    });
}

test("command composition grants every registered command and the exact manifest cone", () => {
    const reading = readings.get("all legitimate rows and exact cone")!;
    expect(reading.exitCode).toBe(0);
    expect(reading.output).toContain("✓ command composition");
});
