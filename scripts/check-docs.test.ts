import { beforeAll, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import {
    cpSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { EXAMPLE_GATES } from "./example-gates";
import { OCEAN_CPU_GATES } from "./ocean-oracle-gates";

const root = resolve(import.meta.dir, "..");
type Reading = { exitCode: number; output: string };
const readings = new Map<string, Reading>();
const baselineFile = "scripts/instruction-budget.json";
const hash = (text: string) => createHash("sha256").update(text).digest("hex");

// A complete tracked-tree fixture keeps every preceding production guard live. Its index
// preserves the real population; only the named mutation differs from the working tree.
beforeAll(async () => {
    const container = mkdtempSync(join(tmpdir(), "shallot-check-docs-"));
    const fixture = join(container, "member");
    mkdirSync(fixture);
    expect(Bun.spawnSync(["git", "init", "--quiet", container]).exitCode).toBe(0);
    writeFileSync(join(container, "checks.md"), "# private parent rule\n");
    expect(Bun.spawnSync(["git", "add", "checks.md"], { cwd: container }).exitCode).toBe(0);
    const index = Bun.spawnSync(["git", "ls-files", "--stage"], { cwd: root });
    expect(index.exitCode).toBe(0);
    try {
        for (const entry of index.stdout.toString().trim().split("\n")) {
            const file = entry.slice(entry.indexOf("\t") + 1);
            mkdirSync(dirname(join(fixture, file)), { recursive: true });
            cpSync(join(root, file), join(fixture, file));
        }
        cpSync(join(root, baselineFile), join(fixture, baselineFile));
        expect(Bun.spawnSync(["git", "init", "--quiet", fixture]).exitCode).toBe(0);
        expect(
            Bun.spawnSync(["git", "update-index", "--index-info"], {
                cwd: fixture,
                stdin: index.stdout,
            }).exitCode,
        ).toBe(0);
        symlinkSync(join(root, "node_modules"), join(fixture, "node_modules"));
        const run = async (
            name: string,
            edits: Record<string, (text: string) => string>,
            args: string[] = [],
            setup?: () => () => void,
        ) => {
            const originals = new Map<string, string>();
            const baselinePath = join(fixture, baselineFile);
            const baselineOriginal = readFileSync(baselinePath, "utf8");
            let cleanup: (() => void) | undefined;
            try {
                for (const [file, mutate] of Object.entries(edits)) {
                    const path = join(fixture, file);
                    const original = readFileSync(path, "utf8");
                    originals.set(path, original);
                    const changed = mutate(original);
                    expect(changed).not.toBe(original);
                    writeFileSync(path, changed);
                }
                cleanup = setup?.();
                const before = readFileSync(baselinePath, "utf8");
                const proc = Bun.spawn(["bun", "run", "scripts/check-docs.ts", ...args], {
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
                const after = readFileSync(baselinePath, "utf8");
                console.log(
                    `[docs control: ${name}] exit=${exitCode} before=${hash(before)} after=${hash(after)}\n${reading.output}`,
                );
                if (!args.length || exitCode !== 0) expect(after).toBe(before);
                if (name === "valid lowering") {
                    expect(exitCode).toBe(0);
                    const oldBudget = JSON.parse(before);
                    const newBudget = JSON.parse(after);
                    expect(newBudget.total).toBeLessThan(oldBudget.total);
                    for (const [file, size] of Object.entries(newBudget.files) as [
                        string,
                        { bytes: number; paragraph: number },
                    ][]) {
                        expect(size.bytes).toBeLessThanOrEqual(oldBudget.files[file].bytes);
                        expect(size.paragraph).toBeLessThanOrEqual(oldBudget.files[file].paragraph);
                    }
                    expect(hash(after)).not.toBe(hash(before));
                }
            } finally {
                cleanup?.();
                for (const [path, original] of originals) writeFileSync(path, original);
                writeFileSync(baselinePath, baselineOriginal);
            }
        };
        await run("baseline", {});
        await run("prototype is not a declared script", {
            "AGENTS.md": (text) => text.replace("bun run build", "bun toString"),
        });
        expect(readings.get("prototype is not a declared script")?.exitCode).toBe(1);
        expect(readings.get("prototype is not a declared script")?.output).toContain(
            "unreachable repository command: bun toString",
        );
        await run("unreachable documented path", {
            "AGENTS.md": (text) =>
                text.replace("packages/shallot-cli/bin/cli.ts", "packages/shallot-cli/bin/gone.ts"),
        });
        await run("registry lookup is not tree resolution", {
            "AGENTS.md": (text) =>
                text.replace("bun packages/shallot-cli/bin/cli.ts", "bunx shallot"),
        });
        for (const kind of ["bin", "files"]) {
            await run(`unregistered ${kind} target`, {
                "packages/shallot/package.json": (text) => {
                    const pkg = JSON.parse(text);
                    if (kind === "bin") pkg.bin.shallot = "./bin/gone.ts";
                    else pkg.files.push("unregistered-pack-output");
                    return JSON.stringify(pkg);
                },
            });
        }
        const audio = ".claude/rules/audio.md";
        const deadAudio = ["Missing", "AudioCitation"].join("");
        for (const marker of ["retired", "gone", "anti-pattern"]) {
            await run(`dead citation with ${marker}`, {
                [audio]: (text) => text.replace("`NodeType`", `\`${deadAudio}\` (${marker})`),
            });
        }
        await run("bare dead citation", {
            [audio]: (text) => text.replace("`NodeType`", deadAudio),
        });
        await run("live citation without exemption", {
            [audio]: (text) => text.replace("`NodeType`", "`LiveSkin`"),
        });
        const suffixes = ["oracle", "tier", "lab"];
        for (const [name, roster] of [
            ["regex", suffixes.join("|")],
            ["array", JSON.stringify(suffixes.map((suffix) => `.${suffix}.ts`))],
        ]) {
            await run(`tier ${name} restatement`, {
                "scripts/foreign-namespaces.ts": (text) => `${text}\n// ${roster}\n`,
            });
        }
        const visual = ".claude/rules/visual-identity.md";
        const byteGrowth = { [visual]: (text: string) => `${text}\n\nsmall addition\n` };
        const paragraphGrowth = { [visual]: (text: string) => text.replace(/\n\s*\n/g, " ") };
        await run("byte growth", byteGrowth);
        await run("paragraph growth", paragraphGrowth);
        await run("lower refuses byte growth", byteGrowth, ["--lower"]);
        await run("lower refuses paragraph growth", paragraphGrowth, ["--lower"]);
        const calibration = Bun.spawnSync(["bun", "run", "scripts/check-docs.ts", "--lower"], {
            cwd: fixture,
        });
        console.log(
            `[docs fixture calibration] exit=${calibration.exitCode}\n${calibration.stdout}${calibration.stderr}`,
        );
        expect(calibration.exitCode).toBe(0);
        const totalGrowth = {
            [baselineFile]: (text: string) => {
                const budget = JSON.parse(text);
                budget.total--;
                return JSON.stringify(budget);
            },
        };
        await run("corpus growth", totalGrowth);
        await run("lower refuses corpus growth", totalGrowth, ["--lower"]);
        await run(
            "valid lowering",
            {
                [visual]: (text) =>
                    text.replace("Shipped UI: examples, overlays, profiler HUD.\n", ""),
            },
            ["--lower"],
        );
        const addMember =
            (file: string, link = false, tracked = false) =>
            () => {
                const path = join(fixture, file);
                mkdirSync(dirname(path), { recursive: true });
                if (link) symlinkSync(join(fixture, "CLAUDE.md"), path);
                else writeFileSync(path, "# local instructions\n");
                if (tracked)
                    expect(
                        Bun.spawnSync(["git", "add", "--", file], { cwd: fixture }).exitCode,
                    ).toBe(0);
                return () => {
                    if (tracked)
                        expect(
                            Bun.spawnSync(["git", "update-index", "--force-remove", "--", file], {
                                cwd: fixture,
                            }).exitCode,
                        ).toBe(0);
                    rmSync(path);
                };
            };
        await run("untracked rule", {}, [], addMember("nested/.claude/rules/new.md"));
        await run("nested entry", {}, [], addMember("nested/deeper/AGENTS.md"));
        await run("tracked symlink", {}, [], addMember("nested/CLAUDE.md", true, true));
        await run("tracked symlink mode", {}, [], () => {
            const cleanup = addMember("nested/CLAUDE.md", true, true)();
            rmSync(join(fixture, "nested/CLAUDE.md"));
            writeFileSync(join(fixture, "nested/CLAUDE.md"), "# regular working file\n");
            return cleanup;
        });
        await run("untracked symlink", {}, [], addMember("nested/AGENTS.md", true));
        await run("lower refuses unlisted", {}, ["--lower"], addMember("nested/AGENTS.md"));
        await run("closed vocabulary", {}, [], addMember("nested/INSTRUCTIONS.md"));
        await run("ignored member", {}, [], addMember("dist/AGENTS.md"));
        await run("pointers", {
            "scripts/foreign-namespaces.ts": (text) =>
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
        const command = /(?:^| && )bun test\s+([^&]+)$/.exec(pkg.scripts.test);
        expect(command).not.toBeNull();
        const paths = command![1].trim().split(/\s+/);
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
                    cone.replace(" (bun-webgpu)", ", bench/src (bun-webgpu)"),
                ),
        });
        await run("manifest moves independently", {
            "README.md": (text) => text.replace(/^bun run test\s+#.*$/m, cone),
            "package.json": (text) => {
                const changed = JSON.parse(text);
                changed.scripts.test += " bench/src";
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
        rmSync(container, { recursive: true, force: true });
    }
}, 60000);

for (const [name, diagnostic] of [
    ["prototype is not a declared script", "unreachable repository command: bun toString"],
    ["unreachable documented path", "unreachable repository command"],
    ["registry lookup is not tree resolution", "unreachable repository command: bunx shallot"],
    ["unregistered bin target", "bin: ./bin/gone.ts is missing"],
    ["unregistered files target", "files: unregistered-pack-output is missing"],
]) {
    test(`command resolution refuses ${name}`, () => {
        const reading = readings.get(name)!;
        expect(reading.exitCode).toBe(1);
        expect(reading.output).toContain("✗ command resolution:");
        expect(reading.output).toContain(diagnostic);
    });
}

for (const name of [
    "baseline",
    "valid lowering",
    "closed vocabulary",
    "ignored member",
    "live citation without exemption",
]) {
    test(`production docs consumer grants ${name}`, () => {
        expect(readings.get(name)?.exitCode).toBe(0);
    });
}

for (const [name, diagnostic] of [
    ["byte growth", "byte growth:"],
    ["paragraph growth", "paragraph growth:"],
    ["corpus growth", "corpus growth:"],
    ["lower refuses byte growth", "byte growth:"],
    ["lower refuses paragraph growth", "paragraph growth:"],
    ["lower refuses corpus growth", "corpus growth:"],
    ["untracked rule", "unlisted member: nested/.claude/rules/new.md"],
    ["nested entry", "unlisted member: nested/deeper/AGENTS.md"],
    ["tracked symlink", "symlink member: nested/CLAUDE.md"],
    ["tracked symlink mode", "symlink member: nested/CLAUDE.md"],
    ["untracked symlink", "symlink member: nested/AGENTS.md"],
    ["lower refuses unlisted", "unlisted member: nested/AGENTS.md"],
]) {
    test(`instruction ratchet refuses ${name} after preceding guards`, () => {
        const reading = readings.get(name)!;
        expect(reading.exitCode).toBe(1);
        expect(reading.output).toContain("✓ command composition");
        expect(reading.output).toContain("✗ instruction ratchet:");
        expect(reading.output).toContain(diagnostic);
        if (name.includes("paragraph")) expect(reading.output).not.toContain("byte growth:");
        if (name.includes("byte")) expect(reading.output).not.toContain("paragraph growth:");
    });
}

for (const name of [
    "dead citation with retired",
    "dead citation with gone",
    "dead citation with anti-pattern",
    "bare dead citation",
]) {
    test(`citation resolution refuses ${name} without a marker escape`, () => {
        const reading = readings.get(name)!;
        expect(reading.exitCode).toBe(1);
        expect(reading.output).toContain("✗ citation resolution:");
        expect(reading.output).toContain(["Missing", "AudioCitation"].join(""));
        expect(reading.output).not.toContain("count below floor");
    });
}

for (const shape of ["regex", "array"]) {
    test(`tier roster refuses ${shape} restatement without prose enumeration`, () => {
        const reading = readings.get(`tier ${shape} restatement`)!;
        expect(reading.exitCode).toBe(1);
        expect(reading.output).toContain("✗ tier-suffix roster arm:");
        expect(reading.output).toContain("scripts/foreign-namespaces.ts:");
        expect(reading.output).toContain("carries a literal tier-suffix roster");
    });
}

test("pointer validity refuses dead and private-only basenames, grants a live basename", () => {
    const reading = readings.get("pointers")!;
    expect(reading.exitCode).toBe(1);
    expect(reading.output).toMatch(/foreign-namespaces\.ts:\d+: zzz-vacuity-dead-seed\.md/);
    expect(reading.output).toMatch(/foreign-namespaces\.ts:\d+: checks\.md/);
    expect(reading.output).not.toMatch(/foreign-namespaces\.ts:\d+: README\.md/);
});

for (const [name, diagnostic] of [
    ["old release clause", "false subsumption: bun run demos"],
    ["old README cone", "stale root test cone"],
    ["extra cone member", "extra [bench/src]"],
    ["manifest moves independently", "missing [bench/src]"],
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
