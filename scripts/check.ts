import { resolve } from "node:path";
import { Glob } from "bun";
import { readProjectPolicy } from "../src/project/policy";

// `bun run check`: every read-only gate in order, one line per arm, stopping at the first red.
// `bun run` puts node_modules/.bin on PATH, so `tsc` and `biome` resolve to the pinned copies.

const root = resolve(import.meta.dir, "..");
const readers = [...new Glob("check-*.ts").scanSync(import.meta.dir)]
    .filter((file) => !/\.(?:test|node|oracle)\.ts$/.test(file))
    .sort();
const arms: [string, string[]][] = [
    ["tsc", ["tsc"]],
    ["biome", ["biome", "check"]],
    ...readers.map((file): [string, string[]] => [
        file.replace(/\.ts$/, ""),
        ["bun", resolve(import.meta.dir, file)],
    ]),
    ["cargo fmt", ["cargo", "fmt", "--all", "--check"]],
    [
        "cargo clippy (physics)",
        ["cargo", "clippy", "-p", "shallot-physics", "--all-targets", "--", "-D", "warnings"],
    ],
    [
        "cargo clippy (physics wasm32)",
        [
            "cargo",
            "clippy",
            "-p",
            "shallot-physics",
            "--lib",
            "--target",
            "wasm32-unknown-unknown",
            "--",
            "-D",
            "warnings",
        ],
    ],
];

const policyViolations = readProjectPolicy(root);
let failed = policyViolations.length > 0;
if (failed) {
    for (const violation of policyViolations) console.error(`✗ ${violation}`);
} else {
    console.log("✓ project policy");
}
for (const [name, command] of arms) {
    const start = performance.now();
    const proc = Bun.spawnSync(command, { cwd: root, stdout: "pipe", stderr: "pipe" });
    const seconds = ((performance.now() - start) / 1000).toFixed(1);
    if (!proc.success) {
        process.stdout.write(proc.stdout);
        process.stderr.write(proc.stderr);
        console.error(`✗ ${name} (${seconds}s)`);
        failed = true;
        continue;
    }
    console.log(`✓ ${name} (${seconds}s)`);
}
if (failed) process.exit(1);
