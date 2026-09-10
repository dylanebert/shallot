import { resolve } from "node:path";
import { Glob } from "bun";

// `bun run check`: every read-only gate in order, one line per arm, stopping at the first red.
// `bun run` puts node_modules/.bin on PATH, so `tsc` and `biome` resolve to the pinned copies.

const root = resolve(import.meta.dir, "..");
const readers = [...new Glob("check-*.ts").scanSync(import.meta.dir)].sort();
const arms: [string, string[]][] = [
    ["tsc", ["tsc"]],
    ["biome", ["biome", "check"]],
    ...readers.map((file): [string, string[]] => [
        file.replace(/\.ts$/, ""),
        ["bun", resolve(import.meta.dir, file)],
    ]),
    ["scene format", ["bun", resolve(import.meta.dir, "format.ts"), "--check"]],
    ["cargo fmt", ["cargo", "fmt", "--all", "--check"]],
];

for (const [name, command] of arms) {
    const start = performance.now();
    const proc = Bun.spawnSync(command, { cwd: root, stdout: "pipe", stderr: "pipe" });
    const seconds = ((performance.now() - start) / 1000).toFixed(1);
    if (!proc.success) {
        process.stdout.write(proc.stdout);
        process.stderr.write(proc.stderr);
        console.error(`✗ ${name} (${seconds}s)`);
        process.exit(proc.exitCode || 1);
    }
    console.log(`✓ ${name} (${seconds}s)`);
}
