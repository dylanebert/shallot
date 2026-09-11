import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const args = Bun.argv.slice(2);
const base = args[args.indexOf("--base") + 1] ?? "HEAD^";
const diff = args[args.indexOf("--diff") + 1] ?? "HEAD";
const all = args.includes("--all");

function run(command: string[]): number {
    const proc = Bun.spawnSync(command, { cwd: root, stdout: "inherit", stderr: "inherit" });
    return proc.exitCode ?? 1;
}

const started = performance.now();
if (run([process.execPath, "run", "scripts/test-runner.ts", "--unit"]) !== 0) process.exit(1);
if (run([process.execPath, "run", "scripts/test-runner.ts", "--integration"]) !== 0)
    process.exit(1);

const changed = all
    ? ["--forced"]
    : Bun.spawnSync(["git", "diff", "--name-only", base, diff], {
          cwd: root,
          stdout: "pipe",
          stderr: "inherit",
      })
          .stdout.toString()
          .trim()
          .split("\n")
          .filter(Boolean);
const needsCargo =
    all ||
    changed.some(
        (file) =>
            file.startsWith("crates/") ||
            /^Cargo(?:\.lock|\.toml)?$/.test(file) ||
            file.includes("solver/fixtures"),
    );
if (
    needsCargo &&
    run(["cargo", "test", "--workspace", "--exclude", "shallot-native", "--quiet"]) !== 0
)
    process.exit(1);
console.log(`suite wall time: ${Math.round(performance.now() - started)}ms`);
