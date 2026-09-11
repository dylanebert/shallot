import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const mode = Bun.argv.includes("--integration")
    ? "integration"
    : Bun.argv.includes("--all")
      ? "all"
      : "unit";
const env = { ...process.env };
if (mode === "unit") env.SHALLOT_UNIT_ONLY = "1";
// Integration runs include unit rows because some integration fixtures need their setup hooks.
const started = performance.now();
const proc = Bun.spawnSync(
    [
        process.execPath,
        "test",
        "--max-concurrency=1",
        "--pass-with-no-tests",
        "src",
        "scripts",
        "examples",
    ],
    { cwd: root, env, stdout: "pipe", stderr: "pipe" },
);
await new Promise<void>((done, fail) =>
    process.stdout.write(proc.stdout.toString() + proc.stderr.toString(), "utf8", (error) =>
        error ? fail(error) : done(),
    ),
);
console.log(`suite wall time: ${Math.round(performance.now() - started)}ms`);
process.exit(proc.exitCode ?? 1);
