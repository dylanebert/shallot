import { resolve } from "path";

// The published tarball must ship only what a consumer needs. Test files and glTF fixtures are
// dev-only weight — this asserts against the real `bun pm pack`
// output, not the `files` field in isolation, so a future files-field edit can't silently regress it.
const pkgDir = resolve(import.meta.dir, "../packages/shallot");

const proc = Bun.spawn(["bun", "pm", "pack", "--dry-run"], {
    cwd: pkgDir,
    stdout: "pipe",
    stderr: "pipe",
});
const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
]);

if (exitCode !== 0) {
    console.error("✗ bun pm pack --dry-run failed:\n", stderr);
    process.exit(1);
}

// bun colors `packed` when the ambient env asks for it (FORCE_COLOR, a TTY), and the SGR codes sit
// between `^` and the word — a gate whose verdict depends on ambient color config is a latent red.
// biome-ignore lint/suspicious/noControlCharactersInRegex: ESC is the SGR introducer — matching it is the point.
const output = `${stdout}\n${stderr}`.replaceAll(/\x1b\[[0-9;]*m/g, "");
const files = [...output.matchAll(/^packed\s+\S+\s+(.+)$/gm)].map((m) => m[1]);

if (files.length === 0) {
    console.error("✗ check-pack: no packed files parsed from `bun pm pack --dry-run` output");
    process.exit(1);
}

// `.probes.ts` is the by-path gate suffix — a test file the default `bun test` glob deliberately misses
// (suite-speed discipline), which is exactly why it also slips a `.test.ts`-only pack check.
const violations = files.filter(
    (f) => f.endsWith(".test.ts") || f.endsWith(".probes.ts") || f.includes("/fixtures/"),
);

if (violations.length > 0) {
    console.error(`✗ ${violations.length} file(s) that must not ship in the npm pack:\n`);
    for (const f of violations) console.error(`  ${f}`);
    console.error(
        "\nTest files and glTF fixtures are dev-only weight. Exclude them via the `files` field\n" +
            "in packages/shallot/package.json.",
    );
    process.exit(1);
}

// the native-host crate ships in the tarball so `shallot build --target <native>` compiles it lazily
// from a standard install. Assert the crate source and the relocated icon are present, and that the
// build artifacts (target/) never ship.
const required = ["rust/window/Cargo.toml", "rust/window/Cargo.lock", "assets/icon-1024.png"];
const missing = required.filter((f) => !files.includes(f));
if (missing.length > 0) {
    console.error(`✗ ${missing.length} required file(s) missing from the npm pack:\n`);
    for (const f of missing) console.error(`  ${f}`);
    console.error(
        "\nThe rust/window crate source and the icon must ship for native builds from an install.",
    );
    process.exit(1);
}

const targetLeaks = files.filter((f) => f.startsWith("rust/window/target/"));
if (targetLeaks.length > 0) {
    console.error(
        `✗ ${targetLeaks.length} rust/window/target/ file(s) leaked into the npm pack:\n`,
    );
    for (const f of targetLeaks) console.error(`  ${f}`);
    console.error(
        "\nBuild artifacts must not ship. Exclude them via the `files` field in package.json.",
    );
    process.exit(1);
}

console.log(
    `✓ tarball clean (${files.length} files, no test.ts or fixtures, crate + icon present)`,
);
