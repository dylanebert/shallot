import { readFileSync } from "node:fs";
import { resolve } from "path";
import { TEST_TIER_SUFFIXES } from "../tests/test-tiers";

// The published tarball ships source, the CLI, the compiled tooling leaves, the Rust audio WASM and
// native-window crate, the icon, recipes and consumer docs; never tests, oracles, tiers, probes,
// fixtures, goldens or build output. Asserted against the real `bun pm pack` output, not the `files`
// allowlist in isolation, so a negation the packer ignores still reds.
//
// Recipes ship with their `src/smoke.ts`: each recipe's `shallot.json` names it, so a copied-out
// recipe needs it.
const pkgDir = resolve(import.meta.dir, "..");

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

const forbidden: [string, (f: string) => boolean][] = [
    ["test tiers", (f) => TEST_TIER_SUFFIXES.test(f) || f.endsWith(".fixture.ts")],
    ["goldens", (f) => f.endsWith(".gold.json")],
    ["fixtures", (f) => f.includes("/fixtures/")],
    ["tests/", (f) => f.startsWith("tests/")],
    ["build output", (f) => f.includes("/target/") || f.includes("/node_modules/")],
    ["site assets", (f) => f.startsWith("assets/") && f !== "assets/icon-1024.png"],
    ["maintainer docs", (f) => f === "MAINTAINERS.md" || f === "CONTRIBUTING.md"],
];
const violations = files.flatMap((f) =>
    forbidden.filter(([, match]) => match(f)).map(([kind]) => `${f} (${kind})`),
);

const required = [
    "src/index.ts",
    "bin/cli.ts",
    "dist/vite.js",
    "dist/harness-browser.js",
    "rust/window/Cargo.toml",
    "rust/window/Cargo.lock",
    "assets/icon-1024.png",
    "AGENTS.md",
    "MIGRATION.md",
    "examples/AGENTS.md",
    "shallot.schema.json",
];
// A shipped manifest must not name the engine by a local path: copy-out pins the installed version,
// and a `file:`/`link:`/`workspace:` spec in the tarball points at a directory no consumer has.
const ENGINE = "@dylanebert/shallot";
for (const f of files.filter((f) => f.endsWith("package.json") && f !== "package.json")) {
    const pkg = JSON.parse(readFileSync(resolve(pkgDir, f), "utf8"));
    for (const field of ["dependencies", "devDependencies", "optionalDependencies"]) {
        const range = pkg[field]?.[ENGINE];
        if (typeof range === "string" && /^(?:workspace:|file:|link:|portal:)/.test(range))
            violations.push(`${f} (local engine dependency ${field}: ${range})`);
    }
}

const missing = required.filter((f) => !files.includes(f));
if (!files.some((f) => f.startsWith("rust/audio/pkg/"))) missing.push("rust/audio/pkg/");
if (!files.some((f) => f.startsWith("examples/recipes/"))) missing.push("examples/recipes/");

if (violations.length > 0) {
    console.error(`✗ ${violations.length} file(s) that must not ship in the npm pack:\n`);
    for (const v of violations) console.error(`  ${v}`);
}
if (missing.length > 0) {
    console.error(`✗ ${missing.length} required file(s) missing from the npm pack:\n`);
    for (const f of missing) console.error(`  ${f}`);
}
if (violations.length > 0 || missing.length > 0) {
    console.error("\nFix the `files` allowlist in package.json (dist/ needs `bun run build`).");
    process.exit(1);
}

console.log(`✓ tarball matches the files allowlist (${files.length} files)`);
