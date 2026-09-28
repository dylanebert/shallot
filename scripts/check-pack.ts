import { existsSync, readdirSync, readFileSync } from "node:fs";
import { Glob } from "bun";
import { resolve } from "path";

// The published tarball ships source, the CLI, the compiled tooling leaves, the Rust audio WASM and
// native-window crate, the icon and recipes; never a path the `files` negations name, nor build
// output. Asserted against the real `bun pm pack` output, not the `files`
// allowlist in isolation, so a negation the packer ignores still reds.
const pkgDir = resolve(import.meta.dir, "..");
const packageManifest = JSON.parse(readFileSync(resolve(pkgDir, "package.json"), "utf8"));

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

// The `**/` negations in `files`, read back from package.json so the pack gate and the allowlist can't disagree.
const packageFiles = packageManifest.files as string[];
const negated = packageFiles
    .filter((entry) => entry.startsWith("!"))
    .map((entry) => new Glob(entry.slice(1)));
const requiredNegations = [
    "!src/transitional/physics/oracle/**",
    "!**/fixtures",
    "!**/*.test.ts",
    "!**/*.oracle.ts",
    "!**/*.node.ts",
    "!**/*.e2e.ts",
];
const missingNegations = requiredNegations.filter((entry) => !packageFiles.includes(entry));
const isNegated = (file: string): boolean => {
    const parts = file.split("/");
    const ancestors = parts.map((_, index) => parts.slice(0, index + 1).join("/"));
    return negated.some((glob) => ancestors.some((path) => glob.match(path)));
};
const forbidden: [string, (f: string) => boolean][] = [
    ["files negation", isNegated],
    ["non-carrier script", (f) => f.startsWith("scripts/") && f.endsWith(".ts")],
    ["build output", (f) => f.includes("/node_modules/")],
    ["site assets", (f) => f.startsWith("assets/") && f !== "assets/icon-1024.png"],
    ["repo docs", (f) => f.endsWith(".md") && f !== "README.md" && !f.startsWith("examples/")],
    ["test-support source", (f) => /^src\/(?:harness|testing)\//.test(f)],
    ["physics oracle test data", (f) => f.startsWith("src/transitional/physics/oracle/")],
];
const violations = files.flatMap((f) =>
    forbidden.filter(([, match]) => match(f)).map(([kind]) => `${f} (${kind})`),
);
for (const file of files.filter((file) => /^examples\/[^/]+\/package\.json$/.test(file)))
    violations.push(`${file} (example package manifest)`);

const examplesDir = resolve(pkgDir, "examples");
const recipes = readdirSync(examplesDir)
    .filter((name) => existsSync(resolve(examplesDir, name, "shallot.json")))
    .filter(
        (name) =>
            JSON.parse(readFileSync(resolve(examplesDir, name, "shallot.json"), "utf8")).kind ===
            "recipe",
    )
    .sort();

const required = [
    "src/index.ts",
    "src/project/bun.ts",
    "tsconfig.base.json",
    "src/core/rendering/capture.ts",
    "bin/shallot.ts",
    "src/project/policy.ts",
    "src/cli/index.ts",
    "dist/vite.js",
    "crates/native/Cargo.toml",
    "crates/native/Cargo.lock",
    "assets/icon-1024.png",
    "shallot.schema.json",
    "crates/audio/pkg/shallot_audio.js",
    "crates/audio/pkg/shallot_audio.d.ts",
    "crates/audio/pkg/shallot_audio.wasm",
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

const removedTestSupport = Object.keys(packageManifest.exports ?? {}).filter((entry) =>
    /^\.\/(?:harness|testing)(?:\/|$)/.test(entry),
);
for (const entry of removedTestSupport) violations.push(`${entry} (test-support export)`);

const missing = required.filter((f) => !files.includes(f));
for (const name of recipes) {
    const config = `examples/${name}/tsconfig.json`;
    if (!files.includes(config)) missing.push(config);
}
for (const entry of ["./bun", "./tsconfig.json"]) {
    if (!(entry in (packageManifest.exports ?? {})))
        violations.push(`${entry} (missing package export)`);
}
for (const entry of missingNegations) violations.push(`${entry} (missing files negation)`);
// The shipped example set is exactly the `kind: "recipe"` manifests: `files` negates showcases by
// name, so a new showcase that misses its negation (or a recipe caught by one) reds here.
const declaredExamples = packageFiles
    .filter((entry) => /^examples\/[^/]+$/.test(entry))
    .map((entry) => entry.slice("examples/".length))
    .sort();
const shipped = [
    ...new Set(files.filter((f) => f.startsWith("examples/")).map((f) => f.split("/")[1])),
].sort();
if (recipes.length === 0) missing.push("examples/<recipe>/");
if (declaredExamples.join() !== recipes.join())
    violations.push(
        `package files allowlist names [${declaredExamples.join(", ")}], recipes are [${recipes.join(", ")}]`,
    );
if (shipped.join() !== recipes.join())
    violations.push(`examples/ ships [${shipped.join(", ")}], recipes are [${recipes.join(", ")}]`);

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
