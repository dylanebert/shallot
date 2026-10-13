import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "@dylanebert/shallot/testing";

setDefaultTimeout(CEILING.node);

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import { Glob } from "bun";
import { PROJECT_OUTPUT } from "../src/cli/create";
import { normalizePath } from "../src/project/paths";
import { importsIn, resolveRelative } from "./example-imports";

// The published tarball ships source, the CLI, the compiled tooling leaves, the Rust audio WASM and
// native-window crate, the icon and recipes; never a path the `files` negations name, nor build
// output. Asserted against the real `bun pm pack` output, not the `files`
// allowlist in isolation, so a negation the packer ignores still reds.
test("the npm package contains only declared files and required consumers", async () => {
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

    if (exitCode !== 0) throw new Error(`bun pm pack --dry-run failed:\n${stderr}`);

    // bun colors `packed` when the ambient env asks for it (FORCE_COLOR, a TTY), and the SGR codes sit
    // between `^` and the word — a gate whose verdict depends on ambient color config is a latent red.
    // biome-ignore lint/suspicious/noControlCharactersInRegex: ESC is the SGR introducer — matching it is the point.
    const output = `${stdout}\n${stderr}`.replaceAll(/\x1b\[[0-9;]*m/g, "");
    const files = [...output.matchAll(/^packed\s+\S+\s+(.+)$/gm)].map((m) => m[1]);

    if (files.length === 0)
        throw new Error("no packed files parsed from `bun pm pack --dry-run` output");

    // The `**/` negations in `files`, read back from package.json so the pack gate and the allowlist can't disagree.
    const packageFiles = packageManifest.files as string[];
    const negated = packageFiles
        .filter((entry) => entry.startsWith("!"))
        .map((entry) => new Glob(entry.slice(1)));
    const requiredNegations = [
        "!src/standard/physics/oracle/**",
        "!**/fixtures",
        "!src/**/*.test.ts",
        "!**/*.oracle.ts",
        "!src/**/*.node.ts",
        "!src/**/*.e2e.ts",
        ...PROJECT_OUTPUT.map((directory) => `!examples/*/${directory}`),
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
        [
            "example project output",
            (f) =>
                f.startsWith("examples/") &&
                f.split("/").some((part) => PROJECT_OUTPUT.includes(part)),
        ],
        ["site assets", (f) => f.startsWith("assets/") && f !== "assets/icon-1024.png"],
        [
            "repo docs",
            (f) =>
                f.endsWith(".md") &&
                f !== "README.md" &&
                f !== "src/cli/skill.md" &&
                !f.startsWith("examples/"),
        ],
        ["test-support source", (f) => /^src\/(?:harness|testing)\//.test(f)],
        ["physics oracle test data", (f) => f.startsWith("src/standard/physics/oracle/")],
    ];
    const violations = files.flatMap((f) =>
        forbidden.filter(([, match]) => match(f)).map(([kind]) => `${f} (${kind})`),
    );

    const packedFiles = new Set(files);
    for (const file of files.filter((file) => /\.(?:[cm]?js|[cm]?ts|jsx|tsx)$/.test(file))) {
        const source = readFileSync(resolve(pkgDir, file), "utf8");
        for (const specifier of importsIn(source, file).specifiers) {
            if (!specifier.startsWith("./") && !specifier.startsWith("../")) continue;
            const clean = specifier.split(/[?#]/, 1)[0]!;
            const targetFile = resolveRelative(resolve(pkgDir, file), clean);
            if (!targetFile) continue;
            const target = normalizePath(relative(pkgDir, targetFile));
            if (!packedFiles.has(target))
                violations.push(
                    `${file} -> ${specifier} resolves to unshipped ${target} (relative import)`,
                );
        }
    }
    const examplesDir = resolve(pkgDir, "examples");
    const recipes = readdirSync(examplesDir, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name)
        .sort();

    const required = [
        "src/index.ts",
        "src/project/bun.ts",
        "tsconfig.base.json",
        "src/core/rendering/capture.ts",
        "src/extras/text/Inter-Regular.ttf",
        "src/extras/text/Inter-OFL.txt",
        "bin/shallot.ts",
        "src/project/policy.ts",
        "src/cli/index.ts",
        "src/cli/skill.md",
        "src/project/assets.js",
        "src/project/testing.js",
        "src/project/vite.js",
        "examples/minimal/index.html",
        "examples/minimal/playwright.config.ts",
        "examples/minimal/public/icon.svg",
        "examples/minimal/src/welcome/index.ts",
        "examples/minimal/tsconfig.json",
        "examples/minimal/vite.config.ts",
        "examples/first-person/playwright.config.ts",
        "examples/first-person/tests/browser-setup.ts",
        "examples/first-person/tests/browser-install.ts",
        "examples/first-person/tests/capture-page.ts",
        "examples/first-person/src/plugins.ts",
        "dist/testing.d.ts",
        "dist/vite.d.ts",
        "crates/native/Cargo.toml",
        "crates/native/Cargo.lock",
        "assets/icon-1024.png",
        "crates/audio/pkg/shallot_audio.js",
        "crates/audio/pkg/shallot_audio.d.ts",
        "crates/audio/pkg/shallot_audio.wasm",
    ];
    // An example manifest would shadow the root package self-reference; `shallot create` writes the copied project's manifest.
    for (const name of recipes)
        if (existsSync(resolve(examplesDir, name, "package.json")))
            violations.push(
                `examples/${name}/package.json (example manifest; shallot create writes it)`,
            );

    const removedTestSupport = Object.keys(packageManifest.exports ?? {}).filter(
        (entry) => /^\.\/harness(?:\/|$)/.test(entry) || entry.startsWith("./testing/"),
    );
    for (const entry of removedTestSupport) violations.push(`${entry} (test-support export)`);

    const missing = required.filter((f) => !files.includes(f));
    const exampleChecks = ["examples/**/*.test.ts", "examples/**/*.node.ts", "examples/**/*.e2e.ts"]
        .flatMap((pattern) => Array.from(new Glob(pattern).scanSync(pkgDir), normalizePath))
        .sort();
    for (const check of exampleChecks)
        if (!files.includes(check)) missing.push(`${check} (example check not shipped)`);
    for (const name of recipes) {
        const config = `examples/${name}/tsconfig.json`;
        if (!files.includes(config)) missing.push(config);
    }
    for (const entry of ["./bun", "./testing", "./tsconfig.json"]) {
        if (!(entry in (packageManifest.exports ?? {})))
            violations.push(`${entry} (missing package export)`);
    }
    for (const entry of missingNegations) violations.push(`${entry} (missing files negation)`);
    // Every example directory must be declared and shipped.
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
        violations.push(
            `examples/ ships [${shipped.join(", ")}], recipes are [${recipes.join(", ")}]`,
        );

    if (violations.length > 0) {
        console.error(`✗ ${violations.length} file(s) that must not ship in the npm pack:\n`);
        for (const v of violations) console.error(`  ${v}`);
    }
    if (missing.length > 0) {
        console.error(`✗ ${missing.length} required file(s) missing from the npm pack:\n`);
        for (const f of missing) console.error(`  ${f}`);
    }
    expect(
        { violations, missing },
        `Fix the package files allowlist (dist/ needs \`bun run build\`); ${files.length} files packed`,
    ).toEqual({ violations: [], missing: [] });
});
