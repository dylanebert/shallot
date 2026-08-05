// Compile the two Node-reachable tooling exports (`exports.md`'s compiled-tooling-export law): a
// `vite.config.ts` and a `playwright.config.ts` both resolve through Node's plain ESM loader, which
// throws `ERR_UNKNOWN_FILE_EXTENSION` on the package's raw `.ts` source — the reason every other export
// stays raw `.ts` (the mandatory TypeGPU transform must see engine source untransformed) doesn't reach
// these two, whose only consumption context is Node. `dist/` is generated here, gitignored, never
// committed — `postpack.ts` removes it after pack, same shape as the examples projection in
// `prepack.ts`. `tsc` still type-checks against source (`package.json`'s `types` condition, which only
// `tsc` reads — a bundler resolves straight to `default`, not `types`), so this emits no `.d.ts`.
//
// `src/project/` is a closed island — node builtins plus the `vite` / `unplugin-typegpu` externals it
// imports, no engine runtime, no TGSL — so bundling it carries no duplicate-TypeGPU-identity risk. Kept
// external explicitly (not just left to Bun's default builtin handling) so a bundled copy of either
// package, which would be the actual regression this build must not introduce, fails loud in review
// rather than silently inlining.

import { mkdirSync, readFileSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dir, ".."); // packages/shallot
const OUT = resolve(ROOT, "dist");

// dist/ is regenerated on every pack — never accumulate a stale build's leftovers (a manual `shallot
// build` run against this package as its own project would otherwise land unrelated output here too).
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const EXTERNAL = [
    "vite",
    "unplugin-typegpu",
    "unplugin-typegpu/vite",
    "node:fs",
    "node:path",
    "fs",
    "path",
];

const entries: { entry: string; out: string }[] = [
    { entry: resolve(ROOT, "src/project/vite.ts"), out: "vite" },
    { entry: resolve(ROOT, "src/harness/browser.ts"), out: "harness-browser" },
];

for (const { entry, out } of entries) {
    const result = await Bun.build({
        entrypoints: [entry],
        outdir: OUT,
        target: "node",
        format: "esm",
        naming: `${out}.js`,
        external: EXTERNAL,
        minify: false,
    });
    if (!result.success) {
        for (const log of result.logs) console.error(log);
        throw new Error(`build-tooling: failed to compile ${entry}`);
    }
}

// `src/project/` is a closed island (engine.ts/generate.ts/manifest.ts, no engine runtime, no TGSL) —
// verify empirically rather than assuming, since a duplicate engine identity landing in a Node-loaded
// bundle would be a serious regression (the exact defect class this whole spec exists to fix).
//
// The real check is the emitted bundle's remaining top-level imports: bundling inlines an internal
// import and erases its statement entirely, so a marker on the erased text (`'from "../engine'`, …)
// can never fire — it would pass silently on a leaked import whose source text just doesn't happen to
// contain `GPUShaderStage` / `requestGPU` / `tgpu.resolve`. What survives bundling either way is the
// import statement for anything Bun couldn't (or was told not to) inline, so asserting that surviving
// set is exactly the declared `EXTERNAL` list — no more, no less — catches both an engine module
// surfacing as an import (a leak) and any other unexpected external the entries shouldn't carry.
function importsOf(js: string): Set<string> {
    // matches `import x from "y"`, `import { a, b } from "y"` (multi-line specifier lists included),
    // and the bare `import "y"` side-effect form — not dynamic `import(...)`, which none of these
    // entries use and which Bun.build never rewrites into a static `import` anyway.
    const re = /(?:^|\n)\s*import\s*(?:[^;\n]*?\sfrom\s*)?["']([^"']+)["']/g;
    const specs = new Set<string>();
    for (const m of js.matchAll(re)) specs.add(m[1]);
    return specs;
}

for (const { out } of entries) {
    const emitted = readFileSync(resolve(OUT, `${out}.js`), "utf8");
    const leaked = [...importsOf(emitted)].filter((spec) => !EXTERNAL.includes(spec));
    if (leaked.length) {
        throw new Error(
            `build-tooling: dist/${out}.js carries an import outside the declared externals, expected only ${EXTERNAL.join(", ")}: ${leaked.join(", ")}`,
        );
    }
}

// Content markers stay as a second, cheap check for the case an engine-runtime symbol got inlined
// (not imported) into the bundle — the import-set check above can't see that shape, since an inlined
// symbol carries no import statement at all.
const emitted = readFileSync(resolve(OUT, "vite.js"), "utf8");
const forbidden = ["GPUShaderStage", "requestGPU", "tgpu.resolve"];
const leakedContent = forbidden.filter((needle) => emitted.includes(needle));
if (leakedContent.length) {
    throw new Error(
        `build-tooling: dist/vite.js carries engine-runtime content, expected a pure tooling bundle: ${leakedContent.join(", ")}`,
    );
}

console.log(`build-tooling: compiled ${entries.map((e) => `dist/${e.out}.js`).join(", ")}`);
