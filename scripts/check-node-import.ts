// The package must load under *real Node*, not only under bun — and no bun-side gate in this repo can
// see whether it does. bun tolerates two things Node ≥26 refuses: a JSON import with no
// `with { type: "json" }` attribute, and a *named* import off a JSON module. 0.9.2 shipped
// `src/standard/loading/index.ts` with the bare form, so every Node-side consumer of the package — a
// Playwright driver importing engine source, a `vite.config.ts` / `playwright.config.ts` reaching the
// engine — died at load before running. Witnessed on node v26.7.0, 2026-08-25, through this very arm:
//
//   bare import                      → TypeError: Module ".../packages/shallot/package.json" needs an
//                                      import attribute of "type: json"
//   attribute + named import         → SyntaxError: The requested module '../../../package.json' does
//                                      not provide an export named 'version'
//   attribute + default import       → 1 passed
//
// **Why the reader is Playwright's loader, spawned under `node`.** Plain `node` cannot read this
// package's raw `.ts` source at all, and neither substitute is honest:
//   · node's own ESM loader rejects the package's bundler-style specifiers (extensionless / directory
//     imports → ERR_UNSUPPORTED_DIR_IMPORT), and node 26 ships strip-only type stripping, which rejects
//     the parameter property in `engine/ecs/query.ts`;
//   · a bundler (`Bun.build`) resolves the JSON import itself, so the artifact never asks Node the
//     question — and measured 2026-08-25, bun's bundler *drops* the `with { type: "json" }` attribute when
//     the JSON is held external, so a bundled arm reds identically on fixed and broken source: it cannot
//     discriminate the fix at all;
//   · `jiti` rewrites the import to a CJS `require`, which is green on the defect.
// Playwright's per-file transform is the reader the field failure actually went through, it is already a
// devDependency here, and it hands the JSON import to Node untouched. The child is a real `node` process
// (`node node_modules/@playwright/test/cli.js`), never bun.
//
// **Scope, stated rather than implied.** The subject is the published specifier
// `@dylanebert/shallot/src/standard/loading/index.ts` (`package.json`'s `"./src/*"` export) and its
// transitive graph, which is where the JSON import lives. It is *not* the package barrel: `src/index.ts`
// reaches `standard/sear/codegen.ts`, which reads `GPUTextureUsage` at module scope, so a Node-side import
// of the root entry still dies with `ReferenceError: GPUTextureUsage is not defined` — a separate,
// pre-existing defect class (module-scope WebGPU globals) this arm does not claim to cover.
//
// Run: `bun run scripts/check-node-import.ts` (in `bun check`).

import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    realpathSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";

const REPO_ROOT = resolve(import.meta.dir, "..");
const NODE_MODULES = resolve(REPO_ROOT, "node_modules");
const PW_CLI = resolve(NODE_MODULES, "@playwright/test/cli.js");
/** the published specifier whose graph carries the JSON import (see the scope note above). */
const SUBJECT = "@dylanebert/shallot/src/standard/loading/index.ts";
/** real exported symbols the child must find — an import that resolves and exports nothing real is not a
 *  loading package (`testing.md` § Install gate: "a real symbol came through, not just that the import
 *  didn't throw"). They aren't *called*: they build DOM nodes and Node has no `document`. */
const EXPECTED_EXPORTS = ["shallotDark", "shallotLight", "minimalDark", "minimalLight"];

function fail(message: string, detail = ""): never {
    console.error(`✗ check-node-import: ${message}`);
    if (detail) console.error(detail);
    console.error(
        "\nNode ≥26 rejects both a bare JSON import and a named import off a JSON module. Import the\n" +
            'manifest as `import pkg from "…/package.json" with { type: "json" }` and read `pkg.version`.',
    );
    process.exit(1);
}

// No skip arm anywhere in this file: `node` and Playwright are hard prerequisites of this repo's browser
// tiers, and a gate that reports green where its subject never ran is the exact failure mode this arm
// exists to end. A missing prerequisite is a red.
if (!existsSync(PW_CLI)) fail(`Playwright's CLI is missing (${PW_CLI}) — run \`bun install\``);

const work = mkdtempSync(join(tmpdir(), "shallot-node-import-"));
try {
    // The fixture lives outside `node_modules` (Playwright's transform excludes that tree, and node's own
    // stripping refuses `.ts` under it), with the repo's `node_modules` symlinked in so both
    // `@playwright/test` and `@dylanebert/shallot` resolve — the workspace link realpaths back into
    // `packages/shallot`, i.e. the arm reads the same source the tarball ships.
    symlinkSync(NODE_MODULES, join(work, "node_modules"));
    // The arm asserts its subject: the workspace link must realpath back into `packages/shallot` (the
    // source the tarball ships). If root `node_modules/@dylanebert/shallot` ever resolves to an extracted
    // tarball or registry copy, the arm would green on a published artifact while the tree is broken —
    // fail instead of reading the wrong subject.
    const shallotPkg = join(REPO_ROOT, "packages/shallot");
    const resolvedSubject = realpathSync(join(NODE_MODULES, "@dylanebert/shallot"));
    if (resolvedSubject !== shallotPkg && !resolvedSubject.startsWith(shallotPkg + sep)) {
        fail(
            `the resolved subject realpaths to ${resolvedSubject}, not under ${shallotPkg} — the arm would read a published artifact, not the tree`,
        );
    }
    mkdirSync(join(work, "test"));
    // `"type": "module"` is load-bearing, not boilerplate: without it Playwright compiles the spec (and the
    // engine source it pulls in) to **CJS**, so the JSON import becomes a `require` that Node is perfectly
    // happy with — measured 2026-08-25, the arm went green on the unfixed source. ESM is also what every
    // real consumer of this package is (`packages/shallot/package.json` is `"type": "module"` itself).
    writeFileSync(
        join(work, "package.json"),
        `${JSON.stringify({ name: "shallot-node-import-arm", private: true, type: "module" }, null, 2)}\n`,
    );
    writeFileSync(
        join(work, "playwright.config.mjs"),
        // no `webServer`, no browser: the spec uses no `page` fixture, so nothing is launched.
        `export default { testDir: "./test", reporter: [["list"]], workers: 1, timeout: 60000 };\n`,
    );
    writeFileSync(
        join(work, "test", "node-import.spec.ts"),
        `import { expect, test } from "@playwright/test";\n` +
            `import * as subject from ${JSON.stringify(SUBJECT)};\n` +
            `test("real node imports ${SUBJECT}", () => {\n` +
            `    for (const key of ${JSON.stringify(EXPECTED_EXPORTS)}) {\n` +
            `        expect(typeof (subject as Record<string, unknown>)[key]).toBe("function");\n` +
            `    }\n` +
            `});\n`,
    );

    // Read the spawned `node`'s version *before* the run: a stale-PATH `node` that predates import
    // attributes (<20.10) must red with its own diagnostic rather than look like a source defect, and a
    // missing `node` gets the script's own message instead of an uncaught `Bun.spawnSync` throw.
    const versionProc = Bun.spawnSync(["node", "--version"], { stdout: "pipe", stderr: "pipe" });
    if (versionProc.exitCode !== 0 || !versionProc.stdout.toString().trim()) {
        fail(
            "`node` is missing or not on PATH — this arm spawns a real `node` binary, never bun",
            versionProc.stderr.toString().trim(),
        );
    }
    const nodeVersion = versionProc.stdout.toString().trim();
    const [major, minor] = nodeVersion.replace(/^v/, "").split(".").map(Number);
    if (major < 20 || (major === 20 && minor < 10)) {
        fail(
            `\`node\` ${nodeVersion} predates import-attribute support — the arm needs Node >= 20.10`,
        );
    }

    const proc = Bun.spawnSync(["node", PW_CLI, "test", "--config", "playwright.config.mjs"], {
        cwd: work,
        stdout: "pipe",
        stderr: "pipe",
    });
    const out = `${proc.stdout.toString()}${proc.stderr.toString()}`;
    // Assert the test *ran*, not merely that the process was happy: a module-load failure leaves
    // `playwright test --list` exiting 0 with "Total: 0 tests in 0 files" (measured 2026-08-25), so an
    // exit code alone is not a witness that the subject was ever imported (`checks.md`: "an arm whose
    // subject never launched reads clean on a predicate that checks only survivors").
    if (proc.exitCode !== 0 || !/\b1 passed\b/.test(out)) {
        fail(
            `real node could not import ${SUBJECT} (exit ${proc.exitCode})`,
            out.trim().split("\n").slice(-25).join("\n"),
        );
    }
    console.log(
        `✓ real node (${nodeVersion}) imports ${SUBJECT} with ${EXPECTED_EXPORTS.length} live exports`,
    );
} finally {
    rmSync(work, { recursive: true, force: true });
}
