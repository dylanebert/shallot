// Real-user install flow, sandboxed — the gate the dev symlink can't be. It packs the engine
// (`bun pm pack`, so it exercises the published `files` surface: src, bin, the audio `.wasm`, the
// schema) and a custom plugin library, installs both into a throwaway project via `bun install` (a real
// node_modules layout, not a workspace symlink), then runs every headless CLI flow against the installed
// engine: `shallot build` (manifest resolves an installed plugin by subpath + a local plugin, the wasm
// bundles), `shallot dev` (the server resolves the same + serves the wasm over /@fs), and `bun create
// shallot` (scaffold → install → build the starter). The packaging / resolution / asset failures the
// repo's own symlinked dev setup hides. Run: `bun run scripts/install-test.ts` (or `bun run test:install`).

import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    realpathSync,
    renameSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
    type ShaderArtifactSummary,
    skipReason,
    teardownBridge,
    type VerifyResult,
    verify,
} from "./verify";

const ENGINE_DIR = resolve(import.meta.dir, "../packages/shallot");
const WIDGET_DIR = resolve(import.meta.dir, "install-test/widget");
const CREATE_SHALLOT = resolve(import.meta.dir, "../packages/create-shallot/index.ts");
const CLI = "node_modules/@dylanebert/shallot/bin/cli.ts"; // the installed CLI, run as a real user would

const freePort = (): Promise<number> =>
    new Promise((res, rej) => {
        const s = createServer();
        s.on("error", rej);
        s.listen(0, "127.0.0.1", () => {
            const p = (s.address() as { port: number }).port;
            s.close(() => res(p));
        });
    });

function run(cmd: string[], cwd: string): { ok: boolean; out: string } {
    const p = Bun.spawnSync(cmd, { cwd, stdout: "pipe", stderr: "pipe" });
    return { ok: p.exitCode === 0, out: `${p.stdout.toString()}\n${p.stderr.toString()}` };
}

function pack(dir: string, dest: string): string {
    const r = run(["bun", "pm", "pack", "--destination", dest], dir);
    if (!r.ok) throw new Error(`pack ${dir} failed:\n${r.out}`);
    const tgz = readdirSync(dest).find((f) => f.endsWith(".tgz") && !f.startsWith("."));
    if (!tgz) throw new Error(`no tarball produced in ${dest}:\n${r.out}`);
    return join(dest, tgz);
}

/** vite colors its banner and its errors; match against the plain text so a TTY can't change a verdict. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: ESC is the SGR introducer — matching it is the point.
const strip = (s: string) => s.replaceAll(/\x1b\[[0-9;]*m/g, "");

async function waitFor(cond: () => Promise<boolean>, ms: number): Promise<boolean> {
    const deadline = Date.now() + ms;
    while (Date.now() < deadline) {
        if (await cond()) return true;
        await Bun.sleep(250);
    }
    return false;
}

const fails: string[] = [];
const check = (name: string, cond: boolean, detail = "") => {
    console.log(`  ${cond ? "✓" : "✗"} ${name}${detail ? ` — ${detail}` : ""}`);
    if (!cond) fails.push(name);
};

/** extract the page's diagnostic from a verify result so a red is legible, or name the absence as an
 *  instrument fault. The ejected boot arm was once dismissed as a flake because its detail printed `[]`
 *  — an empty `errors` array that a future reader can wave away. This surfaces the diagnostic wherever
 *  it landed (page errors, setup error, verdict checks, shader artifacts) and, when none carried one,
 *  reports a named instrument fault with the full field state rather than an empty container. The arm
 *  asserts `pass === true && booted === true && rendered === true`, so the instrument-fault string names
 *  which of those three predicates failed alongside the result's own field values — a reader is never
 *  at a dead end. On a pass the errors array is the detail (empty = no errors), never an instrument
 *  fault. */
export function verifyDiagnostic(result: VerifyResult | null): string {
    if (!result) return "no verify result";
    const errors = result.errors ?? [];
    if (errors.length > 0) return errors.join(" | ");
    if (result.error) return result.error;
    if (result.pass) return "pass";
    const failedChecks = (result.verdict?.checks ?? []).filter((c) => !c.ok);
    if (failedChecks.length > 0) return JSON.stringify(failedChecks);
    const artifacts = result.artifacts ?? [];
    const diagArtifacts = artifacts.filter(
        (a: ShaderArtifactSummary) => a.compilationError || (a.messages && a.messages.length > 0),
    );
    if (diagArtifacts.length > 0)
        return JSON.stringify(
            diagArtifacts.map((a) => ({
                label: a.label,
                compilationError: a.compilationError,
                messages: a.messages,
            })),
        );
    // Instrument fault: the result is red but carries no diagnostic text — no page errors, no setup
    // error, no failed verdict checks, no shader artifacts. Name which of the arm's three predicates
    // (pass===true, booted===true, rendered===true) failed and the full field state, so a reader can
    // see what the instrument saw rather than staring at an empty container.
    const failed: string[] = [];
    // `pass` is always false here — the `if (result.pass) return "pass"` guard above returned
    // early — so it always names a failed predicate without a comparison TS would flag as tautological.
    failed.push("pass===true");
    if (result.booted !== true) failed.push("booted===true");
    if (result.rendered !== true) failed.push("rendered===true");
    const fields: string[] = [
        `pass=${result.pass}`,
        `booted=${result.booted ?? "undefined"}`,
        `rendered=${result.rendered ?? "undefined"}`,
        `hardware=${result.hardware ?? "undefined"}`,
    ];
    if (result.verdict) {
        fields.push(`verdict.ok=${result.verdict.ok ?? "undefined"}`);
        fields.push(`verdict.checks=${result.verdict.checks?.length ?? 0}`);
    } else {
        fields.push("verdict=absent");
    }
    fields.push(result.memory !== undefined ? "memory=present" : "memory=absent");
    // the render probe — the pixel evidence behind the `rendered` verdict. A blank-render red (the
    // reading this exists for) carries its measurement here: samples taken, the last centre/corner RGB,
    // the spread against `structured`'s threshold, elapsed, and how the wait concluded. When no probe
    // was captured at all (a crash before the settle path ran), the fault stays named under its own
    // shape — never an empty container a future reader can dismiss.
    if (result.renderProbe) {
        const p = result.renderProbe;
        fields.push(
            `renderProbe={samples=${p.samples}, center=[${p.center ?? "null"}], corner=[${p.corner ?? "null"}], spread=${p.spread ?? "null"}, threshold=${p.threshold}, elapsed=${p.elapsed}ms, outcome=${p.outcome}}`,
        );
    } else {
        fields.push("renderProbe=absent");
    }
    return `instrument fault: verify red with no diagnostic (no page errors, no verdict checks, no shader artifacts); failed predicates: ${failed.join(", ")}; ${fields.join(", ")}`;
}

/** the `identity-check.ts` probe body — brand-checks the engine-built canary against the app's own
 * `typegpu` resolution (`GREEN`) and a second physical copy (`RED`); `ejectedFlow`, `identityFlow`, and
 * `pmIdentityFlow` all reuse it verbatim. `withPaths` adds the two resolved-path lines `pmIdentityFlow`
 * needs to name which manager diverged. */
function identityProbeScript(withPaths = false): string {
    return (
        `import { isTgpuFn } from "typegpu";\n` +
        `import { isTgpuFn as isTgpuFn2 } from "typegpu2";\n` +
        `import { tgslCanary } from "@dylanebert/shallot/runtime";\n` +
        `console.log("GREEN=" + isTgpuFn(tgslCanary));\n` +
        `console.log("RED=" + isTgpuFn2(tgslCanary));\n` +
        (withPaths
            ? `console.log("PATH_TYPEGPU=" + Bun.resolveSync("typegpu", import.meta.dir));\n` +
              `console.log("PATH_TYPEGPU2=" + Bun.resolveSync("typegpu2", import.meta.dir));\n`
            : "")
    );
}

// `bun create shallot` → install the packed engine → build — the brand-new-user path (the starter
// template is an index.html-free manifest project; this is the asserted form of `bun local`'s scaffold).
function createShallotFlow(work: string, engineTgz: string) {
    console.log("bun create shallot (scaffold → install → build the starter)…");
    const parent = join(work, "scaffold");
    mkdirSync(parent, { recursive: true });
    const created = run(["bun", CREATE_SHALLOT, "starter-app"], parent);
    check(
        "create-shallot scaffolds a project",
        created.ok,
        created.ok ? "" : created.out.slice(-400),
    );
    const proj = join(parent, "starter-app");
    if (!existsSync(join(proj, "package.json"))) return;
    // a real user installs the published engine; here, the packed tarball stands in
    const pkg = JSON.parse(readFileSync(join(proj, "package.json"), "utf8"));
    pkg.dependencies["@dylanebert/shallot"] = `file:${engineTgz}`;
    writeFileSync(join(proj, "package.json"), `${JSON.stringify(pkg, null, 2)}\n`);
    const inst = run(["bun", "install"], proj);
    check("the scaffolded starter installs", inst.ok, inst.ok ? "" : inst.out.slice(-400));
    if (!inst.ok) return;
    const built = run(["bun", CLI, "build", "."], proj);
    check("the scaffolded starter builds", built.ok, built.ok ? "" : built.out.slice(-600));
    check("starter build produced dist/index.html", existsSync(join(proj, "dist", "index.html")));

    // the emitted docs must point an agent at the installed engine (node_modules), not a repo URL, and
    // every path they name must resolve inside the freshly installed project.
    const doc = readFileSync(join(proj, "AGENTS.md"), "utf8");
    check(
        "scaffold docs point at node_modules, not a GitHub URL",
        /node_modules\/@dylanebert\/shallot\/AGENTS\.md/.test(doc) &&
            /node_modules\/@dylanebert\/shallot\/examples\/AGENTS\.md/.test(doc) &&
            !/github\.com\/dylanebert\/shallot/.test(doc),
    );
    for (const rel of [
        "node_modules/@dylanebert/shallot/AGENTS.md",
        "node_modules/@dylanebert/shallot/examples/AGENTS.md",
    ]) {
        check(`the scaffold's ${rel} pointer resolves`, existsSync(join(proj, rel)));
    }
    check(
        "scaffold docs name `shallot verify` as the verification step",
        /shallot verify/.test(doc),
    );
    check(
        "scaffold docs name `shallot recipe` as the copy-out command",
        /shallot recipe/.test(doc),
    );
    // runnable command lines standardize on `bunx shallot <cmd>` — a bare `shallot <cmd>` at a line or
    // `&&`-chain start only resolves when globally linked (check-docs.ts guards the repo's own docs; this
    // guards the docs create-shallot emits, which check-docs can't scan statically). Prose naming the CLI
    // surface (backtick-preceded) is unaffected.
    check(
        "scaffold docs carry no bare `shallot <cmd>` runnable line",
        !/(^|&&)\s*shallot\s+(dev|build|run|verify|recipe)\b/m.test(doc),
    );

    // the shipped verify gate, run as an installed agent would: --help is a clean exit, and a project
    // with no playwright gets the distinct exit 3 + the actionable install command (a browser run itself
    // is display/GPU-gated in WSL — not asserted here).
    const help = run(["bun", CLI, "verify", "--help"], proj);
    check("shallot verify --help exits 0", help.ok, help.ok ? "" : help.out.slice(-200));
    const noPw = Bun.spawnSync(["bun", CLI, "verify", "."], {
        cwd: proj,
        stdout: "pipe",
        stderr: "pipe",
    });
    check(
        "shallot verify exits 3 with an install remedy when playwright is absent",
        noPw.exitCode === 3 &&
            /playwright install chromium/.test(
                `${noPw.stdout.toString()}\n${noPw.stderr.toString()}`,
            ),
        `exit ${noPw.exitCode}`,
    );

    // the TS18003 trap: deleting the demo plugin (its comment invites it) empties src/ but must not break
    // the scaffold's documented `bunx tsc --noEmit` — the env.d.ts anchor keeps `include: ["src"]` matched.
    rmSync(join(proj, "src", "spin.ts"));
    const tsc = run(["bunx", "tsc", "--noEmit"], proj);
    check(
        "tsc --noEmit stays green with an emptied src/ (no TS18003)",
        tsc.ok,
        tsc.ok ? "" : tsc.out.slice(-400),
    );
}

// `shallot recipe <name> <dir>` copies a recipe out of the installed package into a runnable project;
// the copy's engine dep is version-pinned by the CLI, so here we point it back at the packed tarball
// (as a real user's registry install would resolve) and build it headlessly. Guards the whole copy-out
// path: recipe present in the pack, CLI copies it, the pinned dep installs, the project builds.
function recipeFlow(work: string, engineTgz: string, sandbox: string) {
    console.log("shallot recipe (copy a recipe out → install → build)…");
    const dest = join(work, "recipe-out", "joints");
    const copied = run(["bun", CLI, "recipe", "joints", dest], sandbox);
    check("shallot recipe copies a recipe out", copied.ok, copied.ok ? "" : copied.out.slice(-400));
    if (!existsSync(join(dest, "package.json"))) return;
    // the CLI pins the engine to the installed version; swap it for the packed tarball the test has
    const pkg = JSON.parse(readFileSync(join(dest, "package.json"), "utf8"));
    check(
        "the copy's engine dep is version-pinned (no workspace:*)",
        typeof pkg.dependencies?.["@dylanebert/shallot"] === "string" &&
            !pkg.dependencies["@dylanebert/shallot"].startsWith("workspace:"),
        String(pkg.dependencies?.["@dylanebert/shallot"]),
    );
    pkg.dependencies["@dylanebert/shallot"] = `file:${engineTgz}`;
    writeFileSync(join(dest, "package.json"), `${JSON.stringify(pkg, null, 4)}\n`);
    const inst = run(["bun", "install"], dest);
    check("the copied recipe installs", inst.ok, inst.ok ? "" : inst.out.slice(-400));
    if (!inst.ok) return;

    // the copy-out synthesizes the standalone scaffold the monorepo recipe lacks: the agent-surface
    // pointer (AGENTS.md, imported by CLAUDE.md) an installed harness follows, and a tsconfig. Assert the
    // pointer names node_modules and resolves in the real install (the reach the distribution decision
    // rests on), and that CLAUDE.md reaches it through the import rather than copying it.
    check(
        "the copied recipe's AGENTS.md points at node_modules",
        /node_modules\/@dylanebert\/shallot\/AGENTS\.md/.test(
            readFileSync(join(dest, "AGENTS.md"), "utf8"),
        ),
    );
    const claude = readFileSync(join(dest, "CLAUDE.md"), "utf8");
    check(
        "the copied recipe's CLAUDE.md imports AGENTS.md",
        claude.split("\n")[0] === "@AGENTS.md",
    );
    // the import expands only for a session rooted in the recipe itself; opened from a parent directory
    // the line stays literal text, and the prose pointer is the reader's only route to the contract
    check(
        "the copied recipe's CLAUDE.md also names AGENTS.md in prose",
        claude.replace("@AGENTS.md", "").includes("AGENTS.md"),
    );
    check(
        "the copied recipe's engine-pointer path resolves after install",
        existsSync(join(dest, "node_modules/@dylanebert/shallot/AGENTS.md")),
    );
    check(
        "the copied recipe carries a standalone tsconfig",
        existsSync(join(dest, "tsconfig.json")),
    );

    const built = run(["bun", CLI, "build", "."], dest);
    check("the copied recipe builds", built.ok, built.ok ? "" : built.out.slice(-600));
    check(
        "copied recipe build produced dist/index.html",
        existsSync(join(dest, "dist", "index.html")),
    );
}

// MIGRATION.md's "An ejected Vite project" recipe, read straight out of the doc rather than
// re-typed — the failure this flow exists to catch is the recipe itself shipping broken, and a
// hand-written copy could silently diverge from what a consumer actually pastes. Anchored on the
// `// vite.config.ts` comment the doc's fenced block opens with, through the fence's closing ``` .
function ejectedViteConfig(): string {
    const doc = readFileSync(resolve(import.meta.dir, "../packages/shallot/MIGRATION.md"), "utf8");
    const start = doc.indexOf("// vite.config.ts");
    if (start === -1) throw new Error("MIGRATION.md's ejected Vite recipe marker not found");
    const fenceEnd = doc.indexOf("\n```", start);
    if (fenceEnd === -1) throw new Error("MIGRATION.md's ejected Vite recipe has no closing fence");
    return doc.slice(start, fenceEnd);
}

// a true ejected shape (a real consumer's ejected shape, not the CLI's zero-config path): the
// project owns its own index.html + vite.config.ts, so `shallot verify` roots a plain vite server
// there (`bin/verify.ts`'s `serveEjected`) and vite auto-loads the config from disk — nothing merges
// or overrides it, so this is the config a real boot actually runs. MIGRATION.md's block is the
// vite.config only; the manifest, scene, index.html, and entry below are the harness every real
// ejected consumer supplies around it, the same shape the CLI's own synthesized entry uses.
//
// Against a registry install, Vite's default `configLoader` ("bundle") loads vite.config.ts by
// spawning a real `node` subprocess, which applies no TS transform to a `node_modules` import — the same
// class 5b-2f-4b found for Playwright's config loader (`harness/browser.ts`'s header comment). `./vite`
// (and `./harness/browser`) are the two exports whose only consumption context is Node, so they ship a
// compiled `dist/*.js` default alongside the `.ts` source `types` (`exports.md`), the reason this recipe
// boots at all against a packed tarball.
async function ejectedFlow(work: string, engineTgz: string) {
    console.log("ejected Vite project (MIGRATION.md's recipe, booted verbatim)…");
    const proj = join(work, "ejected");
    mkdirSync(join(proj, "src"), { recursive: true });
    mkdirSync(join(proj, "scenes"), { recursive: true });
    writeFileSync(
        join(proj, "package.json"),
        `${JSON.stringify(
            {
                name: "ejected-sandbox",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:${engineTgz}`,
                    typegpu: "~0.12.0",
                    // a second physical copy for the identity probe below — see `identityFlow`'s comment
                    typegpu2: "npm:typegpu@~0.12.0",
                },
                devDependencies: { vite: "^8.0.0", "unplugin-typegpu": "~0.12.1" },
            },
            null,
            2,
        )}\n`,
    );
    const config = ejectedViteConfig();
    writeFileSync(join(proj, "vite.config.ts"), `${config}\n`);
    writeFileSync(
        join(proj, "shallot.json"),
        `${JSON.stringify({ scene: "scenes/main.scene", plugins: { Orbit: true } }, null, 2)}\n`,
    );
    writeFileSync(
        join(proj, "scenes", "main.scene"),
        `<scene>\n    <a ambient-light="color: 0xd0dcec; intensity: 0.5" />\n    <a directional-light="direction: -0.4 -1 -0.55; color: 0xfff4e0; intensity: 1.1" />\n    <a camera sear orbit="distance: 5; yaw: 0.6; pitch: 0.25" transform />\n    <a part transform="pos: 0 0 0" color="rgba: 0.85 0.55 0.35 1" />\n</scene>\n`,
    );
    writeFileSync(
        join(proj, "index.html"),
        `<!doctype html>\n<html lang="en">\n<head><meta charset="UTF-8" /></head>\n<body>\n<canvas id="canvas" style="display:block;width:100vw;height:100vh"></canvas>\n<script type="module" src="/src/main.ts"></script>\n</body>\n</html>\n`,
    );
    writeFileSync(
        join(proj, "src", "main.ts"),
        `import { run } from "@dylanebert/shallot";\nimport project from "virtual:project";\nawait run({ plugins: project.plugins, scene: project.scene ?? undefined, defaults: false, capacity: project.capacity ?? undefined });\n`,
    );

    console.log("bun install (the ejected project's own deps)…");
    const install = run(["bun", "install"], proj);
    check("the ejected project installs", install.ok, install.ok ? "" : install.out.slice(-600));
    if (!install.ok) return;

    // display-independent, so it runs even where the real-browser rung below is skipped: a plain
    // `node` process dynamic-importing both compiled tooling exports is exactly the resolution shape
    // Playwright's/Vite's own config loader uses (`ERR_UNKNOWN_FILE_EXTENSION` on raw `.ts`), with no
    // GPU involved — pure module resolution. Checks a real symbol came through, not just that the
    // import didn't throw (5b-2f-6 Locked decision, "Raw `.ts` exports are the runtime contract").
    console.log("node module resolution (the two compiled tooling exports, no browser needed)…");
    writeFileSync(
        join(proj, "node-resolve-check.mjs"),
        `import { projectPlugin } from "@dylanebert/shallot/vite";\n` +
            `import { REAL_GPU_LAUNCH } from "@dylanebert/shallot/harness/browser";\n` +
            `if (typeof projectPlugin !== "function") throw new Error("projectPlugin: not a function");\n` +
            `if (typeof REAL_GPU_LAUNCH?.channel !== "string") throw new Error("REAL_GPU_LAUNCH: no channel");\n` +
            `console.log("NODE_RESOLVE_OK " + REAL_GPU_LAUNCH.channel);\n`,
    );
    const nodeResolve = run(["node", "node-resolve-check.mjs"], proj);
    check(
        "a plain Node import resolves both compiled tooling exports (playwright/vite config-loader shape)",
        nodeResolve.ok && nodeResolve.out.includes("NODE_RESOLVE_OK"),
        nodeResolve.ok ? nodeResolve.out.trim().slice(-200) : nodeResolve.out.slice(-500),
    );

    // the ejected fixture's own arm of `identityFlow`'s brand check (see its comment above): a
    // materially different install shape (own vite.config, no CLI-synthesized entry), so the probe
    // runs here too rather than trusting the sandbox arm to stand in for it. `bun`, not `node` — this
    // one imports `@dylanebert/shallot/runtime`, which ships raw `.ts` (no compiled arm), and needs no
    // GPU either, so it still runs before the skip below.
    console.log("typegpu peer identity (ejected fixture, brand check)…");
    writeFileSync(join(proj, "identity-check.ts"), identityProbeScript());
    const identity = run(["bun", "identity-check.ts"], proj);
    check(
        "ejected fixture: the app's own typegpu resolution brands the canary true (same physical copy)",
        identity.ok && /GREEN=true/.test(identity.out),
        identity.ok ? identity.out.trim().slice(-200) : identity.out.slice(-400),
    );
    check(
        "ejected fixture: a second physical copy brands the same canary false",
        identity.ok && /RED=false/.test(identity.out),
        identity.ok ? identity.out.trim().slice(-200) : identity.out.slice(-400),
    );

    const skip = skipReason();
    if (skip) {
        console.log(`  · skipped (needs native hardware: ${skip})`);
        return;
    }

    console.log("real browser boot — the recipe as documented, not as known-good…");
    const green = await verify(proj, ["--timeout", "30000"], true);
    check(
        "the documented ejected recipe boots and warms its pipelines as written",
        green?.pass === true && green.booted === true && green.rendered === true,
        verifyDiagnostic(green),
    );

    // No red-proof mutant here (found investigating 5b-2f-6, after the check above went green for the
    // first time): stripping this recipe's `optimizeDeps` line does NOT reproduce 5b-2f-5's
    // prebundle-before-transform defect for this boot path — verified three consecutive real-hardware
    // runs, on both this minimal scene and one with a direct `typegpu/data` import (a consumer
    // importing a typegpu subpath directly). A controlled comparison against a genuine zero-config `shallot dev` project — identical
    // scene, `devConfig`'s exclusion removed the same way — DOES reproduce the exact `Cannot resolve
    // struct cast from 'vertexVsOut' to 'vertexVs_Output'` error 5b-2f-5 diagnosed, so the defect is
    // real and the boot mechanism is the variable, not the scene: `serveEjected` (this flow, gym,
    // flows, showcase) lets Vite auto-load a real on-disk `vite.config.ts` against a real on-disk
    // `index.html`, where `devConfig` (`bin/dev.ts`) sets `configFile: false` and serves a
    // middleware-synthesized entry instead — evidently a different dependency-discovery path. Left
    // unexplained rather than routed around; the exclusion stays documented in MIGRATION.md as the
    // defensive default (it IS load-bearing for the zero-config path, red-proven at 5b-2f-5 and pinned
    // by `toolchain.test.ts`), but asserting it must break an ejected boot would pin a false invariant.
    await teardownBridge();
}

// The engine's shaders are TGSL: JS function bodies transpiled to WGSL at BUILD time by
// `unplugin-typegpu`, with no runtime fallback. A consumer whose bundler lacks the transform gets no
// metadata at all — silently wrong shaders, NaN out of CPU-called kernels — and nothing about the
// packaging surface shows it, so this flow is the only place the contract is checked end to end. It
// runs an engine TGSL fn through `tgpu.resolve` in the installed project and asserts the emitted
// WGSL, both with the transform (green) and without it (red — the metadata really does come from the
// plugin, not from anything shipped in the tarball).
// the canary's transpiled body, quote-agnostic (a minifier picks the quote style). NOT
// `__TYPEGPU_META__` — typegpu's own runtime reads that name, so it ships either way and a check on it
// passes with no transform at all (measured, 2026-07-29).
const TRANSPILED =
    /["'`]?body["'`]?:\[0,\[\[10,\[1,["'`]x["'`],["'`]\+["'`],\[5,["'`]1["'`]\]\]\]\]\]/;

// The peer-identity probe: typegpu's brand symbols are per-copy `Symbol(...)`, never `Symbol.for`
// (`shared/symbols.js`), so `isTgpuFn(x)` imported from a consumer's own resolution of `typegpu`
// returns true for an engine-built fn iff both resolve one physical copy — the property the 0.9.0
// break actually turned on. Pure module resolution + a symbol check, no GPU involved, so this runs (and
// is asserted) before any `skipReason()` guard: a display-less host still exercises it (testing.md
// "Install gate", "a display-gated rung owes a display-independent sibling"). `typegpu2` is a genuine
// second physical copy — an alias install of the identical version (`npm:typegpu@~0.12.0`) landing in
// its own `node_modules/typegpu2`, a distinct module-graph evaluation with its own `Symbol()` calls, not
// a re-export of the first. The red arm is the whole point: a probe green on first run with no witnessed
// red proves nothing.
function identityFlow(work: string, engineTgz: string) {
    console.log("typegpu peer identity (brand check, red-first)…");
    const proj = join(work, "identity");
    mkdirSync(proj, { recursive: true });
    writeFileSync(
        join(proj, "package.json"),
        `${JSON.stringify(
            {
                name: "identity-sandbox",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:${engineTgz}`,
                    typegpu: "~0.12.0",
                    typegpu2: "npm:typegpu@~0.12.0",
                },
            },
            null,
            2,
        )}\n`,
    );
    const install = run(["bun", "install"], proj);
    check(
        "the identity fixture installs (real + aliased typegpu copy)",
        install.ok,
        install.ok ? "" : install.out.slice(-600),
    );
    if (!install.ok) return;

    writeFileSync(join(proj, "identity-check.ts"), identityProbeScript());
    const result = run(["bun", "identity-check.ts"], proj);
    check(
        "the app's own typegpu resolution brands the engine-built canary true (same physical copy)",
        result.ok && /GREEN=true/.test(result.out),
        result.ok ? result.out.trim().slice(-200) : result.out.slice(-400),
    );
    // a second physical copy must read false — this is the observation the boot checks above it can't make
    check(
        "a second physical copy of typegpu brands the same canary false (peer identity actually observed)",
        result.ok && /RED=false/.test(result.out),
        result.ok ? result.out.trim().slice(-200) : result.out.slice(-400),
    );
}

// Stage 3: the browser-side arm of the peer-identity probe. Node-side probes (identityFlow above,
// pmIdentityFlow below) can't see Vite's own resolution — the dependency scanner rewrites the module
// graph on real page load, a rewrite no `bun`/`node` process ever performs. This is the ordering-
// independent observation of the standing ejected-path question (spec "Approach" stage 3): do the app's
// own `isTgpuFn` and the engine-built `tgslCanary` meet (one physical typegpu copy) or diverge (two),
// under the normal config and under the perturbation that reproduced the class for real (5b-2f-5:
// stripping the zero-config path's `optimizeDeps` exclusion so `typegpu` gets prebundled ahead of the
// transform). The zero-config sandbox's own perturbed arm turned out unrealizable in this harness (see the
// comment at its call site in `identityBrowserFlow`, below) — the ejected fixture is where the
// perturbation is both real and observed.
//
// The own-vs-canary fixtures below carry no second `typegpu` copy: a genuine duplicate trips the
// engine's own pre-existing `checkTgsl` write-counter (`engine/runtime/gpu.ts`) inside `requestGPU`,
// which `App.build()` calls before any plugin's `initialize` hook runs — measured directly (a first
// attempt bundling the red-proof's `typegpu2` copy into the same page crashed `run()` before the
// harness ever installed, `window.__harness` never defined, every "published a verdict" check failing
// with `checkTgsl`'s own "Two copies of typegpu are loaded" as the page error). That crash is the
// engine working, not the instrument — so the red-proof (a real second copy reading false) lives in its
// own fixture below that never calls `run()` at all, and these two never carry `typegpu2`.
const IDENTITY_OWN_IMPORTS =
    `import { isTgpuFn } from "typegpu";\n` +
    `import { tgslCanary } from "@dylanebert/shallot/runtime";\n`;

// The `prebundled` line is the perturbation's own control, computed in-page rather than read off disk
// afterward. Reading `node_modules/.vite/deps/_metadata.json` from the driving process was tried first
// and is blind: this plugin loads off the manifest (invisible to Vite's static pre-scan), so `typegpu`
// is a runtime-*discovered* dep whose commit is a debounced (~100ms) esbuild rename — `verify`'s own
// `server.close()`, called the instant the harness's forced-ready flips true, cancels that in-flight
// commit before it lands, independent of whether the perturbation reached the server (measured
// 2026-08-10: a standalone repro outside this harness's fast-teardown path writes the manifest
// correctly within ~1s; inside it, both arms read back empty). Self-fetching `import.meta.url` right
// after module eval is equally blind, for the same race: the FIRST transform of this very file is
// served optimistically off the raw specifier, before Vite's import-analysis has registered `typegpu`
// as discovered — the rewrite to `.vite/deps/typegpu` only appears on a second read, once the debounced
// re-optimize has actually run (measured 2026-08-10: an immediate self-fetch reads the pre-discovery
// raw path even under the perturbation). So poll the self-fetch a few times over ~3s — short enough
// that it can't be mistaken for waiting on a render, since it resolves before `run()`'s pipeline warm
// (the thing that may never finish) is even reached.
function identityOwnBody(publish: string): string {
    return (
        `const own = isTgpuFn(tgslCanary);\n` +
        // split across two literals — a self-fetch reads THIS file's own source, and the whole marker
        // written as one literal would always self-match (found red-handed: the first version of this
        // check read prebundled=true unconditionally, because the search string itself is verbatim text
        // inside the file it searches). Split, the two halves only join into the real marker at runtime.
        `const marker = "/node_modules/.vite/deps/" + "typegpu";\n` +
        `let prebundled = false;\n` +
        `for (let i = 0; i < 15 && !prebundled; i++) {\n` +
        `    if (i > 0) await new Promise((r) => setTimeout(r, 200));\n` +
        `    const selfSrc = await fetch(import.meta.url).then((r) => r.text()).catch(() => "");\n` +
        `    prebundled = selfSrc.includes(marker);\n` +
        `}\n` +
        `const verdict = {\n` +
        `    ok: true,\n` +
        `    checks: [\n` +
        `        { name: "app resolution brands the engine canary true (peer identity)", ok: own, detail: "own=" + own },\n` +
        `        { name: "typegpu import resolved through the prebundle path (self-read, the perturbation's control)", ok: prebundled, detail: "prebundled=" + prebundled },\n` +
        `    ],\n` +
        `};\n` +
        `${publish}\n`
    );
}

// the zero-config manifest shape: a real page.ts entry is synthesized by the CLI (build.ts/dev.ts), not
// controllable from here, so the only injection point is a local plugin's `initialize` — run before
// scene parse, itself a side effect of `run()`'s internal build sequence that survives `run()`'s promise
// later rejecting (a JS engine never rolls back a completed synchronous side effect on a later throw).
// The identity read itself happens even earlier, at module-eval time (the top-level `const own =` line,
// evaluated when the plugin module loads) — before `initialize` is even called — so it reflects
// whichever physical module Vite resolved regardless of anything that crashes afterward.
function identityPluginSource(): string {
    return (
        `import { installHarness } from "@dylanebert/shallot/harness";\n` +
        IDENTITY_OWN_IMPORTS +
        `\n` +
        identityOwnBody("") +
        `const IdentityPlugin = {\n` +
        `    name: "Identity",\n` +
        `    initialize(state) {\n` +
        `        const harness = installHarness(state);\n` +
        `        // the read above needs no drawn frame, and under the perturbation the pipeline may\n` +
        `        // never warm at all — force ready so verify's wait loop reads the verdict regardless.\n` +
        `        Object.defineProperty(harness, "ready", { value: true, configurable: true });\n` +
        `        harness.run = async () => verdict;\n` +
        `    },\n` +
        `};\n` +
        `export default IdentityPlugin;\n`
    );
}

function writeIdentityZeroConfig(dir: string, engineTgz: string) {
    mkdirSync(join(dir, "src"), { recursive: true });
    mkdirSync(join(dir, "scenes"), { recursive: true });
    writeFileSync(
        join(dir, "package.json"),
        `${JSON.stringify(
            {
                name: "identity-browser-sandbox",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:${engineTgz}`,
                    typegpu: "~0.12.0",
                },
            },
            null,
            2,
        )}\n`,
    );
    writeFileSync(
        join(dir, "shallot.json"),
        `${JSON.stringify(
            { scene: "scenes/main.scene", plugins: { Identity: "./src/identity-plugin" } },
            null,
            2,
        )}\n`,
    );
    // deliberately left unframed (camera at the origin, inside the part) unlike the two boot fixtures:
    // this arm reads `verdict.checks` and never `rendered`, so no pixel is asserted here. Do not copy this
    // scene shape into an arm that does — an origin camera sees only the cleared background.
    writeFileSync(
        join(dir, "scenes", "main.scene"),
        `<scene>\n    <a ambient-light="intensity: 0.6" />\n    <a camera sear transform />\n    <a part transform color="rgba: 0.8 0.5 0.3" />\n</scene>\n`,
    );
    writeFileSync(join(dir, "src", "identity-plugin.ts"), identityPluginSource());
}

// the ejected shape: the fixture owns its entry, so the brand check publishes `window.__harness`
// directly, ahead of calling `run()` — the read needs no drawn frame, and under the perturbation `run()`
// (or the pipeline it warms) may never resolve at all.
//
// No `shallot.json`/`.scene` in this fixture — deliberately. `bin/verify.ts`'s `bootArm` picks the CLI's
// synthesized-entry "project" arm whenever `isProject()` sees either one, unconditionally over an
// on-disk `index.html` (`bootArm(true, true) === "project"`, pinned in `verify.test.ts`), which serves
// its OWN generated `index.html` (`synthIndexPlugin`'s `configureServer` middleware wins the "/" route
// ahead of Vite's static file serving) — this fixture's real `index.html`/`vite.config.ts`/`main.ts`
// never load at all, silently. Found by instrumenting: `bootArm(true, true)`'s synthesized page carried
// no marker this fixture's own `index.html` writes, and the probe's module-eval console lines never
// appeared. `projectPlugin(".")` (this fixture's `vite.config.ts`, verbatim from MIGRATION.md) reads a
// missing `shallot.json` as the documented empty-manifest fallback (MIGRATION.md: "omit it and
// `virtual:project` resolves to an empty manifest") — default engine plugins, no scene — which is fine:
// this probe's brand check runs before `run()` is even called and needs no rendered scene.
function identityEjectedEntrySource(): string {
    return (
        `import { run } from "@dylanebert/shallot";\n` +
        IDENTITY_OWN_IMPORTS +
        `import project from "virtual:project";\n` +
        `\n` +
        identityOwnBody("window.__harness = { ready: true, run: async () => verdict };") +
        `run({ plugins: project.plugins, scene: project.scene ?? undefined, defaults: false, capacity: project.capacity ?? undefined }).catch(() => {});\n`
    );
}

function writeIdentityEjected(dir: string, engineTgz: string, configText: string) {
    mkdirSync(join(dir, "src"), { recursive: true });
    writeFileSync(
        join(dir, "package.json"),
        `${JSON.stringify(
            {
                name: "identity-browser-ejected",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:${engineTgz}`,
                    typegpu: "~0.12.0",
                },
                devDependencies: { vite: "^8.0.0", "unplugin-typegpu": "~0.12.1" },
            },
            null,
            2,
        )}\n`,
    );
    writeFileSync(join(dir, "vite.config.ts"), `${configText}\n`);
    writeFileSync(
        join(dir, "index.html"),
        `<!doctype html>\n<html lang="en">\n<head><meta charset="UTF-8" /></head>\n<body>\n<canvas id="canvas" style="display:block;width:100vw;height:100vh"></canvas>\n<script type="module" src="/src/main.ts"></script>\n</body>\n</html>\n`,
    );
    writeFileSync(join(dir, "src", "main.ts"), identityEjectedEntrySource());
}

// the red-proof fixture: a genuine second physical copy must brand the canary false — the witnessed red
// that proves the browser-side instrument (not just the node-side one, stage 1) can produce a false
// read. Ejected-shaped but never calls `run()` — no App boot, so `checkTgsl`'s duplicate-write counter
// never runs (it fires only inside `requestGPU`, itself only reachable from `App.build()`), and the two
// real physical copies this fixture needs can safely coexist with the engine's own peer `typegpu` copy
// on one page. `isTgpuFn`/`tgslCanary` need no App/device either (proven already: `identityFlow`'s
// node-side probe imports the same symbols under plain `bun`, no GPU).
function identityRedProofEntrySource(): string {
    return (
        `import { isTgpuFn } from "typegpu";\n` +
        `import { isTgpuFn as isTgpuFn2 } from "typegpu2";\n` +
        `import { tgslCanary } from "@dylanebert/shallot/runtime";\n` +
        `\n` +
        `const own = isTgpuFn(tgslCanary);\n` +
        `const distinct = isTgpuFn2(tgslCanary);\n` +
        `window.__harness = {\n` +
        `    ready: true,\n` +
        `    run: async () => ({\n` +
        `        ok: true,\n` +
        `        checks: [\n` +
        `            { name: "app resolution brands the engine canary true (peer identity)", ok: own, detail: "own=" + own },\n` +
        `            { name: "a distinct second copy brands the same canary false (red-proof)", ok: !distinct, detail: "distinct=" + distinct },\n` +
        `        ],\n` +
        `    }),\n` +
        `};\n`
    );
}

function writeIdentityRedProof(dir: string, engineTgz: string, configText: string) {
    mkdirSync(join(dir, "src"), { recursive: true });
    writeFileSync(
        join(dir, "package.json"),
        `${JSON.stringify(
            {
                name: "identity-browser-red-proof",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:${engineTgz}`,
                    typegpu: "~0.12.0",
                    typegpu2: "npm:typegpu@~0.12.0",
                },
                devDependencies: { vite: "^8.0.0", "unplugin-typegpu": "~0.12.1" },
            },
            null,
            2,
        )}\n`,
    );
    writeFileSync(join(dir, "vite.config.ts"), `${configText}\n`);
    writeFileSync(
        join(dir, "index.html"),
        `<!doctype html>\n<html lang="en">\n<head><meta charset="UTF-8" /></head>\n<body>\n<canvas id="canvas" style="display:block;width:100vw;height:100vh"></canvas>\n<script type="module" src="/src/main.ts"></script>\n</body>\n</html>\n`,
    );
    writeFileSync(join(dir, "src", "main.ts"), identityRedProofEntrySource());
}

// strip `typegpu` from MIGRATION.md's ejected `optimizeDeps.exclude` line — the perturbation that
// reproduced the class on the zero-config path (5b-2f-5). Throws rather than silently no-op if the doc's
// wording moves, so a doc edit reds this loud instead of quietly testing the unperturbed config twice.
function stripTypegpuExclude(config: string): string {
    const line = `optimizeDeps: { exclude: ["@dylanebert/shallot", "typegpu"] },`;
    if (!config.includes(line)) {
        throw new Error(
            "MIGRATION.md's ejected optimizeDeps line didn't match the expected text — update stripTypegpuExclude",
        );
    }
    return config.replace(line, `optimizeDeps: { exclude: ["@dylanebert/shallot"] },`);
}

// read the "own vs canary" check the fixtures above publish. Reported for every arm; additionally
// asserted true under the normal config, where stage 1/2's node-side probes already established a
// single clean physical copy — the perturbation arm is recorded, not asserted, since it's the open
// question this stage answers.
function checkIdentityVerdict(label: string, result: VerifyResult | null, expectOwnTrue: boolean) {
    const own = (result?.verdict?.checks ?? []).find((c) => c.name.startsWith("app resolution"));
    check(
        `${label}: the browser probe published a verdict`,
        own !== undefined,
        own
            ? ""
            : result
              ? JSON.stringify(result.errors ?? result.error ?? result)
              : "no verify result",
    );
    if (!own) return;
    check(
        `${label}: app resolution vs engine canary — ${expectOwnTrue ? "expect single identity" : "recorded, not asserted (perturbation arm)"}`,
        expectOwnTrue ? own.ok === true : true,
        own.detail ?? "",
    );
}

// The perturbation's own control. `optimizeDeps.exclude` is a claim made to Vite's config; nothing in
// the brand verdict says whether Vite acted on it, and the whole perturbed arm is unreadable without
// that — "the two identities meet harmlessly" and "the strip never reached the dev server" publish the
// identical green (a green check can pin nothing). Reads the `prebundled` check the
// fixtures' `identityOwnBody` publishes (self-read of the transformed source, not a post-hoc disk read —
// see that function's comment for why the disk manifest is blind under this harness's fast teardown).
function checkPrebundled(label: string, result: VerifyResult | null, expect: boolean) {
    const prebundled = (result?.verdict?.checks ?? []).find((c) =>
        c.name.startsWith("typegpu import resolved through the prebundle path"),
    );
    check(
        `${label}: the browser probe published a prebundle read`,
        prebundled !== undefined,
        prebundled
            ? ""
            : result
              ? JSON.stringify(result.errors ?? result.error ?? result)
              : "no verify result",
    );
    if (!prebundled) return;
    check(
        `${label}: typegpu ${expect ? "IS" : "is NOT"} prebundled (the perturbation's control)`,
        prebundled.ok === expect,
        prebundled.detail ?? "",
    );
}

async function identityBrowserFlow(work: string, engineTgz: string) {
    const skip = skipReason();
    if (skip) {
        console.log(
            `typegpu peer identity (browser, both boot paths)… skipped (needs native hardware: ${skip})`,
        );
        return;
    }

    console.log("typegpu peer identity (browser, red-proof fixture)…");
    const redProofDir = join(work, "identity-browser-red-proof");
    writeIdentityRedProof(redProofDir, engineTgz, ejectedViteConfig());
    const redProofInstall = run(["bun", "install"], redProofDir);
    check(
        "identity-browser (red-proof): the fixture installs",
        redProofInstall.ok,
        redProofInstall.ok ? "" : redProofInstall.out.slice(-600),
    );
    if (redProofInstall.ok) {
        const result = await verify(redProofDir, ["--timeout", "30000"], true);
        const checks = result?.verdict?.checks ?? [];
        const own = checks.find((c) => c.name.startsWith("app resolution"));
        const distinct = checks.find((c) => c.name.startsWith("a distinct second copy"));
        check(
            "red-proof: the browser probe published a verdict",
            own !== undefined && distinct !== undefined,
            own && distinct
                ? ""
                : result
                  ? JSON.stringify(result.errors ?? result.error ?? result)
                  : "no verify result",
        );
        if (own && distinct) {
            check(
                "red-proof: app resolution brands the canary true (same physical copy)",
                own.ok === true,
                own.detail ?? "",
            );
            check(
                "red-proof: a distinct second copy brands the same canary false (witnessed red, browser)",
                distinct.ok === true,
                distinct.detail ?? "",
            );
        }
    }
    await teardownBridge();

    console.log("typegpu peer identity (browser, zero-config sandbox)…");
    const sandboxDir = join(work, "identity-browser-sandbox");
    writeIdentityZeroConfig(sandboxDir, engineTgz);
    const sandboxInstall = run(["bun", "install"], sandboxDir);
    check(
        "identity-browser (zero-config sandbox): the fixture installs",
        sandboxInstall.ok,
        sandboxInstall.ok ? "" : sandboxInstall.out.slice(-600),
    );
    if (sandboxInstall.ok) {
        const normal = await verify(sandboxDir, ["--timeout", "30000"], true);
        checkIdentityVerdict("zero-config sandbox, normal config", normal, true);
        checkPrebundled("zero-config sandbox, normal config", normal, false);
    }
    await teardownBridge();

    // No perturbed arm for this fixture — measured 2026-08-10, two ways: (1) stripping `typegpu` out of
    // `devConfig`'s `optimizeDeps.exclude` (`SHALLOT_TEST_STRIP_OPTIMIZE_EXCLUDE`, tried first) and (2)
    // forcing it into `optimizeDeps.include` instead (tried as the deterministic fix once (1) read
    // prebundled=false with no crash) both left `node_modules/.vite/deps/_metadata.json` completely
    // empty after the run, under either knob. Root cause: this fixture's entry is the CLI's
    // middleware-synthesized index (`synthIndexPlugin`), never written to disk, so Vite's dep-optimizer
    // has no on-disk HTML to crawl at server start — `typegpu` is discoverable only when the browser
    // actually requests the manifest-loaded plugin module, which is `identityOwnBody`'s "runtime-
    // discovered dep" case documented above, and forcing `include` doesn't change that: an explicit
    // `include` still commits through the same async optimizer run this harness's single page load /
    // fast teardown never survives long enough to observe (confirmed: even reading disk metadata well
    // past the self-read's 3s retry window, after the whole run finished, showed no commit). A real
    // instrument here needs a harness that waits for and re-navigates on Vite's full-reload event once
    // the optimizer commits — out of this stage's footprint (`scripts/install-test.ts`'s existing verify
    // plumbing has no such wait). The ejected fixture below reaches a real, observable perturbation
    // instead: its entry is a real on-disk index.html, statically crawlable, so the exclusion-strip
    // there needs no runtime discovery at all. Both dev.ts escape hatches this arm would have needed
    // (`SHALLOT_TEST_STRIP_OPTIMIZE_EXCLUDE`, `SHALLOT_TEST_FORCE_OPTIMIZE_INCLUDE`) were reverted with
    // it — a perturbation arm that cannot be shown to have taken effect is worth less than no arm.

    console.log("typegpu peer identity (browser, ejected fixture)…");
    const normalConfig = ejectedViteConfig();
    const ejectedNormalDir = join(work, "identity-browser-ejected");
    writeIdentityEjected(ejectedNormalDir, engineTgz, normalConfig);
    const ejectedNormalInstall = run(["bun", "install"], ejectedNormalDir);
    check(
        "identity-browser (ejected, normal config): the fixture installs",
        ejectedNormalInstall.ok,
        ejectedNormalInstall.ok ? "" : ejectedNormalInstall.out.slice(-600),
    );
    if (ejectedNormalInstall.ok) {
        const normal = await verify(ejectedNormalDir, ["--timeout", "30000"], true);
        checkIdentityVerdict("ejected fixture, normal config", normal, true);
        checkPrebundled("ejected fixture, normal config", normal, false);
    }
    await teardownBridge();

    const perturbedConfig = stripTypegpuExclude(normalConfig);
    const ejectedPerturbedDir = join(work, "identity-browser-ejected-perturbed");
    writeIdentityEjected(ejectedPerturbedDir, engineTgz, perturbedConfig);
    const ejectedPerturbedInstall = run(["bun", "install"], ejectedPerturbedDir);
    check(
        "identity-browser (ejected, perturbed — typegpu prebundled): the fixture installs",
        ejectedPerturbedInstall.ok,
        ejectedPerturbedInstall.ok ? "" : ejectedPerturbedInstall.out.slice(-600),
    );
    if (ejectedPerturbedInstall.ok) {
        // Measured 2026-08-10: stripping the exclusion on the ejected path never reaches a second
        // identity — the page dies first, at `unplugin-typegpu`'s transform, on typegpu's own
        // `Invalid property key 'type': Identifiers cannot start with reserved keywords` (the symptom
        // `bin/dev.ts`'s exclusion comment already records). Prebundled chunks are pre-transformed, so
        // the plugin never sees typegpu's source and the TGSL it must rewrite ships raw. That failure IS
        // the answer for this arm, and it's the whole control: ESM import hoisting means `main.ts`'s own
        // top-level statements (`identityOwnBody`'s self-read included) never run at all once
        // `import { run } from "@dylanebert/shallot"` fails to resolve its module graph — `run`
        // transitively pulls in the same `image.ts` the transform dies inside, before any of this
        // fixture's own code executes. A previous version of this check asserted an in-page prebundle
        // read here too, on the (wrong, corrected after measurement) assumption that the self-read ran
        // ahead of `run()`'s transform crash; it doesn't, since the crash is at *import resolution*, not
        // inside `run()`'s own body. No `window.__harness` is ever assigned on this arm — the loud
        // transform failure above is the only observable, and it's sufficient: a page that dies before
        // any app code runs cannot silently duplicate identity.
        const perturbed = await verify(ejectedPerturbedDir, ["--timeout", "30000"], true);
        const errors = JSON.stringify(perturbed?.errors ?? perturbed?.error ?? perturbed ?? "");
        check(
            "ejected fixture, perturbed: the strip fails loud at transform, never silently duplicating identity",
            errors.includes("Identifiers cannot start with reserved keywords"),
            errors.slice(0, 400),
        );
    }
    await teardownBridge();
}

// A pinned, deliberately different patch version for the second physical copy — not the identical
// `~0.12.0` alias `identityFlow` uses above. Measured 2026-08-10 (0.11-era pins): pnpm's content-addressable store
// dedupes an alias install (`typegpu2: npm:typegpu@~0.11.9`) against the real `typegpu@~0.11.9`
// dependency down to one physical store entry regardless of the alias name, so the two imports
// shared a single module instance and the red arm read `RED=true` — a false negative in the harness,
// not a real duplicate (the actual peer case, `typegpu` resolved through the engine's own
// `peerDependencies`, still reads `GREEN=true` clean under pnpm — see the Live log). A genuinely
// distinct version forces two store entries under every manager; `isTgpuFn`'s brand check is
// version-agnostic in this range, so 0.11.8 still probes the same property.
const PM_RED_COPY_VERSION = "0.11.8";

const managerInstall: Record<"npm" | "pnpm", string[]> = {
    npm: ["npm", "install", "--no-audit", "--no-fund"],
    pnpm: ["pnpm", "install"],
};

// npm + pnpm arms: the stage-1 brand-check probe, reused verbatim, installed under the two package
// managers bun can't stand in for — not a second `test:install` flow, a minimal fixture (engine
// tarball + typegpu + the probe). npm 7+ auto-installs and usually hoists the `typegpu` peer to one
// copy; pnpm's isolated `node_modules` is the high-risk arm the spec calls out. Failure output names
// the manager and both resolved paths, so a real cross-manager divergence is diagnosable from the
// gate log alone.
function pmIdentityFlow(work: string, engineTgz: string, manager: "npm" | "pnpm") {
    console.log(`typegpu peer identity (${manager}, brand check, red-first)…`);
    const proj = join(work, `identity-${manager}`);
    mkdirSync(proj, { recursive: true });
    writeFileSync(
        join(proj, "package.json"),
        `${JSON.stringify(
            {
                name: "identity-sandbox",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:${engineTgz}`,
                    typegpu: "~0.12.0",
                    typegpu2: `npm:typegpu@${PM_RED_COPY_VERSION}`,
                },
            },
            null,
            2,
        )}\n`,
    );

    if (!Bun.which(manager)) {
        check(`${manager} is on PATH`, false, `${manager} not found — cannot run this arm`);
        return;
    }

    const install = run(managerInstall[manager], proj);
    check(
        `the identity fixture installs under ${manager} (real + aliased typegpu copy)`,
        install.ok,
        install.ok ? "" : install.out.slice(-600),
    );
    if (!install.ok) return;

    writeFileSync(join(proj, "identity-check.ts"), identityProbeScript(true));
    const result = run(["bun", "identity-check.ts"], proj);
    const paths =
        result.out.match(/PATH_TYPEGPU2?=\S+/g)?.join(" ") ?? "(no resolved paths printed)";
    check(
        `${manager}: the app's own typegpu resolution brands the engine-built canary true (same physical copy)`,
        result.ok && /GREEN=true/.test(result.out),
        result.ok ? paths : `${manager}: ${result.out.slice(-400)}`,
    );
    check(
        `${manager}: a second physical copy of typegpu brands the same canary false (peer identity actually observed)`,
        result.ok && /RED=false/.test(result.out),
        result.ok ? paths : `${manager}: ${result.out.slice(-400)}`,
    );
}

function tgslFlow(sandbox: string, dist: string) {
    console.log("TGSL distribution (the build transform + an executing resolve)…");

    // the vite arm: the sandbox's own sources carry no typegpu, so any metadata in dist came from
    // engine source inside node_modules — what the CLI's synthesized config has to reach.
    // Split into two named checks so a red names which term failed — the old conjunction reported
    // only the assets list, leaving the failing half unmeasured after five red runs.
    const assets = existsSync(join(dist, "assets")) ? readdirSync(join(dist, "assets")) : [];
    const bundled = assets
        .filter((f) => f.endsWith(".js"))
        .map((f) => readFileSync(join(dist, "assets", f), "utf8"))
        .join("");
    const transpiledOk = TRANSPILED.test(bundled);
    // Metadata format version term: unplugin-typegpu emits the metadata object as
    // `{v: <METADATA_FORMAT_VERSION>, name: …, ast: …, externals: …}` — the vite build resolves the
    // ESM variant (factory-BSXzHM_n.js:23, METADATA_FORMAT_VERSION = 2 at common-Dilxou2I.js:430;
    // the CJS twins factory-CXjNN-uX.cjs:30 / common-hMfUmUk0.cjs:457 carry identical code). The
    // engine's own runtime never reads this `v` field — typegpu's `normalizeMetadata` does
    // (typegpu/shared/normalizeMetadata.js:18,23) — and typegpu exports no version constant, so the
    // expectation is a literal here. It is a literal in exactly ONE place: the extraction below is
    // version-agnostic and this constant is the only expectation, so a bump cannot leave a name and
    // an assertion disagreeing (the duplicated-threshold defect this unit already paid for at S4).
    // Pinning the version rather than a field name means the next V1→V2-style rename reds as a
    // version change instead of a mystery string absence.
    //
    // Red-first witness, taken against THIS `check()` on the live gate rather than a
    // reimplementation of the predicate: setting the constant below to 3 and running
    // `bun run test:install` reds this arm by name — `✗ the vite build emitted typegpu metadata in
    // the expected shape and version — … expected=3; metaVersions=2×118; …` — and the run's own FAIL
    // line names it. Restored, the arm greens with `expected=2`. Mutating the *expectation* is the
    // witness that matters; deleting the assertion greens for free and witnesses nothing.
    //
    // NB the spelling: `bun run format` runs `biome check --write`, whose naming convention rewrites a SCREAMING_SNAKE local const to PascalCase here.
    const ExpectedMetaVersion = 2;
    // TRANSPILED term diagnostic: bundle byte length and whether any transpiled-shape marker appears
    // at all. `body:[0,` is the TRANSPILED regex's head; `__TYPEGPU_META__` ships with the typegpu
    // runtime regardless of the transform, so its presence alone is not a transform witness.
    const bodyShapeMarker = /["'`]?body["'`]?\s*:\s*\[0,/.test(bundled);
    const typegpuMetaPresent = bundled.includes("__TYPEGPU_META__");
    // Metadata format version diagnostic: which `v:N` values appear in objects shaped like the
    // typegpu metadata ({v:N,name:…}), tallied rather than listed — the bundle carries one metadata
    // object per transformed TGSL fn, so the raw list is one number repeated per fn and illegible
    // at any real bundle size. Read the tally as the mechanism: a single expected version with a
    // count is a healthy bundle, one differing number is a version bump, a mixed tally a
    // half-transformed graph, and `(none)` a *shape* miss (a renamed `v` key, quoted keys from a
    // new minifier) rather than a version change. The arm's name states both legs it asserts —
    // shape and version — and no number, so a red never tells a version story about a shape event
    // and no bump can leave the name disagreeing with the assertion. `externals` is the V2 field
    // name; `externalNames` was the V1 field, retained in the identifier list so a future rename
    // stays visible beside the version.
    const nearbyIdentifiers = ["externals", "externalNames", "__TYPEGPU_META__"].filter((id) =>
        bundled.includes(id),
    );
    const metaVersions = [...bundled.matchAll(/\{v:\s*(\d+)\s*,\s*name:/g)].map((m) => m[1]);
    const metaVersionTally =
        [...new Set(metaVersions)]
            .sort()
            .map((v) => `${v}×${metaVersions.filter((x) => x === v).length}`)
            .join(",") || "(none)";
    // Every match must carry the expected version, and there must be at least one: a bare
    // `.test()` for the expected shape greens on a bundle where one object is V2 and the rest are
    // not, and an `every` with no non-empty guard greens on a bundle with no metadata at all.
    const metaVersionOk =
        metaVersions.length > 0 && metaVersions.every((v) => v === String(ExpectedMetaVersion));
    check(
        "the vite build transformed engine TGSL body inside node_modules (TRANSPILED shape)",
        transpiledOk,
        `bundle=${bundled.length} bytes; body:[0, marker=${bodyShapeMarker}; __TYPEGPU_META__=${typegpuMetaPresent}`,
    );
    check(
        "the vite build emitted typegpu metadata in the expected shape and version",
        metaVersionOk,
        `bundle=${bundled.length} bytes; expected=${ExpectedMetaVersion}; metaVersions=${metaVersionTally}; identifiers: ${nearbyIdentifiers.join(", ") || "(none)"}`,
    );

    // the recipe a plain bun/node consumer follows, exactly as the repo runs it (packages/shallot/
    // tests/tgsl.ts): `.ts` only, because the bun arm re-emits every file its filter matches with an
    // explicit loader — including the ones it prunes — which strips CJS default-export interop from a
    // plain `.js` dependency. Shipping the unfiltered form here would document a recipe that breaks
    // the moment the consumer imports a CJS package.
    writeFileSync(
        join(sandbox, "tgsl-preload.ts"),
        `import { plugin } from "bun";\n` +
            `import typegpu from "unplugin-typegpu/bun";\n` +
            `plugin(typegpu({ include: /\\.tsx?$/ }));\n`,
    );
    writeFileSync(
        join(sandbox, "tgsl-check.ts"),
        `import tgpu from "typegpu";\n` +
            `import { tgslCanary } from "@dylanebert/shallot/runtime";\n` +
            `const wgsl = tgpu.resolve([tgslCanary]);\n` +
            `if (!wgsl.includes("(x + 1u)")) { console.error("unexpected WGSL:\\n" + wgsl); process.exit(1); }\n` +
            `console.log("TGSL_OK " + wgsl.replace(/\\s+/g, " "));\n`,
    );

    const green = run(["bun", "--preload", "./tgsl-preload.ts", "tgsl-check.ts"], sandbox);
    check(
        "an installed engine TGSL fn resolves to its WGSL under the transform",
        green.ok && /TGSL_OK .*fn \w+\(x: u32\) -> u32/.test(green.out),
        green.ok ? green.out.trim().slice(0, 200) : green.out.slice(-400),
    );
    // the red arm — without it the check above passes on a tarball that shipped no metadata at all
    const red = run(["bun", "tgsl-check.ts"], sandbox);
    check(
        "the same resolve fails without the transform (metadata comes from the plugin)",
        !red.ok && /Missing metadata/.test(red.out),
        red.ok ? "resolved without the plugin — the check proves nothing" : "",
    );
}

// realpath: macOS tmpdir is a symlink (/var → /private/var); vite realpaths files before the
// fs.allow prefix check, so the sandbox paths must be the resolved form or /@fs requests 403
if (import.meta.main) {
    const work = realpathSync(mkdtempSync(join(tmpdir(), "shallot-install-")));
    const sandbox = join(work, "app");
    try {
        console.log("packing engine + widget…");
        const engineTgz = pack(ENGINE_DIR, join(work, "engine-pack"));
        const widgetTgz = pack(WIDGET_DIR, join(work, "widget-pack"));

        // display-independent, so it runs first: no GPU, no `skipReason()` guard anywhere above it
        identityFlow(work, engineTgz);
        pmIdentityFlow(work, engineTgz, "npm");
        pmIdentityFlow(work, engineTgz, "pnpm");

        // a real manifest project: installed engine + an installed plugin library + a local plugin, the
        // audio plugin pulling its wasm in. No vite.config, no index.html — the CLI supplies the harness.
        for (const d of ["scenes", "src", "public"])
            mkdirSync(join(sandbox, d), { recursive: true });
        writeFileSync(
            join(sandbox, "package.json"),
            `${JSON.stringify(
                {
                    name: "install-sandbox",
                    private: true,
                    type: "module",
                    dependencies: {
                        "@dylanebert/shallot": `file:${engineTgz}`,
                        "shallot-widget-fixture": `file:${widgetTgz}`,
                        // the shape a real consumer actually ships: a consumer's own source imports
                        // `typegpu/data` directly (not just transitively through the engine peer dep) —
                        // src/spin.ts below exercises it, so the prebundle-exclusion rung covers it too.
                        typegpu: "~0.12.0",
                    },
                },
                null,
                2,
            )}\n`,
        );
        writeFileSync(
            join(sandbox, "shallot.json"),
            `${JSON.stringify(
                {
                    scene: "scenes/main.scene",
                    plugins: {
                        Orbit: true, // the orbit camera the scene's `orbit` attribute drives
                        Audio: true, // an engine extra whose wasm must ship
                        Widget: "shallot-widget-fixture/widget", // an installed plugin by subpath
                        Spin: "./src/spin", // a local plugin
                    },
                },
                null,
                2,
            )}\n`,
        );
        writeFileSync(
            join(sandbox, "scenes", "main.scene"),
            `<scene>\n    <a ambient-light="color: 0xd0dcec; intensity: 0.5" />\n    <a directional-light="direction: -0.4 -1 -0.55; color: 0xfff4e0; intensity: 1.1" />\n    <a camera sear orbit="distance: 5; yaw: 0.6; pitch: 0.25" transform />\n    <a part transform="pos: 0 0 0" color="rgba: 0.85 0.55 0.35 1" />\n</scene>\n`,
        );
        writeFileSync(
            join(sandbox, "src", "spin.ts"),
            // the `typegpu/data` import is load-bearing, not decoration: a consumer whose own source
            // imports typegpu directly (a real consumer's shape) is a second entry into Vite's dep
            // scanner distinct from the engine's own bare specifier — the browser-boot rung below only
            // covers this shape because this file reaches it.
            `import type { Plugin, State, System } from "@dylanebert/shallot";\nimport * as d from "typegpu/data";\nconst SpinSystem: System = { group: "simulation", update(_s: State) {} };\nconst SpinPlugin: Plugin = { name: "Spin", systems: [SpinSystem] };\nconsole.log(d.f32);\nexport default SpinPlugin;\n`,
        );
        writeFileSync(
            join(sandbox, "public", "icon.svg"),
            `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>\n`,
        );

        console.log("bun install (the engine + widget tarballs + their deps)…");
        const install = run(["bun", "install"], sandbox);
        check(
            "bun install succeeds from the packed tarballs",
            install.ok,
            install.ok ? "" : install.out.slice(-600),
        );
        check(
            "the engine's audio wasm shipped in the tarball (files surface)",
            existsSync(
                join(sandbox, "node_modules/@dylanebert/shallot/rust/audio/pkg/shallot_audio.wasm"),
            ),
        );
        check(
            "the schema shipped in the tarball",
            existsSync(join(sandbox, "node_modules/@dylanebert/shallot/shallot.schema.json")),
        );
        // the version-matched agent context: the prepack projection must ship (engine AGENTS.md + the
        // examples index + the recipes corpus), and the shipped index must not dangle at tiers the tarball
        // omits (gym/showcase live in the repo only).
        const shipped = join(sandbox, "node_modules/@dylanebert/shallot");
        check(
            "the engine AGENTS.md shipped in the tarball",
            existsSync(join(shipped, "AGENTS.md")),
        );
        check(
            "the 0.9 migration guide shipped in the tarball",
            existsSync(join(shipped, "MIGRATION.md")),
        );
        check(
            "the recipes corpus shipped in the tarball (prepack projection)",
            existsSync(join(shipped, "examples/AGENTS.md")) &&
                existsSync(join(shipped, "examples/recipes/build-a-scene/src/build.ts")) &&
                existsSync(join(shipped, "examples/recipes/save-and-restore/shallot.json")),
        );
        check(
            "a shipped recipe carries its package.json (the copy-out project surface)",
            existsSync(join(shipped, "examples/recipes/build-a-scene/package.json")),
        );
        check(
            "no monorepo-only plumbing leaked into a shipped recipe (tsconfig / node_modules)",
            !existsSync(join(shipped, "examples/recipes/build-a-scene/tsconfig.json")) &&
                !existsSync(join(shipped, "examples/recipes/build-a-scene/node_modules")),
        );
        // repo test files import across the monorepo root (scripts/, examples/gym), paths that dangle
        // in a consumer install — the `files` surface must exclude every *.test.ts, bin included.
        const leakedTests = [...new Bun.Glob("**/*.test.ts").scanSync({ cwd: shipped })];
        check(
            "no test files shipped in the tarball (files surface excludes *.test.ts)",
            leakedTests.length === 0,
            leakedTests.slice(0, 5).join(", "),
        );
        // the dynamics-smoke plugins are CI scaffolding — stripped from the shipped corpus (file + manifest
        // entry) so a copied-out physics recipe carries no `./src/smoke` reference that would fail to build.
        check(
            "the shipped physics recipe dropped its smoke plugin (file + manifest entry)",
            !existsSync(join(shipped, "examples/recipes/joints/src/smoke.ts")) &&
                !/smoke/.test(
                    readFileSync(join(shipped, "examples/recipes/joints/shallot.json"), "utf8"),
                ),
        );
        const idx = existsSync(join(shipped, "examples/AGENTS.md"))
            ? readFileSync(join(shipped, "examples/AGENTS.md"), "utf8")
            : "";
        check(
            "the shipped index carries recipes with no dangling gym/showcase tier",
            /## Recipes/.test(idx) && !/## Gym/.test(idx) && !/## Showcase/.test(idx),
        );
        check(
            "the shipped index names `shallot recipe` as the copy-out command",
            /shallot recipe/.test(idx),
        );

        if (install.ok) {
            console.log("shallot build (the installed CLI, manifest project)…");
            const build = run(["bun", CLI, "build", "."], sandbox);
            check("shallot build exits clean", build.ok, build.ok ? "" : build.out.slice(-900));
            check(
                "no unresolved imports in the build",
                !/Failed to resolve|does not provide an export/i.test(build.out),
            );

            const dist = join(sandbox, "dist");
            check("dist/index.html produced", existsSync(join(dist, "index.html")));
            const assets = existsSync(join(dist, "assets"))
                ? readdirSync(join(dist, "assets"))
                : [];
            check(
                "the audio wasm bundled into dist",
                assets.some((f) => f.endsWith(".wasm")),
                assets.join(", ") || "(no assets dir)",
            );

            // rust/window ships in the tarball (package.json `files` includes `rust/window` minus
            // `target/`), so `shallot build --target <os>` from an installed package compiles the crate
            // lazily via cargo. A real native build is a multi-minute cargo/CEF arm — gated out of the
            // default suite (suite-speed budgets). Here we assert the crate is present and
            // resolvable in the installed layout; the premise builds (spec gates 4+5) run it for real.
            check(
                "the rust/window crate ships in the installed package (lazy native-build source)",
                existsSync(join(shipped, "rust/window/Cargo.toml")) &&
                    existsSync(join(shipped, "rust/window/Cargo.lock")),
            );

            // the crate-present check above says the file crossed the pack/install boundary; it says nothing
            // about the CLI's behavior when it hasn't. `requireRustCrate` runs before cargo is spawned, so
            // hiding the crate exercises the whole diagnostic path — resolution, message, non-zero exit —
            // for the price of a rename, with no toolchain involved.
            console.log("shallot build --target linux with the crate hidden (ENOENT guard fires)…");
            const crate = join(shipped, "rust/window");
            const hidden = `${crate}.hidden`;
            renameSync(crate, hidden);
            const guarded = run(["bun", CLI, "build", ".", "--target", "linux"], sandbox);
            renameSync(hidden, crate);
            check(
                "a missing crate fails with the corrupt-install diagnostic, not a raw ENOENT from cargo",
                !guarded.ok &&
                    /corrupt install/.test(guarded.out) &&
                    !/ENOENT|No such file or directory/.test(guarded.out),
                guarded.out.slice(-900),
            );
            check("the hidden crate is restored", existsSync(join(crate, "Cargo.toml")));

            tgslFlow(sandbox, dist);

            // the dev server: live resolution + asset serving over vite (a different path than the build
            // bundle — it's where the cross-repo fs.allow / wasm-serving lives).
            console.log("shallot dev (boot + resolve + serve the wasm)…");
            const port = await freePort();
            const dev = Bun.spawn(["bun", CLI, "dev", ".", "--port", String(port)], {
                cwd: sandbox,
                stdout: "pipe",
                stderr: "pipe",
            });
            const dec = new TextDecoder();
            let devLog = "";
            const pump = (stream: ReadableStream<Uint8Array>) =>
                (async () => {
                    const reader = stream.getReader();
                    for (;;) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        if (value) devLog += dec.decode(value);
                    }
                })();
            const drain = pump(dev.stdout);
            const drainErr = pump(dev.stderr);
            try {
                // Wait on the server's own ready signal, not a wall clock: `dev.ts` calls
                // `server.printUrls()` immediately after `server.listen()` resolves, so the banner is the
                // moment the socket is up. A fixed HTTP-poll deadline conflated "still starting" with
                // "never coming" and flaked 2 runs in 5. The child exiting ends the wait too — a crashed
                // dev server has nothing to wait out. `printUrls` colors its output, so strip SGR first.
                const banner = new RegExp(`Local:\\s+\\S*http://localhost:${port}/`);
                const listening = await waitFor(
                    async () => banner.test(strip(devLog)) || dev.exitCode !== null,
                    90000,
                );
                // Once listening, a request that can't be served in a second is a real failure, not slow boot.
                const up =
                    listening &&
                    dev.exitCode === null &&
                    (await waitFor(async () => {
                        try {
                            // localhost, not 127.0.0.1 — vite binds the family localhost resolves to
                            // (IPv6-only on macOS), and it's the host the banner advertises
                            return (await fetch(`http://localhost:${port}/`)).ok;
                        } catch {
                            return false;
                        }
                    }, 5000));
                check(
                    "shallot dev boots a server",
                    up,
                    up
                        ? ""
                        : `${dev.exitCode !== null ? `dev exited ${dev.exitCode}; ` : listening ? "banner printed, no response; " : "no ready banner; "}${devLog.slice(-400)}`,
                );
                if (up) {
                    const mod = await fetch(
                        `http://localhost:${port}/@id/__x00__virtual:project`,
                    ).then((r) => r.text());
                    check(
                        "dev resolves the manifest (installed subpath + local + engine)",
                        /shallot-widget-fixture/.test(mod) &&
                            /src\/spin/.test(mod) &&
                            !/Failed to resolve/i.test(mod),
                    );
                    const wasmFs = resolve(
                        sandbox,
                        "node_modules/@dylanebert/shallot/rust/audio/pkg/shallot_audio.wasm",
                    );
                    const res = await fetch(`http://localhost:${port}/@fs${wasmFs}`);
                    const magic = new Uint8Array(
                        (res.ok ? await res.arrayBuffer() : new ArrayBuffer(0)).slice(0, 4),
                    );
                    // the dev arm of the TGSL contract: `shallot dev` synthesizes its own vite config, a
                    // different path than the build above, and it must transform engine source in
                    // node_modules too. Ask the server for the module the page imports and read what it
                    // serves. (It does not cover a prebundled `.vite/deps` copy — dep optimization runs on
                    // page load, which this headless boot never performs. The real-browser rung below does.)
                    const engineMod = await fetch(
                        `http://localhost:${port}/node_modules/@dylanebert/shallot/src/engine/runtime/gpu.ts`,
                    );
                    const served = engineMod.ok ? await engineMod.text() : "";
                    check(
                        "dev serves engine TGSL through the transform",
                        TRANSPILED.test(served),
                        engineMod.ok ? "" : `HTTP ${engineMod.status}`,
                    );
                    check(
                        "dev serves the audio wasm (fs.allow ok, valid magic 0061736d)",
                        res.ok &&
                            magic[0] === 0x00 &&
                            magic[1] === 0x61 &&
                            magic[2] === 0x73 &&
                            magic[3] === 0x6d,
                        res.ok ? "" : `HTTP ${res.status}`,
                    );
                }
                check(
                    "no dev-server errors (resolve / fs allow-list)",
                    !/Failed to resolve|outside of Vite serving allow list/i.test(strip(devLog)),
                );
            } finally {
                dev.kill();
                await Promise.race([Promise.all([drain, drainErr]), Bun.sleep(1500)]);
            }

            // the real-device boot rung the fetch-only checks above can't be: Vite's dependency optimizer
            // prebundles a bare `@dylanebert/shallot` import only on an actual browser page load (esbuild
            // scans the entry HTML's script graph), never on a raw HTTP fetch of a known module path — so
            // this sandbox (a genuine `node_modules` install off a packed tarball, same resolution shape a
            // registry install produces — not a symlink) is booted through a real browser and its rendered
            // pipelines are asserted, catching the 5b-2f-5 prebundle-before-transform defect the checks
            // above structurally could not (`scripts/verify.ts`'s `verify()` is the same shipped gate
            // `bun bench` / `bun run flows` / `bun run recipes` drive; display-gated identically, native
            // hardware only — WSL runs it for real against the Windows host's GPU via the bridge).
            console.log(
                "shallot verify (a real browser boot — warms the installed engine's pipelines)…",
            );
            const skip = skipReason();
            if (skip) {
                console.log(`  · skipped (needs native hardware: ${skip})`);
            } else {
                const result = await verify(sandbox, ["--timeout", "30000"], true);
                check(
                    "a real browser boots the installed engine and warms its pipelines",
                    result?.pass === true && result.booted === true && result.rendered === true,
                    verifyDiagnostic(result),
                );
            }
            await teardownBridge();
        }

        if (install.ok) recipeFlow(work, engineTgz, sandbox);

        await ejectedFlow(work, engineTgz);

        await identityBrowserFlow(work, engineTgz);

        createShallotFlow(work, engineTgz);
    } finally {
        rmSync(work, { recursive: true, force: true });
    }

    if (fails.length) {
        console.error(`\nFAIL: ${fails.length} check(s) failed: ${fails.join("; ")}`);
        process.exit(1);
    }
    console.log("\nPASS: real-install flow clean");
} // import.meta.main
