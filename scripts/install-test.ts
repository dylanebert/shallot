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
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { skipReason, teardownBridge, verify } from "./verify";

const ENGINE_DIR = resolve(import.meta.dir, "../packages/shallot");
const WIDGET_DIR = resolve(import.meta.dir, "install-test/widget");
const CREATE_SHALLOT = resolve(import.meta.dir, "../packages/create-shallot/index.ts");
const CLI = "node_modules/@dylanebert/shallot/bin/cli.ts"; // the installed CLI, run as a real user would

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
function recipeFlow(work: string, engineTgz: string) {
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

function tgslFlow(sandbox: string, dist: string) {
    console.log("TGSL distribution (the build transform + an executing resolve)…");

    // the vite arm: the sandbox's own sources carry no typegpu, so any metadata in dist came from
    // engine source inside node_modules — what the CLI's synthesized config has to reach
    const assets = existsSync(join(dist, "assets")) ? readdirSync(join(dist, "assets")) : [];
    const bundled = assets
        .filter((f) => f.endsWith(".js"))
        .map((f) => readFileSync(join(dist, "assets", f), "utf8"))
        .join("");
    check(
        "the vite build transformed engine TGSL inside node_modules",
        TRANSPILED.test(bundled) && bundled.includes("externalNames"),
        assets.join(", ") || "(no assets dir)",
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
const work = realpathSync(mkdtempSync(join(tmpdir(), "shallot-install-")));
const sandbox = join(work, "app");
try {
    console.log("packing engine + widget…");
    const engineTgz = pack(ENGINE_DIR, join(work, "engine-pack"));
    const widgetTgz = pack(WIDGET_DIR, join(work, "widget-pack"));

    // a real manifest project: installed engine + an installed plugin library + a local plugin, the
    // audio plugin pulling its wasm in. No vite.config, no index.html — the CLI supplies the harness.
    for (const d of ["scenes", "src", "public"]) mkdirSync(join(sandbox, d), { recursive: true });
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
                    // the shape orrstead/spindle actually ship: a consumer's own source imports
                    // `typegpu/data` directly (not just transitively through the engine peer dep) —
                    // src/spin.ts below exercises it, so the prebundle-exclusion rung covers it too.
                    typegpu: "~0.11.9",
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
        `<scene>\n    <a ambient-light="intensity: 0.6" />\n    <a camera sear transform />\n    <a part transform color="rgba: 0.8 0.5 0.3" />\n</scene>\n`,
    );
    writeFileSync(
        join(sandbox, "src", "spin.ts"),
        // the `typegpu/data` import is load-bearing, not decoration: a consumer whose own source
        // imports typegpu directly (orrstead/spindle's real shape) is a second entry into Vite's dep
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
    check("the engine AGENTS.md shipped in the tarball", existsSync(join(shipped, "AGENTS.md")));
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
        const assets = existsSync(join(dist, "assets")) ? readdirSync(join(dist, "assets")) : [];
        check(
            "the audio wasm bundled into dist",
            assets.some((f) => f.endsWith(".wasm")),
            assets.join(", ") || "(no assets dir)",
        );

        tgslFlow(sandbox, dist);

        // the dev server: live resolution + asset serving over vite (a different path than the build
        // bundle — it's where the cross-repo fs.allow / wasm-serving lives).
        console.log("shallot dev (boot + resolve + serve the wasm)…");
        const port = 5191;
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
            const up = await waitFor(async () => {
                try {
                    // localhost, not 127.0.0.1 — vite binds the family localhost resolves to
                    // (IPv6-only on macOS), and it's the host the banner advertises
                    return (await fetch(`http://localhost:${port}/`)).ok;
                } catch {
                    return false;
                }
            }, 40000);
            check("shallot dev boots a server", up, up ? "" : devLog.slice(-400));
            if (up) {
                const mod = await fetch(`http://localhost:${port}/@id/__x00__virtual:project`).then(
                    (r) => r.text(),
                );
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
                !/Failed to resolve|outside of Vite serving allow list/i.test(devLog),
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
                result
                    ? JSON.stringify(result.errors ?? result.error ?? result)
                    : "no verify result",
            );
        }
        await teardownBridge();
    }

    if (install.ok) recipeFlow(work, engineTgz);

    createShallotFlow(work, engineTgz);
} finally {
    rmSync(work, { recursive: true, force: true });
}

if (fails.length) {
    console.error(`\nFAIL: ${fails.length} check(s) failed: ${fails.join("; ")}`);
    process.exit(1);
}
console.log("\nPASS: real-install flow clean");
