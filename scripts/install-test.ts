// Real-user install flow, sandboxed — the gate the dev symlink can't be. It packs the engine
// (`bun pm pack`, so it exercises the published `files` surface: src, bin, the audio `.wasm`, the
// schema) and a custom plugin library, installs both into a throwaway project via `bun install` (a real
// node_modules layout, not a workspace symlink), then runs every headless CLI flow against the installed
// engine: `shallot build` (manifest resolves an installed plugin by subpath + a local plugin, the wasm
// bundles), `shallot dev` (the server resolves the same + serves the wasm over /@fs), and `bun create
// shallot` (scaffold → install → build the starter). The packaging / resolution / asset failures the
// repo's own symlinked dev setup hides. Run: `bun run scripts/install-test.ts` (or `bun run test:install`).

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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
import { dirname, join, resolve } from "node:path";
import { harnessArms, harnessContract } from "./install-test/harness";
import { outputFlow } from "./install-test/output";
import { physicsArms, projectPhysics } from "./install-test/physics";
import { projectFlow } from "./install-test/project";
import { runtimeArms } from "./install-test/runtime";

// `shallot verify` is archived; `check` replaces its browser-boot arms in this version.
const UNAVAILABLE = "unavailable — `check` replaces `shallot verify` in this version";

const REPO_ROOT = resolve(import.meta.dir, "..");
const ENGINE_DIR = resolve(REPO_ROOT);
const WIDGET_DIR = resolve(import.meta.dir, "install-test/widget");
const CLI = "node_modules/.bin/shallot"; // execute the installation's declared public bin

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

function pkgJson(fields: Record<string, unknown>, indent = 2): string {
    const pkg: Record<string, unknown> = { ...fields };
    return JSON.stringify(pkg, null, indent);
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

// The engine's own bridge spec, so the consumer installs what the engine tests against.
const BRIDGE: string = JSON.parse(readFileSync(join(import.meta.dir, "../package.json"), "utf8"))
    .devDependencies["bun-webgpu"];

/** Exercise the shipped native loader against the bridge peer in physical, external installs; retain raw child receipts. */
export function nativeFlow(work: string, engineTgz: string): void {
    const evidence = join(work, "native");
    mkdirSync(evidence, { recursive: true });
    let sequence = 0;
    const exec = (name: string, cmd: string[], cwd: string) => {
        const result = Bun.spawnSync(cmd, { cwd, stdout: "pipe", stderr: "pipe", timeout: 120000 });
        const stem = join(evidence, `${++sequence}-${name}`);
        writeFileSync(`${stem}.stdout`, result.stdout);
        writeFileSync(`${stem}.stderr`, result.stderr);
        writeFileSync(
            `${stem}.json`,
            JSON.stringify({
                cmd,
                cwd,
                exit: result.exitCode,
                signal: result.signalCode,
                runtime: Bun.version,
            }),
        );
        return {
            exit: result.exitCode,
            out: `${result.stdout.toString()}\n${result.stderr.toString()}`,
        };
    };
    const expect = (
        name: string,
        result: { exit: number; out: string },
        exit: number,
        text: RegExp,
    ) => {
        assert.equal(result.exit, exit, `${name}: ${result.out}`);
        assert.match(result.out, text, name);
        console.log(`native: ${name}`);
    };
    for (const layout of ["absent", "normal", "nested"]) {
        const project =
            layout === "nested"
                ? join(evidence, layout, "node_modules/native-consumer")
                : join(evidence, layout);
        mkdirSync(project, { recursive: true });
        writeFileSync(
            join(project, "package.json"),
            pkgJson({
                name: `native-${layout}`,
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:${engineTgz}`,
                    typegpu: "~0.12.5",
                    ...(layout === "absent" ? {} : { "bun-webgpu": BRIDGE }),
                },
            }),
        );
        const install = exec(
            `${layout}-install`,
            layout === "nested"
                ? [
                      "npm",
                      "install",
                      "--install-strategy=nested",
                      "--ignore-scripts",
                      "--no-audit",
                      "--no-fund",
                  ]
                : ["bun", "install"],
            project,
        );
        assert.equal(install.exit, 0, install.out);
        const shipped = join(project, "node_modules/@dylanebert/shallot");
        assert.equal(realpathSync(shipped), shipped, "engine is a physical install");
        assert(existsSync(join(shipped, "bin/bun-native.ts")), "tar contains bin/bun-native.ts");
        writeFileSync(
            join(project, "native.fixture.ts"),
            readFileSync(join(import.meta.dir, "install-test/native.fixture.ts")),
        );
        expect(
            `${layout}-import`,
            exec(
                `${layout}-import`,
                [
                    "bun",
                    "-e",
                    'import { run } from "@dylanebert/shallot"; if(typeof run !== "function") throw Error("run missing"); console.log("NONNATIVE_IMPORT_OK")',
                ],
                project,
            ),
            0,
            /NONNATIVE_IMPORT_OK/,
        );
        if (layout === "absent") {
            assert(!existsSync(join(project, "node_modules/bun-webgpu")));
            continue;
        }
        const peer = join(project, "node_modules/bun-webgpu");
        assert.equal(realpathSync(peer), peer, "peer is a physical install");
        expect(
            `${layout}-acquire`,
            exec(`${layout}-acquire`, ["bun", "native.fixture.ts"], project),
            0,
            /NATIVE_ACQUIRED/,
        );
    }
    console.log(`native: ${sequence} commands completed`);
}

const fails: string[] = [];
const check = (name: string, cond: boolean, detail = "") => {
    console.log(`  ${cond ? "✓" : "✗"} ${name}${detail ? ` — ${detail}` : ""}`);
    if (!cond) fails.push(name);
};

/** The install gate's missing-crate verdict must require a failed command and its owned diagnostic. */
export function missingCrateDiagnosticPass(result: { ok: boolean; out: string }): boolean {
    return (
        !result.ok &&
        /corrupt install/.test(result.out) &&
        !/ENOENT|No such file or directory/.test(result.out)
    );
}

/** Hash every file's path and bytes so a hidden packed crate must return unchanged. */
function directoryDigest(rootDir: string): string {
    const digest = createHash("sha256");
    const files = [...new Bun.Glob("**/*").scanSync({ cwd: rootDir })]
        .filter((file) => !file.endsWith("/"))
        .sort();
    for (const file of files) {
        digest.update(file);
        digest.update(readFileSync(join(rootDir, file)));
    }
    return digest.digest("hex");
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

// `shallot recipe <name> <dir>` copies a recipe out of the installed package into a runnable project;
// the copy's engine dep is version-pinned by the CLI, so here we point it back at the packed tarball
// (as a real user's registry install would resolve) and build it headlessly. Guards the whole copy-out
// path: recipe present in the pack, CLI copies it, the pinned dep installs, the project builds.
async function recipeFlow(work: string, engineTgz: string, sandbox: string, name: string) {
    console.log(`shallot recipe ${name} (copy a recipe out → install → build)…`);
    const dest = join(work, "recipe-out", name);
    const copied = run(["bun", CLI, "recipe", name, dest], sandbox);
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

    if (name === "gpu-particles")
        console.log(`copied-out producer recipe browser boot: ${UNAVAILABLE}`);
}

// MIGRATION.md's "An ejected Vite project" recipe, read straight out of the doc rather than
// re-typed — the failure this flow exists to catch is the recipe itself shipping broken, and a
// hand-written copy could silently diverge from what a consumer actually pastes. Anchored on the
// `// vite.config.ts` comment the doc's fenced block opens with, through the fence's closing ``` .
function ejectedViteConfig(): string {
    const doc = readFileSync(resolve(import.meta.dir, "../MIGRATION.md"), "utf8");
    const start = doc.indexOf("// vite.config.ts");
    if (start === -1) throw new Error("MIGRATION.md's ejected Vite recipe marker not found");
    const fenceEnd = doc.indexOf("\n```", start);
    if (fenceEnd === -1) throw new Error("MIGRATION.md's ejected Vite recipe has no closing fence");
    return doc.slice(start, fenceEnd);
}

// a true ejected shape (a real consumer's ejected shape, not the CLI's zero-config path): the
// project owns its own index.html + vite.config.ts. `./vite` ships a compiled `dist/vite.js` default
// alongside its `.ts` source `types` (`exports.md`), because Vite loads vite.config.ts in a plain
// `node` subprocess that applies no TS transform to a `node_modules` import.
async function ejectedFlow(work: string, engineTgz: string) {
    console.log("ejected Vite project (MIGRATION.md's recipe, booted verbatim)…");
    const proj = join(work, "ejected");
    mkdirSync(join(proj, "src"), { recursive: true });
    mkdirSync(join(proj, "scenes"), { recursive: true });
    writeFileSync(
        join(proj, "package.json"),
        `${pkgJson({
            name: "ejected-sandbox",
            private: true,
            type: "module",
            dependencies: {
                "@dylanebert/shallot": `file:${engineTgz}`,
                typegpu: "~0.12.5",
                // a second physical copy for the identity probe below — see `identityFlow`'s comment
                typegpu2: "npm:typegpu@~0.12.5",
            },
            devDependencies: { vite: "^8.0.0", "unplugin-typegpu": "~0.12.3" },
        })}\n`,
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
            `if (JSON.stringify(REAL_GPU_LAUNCH) !== '${JSON.stringify({ channel: "chromium", args: ["--enable-unsafe-webgpu", "--enable-features=WebGPUDeveloperFeatures", "--class=kex-gate"] })}') throw new Error("REAL_GPU_LAUNCH: wrong options");\n` +
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

    console.log(`documented ejected recipe browser boot: ${UNAVAILABLE}`);
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
// is asserted) before any browser refusal: a display-less host still exercises it (testing.md
// "Install gate"). `typegpu2` is a genuine
// second physical copy — an alias install of the identical version (`npm:typegpu@~0.12.5`) landing in
// its own `node_modules/typegpu2`, a distinct module-graph evaluation with its own `Symbol()` calls, not
// a re-export of the first. The red arm is the whole point: a probe green on first run with no witnessed
// red proves nothing.
function identityFlow(work: string, engineTgz: string) {
    console.log("typegpu peer identity (brand check, red-first)…");
    const proj = join(work, "identity");
    mkdirSync(proj, { recursive: true });
    writeFileSync(
        join(proj, "package.json"),
        `${pkgJson({
            name: "identity-sandbox",
            private: true,
            type: "module",
            dependencies: {
                "@dylanebert/shallot": `file:${engineTgz}`,
                typegpu: "~0.12.5",
                typegpu2: "npm:typegpu@~0.12.5",
            },
        })}\n`,
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

// A pinned, deliberately different patch version for the second physical copy — not the identical
// `~0.12.5` alias `identityFlow` uses above. Measured 2026-08-10 (0.11-era pins): pnpm's content-addressable store
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

// npm + pnpm arms: the brand-check probe, reused verbatim, installed under the two package
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
        `${pkgJson({
            name: "identity-sandbox",
            private: true,
            type: "module",
            dependencies: {
                "@dylanebert/shallot": `file:${engineTgz}`,
                typegpu: "~0.12.5",
                typegpu2: `npm:typegpu@${PM_RED_COPY_VERSION}`,
            },
        })}\n`,
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
    // an assertion disagreeing (the duplicated-threshold defect).
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

    // the recipe a plain bun/node consumer follows, exactly as the repo runs it (
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
        const engineTgz = projectPhysics(work, pack(ENGINE_DIR, join(work, "engine-pack")));
        const widgetTgz = pack(WIDGET_DIR, join(work, "widget-pack"));

        // display-independent, so it runs first: no GPU and no browser refusal anywhere above it
        projectFlow(engineTgz, join(work, "project-seam"));
        identityFlow(work, engineTgz);
        pmIdentityFlow(work, engineTgz, "npm");
        pmIdentityFlow(work, engineTgz, "pnpm");
        nativeFlow(work, engineTgz);

        // a real manifest project: installed engine + an installed plugin library + a local plugin, the
        // audio plugin pulling its wasm in. No vite.config, no index.html — the CLI supplies the harness.
        for (const d of ["scenes", "src", "public"])
            mkdirSync(join(sandbox, d), { recursive: true });
        writeFileSync(
            join(sandbox, "package.json"),
            `${pkgJson({
                name: "install-sandbox",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:${engineTgz}`,
                    "shallot-widget-fixture": `file:${widgetTgz}`,
                    // the shape a real consumer actually ships: a consumer's own source imports
                    // `typegpu/data` directly (not just transitively through the engine peer dep) —
                    // src/spin.ts below exercises it, so the prebundle-exclusion rung covers it too.
                    typegpu: "~0.12.5",
                },
            })}\n`,
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
            harnessContract +
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
        if (install.ok) {
            harnessArms(sandbox);
            runtimeArms(sandbox);
            physicsArms(sandbox);
            writeFileSync(
                join(sandbox, "missing-plugin.ts"),
                `import { build } from "@dylanebert/shallot";\n` +
                    `const Required = { name: "PackedRequired" };\n` +
                    `const Consumer = { name: "PackedConsumer", dependencies: [Required] };\n` +
                    `await build({ plugins: [Consumer], defaults: false });\n`,
            );
            const missing = run(["bun", "missing-plugin.ts"], sandbox);
            check(
                "the packed engine rejects a custom plugin's missing required edge at build",
                !missing.ok && /PackedConsumer requires PackedRequired/.test(missing.out),
                missing.out.slice(-400),
            );
        }
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
        // the version-matched agent context must ship (engine AGENTS.md + the
        // examples index + the recipes corpus), and the shipped index must not dangle at tiers the tarball
        // omits (showcase lives in the repo only).
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
            "the recipes corpus shipped in the tarball",
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
        // repo test files import across the monorepo root (scripts/), paths that dangle
        // in a consumer install — the `files` surface must exclude every *.test.ts, bin included.
        const leakedTests = [...new Bun.Glob("**/*.test.ts").scanSync({ cwd: shipped })];
        check(
            "no test files shipped in the tarball (files surface excludes *.test.ts)",
            leakedTests.length === 0,
            leakedTests.slice(0, 5).join(", "),
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
            // resolvable in the installed layout; the premise builds run it for real.
            check(
                "the rust/window crate ships in the installed package (lazy native-build source)",
                existsSync(join(shipped, "rust/window/Cargo.toml")) &&
                    existsSync(join(shipped, "rust/window/Cargo.lock")),
            );

            // the crate-present check above says the file crossed the pack/install boundary; it says nothing
            // about the CLI's behavior when it hasn't. `requireRustCrate` runs before cargo is spawned, so
            // hiding the crate exercises the whole diagnostic path — resolution, message, non-zero exit —
            // for the price of a rename, with no toolchain involved. Linux's system backend refusal runs
            // earlier, so this invocation uses --portable only to reach the intended diagnostic boundary.
            console.log(
                "shallot build --target linux --portable with the crate hidden (ENOENT guard fires)…",
            );
            const crate = join(shipped, "rust/window");
            const hidden = `${crate}.hidden`;
            const crateDigest = directoryDigest(crate);
            assert(!existsSync(hidden), "hidden crate destination must be absent");
            let guarded: { ok: boolean; out: string } | null = null;
            try {
                renameSync(crate, hidden);
                guarded = run(
                    ["bun", CLI, "build", ".", "--target", "linux", "--portable"],
                    sandbox,
                );
            } finally {
                if (existsSync(hidden)) renameSync(hidden, crate);
            }
            check(
                "a missing crate fails with the corrupt-install diagnostic, not a raw ENOENT from cargo",
                guarded !== null && missingCrateDiagnosticPass(guarded),
                guarded?.out.slice(-900) ?? "the build invocation did not return",
            );
            check(
                "the hidden crate is restored byte-for-byte",
                existsSync(join(crate, "Cargo.toml")) && directoryDigest(crate) === crateDigest,
            );

            tgslFlow(sandbox, dist);

            // the dev server: live resolution + asset serving over vite (a different path than the build
            // bundle — it's where the cross-repo fs.allow / wasm-serving lives). First prove the installed
            // public bin's human default reaches a controlled opener; then prove the gate's corrected
            // --no-open argv suppresses that same sink. BROWSER is an explicit test sink, not a policy that
            // disables opening globally.
            const openerSink = join(sandbox, "dev-opener-sink.js");
            const openerLog = join(sandbox, "dev-opener.log");
            writeFileSync(
                openerSink,
                `import { appendFileSync } from "node:fs";\n` +
                    `appendFileSync(${JSON.stringify(openerLog)}, JSON.stringify(process.argv.slice(2)) + "\\n");\n`,
            );
            rmSync(openerLog, { force: true });
            const openerPort = await freePort();
            const opener = Bun.spawn(["bun", CLI, "dev", ".", "--port", String(openerPort)], {
                cwd: sandbox,
                stdout: "pipe",
                stderr: "pipe",
                env: { ...process.env, BROWSER: openerSink },
            });
            try {
                const opened = await waitFor(
                    async () => existsSync(openerLog) || opener.exitCode !== null,
                    90000,
                );
                const argv = existsSync(openerLog)
                    ? (JSON.parse(readFileSync(openerLog, "utf8").trim()) as string[])
                    : [];
                check(
                    "installed dev forwards its original argv to the controlled opener",
                    opened &&
                        argv.slice(0, 4).join(" ") === `dev . --port ${openerPort}` &&
                        argv.at(-1)?.startsWith(`http://localhost:${openerPort}/`) === true,
                    argv.join(" "),
                );
            } finally {
                opener.kill();
                await opener.exited;
                rmSync(openerLog, { force: true });
            }

            console.log("shallot dev (boot + resolve + serve the wasm)…");
            const port = await freePort();
            const dev = Bun.spawn(["bun", CLI, "dev", ".", "--port", String(port), "--no-open"], {
                cwd: sandbox,
                stdout: "pipe",
                stderr: "pipe",
                env: { ...process.env, BROWSER: openerSink },
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
                check(
                    "installed dev --no-open suppresses the controlled opener",
                    !existsSync(openerLog),
                    existsSync(openerLog) ? readFileSync(openerLog, "utf8") : "",
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
                    const installedRoot = join(sandbox, "node_modules/@dylanebert/shallot");
                    const runtimeTarget = JSON.parse(
                        readFileSync(join(installedRoot, "package.json"), "utf8"),
                    ).exports["./runtime"];
                    assert.equal(typeof runtimeTarget, "string", "declared raw runtime export");
                    const runtimeEntry = resolve(installedRoot, runtimeTarget);
                    const canaryHops = [
                        ...readFileSync(runtimeEntry, "utf8").matchAll(
                            /export\s*\{[^}]*\btgslCanary\b[^}]*\}\s*from\s*["']([^"']+)["']/g,
                        ),
                    ];
                    assert.equal(canaryHops.length, 1, "one resolved public canary re-export");
                    const canarySource = resolve(dirname(runtimeEntry), `${canaryHops[0][1]}.ts`);
                    assert(
                        canarySource.startsWith(installedRoot + "/") && existsSync(canarySource),
                        "canary definition belongs to the installation",
                    );
                    const engineMod = await fetch(`http://localhost:${port}/@fs${canarySource}`);
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
                rmSync(openerSink, { force: true });
                rmSync(openerLog, { force: true });
            }

            console.log(`installed-engine browser boot: ${UNAVAILABLE}`);
        }

        if (install.ok) {
            await recipeFlow(work, engineTgz, sandbox, "joints");
            await recipeFlow(work, engineTgz, sandbox, "gpu-particles");
        }

        await ejectedFlow(work, engineTgz);

        console.log(`typegpu peer identity (browser): ${UNAVAILABLE}`);

        await outputFlow(work, engineTgz);
    } finally {
        if (process.env.SHALLOT_INSTALL_KEEP === "1")
            console.log(`install artifacts retained: ${work}`);
        else rmSync(work, { recursive: true, force: true });
    }

    if (fails.length) {
        console.error(`\nFAIL: ${fails.length} check(s) failed: ${fails.join("; ")}`);
        process.exit(1);
    }
    console.log("\nPASS: real-install flow clean");
} // import.meta.main
