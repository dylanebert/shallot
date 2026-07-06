// Real-user install flow, sandboxed — the gate the dev symlink can't be. It packs the engine
// (`bun pm pack`, so it exercises the published `files` surface: src, bin, editor, the audio `.wasm`, the
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
        `import type { Plugin, State, System } from "@dylanebert/shallot";\nconst SpinSystem: System = { group: "simulation", update(_s: State) {} };\nconst SpinPlugin: Plugin = { name: "Spin", systems: [SpinSystem] };\nexport default SpinPlugin;\n`,
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
    }

    createShallotFlow(work, engineTgz);
} finally {
    rmSync(work, { recursive: true, force: true });
}

if (fails.length) {
    console.error(`\nFAIL: ${fails.length} check(s) failed: ${fails.join("; ")}`);
    process.exit(1);
}
console.log("\nPASS: real-install flow clean");
