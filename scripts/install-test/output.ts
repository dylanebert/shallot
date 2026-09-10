import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
    cpSync,
    existsSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    realpathSync,
    renameSync,
    writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { brotliCompressSync, gzipSync } from "node:zlib";
import { parse } from "@babel/parser";
import { startBrowserServer, verifyArgs } from "./browser-server";

const root = resolve(import.meta.dir, "../..");
const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const files = (dir: string) =>
    readdirSync(dir, { recursive: true, withFileTypes: true })
        .filter((e) => e.isFile())
        .map((e) => join(e.parentPath, e.name));
const graphPlugin = `{
 name: "consumer-output-audit",
 generateBundle(_, bundle) {
  this.emitFile({type:"asset", fileName:"bundle-audit.json", source:JSON.stringify(Object.values(bundle).filter(x=>x.type==="chunk").map(x=>({file:x.fileName,imports:x.imports,dynamicImports:x.dynamicImports,modules:Object.entries(x.modules).filter(([,m])=>m.renderedLength>0).map(([id,m])=>({id,bytes:m.renderedLength}))}))) });
 }
}`;
// `route` is client-side, so it instruments a browser the same whether the CLI launched it locally or
// attached to one over `--connect`. Both entry points are wrapped: a verify driven over a browser
// server still goes through `newContext`, and wrapping only `launch` leaves that run with an empty
// receipt — an instrument that silently sees nothing rather than a gate.
const networkPreload = `import {writeFileSync} from "node:fs";
import {chromium} from "playwright";
const records=[]; const save=()=>writeFileSync(process.env.NETWORK_RECEIPT,JSON.stringify(records));
const instrument=browser=>{
 const create=browser.newContext.bind(browser);
 browser.newContext=async options=>{
  const context=await create({...options,serviceWorkers:"block"}); let origin;
  await context.route("**/*",route=>{
   const request=route.request(); const url=new URL(request.url());
   if(!origin && request.isNavigationRequest() && ["127.0.0.1","localhost"].includes(url.hostname)) origin=url.origin;
   const allowed=url.origin===origin; records.push({url:url.href,allowed}); save();
   return allowed?route.continue():route.abort("blockedbyclient");
  }); return context;
 };return browser;
};
const launch=chromium.launch.bind(chromium);
chromium.launch=async (...args)=>instrument(await launch(...args));
const connect=chromium.connect.bind(chromium);
chromium.connect=async (...args)=>instrument(await connect(...args));
save();
`;

/** Inspect real emitted imports and retained modules, not source marker absence after tree shaking. */
export function inspectOutput(dist: string) {
    const graph = JSON.parse(readFileSync(join(dist, "bundle-audit.json"), "utf8")) as {
        file: string;
        imports: string[];
        dynamicImports: string[];
        modules: { id: string; bytes: number }[];
    }[];
    assert(graph.length > 0, "built output: empty chunk population");
    const modules = graph.flatMap((chunk) => chunk.modules);
    assert(modules.length > 10, "built output: empty retained module population");
    const forbidden =
        /(?:\/node_modules\/(?:vite|unplugin-typegpu|playwright|@playwright)\/|\/@dylanebert\/shallot\/(?:bin\/|src\/project\/|dist\/(?:vite|native|harness-browser)\.js))/;
    for (const module of modules)
        assert(!forbidden.test(module.id), `built output: tooling module ${module.id}`);
    for (const chunk of graph)
        for (const imported of [...chunk.imports, ...chunk.dynamicImports])
            assert(
                existsSync(join(dist, imported)),
                `built output: unresolved graph import ${imported}`,
            );
    for (const file of files(dist).filter((file) => /\.[cm]?js$/.test(file))) {
        const tree = parse(readFileSync(file, "utf8"), { sourceType: "module" });
        const walk = (node: unknown) => {
            if (!node || typeof node !== "object") return;
            if (Array.isArray(node)) {
                for (const child of node) walk(child);
                return;
            }
            const n = node as {
                type?: string;
                source?: { value?: string };
                callee?: { type?: string };
                arguments?: { type?: string; value?: string }[];
                [key: string]: unknown;
            };
            const source = [
                "ImportDeclaration",
                "ExportNamedDeclaration",
                "ExportAllDeclaration",
            ].includes(n.type ?? "")
                ? n.source?.value
                : n.type === "CallExpression" &&
                    n.callee?.type === "Import" &&
                    n.arguments?.[0]?.type === "StringLiteral"
                  ? n.arguments[0].value
                  : undefined;
            if (source) {
                assert(source.startsWith("."), `built output: unresolved emitted import ${source}`);
                assert(
                    existsSync(resolve(dirname(file), source)),
                    `built output: missing emitted import ${source}`,
                );
            }
            for (const [key, child] of Object.entries(n))
                if (!["loc", "start", "end", "comments"].includes(key)) walk(child);
        };
        walk(tree);
    }
    const inventory = files(dist)
        .filter((file) => !file.endsWith("bundle-audit.json"))
        .map((file) => {
            const bytes = readFileSync(file);
            return {
                file: file.slice(dist.length + 1),
                sha256: digest(bytes),
                raw: bytes.length,
                gzip: gzipSync(bytes).length,
                brotli: brotliCompressSync(bytes).length,
            };
        });
    return {
        inventory,
        modules: modules.map((m) => ({
            ...m,
            id: m.id.replace(/^.*\/node_modules\//, "node_modules/"),
        })),
        raw: inventory.reduce((n, f) => n + f.raw, 0),
        gzip: inventory.reduce((n, f) => n + f.gzip, 0),
        brotli: inventory.reduce((n, f) => n + f.brotli, 0),
    };
}

/** Same-input old/candidate physical builds; the installed CLI owns every bounded browser session. */
export async function outputFlow(work: string, candidate: string): Promise<void> {
    const evidence = join(work, "built-output");
    mkdirSync(evidence, { recursive: true });
    const oldDir = join(root, "scripts/install-test/compat-0.9.5/tarballs");
    const old = files(oldDir).find((file) => /\/shallot-0\.9\.5\.tgz$/.test(file));
    assert(old, "frozen previous engine tarball is required");
    let sequence = 0;
    const exec = (
        name: string,
        cmd: string[],
        cwd: string,
        pass = true,
        env: Record<string, string> = {},
    ) => {
        const p = Bun.spawnSync(cmd, {
            cwd,
            env: { ...process.env, ...env },
            stdout: "pipe",
            stderr: "pipe",
        });
        const output = p.stdout.toString() + p.stderr.toString();
        const stem = join(evidence, `${++sequence}-${name}`);
        writeFileSync(`${stem}.log`, output);
        writeFileSync(
            `${stem}.json`,
            JSON.stringify({
                cmd,
                cwd,
                exit: p.exitCode,
                signal: p.signalCode,
                runtime: Bun.version,
            }),
        );
        assert.equal(p.exitCode === 0, pass, `${name}: ${output.slice(-1800)}`);
        console.log(`output: ${name} exit=${p.exitCode}`);
        return output;
    };
    const compare: Record<string, unknown> = {};
    for (const [label, tar] of [
        ["previous", old],
        ["candidate", candidate],
    ] as const) {
        // one server per label, serving every `kind`: the frozen previous CLI can only reach real
        // hardware over `--connect`. The instrument above wraps `connect` as it wraps `launch`, so the
        // network receipt is the same evidence either way.
        const held: { server: { endpoint: string; stop(): void } | null } = { server: null };
        const transport = async (dir: string): Promise<string> =>
            label === "candidate"
                ? ""
                : (held.server ??= await startBrowserServer(dir, evidence)).endpoint;
        try {
            for (const kind of ["minimal", "text-assets", "physics", "blank"]) {
                const app = join(evidence, `${label}-${kind}`);
                mkdirSync(join(app, "public"), { recursive: true });
                mkdirSync(join(app, "src"));
                writeFileSync(
                    join(app, "package.json"),
                    JSON.stringify({
                        name: `output-${kind}`,
                        private: true,
                        type: "module",
                        dependencies: { "@dylanebert/shallot": `file:${tar}`, typegpu: "~0.12.4" },
                        devDependencies: {
                            vite: "^8.0.16",
                            "unplugin-typegpu": "~0.12.3",
                            playwright: "1.62.1",
                            ...(kind === "text-assets"
                                ? { svelte: "^5.0.0", "@sveltejs/vite-plugin-svelte": "^6.0.0" }
                                : {}),
                        },
                    }),
                );
                const text = kind === "text-assets",
                    physics = kind === "physics",
                    blank = kind === "blank";
                writeFileSync(
                    join(app, "shallot.json"),
                    JSON.stringify({
                        scene: "main.scene",
                        plugins: {
                            Orbit: true,
                            ...(text ? { Text: true } : {}),
                            ...(physics ? { Tumble: true } : {}),
                        },
                    }),
                );
                writeFileSync(
                    join(app, "public/main.scene"),
                    `<scene><a ambient-light="intensity: 0.6" /><a directional-light="direction: -1 -1 -1" /><a camera sear orbit="distance: 7; pitch: 0.2; pan: 0 1 0" transform /><a id="box" part ${physics ? 'body="pos: 0 3 0; mass: 1"' : 'transform="pos: 0 1 0"'} color="rgba: 0.85 0.55 0.35 1" />${physics ? '<a part body="mass: 0; pos: 0 -0.5 0; half-extents: 5 0.5 5" color="rgba: 0.3 0.4 0.3 1" />' : ""}${text ? '<a text="content: Local font; font: fixture; font-size: 0.4" transform="pos: 0 2 0" />' : ""}</scene>`,
                );
                if (blank)
                    writeFileSync(
                        join(app, "public/main.scene"),
                        "<scene><a camera sear transform /></scene>",
                    );
                if (text) {
                    cpSync(join(root, "assets/font.ttf"), join(app, "public/font.ttf"));
                    writeFileSync(join(app, "public/asset.bin"), new Uint8Array([41, 42, 43]));
                }
                writeFileSync(
                    join(app, "index.html"),
                    '<!doctype html><html><body style="margin:0"><canvas style="display:block;width:100vw;height:100vh"></canvas><script>window.__harness={ready:false};</script><script type="module" src="/src/main.ts"></script></body></html>',
                );
                writeFileSync(
                    join(app, "vite.config.ts"),
                    `import typegpu from "unplugin-typegpu/vite";import {projectPlugin} from "@dylanebert/shallot/vite";${text ? 'import {svelte} from "@sveltejs/vite-plugin-svelte";' : ""}\nexport default {base:"./",plugins:[${text ? 'svelte(),typegpu({enforce:"post",include:[/\\.m?[jt]sx?(?:\\?.*)?$/,/\\.svelte(?:\\?.*)?$/]}),' : "typegpu(),"}projectPlugin("."),${graphPlugin}],optimizeDeps:{exclude:["@dylanebert/shallot","typegpu"]},build:{target:"esnext"}};\n`,
                );
                if (text)
                    writeFileSync(
                        join(app, "src/Proof.svelte"),
                        `<script lang="ts">import tgpu from "typegpu";import * as d from "typegpu/data";const add=tgpu.fn([d.u32],d.u32)((x)=>{"use gpu";return x+7;});(window as any).__framework={value:add(5),code:tgpu.resolve([add])};</script><span hidden>transform witness</span>`,
                    );
                writeFileSync(
                    join(app, "src/main.ts"),
                    `import {run${text ? ",font" : ""}} from "@dylanebert/shallot";import {installHarness} from "@dylanebert/shallot/harness";import project from "virtual:project";${physics ? 'import {init,threads} from "@dylanebert/shallot/tumble/core";await init({threads:2});' : ""}${text ? 'import {Draws} from "@dylanebert/shallot/render/core";import {mount,unmount} from "svelte";import Proof from "./Proof.svelte";font("./font.ttf","fixture");const asset=new Uint8Array(await (await fetch("./asset.bin")).arrayBuffer());' : ""}\nconst app=await run({plugins:project.plugins,scene:project.scene??undefined,defaults:false${text ? ",ui(container){const component=mount(Proof,{target:container});return ()=>{void unmount(component);};}" : ""}});const harness=installHarness(app.state);harness.run=async()=>{for(let n=0;n<240&&app.state.time.elapsed<0.5;n++)await new Promise(requestAnimationFrame);const checks=[{name:"scene advanced",ok:app.state.time.elapsed>=0.5}${text ? ',{name:"local asset",ok:asset.join(",")==="41,42,43"},{name:"text atlas draw",ok:!!Draws.get("text0")},{name:"framework TGSL",ok:(window as any).__framework?.value===12&&/7u/.test((window as any).__framework?.code)}' : ""}${physics ? ',{name:"two real physics threads",ok:threads()===2},{name:"body fell",ok:(harness.read!([...app.state.identity.authored].find(id=>app.state.identity.id(id)==="box")!)?.pos[1]??3)<2.5}' : ""}];return {ok:checks.every(c=>c.ok),checks};};\n`,
                );
                writeFileSync(join(app, "network-preload.ts"), networkPreload);
                exec(`${label}-${kind}-install`, ["bun", "install"], app);
                const installed = join(app, "node_modules/@dylanebert/shallot");
                assert.equal(realpathSync(installed), installed, "physical runtime installation");
                exec(
                    `${label}-${kind}-build`,
                    ["bun", "node_modules/.bin/shallot", "build", "."],
                    app,
                );
                const dist = join(app, "dist");
                const summary = inspectOutput(dist);
                compare[`${label}-${kind}`] = { tar: digest(readFileSync(tar)), ...summary };
                writeFileSync(
                    join(evidence, `${label}-${kind}-summary.json`),
                    JSON.stringify(compare[`${label}-${kind}`], null, 2),
                );
                const endpoint = await transport(app);
                const verify = (name: string, pass = true, blocked = false, served = app) => {
                    const receipt = join(app, `${name}-requests.json`);
                    const output = exec(
                        `${label}-${kind}-${name}`,
                        [
                            "bun",
                            "--preload",
                            "./network-preload.ts",
                            "node_modules/.bin/shallot",
                            ...verifyArgs(label, endpoint, served),
                        ],
                        app,
                        pass,
                        { NETWORK_RECEIPT: receipt },
                    );
                    const requests = JSON.parse(readFileSync(receipt, "utf8")) as {
                        url: string;
                        allowed: boolean;
                    }[];
                    assert(
                        requests.length > 0,
                        "blocked-network instrument reached actual requests",
                    );
                    if (blocked)
                        assert(
                            requests.some(
                                (r) =>
                                    !r.allowed && r.url === "https://outside.invalid/required.bin",
                            ),
                            "external request was intercepted and blocked",
                        );
                    else
                        assert(
                            requests.every((r) => r.allowed),
                            `external requests: ${JSON.stringify(requests.filter((r) => !r.allowed))}`,
                        );
                    if (pass) {
                        assert(/"pass"\s*:\s*true/.test(output), "real installed browser verdict");
                        const servedDist = join(served, "dist");
                        for (const request of requests) {
                            const pathname = decodeURIComponent(new URL(request.url).pathname);
                            const resource = resolve(
                                servedDist,
                                pathname === "/" ? "index.html" : `.${pathname}`,
                            );
                            assert(
                                resource.startsWith(servedDist + "/") && existsSync(resource),
                                `built output: request is not supplied by static output ${pathname}`,
                            );
                        }
                    }
                    return output;
                };
                const positive = verify(blank ? "blank-control" : "self-contained", !blank);
                if (!blank) {
                    const standalone = join(app, "standalone");
                    mkdirSync(standalone);
                    cpSync(dist, join(standalone, "dist"), { recursive: true });
                    assert.deepEqual(
                        readdirSync(standalone),
                        ["dist"],
                        "static publication has no source, configuration or npm installation",
                    );
                    verify("static-only", true, false, standalone);
                }
                if (blank)
                    assert(
                        /"rendered"\s*:\s*false/.test(positive),
                        "blank reaches the real pixel refusal",
                    );
                if (label === "candidate" && !blank) {
                    const mainPath = join(app, "src/main.ts");
                    const mainSource = readFileSync(mainPath, "utf8");
                    try {
                        writeFileSync(
                            mainPath,
                            mainSource +
                                '\nimport {REAL_GPU_LAUNCH} from "@dylanebert/shallot/harness/browser"; (window as any).__toolingLeak=REAL_GPU_LAUNCH;\n',
                        );
                        exec(
                            `${label}-${kind}-tool-leak-build`,
                            ["bun", "node_modules/.bin/shallot", "build", "."],
                            app,
                        );
                        assert.throws(() => inspectOutput(dist), /tooling module/);
                    } finally {
                        writeFileSync(mainPath, mainSource);
                        exec(
                            `${label}-${kind}-restore-build`,
                            ["bun", "node_modules/.bin/shallot", "build", "."],
                            app,
                        );
                    }
                    if (kind === "minimal") {
                        try {
                            const marker = "harness.run=async()=>{";
                            assert.equal(mainSource.split(marker).length, 2);
                            writeFileSync(
                                mainPath,
                                mainSource.replace(
                                    marker,
                                    `${marker}await fetch("https://outside.invalid/required.bin");`,
                                ),
                            );
                            exec(
                                `${label}-${kind}-remote-build`,
                                ["bun", "node_modules/.bin/shallot", "build", "."],
                                app,
                            );
                            verify("external-network", false, true);
                        } finally {
                            writeFileSync(mainPath, mainSource);
                            exec(
                                `${label}-${kind}-remote-restore`,
                                ["bun", "node_modules/.bin/shallot", "build", "."],
                                app,
                            );
                        }
                    }
                    assert.deepEqual(
                        inspectOutput(dist),
                        summary,
                        "restored same-input output identity",
                    );
                    const chunk = files(dist).find(
                        (file) => /\.js$/.test(file) && !file.includes("kernel-mt"),
                    )!;
                    const original = readFileSync(chunk, "utf8");
                    try {
                        writeFileSync(chunk, `import "unresolved-runtime-fixture";\n${original}`);
                        assert.throws(() => inspectOutput(dist), /unresolved emitted import/);
                        verify("unresolved-import", false);
                    } finally {
                        writeFileSync(chunk, original);
                    }
                    if (text) {
                        for (const asset of ["font.ttf", "asset.bin"]) {
                            const path = join(dist, asset);
                            try {
                                renameSync(path, `${path}.absent`);
                                verify(`missing-${asset}`, false);
                            } finally {
                                renameSync(`${path}.absent`, path);
                            }
                        }
                    }
                    if (physics) {
                        const carrier = JSON.parse(
                            readFileSync(join(dist, "bundle-audit.json"), "utf8"),
                        ).find((chunk: any) =>
                            chunk.modules.some((module: any) =>
                                module.id.endsWith("/kernel.shared.wasm.ts"),
                            ),
                        );
                        assert(carrier, "lazy shared-memory WASM carrier emitted");
                        const wasm = join(dist, carrier.file);
                        try {
                            renameSync(wasm, `${wasm}.absent`);
                            verify("missing-wasm", false);
                        } finally {
                            renameSync(`${wasm}.absent`, wasm);
                        }
                    }
                    if (physics) {
                        let witness:
                            | { file: string; source: string; start: number; end: number }
                            | undefined;
                        for (const file of files(dist).filter((f) => /\.js$/.test(f))) {
                            const source = readFileSync(file, "utf8");
                            const tree = parse(source, { sourceType: "module" });
                            const walk = (node: any) => {
                                if (!node || typeof node !== "object") return;
                                if (
                                    (node.type === "StringLiteral" &&
                                        node.value.includes("ex.workerMain")) ||
                                    (node.type === "TemplateLiteral" &&
                                        node.quasis.some((q: any) =>
                                            q.value.raw.includes("ex.workerMain"),
                                        ))
                                ) {
                                    assert(!witness, "one worker payload carrier");
                                    witness = { file, source, start: node.start, end: node.end };
                                    return;
                                }
                                for (const [k, v] of Object.entries(node))
                                    if (!["loc", "start", "end", "comments"].includes(k)) {
                                        if (Array.isArray(v)) v.forEach(walk);
                                        else walk(v);
                                    }
                            };
                            walk(tree);
                        }
                        assert(witness, "actual emitted worker payload found");
                        const w = witness;
                        try {
                            writeFileSync(
                                w.file,
                                w.source.slice(0, w.start) + '""' + w.source.slice(w.end),
                            );
                            verify("missing-worker", false);
                        } finally {
                            writeFileSync(w.file, w.source);
                        }
                    }
                    verify("restored");
                }
            }
        } finally {
            held.server?.stop();
        }
    }
    writeFileSync(join(evidence, "comparison.json"), JSON.stringify(compare, null, 2));
}

if (import.meta.main) {
    const [work, tar] = process.argv.slice(2);
    assert(work && tar, "output.ts <owned evidence directory> <candidate tarball>");
    await outputFlow(resolve(work), resolve(tar));
}
