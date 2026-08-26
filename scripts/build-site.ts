import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";
import { Glob } from "bun";
import { type DemoEntry, ROSTER } from "../site/roster";
import {
    RUM_CONFIG,
    RUM_ENV_SNIPPET,
    RUM_ENV_USAGE,
    RUM_INJECTION_MARKER,
} from "../site/rum-config";

// `bun run site` — build every showcase demo as an ejected consumer of the *published* package,
// then assemble the site index. Each demo is copied out of the workspace to a scratch tree under
// /tmp with two files rewritten: a standalone `package.json` pinning `@dylanebert/shallot` to the
// release version (every other dep carried over as authored — those are the consumer's own pins),
// and a standalone `tsconfig.json` (the in-repo one extends a repo-root path that doesn't exist
// outside the workspace). The scratch tree installs against npm and builds with `shallot build`,
// so the artifact is a real published-consumer build — not a workspace build that silently carries
// gitignored wasm the clean path does not (the Locked decision's measured fact).
//
// The page itself is static HTML: system monospace, no web fonts, no JS, one small style block,
// readable at 360px. Each row carries a play link to the built demo and a code link to the
// version-pinned tag path on GitHub. The page labels what it was built from (version plus ref).

const root = resolve(import.meta.dir, "..");
const showcaseDir = resolve(root, "examples/showcase");
const outDir = resolve(root, "out/site");

// Datadog RUM browser-agent CDN major, pinned — checked 2026-08-25 against the served
// `/us1/v6/datadog-rum.js` bundle: `addDurationVital(name, {startTime, duration, context})` is
// present as the one-shot form the Locked decision calls for (`startDurationVital`/
// `stopDurationVital` also exist, unused here). Bump this only after re-checking that shape.
const DATADOG_RUM_CDN_MAJOR = 6;
const DATADOG_RUM_CDN_URL = `https://www.datadoghq-browser-agent.com/us1/v${DATADOG_RUM_CDN_MAJOR}/datadog-rum.js`;

// `crossOrigin='anonymous'` on the injected script element: `shallot verify`'s dist/dev preview sends
// `Cross-Origin-Embedder-Policy: require-corp` (`packages/shallot/src/project/vite.ts`, unconditional on
// every serve surface — for the multithreaded WASM kernel, unrelated to RUM) and the CDN never sends a
// `Cross-Origin-Resource-Policy` header, so a plain no-cors `<script src>` load is blocked
// (`net::ERR_BLOCKED_BY_RESPONSE.NotSameOriginAfterDefaultedToSameOriginByCoep`, reproduced 2026-08-25 —
// every `bun run demos` entry point failed on it). The CDN does answer a CORS request with
// `Access-Control-Allow-Origin: *` (verified against a request carrying an `Origin` header), and a
// CORS-mode load is exempt from the CORP check entirely — so `crossOrigin` fixes the verify-only failure
// without needing a header change in `packages/shallot` (out of scope) or the deployed site, which never
// sets COEP (a static host can't set headers, the doc comment above `CROSS_ORIGIN_ISOLATION` already notes).
function datadogInitSnippet(): string {
    return `${RUM_INJECTION_MARKER}
<script>
(function(h,o,u,n,d) {
    h=h[d]=h[d]||{q:[],onReady:function(c){h.q.push(c)}}
    d=o.createElement(u);d.async=1;d.src=n;d.crossOrigin='anonymous'
    n=o.getElementsByTagName(u)[0];n.parentNode.insertBefore(d,n)
})(window,document,'script','${DATADOG_RUM_CDN_URL}','DD_RUM')
window.DD_RUM.onReady(function() {
    ${RUM_ENV_SNIPPET}
    window.DD_RUM.init(${RUM_ENV_USAGE}${JSON.stringify(RUM_CONFIG)});
});
</script>
`;
}

/** Bundles `site/rum-runtime.ts` (which imports the pure sampler) to a single browser-target ESM
 * script — inlined so every demo page, at any output depth (`visualization/demos/*.html`
 * included), carries it with no relative-path plumbing. */
async function buildRumRuntimeBundle(): Promise<string> {
    const result = await Bun.build({
        entrypoints: [resolve(root, "site/rum-runtime.ts")],
        target: "browser",
        minify: false,
    });
    if (!result.success) {
        for (const log of result.logs) console.error(log);
        throw new Error("failed to bundle site/rum-runtime.ts");
    }
    const output = result.outputs[0];
    if (!output) throw new Error("site/rum-runtime.ts bundle produced no output");
    return await output.text();
}

/** Injects the Datadog init snippet plus the bundled sampler into every `*.html` file under
 * `dir` (recursive — a demo like `visualization` emits nested pages under `demos/`), right
 * before `</body>`. */
function injectRum(dir: string, runtimeBundle: string): void {
    const snippet = `${datadogInitSnippet()}<script type="module">\n${runtimeBundle}</script>\n`;
    const glob = new Glob("**/*.html");
    for (const path of glob.scanSync({ cwd: dir })) {
        const full = resolve(dir, path);
        const html = readFileSync(full, "utf8");
        const closeBodyRe = /<\/body>/i;
        if (!closeBodyRe.test(html)) {
            console.error(`✗ ${path} has no </body> — cannot inject RUM snippet`);
            process.exit(1);
        }
        writeFileSync(full, html.replace(closeBodyRe, `${snippet}</body>`));
    }
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run site [--demo <slug>]

Builds every showcase demo as an ejected consumer of the published @dylanebert/shallot,
assembles out/site/<slug>/ per demo, and emits out/site/index.html.

Options:
  --demo <slug>   Build a single demo by its roster slug`);
        process.exit(0);
    }

    const idx = args.indexOf("--demo");
    const only = idx !== -1 ? args[idx + 1] : undefined;
    if (only && !ROSTER.some((d) => d.slug === only)) {
        console.error(`no demo "${only}" — one of: ${ROSTER.map((d) => d.slug).join(", ")}`);
        process.exit(2);
    }

    const pkg = (await Bun.file(resolve(root, "packages/shallot/package.json")).json()) as {
        version: string;
    };
    const version = pkg.version;

    const ref = Bun.spawnSync(["git", "rev-parse", "--short", "HEAD"], { cwd: root });
    const refShort = ref.stdout.toString().trim() || "unknown";

    const demos = only ? ROSTER.filter((d) => d.slug === only) : ROSTER;

    // clean + recreate the output dir — a single-demo build only clears that demo's slot,
    // so a prior full build's other demos survive
    if (only) {
        rmSync(resolve(outDir, only), { recursive: true, force: true });
    } else {
        rmSync(outDir, { recursive: true, force: true });
    }
    mkdirSync(outDir, { recursive: true });

    const rumRuntimeBundle = await buildRumRuntimeBundle();

    const sizes: { slug: string; size: string }[] = [];

    for (const demo of demos) {
        const slug = demo.slug;
        const srcDir = resolve(showcaseDir, slug);
        if (!existsSync(srcDir)) {
            console.error(`✗ showcase dir not found: ${srcDir}`);
            process.exit(1);
        }

        // scratch tree under /tmp — the eject/install/build happens outside the workspace
        const scratch = join(tmpdir(), `shallot-site-${slug}-${Date.now()}`);
        mkdirSync(scratch, { recursive: true });

        try {
            console.log(`\n=== ${slug} ===`);

            // copy the showcase dir verbatim (node_modules/dist excluded by the dir's own .gitignore
            // patterns — but a fresh worktree has none, so just copy everything except those)
            const nodeModulesDir = resolve(srcDir, "node_modules");
            const distDir = resolve(srcDir, "dist");
            cpSync(srcDir, scratch, {
                recursive: true,
                filter: (s) =>
                    s !== nodeModulesDir &&
                    !s.startsWith(nodeModulesDir + sep) &&
                    s !== distDir &&
                    !s.startsWith(distDir + sep),
            });

            // rewrite package.json: pin @dylanebert/shallot to the release version, carry every
            // other dep over as authored
            const demoPkg = (await Bun.file(resolve(scratch, "package.json")).json()) as {
                dependencies?: Record<string, string>;
                devDependencies?: Record<string, string>;
                [key: string]: unknown;
            };
            if (demoPkg.dependencies?.["@dylanebert/shallot"]) {
                demoPkg.dependencies["@dylanebert/shallot"] = version;
            }
            writeFileSync(
                resolve(scratch, "package.json"),
                `${JSON.stringify(demoPkg, null, 4)}\n`,
            );

            // rewrite tsconfig.json: standalone (the in-repo one extends a repo-root path that
            // doesn't exist outside the workspace). Inline the compilerOptions from the root
            // tsconfig.json, minus the workspace-only `paths` mapping.
            writeFileSync(resolve(scratch, "tsconfig.json"), `${standaloneTsconfig()}\n`);

            // install against the published package from npm
            console.log(`  installing...`);
            const install = Bun.spawnSync(["bun", "install"], {
                cwd: scratch,
                stdout: "inherit",
                stderr: "inherit",
            });
            if (install.exitCode !== 0) {
                console.error(`✗ install failed for ${slug}`);
                process.exit(1);
            }

            // build with the published CLI
            console.log(`  building...`);
            const build = Bun.spawnSync(["bunx", "shallot", "build"], {
                cwd: scratch,
                stdout: "inherit",
                stderr: "inherit",
            });
            if (build.exitCode !== 0) {
                console.error(`✗ build failed for ${slug}`);
                process.exit(1);
            }

            // assemble out/site/<slug>/ from dist/
            const dist = resolve(scratch, "dist");
            if (!existsSync(dist)) {
                console.error(`✗ no dist/ produced for ${slug}`);
                process.exit(1);
            }
            const demoOut = resolve(outDir, slug);
            cpSync(dist, demoOut, { recursive: true });
            injectRum(demoOut, rumRuntimeBundle);

            const sizeBytes = dirSize(demoOut);
            sizes.push({ slug, size: formatSize(sizeBytes) });
            console.log(`  done — ${formatSize(sizeBytes)}`);
        } finally {
            rmSync(scratch, { recursive: true, force: true });
        }
    }

    // emit the site index — always lists the full roster so a single-demo build's index
    // still references the other demos from a prior full build
    writeFileSync(resolve(outDir, "index.html"), siteIndex(ROSTER, version, refShort));

    const total = sizes.reduce((sum, s) => sum + parseSize(s.size), 0);
    console.log(`\n=== summary ===`);
    for (const { slug, size } of sizes) {
        console.log(`  ${slug}: ${size}`);
    }
    console.log(`  total: ${formatSize(total)}`);
    console.log(`\n  index: ${resolve(outDir, "index.html")}`);
    console.log(`  built from: v${version} (${refShort})`);
}

// The standalone tsconfig — the root tsconfig.json's compilerOptions inlined, minus the
// workspace-only `paths` mapping (the published package resolves through node_modules, not
// a repo-relative alias). `@webgpu/types` is a dependency of the published package, so it
// resolves in the ejected install.
function standaloneTsconfig(): string {
    return JSON.stringify(
        {
            compilerOptions: {
                lib: ["ESNext", "DOM"],
                target: "ESNext",
                module: "ESNext",
                moduleDetection: "force",
                allowJs: true,
                moduleResolution: "bundler",
                verbatimModuleSyntax: true,
                resolveJsonModule: true,
                noEmit: true,
                strict: true,
                skipLibCheck: true,
                noFallthroughCasesInSwitch: true,
                noImplicitOverride: true,
                noUnusedLocals: false,
                noUnusedParameters: false,
                noPropertyAccessFromIndexSignature: false,
                types: ["@webgpu/types"],
            },
        },
        null,
        4,
    );
}

function dirSize(dir: string): number {
    let total = 0;
    const glob = new Glob("**/*");
    for (const path of glob.scanSync({ cwd: dir, onlyFiles: true })) {
        total += Bun.file(join(dir, path)).size;
    }
    return total;
}

function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}K`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}M`;
}

function parseSize(s: string): number {
    const m = s.match(/^([\d.]+)([KMB]?)$/);
    if (!m) return 0;
    const n = Number(m[1]);
    const unit = m[2];
    if (unit === "K") return n * 1024;
    if (unit === "M") return n * 1024 * 1024;
    return n;
}

function siteIndex(demos: DemoEntry[], version: string, ref: string): string {
    const codeUrl = (slug: string) =>
        `https://github.com/dylanebert/shallot/tree/v${version}/examples/showcase/${slug}`;

    const rows = demos
        .map((d) => {
            const play = `./${d.slug}/`;
            const code = codeUrl(d.slug);
            return `            <tr>
                <td><a href="${play}">${d.title}</a></td>
                <td><a href="${code}">code</a></td>
            </tr>`;
        })
        .join("\n");

    return `<!doctype html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>shallot — demos</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                background: #0c0a09;
                color: #e6e0d8;
                font-family: ui-monospace, "SF Mono", "Cascadia Mono", "Menlo", "Consolas", monospace;
                padding: 1.25rem 1rem 3rem;
                line-height: 1.5;
            }
            h1 { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.25rem; }
            .meta { color: #8a8078; font-size: 0.8rem; margin-bottom: 1rem; }
            .warn { color: #c9a227; font-size: 0.8rem; margin-bottom: 1.25rem; }
            table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
            td { padding: 0.45rem 0.6rem 0.45rem 0; vertical-align: top; }
            tr { border-bottom: 1px solid #1e1a16; }
            tr:last-child { border-bottom: none; }
            a { color: #6cb6ff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            td:first-child { white-space: nowrap; }
            td:last-child { white-space: nowrap; text-align: right; }
            @media (max-width: 360px) {
                body { padding: 1rem 0.75rem 2rem; }
                td { padding: 0.4rem 0.4rem 0.4rem 0; }
            }
        </style>
    </head>
    <body>
        <h1>shallot</h1>
        <p class="meta">v${version} · ${ref}</p>
        <p class="warn">WebGPU required — Chrome, Edge, or Safari 26+ on desktop.</p>
        <table>
            <tbody>
${rows}
            </tbody>
        </table>
    </body>
</html>
`;
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});
