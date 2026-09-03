import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, resolve, sep } from "node:path";
import { Glob } from "bun";
import { ROSTER } from "../site/roster";
import {
    RUM_ENV_SNIPPET,
    RUM_ENV_SNIPPET_STAGING,
    RUM_ENV_USAGE,
    RUM_INJECTION_MARKER,
} from "../site/rum-config";
import { readStamp, STAMP_FILE, staleDemos } from "../site/site-stamp";
import { nonWorkspaceShallotDependencies } from "./build-site";

// `bun run scripts/check-site.ts` — the site membership gate, wired into `bun check`. Seven
// clauses, and a freshness precondition standing in front of the four (4-7) that read the built
// artifact:
//
//   0. every present `out/site/<slug>/` is a build of *this* tree's sources — the build stamps
//      each demo with a content fingerprint of what it built that demo from (`site/site-stamp.ts`),
//      and a slot whose fingerprint no longer matches the tree is refused as stale instead of
//      being read as a live defect. Without this relation clause 6 reported a stale local artifact
//      (built before the title fix at `bf1c25f`, never rebuilt) as a present-tense title defect,
//      and a roadmap item was written from the false red. Stale under `SITE_OUT_REQUIRED=1` (the
//      deploy path, where the build just ran) is itself the failure; otherwise the artifact
//      clauses skip with a note, the same shape as an absent `out/site/`.
//
//
//   1. every entry's dir has a manifest — a showcase dir without `shallot.json` (or, for an
//      ejected project, `index.html`) is not a buildable demo.
//   2. the ejected manifest names the release version — the in-repo `package.json` pins
//      `@dylanebert/shallot` as `workspace:*`; the build script rewrites it to the release version
//      from `packages/shallot/package.json`. This clause verifies the in-repo form is `workspace:*`
//      and that the release version is a real semver (not `workspace:*`), so the ejection would
//      produce a published-consumer pin, not a workspace alias that can't resolve outside the repo.
//   3. no import escapes a demo's own directory — an ejected demo importing anything outside its
//      own dir would break, since only the demo's dir is copied to the scratch tree. A relative
//      import that climbs out is the failure mode; package imports (`@dylanebert/shallot`,
//      `typegpu`) resolve through node_modules and are fine.
//   4. no generated path is root-absolute — the site must work at any base path (GitHub Pages
//      serves at `/shallot/`, a dry-run artifact at root, a local out dir at `file://`). Scans
//      the output dir if it exists; skips with a note if not (the build hasn't run yet).
//   5. the Datadog RUM slow-frame injection reaches every demo page and skips the index — every
//      `*.html` under each `out/site/<slug>/` (including a nested page like
//      `visualization/demos/*.html`) carries `RUM_INJECTION_MARKER`, `RUM_ENV_SNIPPET` (the
//      hostname-derived `env: "prod" | "local"` derivation — localhost previews tag "local" so
//      they never pollute prod's slow-frame vitals) and `RUM_ENV_USAGE` (the wiring that actually
//      reads the derived `env` into `DD_RUM.init` — the derivation alone is a silent no-op if the
//      wiring drops it); the
//      injected `<script>` block also has to parse (`new Function(src)`) — the substring checks
//      pin the seam, this pins that the composed call is runnable, since a dropped paren between
//      two present fragments passes every substring check while still being broken syntax.
//      `out/site/index.html` does not carry any of this, since it has no frame loop to observe.
//      Same `SITE_OUT_REQUIRED` gate as clause 4.
//   6. every built demo root page has a human-readable <title> — a manifest demo's is the bare
//      slug (`synthIndex` titles from the ejected dir's basename); an own-index demo's just must
//      not be scratch-shaped.
//   7. no built page carries a placeholder RUM credential — gated behind RUM_CONFIG_REQUIRED
//      (armed on the deploy path only; see the clause body).
//
// The roster is derived from `examples/showcase/` by enumeration (site/roster.ts), so set membership
// is by construction — a roster-equals-disk clause would compare code against itself, the
// self-referential shape the spec names. The live defect that clause existed to catch
// (`examples/showcase/roads/` indexed nowhere) is caught earlier by derivation, since an unindexed
// dir is not a state the build can represent.

const root = resolve(import.meta.dir, "..");
const showcaseDir = resolve(root, "examples/showcase");
// `SITE_OUT_DIR` points the artifact clauses at an output dir other than the default — a CI job
// that downloaded a build artifact elsewhere, or a fixture tree proving clause ordering. The
// source-side clauses (1-3) always read this repo.
const outDir = process.env.SITE_OUT_DIR
    ? resolve(process.env.SITE_OUT_DIR)
    : resolve(root, "out/site");

function fail(msg: string): never {
    console.error(msg);
    process.exit(1);
}

// --- clause 1: every entry's dir has a manifest -------------------------------------------

const noManifest: string[] = [];
for (const { slug } of ROSTER) {
    const dir = resolve(showcaseDir, slug);
    const hasManifest = existsSync(resolve(dir, "shallot.json"));
    const hasIndex = existsSync(resolve(dir, "index.html"));
    if (!hasManifest && !hasIndex) {
        noManifest.push(slug);
    }
}

if (noManifest.length > 0) {
    console.error(`✗ showcase dir(s) without a manifest (shallot.json or index.html):\n`);
    for (const s of noManifest) console.error(`  ${s}`);
    process.exit(1);
}

// --- clause 2: the ejected manifest names the release version -----------------------------

const pkg = (await Bun.file(resolve(root, "packages/shallot/package.json")).json()) as {
    version: string;
};
const version = pkg.version;

const semverRe = /^\d+\.\d+\.\d+/;
if (!semverRe.test(version)) {
    fail(`✗ packages/shallot/package.json version "${version}" is not a semver release`);
}

const badVersion: string[] = [];
for (const { slug } of ROSTER) {
    const dir = resolve(showcaseDir, slug);
    const demoPkg = (await Bun.file(resolve(dir, "package.json")).json()) as {
        dependencies?: Record<string, string>;
    };
    const dependencies = demoPkg.dependencies ?? {};
    const engine = dependencies["@dylanebert/shallot"];
    if (!engine) {
        badVersion.push(`${slug}: no @dylanebert/shallot dependency`);
    }
    for (const [name, pin] of nonWorkspaceShallotDependencies(demoPkg)) {
        badVersion.push(`${slug}: ${name} is "${pin}", not "workspace:*"`);
    }
}

if (badVersion.length > 0) {
    console.error(`✗ showcase package.json Shallot dependency pin(s) not in workspace form:\n`);
    for (const s of badVersion) console.error(`  ${s}`);
    console.error(
        "\nThe in-repo form must be `workspace:*`; the build script rewrites it to the release" +
            ` version (${version}) at ejection time.`,
    );
    process.exit(1);
}

// --- clause 3: no import escapes a demo's own directory ----------------------------------

const importRe = /(?:from\s+["']|import\s+["']|import\s*\(\s*["'])([^"']+)["']/g;
const escaped: { slug: string; file: string; line: number; spec: string }[] = [];

for (const { slug } of ROSTER) {
    const dir = resolve(showcaseDir, slug);
    const glob = new Glob("**/*.{ts,svelte,js,mjs}");
    for await (const path of glob.scan({ cwd: dir })) {
        if (path.includes("node_modules") || path.includes("dist")) continue;
        if (path.endsWith(".test.ts")) continue; // tests aren't part of the build
        const full = resolve(dir, path);
        const lines = (await Bun.file(full).text()).split("\n");
        for (let i = 0; i < lines.length; i++) {
            for (const match of lines[i].matchAll(importRe)) {
                const spec = match[1];
                if (!spec.startsWith(".")) continue; // package imports are fine
                const resolved = resolve(dirname(full), spec);
                if (resolved === dir || resolved.startsWith(dir + sep)) continue;
                escaped.push({ slug, file: `${slug}/${path}`, line: i + 1, spec });
            }
        }
    }
}

if (escaped.length > 0) {
    console.error(`✗ ${escaped.length} import(s) escape a demo's own directory:\n`);
    for (const e of escaped) {
        console.error(`  ${e.file}:${e.line}`);
        console.error(`    import "${e.spec}"`);
    }
    console.error(
        "\nAn ejected demo is copied alone to a scratch tree — a relative import that climbs" +
            " out of the demo dir would break the build.",
    );
    process.exit(1);
}

// --- clause 4: no generated path is root-absolute -----------------------------------------

if (!existsSync(outDir)) {
    if (process.env.SITE_OUT_REQUIRED === "1") {
        console.error(`✗ out/site/ is absent — run \`bun run site\` first, then re-run this check`);
        process.exit(1);
    }
    console.log(
        `✓ site roster clean (${ROSTER.length} demos, ` +
            `all manifested, all workspace-pinned, no escaping imports) — ` +
            `out/site/ not built yet (run \`bun run site\`), skipping root-absolute path check`,
    );
    process.exit(0);
}

if (process.env.SITE_OUT_REQUIRED === "1" && readdirSync(outDir).length === 0) {
    console.error(`✗ out/site/ is empty — run \`bun run site\` first, then re-run this check`);
    process.exit(1);
}

// --- clause 0: the artifact is a build of this tree's sources ----------------------------
//
// Content-derived, never mtime: the build records a fingerprint of each demo's own build inputs
// (`site/site-stamp.ts`) and this compares the recorded fingerprint against the tree's current
// one. mtime cannot carry this — the founding stale artifact's mtime (22:52) sat *after* the fix
// commit it predated (22:50), so "newer than the fix" was true of a dir built without it.
// Everything below reads the artifact, so a stale artifact stops here.
const stale = staleDemos(
    root,
    outDir,
    ROSTER.map((d) => d.slug),
);
if (stale.length > 0) {
    if (process.env.SITE_OUT_REQUIRED === "1") {
        console.error(
            `✗ ${stale.length} built demo dir(s) are stale — not a build of this tree's sources:\n`,
        );
        for (const { slug, reason } of stale) console.error(`  ${slug}: ${reason}`);
        console.error(
            `\nThe artifact clauses (root-absolute paths, RUM injection, <title>) read the built` +
                ` output as present-tense truth about this tree, so a stale slot would report a` +
                ` defect the sources no longer carry. Run \`bun run site\` and re-run this check.`,
        );
        process.exit(1);
    }
    console.log(
        `✓ site roster clean (${ROSTER.length} demos, ` +
            `all manifested, all workspace-pinned, no escaping imports) — ` +
            `${stale.length} built demo dir(s) stale against the current sources ` +
            `(${stale.map((s) => s.slug).join(", ")}; ${STAMP_FILE}), ` +
            `skipping the artifact clauses (run \`bun run site\` to judge them)`,
    );
    process.exit(0);
}

// --- clause 2 (artifact leg): the ejected pin actually used matches the declared mode -----
//
// The source-side half of clause 2 (above) can only assert the in-repo form (`workspace:*`) and
// that the release version is a real semver — the pin actually written into each demo's ejected
// `package.json` exists only in a scratch tree that `scripts/build-site.ts` deletes before this
// script ever runs. So the build stamp records which pin the run used
// (`site/site-stamp.ts`'s `SiteMode`), and this leg reads it back: a prod-mode stamp must have
// pinned the current release version, a staging-mode stamp must have pinned a `file:`-form
// workspace tarball — never the other way, so a mode mix-up reds here instead of the built
// artifact silently carrying the wrong pin.
const stamp = readStamp(outDir);
if (stamp) {
    if (stamp.mode.kind === "prod") {
        if (stamp.mode.version !== version) {
            fail(
                `✗ build stamp records prod mode pinned to v${stamp.mode.version}, but ` +
                    `packages/shallot/package.json now names v${version} — rebuild with ` +
                    "`bun run site`",
            );
        }
    } else {
        if (!stamp.mode.pin.startsWith("file:") || !stamp.mode.pin.endsWith(".tgz")) {
            fail(
                `✗ build stamp records staging mode with pin "${stamp.mode.pin}" — expected a ` +
                    "`file:<tgz>` workspace pin (`scripts/build-site.ts`'s `bun pm pack` step)",
            );
        }
    }
}
const mode: "prod" | "staging" = stamp?.mode.kind ?? "prod";

// scan every generated HTML/JS/CSS file for root-absolute paths: `href="/..."` or `src="/..."`
// HTML attributes. A root-absolute path would break at a non-root base path (GitHub Pages
// serves at /shallot/).
const rootAbsRe = /(?:href|src)\s*=\s*["']\/(?!\/)/g;
const rootAbsFiles: { file: string; line: number; text: string }[] = [];

const outGlob = new Glob("**/*.{html,js,css}");
for await (const path of outGlob.scan({ cwd: outDir })) {
    const full = resolve(outDir, path);
    const lines = (await Bun.file(full).text()).split("\n");
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].match(rootAbsRe)) {
            rootAbsFiles.push({ file: path, line: i + 1, text: lines[i].trim() });
        }
    }
}

if (rootAbsFiles.length > 0) {
    console.error(`✗ ${rootAbsFiles.length} root-absolute path(s) in generated site output:\n`);
    for (const f of rootAbsFiles) {
        console.error(`  ${f.file}:${f.line}`);
        console.error(`    ${f.text}`);
    }
    console.error(
        "\nThe site must work at any base path (GitHub Pages serves at /shallot/). Built demos" +
            " reference assets relatively (./assets/...); the index must too.",
    );
    process.exit(1);
}

// --- clause 5: the RUM slow-frame injection reaches every demo page, and skips the index --
//
// The env check is two-sided: a page must carry the *mode's own* env literal and must not carry
// the *other* mode's. A one-sided check (present-only) passes a mode mix-up — a prod build that
// somehow injected the staging literal instead would still clear "prod snippet missing? no" if
// nothing ever checked for staging's literal being there — so `wrongModeEnv` reds that shape
// explicitly rather than relying on the two literals happening to differ enough in practice.

const ownEnvSnippet = mode === "staging" ? RUM_ENV_SNIPPET_STAGING : RUM_ENV_SNIPPET;
const otherEnvSnippet = mode === "staging" ? RUM_ENV_SNIPPET : RUM_ENV_SNIPPET_STAGING;

const noInjection: string[] = [];
const noEnv: string[] = [];
const wrongModeEnv: string[] = [];
const noEnvUsage: string[] = [];
const noParse: { file: string; error: string }[] = [];
for (const { slug } of ROSTER) {
    const demoDir = resolve(outDir, slug);
    if (!existsSync(demoDir)) continue; // a `--demo` filtered build only writes some dirs
    const demoGlob = new Glob("**/*.html");
    for (const path of demoGlob.scanSync({ cwd: demoDir })) {
        const full = resolve(demoDir, path);
        const html = readFileSync(full, "utf8");
        if (!html.includes(RUM_INJECTION_MARKER)) {
            noInjection.push(`${slug}/${path}`);
        }
        if (!html.includes(ownEnvSnippet)) {
            noEnv.push(`${slug}/${path}`);
        }
        if (html.includes(otherEnvSnippet)) {
            wrongModeEnv.push(`${slug}/${path}`);
        }
        if (!html.includes(RUM_ENV_USAGE)) {
            noEnvUsage.push(`${slug}/${path}`);
        }

        // The substring checks above pin the seam (each fragment present) but not the
        // composition — a paren dropped between `RUM_ENV_USAGE` and `JSON.stringify(RUM_CONFIG)`
        // still contains every fragment while the composed call is unparseable (measured
        // 2026-08-25: `window.DD_RUM.init(Object.assign({env:ddEnv},{...});` — missing
        // `Object.assign`'s closing paren — threw `Unexpected token ';'` from every built page's
        // real script and none of the substring checks caught it). Extract the injected
        // `<script>` block after the marker (the plain inline one, not the
        // `<script type="module">` sampler bundle right after it) and syntax-check it with
        // `new Function(src)` — construction parses the body without executing it, so this is a
        // syntax check, not an execution one.
        const markerIdx = html.indexOf(RUM_INJECTION_MARKER);
        if (markerIdx !== -1) {
            const scriptMatch = html.slice(markerIdx).match(/<script>([\s\S]*?)<\/script>/);
            if (!scriptMatch) {
                noParse.push({
                    file: `${slug}/${path}`,
                    error: "no <script> block found after the RUM injection marker",
                });
            } else {
                try {
                    new Function(scriptMatch[1]);
                } catch (err) {
                    noParse.push({
                        file: `${slug}/${path}`,
                        error: err instanceof Error ? err.message : String(err),
                    });
                }
            }
        }
    }
}

if (noInjection.length > 0) {
    console.error(`✗ ${noInjection.length} demo page(s) missing the RUM slow-frame injection:\n`);
    for (const f of noInjection) console.error(`  ${f}`);
    console.error(
        "\nEvery built demo page must carry the Datadog RUM init + sampler snippet" +
            " (`scripts/build-site.ts`'s post-copy rewrite).",
    );
    process.exit(1);
}

if (noEnv.length > 0) {
    console.error(
        `✗ ${noEnv.length} demo page(s) missing the RUM env-derivation snippet for ${mode} mode:\n`,
    );
    for (const f of noEnv) console.error(`  ${f}`);
    console.error(
        `\nEvery built demo page must carry the ${mode} env snippet` +
            ` (\`site/rum-config.ts\`'s \`${mode === "staging" ? "RUM_ENV_SNIPPET_STAGING" : "RUM_ENV_SNIPPET"}\`,` +
            " injected by `scripts/build-site.ts`).",
    );
    process.exit(1);
}

if (wrongModeEnv.length > 0) {
    console.error(
        `✗ ${wrongModeEnv.length} demo page(s) built in ${mode} mode carry the other mode's env` +
            " snippet:\n",
    );
    for (const f of wrongModeEnv) console.error(`  ${f}`);
    console.error(
        `\nA ${mode}-mode build must carry only the ${mode} env snippet — a page carrying both` +
            " reads as a mode mix-up in the build itself, not the check.",
    );
    process.exit(1);
}

if (noEnvUsage.length > 0) {
    console.error(`✗ ${noEnvUsage.length} demo page(s) missing the RUM env wiring:\n`);
    for (const f of noEnvUsage) console.error(`  ${f}`);
    console.error(
        "\nThe derived `env` must actually reach `DD_RUM.init` — every built demo page must" +
            " carry `site/rum-config.ts`'s `RUM_ENV_USAGE` literal (`RUM_ENV_SNIPPET` computing" +
            " `ddEnv` with nothing reading it is a silent no-op).",
    );
    process.exit(1);
}

if (noParse.length > 0) {
    console.error(`✗ ${noParse.length} demo page(s) carry an unparseable RUM init script:\n`);
    for (const { file, error } of noParse) console.error(`  ${file}: ${error}`);
    console.error(
        "\nThe injected `<script>` block after the RUM injection marker failed to parse" +
            " (`new Function(src)`) — every fragment substring check above can pass while the" +
            " composed call is still broken syntax (`scripts/build-site.ts`'s" +
            " `datadogInitSnippet`).",
    );
    process.exit(1);
}

const indexPath = resolve(outDir, "index.html");
if (existsSync(indexPath) && readFileSync(indexPath, "utf8").includes(RUM_INJECTION_MARKER)) {
    console.error(
        "✗ out/site/index.html carries the RUM injection marker — the index has no frame loop" +
            " to observe and must stay JS-free.",
    );
    process.exit(1);
}

// --- clause 6: every built demo root page has a human-readable <title> --------------------
//
// `shallot build` synthesizes a manifest project's <title> from the project dir's basename
// (bin/build.ts `synthIndex`), and the site build ejects each demo into a scratch tree — so the
// scratch leaf must be named the bare slug, or every tab reads like a temp path (measured
// 2026-08-25: `shallot-site-collapse-1756…` live on dylanebert.com). Manifest demos (no own
// index.html) must title exactly the slug; a demo shipping its own index.html owns its title,
// which just must not be scratch-shaped. Red-first witnessed 2026-08-25 against a synthetic
// out/site fixture clearing clauses 4/5 (exit 1 listing every scratch-shaped title; exit 0 with
// slug titles) — the real-build leg is CI's SITE_OUT_REQUIRED deploy run.
const badTitle: { file: string; title: string }[] = [];
for (const { slug } of ROSTER) {
    const builtIndex = resolve(outDir, slug, "index.html");
    if (!existsSync(builtIndex)) continue; // a `--demo` filtered build only writes some dirs
    const html = readFileSync(builtIndex, "utf8");
    const title = html.match(/<title>([^<]*)<\/title>/)?.[1] ?? "";
    const ownsIndex = existsSync(resolve(showcaseDir, slug, "index.html"));
    if (ownsIndex ? title.includes("shallot-site-") : title !== slug) {
        badTitle.push({ file: `${slug}/index.html`, title });
    }
}
if (badTitle.length > 0) {
    console.error(`✗ ${badTitle.length} built demo page(s) carry a non-human-readable <title>:\n`);
    for (const t of badTitle) console.error(`  ${t.file}: <title>${t.title}</title>`);
    console.error(
        "\nA manifest demo's <title> is the ejected dir's basename (bin/build.ts `synthIndex`)," +
            " so `scripts/build-site.ts` must eject into `<unique-parent>/<slug>` — the parent" +
            " carries the uniqueness, the leaf carries the name.",
    );
    process.exit(1);
}

// --- clause 7: no built page carries a placeholder RUM credential -------------------------
//
// `site/rum-config.ts` ships `PLACEHOLDER_*` values until the Dogfood-org RUM application
// exists (S1 credentials ask); nothing refused them at build or deploy. Gated behind
// RUM_CONFIG_REQUIRED, same shape as SITE_OUT_REQUIRED for clause 4/5, so today's placeholder
// state doesn't red the default suite. Armed on the real deploy path: `.github/workflows/
// site.yml`'s build job sets `RUM_CONFIG_REQUIRED` on this script's step to an expression that
// mirrors the deploy job's own `if:` gate verbatim (tag push, or workflow_dispatch with
// `deploy: true` on `main`) — so a deploying run reds on a placeholder credential and an
// ordinary artifact-only build stays green while credentials are still pending.
//
// Mutation-proven both directions over the same `out/site/` build (witnessed 2026-08-25):
//   - RUM_CONFIG_REQUIRED=1 over the current placeholder build: reds —
//     "✗ … built page(s) carry a PLACEHOLDER_ RUM credential" listing every demo HTML.
//   - RUM_CONFIG_REQUIRED=1 with `site/rum-config.ts`'s three PLACEHOLDER_ strings swapped for
//     dummy-real values (`pub00000000000000000000000000000000`, an app-id UUID, `datadoghq.com`)
//     and the site rebuilt: passes clean.
if (process.env.RUM_CONFIG_REQUIRED === "1" && existsSync(outDir)) {
    const placeholderHits: { file: string; line: number }[] = [];
    const allHtmlGlob = new Glob("**/*.html");
    for await (const path of allHtmlGlob.scan({ cwd: outDir })) {
        const full = resolve(outDir, path);
        const lines = (await Bun.file(full).text()).split("\n");
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes("PLACEHOLDER_")) {
                placeholderHits.push({ file: path, line: i + 1 });
            }
        }
    }

    if (placeholderHits.length > 0) {
        console.error(
            `✗ ${placeholderHits.length} built page(s) carry a PLACEHOLDER_ RUM credential:\n`,
        );
        for (const h of placeholderHits) console.error(`  ${h.file}:${h.line}`);
        console.error(
            "\nCreate the Dogfood-org RUM browser application and set real applicationId/" +
                "clientToken/site in site/rum-config.ts before deploying.",
        );
        process.exit(1);
    }
}

console.log(
    `✓ site roster clean (${ROSTER.length} demos, ` +
        `all manifested, all workspace-pinned, no escaping imports, no root-absolute paths, ` +
        `RUM injection + env snippet present on every demo page and absent from the index)`,
);
