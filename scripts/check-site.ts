import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, resolve, sep } from "node:path";
import { Glob } from "bun";
import { ROSTER } from "../site/roster";
import { RUM_INJECTION_MARKER } from "../site/rum-config";

// `bun run scripts/check-site.ts` — the site membership gate, wired into `bun check`. Five clauses:
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
//      `out/site/` if it exists; skips with a note if not (the build hasn't run yet).
//   5. the Datadog RUM slow-frame injection reaches every demo page and skips the index — every
//      `*.html` under each `out/site/<slug>/` (including a nested page like
//      `visualization/demos/*.html`) carries `RUM_INJECTION_MARKER`; `out/site/index.html` does
//      not, since it has no frame loop to observe. Same `SITE_OUT_REQUIRED` gate as clause 4.
//
// The roster is derived from `examples/showcase/` by enumeration (site/roster.ts), so set membership
// is by construction — a roster-equals-disk clause would compare code against itself, the
// self-referential shape the spec names. The live defect that clause existed to catch
// (`examples/showcase/roads/` indexed nowhere) is caught earlier by derivation, since an unindexed
// dir is not a state the build can represent.

const root = resolve(import.meta.dir, "..");
const showcaseDir = resolve(root, "examples/showcase");
const outDir = resolve(root, "out/site");

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
    const dep = demoPkg.dependencies?.["@dylanebert/shallot"];
    if (!dep) {
        badVersion.push(`${slug}: no @dylanebert/shallot dependency`);
    } else if (dep !== "workspace:*") {
        badVersion.push(`${slug}: @dylanebert/shallot is "${dep}", not "workspace:*"`);
    }
}

if (badVersion.length > 0) {
    console.error(`✗ showcase package.json @dylanebert/shallot pin(s) not in workspace form:\n`);
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

const noInjection: string[] = [];
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

const indexPath = resolve(outDir, "index.html");
if (existsSync(indexPath) && readFileSync(indexPath, "utf8").includes(RUM_INJECTION_MARKER)) {
    console.error(
        "✗ out/site/index.html carries the RUM injection marker — the index has no frame loop" +
            " to observe and must stay JS-free.",
    );
    process.exit(1);
}

console.log(
    `✓ site roster clean (${ROSTER.length} demos, ` +
        `all manifested, all workspace-pinned, no escaping imports, no root-absolute paths, ` +
        `RUM injection present on every demo page and absent from the index)`,
);
